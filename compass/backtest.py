"""Simulate strategies on daily data with leverage costs, drift and trading friction."""
import re

import numpy as np
import pandas as pd

from . import data
from .strategies import BASE_SERIES, STRATEGIES, Strategy, month_end_mask

START = "2000-01-03"
SLIPPAGE = 0.0005  # per unit of turnover; Robinhood is commission-free
BORROW_SPREAD = 0.005  # annual spread over T-bills paid on leveraged exposure
LEVERED_ER = {1.5: 0.003, 2.0: 0.009, 3.0: 0.009}
LEVERED = re.compile(r"^(\w+?)x([\d.]+)$")


def load(refresh=False):
    px = data.prices(sorted(set(BASE_SERIES.values())), refresh)
    px.columns = [next(k for k, v in BASE_SERIES.items() if v == c) for c in px.columns]
    px = px[px.index >= "1993-01-29"].ffill()
    cash = data.cash_daily_returns(px.index, refresh)
    px["_cash"] = cash
    return px


def sleeve_returns(px: pd.DataFrame, sleeves) -> pd.DataFrame:
    rets = px.drop(columns="_cash").pct_change()
    cash = px["_cash"]
    out = {}
    for s in sleeves:
        if s == "CASH":
            out[s] = cash
            continue
        m = LEVERED.match(s)
        if m:
            base, lev = m.group(1), float(m.group(2))
            r = rets[base]
            out[s] = lev * r - (lev - 1) * (cash + BORROW_SPREAD / 252) - LEVERED_ER[lev] / 252
        else:
            out[s] = rets[s]
    return pd.DataFrame(out).fillna(0.0)


def run(strategy: Strategy, px: pd.DataFrame, start=START):
    target = strategy.weights(px).fillna(0.0)
    target = target.loc[:, (target != 0).any()]
    rets = sleeve_returns(px, target.columns)
    idx = px.index[px.index >= start]
    target, rets = target.reindex(idx).ffill(), rets.reindex(idx)
    month_end = month_end_mask(idx)
    rebal_monthly = strategy.cadence == "monthly"

    t, r = target.values, rets.values
    hold = t[0].copy()
    nav = np.empty(len(idx))
    nav[0], value, trades, turnover = 1.0, 1.0, 0, 0.0
    for i in range(1, len(idx)):
        hold = hold * (1 + r[i])
        value = hold.sum()
        if value <= 0:
            nav[i:] = 0.0
            break
        w_now = hold / value
        changed = not np.allclose(t[i], t[i - 1])
        if changed or (rebal_monthly and month_end[i]):
            to = np.abs(t[i] - w_now).sum()
            value *= 1 - SLIPPAGE * to
            hold = t[i] * value
            turnover += to
            trades += to > 1e-6
        nav[i] = value
    nav = pd.Series(nav, index=idx, name=strategy.key)
    return nav, trades, target


def metrics(nav: pd.Series, trades: int, cash: pd.Series) -> dict:
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    final = nav.iloc[-1]
    cagr = final ** (1 / years) - 1 if final > 0 else -1.0
    dr = nav.pct_change().dropna()
    vol = dr.std() * np.sqrt(252)
    excess = dr - cash.reindex(dr.index).fillna(0)
    sharpe = excess.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    peak = nav.cummax()
    dd = nav / peak - 1
    max_dd = dd.min()
    # longest time below a prior peak
    under = dd < -1e-9
    longest, run_start = 0, None
    for d, u in under.items():
        if u and run_start is None:
            run_start = d
        if (not u or d == under.index[-1]) and run_start is not None:
            longest = max(longest, (d - run_start).days)
            run_start = None
    yearly = nav.resample("YE").last().pct_change()
    yearly.iloc[0] = nav.resample("YE").last().iloc[0] / nav.iloc[0] - 1
    roll10 = nav / nav.shift(2520)
    roll10 = (roll10 ** (1 / 10) - 1).dropna()
    return {
        "cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": max_dd,
        "longest_underwater_years": longest / 365.25,
        "worst_year": yearly.min(), "best_year": yearly.max(),
        "final_10k": 10_000 * final, "trades_per_year": trades / years,
        "pct_10y_windows_15": float((roll10 >= 0.15).mean()) if len(roll10) else None,
        "worst_10y_cagr": float(roll10.min()) if len(roll10) else None,
        "yearly": {str(k.year): float(v) for k, v in yearly.items()},
    }


STRESS = {
    "Dot-com crash (2000-02)": ("2000-01-03", "2002-10-09"),
    "Financial crisis (2007-09)": ("2007-10-09", "2009-03-09"),
    "COVID crash (2020)": ("2020-02-19", "2020-03-23"),
    "2022 bear market": ("2022-01-03", "2022-10-12"),
    "Last 12 months": (None, None),
}


def stress(nav: pd.Series) -> dict:
    out = {}
    for name, (a, b) in STRESS.items():
        if a is None:
            b_ = nav.index[-1]
            a_ = nav.index[nav.index.searchsorted(b_ - pd.Timedelta(days=365))]
            s = nav.loc[a_:b_]
        else:
            s = nav.loc[a:b]
        out[name] = float(s.iloc[-1] / s.iloc[0] - 1) if len(s) > 1 and s.iloc[0] > 0 else -1.0
    return out


def run_all(refresh=False, start=START):
    px = load(refresh)
    results = []
    for st in STRATEGIES:
        nav, trades, _ = run(st, px, start)
        m = metrics(nav, trades, px["_cash"])
        dd = nav / nav.cummax() - 1
        weekly = nav.resample("W-FRI").last().dropna()
        results.append({
            "key": st.key, "name": st.name, "blurb": st.blurb, "rule": st.rule, "risk": st.risk,
            "cadence": st.cadence, "tags": st.tags, "metrics": m, "stress": stress(nav),
            "curve": {"t": [d.strftime("%Y-%m-%d") for d in weekly.index], "v": [round(float(v) * 10000, 2) for v in weekly.values]},
            "dd": {"t": [d.strftime("%Y-%m-%d") for d in weekly.index],
                   "v": [round(float(v), 4) for v in dd.resample("W-FRI").min().reindex(weekly.index).values]},
        })
    return px, results


def validate_leverage(refresh=False):
    """Compare simulated leveraged sleeves to the real ETFs over their live history."""
    px = load(refresh)
    real = data.prices(["SSO", "UPRO", "TQQQ", "QLD"], refresh)
    out = []
    for sleeve, etf in [("SPYx2", "SSO"), ("SPYx3", "UPRO"), ("QQQx2", "QLD"), ("QQQx3", "TQQQ")]:
        sim = sleeve_returns(px, [sleeve])[sleeve]
        act = real[etf].pct_change()
        start = act.dropna().index[0] + pd.Timedelta(days=5)
        both = pd.concat([sim, act], axis=1, keys=["sim", "act"]).loc[start:].dropna()
        yrs = (both.index[-1] - both.index[0]).days / 365.25
        c = (1 + both).prod() ** (1 / yrs) - 1
        out.append({"etf": etf, "since": both.index[0].strftime("%Y-%m-%d"),
                    "sim_cagr": float(c["sim"]), "actual_cagr": float(c["act"]),
                    "daily_corr": float(both.corr().iloc[0, 1])})
    return out
