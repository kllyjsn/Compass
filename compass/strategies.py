"""Strategies as functions of daily prices -> daily target weights.

Each strategy returns a DataFrame of target weights (rows = days, columns =
sleeves). A sleeve is an underlying index sleeve, optionally leveraged, e.g.
"SPY" or "SPYx3". "CASH" earns T-bill yield. Weights are decided using the
close of day t and earn returns from t+1 onward (no lookahead).

Each strategy also names the Robinhood-tradeable ticker for every sleeve so
the daily brief can say exactly what to buy.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr, USMemorialDay,
    USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)

# Sleeve -> ticker you would actually buy in Robinhood.
TRADE_TICKER = {
    "SPY": "SPY", "SPYx1.5": "SPY + SSO (50/50 mix)", "SPYx2": "SSO", "SPYx3": "UPRO",
    "QQQ": "QQQ", "QQQx2": "QLD", "QQQx3": "TQQQ",
    "BND": "BND", "TLTx3": "TMF", "SCV": "VBR", "INTL": "VXUS", "CASH": "SGOV",
    "XLB": "XLB", "XLE": "XLE", "XLF": "XLF", "XLI": "XLI", "XLK": "XLK",
    "XLP": "XLP", "XLU": "XLU", "XLV": "XLV", "XLY": "XLY",
}

# Underlying data series used to simulate each base sleeve (longest daily history).
BASE_SERIES = {
    "SPY": "SPY", "QQQ": "QQQ", "BND": "VBMFX", "TLT": "VUSTX", "SCV": "VISVX", "INTL": "VGTSX",
    **{s: s for s in ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]},
}
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


@dataclass
class Strategy:
    key: str
    name: str
    blurb: str
    rule: str
    weights: Callable[[pd.DataFrame], pd.DataFrame]
    risk: str
    cadence: str = "monthly"
    tags: list = field(default_factory=list)


def _const(w):
    return lambda px: pd.DataFrame({k: v for k, v in w.items()}, index=px.index)


class NYSEHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


NYSE_DAY = pd.offsets.CustomBusinessDay(calendar=NYSEHolidays())


def month_end_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    """True on the last trading day of each month. The final row counts only if
    the next NYSE trading day starts a new month, so a partial month never rebalances."""
    nxt = np.append(idx.month[1:], (idx[-1] + NYSE_DAY).month)
    return idx.month != nxt


def _trend(base, risk_on, band=0.0):
    """Hold `risk_on` sleeve while base > 200-day SMA, else cash. Checked daily.

    `band` adds hysteresis: switch out below SMA*(1-band), back in above SMA*(1+band).
    """
    def f(px):
        p = px[base]
        sma = p.rolling(200).mean()
        on = np.zeros(len(p), dtype=bool)
        state = True
        for i, (x, m) in enumerate(zip(p.values, sma.values)):
            if np.isnan(m):
                state = True
            elif state and x < m * (1 - band):
                state = False
            elif not state and x > m * (1 + band):
                state = True
            on[i] = state
        on = pd.Series(on, index=p.index)
        return pd.DataFrame({risk_on: on.astype(float), "CASH": (~on).astype(float)})
    return f


def _dual_momentum(px):
    """Antonacci GEM: monthly. If US 12m return > T-bills, hold the stronger of US/intl; else bonds."""
    lb = 252
    r_us = px["SPY"].pct_change(lb)
    r_int = px["INTL"].pct_change(lb)
    r_cash = (1 + px["_cash"]).rolling(lb).apply(np.prod, raw=True) - 1
    pick = np.where(r_us > r_cash, np.where(r_us >= r_int, "SPY", "INTL"), "BND")
    pick = pd.Series(pick, index=px.index).where(month_end_mask(px.index)).ffill().fillna("SPY")
    return pd.get_dummies(pick).astype(float).reindex(columns=["SPY", "INTL", "BND"], fill_value=0.0)


def _sector_momentum(px, top=3, lb=126):
    """Monthly: equal-weight the top 3 of 9 SPDR sectors by 6-month return; cash if SPY < 200d SMA."""
    mom = px[SECTORS].pct_change(lb)
    me = month_end_mask(px.index)
    w = pd.DataFrame(np.nan, index=px.index, columns=SECTORS + ["CASH"])
    ranks = mom.rank(axis=1, ascending=False)
    trend_on = px["SPY"] > px["SPY"].rolling(200).mean()
    for i in np.where(me)[0]:
        row = ranks.iloc[i]
        if row.isna().all():
            continue
        if not trend_on.iloc[i]:
            w.iloc[i] = 0.0
            w.iloc[i, -1] = 1.0
            continue
        sel = row[row <= top].index
        w.iloc[i] = 0.0
        w.loc[w.index[i], sel] = 1.0 / len(sel)
    return w.ffill().fillna(pd.Series({**{s: 1 / 9 for s in SECTORS}, "CASH": 0.0}))


STRATEGIES = [
    Strategy("spy", "S&P 500 (buy & hold)", "The benchmark. Own the 500 largest US companies forever.",
             "Hold 100% SPY. Never sell.", _const({"SPY": 1.0}), "Market risk only", "never", ["baseline"]),
    Strategy("6040", "60/40 stocks/bonds", "The classic balanced portfolio.",
             "60% SPY, 40% BND, rebalanced monthly.", _const({"SPY": 0.6, "BND": 0.4}), "Lower", "monthly"),
    Strategy("qqq", "Nasdaq-100 (buy & hold)", "Concentrated in large-cap tech.",
             "Hold 100% QQQ.", _const({"QQQ": 1.0}), "High: -83% in 2000-02", "never"),
    Strategy("scv", "Small-cap value (buy & hold)", "Tilt to small, cheap companies (factor premium).",
             "Hold 100% VBR.", _const({"SCV": 1.0}), "Market risk + long stretches of lagging", "never"),
    Strategy("spy15", "S&P 500 at 1.5x", "Modest leverage, rebalanced daily.",
             "Hold 1.5x S&P exposure: 50% SPY + 50% SSO (2x), rebalanced monthly.", _const({"SPYx1.5": 1.0}), "High", "monthly",
             ["leverage"]),
    Strategy("upro", "S&P 500 at 3x (UPRO)", "3x daily-leveraged S&P, held through everything.",
             "Hold 100% UPRO.", _const({"SPYx3": 1.0}), "Extreme: -97% 2000-09 simulated", "never", ["leverage"]),
    Strategy("tqqq", "Nasdaq at 3x (TQQQ)", "3x daily-leveraged Nasdaq-100, held through everything.",
             "Hold 100% TQQQ.", _const({"QQQx3": 1.0}), "Extreme: effectively wiped out in 2000-02", "never", ["leverage"]),
    Strategy("hfea", "55% UPRO / 45% TMF", "'Hedgefundie' leveraged risk parity.",
             "55% UPRO, 45% TMF, rebalanced monthly.", _const({"SPYx3": 0.55, "TLTx3": 0.45}),
             "Extreme: stocks and bonds fell together in 2022", "monthly", ["leverage"]),
    Strategy("trend", "S&P 500 trend filter", "Hold the S&P only while it's above its 200-day average.",
             "If SPY closes above its 200-day moving average: hold SPY. Otherwise: hold SGOV (T-bills).",
             _trend("SPY", "SPY", 0.01), "Medium: cuts deep crashes, whipsaws in choppy markets", "daily check", ["trend"]),
    Strategy("trend2x", "2x S&P with trend filter", "Leverage only in uptrends; T-bills in downtrends.",
             "If SPY closes above its 200-day moving average: hold SSO (2x). Otherwise: hold SGOV.",
             _trend("SPY", "SPYx2", 0.01), "High", "daily check", ["trend", "leverage"]),
    Strategy("trend3x", "3x S&P with trend filter", "The popular 'UPRO + 200-day SMA' strategy.",
             "If SPY closes above its 200-day moving average: hold UPRO (3x). Otherwise: hold SGOV.",
             _trend("SPY", "SPYx3", 0.01), "Very high: big gap-down days hit before the filter reacts", "daily check",
             ["trend", "leverage"]),
    Strategy("trendq2x", "2x Nasdaq with trend filter", "Leveraged Nasdaq only while it's in an uptrend.",
             "If QQQ closes above its 200-day moving average: hold QLD (2x). Otherwise: hold SGOV.",
             _trend("QQQ", "QQQx2", 0.01), "Very high", "daily check", ["trend", "leverage"]),
    Strategy("gem", "Dual momentum (GEM)", "Antonacci's Global Equities Momentum: US vs international vs bonds.",
             "Month end: if SPY's 12-month return beats T-bills, hold whichever of SPY / VXUS did better over 12 months; "
             "otherwise hold BND.", _dual_momentum, "Medium", "monthly", ["momentum"]),
    Strategy("sectors", "Sector momentum", "Ride the 3 strongest S&P sectors; T-bills in downtrends.",
             "Month end: hold the 3 SPDR sector ETFs with the best 6-month return, equal weight. If SPY is below its "
             "200-day average, hold SGOV instead.", _sector_momentum, "Medium-high", "monthly", ["momentum", "trend"]),
]
BY_KEY = {s.key: s for s in STRATEGIES}
