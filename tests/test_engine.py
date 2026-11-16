import numpy as np
import pandas as pd

from compass.backtest import run, sleeve_returns
from compass.strategies import BY_KEY, month_end_mask


def _px(n=600, seed=0):
    idx = pd.bdate_range("2001-01-01", periods=n)
    rng = np.random.default_rng(seed)
    cols = ["SPY", "QQQ", "BND", "TLT", "SCV", "INTL", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, (n, len(cols))), axis=0)), index=idx, columns=cols)
    px["_cash"] = 0.0001
    return px


def test_month_end_mask_ignores_partial_final_month():
    idx = pd.bdate_range("2024-01-01", "2024-02-14")
    m = month_end_mask(idx)
    assert m[idx.get_loc(pd.Timestamp("2024-01-31"))]
    assert not m[-1] and m.sum() == 1


def test_leverage_math():
    px = _px()
    r = sleeve_returns(px, ["SPY", "SPYx3"])
    expected = 3 * r["SPY"] - 2 * (0.0001 + 0.005 / 252) - 0.009 / 252
    assert np.allclose(r["SPYx3"].iloc[1:], expected.iloc[1:])


def test_no_lookahead_future_prices_do_not_change_past_nav():
    px = _px()
    cut = 400
    a, _, _ = run(BY_KEY["trend3x"], px, start=px.index[0])
    shocked = px.copy()
    shocked.iloc[cut:, :-1] *= 0.5
    b, _, _ = run(BY_KEY["trend3x"], shocked, start=px.index[0])
    assert np.allclose(a.iloc[:cut], b.iloc[:cut])


def test_buy_and_hold_matches_index():
    px = _px()
    nav, trades, _ = run(BY_KEY["spy"], px, start=px.index[0])
    assert trades == 0
    assert np.isclose(nav.iloc[-1], px["SPY"].iloc[-1] / px["SPY"].iloc[0])
