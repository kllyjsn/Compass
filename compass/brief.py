"""Daily brief: what the chosen strategy says to hold today, in Robinhood tickers."""
import numpy as np
import pandas as pd

from .backtest import load, run
from .strategies import BY_KEY, TRADE_TICKER, month_end_mask


def _fmt_alloc(w: pd.Series, value=None):
    lines = []
    for sleeve, wt in w[w > 1e-6].sort_values(ascending=False).items():
        amt = f" (${value * wt:,.0f})" if value else ""
        lines.append(f"- **{TRADE_TICKER.get(sleeve, sleeve)}**: {wt:.0%}{amt}")
    return "\n".join(lines)


def _trend_context(px, base):
    p = px[base].dropna()
    sma = p.rolling(200).mean()
    last, m = p.iloc[-1], sma.iloc[-1]
    return (f"{base} closed at **${last:,.2f}**, {last / m - 1:+.1%} vs its 200-day average (${m:,.2f}). "
            f"Exit level (1% below average): **${m * 0.99:,.2f}**. Re-entry level (1% above): **${m * 1.01:,.2f}**.")


def build(key: str, value: float | None = None, refresh: bool = True) -> str:
    st = BY_KEY[key]
    px = load(refresh)
    nav, _, target = run(st, px, start=px.index[-600])
    today, prev = target.iloc[-1], target.iloc[-2]
    asof = px.index[-1].strftime("%a %b %d, %Y")
    changed = not np.allclose(today.values, prev.values)
    month_end_next = month_end_mask(px.index)[-1]
    monthly = st.cadence == "monthly"

    if changed:
        sells = [TRADE_TICKER.get(s, s) for s in prev.index if prev[s] > today[s] + 1e-6]
        buys = [TRADE_TICKER.get(s, s) for s in today.index if today[s] > prev[s] + 1e-6]
        action = (f"**TRADE TODAY.** Signal changed at the last close. Sell {', '.join(sells) or 'nothing'}; "
                  f"buy {', '.join(buys) or 'nothing'} to match the target below.")
    elif monthly and month_end_next:
        action = "**REBALANCE.** Last trading day of the month: bring holdings back to the target weights below."
    else:
        action = "**NO TRADE.** Signal unchanged. Hold your current positions."

    ctx = ""
    if "trend" in st.tags:
        base = "QQQ" if "QQQ" in st.rule else "SPY"
        ctx = _trend_context(px, base)
    dd = nav.iloc[-1] / nav.cummax().iloc[-1] - 1
    return f"""## Compass daily brief: {st.name}
Data as of the close on {asof}.

{action}

**Target allocation**
{_fmt_alloc(today, value)}

**Rule:** {st.rule}
{ctx}

Strategy is {dd:.1%} from its recent high.

_Mechanical output of the rules you chose, not personalized financial advice. You place every trade yourself._
"""


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy", choices=sorted(BY_KEY))
    ap.add_argument("--value", type=float, help="portfolio value in dollars, to show dollar amounts")
    a = ap.parse_args()
    print(build(a.strategy, a.value))
