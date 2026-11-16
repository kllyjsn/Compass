"""Export backtest results as JSON for the research report."""
import json
from pathlib import Path

import pandas as pd

from .backtest import STRESS, load, metrics, run, stress, validate_leverage
from .strategies import STRATEGIES

WINDOWS = {"2000": "2000-01-03", "2007": "2007-01-03", "2010": "2010-01-04", "2016": "2016-11-14"}


def export(path="report/results.json", refresh=False):
    px = load(refresh)
    out = {"asof": px.index[-1].strftime("%Y-%m-%d"), "windows": WINDOWS, "stress_periods": list(STRESS),
           "validation": validate_leverage(), "strategies": []}
    for st in STRATEGIES:
        nav, trades, _ = run(st, px, WINDOWS["2000"])
        dd_daily = nav / nav.cummax() - 1
        weekly = nav.resample("W-FRI").last().dropna()
        dd = dd_daily.resample("W-FRI").min().reindex(weekly.index)
        starts = [nav.index[nav.index >= s][0] for s in WINDOWS.values()]
        weekly = pd.concat([weekly, nav.loc[starts]]).sort_index()
        dd = pd.concat([dd, dd_daily.loc[starts]]).sort_index()
        per_window = {}
        for w, start in WINDOWS.items():
            n, t, _ = run(st, px, start)
            m = metrics(n, t, px["_cash"])
            m.pop("yearly")
            per_window[w] = m
        full = metrics(nav, trades, px["_cash"])
        out["strategies"].append({
            "key": st.key, "name": st.name, "blurb": st.blurb, "rule": st.rule, "risk": st.risk,
            "cadence": st.cadence, "tags": st.tags, "metrics": per_window, "yearly": full["yearly"],
            "stress": stress(nav),
            "t": [d.strftime("%Y-%m-%d") for d in weekly.index],
            "v": [round(float(v), 5) for v in weekly.values],
            "dd": [round(float(v), 4) for v in dd.values],
        })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    o = export()
    for s in o["strategies"]:
        m = s["metrics"]
        print(f'{s["name"]:32s}', "  ".join(f'{w}: {m[w]["cagr"]*100:5.1f}%/{m[w]["max_dd"]*100:4.0f}%' for w in m))
