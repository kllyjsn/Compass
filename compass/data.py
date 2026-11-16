"""Daily adjusted closes from Yahoo Finance, cached as CSV."""
import os
import time
from pathlib import Path

import pandas as pd
import requests

CACHE = Path(os.environ.get("COMPASS_CACHE", Path(__file__).resolve().parent.parent / "data"))
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{}"


def _download(symbol: str) -> pd.Series:
    r = requests.get(
        URL.format(symbol),
        params={"period1": 0, "period2": int(time.time()) + 86400, "interval": "1d", "events": "div,split"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
    ind = res["indicators"]
    vals = ind["adjclose"][0]["adjclose"] if "adjclose" in ind else ind["quote"][0]["close"]
    s = pd.Series(vals, index=idx, name=symbol, dtype=float).dropna()
    return s[~s.index.duplicated(keep="last")]


def series(symbol: str, refresh: bool = False, max_age_hours: float = 12) -> pd.Series:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{symbol.replace('^', '_')}.csv"
    fresh = path.exists() and time.time() - path.stat().st_mtime < max_age_hours * 3600
    if fresh and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].rename(symbol)
    s = _download(symbol)
    s.to_frame().to_csv(path)
    return s


def prices(symbols, refresh: bool = False) -> pd.DataFrame:
    return pd.concat([series(s, refresh) for s in symbols], axis=1)


def cash_daily_returns(index: pd.DatetimeIndex, refresh: bool = False) -> pd.Series:
    """Daily T-bill return from the 13-week yield (^IRX, annualized percent)."""
    irx = series("^IRX", refresh).reindex(index).ffill().bfill()
    return irx / 100 / 252
