"""Load QQQ daily prices.

Priority:
1. ``source="csv"`` (default) - read the bundled ``data/QQQ.csv``
   (split-adjusted daily open/close, 1999-2025, originally a MacroTrends export).
2. ``source="yfinance"`` - download fresh data via yfinance.
3. ``source="nasdaq"`` - pull from Nasdaq's public historical API.
4. ``source="<path>.csv"`` - any CSV you supply with date + close columns.

In a locked-down/offline environment only option 1 is guaranteed to work; the
others need open outbound network access (run them on your own machine to get
data right up to today).
"""

from __future__ import annotations

import io
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLED_CSV = os.path.join(HERE, "data", "QQQ.csv")


def _from_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    # be liberal about column names
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date") or cols.get("datetime") or df.columns[0]
    close_col = (cols.get("close") or cols.get("close/last") or cols.get("adj close")
                 or cols.get("qqq"))
    if close_col is None:
        raise ValueError(f"No close column found in {path}; columns={list(df.columns)}")
    s = pd.Series(
        pd.to_numeric(df[close_col].astype(str).str.replace("$", "", regex=False),
                      errors="coerce").values,
        index=pd.to_datetime(df[date_col]),
        name="QQQ",
    ).sort_index().dropna()
    return s[~s.index.duplicated(keep="last")]


def _from_yfinance(start: str) -> pd.Series:
    import yfinance as yf
    df = yf.download("QQQ", start=start, auto_adjust=True, progress=False)
    s = df["Close"]
    if hasattr(s, "columns"):          # newer yfinance returns a frame
        s = s.iloc[:, 0]
    return s.rename("QQQ").dropna()


def _from_nasdaq(start: str, end: str) -> pd.Series:
    import requests
    url = (f"https://api.nasdaq.com/api/quote/QQQ/historical"
           f"?assetclass=etf&fromdate={start}&todate={end}&limit=99999")
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    rows = requests.get(url, headers=headers, timeout=30).json()["data"]["tradesTable"]["rows"]
    df = pd.DataFrame(rows)
    s = pd.Series(
        pd.to_numeric(df["close"].str.replace("$", "", regex=False)).values,
        index=pd.to_datetime(df["date"]), name="QQQ").sort_index()
    return s


def load_qqq(source: str = "csv", start: str = "1999-01-01",
             end: str = "2026-12-31") -> pd.Series:
    """Return a daily QQQ close-price series (pandas Series indexed by date)."""
    if source == "csv":
        return _from_csv(BUNDLED_CSV)
    if source == "yfinance":
        return _from_yfinance(start)
    if source == "nasdaq":
        return _from_nasdaq(start, end)
    if source.endswith(".csv"):
        return _from_csv(source)
    raise ValueError(f"Unknown source: {source!r}")


if __name__ == "__main__":
    s = load_qqq()
    print(f"QQQ: {len(s)} rows, {s.index.min().date()} -> {s.index.max().date()}")
    print(s.tail())
