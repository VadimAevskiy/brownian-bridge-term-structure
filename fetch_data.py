#!/usr/bin/env python3
"""
fetch_data.py -- Download convergence event data from Yahoo Finance.

Only downloads data for events with a KNOWN convergence date T and target value.
The Brownian bridge model requires both -- without them the model is not applicable.

Events:
    1. Croatia EUR/HRK -> 7.5345 by 01.01.2023 (euro adoption)
    2. Broadcom / VMware: VMW -> $142.50 by 22.11.2023 (merger close)
    3. T-Mobile / Sprint: S -> 0.10256 * TMUS by 01.04.2020 (merger close)

Usage:
    python fetch_data.py
    python backtest_yahoo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

DATA_DIR = Path(__file__).resolve().parent / "data" / "yahoo"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download(ticker: str, start: str, end: str, interval: str = "1wk") -> pd.DataFrame:
    """Download with error handling."""
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df is not None and len(df) > 0:
            print(f"    OK  {ticker:12s}  {len(df)} rows  [{df.index[0].date()} -- {df.index[-1].date()}]")
            return df
        print(f"    --  {ticker:12s}  (no data)")
    except Exception as e:
        print(f"    ERR {ticker:12s}  {e}")
    return pd.DataFrame()


def fetch_croatia():
    """
    Croatia adopted the euro on 1 January 2023.
    EUR/HRK was fixed at 7.5345.  This is a textbook Brownian bridge:
    known target, known date, observable convergence path.
    """
    print("\n[1/3] Croatia euro adoption: EUR/HRK -> 7.5345 by 01.01.2023")

    fx = download("EURHRK=X", "2020-01-01", "2023-01-15")
    if len(fx) < 10:
        return

    close = fx["Close"].squeeze().dropna()
    df = pd.DataFrame({"close": close})
    df["target"] = 7.5345
    df["spread"] = df["close"] - df["target"]

    adoption = pd.Timestamp("2023-01-01")
    df["weeks_to_T"] = [(adoption - d).days / 7 for d in df.index]
    df["weeks_to_T"] = df["weeks_to_T"].clip(lower=0)

    path = DATA_DIR / "croatia_eurhrk.csv"
    df.to_csv(path)
    print(f"    >>> {path.name}  ({len(df)} rows)")


def fetch_broadcom_vmware():
    """
    Broadcom acquired VMware.  Deal closed 22.11.2023.
    Cash offer ~$142.50 per VMW share.
    The merger basis (VMW price - offer) should converge to 0 by close.
    """
    print("\n[2/3] Broadcom / VMware: VMW -> $142.50 by 22.11.2023")

    vmw = download("VMW", "2022-06-01", "2023-11-22")
    if len(vmw) < 10:
        return

    close = vmw["Close"].squeeze().dropna()
    offer = 142.50
    df = pd.DataFrame({"close": close})
    df["target"] = offer
    df["spread"] = df["close"] - offer
    df["spread_pct"] = df["spread"] / offer * 100

    close_date = pd.Timestamp("2023-11-22")
    df["weeks_to_T"] = [(close_date - d).days / 7 for d in df.index]
    df["weeks_to_T"] = df["weeks_to_T"].clip(lower=0)

    path = DATA_DIR / "ma_broadcom_vmware.csv"
    df.to_csv(path)
    print(f"    >>> {path.name}  ({len(df)} rows)")


def fetch_tmobile_sprint():
    """
    T-Mobile acquired Sprint.  Deal closed 01.04.2020.
    Exchange ratio: 0.10256 TMUS shares per Sprint share.
    Merger basis = Sprint price - ratio * TMUS price -> 0 by close.
    """
    print("\n[3/3] T-Mobile / Sprint: basis -> 0 by 01.04.2020")

    tmus = download("TMUS", "2019-06-01", "2020-04-10")
    sprint = download("S", "2019-06-01", "2020-04-01")

    if len(tmus) < 5 or len(sprint) < 5:
        print("    Sprint (S) may not be available after delisting.")
        return

    ratio = 0.10256
    merged = pd.DataFrame({
        "sprint": sprint["Close"].squeeze(),
        "tmus": tmus["Close"].squeeze(),
    }).dropna()

    merged["implied"] = merged["tmus"] * ratio
    merged["spread"] = merged["sprint"] - merged["implied"]
    merged["spread_pct"] = merged["spread"] / merged["implied"] * 100

    close_date = pd.Timestamp("2020-04-01")
    merged["weeks_to_T"] = [(close_date - d).days / 7 for d in merged.index]
    merged["weeks_to_T"] = merged["weeks_to_T"].clip(lower=0)

    path = DATA_DIR / "ma_tmobile_sprint.csv"
    merged.to_csv(path)
    print(f"    >>> {path.name}  ({len(merged)} rows)")


def main():
    print("=" * 65)
    print("  Convergence Event Data Downloader (Yahoo Finance)")
    print("  Only events with known date T and known target value")
    print("=" * 65)

    fetch_croatia()
    fetch_broadcom_vmware()
    fetch_tmobile_sprint()

    print("\n" + "=" * 65)
    downloaded = sorted(DATA_DIR.glob("*.csv"))
    print(f"  {len(downloaded)} files saved to {DATA_DIR}/")
    for f in downloaded:
        print(f"    {f.name}")
    print("=" * 65)
    print("\n  Next: python backtest_yahoo.py\n")


if __name__ == "__main__":
    main()
