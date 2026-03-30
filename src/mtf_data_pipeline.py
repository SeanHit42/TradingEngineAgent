from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf


TICKERS: List[str] = ["AAPL", "MSFT", "NVDA", "QQQ", "SPY"]

RAW_15M_DIR = Path("data/raw_15m")
RAW_4H_DIR = Path("data/raw_4h")

RAW_15M_DIR.mkdir(parents=True, exist_ok=True)
RAW_4H_DIR.mkdir(parents=True, exist_ok=True)

# yfinance intraday limitation: use recent period only
DOWNLOAD_PERIOD = "60d"
DOWNLOAD_INTERVAL = "15m"

# timezone handling:
# keep as returned by yfinance; then normalize before saving
DROP_TIMEZONE = True

# resample target
RESAMPLE_RULE_4H = "4h"


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def validate_ohlcv(df: pd.DataFrame, ticker: str) -> None:
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing required columns: {missing}")

    if df.empty:
        raise ValueError(f"{ticker}: dataframe is empty")


def download_15m_data(ticker: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=DOWNLOAD_PERIOD,
        interval=DOWNLOAD_INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(f"{ticker}: no data returned from yfinance")

    df = flatten_columns(df)
    validate_ohlcv(df, ticker)

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    if DROP_TIMEZONE:
        try:
            df.index = df.index.tz_localize(None)
        except TypeError:
            # already tz-naive
            pass

    df = df.sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df["Ticker"] = ticker

    return df


def resample_15m_to_4h(df_15m: pd.DataFrame, ticker: str) -> pd.DataFrame:
    ohlc = df_15m[["Open", "High", "Low", "Close"]].resample(
        RESAMPLE_RULE_4H,
        label="right",
        closed="right",
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    })

    vol = df_15m[["Volume"]].resample(
        RESAMPLE_RULE_4H,
        label="right",
        closed="right",
    ).sum()

    df_4h = pd.concat([ohlc, vol], axis=1)

    # drop partial / empty bars
    df_4h = df_4h.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df_4h = df_4h[df_4h["Volume"].notna()].copy()

    # remove zero-volume bars if any appear from empty windows
    df_4h = df_4h[df_4h["Volume"] > 0].copy()

    df_4h["Ticker"] = ticker
    return df_4h


def save_parquet(df: pd.DataFrame, path: Path) -> None:

    out = df.copy()

    out = out.reset_index()

    if "index" in out.columns:
        out = out.rename(columns={"index": "Date"})

    if "Datetime" in out.columns:
        out = out.rename(columns={"Datetime": "Date"})

    if "Date" not in out.columns:
        raise ValueError("Date column missing after reset_index")

    out["Date"] = pd.to_datetime(out["Date"])

    out = out[
        ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker"]
    ]

    out.to_parquet(path, index=False)

def process_ticker(ticker: str) -> None:
    print(f"Downloading 15m data for {ticker}...")
    df_15m = download_15m_data(ticker)

    path_15m = RAW_15M_DIR / f"{ticker}.parquet"
    save_parquet(df_15m, path_15m)
    print(f"Saved 15m -> {path_15m}")

    print(f"Resampling 4h data for {ticker}...")
    df_4h = resample_15m_to_4h(df_15m, ticker)

    path_4h = RAW_4H_DIR / f"{ticker}.parquet"
    save_parquet(df_4h, path_4h)
    print(f"Saved 4h -> {path_4h}")

    print(f"{ticker}: 15m rows={len(df_15m)}, 4h rows={len(df_4h)}")
    print("-" * 60)


def main() -> None:
    for ticker in TICKERS:
        try:
            process_ticker(ticker)
        except Exception as e:
            print(f"Failed on {ticker}: {e}")
            print("-" * 60)


if __name__ == "__main__":
    main()