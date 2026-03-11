from pathlib import Path
import yfinance as yf
import pandas as pd

TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
START_DATE = "2020-01-01"
END_DATE = None  # latest available data(leave it as None)

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_ticker_data(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    # MultiIndex can cause issues when saving/loading parquet, so we flatten it if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    return df


def save_as_parquet(df: pd.DataFrame, ticker: str) -> None:
    output_path = RAW_DATA_DIR / f"{ticker}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Saved {ticker} data to {output_path}")


def main():
    for ticker in TICKERS:
        try:
            print(f"Downloading {ticker}...")
            df = download_ticker_data(ticker, START_DATE, END_DATE)
            save_as_parquet(df, ticker)
            print(df.head())
            print("-" * 50)
        except Exception as e:
            print(f"Failed to process {ticker}: {e}")


if __name__ == "__main__":
    main()