from pathlib import Path
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def validate_ohlcv(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty")


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    validate_ohlcv(df)

    df = df.sort_values("Date").copy()

    df["returns"] = df["Close"].pct_change()
    df["ema_fast"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
    df["ema_slow"] = EMAIndicator(close=df["Close"], window=50).ema_indicator()
    df["ema_trend"] = EMAIndicator(close=df["Close"], window=200).ema_indicator()
    df["rsi"] = RSIIndicator(close=df["Close"], window=14).rsi()
    df["atr"] = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    ).average_true_range()

    df = df.dropna(subset=["ema_fast", "ema_slow", "ema_trend", "rsi", "atr"]).copy()
    df = df.reset_index(drop=True)

    return df


def process_file(file_path: Path):
    print(f"Processing {file_path.name}")
    df = pd.read_parquet(file_path).copy()
    df = compute_indicators(df)

    output_path = PROCESSED_DIR / file_path.name
    df.to_parquet(output_path, index=False)
    print(f"Saved processed data to {output_path}")


def main():
    files = list(RAW_DIR.glob("*.parquet"))
    for file in files:
        process_file(file)


if __name__ == "__main__":
    main()