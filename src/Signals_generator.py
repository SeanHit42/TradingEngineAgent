from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
SIGNALS_DIR = Path("data/signals")
SIGNALS_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = ["Close", "ema_fast", "ema_slow", "ema_trend", "rsi", "atr"]


def validate_feature_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    validate_feature_columns(df)

    df = df.sort_values("Date").copy()
    df["signal"] = 0

    bullish_cross = (df["ema_fast"] > df["ema_slow"]) & (
        df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)
    )
    bearish_cross = (df["ema_fast"] < df["ema_slow"]) & (
        df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)
    )

    trend_filter = df["Close"] > df["ema_trend"]
    momentum_filter = (df["rsi"] > 50) & (df["rsi"] < 70)
    volatility_filter = df["atr"] > df["atr"].rolling(20).mean()

    buy_condition = bullish_cross & trend_filter & momentum_filter & volatility_filter
    sell_condition = bearish_cross

    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1

    return df.reset_index(drop=True)


def process_file(file_path: Path):
    print(f"Generating signals for {file_path.name}")
    df = pd.read_parquet(file_path).copy()
    df = generate_signals(df)

    output_path = SIGNALS_DIR / file_path.name
    df.to_parquet(output_path, index=False)
    print(f"Saved signals to {output_path}")


def main():
    files = list(PROCESSED_DIR.glob("*.parquet"))
    for file in files:
        process_file(file)


if __name__ == "__main__":
    main()