import pandas as pd

df = pd.read_parquet("data/signals/AAPL.parquet")

print(df[["Date","Close","ema_fast","ema_slow","ema_trend","rsi","atr","signal"]].tail(20))

print("\nSignal distribution:")
print(df["signal"].value_counts(dropna=False))