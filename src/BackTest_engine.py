from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
SIGNALS_DIR = Path("data/signals")

TRANSACTION_COST = 0.001  # 0.1% לכל כניסה/יציאה


def build_long_only_position(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["position"] = 0

    in_position = False
    positions = []

    for signal in df["signal"]:
        if signal == 1 and not in_position:
            in_position = True
        elif signal == -1 and in_position:
            in_position = False

        positions.append(1 if in_position else 0)

    df["position"] = positions
    return df


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve / rolling_max) - 1
    return drawdown


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["Date", "Close", "signal"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values("Date").copy()

    df = build_long_only_position(df)

    df["returns"] = df["Close"].pct_change().fillna(0)

    # משתמשים בפוזיציה של יום קודם כדי להימנע מ-lookahead bias
    df["strategy_returns_gross"] = df["position"].shift(1).fillna(0) * df["returns"]

    # עלות מסחר בכל שינוי פוזיציה
    df["trade"] = df["position"].diff().abs().fillna(0)
    df["transaction_cost"] = df["trade"] * TRANSACTION_COST

    df["strategy_returns_net"] = df["strategy_returns_gross"] - df["transaction_cost"]

    df["equity_curve"] = (1 + df["strategy_returns_net"]).cumprod()
    df["buy_and_hold"] = (1 + df["returns"]).cumprod()

    df["drawdown"] = compute_drawdown(df["equity_curve"])

    return df


def print_summary(df: pd.DataFrame) -> None:
    final_equity = df["equity_curve"].iloc[-1]
    strategy_return_pct = (final_equity - 1) * 100

    buy_hold_equity = df["buy_and_hold"].iloc[-1]
    buy_hold_return_pct = (buy_hold_equity - 1) * 100

    max_drawdown_pct = df["drawdown"].min() * 100
    num_entries = int((df["trade"] == 1).sum())

    print("\n===== BACKTEST SUMMARY =====")
    print(f"Final equity: {final_equity:.4f}")
    print(f"Strategy return: {strategy_return_pct:.2f}%")
    print(f"Buy & Hold return: {buy_hold_return_pct:.2f}%")
    print(f"Max drawdown: {max_drawdown_pct:.2f}%")
    print(f"Number of position changes: {num_entries}")



def plot_equity(df, ticker):

    plt.figure(figsize=(10,6))

    plt.plot(df["Date"], df["equity_curve"], label="Strategy")
    plt.plot(df["Date"], df["buy_and_hold"], label="Buy & Hold")

    plt.title(f"{ticker} Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity")

    plt.legend()
    plt.grid(True)

    plt.show()
    

def main():

    files = list(SIGNALS_DIR.glob("*.parquet"))

    results = []

    for file_path in files:

        ticker = file_path.stem

        df = pd.read_parquet(file_path)
        df = run_backtest(df)
        plot_equity(df, ticker)

        final_equity = df["equity_curve"].iloc[-1]
        strategy_return = (final_equity - 1) * 100

        buy_hold = (df["buy_and_hold"].iloc[-1] - 1) * 100
        max_dd = df["drawdown"].min() * 100
        num_trades = df["trade"].sum()

        results.append({
            "ticker": ticker,
            "strategy_return_%": round(strategy_return, 2),
            "buy_hold_%": round(buy_hold, 2),
            "max_drawdown_%": round(max_dd, 2),
            "num_trades": num_trades
        })
    
    
    results_df = pd.DataFrame(results)

    print("\n===== STRATEGY RESULTS =====")
    print(results_df)

if __name__ == "__main__":
    main()