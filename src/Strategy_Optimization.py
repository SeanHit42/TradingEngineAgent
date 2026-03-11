from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange

RAW_DIR = Path("data/raw")

TICKER = "AAPL"
TRANSACTION_COST = 0.001  # 0.1%
MIN_TRADES = 5

# ===== Parameter Grid =====
FAST_WINDOWS = [10, 20]
SLOW_WINDOWS = [30, 50]
TREND_WINDOWS = [100, 200]

RSI_MINS = [45, 50]
RSI_MAXS = [65, 70]

ADX_MINS = [15, 20, 25]

STOP_ATR_MULTIPLIERS = [1.0, 1.5]
TARGET_ATR_MULTIPLIERS = [2.0, 3.0]

SWEEP_LOOKBACKS = [5, 10]
REL_VOL_THRESHOLDS = [1.1, 1.25, 1.5]


def validate_ohlcv(df: pd.DataFrame) -> None:
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Input DataFrame is empty")


def load_raw_ticker_data(ticker: str) -> pd.DataFrame:
    file_path = RAW_DIR / f"{ticker}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    df = pd.read_parquet(file_path)
    validate_ohlcv(df)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_features(
    df: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    trend_window: int,
    sweep_lookback: int,
    rsi_window: int = 14,
    atr_window: int = 14,
    adx_window: int = 14,
    volume_window: int = 20,
) -> pd.DataFrame:
    validate_ohlcv(df)

    df = df.sort_values("Date").copy()

    # Trend / momentum / volatility
    df["ema_fast"] = EMAIndicator(close=df["Close"], window=fast_window).ema_indicator()
    df["ema_slow"] = EMAIndicator(close=df["Close"], window=slow_window).ema_indicator()
    df["ema_trend"] = EMAIndicator(close=df["Close"], window=trend_window).ema_indicator()
    df["rsi"] = RSIIndicator(close=df["Close"], window=rsi_window).rsi()

    atr_indicator = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=atr_window,
    )
    df["atr"] = atr_indicator.average_true_range()

    adx_indicator = ADXIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=adx_window,
    )
    df["adx"] = adx_indicator.adx()

    # Volume / liquidity proxy
    df["vol_ma"] = df["Volume"].rolling(volume_window).mean()
    df["relative_volume"] = df["Volume"] / df["vol_ma"]

    # Previous liquidity zones
    df["prev_low_zone"] = df["Low"].rolling(sweep_lookback).min().shift(1)
    df["prev_high_zone"] = df["High"].rolling(sweep_lookback).max().shift(1)

    # Daily returns for buy-and-hold reference
    df["returns"] = df["Close"].pct_change().fillna(0)

    df = df.dropna(
        subset=[
            "ema_fast",
            "ema_slow",
            "ema_trend",
            "rsi",
            "atr",
            "adx",
            "vol_ma",
            "relative_volume",
            "prev_low_zone",
            "prev_high_zone",
        ]
    ).copy()

    df = df.reset_index(drop=True)
    return df


def bullish_sweep_reclaim(row: pd.Series) -> bool:
    """
    Detects a simple daily liquidity sweep of prior lows:
    - intraday low goes below previous low zone
    - but close reclaims back above that zone
    """
    return (row["Low"] < row["prev_low_zone"]) and (row["Close"] > row["prev_low_zone"])


def bearish_breakdown_signal(row: pd.Series) -> bool:
    """
    A simple bearish failure / exit signal:
    - close below slow EMA
    OR
    - close below previous low zone
    """
    return (row["Close"] < row["ema_slow"]) or (row["Close"] < row["prev_low_zone"])


def generate_entry_signal(
    row: pd.Series,
    prev_row: pd.Series,
    rsi_min: float,
    rsi_max: float,
    adx_min: float,
    rel_vol_threshold: float,
) -> bool:
    fast_above_slow = row["ema_fast"] > row["ema_slow"]
    bullish_cross_or_trend = fast_above_slow and (row["Close"] > row["ema_trend"])

    momentum_filter = rsi_min < row["rsi"] < rsi_max
    strength_filter = row["adx"] > adx_min
    liquidity_filter = row["relative_volume"] > rel_vol_threshold

    sweep_filter = bullish_sweep_reclaim(row)

    # Optional extra confirmation:
    # today closes above yesterday close after sweep
    reclaim_confirmation = row["Close"] > prev_row["Close"]

    return (
        bullish_cross_or_trend
        and momentum_filter
        and strength_filter
        and liquidity_filter
        and sweep_filter
        and reclaim_confirmation
    )


def generate_exit_signal(row: pd.Series, prev_row: pd.Series) -> bool:
    bearish_cross = (row["ema_fast"] < row["ema_slow"]) and (prev_row["ema_fast"] >= prev_row["ema_slow"])
    breakdown = bearish_breakdown_signal(row)
    return bearish_cross or breakdown


def run_trade_backtest(
    df: pd.DataFrame,
    rsi_min: float,
    rsi_max: float,
    adx_min: float,
    rel_vol_threshold: float,
    stop_atr_mult: float,
    target_atr_mult: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "ema_fast", "ema_slow", "ema_trend", "rsi", "atr", "adx",
        "relative_volume", "prev_low_zone", "prev_high_zone", "returns"
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for trade backtest: {missing}")

    df = df.sort_values("Date").reset_index(drop=True).copy()

    equity = 1.0
    equity_curve = []
    in_position = False

    entry_price = None
    entry_date = None
    stop_price = None
    target_price = None
    entry_atr = None

    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        current_date = row["Date"]

        if i == 0:
            equity_curve.append(equity)
            continue

        prev_row = df.iloc[i - 1]

        if not in_position:
            should_enter = generate_entry_signal(
                row=row,
                prev_row=prev_row,
                rsi_min=rsi_min,
                rsi_max=rsi_max,
                adx_min=adx_min,
                rel_vol_threshold=rel_vol_threshold,
            )

            if should_enter:
                entry_price = row["Close"]
                entry_date = current_date
                entry_atr = row["atr"]

                stop_price = entry_price - stop_atr_mult * entry_atr
                target_price = entry_price + target_atr_mult * entry_atr
                in_position = True

                # half cost on entry
                equity *= (1 - TRANSACTION_COST / 2)

            equity_curve.append(equity)
            continue

        # manage open trade
        exit_reason = None
        exit_price = None

        hit_stop = row["Low"] <= stop_price
        hit_target = row["High"] >= target_price

        if hit_stop and hit_target:
            # conservative assumption
            exit_reason = "stop_and_target_same_bar_stop_first"
            exit_price = stop_price
        elif hit_stop:
            exit_reason = "stop"
            exit_price = stop_price
        elif hit_target:
            exit_reason = "target"
            exit_price = target_price
        elif generate_exit_signal(row=row, prev_row=prev_row):
            exit_reason = "signal_exit"
            exit_price = row["Close"]

        if exit_reason is not None:
            gross_trade_return = (exit_price / entry_price) - 1
            net_trade_return = gross_trade_return - (TRANSACTION_COST / 2)

            equity *= (1 + net_trade_return)

            trades.append({
                "entry_date": entry_date,
                "exit_date": current_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "entry_atr": entry_atr,
                "gross_return_pct": gross_trade_return * 100,
                "net_return_pct": net_trade_return * 100,
                "exit_reason": exit_reason,
                "holding_days": (pd.Timestamp(current_date) - pd.Timestamp(entry_date)).days,
            })

            in_position = False
            entry_price = None
            entry_date = None
            stop_price = None
            target_price = None
            entry_atr = None

            equity_curve.append(equity)
            continue

        # mark to market
        close_to_close_return = row["Close"] / prev_row["Close"] - 1
        equity *= (1 + close_to_close_return)
        equity_curve.append(equity)

    result_df = df.copy()
    result_df["equity_curve"] = pd.Series(equity_curve, index=result_df.index)
    result_df["rolling_max"] = result_df["equity_curve"].cummax()
    result_df["drawdown"] = (result_df["equity_curve"] / result_df["rolling_max"]) - 1
    result_df["buy_and_hold"] = (1 + result_df["returns"]).cumprod()

    trades_df = pd.DataFrame(trades)
    return result_df, trades_df


def summarize_trades(result_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    strategy_return_pct = (float(result_df["equity_curve"].iloc[-1]) - 1) * 100
    buy_hold_return_pct = (float(result_df["buy_and_hold"].iloc[-1]) - 1) * 100
    max_drawdown_pct = float(result_df["drawdown"].min() * 100)

    num_trades = len(trades_df)

    if num_trades == 0:
        return {
            "strategy_return_%": round(strategy_return_pct, 2),
            "buy_hold_%": round(buy_hold_return_pct, 2),
            "max_drawdown_%": round(max_drawdown_pct, 2),
            "num_trades": 0,
            "win_rate_%": 0.0,
            "avg_win_%": 0.0,
            "avg_loss_%": 0.0,
            "avg_rr": np.nan,
            "profit_factor": 0.0,
            "expectancy_%": 0.0,
            "score": -999999.0,
        }

    wins = trades_df[trades_df["net_return_pct"] > 0]
    losses = trades_df[trades_df["net_return_pct"] <= 0]

    win_rate = len(wins) / num_trades
    loss_rate = 1 - win_rate

    avg_win = wins["net_return_pct"].mean() if not wins.empty else 0.0
    avg_loss = losses["net_return_pct"].mean() if not losses.empty else 0.0

    gross_profit = wins["net_return_pct"].sum() if not wins.empty else 0.0
    gross_loss_abs = abs(losses["net_return_pct"].sum()) if not losses.empty else 0.0

    if gross_loss_abs == 0:
        profit_factor = np.inf if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss_abs

    expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))
    avg_rr = (avg_win / abs(avg_loss)) if avg_loss < 0 else np.nan

    finite_profit_factor = 10.0 if np.isinf(profit_factor) else profit_factor

    score = (
        strategy_return_pct * 0.30
        + expectancy * 8.0
        + finite_profit_factor * 8.0
        + win_rate * 100 * 0.15
        + max_drawdown_pct * 0.10
    )

    if num_trades < MIN_TRADES:
        score -= 1000

    return {
        "strategy_return_%": round(strategy_return_pct, 2),
        "buy_hold_%": round(buy_hold_return_pct, 2),
        "max_drawdown_%": round(max_drawdown_pct, 2),
        "num_trades": int(num_trades),
        "win_rate_%": round(win_rate * 100, 2),
        "avg_win_%": round(float(avg_win), 2) if not np.isnan(avg_win) else 0.0,
        "avg_loss_%": round(float(avg_loss), 2) if not np.isnan(avg_loss) else 0.0,
        "avg_rr": round(float(avg_rr), 2) if not np.isnan(avg_rr) else np.nan,
        "profit_factor": round(float(finite_profit_factor), 2),
        "expectancy_%": round(float(expectancy), 2),
        "score": round(float(score), 2),
    }


def main() -> None:
    raw_df = load_raw_ticker_data(TICKER)
    results = []

    param_grid = product(
        FAST_WINDOWS,
        SLOW_WINDOWS,
        TREND_WINDOWS,
        RSI_MINS,
        RSI_MAXS,
        ADX_MINS,
        STOP_ATR_MULTIPLIERS,
        TARGET_ATR_MULTIPLIERS,
        SWEEP_LOOKBACKS,
        REL_VOL_THRESHOLDS,
    )

    for (
        fast_window,
        slow_window,
        trend_window,
        rsi_min,
        rsi_max,
        adx_min,
        stop_atr_mult,
        target_atr_mult,
        sweep_lookback,
        rel_vol_threshold,
    ) in param_grid:
        if fast_window >= slow_window:
            continue
        if slow_window >= trend_window:
            continue
        if rsi_min >= rsi_max:
            continue
        if target_atr_mult <= stop_atr_mult:
            continue

        try:
            features_df = compute_features(
                raw_df,
                fast_window=fast_window,
                slow_window=slow_window,
                trend_window=trend_window,
                sweep_lookback=sweep_lookback,
            )

            result_df, trades_df = run_trade_backtest(
                features_df,
                rsi_min=rsi_min,
                rsi_max=rsi_max,
                adx_min=adx_min,
                rel_vol_threshold=rel_vol_threshold,
                stop_atr_mult=stop_atr_mult,
                target_atr_mult=target_atr_mult,
            )

            summary = summarize_trades(result_df, trades_df)

            if summary["num_trades"] < MIN_TRADES:
                continue

            results.append({
                "ticker": TICKER,
                "fast_window": fast_window,
                "slow_window": slow_window,
                "trend_window": trend_window,
                "rsi_min": rsi_min,
                "rsi_max": rsi_max,
                "adx_min": adx_min,
                "stop_atr_mult": stop_atr_mult,
                "target_atr_mult": target_atr_mult,
                "sweep_lookback": sweep_lookback,
                "rel_vol_threshold": rel_vol_threshold,
                **summary,
            })

        except Exception as e:
            print(
                f"Failed params: fast={fast_window}, slow={slow_window}, trend={trend_window}, "
                f"rsi_min={rsi_min}, rsi_max={rsi_max}, adx_min={adx_min}, "
                f"stop_atr={stop_atr_mult}, target_atr={target_atr_mult}, "
                f"sweep_lookback={sweep_lookback}, rel_vol_threshold={rel_vol_threshold} -> {e}"
            )

    if not results:
        raise ValueError("No valid optimization results were produced.")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["score", "expectancy_%", "profit_factor", "win_rate_%"],
        ascending=False,
    ).reset_index(drop=True)

    print("\n===== TOP 15 PARAMETER COMBINATIONS =====")
    print(results_df.head(15).to_string(index=False))

    output_path = Path("data") / f"optimization_results_swing_liquidity_{TICKER}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved full optimization results to: {output_path}")


if __name__ == "__main__":
    main()