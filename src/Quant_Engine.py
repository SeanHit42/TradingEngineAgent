from __future__ import annotations

from pathlib import Path
from itertools import product
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", FutureWarning)

# =========================================================
# CONFIG
# =========================================================

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/quant_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
REGIME_TICKER = "SPY"

INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE_PCT = 0.01
ENTRY_FEE_PCT = 0.0005
EXIT_FEE_PCT = 0.0005

LONG_FAST_LEVEL = 30
LONG_SLOW_LEVEL = 50
SHORT_FAST_LEVEL = 70
SHORT_SLOW_LEVEL = 50

FAST_WINDOWS = [14, 21, 30, 42]
SLOW_WINDOWS = [42, 55, 84]
ATR_WINDOWS = [14]
STOP_ATR_MULTS = [0.8, 1.0]
TP1_MULTS = [1.0, 1.5]
TP2_MULTS = [2.0, 2.5]
TP3_MULTS = [3.0, 4.0]
REGIME_MA_WINDOWS = [100, 150, 200]
MIN_ATR_PCTS = [0.015, 0.02, 0.025]

MIN_TRADES_PER_TICKER = 5
TOP_N_RESULTS = 15


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass(frozen=True)
class StrategyParams:
    fast_window: int
    slow_window: int
    atr_window: int
    stop_atr_mult: float
    tp1_mult: float
    tp2_mult: float
    tp3_mult: float
    regime_ma_window: int
    min_atr_pct: float


# =========================================================
# UTILS
# =========================================================

def validate_ohlcv(df: pd.DataFrame) -> None:
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Input DataFrame is empty")


def crossed_above(prev_value: float, current_value: float, level: float) -> bool:
    return prev_value <= level and current_value > level


def crossed_below(prev_value: float, current_value: float, level: float) -> bool:
    return prev_value >= level and current_value < level


def split_shares_into_tranches(total_shares: int) -> list[int]:
    base = total_shares // 3
    remainder = total_shares % 3
    return [base + (1 if i < remainder else 0) for i in range(3)]


def calc_leg_pnl(side: str, entry_price: float, exit_price: float, qty: int) -> float:
    if side == "long":
        return qty * (exit_price - entry_price)
    return qty * (entry_price - exit_price)


def compute_position_size(equity: float, entry_price: float, stop_price: float) -> int:
    risk_dollars = equity * RISK_PER_TRADE_PCT
    stop_distance = abs(entry_price - stop_price)

    if stop_distance <= 0:
        return 0

    raw_shares = math.floor(risk_dollars / stop_distance)
    max_affordable = math.floor(equity / entry_price) if entry_price > 0 else 0

    return max(0, min(raw_shares, max_affordable))


# =========================================================
# INDICATORS
# =========================================================

def williams_r_normalized(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    highest_high = high.rolling(window).max()
    lowest_low = low.rolling(window).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    willr_raw = -100 * ((highest_high - close) / denom)
    return 100 + willr_raw


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def compute_regime_ma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


# =========================================================
# LOADING + FEATURE STORE
# =========================================================

def load_raw_ticker_data(ticker: str) -> pd.DataFrame:
    file_path = RAW_DIR / f"{ticker}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {file_path}")

    df = pd.read_parquet(file_path)
    validate_ohlcv(df)

    df = df.sort_values("Date").reset_index(drop=True).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_all_data(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        out[ticker] = load_raw_ticker_data(ticker)
    return out


def build_feature_store(
    market_data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    מחשב מראש את כל הפיצ'רים האפשריים שנצטרך,
    כדי לא לחשב אותם מחדש בכל קומבינציה.
    """
    all_williams_windows = sorted(set(FAST_WINDOWS + SLOW_WINDOWS))
    all_atr_windows = sorted(set(ATR_WINDOWS))
    all_regime_windows = sorted(set(REGIME_MA_WINDOWS))

    feature_store: Dict[str, pd.DataFrame] = {}

    for ticker, df in market_data.items():
        feat = df.copy()

        # returns
        feat["returns"] = feat["Close"].pct_change().fillna(0)

        # Williams
        for w in all_williams_windows:
            feat[f"willi_{w}"] = williams_r_normalized(
                feat["High"], feat["Low"], feat["Close"], w
            )

        # ATR + ATR%
        for w in all_atr_windows:
            atr = compute_atr(feat, w)
            feat[f"atr_{w}"] = atr
            feat[f"atr_pct_{w}"] = atr / feat["Close"]

        # regime MA for SPY reference only
        if ticker == REGIME_TICKER:
            for w in all_regime_windows:
                feat[f"regime_ma_{w}"] = compute_regime_ma(feat["Close"], w)
                feat[f"bull_regime_{w}"] = feat["Close"] > feat[f"regime_ma_{w}"]

        feature_store[ticker] = feat

    return feature_store


def merge_regime(asset_df: pd.DataFrame, regime_df: pd.DataFrame, regime_col: str) -> pd.DataFrame:
    merged = asset_df.merge(
        regime_df[["Date", regime_col]],
        on="Date",
        how="left",
    )
    merged[regime_col] = merged[regime_col].where(merged[regime_col].notna(), False)
    merged[regime_col] = merged[regime_col].astype(bool)
    return merged


def build_dataset_for_params(
    ticker: str,
    params: StrategyParams,
    feature_store: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    asset_df = feature_store[ticker].copy()
    regime_df = feature_store[REGIME_TICKER].copy()

    fast_col = f"willi_{params.fast_window}"
    slow_col = f"willi_{params.slow_window}"
    atr_col = f"atr_{params.atr_window}"
    atr_pct_col = f"atr_pct_{params.atr_window}"
    regime_col = f"bull_regime_{params.regime_ma_window}"

    if fast_col not in asset_df.columns:
        raise ValueError(f"Missing column {fast_col} for {ticker}")
    if slow_col not in asset_df.columns:
        raise ValueError(f"Missing column {slow_col} for {ticker}")
    if atr_col not in asset_df.columns:
        raise ValueError(f"Missing column {atr_col} for {ticker}")
    if atr_pct_col not in asset_df.columns:
        raise ValueError(f"Missing column {atr_pct_col} for {ticker}")
    if regime_col not in regime_df.columns:
        raise ValueError(f"Missing regime column {regime_col} for {REGIME_TICKER}")

    asset_df = asset_df.rename(
        columns={
            fast_col: "willi_fast",
            slow_col: "willi_slow",
            atr_col: "atr",
            atr_pct_col: "atr_pct",
        }
    )

    merged = merge_regime(asset_df, regime_df, regime_col=regime_col)
    merged = merged.rename(columns={regime_col: "bull_regime"})

    needed = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "returns", "willi_fast", "willi_slow", "atr", "atr_pct", "bull_regime",
    ]
    merged = merged[needed].dropna().reset_index(drop=True)

    return merged


# =========================================================
# SIGNALS
# =========================================================

def is_long_signal(prev_row: pd.Series, signal_row: pd.Series) -> bool:
    return (
        crossed_above(prev_row["willi_fast"], signal_row["willi_fast"], LONG_FAST_LEVEL)
        and crossed_above(prev_row["willi_slow"], signal_row["willi_slow"], LONG_SLOW_LEVEL)
    )


def is_short_signal(prev_row: pd.Series, signal_row: pd.Series) -> bool:
    return (
        crossed_below(prev_row["willi_fast"], signal_row["willi_fast"], SHORT_FAST_LEVEL)
        and crossed_below(prev_row["willi_slow"], signal_row["willi_slow"], SHORT_SLOW_LEVEL)
    )


# =========================================================
# TRADE OBJECT HELPERS
# =========================================================

def create_trade(
    side: str,
    signal_bar_date,
    entry_date,
    entry_price: float,
    atr: float,
    shares_total: int,
    stop_atr_mult: float,
    tp_mults: Tuple[float, float, float],
) -> dict:
    tp1_mult, tp2_mult, tp3_mult = tp_mults

    if side == "long":
        stop_price = entry_price - stop_atr_mult * atr
        tp_prices = [
            entry_price + tp1_mult * atr,
            entry_price + tp2_mult * atr,
            entry_price + tp3_mult * atr,
        ]
    else:
        stop_price = entry_price + stop_atr_mult * atr
        tp_prices = [
            entry_price - tp1_mult * atr,
            entry_price - tp2_mult * atr,
            entry_price - tp3_mult * atr,
        ]

    tranche_sizes = split_shares_into_tranches(shares_total)
    tranches = []

    for idx, qty in enumerate(tranche_sizes):
        tranches.append(
            {
                "tranche_id": idx + 1,
                "qty": qty,
                "tp_price": tp_prices[idx],
                "is_open": qty > 0,
                "exit_price": None,
                "exit_date": None,
                "exit_reason": None,
                "gross_pnl": 0.0,
                "fees": 0.0,
                "net_pnl": 0.0,
            }
        )

    entry_fee = shares_total * entry_price * ENTRY_FEE_PCT

    return {
        "side": side,
        "signal_bar_date": signal_bar_date,
        "entry_date": entry_date,
        "entry_price": entry_price,
        "initial_stop": stop_price,
        "current_stop": stop_price,
        "tp1_price": tp_prices[0],
        "tp2_price": tp_prices[1],
        "tp3_price": tp_prices[2],
        "shares_total": shares_total,
        "entry_fee": entry_fee,
        "tranches": tranches,
        "is_open": True,
        "exit_reason": None,
    }


def close_tranche(
    trade: dict,
    tranche_idx: int,
    exit_price: float,
    exit_date,
    exit_reason: str,
) -> None:
    tranche = trade["tranches"][tranche_idx]
    if not tranche["is_open"] or tranche["qty"] == 0:
        return

    qty = tranche["qty"]
    gross_pnl = calc_leg_pnl(trade["side"], trade["entry_price"], exit_price, qty)
    exit_fee = qty * exit_price * EXIT_FEE_PCT

    tranche["is_open"] = False
    tranche["exit_price"] = exit_price
    tranche["exit_date"] = exit_date
    tranche["exit_reason"] = exit_reason
    tranche["gross_pnl"] = gross_pnl
    tranche["fees"] += exit_fee
    tranche["net_pnl"] = gross_pnl - exit_fee


def update_stop_after_targets(trade: dict) -> None:
    t1 = trade["tranches"][0]
    t2 = trade["tranches"][1]

    tp1_closed = (not t1["is_open"]) and (t1["exit_reason"] == "tp1")
    tp2_closed = (not t2["is_open"]) and (t2["exit_reason"] == "tp2")

    if tp2_closed:
        trade["current_stop"] = trade["tp1_price"]
    elif tp1_closed:
        trade["current_stop"] = trade["entry_price"]


def all_tranches_closed(trade: dict) -> bool:
    return all((not t["is_open"]) for t in trade["tranches"])


def compute_trade_totals(trade: dict) -> dict:
    gross_pnl = sum(t["gross_pnl"] for t in trade["tranches"])
    exit_fees = sum(t["fees"] for t in trade["tranches"])
    total_fees = trade["entry_fee"] + exit_fees
    net_pnl = gross_pnl - total_fees

    tp_hits = sum(
        1 for idx, t in enumerate(trade["tranches"])
        if t["exit_reason"] == f"tp{idx + 1}"
    )

    exit_dates = [t["exit_date"] for t in trade["tranches"] if t["exit_date"] is not None]
    last_exit_date = max(exit_dates) if exit_dates else None

    return {
        "gross_pnl": gross_pnl,
        "total_fees": total_fees,
        "net_pnl": net_pnl,
        "tp_hits": tp_hits,
        "last_exit_date": last_exit_date,
    }


def mark_to_market_value(trade: Optional[dict], close_price: float) -> float:
    if trade is None:
        return 0.0

    value = 0.0
    for t in trade["tranches"]:
        if t["is_open"] and t["qty"] > 0:
            qty = t["qty"]
            if trade["side"] == "long":
                value += qty * close_price
            else:
                value += qty * (2 * trade["entry_price"] - close_price)
    return value


def handle_open_trade_for_bar(trade: dict, row: pd.Series) -> None:
    current_date = row["Date"]
    current_high = row["High"]
    current_low = row["Low"]

    open_indices = [i for i, t in enumerate(trade["tranches"]) if t["is_open"] and t["qty"] > 0]
    if not open_indices:
        trade["is_open"] = False
        return

    if trade["side"] == "long":
        stop_hit = current_low <= trade["current_stop"]
    else:
        stop_hit = current_high >= trade["current_stop"]

    # conservative daily rule: stop first
    if stop_hit:
        for idx in open_indices:
            close_tranche(trade, idx, trade["current_stop"], current_date, "stop")
        trade["is_open"] = False
        trade["exit_reason"] = "stop"
        return

    # then targets
    for idx in open_indices:
        tranche = trade["tranches"][idx]
        tp_price = tranche["tp_price"]

        if trade["side"] == "long" and current_high >= tp_price:
            close_tranche(trade, idx, tp_price, current_date, f"tp{idx + 1}")
        elif trade["side"] == "short" and current_low <= tp_price:
            close_tranche(trade, idx, tp_price, current_date, f"tp{idx + 1}")

        update_stop_after_targets(trade)

    if all_tranches_closed(trade):
        trade["is_open"] = False
        trade["exit_reason"] = "tp_complete"


# =========================================================
# BACKTEST
# =========================================================

def run_backtest(
    df: pd.DataFrame,
    stop_atr_mult: float,
    tp_mults: Tuple[float, float, float],
    min_atr_pct: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("Date").reset_index(drop=True).copy()

    cash = INITIAL_CAPITAL
    closed_trades = []
    equity_rows = []
    open_trade = None

    first_close = float(df.iloc[0]["Close"])
    buy_hold_shares = INITIAL_CAPITAL / first_close

    for i in range(len(df)):
        row = df.iloc[i]
        current_date = row["Date"]
        buy_and_hold_value = buy_hold_shares * row["Close"]

        if i >= 2 and open_trade is None:
            signal_row = df.iloc[i - 1]
            prev_signal_row = df.iloc[i - 2]

            long_signal = is_long_signal(prev_signal_row, signal_row)
            short_signal = is_short_signal(prev_signal_row, signal_row)

            volatility_ok = signal_row["atr_pct"] >= min_atr_pct
            bull_regime = bool(signal_row["bull_regime"])

            allow_long = bull_regime and volatility_ok
            allow_short = (not bull_regime) and volatility_ok

            if long_signal and allow_long and not short_signal:
                side = "long"
            elif short_signal and allow_short and not long_signal:
                side = "short"
            else:
                side = None

            if side is not None:
                entry_price = float(row["Open"])
                atr = float(signal_row["atr"])
                stop_price = entry_price - stop_atr_mult * atr if side == "long" else entry_price + stop_atr_mult * atr
                shares_total = compute_position_size(cash, entry_price, stop_price)

                if shares_total > 0:
                    trade = create_trade(
                        side=side,
                        signal_bar_date=signal_row["Date"],
                        entry_date=current_date,
                        entry_price=entry_price,
                        atr=atr,
                        shares_total=shares_total,
                        stop_atr_mult=stop_atr_mult,
                        tp_mults=tp_mults,
                    )

                    if side == "long":
                        gross_cost = shares_total * entry_price
                        total_entry_cash = gross_cost + trade["entry_fee"]
                        if total_entry_cash <= cash:
                            cash -= total_entry_cash
                            open_trade = trade
                    else:
                        if trade["entry_fee"] <= cash:
                            cash -= trade["entry_fee"]
                            open_trade = trade

        if open_trade is not None:
            handle_open_trade_for_bar(open_trade, row)

            if not open_trade["is_open"]:
                totals = compute_trade_totals(open_trade)

                if open_trade["side"] == "long":
                    proceeds = sum(t["qty"] * t["exit_price"] for t in open_trade["tranches"])
                    cash += proceeds
                else:
                    cash += open_trade["shares_total"] * open_trade["entry_price"] + totals["net_pnl"]

                closed_trades.append(
                    {
                        "side": open_trade["side"],
                        "signal_bar_date": open_trade["signal_bar_date"],
                        "entry_date": open_trade["entry_date"],
                        "exit_date": totals["last_exit_date"],
                        "entry_price": round(open_trade["entry_price"], 4),
                        "initial_stop": round(open_trade["initial_stop"], 4),
                        "final_stop": round(open_trade["current_stop"], 4),
                        "tp1_price": round(open_trade["tp1_price"], 4),
                        "tp2_price": round(open_trade["tp2_price"], 4),
                        "tp3_price": round(open_trade["tp3_price"], 4),
                        "shares_total": open_trade["shares_total"],
                        "tp_hits": totals["tp_hits"],
                        "exit_reason": open_trade["exit_reason"],
                        "gross_pnl": round(totals["gross_pnl"], 4),
                        "total_fees": round(totals["total_fees"], 4),
                        "net_pnl": round(totals["net_pnl"], 4),
                        "holding_days": (
                            pd.Timestamp(totals["last_exit_date"]) - pd.Timestamp(open_trade["entry_date"])
                        ).days if totals["last_exit_date"] is not None else 0,
                    }
                )

                open_trade = None

        open_value = mark_to_market_value(open_trade, float(row["Close"]))
        equity = cash + open_value

        equity_rows.append(
            {
                "Date": current_date,
                "cash": cash,
                "open_value": open_value,
                "equity_curve": equity,
                "buy_and_hold_value": buy_and_hold_value,
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    equity_df["rolling_max"] = equity_df["equity_curve"].cummax()
    equity_df["drawdown_pct"] = ((equity_df["equity_curve"] / equity_df["rolling_max"]) - 1) * 100

    trades_df = pd.DataFrame(closed_trades)
    return equity_df, trades_df


# =========================================================
# SUMMARY / SCORING
# =========================================================

def build_trade_summary(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
    strategy_return_pct = (float(equity_df["equity_curve"].iloc[-1]) / INITIAL_CAPITAL - 1) * 100
    buy_hold_return_pct = (float(equity_df["buy_and_hold_value"].iloc[-1]) / INITIAL_CAPITAL - 1) * 100
    max_drawdown_pct = float(equity_df["drawdown_pct"].min())

    num_trades = len(trades_df)

    if num_trades == 0:
        return {
            "strategy_return_%": round(strategy_return_pct, 2),
            "buy_hold_%": round(buy_hold_return_pct, 2),
            "max_drawdown_%": round(max_drawdown_pct, 2),
            "num_trades": 0,
            "win_rate_%": 0.0,
            "avg_win_$": 0.0,
            "avg_loss_$": 0.0,
            "avg_rr": np.nan,
            "profit_factor": 0.0,
            "expectancy_$": 0.0,
        }

    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] <= 0]

    win_rate = len(wins) / num_trades
    loss_rate = 1 - win_rate

    avg_win = wins["net_pnl"].mean() if not wins.empty else 0.0
    avg_loss = losses["net_pnl"].mean() if not losses.empty else 0.0

    gross_profit = wins["net_pnl"].sum() if not wins.empty else 0.0
    gross_loss_abs = abs(losses["net_pnl"].sum()) if not losses.empty else 0.0

    if gross_loss_abs == 0:
        profit_factor = np.inf if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss_abs

    expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))
    avg_rr = (avg_win / abs(avg_loss)) if avg_loss < 0 else np.nan

    return {
        "strategy_return_%": round(strategy_return_pct, 2),
        "buy_hold_%": round(buy_hold_return_pct, 2),
        "max_drawdown_%": round(max_drawdown_pct, 2),
        "num_trades": int(num_trades),
        "win_rate_%": round(win_rate * 100, 2),
        "avg_win_$": round(float(avg_win), 2) if not np.isnan(avg_win) else 0.0,
        "avg_loss_$": round(float(avg_loss), 2) if not np.isnan(avg_loss) else 0.0,
        "avg_rr": round(float(avg_rr), 2) if not np.isnan(avg_rr) else np.nan,
        "profit_factor": round(float(profit_factor), 2) if np.isfinite(profit_factor) else 10.0,
        "expectancy_$": round(float(expectancy), 2),
    }


# =========================================================
# RUN PER TICKER
# =========================================================

def run_for_ticker(
    ticker: str,
    params: StrategyParams,
    feature_store: Dict[str, pd.DataFrame],
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    df = build_dataset_for_params(
        ticker=ticker,
        params=params,
        feature_store=feature_store,
    )

    equity_df, trades_df = run_backtest(
        df=df,
        stop_atr_mult=params.stop_atr_mult,
        tp_mults=(params.tp1_mult, params.tp2_mult, params.tp3_mult),
        min_atr_pct=params.min_atr_pct,
    )

    summary = build_trade_summary(trades_df, equity_df)
    summary["ticker"] = ticker

    return summary, equity_df, trades_df


# =========================================================
# OPTIMIZER
# =========================================================

def build_param_grid() -> List[StrategyParams]:
    params: List[StrategyParams] = []

    for (
        fast_window,
        slow_window,
        atr_window,
        stop_atr_mult,
        tp1_mult,
        tp2_mult,
        tp3_mult,
        regime_ma_window,
        min_atr_pct,
    ) in product(
        FAST_WINDOWS,
        SLOW_WINDOWS,
        ATR_WINDOWS,
        STOP_ATR_MULTS,
        TP1_MULTS,
        TP2_MULTS,
        TP3_MULTS,
        REGIME_MA_WINDOWS,
        MIN_ATR_PCTS,
    ):
        if fast_window >= slow_window:
            continue
        if not (tp1_mult < tp2_mult < tp3_mult):
            continue

        params.append(
            StrategyParams(
                fast_window=fast_window,
                slow_window=slow_window,
                atr_window=atr_window,
                stop_atr_mult=stop_atr_mult,
                tp1_mult=tp1_mult,
                tp2_mult=tp2_mult,
                tp3_mult=tp3_mult,
                regime_ma_window=regime_ma_window,
                min_atr_pct=min_atr_pct,
            )
        )

    return params


def optimize(feature_store: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    param_grid = build_param_grid()
    total = len(param_grid)
    results = []

    for idx, params in enumerate(param_grid, start=1):
        ticker_summaries = []
        valid_ticker_count = 0

        for ticker in TICKERS:
            try:
                summary, _, _ = run_for_ticker(
                    ticker=ticker,
                    params=params,
                    feature_store=feature_store,
                )
                ticker_summaries.append(summary)

                if summary["num_trades"] >= MIN_TRADES_PER_TICKER:
                    valid_ticker_count += 1

            except Exception as e:
                print(f"Failed {ticker} with params={params}: {e}")

        if not ticker_summaries:
            continue

        df_sum = pd.DataFrame(ticker_summaries)

        row = {
            "fast_window": params.fast_window,
            "slow_window": params.slow_window,
            "atr_window": params.atr_window,
            "stop_atr_mult": params.stop_atr_mult,
            "tp1_mult": params.tp1_mult,
            "tp2_mult": params.tp2_mult,
            "tp3_mult": params.tp3_mult,
            "regime_ma_window": params.regime_ma_window,
            "min_atr_pct": params.min_atr_pct,
            "avg_strategy_return_%": df_sum["strategy_return_%"].mean(),
            "avg_buy_hold_%": df_sum["buy_hold_%"].mean(),
            "avg_max_drawdown_%": df_sum["max_drawdown_%"].mean(),
            "avg_num_trades": df_sum["num_trades"].mean(),
            "avg_win_rate_%": df_sum["win_rate_%"].mean(),
            "avg_avg_win_$": df_sum["avg_win_$"].mean(),
            "avg_avg_loss_$": df_sum["avg_loss_$"].mean(),
            "avg_avg_rr": df_sum["avg_rr"].mean(skipna=True),
            "avg_profit_factor": df_sum["profit_factor"].mean(),
            "avg_expectancy_$": df_sum["expectancy_$"].mean(),
            "valid_ticker_count": valid_ticker_count,
        }

        row["score"] = (
            row["avg_strategy_return_%"] * 0.25
            + row["avg_profit_factor"] * 20.0
            + row["avg_expectancy_$"] * 0.10
            + row["avg_win_rate_%"] * 0.20
            + row["avg_max_drawdown_%"] * 0.15
            + row["valid_ticker_count"] * 5.0
        )

        results.append(row)

        if idx % 50 == 0 or idx == total:
            print(f"tested {idx}/{total} parameter sets")

    if not results:
        raise ValueError("No optimization results produced.")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="score", ascending=False).reset_index(drop=True)
    return results_df


# =========================================================
# FINAL RUN ON BEST CONFIG
# =========================================================

def run_best_configuration(best_row: pd.Series, feature_store: Dict[str, pd.DataFrame]) -> None:
    params = StrategyParams(
        fast_window=int(best_row["fast_window"]),
        slow_window=int(best_row["slow_window"]),
        atr_window=int(best_row["atr_window"]),
        stop_atr_mult=float(best_row["stop_atr_mult"]),
        tp1_mult=float(best_row["tp1_mult"]),
        tp2_mult=float(best_row["tp2_mult"]),
        tp3_mult=float(best_row["tp3_mult"]),
        regime_ma_window=int(best_row["regime_ma_window"]),
        min_atr_pct=float(best_row["min_atr_pct"]),
    )

    all_results = []

    for ticker in TICKERS:
        try:
            summary, equity_df, trades_df = run_for_ticker(
                ticker=ticker,
                params=params,
                feature_store=feature_store,
            )
            all_results.append(summary)

            trades_path = OUTPUT_DIR / f"{ticker}_optimized_trades.csv"
            equity_path = OUTPUT_DIR / f"{ticker}_optimized_equity.csv"

            trades_df.to_csv(trades_path, index=False)
            equity_df.to_csv(equity_path, index=False)

        except Exception as e:
            print(f"Failed final run on {ticker}: {e}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n===== FINAL SUMMARY (BEST CONFIG) =====")
        print(results_df.to_string(index=False))

        summary_path = OUTPUT_DIR / "optimized_strategy_summary.csv"
        results_df.to_csv(summary_path, index=False)

        numeric_cols = [
            "strategy_return_%",
            "buy_hold_%",
            "max_drawdown_%",
            "num_trades",
            "win_rate_%",
            "avg_win_$",
            "avg_loss_$",
            "avg_rr",
            "profit_factor",
            "expectancy_$",
        ]

        agg = results_df[numeric_cols].mean(numeric_only=True).to_frame().T
        print("\n===== AVERAGE ACROSS ALL TICKERS (BEST CONFIG) =====")
        print(agg.to_string(index=False))

        print(f"\nSaved final per-ticker summary to: {summary_path}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    print("Loading market data...")
    market_data = load_all_data(sorted(set(TICKERS + [REGIME_TICKER])))

    print("Building feature store...")
    feature_store = build_feature_store(market_data)

    print("Running optimizer...")
    results_df = optimize(feature_store)

    print("\n===== TOP 15 PARAMETER COMBINATIONS =====")
    print(results_df.head(TOP_N_RESULTS).to_string(index=False))

    optimization_path = OUTPUT_DIR / "optimizer_results.csv"
    results_df.to_csv(optimization_path, index=False)
    print(f"\nSaved optimizer results to: {optimization_path}")

    best_row = results_df.iloc[0]
    print("\n===== BEST CONFIGURATION =====")
    print(best_row.to_string())

    print("\nRunning best configuration across all tickers...")
    run_best_configuration(best_row, feature_store)


if __name__ == "__main__":
    main()