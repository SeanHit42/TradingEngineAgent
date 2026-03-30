from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple, Optional
import math
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", FutureWarning)

# =========================================================
# CONFIG
# =========================================================

RAW_4H_DIR = Path("data/raw_4h")
RAW_15M_DIR = Path("data/raw_15m")
OUTPUT_DIR = Path("data/quant_v2_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "NVDA", "QQQ"]
REGIME_TICKER = "SPY"

INITIAL_CAPITAL = 10_000.0
RISK_PER_TRADE_PCT = 0.01
ENTRY_FEE_PCT = 0.0005
EXIT_FEE_PCT = 0.0005

# 4H signal parameters
FAST_WINDOWS = [10, 14, 21]
SLOW_WINDOWS = [30, 42, 55, 84]
ATR_WINDOWS = [14]
REGIME_MA_WINDOWS = [30, 50, 100]
MIN_ATR_PCTS = [0.0, 0.005, 0.01]

# liquidity / structure
EQ_LOOKBACKS = [5, 10, 20]
EQ_TOLS = [0.001, 0.002]  # 0.1% / 0.2%
REQUIRE_SWEEP_OPTIONS = [True, False]
REQUIRE_VWAP_CONFIRM_OPTIONS = [True, False]

# execution
STOP_ATR_MULTS = [0.8, 1.0]
TP1_MULTS = [1.0, 1.5]
TP2_MULTS = [2.0, 2.5]
TP3_MULTS = [3.0, 4.0]

# optimizer / robustness
MIN_TRADES_PER_TICKER = 3
MIN_VALID_TICKERS = 2
MIN_AVG_RR = 0.8
TOP_N_RESULTS = 15

MONTE_CARLO_SIMS = 1000
MONTE_CARLO_SEED = 42

# walk-forward
N_SPLITS = 3
MIN_TRAIN_BARS_4H = 40
MIN_TEST_BARS_4H = 20


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass(frozen=True)
class StrategyParams:
    fast_window: int
    slow_window: int
    atr_window: int
    regime_ma_window: int
    min_atr_pct: float
    eq_lookback: int
    eq_tol: float
    require_sweep: bool
    require_vwap_confirm: bool
    stop_atr_mult: float
    tp1_mult: float
    tp2_mult: float
    tp3_mult: float


# =========================================================
# HELPERS
# =========================================================

def validate_ohlcv(df: pd.DataFrame) -> None:
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("DataFrame is empty")


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_parquet(path)
    validate_ohlcv(df)
    df = df.sort_values("Date").reset_index(drop=True).copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_all_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    tickers = sorted(set(TICKERS + [REGIME_TICKER]))
    data_4h = {t: load_parquet(RAW_4H_DIR / f"{t}.parquet") for t in tickers}
    data_15m = {t: load_parquet(RAW_15M_DIR / f"{t}.parquet") for t in tickers}
    return data_4h, data_15m


def crossed_above(prev_value: float, current_value: float, level: float) -> bool:
    return prev_value <= level and current_value > level


def crossed_below(prev_value: float, current_value: float, level: float) -> bool:
    return prev_value >= level and current_value < level


def split_tranches(total_shares: int) -> list[int]:
    base = total_shares // 3
    rem = total_shares % 3
    return [base + (1 if i < rem else 0) for i in range(3)]


def compute_position_size(equity: float, entry_price: float, stop_price: float) -> int:
    risk_dollars = equity * RISK_PER_TRADE_PCT
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0
    raw_shares = math.floor(risk_dollars / stop_distance)
    max_affordable = math.floor(equity / entry_price) if entry_price > 0 else 0
    return max(0, min(raw_shares, max_affordable))


def calc_leg_pnl(side: str, entry_price: float, exit_price: float, qty: int) -> float:
    if side == "long":
        return qty * (exit_price - entry_price)
    return qty * (entry_price - exit_price)


# =========================================================
# INDICATORS / FEATURES
# =========================================================

def williams_r_normalized(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    hh = high.rolling(window).max()
    ll = low.rolling(window).min()
    denom = (hh - ll).replace(0, np.nan)
    raw = -100.0 * ((hh - close) / denom)
    return 100.0 + raw


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def compute_vwap_intraday(df_15m: pd.DataFrame) -> pd.Series:
    d = df_15m.copy()
    session = d["Date"].dt.floor("D")
    typical = (d["High"] + d["Low"] + d["Close"]) / 3.0
    pv = typical * d["Volume"]
    cum_pv = pv.groupby(session).cumsum()
    cum_vol = d["Volume"].groupby(session).cumsum().replace(0, np.nan)
    return cum_pv / cum_vol


def linear_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)

    def _slope(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        xm = x.mean()
        ym = y.mean()
        den = ((x - xm) ** 2).sum()
        if den == 0:
            return 0.0
        num = ((x - xm) * (y - ym)).sum()
        return num / den

    return series.rolling(window).apply(_slope, raw=True)


def compute_equal_high_low(df: pd.DataFrame, lookback: int, tol: float) -> Tuple[pd.Series, pd.Series]:
    prev_high = df["High"].rolling(lookback).max().shift(1)
    prev_low = df["Low"].rolling(lookback).min().shift(1)

    eq_high = ((df["High"] - prev_high).abs() / prev_high) <= tol
    eq_low = ((df["Low"] - prev_low).abs() / prev_low) <= tol

    eq_high = eq_high.fillna(False)
    eq_low = eq_low.fillna(False)
    return eq_high, eq_low


def compute_stop_run(df: pd.DataFrame, lookback: int) -> Tuple[pd.Series, pd.Series]:
    prev_high = df["High"].rolling(lookback).max().shift(1)
    prev_low = df["Low"].rolling(lookback).min().shift(1)

    bull_sweep = (df["Low"] < prev_low) & (df["Close"] > prev_low)
    bear_sweep = (df["High"] > prev_high) & (df["Close"] < prev_high)

    return bull_sweep.fillna(False), bear_sweep.fillna(False)


# =========================================================
# FEATURE STORES
# =========================================================

def build_4h_feature_store(data_4h: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    store: Dict[str, pd.DataFrame] = {}

    all_w = sorted(set(FAST_WINDOWS + SLOW_WINDOWS))
    all_atr = sorted(set(ATR_WINDOWS))
    all_regime = sorted(set(REGIME_MA_WINDOWS))
    all_lb = sorted(set(EQ_LOOKBACKS))
    all_tol = sorted(set(EQ_TOLS))

    for ticker, df in data_4h.items():
        feat = df.copy()
        feat["returns"] = feat["Close"].pct_change().fillna(0)

        for w in all_w:
            feat[f"willi_{w}"] = williams_r_normalized(feat["High"], feat["Low"], feat["Close"], w)

        for w in all_atr:
            atr = compute_atr(feat, w)
            feat[f"atr_{w}"] = atr
            feat[f"atr_pct_{w}"] = atr / feat["Close"]

        feat["trend_slope_10"] = linear_slope(feat["Close"], 10)
        feat["vol_ma_20"] = feat["Volume"].rolling(20).mean()
        feat["vol_imbalance"] = (feat["Volume"] / feat["vol_ma_20"]).replace([np.inf, -np.inf], np.nan)

        for lb in all_lb:
            bull_sweep, bear_sweep = compute_stop_run(feat, lb)
            feat[f"bull_sweep_{lb}"] = bull_sweep
            feat[f"bear_sweep_{lb}"] = bear_sweep

            for tol in all_tol:
                tag = str(tol).replace(".", "_")
                eqh, eql = compute_equal_high_low(feat, lb, tol)
                feat[f"eq_high_{lb}_{tag}"] = eqh
                feat[f"eq_low_{lb}_{tag}"] = eql

        if ticker == REGIME_TICKER:
            for w in all_regime:
                feat[f"regime_ma_{w}"] = feat["Close"].rolling(w).mean()
                feat[f"bull_regime_{w}"] = feat["Close"] > feat[f"regime_ma_{w}"]

        store[ticker] = feat

    return store


def build_15m_feature_store(data_15m: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    store: Dict[str, pd.DataFrame] = {}
    for ticker, df in data_15m.items():
        feat = df.copy()
        feat["vwap"] = compute_vwap_intraday(feat)
        feat["vwap_dev"] = (feat["Close"] - feat["vwap"]) / feat["vwap"]
        feat["prev_high_1"] = feat["High"].shift(1)
        feat["prev_low_1"] = feat["Low"].shift(1)
        store[ticker] = feat
    return store


# =========================================================
# DATASET BUILDERS
# =========================================================

def build_regime_series(spy_4h_df: pd.DataFrame, regime_ma_window: int) -> pd.DataFrame:
    col = f"bull_regime_{regime_ma_window}"
    if col not in spy_4h_df.columns:
        spy_4h_df = spy_4h_df.copy()
        spy_4h_df[f"regime_ma_{regime_ma_window}"] = spy_4h_df["Close"].rolling(regime_ma_window).mean()
        spy_4h_df[col] = spy_4h_df["Close"] > spy_4h_df[f"regime_ma_{regime_ma_window}"]
    out = spy_4h_df[["Date", col]].copy()
    out = out.rename(columns={col: "bull_regime"})
    return out


def build_4h_dataset(
    ticker: str,
    params: StrategyParams,
    store_4h: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    df = store_4h[ticker].copy()
    regime_df = build_regime_series(store_4h[REGIME_TICKER], params.regime_ma_window)

    tol_tag = str(params.eq_tol).replace(".", "_")

    df["willi_fast"] = df[f"willi_{params.fast_window}"]
    df["willi_slow"] = df[f"willi_{params.slow_window}"]
    df["atr"] = df[f"atr_{params.atr_window}"]
    df["atr_pct"] = df[f"atr_pct_{params.atr_window}"]
    df["bull_sweep"] = df[f"bull_sweep_{params.eq_lookback}"]
    df["bear_sweep"] = df[f"bear_sweep_{params.eq_lookback}"]
    df["eq_high"] = df[f"eq_high_{params.eq_lookback}_{tol_tag}"]
    df["eq_low"] = df[f"eq_low_{params.eq_lookback}_{tol_tag}"]

    df = df.merge(regime_df, on="Date", how="left")
    df["bull_regime"] = df["bull_regime"].fillna(False).astype(bool)

    needed = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "returns", "willi_fast", "willi_slow", "atr", "atr_pct",
        "bull_sweep", "bear_sweep", "eq_high", "eq_low",
        "trend_slope_10", "vol_imbalance", "bull_regime",
    ]
    return df[needed].dropna().reset_index(drop=True)


def build_aligned_15m_dataset(
    ticker: str,
    htf_df: pd.DataFrame,
    store_15m: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    ltf = store_15m[ticker].copy()
    right = htf_df[["Date", "filtered_signal", "atr"]].sort_values("Date").copy()

    aligned = pd.merge_asof(
        ltf.sort_values("Date"),
        right,
        on="Date",
        direction="backward",
    )

    aligned["filtered_signal"] = aligned["filtered_signal"].fillna(0).astype(int)
    return aligned


# =========================================================
# SIGNAL ENGINE 4H
# =========================================================

def generate_4h_signals(df_4h: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    df = df_4h.copy()
    df["raw_signal"] = 0

    long_signal = (
        (df["willi_fast"].shift(1) <= 30)
        & (df["willi_fast"] > 30)
        & (df["willi_slow"].shift(1) <= 50)
        & (df["willi_slow"] > 50)
    )

    short_signal = (
        (df["willi_fast"].shift(1) >= 70)
        & (df["willi_fast"] < 70)
        & (df["willi_slow"].shift(1) >= 50)
        & (df["willi_slow"] < 50)
    )

    # structure / context filters
    long_signal = long_signal & (df["trend_slope_10"] > 0)
    short_signal = short_signal & (df["trend_slope_10"] < 0)

    if params.require_sweep:
        long_signal = long_signal & df["bull_sweep"]
        short_signal = short_signal & df["bear_sweep"]

    df.loc[long_signal, "raw_signal"] = 1
    df.loc[short_signal, "raw_signal"] = -1

    # regime + volatility
    df["filtered_signal"] = 0
    vol_ok = df["atr_pct"] >= params.min_atr_pct

    df.loc[(df["raw_signal"] == 1) & df["bull_regime"] & vol_ok, "filtered_signal"] = 1
    df.loc[(df["raw_signal"] == -1) & (~df["bull_regime"]) & vol_ok, "filtered_signal"] = -1

    return df


# =========================================================
# EXECUTION ENGINE 15M
# =========================================================

def create_trade(
    side: str,
    entry_time,
    entry_price: float,
    atr_4h: float,
    shares_total: int,
    params: StrategyParams,
) -> dict:
    if side == "long":
        stop_price = entry_price - params.stop_atr_mult * atr_4h
        tp_prices = [
            entry_price + params.tp1_mult * atr_4h,
            entry_price + params.tp2_mult * atr_4h,
            entry_price + params.tp3_mult * atr_4h,
        ]
    else:
        stop_price = entry_price + params.stop_atr_mult * atr_4h
        tp_prices = [
            entry_price - params.tp1_mult * atr_4h,
            entry_price - params.tp2_mult * atr_4h,
            entry_price - params.tp3_mult * atr_4h,
        ]

    tranche_sizes = split_tranches(shares_total)

    tranches = []
    for i, qty in enumerate(tranche_sizes):
        tranches.append({
            "tranche_id": i + 1,
            "qty": qty,
            "tp_price": tp_prices[i],
            "is_open": qty > 0,
            "exit_price": None,
            "exit_time": None,
            "exit_reason": None,
            "gross_pnl": 0.0,
            "fees": 0.0,
            "net_pnl": 0.0,
        })

    entry_fee = shares_total * entry_price * ENTRY_FEE_PCT

    return {
        "side": side,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "atr_4h": atr_4h,
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


def close_tranche(trade: dict, tranche_idx: int, exit_price: float, exit_time, exit_reason: str) -> None:
    tr = trade["tranches"][tranche_idx]
    if not tr["is_open"] or tr["qty"] == 0:
        return

    qty = tr["qty"]
    gross = calc_leg_pnl(trade["side"], trade["entry_price"], exit_price, qty)
    exit_fee = qty * exit_price * EXIT_FEE_PCT

    tr["is_open"] = False
    tr["exit_price"] = exit_price
    tr["exit_time"] = exit_time
    tr["exit_reason"] = exit_reason
    tr["gross_pnl"] = gross
    tr["fees"] += exit_fee
    tr["net_pnl"] = gross - exit_fee


def update_stop_after_targets(trade: dict) -> None:
    t1 = trade["tranches"][0]
    t2 = trade["tranches"][1]

    if (not t2["is_open"]) and t2["exit_reason"] == "tp2":
        trade["current_stop"] = trade["tp1_price"]
    elif (not t1["is_open"]) and t1["exit_reason"] == "tp1":
        trade["current_stop"] = trade["entry_price"]


def all_closed(trade: dict) -> bool:
    return all(not t["is_open"] for t in trade["tranches"])


def compute_trade_totals(trade: dict) -> dict:
    gross = sum(t["gross_pnl"] for t in trade["tranches"])
    exit_fees = sum(t["fees"] for t in trade["tranches"])
    total_fees = trade["entry_fee"] + exit_fees
    net = gross - total_fees

    exit_times = [t["exit_time"] for t in trade["tranches"] if t["exit_time"] is not None]
    last_exit_time = max(exit_times) if exit_times else None

    tp_hits = sum(
        1 for idx, t in enumerate(trade["tranches"])
        if t["exit_reason"] == f"tp{idx + 1}"
    )

    return {
        "gross_pnl": gross,
        "total_fees": total_fees,
        "net_pnl": net,
        "tp_hits": tp_hits,
        "last_exit_time": last_exit_time,
    }


def mark_to_market_value(trade: Optional[dict], close_price: float) -> float:
    if trade is None:
        return 0.0

    val = 0.0
    for t in trade["tranches"]:
        if t["is_open"] and t["qty"] > 0:
            qty = t["qty"]
            if trade["side"] == "long":
                val += qty * close_price
            else:
                val += qty * (2 * trade["entry_price"] - close_price)
    return val


def run_15m_execution(df_15m: pd.DataFrame, params: StrategyParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_15m.sort_values("Date").reset_index(drop=True).copy()

    cash = INITIAL_CAPITAL
    open_trade = None
    closed_trades = []
    equity_rows = []

    buy_hold_shares = INITIAL_CAPITAL / float(df.iloc[0]["Close"])

    for i in range(len(df)):
        row = df.iloc[i]
        now = row["Date"]
        buy_hold_value = buy_hold_shares * row["Close"]

        if i >= 1 and open_trade is None:
            signal = int(row["filtered_signal"])
            atr_4h = row["atr"]

            long_trigger = (
                signal == 1
                and row["High"] > row["prev_high_1"]
                and (
                    (not params.require_vwap_confirm)
                    or (row["Close"] > row["vwap"])
                )
            )

            short_trigger = (
                signal == -1
                and row["Low"] < row["prev_low_1"]
                and (
                    (not params.require_vwap_confirm)
                    or (row["Close"] < row["vwap"])
                )
            )

            side = "long" if long_trigger else "short" if short_trigger else None

            if side is not None and pd.notna(atr_4h) and atr_4h > 0:
                entry_price = float(row["Open"])
                stop_price = entry_price - params.stop_atr_mult * atr_4h if side == "long" else entry_price + params.stop_atr_mult * atr_4h
                shares_total = compute_position_size(cash, entry_price, stop_price)

                if shares_total > 0:
                    trade = create_trade(side, now, entry_price, float(atr_4h), shares_total, params)

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
            high = row["High"]
            low = row["Low"]

            open_idx = [j for j, t in enumerate(open_trade["tranches"]) if t["is_open"] and t["qty"] > 0]

            if open_trade["side"] == "long":
                stop_hit = low <= open_trade["current_stop"]
            else:
                stop_hit = high >= open_trade["current_stop"]

            if stop_hit:
                for j in open_idx:
                    close_tranche(open_trade, j, open_trade["current_stop"], now, "stop")
                open_trade["is_open"] = False
                open_trade["exit_reason"] = "stop"
            else:
                for j in open_idx:
                    tr = open_trade["tranches"][j]
                    tp = tr["tp_price"]

                    if open_trade["side"] == "long" and high >= tp:
                        close_tranche(open_trade, j, tp, now, f"tp{j + 1}")
                    elif open_trade["side"] == "short" and low <= tp:
                        close_tranche(open_trade, j, tp, now, f"tp{j + 1}")

                    update_stop_after_targets(open_trade)

                if all_closed(open_trade):
                    open_trade["is_open"] = False
                    open_trade["exit_reason"] = "tp_complete"

            if open_trade is not None and not open_trade["is_open"]:
                totals = compute_trade_totals(open_trade)

                if open_trade["side"] == "long":
                    proceeds = sum(t["qty"] * t["exit_price"] for t in open_trade["tranches"])
                    cash += proceeds
                else:
                    cash += open_trade["shares_total"] * open_trade["entry_price"] + totals["net_pnl"]

                closed_trades.append({
                    "side": open_trade["side"],
                    "entry_time": open_trade["entry_time"],
                    "exit_time": totals["last_exit_time"],
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
                    "holding_minutes": int((pd.Timestamp(totals["last_exit_time"]) - pd.Timestamp(open_trade["entry_time"])).total_seconds() / 60.0)
                    if totals["last_exit_time"] is not None else 0,
                })

                open_trade = None

        open_value = mark_to_market_value(open_trade, float(row["Close"]))
        equity_rows.append({
            "Date": now,
            "cash": cash,
            "open_value": open_value,
            "equity_curve": cash + open_value,
            "buy_and_hold_value": buy_hold_value,
        })

    equity_df = pd.DataFrame(equity_rows)
    equity_df["rolling_max"] = equity_df["equity_curve"].cummax()
    equity_df["drawdown_pct"] = ((equity_df["equity_curve"] / equity_df["rolling_max"]) - 1.0) * 100.0

    trades_df = pd.DataFrame(closed_trades)
    return equity_df, trades_df


# =========================================================
# SUMMARY / MONTE CARLO
# =========================================================

def monte_carlo_trade_bootstrap(trades_df: pd.DataFrame, n_sims: int = MONTE_CARLO_SIMS, seed: int = MONTE_CARLO_SEED) -> dict:
    if trades_df.empty or "net_pnl" not in trades_df.columns:
        return {
            "mc_prob_profit_%": 0.0,
            "mc_p05_return_%": 0.0,
            "mc_median_return_%": 0.0,
            "mc_median_max_dd_%": 0.0,
        }

    rng = np.random.default_rng(seed)
    trade_pnls = trades_df["net_pnl"].to_numpy(dtype=float)
    n_trades = len(trade_pnls)

    final_eq = np.zeros(n_sims)
    max_dd = np.zeros(n_sims)

    for i in range(n_sims):
        sample = rng.choice(trade_pnls, size=n_trades, replace=True)
        eq_path = INITIAL_CAPITAL + np.cumsum(sample)
        roll = np.maximum.accumulate(eq_path)
        dd = (eq_path / roll - 1.0) * 100.0
        final_eq[i] = eq_path[-1]
        max_dd[i] = np.min(dd)

    ret = (final_eq / INITIAL_CAPITAL - 1.0) * 100.0

    return {
        "mc_prob_profit_%": round(float(np.mean(final_eq > INITIAL_CAPITAL) * 100.0), 2),
        "mc_p05_return_%": round(float(np.percentile(ret, 5)), 2),
        "mc_median_return_%": round(float(np.median(ret)), 2),
        "mc_median_max_dd_%": round(float(np.median(max_dd)), 2),
    }


def build_trade_summary(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
    strategy_return_pct = (float(equity_df["equity_curve"].iloc[-1]) / INITIAL_CAPITAL - 1.0) * 100.0
    buy_hold_return_pct = (float(equity_df["buy_and_hold_value"].iloc[-1]) / INITIAL_CAPITAL - 1.0) * 100.0
    max_drawdown_pct = float(equity_df["drawdown_pct"].min())

    num_trades = len(trades_df)

    if num_trades == 0:
        out = {
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
        out.update(monte_carlo_trade_bootstrap(trades_df))
        return out

    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] <= 0]

    win_rate = len(wins) / num_trades
    loss_rate = 1.0 - win_rate
    avg_win = wins["net_pnl"].mean() if not wins.empty else 0.0
    avg_loss = losses["net_pnl"].mean() if not losses.empty else 0.0
    gross_profit = wins["net_pnl"].sum() if not wins.empty else 0.0
    gross_loss_abs = abs(losses["net_pnl"].sum()) if not losses.empty else 0.0
    profit_factor = (gross_profit / gross_loss_abs) if gross_loss_abs > 0 else (10.0 if gross_profit > 0 else 0.0)
    expectancy = (win_rate * avg_win) - (loss_rate * abs(avg_loss))
    avg_rr = (avg_win / abs(avg_loss)) if avg_loss < 0 else np.nan

    out = {
        "strategy_return_%": round(strategy_return_pct, 2),
        "buy_hold_%": round(buy_hold_return_pct, 2),
        "max_drawdown_%": round(max_drawdown_pct, 2),
        "num_trades": int(num_trades),
        "win_rate_%": round(win_rate * 100.0, 2),
        "avg_win_$": round(float(avg_win), 2),
        "avg_loss_$": round(float(avg_loss), 2),
        "avg_rr": round(float(avg_rr), 2) if not np.isnan(avg_rr) else np.nan,
        "profit_factor": round(float(profit_factor), 2),
        "expectancy_$": round(float(expectancy), 2),
    }
    out.update(monte_carlo_trade_bootstrap(trades_df))
    return out


# =========================================================
# WALK-FORWARD
# =========================================================

def time_splits(df_4h: pd.DataFrame, n_splits: int = N_SPLITS) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    df = df_4h.sort_values("Date").reset_index(drop=True)
    dates = df["Date"]

    total = len(df)
    if total < (MIN_TRAIN_BARS_4H + MIN_TEST_BARS_4H):
        return []

    step = max(MIN_TEST_BARS_4H, total // (n_splits + 1))
    splits = []

    for i in range(n_splits):
        train_end_idx = MIN_TRAIN_BARS_4H + i * step
        test_end_idx = min(train_end_idx + step, total - 1)

        if train_end_idx >= total or test_end_idx <= train_end_idx:
            continue

        train_start = dates.iloc[0]
        train_end = dates.iloc[train_end_idx]
        test_start = dates.iloc[train_end_idx + 1] if train_end_idx + 1 < total else dates.iloc[train_end_idx]
        test_end = dates.iloc[test_end_idx]

        splits.append((train_start, train_end, test_start, test_end))

    return splits


def subset_4h(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["Date"] >= start) & (df["Date"] <= end)].copy()


def subset_15m_from_4h_window(df_15m: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df_15m[(df_15m["Date"] >= start) & (df_15m["Date"] <= end + pd.Timedelta(hours=4))].copy()


# =========================================================
# RUN PER TICKER / PER SPLIT
# =========================================================

def run_single(
    ticker: str,
    params: StrategyParams,
    data_4h: Dict[str, pd.DataFrame],
    data_15m: Dict[str, pd.DataFrame],
    start_4h: Optional[pd.Timestamp] = None,
    end_4h: Optional[pd.Timestamp] = None,
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    store_4h = build_4h_feature_store(data_4h)
    store_15m = build_15m_feature_store(data_15m)

    htf = build_4h_dataset(ticker, params, store_4h)
    regime = build_regime_series(store_4h[REGIME_TICKER], params.regime_ma_window)

    if start_4h is not None and end_4h is not None:
        htf = subset_4h(htf, start_4h, end_4h)
        regime = subset_4h(regime, start_4h, end_4h)

    htf = htf.merge(regime, on="Date", how="left", suffixes=("", "_reg"))
    if "bull_regime_reg" in htf.columns:
        htf["bull_regime"] = htf["bull_regime_reg"].fillna(False).astype(bool)
        htf = htf.drop(columns=["bull_regime_reg"])

    htf = generate_4h_signals(htf, params)

    ltf = store_15m[ticker].copy()
    if start_4h is not None and end_4h is not None:
        ltf = subset_15m_from_4h_window(ltf, start_4h, end_4h)

    aligned = build_aligned_15m_dataset(ticker, htf, {ticker: ltf})
    equity_df, trades_df = run_15m_execution(aligned, params)
    summary = build_trade_summary(trades_df, equity_df)
    summary["ticker"] = ticker

    return summary, equity_df, trades_df


# =========================================================
# OPTIMIZER
# =========================================================

def build_param_grid() -> List[StrategyParams]:
    out: List[StrategyParams] = []
    for (
        fast_window,
        slow_window,
        atr_window,
        regime_ma_window,
        min_atr_pct,
        eq_lookback,
        eq_tol,
        require_sweep,
        require_vwap_confirm,
        stop_atr_mult,
        tp1_mult,
        tp2_mult,
        tp3_mult,
    ) in product(
        FAST_WINDOWS,
        SLOW_WINDOWS,
        ATR_WINDOWS,
        REGIME_MA_WINDOWS,
        MIN_ATR_PCTS,
        EQ_LOOKBACKS,
        EQ_TOLS,
        REQUIRE_SWEEP_OPTIONS,
        REQUIRE_VWAP_CONFIRM_OPTIONS,
        STOP_ATR_MULTS,
        TP1_MULTS,
        TP2_MULTS,
        TP3_MULTS,
    ):
        if fast_window >= slow_window:
            continue
        if not (tp1_mult < tp2_mult < tp3_mult):
            continue

        out.append(
            StrategyParams(
                fast_window=fast_window,
                slow_window=slow_window,
                atr_window=atr_window,
                regime_ma_window=regime_ma_window,
                min_atr_pct=min_atr_pct,
                eq_lookback=eq_lookback,
                eq_tol=eq_tol,
                require_sweep=require_sweep,
                require_vwap_confirm=require_vwap_confirm,
                stop_atr_mult=stop_atr_mult,
                tp1_mult=tp1_mult,
                tp2_mult=tp2_mult,
                tp3_mult=tp3_mult,
            )
        )
    return out


def score_row(row: dict) -> float:
    score = (
        row["avg_strategy_return_%"] * 0.20
        + row["avg_profit_factor"] * 18.0
        + row["avg_expectancy_$"] * 0.10
        + row["avg_win_rate_%"] * 0.20
        + row["avg_max_drawdown_%"] * 0.20
        + row["valid_ticker_count"] * 8.0
        + row["avg_mc_prob_profit_%"] * 0.10
        + row["avg_mc_p05_return_%"] * 0.10
        + row["avg_mc_median_max_dd_%"] * 0.10
    )
    if row["avg_max_drawdown_%"] < -30:
        score -= 40
    if row["avg_max_drawdown_%"] < -40:
        score -= 80
    return score


def optimize(
    data_4h: Dict[str, pd.DataFrame],
    data_15m: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    param_grid = build_param_grid()
    results = []
    total = len(param_grid)

    base_df = load_parquet(RAW_4H_DIR / f"{TICKERS[0]}.parquet")
    splits = time_splits(base_df)

    if not splits:
        raise ValueError("Not enough 4H data for walk-forward splits.")

    for idx, params in enumerate(param_grid, start=1):
        split_scores = []

        for train_start, train_end, test_start, test_end in splits:
            ticker_summaries = []
            valid_ticker_count = 0

            for ticker in TICKERS:
                try:
                    summary, _, _ = run_single(
                        ticker=ticker,
                        params=params,
                        data_4h=data_4h,
                        data_15m=data_15m,
                        start_4h=test_start,
                        end_4h=test_end,
                    )
                    ticker_summaries.append(summary)
                    if summary["num_trades"] >= MIN_TRADES_PER_TICKER:
                        valid_ticker_count += 1
                except Exception:
                    pass

            if not ticker_summaries:
                continue

            df_sum = pd.DataFrame(ticker_summaries)

            row = {
                "fast_window": params.fast_window,
                "slow_window": params.slow_window,
                "atr_window": params.atr_window,
                "regime_ma_window": params.regime_ma_window,
                "min_atr_pct": params.min_atr_pct,
                "eq_lookback": params.eq_lookback,
                "eq_tol": params.eq_tol,
                "require_sweep": params.require_sweep,
                "require_vwap_confirm": params.require_vwap_confirm,
                "stop_atr_mult": params.stop_atr_mult,
                "tp1_mult": params.tp1_mult,
                "tp2_mult": params.tp2_mult,
                "tp3_mult": params.tp3_mult,
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
                "avg_mc_prob_profit_%": df_sum["mc_prob_profit_%"].mean(),
                "avg_mc_p05_return_%": df_sum["mc_p05_return_%"].mean(),
                "avg_mc_median_max_dd_%": df_sum["mc_median_max_dd_%"].mean(),
                "valid_ticker_count": valid_ticker_count,
            }

            if row["valid_ticker_count"] < MIN_VALID_TICKERS:
                continue
            if row["avg_num_trades"] < MIN_TRADES_PER_TICKER:
                continue
            if pd.notna(row["avg_avg_rr"]) and row["avg_avg_rr"] < MIN_AVG_RR:
                continue

            split_scores.append(row)

        if split_scores:
            tmp = pd.DataFrame(split_scores)
            final_row = tmp.mean(numeric_only=True).to_dict()
            final_row.update({
                "fast_window": params.fast_window,
                "slow_window": params.slow_window,
                "atr_window": params.atr_window,
                "regime_ma_window": params.regime_ma_window,
                "min_atr_pct": params.min_atr_pct,
                "eq_lookback": params.eq_lookback,
                "eq_tol": params.eq_tol,
                "require_sweep": params.require_sweep,
                "require_vwap_confirm": params.require_vwap_confirm,
                "stop_atr_mult": params.stop_atr_mult,
                "tp1_mult": params.tp1_mult,
                "tp2_mult": params.tp2_mult,
                "tp3_mult": params.tp3_mult,
            })
            final_row["score"] = score_row(final_row)
            results.append(final_row)

        if idx % 25 == 0 or idx == total:
            print(f"tested {idx}/{total} parameter sets")

    if not results:
        raise ValueError("No optimization results produced after walk-forward filters.")

    results_df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    return results_df


# =========================================================
# FINAL BEST RUN
# =========================================================

def run_best_configuration(best_row: pd.Series, data_4h: Dict[str, pd.DataFrame], data_15m: Dict[str, pd.DataFrame]) -> None:
    params = StrategyParams(
        fast_window=int(best_row["fast_window"]),
        slow_window=int(best_row["slow_window"]),
        atr_window=int(best_row["atr_window"]),
        regime_ma_window=int(best_row["regime_ma_window"]),
        min_atr_pct=float(best_row["min_atr_pct"]),
        eq_lookback=int(best_row["eq_lookback"]),
        eq_tol=float(best_row["eq_tol"]),
        require_sweep=bool(best_row["require_sweep"]),
        require_vwap_confirm=bool(best_row["require_vwap_confirm"]),
        stop_atr_mult=float(best_row["stop_atr_mult"]),
        tp1_mult=float(best_row["tp1_mult"]),
        tp2_mult=float(best_row["tp2_mult"]),
        tp3_mult=float(best_row["tp3_mult"]),
    )

    all_results = []

    for ticker in TICKERS:
        try:
            summary, equity_df, trades_df = run_single(
                ticker=ticker,
                params=params,
                data_4h=data_4h,
                data_15m=data_15m,
            )
            all_results.append(summary)
            trades_df.to_csv(OUTPUT_DIR / f"{ticker}_optimized_trades.csv", index=False)
            equity_df.to_csv(OUTPUT_DIR / f"{ticker}_optimized_equity.csv", index=False)
        except Exception as e:
            print(f"Failed final run on {ticker}: {e}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n===== FINAL SUMMARY (BEST CONFIG) =====")
        print(results_df.to_string(index=False))
        results_df.to_csv(OUTPUT_DIR / "optimized_strategy_summary.csv", index=False)

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
            "mc_prob_profit_%",
            "mc_p05_return_%",
            "mc_median_return_%",
            "mc_median_max_dd_%",
        ]
        agg = results_df[numeric_cols].mean(numeric_only=True).to_frame().T
        print("\n===== AVERAGE ACROSS ALL TICKERS (BEST CONFIG) =====")
        print(agg.to_string(index=False))
        agg.to_csv(OUTPUT_DIR / "optimized_average_summary.csv", index=False)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    print("Loading 4H and 15m data...")
    data_4h, data_15m = load_all_data()

    print("Running walk-forward optimizer...")
    results_df = optimize(data_4h, data_15m)

    print("\n===== TOP 15 PARAMETER COMBINATIONS =====")
    print(results_df.head(TOP_N_RESULTS).to_string(index=False))
    results_df.to_csv(OUTPUT_DIR / "optimizer_results.csv", index=False)

    best_row = results_df.iloc[0]
    print("\n===== BEST CONFIGURATION =====")
    print(best_row.to_string())

    print("\nRunning best configuration across all tickers...")
    run_best_configuration(best_row, data_4h, data_15m)

#continue later need to run the engine
if __name__ == "__main__":
    main()