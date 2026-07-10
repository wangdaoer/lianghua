from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
from pandas.errors import EmptyDataError

from run_backtest import load_prices, pivot_prices
from run_strong_pullback_satellite import run_satellite_walk_forward
from train_next_open_rank_model import clean_matrix, load_market_exposure


REQUIRED_EVOLUTION_COLUMNS = {
    "date", "symbol", "open", "high", "low", "close", "volume", "amount",
}
FILTER_KEYS = (
    "min_close", "min_avg_amount_20d", "min_pullback_5d", "max_pullback_5d",
    "min_prior_return_20", "min_prior_return_60", "min_return_20d",
    "min_return_60d", "min_distance_ma60", "max_intraday_return",
)


@dataclass(frozen=True)
class PriceBundle:
    close: pd.DataFrame
    open_px: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    amount: pd.DataFrame
    market_exposure: pd.Series


@dataclass(frozen=True)
class StrategyRun:
    equity: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    candidates: pd.DataFrame


def validate_input_schema(path: Path) -> None:
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = REQUIRED_EVOLUTION_COLUMNS - columns
    if missing:
        raise ValueError(f"Missing evolution input columns: {sorted(missing)}")


def load_price_bundle(
    data_path: Path,
    end_date: pd.Timestamp,
    benchmark_path: Path | None,
    params: Mapping[str, object],
) -> PriceBundle:
    validate_input_schema(data_path)
    raw = load_prices(data_path, None, end_date.strftime("%Y-%m-%d"))
    max_abs_return = float(params["max_abs_daily_return"])
    close = clean_matrix(pivot_prices(raw, "close"), max_abs_return)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), max_abs_return)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), max_abs_return)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), max_abs_return)
    amount = pivot_prices(raw, "amount").reindex_like(close)
    market_exposure = load_market_exposure(
        str(benchmark_path) if benchmark_path else None,
        close.index,
        ma_window=int(params["market_ma_window"]),
        risk_off_drawdown_20d=float(params["market_risk_off_drawdown_20d"]),
        below_ma_exposure=float(params["market_below_ma_exposure"]),
        crash_exposure=float(params["market_crash_exposure"]),
    )
    if close.index.max() > end_date:
        raise AssertionError("Price bundle extends beyond requested end date")
    return PriceBundle(close, open_px, high, low, amount, market_exposure)


def execute_strategy_trial(bundle: PriceBundle, params: Mapping[str, object]) -> StrategyRun:
    filter_kwargs = {key: float(params[key]) for key in FILTER_KEYS}
    equity, weights, trades, candidates = run_satellite_walk_forward(
        close=bundle.close,
        open_px=bundle.open_px,
        high=bundle.high,
        low=bundle.low,
        amount=bundle.amount,
        train_days=int(params["train_days"]),
        retrain_frequency=int(params["retrain_frequency"]),
        top_n=int(params["top_n"]),
        rebalance_frequency=int(params["rebalance_frequency"]),
        max_position_weight=float(params["max_position_weight"]),
        leverage=float(params["leverage"]),
        min_score=None if params["min_score"] is None else float(params["min_score"]),
        commission_bps=float(params["commission_bps"]),
        impact_bps=float(params["impact_bps"]),
        max_buy_open_gap=float(params["max_buy_open_gap"]),
        limit_buffer=float(params["limit_buffer"]),
        market_exposure=bundle.market_exposure,
        initial_capital=float(params["initial_capital"]),
        filter_kwargs=filter_kwargs,
        basket_guard_return_20d_min=params["basket_guard_return_20d_min"],
        basket_guard_distance_ma60_min=params["basket_guard_distance_ma60_min"],
        basket_guard_scale=float(params["basket_guard_scale"]),
        rebound_exit_return=params["rebound_exit_return"],
        rebound_exit_scale=float(params["rebound_exit_scale"]),
        rebound_exit_market_exposure_max=params["rebound_exit_market_exposure_max"],
        rebound_exit_market_exposure_min=params["rebound_exit_market_exposure_min"],
    )
    if equity.empty:
        raise ValueError("Trial generated no equity rows")
    return StrategyRun(equity, weights, trades, candidates)


def write_trial_artifacts(
    trial_dir: Path,
    run: StrategyRun,
    metrics: Mapping[str, object],
    trial_state: Mapping[str, object],
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    run.equity.to_csv(trial_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    run.weights.to_csv(trial_dir / "rolling_feature_weights.csv", index=False, encoding="utf-8-sig")
    run.trades.to_csv(trial_dir / "trade_audit.csv", index=False, encoding="utf-8-sig")
    run.candidates.to_csv(trial_dir / "selected_candidates.csv", index=False, encoding="utf-8-sig")
    (trial_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (trial_dir / "trial_state.json").write_text(json.dumps(trial_state, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_csv_or_empty(path: Path, **kwargs: object) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except EmptyDataError:
        return pd.DataFrame()


def load_trial_artifacts(trial_dir: Path) -> StrategyRun:
    return StrategyRun(
        equity=_read_csv_or_empty(trial_dir / "equity_curve.csv", parse_dates=["date"]),
        weights=_read_csv_or_empty(trial_dir / "rolling_feature_weights.csv"),
        trades=_read_csv_or_empty(trial_dir / "trade_audit.csv"),
        candidates=_read_csv_or_empty(trial_dir / "selected_candidates.csv", dtype={"symbol": "string"}),
    )
