"""Rolling training and walk-forward validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._compat import read_text

from .backtest import BacktestResult, run_backtest
from .config import LabConfig
from .data import filter_history_by_date, load_universe_history, parse_market_date
from .report import write_report
from .strategies import build_signal_frames


@dataclass(frozen=True)
class RollingWindow:
    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class WalkForwardValidationDates:
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str


@dataclass(frozen=True)
class ParameterCandidate:
    name: str
    factor_weights: dict[str, float]
    rebalance_interval: int
    max_positions: int
    risk_overrides: dict[str, Any]


def _yyyymmdd(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _display_date(value: str) -> str:
    return pd.Timestamp(parse_market_date(value)).strftime("%Y-%m-%d")


def generate_rolling_windows(
    start_date: str,
    end_date: str,
    train_years: int = 3,
    test_months: int = 12,
    step_months: int = 12,
) -> tuple[RollingWindow, ...]:
    start = parse_market_date(start_date)
    end = parse_market_date(end_date)
    if start is None or end is None:
        raise ValueError("Rolling windows require concrete start and end dates.")
    if train_years <= 0 or test_months <= 0 or step_months <= 0:
        raise ValueError("train_years, test_months, and step_months must be positive.")

    windows: list[RollingWindow] = []
    cursor = start.normalize()
    index = 1
    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        if test_start > end:
            break
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), end)
        windows.append(
            RollingWindow(
                name=f"wf_{index:02d}_{_yyyymmdd(test_start)}_{_yyyymmdd(test_end)}",
                train_start=_yyyymmdd(train_start),
                train_end=_yyyymmdd(train_end),
                test_start=_yyyymmdd(test_start),
                test_end=_yyyymmdd(test_end),
            )
        )
        cursor = cursor + pd.DateOffset(months=step_months)
        index += 1
    return tuple(windows)


def _walk_forward_validation_dates(
    window: RollingWindow,
    validation_months: int = 0,
) -> WalkForwardValidationDates | None:
    if validation_months <= 0:
        return None
    train_start = parse_market_date(window.train_start)
    train_end = parse_market_date(window.train_end)
    if train_start is None or train_end is None:
        raise ValueError("Walk-forward validation requires concrete training dates.")
    validation_start = train_end - pd.DateOffset(months=validation_months) + pd.Timedelta(days=1)
    fit_end = validation_start - pd.Timedelta(days=1)
    if validation_start <= train_start or fit_end < train_start:
        raise ValueError("selection_validation_months must be shorter than the training window.")
    return WalkForwardValidationDates(
        train_start=_yyyymmdd(train_start),
        train_end=_yyyymmdd(fit_end),
        validation_start=_yyyymmdd(validation_start),
        validation_end=_yyyymmdd(train_end),
    )


def _preserve_extra_factor_weights(
    factor_weights: dict[str, float],
    current_weights: dict[str, float],
) -> dict[str, float]:
    extras = {
        name: float(weight)
        for name, weight in current_weights.items()
        if name not in factor_weights and float(weight) != 0.0
    }
    if not extras:
        return dict(factor_weights)

    extra_total = sum(extras.values())
    if extra_total >= 1.0:
        return {name: weight / extra_total for name, weight in extras.items()}

    base_total = sum(float(weight) for weight in factor_weights.values())
    base_scale = (1.0 - extra_total) / base_total if base_total else 0.0
    merged = {name: float(weight) * base_scale for name, weight in factor_weights.items()}
    merged.update(extras)
    return merged


def build_parameter_grid(config: LabConfig, grid: str = "compact") -> tuple[ParameterCandidate, ...]:
    current_weights = dict(config.strategy.factor_weights)
    named_factor_sets = {
        "current": current_weights,
        "base": {"momentum": 0.35, "trend": 0.25, "reversal": 0.15, "volatility": 0.15, "liquidity": 0.10},
        "momentum_trend": {"momentum": 0.40, "trend": 0.30, "reversal": 0.10, "volatility": 0.10, "liquidity": 0.10},
        "defensive_lowvol": {"momentum": 0.15, "trend": 0.15, "reversal": 0.25, "volatility": 0.35, "liquidity": 0.10},
        "reversal_lowvol": {"momentum": 0.10, "trend": 0.15, "reversal": 0.40, "volatility": 0.25, "liquidity": 0.10},
        "reversal_focus": {"momentum": 0.05, "trend": 0.10, "reversal": 0.55, "volatility": 0.20, "liquidity": 0.10},
        "reversal_defensive": {"momentum": 0.05, "trend": 0.05, "reversal": 0.50, "volatility": 0.30, "liquidity": 0.10},
        "broad_balanced": {"momentum": 0.20, "trend": 0.20, "reversal": 0.25, "volatility": 0.25, "liquidity": 0.10},
        "broad_defensive": {"momentum": 0.15, "trend": 0.20, "reversal": 0.25, "volatility": 0.30, "liquidity": 0.10},
        "growth_momentum": {"momentum": 0.45, "trend": 0.30, "reversal": 0.05, "volatility": 0.10, "liquidity": 0.10},
        "trend_quality": {"momentum": 0.35, "trend": 0.35, "reversal": 0.05, "volatility": 0.15, "liquidity": 0.10},
        "pullback_growth": {"momentum": 0.35, "trend": 0.25, "reversal": 0.25, "volatility": 0.05, "liquidity": 0.10},
        "breakout_momentum": {"momentum": 0.60, "trend": 0.25, "reversal": 0.00, "volatility": 0.08, "liquidity": 0.07},
        "trend_accel": {"momentum": 0.50, "trend": 0.35, "reversal": 0.00, "volatility": 0.10, "liquidity": 0.05},
        "strong_gain_quality": {"momentum": 0.52, "trend": 0.28, "reversal": 0.05, "volatility": 0.10, "liquidity": 0.05},
    }
    named_factor_sets = {
        name: _preserve_extra_factor_weights(weights, current_weights)
        for name, weights in named_factor_sets.items()
    }
    compact_specs = (
        ("current", 10),
        ("current", 20),
        ("defensive_lowvol", 20),
        ("reversal_lowvol", 10),
        ("momentum_trend", 10),
    )
    bear_specs = (
        ("current", 20),
        ("defensive_lowvol", 20),
        ("reversal_lowvol", 10),
    )
    opportunity_specs = (
        ("current", 10),
        ("current", 20),
        ("momentum_trend", 10),
        ("defensive_lowvol", 20),
    )
    opportunity_v2_specs = (
        ("current", 8, 8),
        ("current", 10, 10),
        ("current", 20, 10),
        ("breakout_momentum", 8, 8),
        ("breakout_momentum", 10, 10),
        ("trend_accel", 10, 10),
        ("strong_gain_quality", 15, 12),
        ("pullback_growth", 10, 10),
    )
    stable_specs = (
        ("broad_balanced", 20, 10),
        ("broad_balanced", 40, 10),
        ("broad_defensive", 20, 10),
        ("broad_defensive", 40, 15),
        ("defensive_lowvol", 20, 10),
        ("reversal_lowvol", 20, 10),
    )
    stable_v2_specs = (
        ("broad_balanced", 40, 15),
        ("broad_defensive", 40, 15),
        ("defensive_lowvol", 20, 15),
        ("reversal_lowvol", 20, 10),
    )
    reversal_specs = (
        ("reversal_focus", 20, 10),
        ("reversal_focus", 40, 10),
        ("reversal_defensive", 20, 10),
        ("reversal_lowvol", 20, 10),
    )
    satellite_specs = (
        ("growth_momentum", 10, 10),
        ("growth_momentum", 20, 15),
        ("trend_quality", 20, 15),
        ("pullback_growth", 10, 10),
        ("momentum_trend", 10, 10),
        ("momentum_trend", 20, 15),
    )
    risk_specs = [
        (factor_name, rebalance)
        for factor_name in named_factor_sets
        for rebalance in (10, 20)
    ]
    bear_profiles = [
        ("risk_current", {}),
        (
            "risk_bear_guard",
            {
                "benchmark_off_exposure": 0.50,
                "benchmark_crash_exposure": 0.0,
                "benchmark_drop_threshold": -0.06,
                "drawdown_levels": ((0.08, 0.65), (0.15, 0.45), (0.25, 0.15), (0.30, 0.0)),
                "protection_drawdown": 0.30,
                "recovery_drawdown": 0.14,
            },
        ),
        (
            "risk_bear_strict",
            {
                "benchmark_off_exposure": 0.35,
                "benchmark_crash_exposure": 0.0,
                "benchmark_drop_threshold": -0.05,
                "drawdown_levels": ((0.08, 0.55), (0.15, 0.30), (0.25, 0.0)),
                "protection_drawdown": 0.25,
                "recovery_drawdown": 0.12,
            },
        ),
    ]
    opportunity_profiles = [
        ("risk_current", {}),
        (
            "risk_bear_guard",
            {
                "benchmark_off_exposure": 0.50,
                "benchmark_crash_exposure": 0.0,
                "benchmark_drop_threshold": -0.06,
                "drawdown_levels": ((0.08, 0.65), (0.15, 0.45), (0.25, 0.15), (0.30, 0.0)),
                "protection_drawdown": 0.30,
                "recovery_drawdown": 0.14,
            },
        ),
        (
            "risk_bear_growth",
            {
                "benchmark_off_exposure": 0.65,
                "benchmark_crash_exposure": 0.10,
                "benchmark_drop_threshold": -0.07,
                "drawdown_levels": ((0.10, 0.75), (0.20, 0.55), (0.30, 0.25), (0.35, 0.0)),
                "protection_drawdown": 0.35,
                "recovery_drawdown": 0.16,
            },
        ),
    ]
    opportunity_v2_profiles = [
        ("risk_current", {}),
        (
            "risk_loose",
            {
                "benchmark_off_exposure": 0.90,
                "benchmark_crash_exposure": 0.30,
                "drawdown_levels": ((0.10, 0.90), (0.20, 0.75), (0.40, 0.0)),
                "protection_drawdown": 0.40,
                "recovery_drawdown": 0.20,
            },
        ),
        (
            "risk_growth_dd40",
            {
                "benchmark_off_exposure": 0.70,
                "benchmark_crash_exposure": 0.10,
                "benchmark_drop_threshold": -0.07,
                "drawdown_levels": ((0.12, 0.85), (0.24, 0.60), (0.35, 0.20), (0.40, 0.0)),
                "protection_drawdown": 0.40,
                "recovery_drawdown": 0.18,
            },
        ),
        (
            "risk_growth_on",
            {
                "benchmark_off_exposure": 0.90,
                "benchmark_crash_exposure": 0.20,
                "benchmark_drop_threshold": -0.08,
                "drawdown_levels": ((0.15, 0.90), (0.30, 0.65), (0.40, 0.0)),
                "protection_drawdown": 0.40,
                "recovery_drawdown": 0.20,
            },
        ),
    ]
    stable_profiles = [
        (
            "risk_stable_guard",
            {
                "benchmark_off_exposure": 0.50,
                "benchmark_crash_exposure": 0.0,
                "benchmark_drop_threshold": -0.06,
                "drawdown_levels": ((0.08, 0.65), (0.15, 0.45), (0.25, 0.15), (0.30, 0.0)),
                "protection_drawdown": 0.30,
                "recovery_drawdown": 0.14,
            },
        ),
        (
            "risk_stable_strict",
            {
                "benchmark_off_exposure": 0.35,
                "benchmark_crash_exposure": 0.0,
                "benchmark_drop_threshold": -0.05,
                "drawdown_levels": ((0.08, 0.55), (0.15, 0.30), (0.25, 0.0)),
                "protection_drawdown": 0.25,
                "recovery_drawdown": 0.12,
            },
        ),
    ]
    stable_v2_profiles = [
        (
            "risk_stable_strict",
            {
                "benchmark_off_exposure": 0.35,
                "benchmark_crash_exposure": 0.0,
                "benchmark_drop_threshold": -0.05,
                "drawdown_levels": ((0.08, 0.55), (0.15, 0.30), (0.25, 0.0)),
                "protection_drawdown": 0.25,
                "recovery_drawdown": 0.12,
            },
        ),
        (
            "risk_stable_ultra",
            {
                "benchmark_off_exposure": 0.20,
                "benchmark_crash_exposure": 0.0,
                "benchmark_drop_threshold": -0.04,
                "drawdown_levels": ((0.06, 0.45), (0.12, 0.20), (0.20, 0.0)),
                "protection_drawdown": 0.20,
                "recovery_drawdown": 0.08,
            },
        ),
    ]
    reversal_profiles = [
        (
            "risk_reversal_guard",
            {
                "benchmark_off_exposure": 0.50,
                "benchmark_crash_exposure": 0.0,
                "benchmark_drop_threshold": -0.06,
                "drawdown_levels": ((0.08, 0.65), (0.15, 0.45), (0.25, 0.15), (0.30, 0.0)),
                "protection_drawdown": 0.30,
                "recovery_drawdown": 0.14,
            },
        ),
        (
            "risk_reversal_strict",
            {
                "benchmark_off_exposure": 0.35,
                "benchmark_crash_exposure": 0.0,
                "benchmark_drop_threshold": -0.05,
                "drawdown_levels": ((0.08, 0.55), (0.15, 0.30), (0.25, 0.0)),
                "protection_drawdown": 0.25,
                "recovery_drawdown": 0.12,
            },
        ),
    ]
    satellite_profiles = [
        (
            "risk_satellite_guard",
            {
                "benchmark_off_exposure": 0.40,
                "benchmark_crash_exposure": 0.0,
                "benchmark_drop_threshold": -0.06,
                "drawdown_levels": ((0.10, 0.75), (0.20, 0.45), (0.30, 0.0)),
                "protection_drawdown": 0.30,
                "recovery_drawdown": 0.14,
            },
        ),
        (
            "risk_satellite_growth",
            {
                "benchmark_off_exposure": 0.65,
                "benchmark_crash_exposure": 0.10,
                "benchmark_drop_threshold": -0.07,
                "drawdown_levels": ((0.12, 0.85), (0.24, 0.55), (0.35, 0.15), (0.40, 0.0)),
                "protection_drawdown": 0.40,
                "recovery_drawdown": 0.18,
            },
        ),
    ]
    satellite_v2_profiles = [
        (
            "risk_satellite_guard",
            {
                "benchmark_off_exposure": 0.40,
                "benchmark_crash_exposure": 0.0,
                "benchmark_drop_threshold": -0.06,
                "drawdown_levels": ((0.10, 0.75), (0.20, 0.45), (0.30, 0.0)),
                "protection_drawdown": 0.30,
                "recovery_drawdown": 0.14,
            },
        ),
        (
            "risk_satellite_on_only",
            {
                "benchmark_off_exposure": 0.0,
                "benchmark_crash_exposure": 0.0,
                "benchmark_drop_threshold": -0.06,
                "drawdown_levels": ((0.10, 0.70), (0.20, 0.40), (0.30, 0.0)),
                "protection_drawdown": 0.30,
                "recovery_drawdown": 0.14,
            },
        ),
    ]
    risk_profiles: list[tuple[str, dict[str, Any]]] = [("risk_current", {})]
    if grid == "risk":
        risk_profiles.extend(
            [
                (
                    "risk_strict",
                    {
                        "benchmark_off_exposure": 0.60,
                        "benchmark_crash_exposure": 0.0,
                        "drawdown_levels": ((0.10, 0.70), (0.20, 0.45), (0.30, 0.0)),
                        "protection_drawdown": 0.30,
                        "recovery_drawdown": 0.15,
                    },
                ),
                (
                    "risk_loose",
                    {
                        "benchmark_off_exposure": 0.90,
                        "benchmark_crash_exposure": 0.30,
                        "drawdown_levels": ((0.10, 0.90), (0.20, 0.75), (0.40, 0.0)),
                        "protection_drawdown": 0.40,
                        "recovery_drawdown": 0.20,
                    },
                ),
            ]
        )
    elif grid == "bear":
        risk_profiles = bear_profiles
    elif grid == "opportunity":
        risk_profiles = opportunity_profiles
    elif grid == "opportunity_v2":
        risk_profiles = opportunity_v2_profiles
    elif grid == "stable":
        risk_profiles = stable_profiles
    elif grid == "stable_v2":
        risk_profiles = stable_v2_profiles
    elif grid == "reversal":
        risk_profiles = reversal_profiles
    elif grid == "satellite":
        risk_profiles = satellite_profiles
    elif grid == "satellite_v2":
        risk_profiles = satellite_v2_profiles
    elif grid != "compact":
        raise ValueError(
            "grid must be 'compact', 'bear', 'opportunity', 'opportunity_v2', 'stable', 'stable_v2', 'reversal', 'satellite', 'satellite_v2', or 'risk'."
        )

    candidates: list[ParameterCandidate] = []
    max_positions = max(config.strategy.max_positions, 1)
    if grid == "compact":
        factor_specs = compact_specs
    elif grid == "bear":
        factor_specs = bear_specs
    elif grid == "opportunity":
        factor_specs = opportunity_specs
    elif grid == "opportunity_v2":
        factor_specs = opportunity_v2_specs
    elif grid == "stable":
        factor_specs = stable_specs
    elif grid == "stable_v2":
        factor_specs = stable_v2_specs
    elif grid == "reversal":
        factor_specs = reversal_specs
    elif grid == "satellite":
        factor_specs = satellite_specs
    elif grid == "satellite_v2":
        factor_specs = satellite_specs
    else:
        factor_specs = tuple(risk_specs)
    for spec in factor_specs:
        if len(spec) == 2:
            factor_name, rebalance = spec
            candidate_max_positions = max_positions
        else:
            factor_name, rebalance, candidate_max_positions = spec
        weights = named_factor_sets[factor_name]
        for risk_name, risk_overrides in risk_profiles:
            candidates.append(
                ParameterCandidate(
                    name=f"{factor_name}_reb{rebalance}_{risk_name}",
                    factor_weights=weights,
                    rebalance_interval=rebalance,
                    max_positions=max(int(candidate_max_positions), 1),
                    risk_overrides=risk_overrides,
                )
            )
    return tuple(candidates)


def _candidate_config(config: LabConfig, candidate: ParameterCandidate, start_date: str, end_date: str) -> LabConfig:
    strategy = replace(
        config.strategy,
        factor_weights=dict(candidate.factor_weights),
        factor_rebalance_interval=candidate.rebalance_interval,
        max_positions=candidate.max_positions,
    )
    risk = replace(config.risk, **candidate.risk_overrides) if candidate.risk_overrides else config.risk
    return replace(
        config,
        data=replace(config.data, start_date=start_date, end_date=end_date),
        strategy=strategy,
        risk=risk,
    )


def _signal_key(candidate: ParameterCandidate) -> str:
    payload = {
        "factor_weights": candidate.factor_weights,
        "rebalance_interval": candidate.rebalance_interval,
        "max_positions": candidate.max_positions,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _group_candidates_by_signal(candidates: tuple[ParameterCandidate, ...]) -> list[list[ParameterCandidate]]:
    grouped: dict[str, list[ParameterCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(_signal_key(candidate), []).append(candidate)
    return list(grouped.values())


def _slice_histories(
    histories: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    sliced: dict[str, pd.DataFrame] = {}
    for code, frame in histories.items():
        window = filter_history_by_date(frame, start_date, end_date)
        if not window.empty:
            sliced[code] = window
    return sliced


def _signal_warmup_start(config: LabConfig, start_date: str) -> str:
    start = parse_market_date(start_date)
    project_start = parse_market_date(config.data.start_date)
    if start is None:
        raise ValueError("Warm-up start requires a concrete start date.")
    lookback_bars = max(
        config.strategy.factor_min_history,
        config.strategy.factor_momentum_window,
        config.strategy.factor_reversal_window,
        config.strategy.factor_volatility_window,
        config.strategy.factor_liquidity_window,
        config.strategy.factor_trend_window,
        config.strategy.factor_entry_trend_window,
        config.strategy.factor_entry_momentum_window,
        config.strategy.slow_window,
        config.strategy.trend_window,
        config.strategy.rsi_window,
    )
    warmup_start = start - pd.Timedelta(days=max(lookback_bars + 20, 1) * 3)
    if project_start is not None:
        warmup_start = max(project_start, warmup_start)
    return _yyyymmdd(warmup_start)


def _trim_signal_frames(
    signal_frames: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    trimmed: dict[str, pd.DataFrame] = {}
    for code, frame in signal_frames.items():
        window = filter_history_by_date(frame, start_date, end_date)
        if not window.empty:
            trimmed[code] = window
    return trimmed


def _yearly_returns(series: pd.Series) -> dict[int, float]:
    if series.empty:
        return {}
    returns: dict[int, float] = {}
    for year, year_series in series.groupby(series.index.year):
        if year_series.empty:
            continue
        first = float(year_series.iloc[0])
        last = float(year_series.iloc[-1])
        returns[int(year)] = (last / first - 1) if first > 0 else 0.0
    return returns


def _yearly_drawdowns(series: pd.Series) -> dict[int, float]:
    if series.empty:
        return {}
    drawdowns: dict[int, float] = {}
    for year, year_series in series.groupby(series.index.year):
        if year_series.empty:
            continue
        running_max = year_series.cummax()
        drawdowns[int(year)] = float((year_series / running_max - 1).min())
    return drawdowns


def _monthly_return_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    try:
        monthly_equity = series.resample("ME").last()
    except ValueError:
        monthly_equity = series.resample("M").last()
    return monthly_equity.pct_change().dropna()


def training_diagnostics(result: BacktestResult) -> dict[str, float]:
    if result.equity.empty:
        return {
            "worst_year_return": 0.0,
            "negative_year_ratio": 0.0,
            "worst_year_drawdown": 0.0,
            "bear_year_count": 0.0,
            "bear_year_avg_return": 0.0,
            "trades_per_year": 0.0,
            "annual_return_std": 0.0,
            "return_concentration": 0.0,
            "monthly_win_rate": 0.0,
            "worst_month_return": 0.0,
        }

    equity = result.equity.set_index("date")["equity"].sort_index()
    yearly_returns = _yearly_returns(equity)
    yearly_drawdowns = _yearly_drawdowns(equity)
    negative_years = [value for value in yearly_returns.values() if value < 0]
    annual_values = list(yearly_returns.values())
    positive_year_returns = [value for value in annual_values if value > 0]
    positive_return_sum = float(sum(positive_year_returns))
    return_concentration = (
        float(max(positive_year_returns) / positive_return_sum)
        if positive_return_sum > 0
        else (1.0 if annual_values else 0.0)
    )
    monthly_returns = _monthly_return_series(equity)

    benchmark_returns: dict[int, float] = {}
    if not result.benchmark.empty and "benchmark_equity" in result.benchmark:
        benchmark = result.benchmark.set_index("date")["benchmark_equity"].sort_index()
        benchmark_returns = _yearly_returns(benchmark)
    bear_years = [
        year
        for year, benchmark_return in benchmark_returns.items()
        if benchmark_return <= -0.05 and year in yearly_returns
    ]
    bear_returns = [yearly_returns[year] for year in bear_years]

    first_date = pd.Timestamp(equity.index.min())
    last_date = pd.Timestamp(equity.index.max())
    years = max((last_date - first_date).days / 365.25, 1 / 365.25)
    trade_count = float(result.metrics.get("trade_count", 0.0) or 0.0)
    return {
        "worst_year_return": float(min(yearly_returns.values())) if yearly_returns else 0.0,
        "negative_year_ratio": float(len(negative_years) / len(yearly_returns)) if yearly_returns else 0.0,
        "worst_year_drawdown": float(min(yearly_drawdowns.values())) if yearly_drawdowns else 0.0,
        "bear_year_count": float(len(bear_years)),
        "bear_year_avg_return": float(np.mean(bear_returns)) if bear_returns else 0.0,
        "trades_per_year": trade_count / years,
        "annual_return_std": float(np.std(annual_values)) if len(annual_values) > 1 else 0.0,
        "return_concentration": return_concentration,
        "monthly_win_rate": float((monthly_returns > 0).mean()) if len(monthly_returns) else 0.0,
        "worst_month_return": float(monthly_returns.min()) if len(monthly_returns) else 0.0,
    }


def score_training_metrics(
    metrics: dict[str, Any],
    max_drawdown_limit: float = 0.35,
    objective: str = "balanced",
    diagnostics: dict[str, float] | None = None,
) -> float:
    total_return = float(metrics.get("total_return", 0.0) or 0.0)
    cagr = float(metrics.get("cagr", 0.0) or 0.0)
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    max_drawdown = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
    profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
    win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
    average_risk_exposure = float(metrics.get("average_risk_exposure", 1.0) or 1.0)
    drawdown_penalty = max(0.0, max_drawdown - max_drawdown_limit)
    loss_penalty = max(0.0, -total_return)

    if objective == "balanced":
        return sharpe + cagr + 0.25 * total_return - drawdown_penalty * 4.0 - loss_penalty * 0.5
    if objective == "capital":
        return (
            0.28 * sharpe
            + 0.32 * cagr
            + 0.18 * total_return
            + 0.20 * win_rate
            + 0.15 * min(average_risk_exposure, 1.0)
            - drawdown_penalty * 5.0
            - loss_penalty * 0.8
        )
    if objective == "opportunity_q_full":
        monthly_win_rate = float((diagnostics or {}).get("monthly_win_rate", 0.0) or 0.0)
        annual_return_std = float((diagnostics or {}).get("annual_return_std", 0.0) or 0.0)
        return_concentration = float((diagnostics or {}).get("return_concentration", 0.0) or 0.0)
        worst_month_return = abs(float((diagnostics or {}).get("worst_month_return", 0.0) or 0.0))
        base = 0.62 * cagr + 0.46 * total_return + 0.06 * sharpe
        execution_bonus = 0.15 * win_rate + 0.12 * monthly_win_rate + 0.08 * min(profit_factor, 3.0)
        exposure_bonus = 0.20 * min(average_risk_exposure, 1.0)
        utilization_penalty = max(0.0, 0.55 - average_risk_exposure) * 0.35
        stability_penalty = (
            annual_return_std * 0.35
            + max(0.0, return_concentration - 0.72) * 0.60
            + max(0.0, worst_month_return - 0.18) * 1.05
        )
        return (
            base
            + execution_bonus
            + exposure_bonus
            - drawdown_penalty * 5.2
            - loss_penalty * 0.75
            - utilization_penalty
            - stability_penalty
            - max(0.0, average_risk_exposure - 1.0) * 0.20
            - max(0.0, -win_rate) * 0.10
        )
    if objective == "opportunity_growth":
        monthly_win_rate = float((diagnostics or {}).get("monthly_win_rate", 0.0) or 0.0)
        base = 0.40 * cagr + 0.35 * total_return + 0.20 * sharpe
        execution_bonus = 0.12 * win_rate + 0.10 * monthly_win_rate + 0.10 * min(profit_factor, 2.5)
        exposure_bonus = 0.20 * min(average_risk_exposure, 0.95)
        utilization_penalty = max(0.0, 0.60 - average_risk_exposure) * 0.25
        return (
            base
            + execution_bonus
            + exposure_bonus
            - drawdown_penalty * 5.5
            - loss_penalty * 0.9
            - utilization_penalty
            - max(0.0, average_risk_exposure - 1.0) * 0.30
            - max(0.0, -win_rate) * 0.10
        )
    if objective == "opportunity_quality":
        monthly_win_rate = float((diagnostics or {}).get("monthly_win_rate", 0.0) or 0.0)
        base = 0.35 * sharpe + 0.40 * cagr + 0.22 * total_return
        success_bonus = 0.22 * win_rate + 0.12 * monthly_win_rate + 0.10 * min(profit_factor, 2.0)
        utilization_bonus = 0.20 * min(average_risk_exposure, 0.90)
        overuse_penalty = max(0.0, average_risk_exposure - 0.90) * 0.35
        return (
            base
            + success_bonus
            + utilization_bonus
            - drawdown_penalty * 6.0
            - loss_penalty * 0.9
            - overuse_penalty
        )
    if objective not in {"robust", "stable", "satellite"}:
        raise ValueError(
            "objective must be 'balanced', 'robust', 'stable', 'satellite', 'capital', 'opportunity_quality', 'opportunity_growth', or 'opportunity_q_full'."
        )

    diagnostics = diagnostics or {}
    worst_year_return = float(diagnostics.get("worst_year_return", 0.0) or 0.0)
    negative_year_ratio = float(diagnostics.get("negative_year_ratio", 0.0) or 0.0)
    worst_year_drawdown = abs(float(diagnostics.get("worst_year_drawdown", 0.0) or 0.0))
    bear_year_avg_return = float(diagnostics.get("bear_year_avg_return", 0.0) or 0.0)
    trades_per_year = float(diagnostics.get("trades_per_year", 0.0) or 0.0)

    robust_base = 0.45 * sharpe + 0.45 * cagr + 0.15 * total_return
    quality_bonus = 0.08 * min(profit_factor, 2.0) + 0.08 * win_rate
    annual_loss_penalty = max(0.0, -worst_year_return) * 1.25
    negative_year_penalty = negative_year_ratio * 0.55
    annual_drawdown_penalty = max(0.0, worst_year_drawdown - max_drawdown_limit) * 4.0
    bear_penalty = max(0.0, -bear_year_avg_return) * 0.75
    turnover_penalty = max(0.0, trades_per_year - 800.0) / 800.0 * 0.12
    robust_score = (
        robust_base
        + quality_bonus
        - drawdown_penalty * 7.0
        - loss_penalty * 1.0
        - annual_loss_penalty
        - negative_year_penalty
        - annual_drawdown_penalty
        - bear_penalty
        - turnover_penalty
    )
    if objective == "robust":
        return robust_score

    annual_return_std = float(diagnostics.get("annual_return_std", 0.0) or 0.0)
    return_concentration = float(diagnostics.get("return_concentration", 0.0) or 0.0)
    monthly_win_rate = float(diagnostics.get("monthly_win_rate", 0.0) or 0.0)
    worst_month_return = float(diagnostics.get("worst_month_return", 0.0) or 0.0)
    if objective == "satellite":
        satellite_base = 0.35 * sharpe + 0.50 * cagr + 0.28 * total_return
        participation_bonus = 0.08 * min(average_risk_exposure, 0.80)
        monthly_bonus = 0.08 * monthly_win_rate
        unstable_year_penalty = annual_return_std * 0.30
        concentration_penalty = max(0.0, return_concentration - 0.55) * 0.65
        tail_month_penalty = max(0.0, abs(worst_month_return) - 0.15) * 1.00
        too_good_penalty = max(0.0, sharpe - 1.80) * 0.08 + max(0.0, cagr - 0.40) * 0.25
        return (
            satellite_base
            + quality_bonus
            + participation_bonus
            + monthly_bonus
            - drawdown_penalty * 5.0
            - loss_penalty * 0.90
            - annual_loss_penalty * 0.80
            - negative_year_penalty * 0.45
            - annual_drawdown_penalty * 0.65
            - bear_penalty * 0.35
            - turnover_penalty
            - unstable_year_penalty
            - concentration_penalty
            - tail_month_penalty
            - too_good_penalty
        )

    stability_bonus = 0.12 * monthly_win_rate
    annual_volatility_penalty = annual_return_std * 0.45
    concentration_penalty = max(0.0, return_concentration - 0.45) * 0.80
    tail_month_penalty = max(0.0, abs(worst_month_return) - 0.12) * 1.25
    too_good_penalty = max(0.0, sharpe - 1.50) * 0.10 + max(0.0, cagr - 0.30) * 0.35
    risk_exposure_penalty = max(0.0, average_risk_exposure - 0.45) * 0.40
    return (
        robust_score
        + stability_bonus
        - drawdown_penalty * 3.0
        - annual_volatility_penalty
        - concentration_penalty
        - tail_month_penalty
        - too_good_penalty
        - risk_exposure_penalty
    )


def score_training_result(
    result: BacktestResult,
    max_drawdown_limit: float = 0.35,
    objective: str = "robust",
) -> tuple[float, dict[str, float]]:
    diagnostics = training_diagnostics(result)
    score = score_training_metrics(
        result.metrics,
        max_drawdown_limit=max_drawdown_limit,
        objective=objective,
        diagnostics=diagnostics,
    )
    return score, diagnostics


def _metrics_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "final_equity",
        "total_return",
        "benchmark_return",
        "excess_return",
        "cagr",
        "max_drawdown",
        "sharpe",
        "trade_count",
        "win_rate",
        "payoff_ratio",
        "profit_factor",
        "average_position_exposure",
        "average_risk_exposure",
        "risk_off_day_ratio",
        "risk_reduced_day_ratio",
        "average_market_breadth_exposure",
        "market_breadth_reduced_day_ratio",
    ]
    return {f"{prefix}_{key}": metrics.get(key) for key in keys}


def _candidate_params(candidate: ParameterCandidate) -> str:
    payload = {
        "factor_weights": candidate.factor_weights,
        "rebalance_interval": candidate.rebalance_interval,
        "max_positions": candidate.max_positions,
        "risk_overrides": candidate.risk_overrides,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _candidate_selection_gate_reason(
    candidate: ParameterCandidate,
    validation_metrics: dict[str, Any] | None,
    *,
    grid: str,
) -> str | None:
    if grid != "opportunity_v2" or validation_metrics is None:
        return None
    if candidate.name.startswith("current_"):
        return None

    validation_return = float(validation_metrics.get("total_return", 0.0) or 0.0)
    validation_sharpe = float(validation_metrics.get("sharpe", 0.0) or 0.0)
    if validation_return < 0.0:
        return "non_current_negative_validation_return"
    if validation_sharpe < 0.0:
        return "non_current_negative_validation_sharpe"
    return None


def _stitch_oos_equity(results: list[BacktestResult], initial_cash: float) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    window_initial_cash = float(initial_cash)
    base_equity = window_initial_cash
    if window_initial_cash <= 0:
        return pd.DataFrame(columns=["date", "window", "stitched_equity"])
    for result in results:
        equity = result.equity.copy()
        if equity.empty:
            continue
        if float(equity["equity"].iloc[-1]) <= 0:
            continue
        equity["stitched_equity"] = equity["equity"] / window_initial_cash * base_equity
        equity["window"] = result.run_id
        base_equity = float(equity["stitched_equity"].iloc[-1])
        records.append(equity[["date", "window", "stitched_equity"]])
    if not records:
        return pd.DataFrame(columns=["date", "window", "stitched_equity"])
    return pd.concat(records, ignore_index=True)


def _read_csv_if_exists(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)


def _load_backtest_result(run_dir: Path) -> BacktestResult | None:
    metrics_path = run_dir / "metrics.json"
    equity_path = run_dir / "equity_curve.csv"
    if not metrics_path.exists() or not equity_path.exists():
        return None
    metrics = json.loads(read_text(metrics_path))
    return BacktestResult(
        run_id=run_dir.name,
        run_dir=run_dir,
        equity=_read_csv_if_exists(equity_path, parse_dates=["date"]),
        trades=_read_csv_if_exists(run_dir / "trades.csv", parse_dates=["date"]),
        benchmark=_read_csv_if_exists(run_dir / "benchmark.csv", parse_dates=["date"]),
        monthly_returns=_read_csv_if_exists(run_dir / "monthly_returns.csv", parse_dates=["date"]),
        metrics=metrics,
        risk_curve=_read_csv_if_exists(run_dir / "risk_curve.csv", parse_dates=["date"]),
        risk_events=_read_csv_if_exists(run_dir / "risk_events.csv", parse_dates=["date"]),
        cooldown_events=_read_csv_if_exists(run_dir / "cooldown_events.csv", parse_dates=["date"]),
    )


def _find_existing_test_result(config: LabConfig, window_name: str) -> tuple[str | None, BacktestResult | None]:
    if not config.project.output_dir.exists():
        return None, None
    matches = sorted(
        (
            path
            for path in config.project.output_dir.iterdir()
            if path.is_dir() and f"_{window_name}_" in path.name and path.name.endswith("_test")
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in matches:
        result = _load_backtest_result(run_dir)
        if result is None:
            continue
        marker = f"_{window_name}_"
        selected = run_dir.name.split(marker, 1)[1].removesuffix("_test")
        return selected, result
    return None, None


def _load_walk_forward_checkpoint(
    wf_dir: Path,
    windows: tuple[RollingWindow, ...],
    config: LabConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[BacktestResult]]:
    summary_path = wf_dir / "walk_forward_summary.csv"
    candidates_path = wf_dir / "candidate_results.csv"
    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    test_results: list[BacktestResult] = []
    if summary_path.exists():
        summary_rows = pd.read_csv(summary_path).to_dict("records")
        if candidates_path.exists():
            candidate_rows = pd.read_csv(candidates_path).to_dict("records")
        for row in summary_rows:
            run_dir_raw = row.get("test_run_dir")
            if isinstance(run_dir_raw, str) and run_dir_raw:
                result = _load_backtest_result(Path(run_dir_raw))
                if result is not None:
                    test_results.append(result)
        return summary_rows, candidate_rows, test_results

    windows_by_name = {window.name: window for window in windows}
    for params_path in sorted(wf_dir.glob("*_selected_params.json")):
        window_name = params_path.name.removesuffix("_selected_params.json")
        window = windows_by_name.get(window_name)
        if window is None:
            continue
        selected, result = _find_existing_test_result(config, window_name)
        if selected is None or result is None:
            continue
        summary_rows.append(
            {
                "window": window.name,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "selected_candidate": selected,
                "selected_score": None,
                "selected_params_path": str(params_path),
                "test_run_dir": str(result.run_dir),
                **_metrics_row("train", {}),
                **_metrics_row("test", result.metrics),
                "train_test_return_gap": None,
                "train_test_sharpe_gap": None,
            }
        )
        test_results.append(result)
    return summary_rows, candidate_rows, test_results


def _stitched_metrics(stitched: pd.DataFrame, initial_cash: float) -> dict[str, float]:
    if stitched.empty:
        return {"stitched_total_return": 0.0, "stitched_max_drawdown": 0.0, "stitched_sharpe": 0.0}
    series = stitched.set_index("date")["stitched_equity"].sort_index()
    returns = series.pct_change().dropna()
    running_max = series.cummax()
    drawdown = series / running_max - 1
    sharpe = 0.0
    if len(returns) > 1 and float(returns.std()) != 0:
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std())
    return {
        "stitched_total_return": float(series.iloc[-1] / initial_cash - 1),
        "stitched_max_drawdown": float(drawdown.min()),
        "stitched_sharpe": sharpe,
    }


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _write_walk_forward_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    stitched_metrics: dict[str, float],
    grid: str,
    objective: str,
    train_years: int,
    test_months: int,
    step_months: int,
    max_train_drawdown: float,
    selection_validation_months: int,
) -> tuple[Path, Path, Path, Path]:
    summary = pd.DataFrame(rows)
    candidates = pd.DataFrame(candidate_rows)
    summary_path = output_dir / "walk_forward_summary.csv"
    candidates_path = output_dir / "candidate_results.csv"
    audit_path = output_dir / "overfit_audit.csv"
    report_path = output_dir / "summary.md"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    candidates.to_csv(candidates_path, index=False, encoding="utf-8")
    audit_columns = [
        "window",
        "selected_candidate",
        "train_total_return",
        "test_total_return",
        "train_test_return_gap",
        "train_sharpe",
        "test_sharpe",
        "train_test_sharpe_gap",
        "train_max_drawdown",
        "test_max_drawdown",
        "selected_train_annual_return_std",
        "selected_train_return_concentration",
        "selected_train_monthly_win_rate",
        "selected_train_worst_month_return",
    ]
    audit = summary[[column for column in audit_columns if column in summary.columns]].copy()
    audit.to_csv(audit_path, index=False, encoding="utf-8")

    positive_oos_ratio = 0.0
    avg_return_gap = 0.0
    avg_sharpe_gap = 0.0
    worst_oos_return = 0.0
    selected_count = 0
    if not summary.empty:
        test_returns = pd.to_numeric(summary.get("test_total_return"), errors="coerce")
        positive_oos_ratio = float((test_returns > 0).mean())
        worst_oos_return = float(test_returns.min()) if len(test_returns.dropna()) else 0.0
        avg_return_gap = float(pd.to_numeric(summary.get("train_test_return_gap"), errors="coerce").mean())
        avg_sharpe_gap = float(pd.to_numeric(summary.get("train_test_sharpe_gap"), errors="coerce").mean())
        selected_count = int(summary["selected_candidate"].nunique()) if "selected_candidate" in summary else 0

    table = [
        "| window | train | test | selected | train_return | train_dd | train_sharpe | test_return | test_dd | test_sharpe |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            "| {window} | {train_start} to {train_end} | {test_start} to {test_end} | {selected} | "
            "{train_return} | {train_dd} | {train_sharpe:.3f} | {test_return} | {test_dd} | {test_sharpe:.3f} |".format(
                window=row["window"],
                train_start=_display_date(row["train_start"]),
                train_end=_display_date(row["train_end"]),
                test_start=_display_date(row["test_start"]),
                test_end=_display_date(row["test_end"]),
                selected=row["selected_candidate"],
                train_return=_pct(row.get("train_total_return")),
                train_dd=_pct(row.get("train_max_drawdown")),
                train_sharpe=float(row.get("train_sharpe", 0.0) or 0.0),
                test_return=_pct(row.get("test_total_return")),
                test_dd=_pct(row.get("test_max_drawdown")),
                test_sharpe=float(row.get("test_sharpe", 0.0) or 0.0),
            )
        )

    body = f"""# Rolling Training Summary

Output directory: `{output_dir}`

Each window selects parameters using only its training period, then runs the selected configuration on the next out-of-sample period.
This is research analysis only; it does not connect to brokers or provide investment advice.

## Aggregate Out-Of-Sample

| Metric | Value |
| --- | ---: |
| Stitched OOS total return | {_pct(stitched_metrics.get("stitched_total_return"))} |
| Stitched OOS max drawdown | {_pct(stitched_metrics.get("stitched_max_drawdown"))} |
| Stitched OOS Sharpe | {stitched_metrics.get("stitched_sharpe", 0.0):.3f} |
| Training objective | {objective} |
| Candidate grid | {grid} |
| Candidate runs | {len(candidate_rows)} |
| Train years | {train_years} |
| Test months | {test_months} |
| Step months | {step_months} |
| Training drawdown limit | {_pct(max_train_drawdown)} |
| Selection validation months | {selection_validation_months} |

## Overfit Audit

| Check | Value |
| --- | ---: |
| Positive OOS window ratio | {_pct(positive_oos_ratio)} |
| Worst OOS window return | {_pct(worst_oos_return)} |
| Average train minus OOS return gap | {_pct(avg_return_gap)} |
| Average train minus OOS Sharpe gap | {avg_sharpe_gap:.3f} |
| Distinct selected candidates | {selected_count} |

## Window Results

{chr(10).join(table)}

## Files

- `walk_forward_summary.csv`: selected parameter and train/test metrics by window.
- `candidate_results.csv`: all training candidates by window.
- `overfit_audit.csv`: selected candidate stability diagnostics and train/OOS gaps.
- `oos_equity_stitched.csv`: out-of-sample windows compounded in chronological order.
- Selected test runs each have full backtest reports under `outputs/backtests/`.

## Notes

- This run uses a {train_years}-year training window, a {test_months}-month test window, and a {step_months}-month step.
- `balanced` selection uses Sharpe, CAGR, return, and drawdown penalty.
- `robust` selection adds penalties for losing years, worst annual return, annual drawdown, weak bear-year behavior, and very high turnover.
- `stable` selection adds penalties for annual return instability, return concentration, deep monthly tail losses, and overly strong training-only statistics.
- `satellite` selection rewards higher CAGR and participation while still penalizing drawdown breaches, losing years, return concentration, deep monthly tail losses, and too-good-to-trust training statistics.
- `capital` selection emphasizes higher execution success probability and utilization with weights on win rate and average risk exposure, while still penalizing drawdowns and losses.
- `opportunity_growth` selection emphasizes high CAGR and absolute return while still penalizing breaches above the drawdown floor, poor execution, and weak risk utilization.
- `opportunity_q_full` selection emphasizes CAGR/absolute return with controlled tail-risk checks: it favors higher execution quality and participation, keeps a looser under-coverage penalty than conservative modes, and still punishes deep drawdowns and concentrated/unstable return patterns.
- Signals are computed with prior history for indicator warm-up, then trimmed to the train, validation, or out-of-sample window before backtesting. This avoids future data while preventing short windows from being consumed by factor warm-up.
- When selection validation months is greater than 0, the latest part of each training window is held out for candidate selection; `selection_train_*` records the fit segment and `validation_*` records the internal validation segment.
- Each out-of-sample window starts from cash.
"""
    report_path.write_text(body, encoding="utf-8")
    return summary_path, candidates_path, audit_path, report_path


def _write_walk_forward_outputs(
    wf_dir: Path,
    summary_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    test_results: list[BacktestResult],
    config: LabConfig,
    grid: str,
    objective: str,
    train_years: int,
    test_months: int,
    step_months: int,
    max_train_drawdown: float,
    selection_validation_months: int,
) -> tuple[Path, Path, Path, Path, Path, dict[str, float]]:
    stitched = _stitch_oos_equity(test_results, config.project.initial_cash)
    stitched_path = wf_dir / "oos_equity_stitched.csv"
    stitched.to_csv(stitched_path, index=False, encoding="utf-8")
    aggregate = _stitched_metrics(stitched, config.project.initial_cash)
    summary_path, candidates_path, audit_path, report_path = _write_walk_forward_report(
        wf_dir,
        summary_rows,
        candidate_rows,
        aggregate,
        grid,
        objective,
        train_years,
        test_months,
        step_months,
        max_train_drawdown,
        selection_validation_months,
    )
    return summary_path, candidates_path, audit_path, report_path, stitched_path, aggregate


def run_walk_forward(
    config: LabConfig,
    train_years: int = 3,
    test_months: int = 12,
    step_months: int = 12,
    grid: str = "compact",
    objective: str = "robust",
    max_train_drawdown: float = 0.35,
    output_dir: Path | None = None,
    run_id_prefix: str | None = None,
    allow_fetch: bool = False,
    skip_missing: bool = False,
    resume: bool = False,
    window_limit: int | None = None,
    selection_validation_months: int = 0,
) -> tuple[Path, pd.DataFrame]:
    if selection_validation_months and selection_validation_months < 0:
        raise ValueError("selection_validation_months must be non-negative.")
    if window_limit is not None and window_limit <= 0:
        raise ValueError("window_limit must be greater than zero.")
    full_histories = load_universe_history(config, allow_fetch=allow_fetch, skip_missing=skip_missing)
    if not full_histories:
        raise ValueError("No cached histories available for walk-forward training.")

    all_dates = pd.concat([frame["date"] for frame in full_histories.values()])
    start = parse_market_date(config.data.start_date)
    end = parse_market_date(config.data.end_date) if config.data.end_date else pd.Timestamp(all_dates.max())
    if start is None or end is None:
        raise ValueError("Walk-forward training requires concrete date bounds.")

    windows = generate_rolling_windows(_yyyymmdd(start), _yyyymmdd(end), train_years, test_months, step_months)
    if not windows:
        raise ValueError("No walk-forward windows generated. Shorten train_years or extend the data range.")
    if window_limit is not None:
        windows = windows[:window_limit]
    candidates = build_parameter_grid(config, grid=grid)
    candidate_groups = _group_candidates_by_signal(candidates)
    if objective not in {
        "balanced",
        "robust",
        "stable",
        "satellite",
        "capital",
        "opportunity_quality",
        "opportunity_growth",
        "opportunity_q_full",
    }:
        raise ValueError(
            "objective must be 'balanced', 'robust', 'stable', 'satellite', 'capital', 'opportunity_quality', 'opportunity_growth', or 'opportunity_q_full'."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = output_dir or (config.project_root / "outputs" / "walk_forward")
    wf_name = run_id_prefix or f"{stamp}_{config.project.name}"
    wf_dir = root / wf_name
    wf_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    test_results: list[BacktestResult] = []
    completed_windows: set[str] = set()
    if resume:
        summary_rows, candidate_rows, test_results = _load_walk_forward_checkpoint(wf_dir, windows, config)
        completed_windows = {str(row.get("window")) for row in summary_rows if row.get("window")}
    for window in windows:
        if window.name in completed_windows:
            continue
        validation_dates = _walk_forward_validation_dates(window, selection_validation_months)
        selection_train_start = validation_dates.train_start if validation_dates else window.train_start
        selection_train_end = validation_dates.train_end if validation_dates else window.train_end
        validation_start = validation_dates.validation_start if validation_dates else None
        validation_end = validation_dates.validation_end if validation_dates else None

        train_histories = _slice_histories(full_histories, selection_train_start, selection_train_end)
        validation_histories = (
            _slice_histories(full_histories, validation_start, validation_end)
            if validation_start is not None and validation_end is not None
            else {}
        )
        test_histories = _slice_histories(full_histories, window.test_start, window.test_end)
        if not train_histories or not test_histories or (validation_dates is not None and not validation_histories):
            continue

        train_results: list[
            tuple[
                float,
                ParameterCandidate,
                BacktestResult,
                dict[str, float],
                float | None,
                BacktestResult | None,
                dict[str, float],
                str | None,
            ]
        ] = []
        train_warmup_start = _signal_warmup_start(config, selection_train_start)
        train_signal_histories = _slice_histories(full_histories, train_warmup_start, selection_train_end)
        validation_signal_histories: dict[str, pd.DataFrame] = {}
        if validation_start is not None and validation_end is not None:
            validation_warmup_start = _signal_warmup_start(config, validation_start)
            validation_signal_histories = _slice_histories(full_histories, validation_warmup_start, validation_end)
        for candidate_group in candidate_groups:
            signal_config = _candidate_config(config, candidate_group[0], selection_train_start, selection_train_end)
            train_signal_frames = _trim_signal_frames(
                build_signal_frames(train_signal_histories, signal_config.strategy),
                selection_train_start,
                selection_train_end,
            )
            validation_signal_frames: dict[str, pd.DataFrame] = {}
            if validation_start is not None and validation_end is not None:
                validation_signal_config = _candidate_config(config, candidate_group[0], validation_start, validation_end)
                validation_signal_frames = _trim_signal_frames(
                    build_signal_frames(validation_signal_histories, validation_signal_config.strategy),
                    validation_start,
                    validation_end,
                )
            for candidate in candidate_group:
                train_config = _candidate_config(config, candidate, selection_train_start, selection_train_end)
                train_result = run_backtest(
                    train_config,
                    histories=train_histories,
                    signal_frames=train_signal_frames,
                    run_id=f"{stamp}_{window.name}_{candidate.name}_train",
                    write_outputs=False,
                )
                score, diagnostics = score_training_result(
                    train_result,
                    max_drawdown_limit=max_train_drawdown,
                    objective=objective,
                )
                validation_score: float | None = None
                validation_result: BacktestResult | None = None
                validation_diagnostics: dict[str, float] = {}
                selection_score = score
                if validation_start is not None and validation_end is not None:
                    validation_config = _candidate_config(config, candidate, validation_start, validation_end)
                    validation_result = run_backtest(
                        validation_config,
                        histories=validation_histories,
                        signal_frames=validation_signal_frames,
                        run_id=f"{stamp}_{window.name}_{candidate.name}_validation",
                        write_outputs=False,
                    )
                    validation_score, validation_diagnostics = score_training_result(
                        validation_result,
                        max_drawdown_limit=max_train_drawdown,
                        objective=objective,
                    )
                    selection_score = validation_score
                selection_gate_reason = _candidate_selection_gate_reason(
                    candidate,
                    validation_result.metrics if validation_result is not None else None,
                    grid=grid,
                )
                train_results.append(
                    (
                        selection_score,
                        candidate,
                        train_result,
                        diagnostics,
                        validation_score,
                        validation_result,
                        validation_diagnostics,
                        selection_gate_reason,
                    )
                )
                candidate_row = {
                    "window": window.name,
                    "candidate": candidate.name,
                    "score": selection_score,
                    "train_score": score,
                    "validation_score": validation_score,
                    "selection_validation_months": int(selection_validation_months),
                    "selection_train_start": selection_train_start,
                    "selection_train_end": selection_train_end,
                    "validation_start": validation_start,
                    "validation_end": validation_end,
                    "objective": objective,
                    "selection_gate_passed": selection_gate_reason is None,
                    "selection_gate_reason": selection_gate_reason,
                    "parameters": _candidate_params(candidate),
                    **{f"train_{name}": value for name, value in diagnostics.items()},
                    **_metrics_row("train", train_result.metrics),
                }
                if validation_result is not None:
                    candidate_row.update({f"validation_{name}": value for name, value in validation_diagnostics.items()})
                    candidate_row.update(_metrics_row("validation", validation_result.metrics))
                candidate_rows.append(candidate_row)

        eligible_train_results = [item for item in train_results if item[7] is None]
        selection_pool = eligible_train_results or train_results
        selection_pool.sort(key=lambda item: item[0], reverse=True)
        (
            best_score,
            best_candidate,
            best_train_result,
            best_diagnostics,
            best_validation_score,
            best_validation_result,
            best_validation_diagnostics,
            best_selection_gate_reason,
        ) = selection_pool[0]
        test_config = _candidate_config(config, best_candidate, window.test_start, window.test_end)
        test_run_id = f"walk_forward_{stamp}_{window.name}_{best_candidate.name}_test"
        test_warmup_start = _signal_warmup_start(config, window.test_start)
        test_signal_histories = _slice_histories(full_histories, test_warmup_start, window.test_end)
        test_signal_frames = _trim_signal_frames(
            build_signal_frames(test_signal_histories, test_config.strategy),
            window.test_start,
            window.test_end,
        )
        test_result = run_backtest(
            test_config,
            histories=test_histories,
            signal_frames=test_signal_frames,
            run_id=test_run_id,
            write_outputs=True,
        )
        write_report(test_result.run_dir)
        test_results.append(test_result)

        selected_config_path = wf_dir / f"{window.name}_selected_params.json"
        selected_config_path.write_text(_candidate_params(best_candidate), encoding="utf-8")
        summary_rows.append(
            {
                "window": window.name,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "selected_candidate": best_candidate.name,
                "selected_score": best_score,
                "selected_train_score": score_training_metrics(
                    best_train_result.metrics,
                    max_drawdown_limit=max_train_drawdown,
                    objective=objective,
                    diagnostics=best_diagnostics,
                ),
                "selected_validation_score": best_validation_score,
                "selected_selection_gate_reason": best_selection_gate_reason,
                "selection_validation_months": int(selection_validation_months),
                "selection_train_start": selection_train_start,
                "selection_train_end": selection_train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "selected_params_path": str(selected_config_path),
                "test_run_dir": str(test_result.run_dir),
                **{f"selected_train_{name}": value for name, value in best_diagnostics.items()},
                **{f"selected_validation_{name}": value for name, value in best_validation_diagnostics.items()},
                **_metrics_row("train", best_train_result.metrics),
                **_metrics_row("validation", best_validation_result.metrics if best_validation_result is not None else {}),
                **_metrics_row("test", test_result.metrics),
                "train_test_return_gap": (
                    float(best_train_result.metrics.get("total_return", 0.0) or 0.0)
                    - float(test_result.metrics.get("total_return", 0.0) or 0.0)
                ),
                "validation_test_return_gap": (
                    float(best_validation_result.metrics.get("total_return", 0.0) or 0.0)
                    - float(test_result.metrics.get("total_return", 0.0) or 0.0)
                    if best_validation_result is not None
                    else None
                ),
                "train_test_sharpe_gap": (
                    float(best_train_result.metrics.get("sharpe", 0.0) or 0.0)
                    - float(test_result.metrics.get("sharpe", 0.0) or 0.0)
                ),
                "validation_test_sharpe_gap": (
                    float(best_validation_result.metrics.get("sharpe", 0.0) or 0.0)
                    - float(test_result.metrics.get("sharpe", 0.0) or 0.0)
                    if best_validation_result is not None
                    else None
                ),
            }
        )
        _write_walk_forward_outputs(
            wf_dir,
            summary_rows,
            candidate_rows,
            test_results,
            config,
            grid,
            objective,
            train_years,
            test_months,
            step_months,
            max_train_drawdown,
            selection_validation_months,
        )

    if not summary_rows:
        raise ValueError("No walk-forward windows completed.")

    summary_path, candidates_path, audit_path, report_path, stitched_path, aggregate = _write_walk_forward_outputs(
        wf_dir,
        summary_rows,
        candidate_rows,
        test_results,
        config,
        grid,
        objective,
        train_years,
        test_months,
        step_months,
        max_train_drawdown,
        selection_validation_months,
    )
    summary = pd.read_csv(summary_path)
    summary.attrs["summary_csv"] = str(summary_path)
    summary.attrs["candidate_csv"] = str(candidates_path)
    summary.attrs["overfit_audit_csv"] = str(audit_path)
    summary.attrs["summary_md"] = str(report_path)
    summary.attrs["stitched_csv"] = str(stitched_path)
    summary.attrs.update(aggregate)
    return wf_dir, summary
