from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml


DEFAULT_STRATEGY_PARAMS: dict[str, object] = {
    "train_days": 252,
    "retrain_frequency": 20,
    "top_n": 8,
    "rebalance_frequency": 5,
    "max_position_weight": 0.08,
    "leverage": 0.60,
    "min_score": None,
    "commission_bps": 1.0,
    "impact_bps": 0.7,
    "max_buy_open_gap": 0.05,
    "limit_buffer": 0.995,
    "initial_capital": 1_000_000.0,
    "max_abs_daily_return": 0.22,
    "min_close": 2.0,
    "min_avg_amount_20d": 30_000_000.0,
    "min_pullback_5d": 0.03,
    "max_pullback_5d": 0.18,
    "min_prior_return_20": 0.08,
    "min_prior_return_60": 0.18,
    "min_return_20d": -0.12,
    "min_return_60d": 0.0,
    "min_distance_ma60": -0.10,
    "max_intraday_return": 0.05,
    "market_ma_window": 120,
    "market_risk_off_drawdown_20d": -0.08,
    "market_below_ma_exposure": 0.60,
    "market_crash_exposure": 0.0,
    "basket_guard_return_20d_min": None,
    "basket_guard_distance_ma60_min": None,
    "basket_guard_scale": 1.0,
    "rebound_exit_return": None,
    "rebound_exit_scale": 0.0,
    "rebound_exit_market_exposure_max": None,
    "rebound_exit_market_exposure_min": None,
}

ALLOWED_OVERRIDE_PARAMS = frozenset({
    "top_n", "rebalance_frequency", "max_position_weight", "leverage", "min_score",
    "min_close", "min_avg_amount_20d", "min_pullback_5d", "max_pullback_5d",
    "min_prior_return_20", "min_prior_return_60", "min_return_20d",
    "min_return_60d", "min_distance_ma60", "max_intraday_return",
    "basket_guard_return_20d_min", "basket_guard_distance_ma60_min",
    "basket_guard_scale", "rebound_exit_return", "rebound_exit_scale",
    "rebound_exit_market_exposure_max", "rebound_exit_market_exposure_min",
})

SELECTION_KEYS = frozenset({
    "min_validation_days",
    "min_test_days",
    "max_drawdown_floor",
    "min_annualized_return_delta",
    "min_sharpe_delta",
    "max_turnover_ratio",
    "rolling_window_days",
    "max_negative_window_rate",
})


@dataclass(frozen=True)
class EvolutionPeriods:
    research_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp | None


@dataclass(frozen=True)
class SelectionRules:
    min_validation_days: int
    min_test_days: int
    max_drawdown_floor: float
    min_annualized_return_delta: float
    min_sharpe_delta: float
    max_turnover_ratio: float
    rolling_window_days: int
    max_negative_window_rate: float


@dataclass(frozen=True)
class SearchCandidate:
    candidate_id: str
    overrides: dict[str, object]


@dataclass(frozen=True)
class SearchGroup:
    group_id: str
    hypothesis_cn: str
    candidates: tuple[SearchCandidate, ...]


@dataclass(frozen=True)
class EvolutionConfig:
    strategy: str
    periods: EvolutionPeriods
    baseline: dict[str, object]
    search_groups: tuple[SearchGroup, ...]
    selection: SelectionRules


def _timestamp(value: object, name: str, allow_none: bool = False) -> pd.Timestamp | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"Missing required date: {name}")
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid date for {name}: {value!r}") from exc


def _validate_params(params: Mapping[str, object]) -> None:
    unknown = set(params) - set(DEFAULT_STRATEGY_PARAMS)
    if unknown:
        raise ValueError(f"Unknown strategy parameters: {sorted(unknown)}")
    for key in ("train_days", "retrain_frequency", "top_n", "rebalance_frequency"):
        if int(params[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("leverage", "max_position_weight"):
        value = float(params[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive number")
    if float(params["leverage"]) > 1.5:
        raise ValueError("leverage must be <= 1.5")
    if float(params["max_position_weight"]) > 1.0:
        raise ValueError("max_position_weight must be <= 1.0")
    if float(params["min_pullback_5d"]) > float(params["max_pullback_5d"]):
        raise ValueError("min_pullback_5d must be <= max_pullback_5d")


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def parse_evolution_config(raw: Mapping[str, object]) -> EvolutionConfig:
    if raw.get("strategy") != "strong_pullback_satellite":
        raise ValueError("strategy must be strong_pullback_satellite")
    period_raw = dict(raw.get("periods") or {})
    periods = EvolutionPeriods(
        research_start=_timestamp(period_raw.get("research_start"), "research_start"),
        train_end=_timestamp(period_raw.get("train_end"), "train_end"),
        validation_start=_timestamp(period_raw.get("validation_start"), "validation_start"),
        validation_end=_timestamp(period_raw.get("validation_end"), "validation_end"),
        test_start=_timestamp(period_raw.get("test_start"), "test_start"),
        test_end=_timestamp(period_raw.get("test_end"), "test_end", allow_none=True),
    )
    if not (
        periods.research_start <= periods.train_end
        and periods.train_end < periods.validation_start
        and periods.validation_start <= periods.validation_end
        and periods.validation_end < periods.test_start
        and (periods.test_end is None or periods.test_start <= periods.test_end)
    ):
        raise ValueError("Periods must satisfy train_end < validation_start <= validation_end < test_start")

    baseline = {**DEFAULT_STRATEGY_PARAMS, **dict(raw.get("baseline") or {})}
    _validate_params(baseline)
    group_ids: set[str] = set()
    candidate_ids: set[str] = set()
    groups: list[SearchGroup] = []
    for group_raw in list(raw.get("search_groups") or []):
        if not isinstance(group_raw, Mapping):
            raise ValueError("search_groups entries must be mappings")
        group_id = _identifier(group_raw.get("id"), "Group id")
        if group_id in group_ids:
            raise ValueError(f"Duplicate or empty group id: {group_id!r}")
        group_ids.add(group_id)
        candidates: list[SearchCandidate] = []
        for candidate_raw in list(group_raw.get("candidates") or []):
            if not isinstance(candidate_raw, Mapping):
                raise ValueError("candidates entries must be mappings")
            candidate_id = _identifier(candidate_raw.get("id"), "Candidate id")
            if candidate_id in candidate_ids:
                raise ValueError(f"Duplicate candidate id: {candidate_id!r}")
            candidate_ids.add(candidate_id)
            overrides_raw = candidate_raw.get("overrides") or {}
            if not isinstance(overrides_raw, Mapping):
                raise ValueError("candidate overrides must be a mapping")
            overrides = dict(overrides_raw)
            unknown = set(overrides) - ALLOWED_OVERRIDE_PARAMS
            if unknown:
                raise ValueError(f"Unknown strategy parameters: {sorted(unknown)}")
            _validate_params({**baseline, **overrides})
            candidates.append(SearchCandidate(candidate_id, overrides))
        if not candidates:
            raise ValueError(f"Search group {group_id!r} has no candidates")
        groups.append(SearchGroup(group_id, str(group_raw.get("hypothesis_cn", "")).strip(), tuple(candidates)))

    selection_value = raw.get("selection")
    if not isinstance(selection_value, Mapping):
        raise ValueError("selection must be a mapping")
    selection_raw = dict(selection_value)
    unknown_selection = set(selection_raw) - SELECTION_KEYS
    if unknown_selection:
        raise ValueError(f"Unknown selection keys: {sorted(unknown_selection)}")
    missing_selection = SELECTION_KEYS - set(selection_raw)
    if missing_selection:
        raise ValueError(f"Missing selection keys: {sorted(missing_selection)}")
    selection = SelectionRules(**selection_raw)
    if selection.min_validation_days <= 0 or selection.min_test_days <= 0:
        raise ValueError("minimum period days must be positive")
    if not -1.0 <= selection.max_drawdown_floor <= 0.0:
        raise ValueError("max_drawdown_floor must be between -1 and 0")
    if selection.max_turnover_ratio < 1.0:
        raise ValueError("max_turnover_ratio must be >= 1")
    if not 0.0 <= selection.max_negative_window_rate <= 1.0:
        raise ValueError("max_negative_window_rate must be between 0 and 1")
    return EvolutionConfig("strong_pullback_satellite", periods, baseline, tuple(groups), selection)


def load_evolution_config(path: Path) -> EvolutionConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Evolution config must be a YAML mapping")
    return parse_evolution_config(raw)


def build_group_candidates(
    incumbent: Mapping[str, object], group: SearchGroup
) -> tuple[tuple[str, dict[str, object]], ...]:
    return tuple(
        (candidate.candidate_id, {**deepcopy(dict(incumbent)), **deepcopy(candidate.overrides)})
        for candidate in group.candidates
    )
