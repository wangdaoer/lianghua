from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml

from run_backtest import max_drawdown, sharpe_like


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


@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool
    reasons: tuple[str, ...]
    turnover_ratio: float
    robust_score: float


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str
    promotion: PromotionDecision
    metrics: dict[str, float]


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


def calculate_segment_metrics(
    equity: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    rolling_window_days: int,
) -> dict[str, float]:
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))].sort_values("date")
    if frame.empty:
        raise ValueError(f"No equity rows between {start} and {end}")
    returns = pd.to_numeric(frame["gross_return"], errors="coerce") - pd.to_numeric(
        frame["cost"], errors="coerce"
    )
    if returns.isna().any() or (returns <= -1.0).any():
        raise ValueError("Segment contains invalid net returns")
    nav = (1.0 + returns).cumprod()
    total_return = float(nav.iloc[-1] - 1.0)
    annualized = float((1.0 + total_return) ** (252.0 / len(nav)) - 1.0)
    rolling = nav / nav.shift(int(rolling_window_days)) - 1.0
    completed = rolling.dropna()
    negative_rate = float(completed.lt(0.0).mean()) if not completed.empty else 0.0
    worst_rolling = float(completed.min()) if not completed.empty else total_return
    metrics = {
        "total_return": total_return,
        "annualized_return": annualized,
        "max_drawdown": float(max_drawdown(nav)),
        "sharpe_like": float(sharpe_like(returns)),
        "avg_turnover": float(pd.to_numeric(frame["turnover"], errors="coerce").mean()),
        "avg_gross_exposure": float(pd.to_numeric(frame["gross_exposure"], errors="coerce").mean()),
        "trade_days": int(len(frame)),
        "negative_window_rate": negative_rate,
        "worst_rolling_return": worst_rolling,
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("Segment metrics contain non-finite values")
    return metrics


def _turnover_ratio(candidate_turnover: float, incumbent_turnover: float) -> float:
    if abs(incumbent_turnover) <= 1e-12:
        return 1.0 if abs(candidate_turnover) <= 1e-12 else math.inf
    return candidate_turnover / incumbent_turnover


def _robust_score(metrics: Mapping[str, float], turnover_ratio: float) -> float:
    return float(
        metrics["annualized_return"]
        + 0.15 * metrics["sharpe_like"]
        + 0.50 * metrics["max_drawdown"]
        - 0.10 * metrics["negative_window_rate"]
        - 0.05 * max(turnover_ratio - 1.0, 0.0)
    )


def evaluate_promotion(
    candidate_metrics: Mapping[str, float],
    incumbent_metrics: Mapping[str, float],
    rules: SelectionRules,
) -> PromotionDecision:
    required = (
        "annualized_return", "max_drawdown", "sharpe_like", "avg_turnover",
        "trade_days", "negative_window_rate",
    )
    reasons: list[str] = []
    if not all(math.isfinite(float(candidate_metrics[key])) for key in required):
        reasons.append("关键指标存在非有限值")
    turnover_ratio = _turnover_ratio(
        float(candidate_metrics["avg_turnover"]), float(incumbent_metrics["avg_turnover"])
    )
    if int(candidate_metrics["trade_days"]) < rules.min_validation_days:
        reasons.append("验证期有效交易日不足")
    if float(candidate_metrics["max_drawdown"]) < rules.max_drawdown_floor:
        reasons.append("最大回撤超过限制")
    if float(candidate_metrics["annualized_return"]) - float(incumbent_metrics["annualized_return"]) < rules.min_annualized_return_delta - 1e-12:
        reasons.append("年化收益提升不足")
    if float(candidate_metrics["sharpe_like"]) - float(incumbent_metrics["sharpe_like"]) < rules.min_sharpe_delta - 1e-12:
        reasons.append("Sharpe 恶化超过限制")
    if turnover_ratio > rules.max_turnover_ratio + 1e-12:
        reasons.append("换手率放大超过限制")
    if float(candidate_metrics["negative_window_rate"]) > rules.max_negative_window_rate + 1e-12:
        reasons.append("负滚动窗口占比超过限制")
    return PromotionDecision(
        eligible=not reasons,
        reasons=tuple(reasons),
        turnover_ratio=float(turnover_ratio),
        robust_score=_robust_score(candidate_metrics, turnover_ratio),
    )


def choose_group_winner(
    incumbent_id: str,
    incumbent_metrics: Mapping[str, float],
    candidates: tuple[tuple[str, Mapping[str, float]], ...],
    rules: SelectionRules,
) -> tuple[str, tuple[CandidateDecision, ...]]:
    decisions = tuple(
        CandidateDecision(candidate_id, evaluate_promotion(metrics, incumbent_metrics, rules), dict(metrics))
        for candidate_id, metrics in candidates
    )
    eligible = [decision for decision in decisions if decision.promotion.eligible]
    if not eligible:
        return incumbent_id, decisions
    eligible.sort(
        key=lambda item: (
            item.metrics["annualized_return"], item.promotion.robust_score,
            item.metrics["sharpe_like"], item.metrics["max_drawdown"],
            -item.metrics["avg_turnover"],
        ),
        reverse=True,
    )
    return eligible[0].candidate_id, decisions


def assess_test_result(
    baseline_metrics: Mapping[str, float],
    champion_metrics: Mapping[str, float],
    rules: SelectionRules,
) -> tuple[str, str]:
    if min(int(baseline_metrics["trade_days"]), int(champion_metrics["trade_days"])) < rules.min_test_days:
        return "test_warning", "测试期有效交易日不足，暂不判断晋级"
    failures: list[str] = []
    if float(champion_metrics["total_return"]) < float(baseline_metrics["total_return"]):
        failures.append("测试期收益低于基准")
    if float(champion_metrics["max_drawdown"]) < rules.max_drawdown_floor:
        failures.append("测试期最大回撤超过限制")
    if float(champion_metrics["sharpe_like"]) - float(baseline_metrics["sharpe_like"]) < -0.10:
        failures.append("测试期 Sharpe 恶化超过 0.10")
    if failures:
        return "rollback_recommended", "；".join(failures)
    return "ready_for_manual_review", "测试期收益、回撤和 Sharpe 门槛通过，等待人工确认"
