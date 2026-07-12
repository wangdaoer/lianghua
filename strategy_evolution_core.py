from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum


def _canonicalize(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("fingerprint mappings require string keys")
            normalized[key] = _canonicalize(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fingerprint values must be finite")
        return value
    raise TypeError(f"unsupported fingerprint value: {type(value).__name__}")


def fingerprint_payload(value: object) -> str:
    payload = json.dumps(
        _canonicalize(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generate_parameter_candidates(
    champion_parameters: Mapping[str, object],
    candidate_overrides: Sequence[Mapping[str, object]],
    max_candidates: int,
    seed: int,
) -> tuple[dict[str, object], ...]:
    if not isinstance(champion_parameters, Mapping):
        raise TypeError("champion_parameters must be a mapping")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 0:
        raise ValueError("max_candidates must be a non-negative integer")

    champion = copy.deepcopy(dict(champion_parameters))
    candidates: list[dict[str, object]] = []
    fingerprints: set[str] = set()
    for override in candidate_overrides:
        if not isinstance(override, Mapping) or not override:
            raise ValueError("each candidate override must be a non-empty mapping")
        candidate = copy.deepcopy(champion)
        candidate.update(copy.deepcopy(dict(override)))
        if candidate == champion:
            continue
        fingerprint = fingerprint_payload(candidate)
        if fingerprint in fingerprints:
            raise ValueError("conflicting duplicate candidate fingerprints")
        fingerprints.add(fingerprint)
        candidates.append(candidate)

    random.Random(seed).shuffle(candidates)
    return tuple(candidates[:max_candidates])


@dataclass(frozen=True)
class FoldMetrics:
    fold_id: str
    total_return: float
    max_drawdown: float
    sharpe: float
    filled_trades: int
    average_turnover: float
    pnl_concentration: float


@dataclass(frozen=True)
class PromotionPolicy:
    min_folds: int = 3
    min_filled_trades_per_fold: int = 5
    min_positive_fold_ratio: float = 2.0 / 3.0
    min_mean_return_improvement: float = 0.01
    max_drawdown_floor: float = -0.40
    max_drawdown_worsening: float = 0.05
    max_turnover_ratio: float = 1.50
    max_pnl_concentration: float = 0.50


@dataclass(frozen=True)
class PromotionDecision:
    status: str
    failed_gates: tuple[str, ...]
    gates: dict[str, dict[str, object]]


def _fold_map(folds: Sequence[FoldMetrics]) -> tuple[dict[str, FoldMetrics] | None, str | None]:
    try:
        values = tuple(folds)
    except TypeError:
        return None, "invalid_folds"
    result: dict[str, FoldMetrics] = {}
    for fold in values:
        if not isinstance(fold, FoldMetrics):
            return None, "invalid_folds"
        if not isinstance(fold.fold_id, str) or not fold.fold_id:
            return None, "invalid_fold_id"
        if fold.fold_id in result:
            return None, "duplicate_fold_ids"
        result[fold.fold_id] = fold
    return result, None


def _finite_metrics(fold: FoldMetrics) -> bool:
    try:
        return all(
            math.isfinite(float(value))
            for value in (
                fold.total_return,
                fold.max_drawdown,
                fold.sharpe,
                fold.filled_trades,
                fold.average_turnover,
                fold.pnl_concentration,
            )
        )
    except (TypeError, ValueError):
        return False


def _gate(passed: bool, value: object, threshold: object) -> dict[str, object]:
    return {"passed": passed, "value": value, "threshold": threshold}


def evaluate_candidate(
    challenger: Sequence[FoldMetrics],
    champion: Sequence[FoldMetrics],
    policy: PromotionPolicy,
) -> PromotionDecision:
    challenger_by_id, challenger_error = _fold_map(challenger)
    champion_by_id, champion_error = _fold_map(champion)
    if challenger_error or champion_error:
        error = challenger_error or champion_error or "invalid_folds"
        return PromotionDecision("rejected", (error,), {"fold_alignment": {"passed": False, "error": error}})
    assert challenger_by_id is not None and champion_by_id is not None
    if set(challenger_by_id) != set(champion_by_id):
        return PromotionDecision(
            "rejected",
            ("fold_alignment",),
            {
                "fold_alignment": {
                    "passed": False,
                    "challenger_fold_ids": tuple(sorted(challenger_by_id)),
                    "champion_fold_ids": tuple(sorted(champion_by_id)),
                }
            },
        )

    fold_ids = tuple(sorted(challenger_by_id))
    challenger_folds = tuple(challenger_by_id[fold_id] for fold_id in fold_ids)
    champion_folds = tuple(champion_by_id[fold_id] for fold_id in fold_ids)
    all_finite = all(_finite_metrics(fold) for fold in (*challenger_folds, *champion_folds))
    fold_count = len(fold_ids)
    min_trades = min((fold.filled_trades for fold in challenger_folds), default=0)
    positive_ratio = (
        sum(fold.total_return > 0 for fold in challenger_folds) / fold_count
        if fold_count
        else 0.0
    )
    mean_return_improvement = (
        sum(fold.total_return for fold in challenger_folds) / fold_count
        - sum(fold.total_return for fold in champion_folds) / fold_count
        if fold_count
        else 0.0
    )
    worst_drawdown = min((fold.max_drawdown for fold in challenger_folds), default=math.inf)
    worst_drawdown_worsening = max(
        (max(champion_fold.max_drawdown - challenger_fold.max_drawdown, 0.0)
         for challenger_fold, champion_fold in zip(challenger_folds, champion_folds)),
        default=math.inf,
    )
    champion_turnover = sum(fold.average_turnover for fold in champion_folds) / fold_count if fold_count else 0.0
    challenger_turnover = sum(fold.average_turnover for fold in challenger_folds) / fold_count if fold_count else 0.0
    if abs(champion_turnover) <= 1e-12:
        turnover_ratio = 1.0 if abs(challenger_turnover) <= 1e-12 else math.inf
    else:
        turnover_ratio = challenger_turnover / champion_turnover
    max_concentration = max((fold.pnl_concentration for fold in challenger_folds), default=math.inf)

    gates = {
        "min_folds": _gate(fold_count >= policy.min_folds, fold_count, policy.min_folds),
        "min_filled_trades": _gate(
            min_trades >= policy.min_filled_trades_per_fold,
            min_trades,
            policy.min_filled_trades_per_fold,
        ),
        "positive_fold_ratio": _gate(
            positive_ratio >= policy.min_positive_fold_ratio,
            positive_ratio,
            policy.min_positive_fold_ratio,
        ),
        "mean_return_improvement": _gate(
            mean_return_improvement >= policy.min_mean_return_improvement,
            mean_return_improvement,
            policy.min_mean_return_improvement,
        ),
        "max_drawdown": _gate(
            worst_drawdown >= policy.max_drawdown_floor,
            worst_drawdown,
            policy.max_drawdown_floor,
        ),
        "drawdown_worsening": _gate(
            worst_drawdown_worsening <= policy.max_drawdown_worsening,
            worst_drawdown_worsening,
            policy.max_drawdown_worsening,
        ),
        "turnover_ratio": _gate(
            turnover_ratio <= policy.max_turnover_ratio,
            turnover_ratio,
            policy.max_turnover_ratio,
        ),
        "pnl_concentration": _gate(
            max_concentration <= policy.max_pnl_concentration,
            max_concentration,
            policy.max_pnl_concentration,
        ),
    }
    failed = [name for name, result in gates.items() if not result["passed"]]
    if not all_finite:
        gates["finite_metrics"] = {"passed": False, "value": False, "threshold": True}
        failed.append("finite_metrics")
    status = "insufficient_evidence" if not all_finite or "min_folds" in failed or "min_filled_trades" in failed else (
        "rejected" if failed else "eligible_for_shadow"
    )
    return PromotionDecision(status, tuple(failed), gates)


def _copy_parameters(parameters: Mapping[str, object], name: str) -> dict[str, object]:
    if not isinstance(parameters, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return copy.deepcopy(dict(parameters))


def _state_time(now: object | None) -> str:
    if now is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(now, datetime):
        return now.isoformat()
    if isinstance(now, str) and now:
        return now
    raise ValueError("now must be a non-empty string or datetime")


@dataclass(frozen=True)
class EvolutionState:
    schema_version: int
    champion_version: str
    champion_parameters: dict[str, object]
    shadow_status: str
    shadow_version: str | None = None
    shadow_parameters: dict[str, object] | None = None
    shadow_experiment_id: str | None = None
    shadow_run_id: str | None = None
    shadow_data_fingerprint: str | None = None
    previous_champion_version: str | None = None
    previous_champion_parameters: dict[str, object] | None = None
    last_completed_run_id: str | None = None
    last_data_fingerprint: str | None = None
    blocked_reason: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "champion_parameters", copy.deepcopy(dict(self.champion_parameters)))
        if self.shadow_parameters is not None:
            object.__setattr__(self, "shadow_parameters", copy.deepcopy(dict(self.shadow_parameters)))
        if self.previous_champion_parameters is not None:
            object.__setattr__(self, "previous_champion_parameters", copy.deepcopy(dict(self.previous_champion_parameters)))

    @classmethod
    def initial(
        cls,
        champion_version: str,
        champion_parameters: Mapping[str, object],
        *,
        now: object | None = None,
    ) -> EvolutionState:
        return cls(
            schema_version=1,
            champion_version=champion_version,
            champion_parameters=_copy_parameters(champion_parameters, "champion_parameters"),
            shadow_status="none",
            updated_at=_state_time(now),
        )

    @property
    def experiment_id(self) -> str | None:
        return self.shadow_experiment_id

    @property
    def run_id(self) -> str | None:
        return self.shadow_run_id

    @property
    def data_fingerprint(self) -> str | None:
        return self.shadow_data_fingerprint

    @property
    def last_run_id(self) -> str | None:
        return self.last_completed_run_id


def promote_to_shadow(
    state: EvolutionState,
    challenger_version: str,
    challenger_parameters: Mapping[str, object],
    experiment_id: str,
    run_id: str,
    data_fingerprint: str,
    now: object | None = None,
) -> EvolutionState:
    return EvolutionState(
        schema_version=state.schema_version,
        champion_version=state.champion_version,
        champion_parameters=state.champion_parameters,
        shadow_status="shadow",
        shadow_version=challenger_version,
        shadow_parameters=_copy_parameters(challenger_parameters, "challenger_parameters"),
        shadow_experiment_id=experiment_id,
        shadow_run_id=run_id,
        shadow_data_fingerprint=data_fingerprint,
        previous_champion_version=state.champion_version,
        previous_champion_parameters=state.champion_parameters,
        last_completed_run_id=run_id,
        last_data_fingerprint=data_fingerprint,
        blocked_reason=None,
        updated_at=_state_time(now),
    )


def rollback_shadow(
    state: EvolutionState,
    reason: str,
    now: object | None = None,
) -> EvolutionState:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    champion_version = state.previous_champion_version or state.champion_version
    champion_parameters = (
        state.previous_champion_parameters
        if state.previous_champion_parameters is not None
        else state.champion_parameters
    )
    return EvolutionState(
        schema_version=state.schema_version,
        champion_version=champion_version,
        champion_parameters=champion_parameters,
        shadow_status="rolled_back",
        shadow_version=state.shadow_version,
        shadow_parameters=state.shadow_parameters,
        shadow_experiment_id=state.shadow_experiment_id,
        shadow_run_id=state.shadow_run_id,
        shadow_data_fingerprint=state.shadow_data_fingerprint,
        previous_champion_version=None,
        previous_champion_parameters=None,
        last_completed_run_id=state.last_completed_run_id,
        last_data_fingerprint=state.last_data_fingerprint,
        blocked_reason=reason,
        updated_at=_state_time(now),
    )
