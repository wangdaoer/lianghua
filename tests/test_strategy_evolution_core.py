import json

import pytest

from strategy_evolution_core import (
    FoldMetrics,
    PromotionPolicy,
    EvolutionState,
    evaluate_candidate,
    fingerprint_payload,
    generate_parameter_candidates,
    promote_to_shadow,
    rollback_shadow,
)


def _fold(fold_id: str, total_return: float, *, trades: int = 20) -> FoldMetrics:
    return FoldMetrics(
        fold_id=fold_id,
        total_return=total_return,
        max_drawdown=-0.12,
        sharpe=1.1,
        filled_trades=trades,
        average_turnover=0.20,
        pnl_concentration=0.30,
    )


def test_fingerprint_is_canonical_and_sensitive():
    assert fingerprint_payload({"a": 1, "b": [2, 3]}) == fingerprint_payload({"b": [2, 3], "a": 1})
    assert fingerprint_payload({"a": 1}) != fingerprint_payload({"a": 2})


def test_candidate_passes_all_fold_gates():
    champion = tuple(_fold(f"f{i}", 0.04) for i in range(3))
    challenger = tuple(_fold(f"f{i}", 0.08) for i in range(3))
    decision = evaluate_candidate(challenger, champion, PromotionPolicy())
    assert decision.status == "eligible_for_shadow"
    assert decision.failed_gates == ()


def test_candidate_with_too_few_trades_is_insufficient_evidence():
    champion = tuple(_fold(f"f{i}", 0.04) for i in range(3))
    challenger = tuple(_fold(f"f{i}", 0.08, trades=1) for i in range(3))
    decision = evaluate_candidate(challenger, champion, PromotionPolicy(min_filled_trades_per_fold=5))
    assert decision.status == "insufficient_evidence"
    assert "min_filled_trades" in decision.failed_gates


def test_candidate_generation_is_seeded_and_does_not_mutate_champion():
    champion = {"leverage": 0.60, "top_n": 8}
    overrides = (
        {"leverage": 0.75},
        {"leverage": 0.90},
        {"top_n": 6},
        {"top_n": 10},
    )
    first = generate_parameter_candidates(champion, overrides, max_candidates=4, seed=20260712)
    second = generate_parameter_candidates(champion, overrides, max_candidates=4, seed=20260712)
    assert first == second
    assert len(first) == 4
    assert champion == {"leverage": 0.60, "top_n": 8}
    assert all(candidate != champion for candidate in first)


def test_non_finite_metric_cannot_promote():
    challenger = (_fold("f1", float("nan")), _fold("f2", 0.1), _fold("f3", 0.1))
    champion = tuple(_fold(f"f{i}", 0.04) for i in range(1, 4))
    assert evaluate_candidate(challenger, champion, PromotionPolicy()).status == "insufficient_evidence"


def test_shadow_promotion_and_rollback_restore_previous_champion():
    initial = EvolutionState.initial("v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00")
    shadow = promote_to_shadow(
        initial,
        challenger_version="v2",
        challenger_parameters={"leverage": 0.75},
        experiment_id="exp-1",
        run_id="run-1",
        data_fingerprint="data-1",
        now="2026-07-12T01:00:00+00:00",
    )
    restored = rollback_shadow(shadow, reason="drawdown", now="2026-07-12T02:00:00+00:00")
    assert shadow.shadow_status == "shadow"
    assert restored.champion_version == "v1"
    assert restored.shadow_status == "rolled_back"
    assert restored.blocked_reason == "drawdown"


def test_candidate_with_too_few_folds_is_insufficient_evidence():
    folds = tuple(_fold(f"f{i}", 0.08) for i in range(2))
    champion = tuple(_fold(f"f{i}", 0.04) for i in range(2))
    decision = evaluate_candidate(folds, champion, PromotionPolicy(min_folds=3))
    assert decision.status == "insufficient_evidence"
    assert "min_folds" in decision.failed_gates


def test_mismatched_or_duplicate_folds_are_rejected():
    champion = tuple(_fold(f"f{i}", 0.04) for i in range(3))
    mismatch = tuple(_fold(f"g{i}", 0.08) for i in range(3))
    duplicate = (_fold("f0", 0.08), _fold("f0", 0.08), _fold("f2", 0.08))
    assert evaluate_candidate(mismatch, champion, PromotionPolicy()).status == "rejected"
    assert evaluate_candidate(duplicate, champion, PromotionPolicy()).status == "rejected"


def test_candidate_reports_all_failed_hard_gates():
    champion = tuple(_fold(f"f{i}", 0.10, trades=20) for i in range(3))
    challenger = tuple(
        FoldMetrics(
            fold_id=f"f{i}",
            total_return=-0.10,
            max_drawdown=-0.60,
            sharpe=-1.0,
            filled_trades=1,
            average_turnover=0.40,
            pnl_concentration=0.80,
        )
        for i in range(3)
    )
    decision = evaluate_candidate(challenger, champion, PromotionPolicy())
    assert decision.status == "insufficient_evidence"
    assert {
        "min_filled_trades",
        "positive_fold_ratio",
        "mean_return_improvement",
        "max_drawdown",
        "drawdown_worsening",
        "turnover_ratio",
        "pnl_concentration",
    } <= set(decision.failed_gates)


def test_candidate_overrides_must_be_unique_non_empty_and_independent():
    champion = {"risk": {"leverage": 0.6}}
    with pytest.raises(ValueError, match="non-empty mapping"):
        generate_parameter_candidates(champion, ({},), max_candidates=2, seed=1)
    with pytest.raises(ValueError, match="duplicate candidate fingerprints"):
        generate_parameter_candidates(champion, ({"risk": {"leverage": 0.7}}, {"risk": {"leverage": 0.7}}), max_candidates=2, seed=1)
    candidates = generate_parameter_candidates(
        champion,
        ({"risk": {"leverage": 0.7}}, {"risk": {"leverage": 0.8}}),
        max_candidates=1,
        seed=1,
    )
    candidates[0]["risk"]["leverage"] = 0.1
    assert champion == {"risk": {"leverage": 0.6}}


def test_missing_metric_returns_full_insufficient_evidence_audit():
    challenger = (
        FoldMetrics(
            fold_id="f1",
            total_return=None,
            max_drawdown=-0.12,
            sharpe=1.1,
            filled_trades=20,
            average_turnover=0.20,
            pnl_concentration=0.30,
        ),
        _fold("f2", 0.08),
        _fold("f3", 0.08),
    )
    champion = tuple(_fold(f"f{i}", 0.04) for i in range(1, 4))
    decision = evaluate_candidate(challenger, champion, PromotionPolicy())
    assert decision.status == "insufficient_evidence"
    assert "finite_metrics" in decision.failed_gates
    assert {
        "min_folds",
        "min_filled_trades",
        "positive_fold_ratio",
        "mean_return_improvement",
        "max_drawdown",
        "drawdown_worsening",
        "turnover_ratio",
        "pnl_concentration",
        "finite_metrics",
    } <= set(decision.gates)


def test_non_numeric_metric_returns_insufficient_evidence_without_arithmetic_error():
    challenger = (
        FoldMetrics(
            fold_id="f1",
            total_return=0.08,
            max_drawdown=-0.12,
            sharpe=1.1,
            filled_trades=20,
            average_turnover="not-a-number",
            pnl_concentration=0.30,
        ),
        _fold("f2", 0.08),
        _fold("f3", 0.08),
    )
    champion = tuple(_fold(f"f{i}", 0.04) for i in range(1, 4))
    decision = evaluate_candidate(challenger, champion, PromotionPolicy())
    assert decision.status == "insufficient_evidence"
    assert "finite_metrics" in decision.failed_gates


def test_parseable_numeric_string_metric_returns_insufficient_evidence():
    challenger = (
        FoldMetrics(
            fold_id="f1",
            total_return=0.08,
            max_drawdown=-0.12,
            sharpe=1.1,
            filled_trades=20,
            average_turnover="0.20",
            pnl_concentration=0.30,
        ),
        _fold("f2", 0.08),
        _fold("f3", 0.08),
    )
    champion = tuple(_fold(f"f{i}", 0.04) for i in range(1, 4))
    decision = evaluate_candidate(challenger, champion, PromotionPolicy())
    assert decision.status == "insufficient_evidence"
    assert "finite_metrics" in decision.failed_gates


def test_evolution_state_mappings_are_immutable_and_serializable():
    initial = EvolutionState.initial(
        "v1",
        {"risk": {"leverage": 0.6}, "levels": [1, 2]},
        now="2026-07-12T00:00:00+00:00",
    )
    shadow = promote_to_shadow(
        initial,
        challenger_version="v2",
        challenger_parameters={"risk": {"leverage": 0.75}},
        experiment_id="exp-1",
        run_id="run-1",
        data_fingerprint="data-1",
        now="2026-07-12T01:00:00+00:00",
    )

    with pytest.raises(TypeError):
        initial.champion_parameters["risk"] = {"leverage": 0.1}
    with pytest.raises(TypeError):
        initial.champion_parameters["risk"]["leverage"] = 0.1
    with pytest.raises(TypeError):
        initial.champion_parameters["levels"].append(3)
    with pytest.raises(TypeError):
        shadow.shadow_parameters["risk"]["leverage"] = 0.1
    with pytest.raises(TypeError):
        shadow.previous_champion_parameters["risk"]["leverage"] = 0.1

    assert json.dumps(initial.champion_parameters, sort_keys=True) == json.dumps(
        {"risk": {"leverage": 0.6}, "levels": [1, 2]}, sort_keys=True
    )
    assert fingerprint_payload(initial.champion_parameters) == fingerprint_payload(
        {"risk": {"leverage": 0.6}, "levels": [1, 2]}
    )
