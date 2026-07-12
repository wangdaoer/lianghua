import json
import multiprocessing
from dataclasses import replace
from pathlib import Path

import pytest

from strategy_evolution_core import (
    EvolutionTransitionJournal,
    EvolutionStateError,
    FoldMetrics,
    PromotionPolicy,
    EvolutionState,
    StateWriteExpectation,
    commit_evolution_state_transition,
    evaluate_candidate,
    fingerprint_payload,
    generate_parameter_candidates,
    load_evolution_state,
    load_evolution_transition_journal,
    promote_to_shadow,
    reconcile_evolution_transition,
    rollback_shadow,
    stage_evolution_transition,
    write_evolution_state_atomic,
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


def _concurrent_state_writer(path, expected_fingerprint, run_id, barrier, results):
    state = EvolutionState.initial(
        "v1",
        {"blob": "x" * 8_000_000},
        now="2026-07-12T00:00:00+00:00",
    )
    state = replace(state, last_completed_run_id=run_id)
    barrier.wait()
    try:
        write_evolution_state_atomic(
            Path(path), state, expected_previous_fingerprint=expected_fingerprint
        )
        results.put(("written", run_id))
    except EvolutionStateError:
        results.put(("rejected", run_id))


def _concurrent_first_state_writer(path, run_id, barrier, results):
    state = EvolutionState.initial(
        "v1",
        {"writer": run_id},
        now="2026-07-12T00:00:00+00:00",
    )
    barrier.wait()
    try:
        write_evolution_state_atomic(
            Path(path),
            state,
            expected_previous_fingerprint=StateWriteExpectation.ABSENT,
        )
        results.put(("written", run_id))
    except EvolutionStateError:
        results.put(("rejected", run_id))


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
        data_asof_date="2026-07-12",
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
        data_asof_date="2026-07-12",
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


def test_atomic_state_write_round_trips_and_checks_previous_fingerprint(tmp_path):
    path = tmp_path / "state.json"
    first = EvolutionState.initial(
        "v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00"
    )

    first_fp = write_evolution_state_atomic(path, first)

    assert load_evolution_state(path, first) == first
    second = replace(first, last_completed_run_id="run-2")
    with pytest.raises(EvolutionStateError, match="previous state fingerprint"):
        write_evolution_state_atomic(
            path, second, expected_previous_fingerprint="stale"
        )
    assert write_evolution_state_atomic(
        path, second, expected_previous_fingerprint=first_fp
    ) == fingerprint_payload(second)
    assert load_evolution_state(path, first) == second


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda payload: payload.update({"unknown": True}), "Unknown state fields"),
        (lambda payload: payload.pop("champion_version"), "Missing state fields"),
        (lambda payload: payload.update({"schema_version": 1}), "schema_version"),
    ],
)
def test_state_load_rejects_unknown_missing_and_unsupported_schema(
    tmp_path, mutation, message
):
    path = tmp_path / "state.json"
    state = EvolutionState.initial(
        "v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00"
    )
    payload = json.loads(json.dumps(state.__dict__))
    mutation(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvolutionStateError, match=message):
        load_evolution_state(path, state)


def test_state_load_returns_default_when_file_is_absent(tmp_path):
    default = EvolutionState.initial(
        "v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00"
    )

    assert load_evolution_state(tmp_path / "missing.json", default) is default


def test_state_write_rejects_unsupported_schema(tmp_path):
    with pytest.raises(EvolutionStateError, match="schema_version"):
        replace(EvolutionState.initial("v1", {"leverage": 0.6}), schema_version=1)


def test_evolution_state_schema_v2_marks_semantic_data_date_contract():
    state = EvolutionState.initial(
        "v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00"
    )

    assert state.schema_version == 2


def test_concurrent_writers_with_same_previous_fingerprint_reject_lost_update(tmp_path):
    path = tmp_path / "state.json"
    initial = EvolutionState.initial(
        "v1",
        {"blob": "x" * 8_000_000},
        now="2026-07-12T00:00:00+00:00",
    )
    previous_fingerprint = write_evolution_state_atomic(path, initial)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_state_writer,
            args=(str(path), previous_fingerprint, run_id, barrier, results),
        )
        for run_id in ("run-a", "run-b")
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    outcomes = sorted(results.get(timeout=5)[0] for _ in processes)
    assert outcomes == ["rejected", "written"]
    assert load_evolution_state(path, initial).last_completed_run_id in {"run-a", "run-b"}


def test_concurrent_first_writers_require_expected_absent_cas(tmp_path):
    path = tmp_path / "state.json"
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_first_state_writer,
            args=(str(path), run_id, barrier, results),
        )
        for run_id in ("run-a", "run-b")
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    outcomes = sorted(results.get(timeout=5)[0] for _ in processes)
    assert outcomes == ["rejected", "written"]
    written = load_evolution_state(
        path, EvolutionState.initial("unused", {"writer": "unused"})
    )
    assert written.champion_parameters["writer"] in {"run-a", "run-b"}


def test_state_write_rejects_older_data_asof_even_with_current_cas(tmp_path):
    path = tmp_path / "state.json"
    initial = EvolutionState.initial(
        "v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00"
    )
    newer = promote_to_shadow(
        initial,
        challenger_version="v2",
        challenger_parameters={"leverage": 0.75},
        experiment_id="exp-newer",
        run_id="run-newer",
        data_fingerprint="data-newer",
        data_asof_date="2026-07-12",
        now="2026-07-12T01:00:00+00:00",
    )
    current_fingerprint = write_evolution_state_atomic(
        path, newer, expected_previous_fingerprint=StateWriteExpectation.ABSENT
    )
    older = replace(
        newer,
        shadow_run_id="run-older",
        last_completed_run_id="run-older",
        data_asof_date="2026-07-11",
    )

    with pytest.raises(EvolutionStateError, match="data_asof_date"):
        write_evolution_state_atomic(
            path, older, expected_previous_fingerprint=current_fingerprint
        )

    assert load_evolution_state(path, initial) == newer


def test_pending_transition_reconciles_committed_when_target_state_exists(tmp_path):
    path = tmp_path / "state.json"
    initial = EvolutionState.initial(
        "v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00"
    )
    target = promote_to_shadow(
        initial,
        challenger_version="v2",
        challenger_parameters={"leverage": 0.75},
        experiment_id="exp-1",
        run_id="run-1",
        data_fingerprint="data-1",
        data_asof_date="2026-07-12",
        now="2026-07-12T01:00:00+00:00",
    )
    write_evolution_state_atomic(
        path, target, expected_previous_fingerprint=StateWriteExpectation.ABSENT
    )
    pending = stage_evolution_transition(
        path,
        target,
        operation="promote",
        run_id="run-1",
        experiment_id="exp-1",
        now="2026-07-12T01:00:00+00:00",
    )

    reconciled = reconcile_evolution_transition(
        path, initial, now="2026-07-12T02:00:00+00:00"
    )

    assert isinstance(pending, EvolutionTransitionJournal)
    assert pending.status == "pending"
    assert reconciled is not None
    assert reconciled.status == "committed"
    assert load_evolution_transition_journal(path) == reconciled


def test_pending_transition_reconciles_rejected_when_target_state_is_uncommitted(tmp_path):
    path = tmp_path / "state.json"
    initial = EvolutionState.initial(
        "v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00"
    )
    target = promote_to_shadow(
        initial,
        challenger_version="v2",
        challenger_parameters={"leverage": 0.75},
        experiment_id="exp-1",
        run_id="run-1",
        data_fingerprint="data-1",
        data_asof_date="2026-07-12",
        now="2026-07-12T01:00:00+00:00",
    )
    write_evolution_state_atomic(
        path, initial, expected_previous_fingerprint=StateWriteExpectation.ABSENT
    )
    stage_evolution_transition(
        path,
        target,
        operation="promote",
        run_id="run-1",
        experiment_id="exp-1",
        now="2026-07-12T01:00:00+00:00",
    )

    reconciled = reconcile_evolution_transition(
        path, initial, now="2026-07-12T02:00:00+00:00"
    )

    assert reconciled is not None
    assert reconciled.status == "rejected"
    assert "target state was not committed" in (reconciled.reason or "")
    assert load_evolution_state(path, initial) == initial


def test_commit_transition_writes_state_and_committed_journal(tmp_path):
    path = tmp_path / "state.json"
    initial = EvolutionState.initial(
        "v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00"
    )
    target = promote_to_shadow(
        initial,
        challenger_version="v2",
        challenger_parameters={"leverage": 0.75},
        experiment_id="exp-1",
        run_id="run-1",
        data_fingerprint="data-1",
        data_asof_date="2026-07-12",
        now="2026-07-12T01:00:00+00:00",
    )

    fingerprint = commit_evolution_state_transition(
        path,
        target,
        operation="promote",
        run_id="run-1",
        experiment_id="exp-1",
        expected_previous_fingerprint=StateWriteExpectation.ABSENT,
        now="2026-07-12T01:00:00+00:00",
    )

    assert fingerprint == fingerprint_payload(target)
    assert load_evolution_state(path, initial) == target
    journal = load_evolution_transition_journal(path)
    assert journal is not None
    assert journal.status == "committed"


@pytest.mark.parametrize(
    "policy_kwargs, message",
    [
        ({"min_folds": True}, "min_folds"),
        ({"min_filled_trades_per_fold": 0}, "min_filled_trades_per_fold"),
        ({"min_positive_fold_ratio": 1.01}, "min_positive_fold_ratio"),
        ({"min_mean_return_improvement": float("nan")}, "min_mean_return_improvement"),
        ({"min_mean_return_improvement": -0.01}, "min_mean_return_improvement"),
        ({"max_drawdown_floor": 0.01}, "max_drawdown_floor"),
        ({"max_drawdown_worsening": -0.01}, "max_drawdown_worsening"),
        ({"max_drawdown_worsening": 1.01}, "max_drawdown_worsening"),
        ({"max_turnover_ratio": 0.0}, "max_turnover_ratio"),
        ({"max_pnl_concentration": -0.01}, "max_pnl_concentration"),
    ],
)
def test_promotion_policy_rejects_invalid_semantics(policy_kwargs, message):
    with pytest.raises(ValueError, match=message):
        PromotionPolicy(**policy_kwargs)


def test_evolution_state_rejects_invalid_enums_identifiers_and_timestamps():
    initial = EvolutionState.initial(
        "v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00"
    )

    with pytest.raises(EvolutionStateError, match="shadow_status"):
        replace(initial, shadow_status="mystery")
    with pytest.raises(EvolutionStateError, match="champion_version"):
        replace(initial, champion_version=123)
    with pytest.raises(EvolutionStateError, match="updated_at"):
        replace(initial, updated_at="not-a-timestamp")
    with pytest.raises(EvolutionStateError, match="data_asof_date"):
        replace(initial, data_asof_date="2026/07/12")


def test_evolution_state_rejects_incomplete_shadow_cross_fields():
    initial = EvolutionState.initial(
        "v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00"
    )

    with pytest.raises(EvolutionStateError, match="shadow state requires"):
        replace(initial, shadow_status="shadow")
