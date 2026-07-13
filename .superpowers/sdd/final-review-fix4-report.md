# Final Review Fix Wave 4 Report

## Status

`DONE`

- Base commit: `60a6775` (`docs: record final review fix wave 3 evidence`)
- Implementation commit: `0cff857` (`fix: gate persisted shadow parenting`)
- Scope remained inside persisted-shadow evolution orchestration and its tests.
- No config, formal strategy YAML, broker, order, or daily-pipeline file changed.

## Finding Map

### Persisted shadow must pass both selection-local policies before parenting

Implemented:

- Persisted-shadow reevaluation now consumes the selection-local `FoldMetrics` already produced by `execute_one()`.
- The shadow is compared with the locked champion/baseline through the same `evaluate_candidate(..., _promotion_policy(config))` call used by ordinary group transitions.
- A persisted shadow becomes the incumbent and can parent a group only when both the legacy selection decision is eligible and the generic selection-local decision has status `eligible_for_shadow`.
- Generic rejection and generic insufficient-evidence statuses both take the same fail-closed rollback branch.
- The rollback branch returns through `finish_without_holdout()` before `build_group_candidates()` can run, so a strong replacement cannot execute in the same run.
- The manifest preserves the concrete generic failed gates. The trial audit records generic status and failed gates.

Exact regression:

- The persisted shadow has stronger validation returns than baseline, so the legacy policy passes.
- Its selection-local realized symbol PnL is concentrated in one symbol, so the generic policy fails `pnl_concentration` at the configured `0.50` ceiling.
- A stronger `top_n=7` child is configured and would otherwise win.
- `build_group_candidates` is asserted never called.
- Explicit mutation restores the persisted state through rollback journal/CAS; dry-run leaves persisted state and journal untouched while writing a rollback recommendation.
- The run-local state snapshot has `shadow_status=rolled_back`, active `champion_parameters` equal to the locked baseline, and `champion_candidate.yaml` equal to the baseline. No `final/` holdout directory is opened.
- Failed-shadow parameters remain only in the state schema's dedicated shadow audit fields; they do not become the active champion or final candidate snapshot.

Proof test:

- `test_persisted_shadow_requires_generic_selection_gate_before_parenting`

### Rollback persistence failures must not be mislabeled as evidence failures

Implemented:

- The narrow evidence `try` now contains only shadow trial execution plus legacy and generic evidence evaluation.
- Trial-row construction, freshness authorization, rollback snapshot construction, artifact persistence, transition journaling, and CAS commit all execute outside that evidence `try`.
- A persistence, artifact, freshness, or CAS exception therefore reaches the outer run-failure handler with its original type and message.
- The manifest and `shadow_decision.md` can no longer rewrite a rollback CAS failure as `shadow reevaluation evidence failed`.

Proof test:

- `test_shadow_rollback_cas_failure_is_not_labeled_as_evidence_failure`
- The test injects `RuntimeError: rollback CAS exploded`, verifies persisted state remains unchanged, verifies persistence status is `rejected`, and verifies neither manifest nor decision text contains `evidence failed`.

## RED Evidence

Command:

`python -m pytest -q tests/test_strong_pullback_evolution.py::EvolutionOrchestrationTest::test_persisted_shadow_requires_generic_selection_gate_before_parenting tests/test_strong_pullback_evolution.py::EvolutionOrchestrationTest::test_shadow_rollback_cas_failure_is_not_labeled_as_evidence_failure`

Observed before production changes:

- Both concentration subtests failed because `build_group_candidates` was called once with the persisted shadow as parent.
- The CAS provenance test failed because the manifest error was `RuntimeError: shadow reevaluation evidence failed: RuntimeError: rollback CAS exploded`.
- Pytest summary: `3 failed, 1 passed in 3.84s`.

## GREEN Evidence

Targeted regression:

- Same command as RED.
- Result: `2 passed, 2 subtests passed in 2.29s`.

All shadow paths:

- `python -m pytest -q tests/test_strong_pullback_evolution.py -k shadow`
- Result: `14 passed, 86 deselected, 4 subtests passed in 6.60s`.

Focused evolution/execution regression:

- `python -m pytest -q tests/test_strategy_evolution_core.py tests/test_strong_pullback_evolution.py tests/test_strong_pullback_satellite.py tests/test_execution_rules.py`
- Result: `153 passed, 188 subtests passed in 77.23s`.

Full regression:

- `python -m pytest -q`
- Result: `243 passed, 188 subtests passed in 73.72s`.

Compilation:

- `python -m compileall strategy_evolution_core.py strong_pullback_evolution.py run_strong_pullback_evolution.py run_strong_pullback_satellite.py execution_rules.py train_next_open_rank_model.py run_backtest.py generate_strong_pullback_candidates.py -q`
- Result: exit code `0`.

Reduced paper-only real-engine smoke:

- `python -m pytest -q tests/test_strong_pullback_evolution.py::EvolutionCliTest::test_real_engine_evolution_writes_versioned_outputs`
- Result: `1 passed in 45.36s`.
- The smoke retained default dry-run behavior and did not mutate formal config or global shadow state.

Diff hygiene:

- `git diff --check`
- Result: exit code `0`.
- Pre-report changed-file audit contained only `run_strong_pullback_evolution.py` and `tests/test_strong_pullback_evolution.py`.

The explicitly excluded 4.8M-row replay was not run.

## Changed Files

- `run_strong_pullback_evolution.py`
- `tests/test_strong_pullback_evolution.py`
- `.superpowers/sdd/final-review-fix4-report.md`

## Safety Boundaries

- CLI defaults remain `dry_run=True` and `promote_shadow=False`.
- Global state mutation still requires explicit non-dry-run plus shadow-promotion authorization.
- Rejection uses the existing durable rollback journal and compare-and-swap path.
- Dry-run writes a local rollback recommendation without mutating persisted state or creating a transition journal.
- Selection-local reevaluation does not open locked core-test or final-holdout evidence.
- Formal configuration and broker/order boundaries remain untouched.

## Self-Review

- The persisted shadow uses the exact generic policy helper and baseline comparator used by ordinary transitions.
- Eligibility is conjunctive; rejection and insufficient evidence cannot enter the group loop.
- The early return is before the only `build_group_candidates()` call in the orchestration path.
- Generic failed-gate audit is preserved in the manifest and trials artifact.
- Rollback persistence is structurally outside the evidence exception boundary, not merely relabeled after catching.
- Healthy-shadow continuation, evidence-error handling, core rollback, holdout rollback, journal recovery, and dry-run behavior remain covered and green.
- The final diff does not touch unrelated or prohibited files.

## Concerns

- No known correctness concern remains in the targeted scope.
- Rolled-back state intentionally retains failed-shadow parameters in dedicated audit fields; active champion parameters and the emitted champion candidate are restored to the locked baseline.
- The expensive 4.8M-row replay remains unexecuted by explicit review-wave convention.
