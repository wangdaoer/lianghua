# Final Review Fix Report

Status: `DONE`

Worktree: `D:\codex\智能化\high_risk_quant_model3\.worktrees\generic-strategy-evolution-core`

Binding base: `b1cac5e`

## Commits

- `abdaee3` - `fix: harden strategy evolution final review`
- `f7e2501` - `fix: version semantic evolution state schema`

This report is committed in a follow-up documentation commit. Its own hash is reported by the final task response because a commit cannot embed its own final hash.

## Scope And Preserved Boundaries

- Research/paper-only evolution, satellite, and generic-core paths were changed.
- No broker, order, or daily production pipeline file was changed.
- No repository `configs/*.yaml` file was changed; `configs/evolution_strong_pullback.yaml` needed no schema extension because shadow reevaluation reuses the existing legacy and generic-core policies.
- CLI default remains `dry_run=True`; state mutation still requires both `--no-dry-run` and `--promote-shadow`.
- Runtime state paths are restricted to a JSON file directly under a dedicated `evolution_state` directory.
- Formal YAML writes remain prohibited.

## Finding Map

### C1 - Execute Real Out-Of-Sample Folds

Implemented:

- Added `evaluate_strategy_folds()` and a complete `PriceBundle` slicer.
- Every fold now calls the injected real strategy executor with `close`, `open_px`, `high`, `low`, `amount`, and `market_exposure` physically truncated at `fold.test_end`.
- Legacy train/validation metrics still come from the separate full research replay; generic promotion gates consume only metrics from the independent fold replays.
- Fold metrics are stored in trial cache payloads and reused only when evidence, parameters, artifact schema, fold IDs, and code fingerprint are unchanged.

Regression coverage:

- `test_orchestration_executes_every_trial_at_each_physical_fold_end`
- `test_future_bundle_changes_cannot_change_prior_fold_metrics_or_decision`

### C2 - Remove One-Session IC Lookahead

Implemented:

- Added `observable_ic_training_window()`.
- At signal close `i`, retraining uses exactly `train_days` IC rows ending at `i-2`; IC row `i-1` is excluded.
- The first signal/retrain moves to `train_days + 1` so the configured observation count is retained.
- Rebalance cadence is anchored to the same first causally fitted signal.

Regression coverage:

- `test_signal_close_ic_fit_excludes_unobservable_future_open_labels`
- Full satellite and execution-rule regressions.

### C3 - Enforce Core Gates At Every Group Transition

Implemented:

- A legacy group winner advances only when its generic-core decision is also `eligible_for_shadow`.
- A core-ineligible legacy winner leaves the incumbent ID, parameters, metrics, and trial key unchanged.
- Before shadow authorization, the final accumulated candidate is reevaluated directly against the locked starting champion on the real folds and on legacy validation metrics.
- The transition-blocking core decision remains in the manifest audit even when no candidate advances.

Regression coverage:

- `test_core_ineligible_group_winner_never_becomes_next_parent_or_shadow`
- `test_final_accumulated_candidate_must_pass_core_gates_against_locked_champion`

### I1 - Atomic First-State Creation

Implemented:

- Added `StateWriteExpectation.ABSENT`, distinct from `None` (no comparison).
- First creation checks absence while holding the same cross-process state lock used for CAS and replacement.
- Runner mutations automatically use expected-absent when the state did not exist at run start.

Regression coverage:

- `test_concurrent_first_writers_require_expected_absent_cas` uses two spawned processes and observes exactly one write and one rejection.

### I2 - Correct Fold Evidence Metrics

Implemented:

- Missing/invalid `realize_date`, missing/non-finite `turnover`, and malformed/non-mapping/non-finite contribution JSON fail closed.
- Trade audit rows are restricted to the fold test dates.
- `filled_trades` counts rows with nonzero turnover rather than all audit rows.
- Signed symbol contributions are aggregated across the complete fold before positive net symbols are selected for concentration.
- Concentration is largest positive net symbol contribution divided by total positive net contribution; no positive net contribution remains non-finite evidence.

Regression coverage:

- `test_fold_metrics_filter_dates_count_execution_days_and_net_signed_contributions`
- `test_fold_metrics_fail_closed_without_realize_date`
- `test_fold_metrics_fail_closed_on_malformed_contribution_json`

### I3 - Consume And Evaluate Existing Shadow State

Implemented:

- A persisted `shadow` version and its parameters are explicitly replayed as the evaluated incumbent.
- A completed shadow reevaluation must pass both legacy validation and generic-core fold gates before its parameters can parent a search group.
- Gate failure calls `rollback_shadow()` with the new semantic data date and persists through the same CAS/journal path when both explicit flags are present.
- Evidence/executor exceptions abort the run without mutating state; they are not misclassified as configured rollback-gate failures.

Regression coverage:

- `test_existing_shadow_is_reevaluated_and_used_as_candidate_parent`
- `test_failing_existing_shadow_rolls_back_with_cas_and_audit`
- `test_shadow_evidence_error_aborts_without_mutating_state`

### I4 - Crash-Consistent Promotion Journal And Shared Metadata

Implemented:

- Added a sibling `strong_pullback.promotion_journal.json` with strict `pending`, `committed`, and `rejected` states.
- Transition sequence is journal-pending -> state CAS -> journal-committed/rejected under a cross-process transition lock.
- Startup reconciles pending journal target fingerprints against actual global state and atomically repairs the interrupted run manifest.
- State and journal temporary files are flushed with `fsync` before atomic replacement.
- `latest.json` is atomic; `evolution_registry.jsonl` is parsed, deduplicated by `run_id`, and atomically rewritten under a shared cross-process metadata lock.

Regression coverage:

- `test_pending_transition_reconciles_committed_when_target_state_exists`
- `test_pending_transition_reconciles_rejected_when_target_state_is_uncommitted`
- `test_commit_transition_writes_state_and_committed_journal`
- `test_startup_reconciles_pending_committed_journal_and_prior_manifest`
- `test_concurrent_run_metadata_publication_is_atomic_and_deduplicated`
- Existing post-CAS artifact-failure reconciliation regression remains green.

### I5 - Prevent Stale Research Data From Replacing Newer State

Implemented:

- State-changing runs require `asof_date` to equal the panel's actual maximum date.
- State-changing runs require a benchmark file whose maximum date reaches the panel maximum.
- `EvolutionState` schema v2 persists `data_asof_date`.
- State replacement rejects removal or regression of the current semantic data date while holding the state lock.
- Dry-run research may intentionally use an older as-of date and remains non-mutating.
- Old schema-v1 state payloads are explicitly rejected; they must be manually reviewed and rebuilt because they lack semantic date evidence.

Regression coverage:

- `test_shadow_promotion_rejects_explicit_stale_asof_date`
- `test_shadow_promotion_rejects_benchmark_ending_before_panel`
- `test_dry_run_allows_older_asof_without_mutating_global_state`
- `test_state_write_rejects_older_data_asof_even_with_current_cas`
- `test_evolution_state_schema_v2_marks_semantic_data_date_contract`

### I6 - Strictly Validate Legacy Selection Config

Implemented:

- Selection mapping rejects unknown and missing fields before dataclass construction.
- Integer day/window fields require runtime `int`, exclude `bool`, and must be positive.
- Numeric fields exclude `bool`, require finite runtime numbers, and enforce logical bounds.
- `min_annualized_return_delta` must be non-negative; drawdown floor is in `[-1, 0]`; turnover ratio is at least `1`; negative-window ratio is in `[0, 1]`.
- `calculate_segment_metrics()` independently rejects invalid rolling windows before calling `shift()`.

Regression coverage:

- `test_selection_rules_reject_boolean_nan_and_wrong_runtime_types`
- `test_selection_rules_reject_non_logical_ranges`
- `test_selection_rules_accept_exact_boundaries`
- `test_segment_metrics_rejects_negative_rolling_window_before_shift`

## Minor Hardening Included

- Strict `EvolutionState` status, identifier, aware timestamp, semantic date, and lifecycle cross-field validation.
- Strict `PromotionPolicy` type, finiteness, and range validation.
- Dedicated runtime state-path root.
- Strategy/core code fingerprint included in run evidence and experiment identity.
- Atomic run metadata publication.
- `elapsed_seconds` and process peak working-set memory recorded in success/failure manifests.
- `MODEL_OPERATING_SYSTEM.md` and `README.md` now describe the benchmarked verified workflow, real folds, legacy full-exposure fallback, explicit freshness mutation gates, journal recovery, and state schema v2.

## TDD RED Evidence

Each Critical and Important finding was captured before its production change:

| Finding | RED command/evidence |
| --- | --- |
| C1 | `python -m pytest tests/test_strong_pullback_evolution.py -q -k "real_fold_replays or future_bundle_changes"` -> collection error: missing `evaluate_strategy_folds`; orchestration then proved missing fold executions. |
| C2 | `python -m pytest tests/test_strong_pullback_satellite.py -q -k signal_close_ic_fit` -> collection error: missing `observable_ic_training_window`. |
| C3 | `python -m pytest tests/test_strong_pullback_evolution.py -q -k "core_ineligible_group_winner or final_accumulated_candidate"` -> `2 failed`; bad group advanced and path-dependent final candidate entered shadow. |
| I1 | `python -m pytest tests/test_strategy_evolution_core.py -q -k concurrent_first_writers` -> collection error: missing explicit absent expectation. |
| I2 | `python -m pytest tests/test_strong_pullback_evolution.py -q -k fold_metrics` -> `3 failed, 1 passed`; missing date/JSON did not fail and all rows counted as fills. |
| I3 | `python -m pytest tests/test_strong_pullback_evolution.py -q -k "existing_shadow_is_reevaluated or failing_existing_shadow"` -> `2 failed`; shadow was ignored and not rolled back. |
| I4 | Core journal tests initially failed collection on missing journal API; metadata publication failed collection on missing `publish_run_metadata`; startup recovery remained `pending` in a dedicated `1 failed` run. |
| I5 | Core stale-date test failed on missing `data_asof_date`; runner freshness command produced `2 failed, 1 passed` because stale as-of and short benchmark were accepted. |
| I6 | `python -m pytest tests/test_strong_pullback_evolution.py -q -k "selection_rules or negative_rolling_window"` -> `17 failed, 3 passed, 58 deselected, 9 subtests passed`. |

Additional self-review RED:

- `test_shadow_evidence_error_aborts_without_mutating_state` initially failed because an evidence exception was converted into rollback.
- State schema-v2 contract tests initially produced `3 failed, 2 passed` because the changed state payload was still labeled v1.

## GREEN And Final Verification

Finding-sized GREEN runs were completed after each implementation. Final fresh verification on the exact committed implementation candidate:

1. Focused evolution/satellite/core/execution tests:

   `python -m pytest tests/test_strategy_evolution_core.py tests/test_strong_pullback_evolution.py tests/test_strong_pullback_satellite.py tests/test_execution_rules.py -q`

   Result: `131 passed, 112 subtests passed in 50.67s`.

2. Full project regression:

   `python -m pytest -q`

   Result: `221 passed, 112 subtests passed in 54.90s`.

3. Compile verification:

   `python -m compileall strategy_evolution_core.py strong_pullback_evolution.py run_strong_pullback_evolution.py run_strong_pullback_satellite.py execution_rules.py -q`

   Result: exit `0`, no output.

4. Diff whitespace verification:

   `git diff --check HEAD`

   Result: exit `0`, no output.

5. Reduced real-engine paper-only smoke:

   `python -m pytest tests/test_strong_pullback_evolution.py::EvolutionCliTest::test_real_engine_evolution_writes_versioned_outputs -q`

   Result: `1 passed in 25.16s`.

6. Scope/config audit:

   - `git diff --name-only b1cac5e` contained only owned evolution/satellite/core tests and the two requested docs, plus this report.
   - `git diff --exit-code b1cac5e -- configs` exited `0`.
   - No test function was deleted. The prior resume assertion was strengthened from one replay to one legacy replay plus every physical fold; old state-path assertions were updated only to the dedicated safe directory.

## Changed Files

- `strategy_evolution_core.py`
- `strong_pullback_evolution.py`
- `run_strong_pullback_evolution.py`
- `run_strong_pullback_satellite.py`
- `tests/test_strategy_evolution_core.py`
- `tests/test_strong_pullback_evolution.py`
- `tests/test_strong_pullback_satellite.py`
- `MODEL_OPERATING_SYSTEM.md`
- `README.md`
- `.superpowers/sdd/final-review-fix-report.md`

Intentionally unchanged:

- `execution_rules.py` (compiled and regression-tested, but no source change was required).
- `configs/evolution_strong_pullback.yaml`.
- All broker/order/daily-pipeline files.

## Self-Review

- Causal boundary: executor inputs and IC windows are physically/time causally bounded; tests mutate future data and verify unchanged earlier outcomes.
- Transition integrity: both gate families are enforced per group, and final accumulated parameters are directly compared with the locked starting champion.
- Persistence integrity: first-create CAS, semantic-date monotonicity, journal recovery, and metadata locks are all exercised with spawned-process or crash-state tests.
- Fail-closed behavior: malformed fold audit and shadow evidence errors do not silently produce eligible metrics or mutate state.
- Preservation: dry-run default, dual mutation flags, formal-config prohibition, and research-only boundaries remain directly tested.
- Test integrity: no tests or assertions were removed to obtain GREEN; fixture changes add required dates/benchmarks or account for the newly required physical fold replays.

## Concerns And Deferred Work

- No unresolved Critical or Important concern remains within this fix-wave scope.
- The 4.8-million-row double replay was intentionally not run, exactly as directed by the brief. The reduced real-engine paper smoke passed; the large replay remains the next final-review activity after causal/gate wiring approval.
- Schema-v1 state files are intentionally fail-closed because they lack `data_asof_date`; operators must manually review and rebuild them as documented rather than silently migrate unverifiable freshness.
