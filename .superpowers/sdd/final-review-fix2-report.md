# Final Review Fix Wave 2 Report

## Status

`DONE`

- Binding brief: `.superpowers/sdd/final-review-fix2-brief.md`
- Base commit: `61a50c3`
- Implementation commit: `5a483e6` (`fix: isolate strategy evolution evidence stages`)
- Scope: evolution core/adapter/runner tests, research evolution config, README, and model operating-system documentation only.
- Excluded by design: broker, order, daily-pipeline, and formal strategy configuration files.

## Finding Map

### C1 - Fully disjoint parameter selection and core-test folds

Implemented:

- Extended `EvolutionPeriods` with required `core_test_start` and `core_test_end`.
- Enforced `train_end < validation_start <= validation_end < core_test_start <= core_test_end < test_start`.
- Changed the default calendar to selection through `2025-06-30`, core test from `2025-07-01` through `2025-12-31`, and final holdout from `2026-01-01`.
- Added `build_core_test_segments()`, which partitions effective trading dates into `min_folds` chronological, complete, pairwise-disjoint segments.
- Physically caps all parameter/group selection replays at `validation_end` and computes selection metrics only inside the validation interval.
- Locks the accumulated candidate path before loading core-test data.
- Replays only the locked candidate and locked starting champion at each core segment end; metrics are restricted to the segment's dates.
- Opens final holdout data only after the locked candidate passes all generic core gates.
- Writes core segment fingerprints to `folds.json` and records `core_test_status` separately from selection and final holdout status.

Proof tests:

- `test_requires_disjoint_selection_core_and_holdout_periods`
- `test_default_core_test_segments_are_complete_disjoint_and_non_overlapping`
- `test_core_rejection_keeps_final_holdout_untouched_after_selection_lock`
- `test_core_and_holdout_changes_cannot_reopen_locked_selection_path`
- `test_old_full_2025_selection_winner_no_longer_uses_core_test_rows`
- `test_selection_load_precedes_core_and_gated_holdout_load`
- `test_orchestration_replays_only_locked_candidate_at_core_segment_ends`

RED evidence:

- Period test failed with `AttributeError: 'EvolutionPeriods' object has no attribute 'core_test_start'`.
- Default config test failed with `ValueError: Missing required date: core_test_start`.
- Segment test failed because `build_core_test_segments` did not exist.
- Orchestration test failed because the executor observed the final holdout before any core-test gate: `AssertionError: final holdout opened before core gates passed`.

GREEN evidence:

- Period chronology, default config, segment partition, core-gated holdout, literal core/holdout price isolation, and old-full-2025 regression fixtures all pass.
- Default config produces three complete non-overlapping core segments with no selection date in any segment.

### I3 - Persist rollback before any new search

Implemented:

- Existing-shadow reevaluation uses only the disjoint selection evidence available before path lock.
- A failed shadow produces a rollback snapshot immediately.
- With explicit mutation flags, effective loaded freshness is validated, then rollback is committed as its own rollback journal/CAS event.
- The run finalizes with `not_opened_shadow_rollback` and returns before `build_group_candidates()` can run.
- The rollback manifest and decision retain the failed shadow experiment ID.
- Dry-run still stops after a local rollback decision without changing global state.

Proof test:

- `test_failing_shadow_persists_rollback_and_stops_before_strong_replacement_search`

RED evidence:

- The strong-replacement fixture failed with `Expected 'build_group_candidates' to not have been called. Called 1 times.`

GREEN evidence:

- Candidate search is not called.
- Persisted state is `rolled_back`.
- Journal operation/status are `rollback`/`committed`.
- Manifest identifies `prior-exp` and no replacement is promoted.

### I4 - Recover committed-but-unfinalized journal

Implemented:

- Startup reconciliation now handles both pending and already committed journals.
- Pending journals are first reconciled against actual state; matching state commits and nonmatching state rejects.
- A committed journal paired with a pending/running manifest is finalized idempotently without writing state again.
- Decision markdown, manifest truth, `latest.json`, and deduplicated registry metadata are finalized once.
- A rejected transition finalizes the interrupted manifest as failed and is not published as latest.

Proof tests:

- `test_startup_finalizes_committed_but_unfinalized_run_idempotently`
- `test_startup_reconciles_pending_committed_journal_and_prior_manifest`
- Existing core journal matching/nonmatching reconciliation tests remain green.

RED evidence:

- The committed crash fixture left no registry and failed with `FileNotFoundError` for `evolution_registry.jsonl`.

GREEN evidence:

- Reconciliation twice leaves state bytes unchanged.
- Manifest becomes `success` with persistence `committed` and `global_state_changed=true`.
- Decision records committed truth.
- Registry contains exactly one record for the recovered run.
- Second reconciliation makes no manifest change.

### I5 - Use effective loaded data freshness

Implemented:

- Retained the raw-date precheck for explicit state-changing runs.
- Added a second mandatory check against the loaded replay bundle after normal loader filtering and matrix cleaning.
- Requires close, open, high, low, amount, a common valid symbol observation, and the benchmark's effective source series to reach `asof_date`.
- Real benchmark loading records the last valid date before exposure reindex/forward fill, preventing a filled exposure from disguising stale source data.
- Promotion and rollback persist the effective loaded date, not merely the raw CSV maximum.
- Historical dry-run research remains non-mutating and may use an older as-of date.

Proof tests:

- `test_effective_loaded_freshness_rejects_raw_latest_row_dropped_by_cleaning`
- `test_effective_loaded_freshness_rejects_invalid_latest_benchmark_row`
- `test_effective_loaded_freshness_requires_common_valid_price_observation`
- `test_shadow_promotion_rejects_effective_loaded_bundle_older_than_raw_max`
- Existing raw panel/benchmark freshness and dry-run tests remain green.

RED evidence:

- Initial fixture failed because `validate_loaded_bundle_freshness` did not exist.
- Common-observation fixture initially raised no error when required matrices had no common valid symbol on the latest date.

GREEN evidence:

- Invalid latest panel or benchmark rows are rejected after loading.
- A raw-current but effectively stale promotion run raises before state creation.
- Common matrix validity and benchmark source validity are both enforced.

### Path safety

Implemented:

- Group and candidate IDs must be nonempty safe filename components.
- Rejects POSIX/Windows absolute paths, separators, traversal, dot components, control characters, Windows-invalid filename characters, and trailing dots.
- Rejects reserved internal names case-insensitively, including `baseline`, `champion`, `final`, and `experiments`.
- Group and candidate uniqueness is enforced after case normalization.
- `_resolve_trial_dir()` independently validates every trial ID and proves the resolved path is exactly one direct child of `trials/`.

Proof tests:

- `test_rejects_unsafe_reserved_and_case_colliding_group_candidate_ids`
- `test_trial_path_resolver_keeps_every_trial_as_a_direct_child`

RED evidence:

- Initial path matrix produced 27 failures because traversal, absolute, reserved, control, and case-collision values were accepted.
- Runtime resolver test failed because `_resolve_trial_dir` did not exist.
- Self-review RED additionally showed Windows-invalid colon and trailing-dot IDs were accepted.

GREEN evidence:

- All unsafe/reserved/collision subtests pass.
- Safe trial IDs resolve to one direct child; traversal and absolute inputs fail before filesystem writes.

### Code fingerprint

Implemented:

- Added `STRATEGY_CODE_DEPENDENCIES` and `build_code_fingerprint()`.
- Fingerprint covers:
  - `strategy_evolution_core.py`
  - `strong_pullback_evolution.py`
  - `run_strong_pullback_evolution.py`
  - `run_strong_pullback_satellite.py`
  - `train_next_open_rank_model.py`
  - `run_backtest.py`
  - `execution_rules.py`
  - `generate_strong_pullback_candidates.py`
- `RunEvidence.code_fingerprint` uses that aggregate, so any dependency change alters run evidence and invalidates resume identity.

Proof test:

- `test_transitive_strategy_dependencies_change_code_and_resume_identity`

RED evidence:

- Test failed because `STRATEGY_CODE_DEPENDENCIES` did not exist.

GREEN evidence:

- Changing each listed dependency independently changes the code fingerprint.
- Changing only the code fingerprint changes `RunEvidence.fingerprint`.

## Safety Boundaries

- CLI default remains `dry_run=True` and `promote_shadow=False`.
- State mutation still requires both `--no-dry-run` and `--promote-shadow`.
- State paths remain restricted to a dedicated `evolution_state` directory.
- No formal strategy YAML is written; only `configs/evolution_strong_pullback.yaml`, the research evolution schema, changed.
- No broker, order, or daily-pipeline file changed.
- No broker connection or order generation was introduced.
- Existing causal satellite IC window and A-share execution-rule tests remain green.

## Changed Files

- `strong_pullback_evolution.py`
- `run_strong_pullback_evolution.py`
- `configs/evolution_strong_pullback.yaml`
- `tests/test_strong_pullback_evolution.py`
- `README.md`
- `MODEL_OPERATING_SYSTEM.md`
- `.superpowers/sdd/final-review-fix2-report.md`

## Verification

Baseline before wave 2:

- `python -m pytest tests/test_strategy_evolution_core.py tests/test_strong_pullback_evolution.py tests/test_strong_pullback_satellite.py tests/test_execution_rules.py -q`
- Result: `131 passed, 112 subtests passed in 63.57s`.

Final focused regression:

- Same focused command.
- Result: `145 passed, 156 subtests passed in 60.00s`.

Final full regression:

- `python -m pytest -q`
- Result: `235 passed, 156 subtests passed in 56.88s`.

Compilation:

- `python -m compileall strategy_evolution_core.py strong_pullback_evolution.py run_strong_pullback_evolution.py run_strong_pullback_satellite.py execution_rules.py train_next_open_rank_model.py run_backtest.py generate_strong_pullback_candidates.py -q`
- Result: exit code `0`.

Diff hygiene and scope:

- `git diff --check`
- Result: exit code `0`.
- Changed-file inspection found only the owned evolution/config/test/docs files listed above.

Reduced paper-only real-engine smoke:

- `python -m pytest tests/test_strong_pullback_evolution.py::EvolutionCliTest::test_real_engine_evolution_writes_versioned_outputs -q`
- Result: `1 passed in 37.65s`.
- The default dry-run path was used and the synthetic smoke calendar contains disjoint selection, core-test, and final-holdout periods.

The 4.8M-row double replay was intentionally not run, exactly as required by the binding brief pending fresh final review.

## Self-Review

- Temporal decisions are one-way: selection can affect the locked path; core and holdout cannot feed back into it; holdout cannot feed back into core.
- Core segments cover the entire configured core interval exactly once and fail closed when fewer than `min_folds` effective trading dates exist.
- Rollback commits before any replacement search and terminates the run.
- Committed crash recovery changes journal/run metadata only and does not rewrite state.
- Effective freshness is checked on the same cleaned structures used by replay, with benchmark source freshness retained separately from forward-filled exposure.
- Candidate/group IDs are checked both at config parsing and at path resolution.
- Fingerprint coverage follows every local behavior-bearing replay/metric dependency in the import chain.
- Formal configuration and broker/daily boundaries are unchanged.

## Concerns

- No known blocking or correctness concern remains in the reviewed scope.
- The intentionally excluded 4.8M-row double replay remains the next expensive validation only after fresh final-review approval.
