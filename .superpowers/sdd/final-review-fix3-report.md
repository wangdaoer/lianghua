# Final Review Fix Wave 3 Report

## Status

`DONE`

- Binding brief: `.superpowers/sdd/final-review-fix3-brief.md`
- Base commit: `658884f`
- Implementation commit: `828bb63` (`fix: restore guarded evolution group transitions`)
- Scope remained inside evolution/core/adapter tests and factual operator documentation.
- No broker, order, daily-pipeline, or formal strategy configuration file changed.

## Finding Map

### Critical - Generic hard gates at every group transition

Implemented:

- Added `build_selection_segments()` using a shared chronological period partitioner.
- Selection segments cover only `validation_start..validation_end`, are pairwise disjoint, and contain no core-test or holdout date.
- Every incumbent and candidate gets one full selection replay plus causal replays physically truncated at each selection segment end.
- Selection-local `FoldMetrics` are cached and validated with the trial artifacts.
- Every candidate is compared with the current group incumbent through generic `evaluate_candidate()` gates using only selection-local fold metrics.
- A group candidate becomes the next parent only when both the legacy selection decision and generic selection-local decision are eligible.
- If the generic decision fails, the incumbent remains unchanged; failed parameters cannot enter later groups or the final shadow.
- Candidate score and experiment artifacts now preserve selection-local fold metrics, generic gate status, and generic gate audit.
- The separate locked-candidate core-test construction, replay, and final OOS comparison remain after all group transitions and still use only `core_test_start..core_test_end`.

Proof tests:

- `test_core_ineligible_group_winner_never_becomes_next_parent_or_shadow`
- `test_selection_local_generic_segments_never_read_core_or_holdout_dates`
- `test_orchestration_replays_only_locked_candidate_at_core_segment_ends`
- `test_final_accumulated_candidate_must_pass_core_gates_against_locked_champion`
- Resume-cache corruption tests now require one full selection replay plus `min_folds` selection-local replays.

RED evidence:

- The inverted regression failed because group one selected `bad_core` instead of retaining `baseline`.
- The isolation test failed with `AttributeError` because `build_selection_segments` did not exist.

GREEN evidence:

- The generic-ineligible candidate no longer parents group two.
- Later candidate parameters retain the original `0.60` leverage, and final shadow parameters do not contain the failed `0.75` leverage.
- Every replay of the failed candidate ends on or before `validation_end`.
- Selection segments are all before `core_test_start`; locked core segments remain separate and unchanged.

### Important - Durable rollback for an existing shadow at core or holdout failure

Implemented:

- The runner explicitly identifies when the locked candidate is the persisted existing shadow rather than a new experiment.
- Core-test rejection or insufficient evidence creates a rollback snapshot, validates effective freshness for explicit mutation, commits rollback through journal/CAS, and stops.
- Final holdout `rollback_recommended` creates and commits the same durable rollback event and stops without raising a failed-run exception.
- Dry-run produces a local rollback snapshot and explicit rollback recommendation while leaving persisted state and journal unchanged.
- Core and holdout rollback manifests retain the failed shadow experiment ID, core status, rollback reason, persistence truth, and global-state truth.
- Holdout rollback preserves the completed final baseline/champion artifacts and comparison metrics.
- No candidate search or promotion occurs after a core or holdout rollback decision.

Proof tests:

- `test_locked_existing_shadow_core_failure_rolls_back_durably_or_recommends_in_dry_run`
- `test_locked_existing_shadow_holdout_failure_persists_rollback_and_stops`
- Existing selection-reevaluation rollback and strong-replacement stop test remains green.

RED evidence:

- Core failure returned `not_opened_core_rejected` instead of a shadow rollback for both dry and mutating runs.
- Holdout failure raised `RuntimeError: Holdout test rollback_recommended` and left the existing shadow unrolled back.

GREEN evidence:

- Mutating core and holdout failures reread as `shadow_status=rolled_back`.
- Journal operation/status are `rollback`/`committed`.
- Manifest persistence is `committed`, `global_state_changed=true`, and `selected_experiment_id=existing-exp`.
- Dry-run rereads the original shadow unchanged, has no journal, and records `not_changed` rollback recommendation truth.
- Holdout rollback finishes successfully and retains `final/champion/metrics.json`.

### Important - Finite effective freshness

Implemented:

- Required matrix coverage now uses finite numeric masks rather than non-null masks.
- `NaN`, `+inf`, and `-inf` cannot establish the latest effective date for close, open, high, low, or amount.
- The common valid symbol/date observation must also be finite across every required matrix.
- Raw benchmark source coverage filters to finite numeric closes before recording `benchmark_effective_date`.
- A benchmark with no finite effective close is rejected during loading.
- Existing raw-date and effective-loaded-date checks remain required before any state mutation.

Proof test:

- `test_effective_loaded_freshness_rejects_non_finite_latest_values`

Covered fixtures:

- Latest `amount=+inf`.
- Latest price `-inf`.
- Latest benchmark close `+inf`.

RED evidence:

- The amount and benchmark fixtures raised no error and incorrectly established latest coverage.
- The negative-infinity price fixture already failed closed because normal price loading removed the invalid row.

GREEN evidence:

- All three non-finite fixtures reject effective freshness authorization.
- Existing invalid-row, stale-bundle, common-observation, and benchmark-source freshness tests remain green.

### Minor hardening - Strict mappings and Windows device names

Implemented:

- Added exact allowed-key sets for the top-level config, periods, baseline, search-group, and candidate mappings.
- Existing selection and evolution-core strict key validation remains in force.
- Unknown fields now fail at every required mapping level while the checked-in config parses unchanged.
- Added case-insensitive Windows reserved-device rejection for `CON`, `PRN`, `AUX`, `NUL`, `COM1..COM9`, and `LPT1..LPT9`.
- Device names are rejected with extensions and trailing dot/space forms.
- The same device-name predicate also protects direct trial-path resolution.

Proof tests:

- `test_rejects_unknown_fields_at_every_config_mapping_level`
- `test_rejects_windows_reserved_device_names_with_extensions_and_trailing_forms`
- Existing traversal, absolute-path, reserved-internal-name, and case-collision tests remain green.

RED evidence:

- Unknown fields were accepted at top-level, periods, search-group, and candidate mappings.
- Windows device aliases were accepted for both group and candidate IDs, including extension forms.

GREEN evidence:

- All seven config mapping levels reject unknown fields.
- All device-name variants reject case-insensitively for group and candidate IDs.
- The checked-in `configs/evolution_strong_pullback.yaml` remains exact-field compatible.

## Safety Boundaries

- CLI still defaults to `dry_run=True` and `promote_shadow=False`.
- Global mutation still requires both `--no-dry-run` and `--promote-shadow`.
- Formal YAML, broker, order, and daily-pipeline boundaries remain untouched.
- Selection-local evidence cannot read core-test or final-holdout dates.
- Locked core-test evidence remains excluded from every group transition.
- Holdout still opens only after the locked candidate passes core tests.
- Existing satellite causality and A-share execution-rule tests remain green.

## Changed Files

- `strong_pullback_evolution.py`
- `run_strong_pullback_evolution.py`
- `tests/test_strong_pullback_evolution.py`
- `README.md`
- `MODEL_OPERATING_SYSTEM.md`
- `.superpowers/sdd/final-review-fix3-report.md`

## Verification

Wave-2 baseline:

- Full regression: `235 passed, 156 subtests passed`.

Final evolution module:

- `python -m pytest tests/test_strong_pullback_evolution.py -q`
- Result: `98 passed, 186 subtests passed in 63.74s`.

Final focused regression:

- `python -m pytest tests/test_strategy_evolution_core.py tests/test_strong_pullback_evolution.py tests/test_strong_pullback_satellite.py tests/test_execution_rules.py -q`
- Result: `151 passed, 186 subtests passed in 65.09s`.

Final full regression:

- `python -m pytest -q`
- Result: `241 passed, 186 subtests passed in 63.23s`.

Compilation:

- `python -m compileall strategy_evolution_core.py strong_pullback_evolution.py run_strong_pullback_evolution.py run_strong_pullback_satellite.py execution_rules.py train_next_open_rank_model.py run_backtest.py generate_strong_pullback_candidates.py -q`
- Result: exit code `0`.

Diff hygiene and scope:

- `git diff --check`
- Result: exit code `0`.
- Changed-file audit found only the five owned implementation/test/docs files plus this report.

Reduced paper-only real-engine smoke:

- `python -m pytest tests/test_strong_pullback_evolution.py::EvolutionCliTest::test_real_engine_evolution_writes_versioned_outputs -q`
- Result: `1 passed in 33.26s`.
- The smoke used default dry-run behavior and the revised selection-local/core/holdout sequence.

The 4.8M-row replay was intentionally not run, exactly as required pending fresh final review.

## Self-Review

- Generic group gates use only selection-local segments and compare against the current group incumbent.
- Failed generic candidates are excluded before legacy winner ranking can make them parents.
- Selection and core artifacts are separate: `selection_folds.json` versus locked-core `folds.json`.
- Existing-shadow rollback uses journal/CAS at all configured failure stages and preserves dry-run non-mutation.
- Holdout rollback preserves final evidence rather than rewriting it as an unopened holdout.
- Effective freshness requires finite values and finite raw benchmark source coverage.
- Config schema checks fail before candidate path construction or filesystem writes.
- No unrelated file or operational trading boundary changed.

## Concerns

- No known correctness concern remains in the reviewed scope.
- Per-group generic gating intentionally adds `min_folds` causal selection replays per candidate, increasing research runtime.
- The explicitly excluded 4.8M-row replay remains the next expensive validation only after fresh review approval.
