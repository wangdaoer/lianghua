# Final Review Fix Wave 5 Report

## Status

`DONE`

- Base commit: `d506026` (`docs: record final review fix wave 4 evidence`)
- Implementation commit: `bd24523` (`fix: harden evolution recovery and feasible gates`)
- Scope remained inside evolution config/schema/orchestration, corresponding tests, and factual evolution documentation.
- No broker, order, or daily-pipeline file changed.

## Finding Map

### 1. Feasible checked-in selection evidence

Implemented:

- Changed checked-in `selection.min_validation_days` from `120` to `100`.
- Changed checked-in `selection.rolling_window_days` from `126` to `63`.
- Preserved the disjoint parameter-selection period at 2025-01-01..2025-06-30, locked core at 2025-07-01..2025-12-31, and final holdout from 2026-01-01.
- Preserved all checked-in selection/core risk thresholds: drawdown, Sharpe delta, turnover ratio, negative rolling-window rate, drawdown worsening, and PnL concentration.
- Documented that the selection period has 117 actual A-share sessions, the 100-day floor allows limited suspension/data gaps, and a 63-session window yields 54 completed windows on full coverage.
- Updated README, operating-system documentation, original design, and both relevant implementation plans so no checked-in documentation advertises the impossible `120`/`126` pair.

Proof test:

- `test_checked_in_selection_gates_are_feasible_on_real_2025_calendar`
- The test removes the 12 official weekday exchange closures from the 2025 first-half business-day calendar and asserts 117 sessions.
- A superior synthetic candidate records 117 trade days, 54 completed rolling windows, and passes the checked-in selection policy.
- An otherwise high-return synthetic candidate with a 45% drawdown is still rejected by the unchanged 40% drawdown floor.

### 2. State truth after committed-journal finalization failure

Implemented:

- The outer failure handler no longer treats a same-run pending journal as proof that global state did not change.
- It loads the actual state and requires all of the following before reporting commit truth:
  - actual-state fingerprint equals the journal target fingerprint;
  - actual `last_completed_run_id` equals the current run;
  - actual shadow experiment equals the selected/journal experiment;
  - journal run and experiment also match the current transition.
- If those checks pass and the journal is pending, the handler calls the existing locked `reconcile_evolution_transition()` path.
- Reconciliation is idempotent and writes the pending journal as committed when the target state is already present.
- The original journal-write exception remains visible in failed-run artifacts, while manifest and decision truthfully record `promotion_persistence_status=committed` and `global_state_changed=true`.

Proof test:

- `test_post_cas_committed_journal_write_failure_reconciles_state_truth`
- The test injects one `OSError` only for the first committed-journal write, after state CAS succeeds.
- It verifies the state run/experiment, journal target fingerprint, reconciled committed journal, manifest truth, decision truth, and retained original error.

### 3. Strict crash-recovery run path containment

Implemented:

- Strengthened `_resolve_run_dir()` to apply the strict direct-child rules already used by trial paths: non-empty, no `.`/`..`, no slash/backslash, no absolute POSIX or Windows path, no control/Windows-invalid characters, no trailing dot, and no Windows reserved device name including extensions.
- `reconcile_startup_transition()` now calls `_resolve_run_dir()` immediately after reading the journal and before reconciling a pending journal or probing any manifest, decision, latest pointer, or registry metadata.
- Recovery uses the resolved, containment-proven root and direct-child run directory for all later artifact paths.
- The pending journal remains pending when its run ID is rejected; no recovery artifact or metadata is touched.

Proof test:

- `test_startup_recovery_rejects_unsafe_journal_run_ids_before_any_probe`
- Covered IDs: `.`, `..`, forward/backslash children, forward/backslash traversal, Windows absolute path, `CON`, `nul.txt`, and trailing-dot form.
- All ten probes assert the shared resolver is called, all manifest/decision/metadata write probes remain untouched, and the journal remains pending.

### 4. Remove unused evolution-core walk-forward fields

Implemented:

- Removed `train_days`, `validation_days`, `test_days`, and `step_days` from `EVOLUTION_CORE_KEYS`.
- Removed the four fields from frozen `EvolutionCoreConfig` and `_parse_evolution_core()`.
- Removed them from the checked-in YAML and all current schema documentation.
- Strict compatibility behavior is explicit: supplying any removed field now raises `Unknown evolution_core keys`.
- The disjoint `periods` date boundaries remain the sole orchestration timing contract.
- The independent `build_trading_folds(trading_dates, train_days, validation_days, test_days, step_days)` utility remains supported and tested; those function arguments are not config schema fields.
- Baseline strategy `train_days: 252` remains because it is consumed by the real model executor and is unrelated to the removed orchestration fields.

Proof test:

- `test_period_driven_core_schema_rejects_removed_walk_forward_fields`
- Verifies a period-driven config parses without the fields, the dataclass/resolved config omit them, and each legacy field is rejected when reintroduced.

## RED Evidence

Command:

`python -m pytest -q tests/test_strong_pullback_evolution.py::EvolutionConfigTest::test_period_driven_core_schema_rejects_removed_walk_forward_fields tests/test_strong_pullback_evolution.py::EvolutionOrchestrationTest::test_post_cas_committed_journal_write_failure_reconciles_state_truth tests/test_strong_pullback_evolution.py::EvolutionOrchestrationTest::test_startup_recovery_rejects_unsafe_journal_run_ids_before_any_probe tests/test_strong_pullback_evolution.py::EvolutionDecisionTest::test_checked_in_selection_gates_are_feasible_on_real_2025_calendar`

Observed before production changes:

- Period-driven config failed with missing legacy core fields.
- Post-CAS failure left the transition journal `pending`.
- Recovery reconciled before validating the run path, accepted `..`, Windows device names, and trailing-dot forms, and did not call the shared resolver.
- Checked-in selection still required `120` days and a `126`-day rolling window.
- Pytest summary: `13 failed, 1 passed in 4.18s` including unsafe-ID subtests.

## GREEN Evidence

Targeted wave-5 regression:

- Same command as RED.
- Result: `4 passed, 14 subtests passed in 1.83s`.

Adjacent config/recovery/post-CAS regression:

- `python -m pytest -q tests/test_strong_pullback_evolution.py -k "default_config or evolution_core or post_cas or startup or checked_in_selection"`
- Result: `10 passed, 94 deselected, 34 subtests passed in 4.45s`.

Evolution module:

- `python -m pytest -q tests/test_strong_pullback_evolution.py`
- Result: `104 passed, 194 subtests passed in 63.00s`.

Focused evolution/execution regression:

- `python -m pytest -q tests/test_strategy_evolution_core.py tests/test_strong_pullback_evolution.py tests/test_strong_pullback_satellite.py tests/test_execution_rules.py`
- Result: `157 passed, 194 subtests passed in 67.25s`.

Full regression:

- `python -m pytest -q`
- Result: `247 passed, 194 subtests passed in 71.75s`.

Compilation:

- `python -m compileall strategy_evolution_core.py strong_pullback_evolution.py run_strong_pullback_evolution.py run_strong_pullback_satellite.py execution_rules.py train_next_open_rank_model.py run_backtest.py generate_strong_pullback_candidates.py -q`
- Result: exit code `0`.

Reduced paper-only real-engine smoke:

- `python -m pytest -q tests/test_strong_pullback_evolution.py::EvolutionCliTest::test_real_engine_evolution_writes_versioned_outputs`
- Result: `1 passed in 37.92s`.
- Default dry-run behavior remained in force.

Diff and documentation hygiene:

- `git diff --check`: exit code `0`.
- Stale-doc searches found no remaining checked-in YAML/Markdown `min_validation_days: 120`, `rolling_window_days: 126`, or config block advertising removed core fields.
- Risk-threshold diff confirms only the day/window evidence values changed; no core/holdout risk threshold changed.

The explicitly excluded 4.8M-row replay was not run.

## Changed Files

- `configs/evolution_strong_pullback.yaml`
- `strong_pullback_evolution.py`
- `run_strong_pullback_evolution.py`
- `tests/test_strong_pullback_evolution.py`
- `README.md`
- `MODEL_OPERATING_SYSTEM.md`
- `docs/superpowers/specs/2026-07-10-strong-pullback-self-evolution-design.md`
- `docs/superpowers/plans/2026-07-10-strong-pullback-self-evolution.md`
- `docs/superpowers/plans/2026-07-12-generic-strategy-evolution-core.md`
- `.superpowers/sdd/final-review-fix5-report.md`

## Safety Boundaries

- CLI still defaults to `dry_run=True` and `promote_shadow=False`.
- State mutation still requires explicit non-dry-run plus promotion authorization and freshness checks.
- Formal strategy parameters, broker paths, order paths, and daily pipelines remain untouched.
- Parameter selection, locked core, and final holdout periods remain disjoint and causally ordered.
- Recovery validates and contains journal-derived run paths before any run artifact or metadata probe.
- Post-CAS error handling reports actual committed state truth without swallowing the original failure.

## Self-Review

- Compared the complete config diff to confirm no risk threshold was lowered.
- Searched all checked-in YAML/Markdown for stale selection evidence values and removed schema fields.
- Verified every removed field is rejected rather than silently ignored.
- Verified the shared run resolver is used before pending-journal reconciliation and before artifact/metadata access.
- Verified actual-state fingerprint, run, and experiment all participate in post-CAS truth determination.
- Verified normal run IDs, healthy startup recovery, rejected startup recovery, post-CAS artifact handling, and dry-run behavior remain green.
- Verified the final changed-file list contains no prohibited operational trading file.

## Concerns

- No known correctness concern remains in the targeted scope.
- The actual-calendar test pins the 2025 first-half exchange closures in the fixture; a future selection period requires its own reviewed calendar fixture.
- The expensive 4.8M-row replay remains unexecuted by explicit review-wave convention.
