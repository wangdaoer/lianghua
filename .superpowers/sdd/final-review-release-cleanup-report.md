# Final Review Release Cleanup Report

## Status

`DONE`

- Base commit: `6f5ef4c` (`docs: record final review fix wave 5 evidence`)
- Implementation commit: `eae27a2` (`fix: close evolution release gates`)
- Scope remained inside evolution adapter/orchestration, corresponding tests, and the two requested design specifications.
- No config risk threshold, broker, order, formal strategy, or daily-pipeline file changed.

## Finding Map

### 1. Exact baseline and override parameter types

Implemented:

- Added an explicit partition of every strategy parameter into integer, finite numeric, and optional finite numeric controls.
- Integer controls require `type(value) is int`; bool, fractional float, numeric string, NaN, and infinity are rejected before any coercion.
- Numeric controls require `numbers.Real`, reject bool and strings, and require a finite value.
- Optional numeric controls accept only `None` or a finite `Real`; bool, strings, NaN, and infinities are rejected.
- Existing logical/range checks still run after exact type validation.
- Baseline parsing and merged candidate overrides use the same `_validate_params()` path.

Proof test:

- `test_baseline_and_overrides_require_exact_parameter_runtime_types`
- The test proves the category partition covers every `DEFAULT_STRATEGY_PARAMS` key and every allowed override.
- Matrix coverage includes all baseline controls and all override-eligible controls.
- Invalid cases: bool, fractional integer, numeric string, NaN, positive infinity, and negative infinity as applicable.
- Optional controls additionally prove `None` and finite real values remain accepted.

### 2. Metadata recovery for already successful committed runs

Implemented:

- Startup recovery no longer returns immediately when journal and manifest already say committed/success.
- A committed journal plus truthful successful committed manifest is republished through `publish_run_metadata()`.
- Publication uses the existing cross-process metadata lock and deduplicating registry replacement.
- Missing `latest.json`, missing registry, or partial metadata is repaired without rewriting the evolution state, manifest, or shadow decision.
- Repeated startup recovery is byte-stable for latest/registry output and retains a single registry record for the run.

Proof tests:

- `test_startup_committed_success_manifest_repairs_missing_metadata_only`
- `test_startup_committed_success_metadata_repair_is_repeatable_and_deduplicated`
- Covered crash windows: both metadata files missing, and registry present while latest is missing.
- Both tests compare state and run artifacts byte-for-byte before and after recovery.

### 3. Final locked-candidate legacy gate before later evidence

Implemented:

- After all groups lock a cumulative candidate, the runner immediately compares its full selection metrics directly with the locked starting champion through the legacy policy.
- A final legacy rejection stops before loading locked core or final holdout evidence.
- The selected candidate score records `final_legacy_status`, JSON reasons, overall rejection status, and failed gates.
- The selected experiment records the same status/reasons and rejection state.
- Manifest records champion, selected experiment, final legacy status/reasons, and `legacy:<reason>` failed gates before later evidence can open.
- `shadow_decision.md` records final locked-candidate legacy status and reasons.
- Eligible final legacy decisions remain present through core, holdout, persistence, recovery, and failure artifacts.
- Audit defaults are initialized before the broad run `try`, so an earlier input/evidence failure retains its original exception.

Proof test:

- `test_final_accumulated_legacy_rejection_is_persisted_before_holdout`
- Two groups each make an individually legal turnover increase (`1.4x`, then about `1.36x`), while the cumulative locked candidate reaches `1.9x` against baseline and fails the `1.5x` legacy limit.
- The test verifies no core/holdout bundle is loaded, no `final/` artifacts exist, and score/experiment/manifest/decision all contain the turnover rejection.
- Existing direct-core cumulative test now uses a permissive legacy limit and strict core limit so the legacy and core regressions remain independent.

### 4. Current design documentation

Implemented in both design specifications:

- Added release-state notes marking the original two-stage and single-flag passages as superseded historical design, not executable current behavior.
- Documented parameter selection at 2025-01-01..2025-06-30 with `min_validation_days: 100` and `rolling_window_days: 63`.
- Documented locked core testing at 2025-07-01..2025-12-31.
- Documented untouched final holdout from 2026-01-01, opened only after selection and core gates pass.
- Documented per-group legacy plus generic selection gates and the cumulative final locked-candidate legacy gate.
- Documented that global state mutation requires both `--no-dry-run --promote-shadow`; either flag alone is insufficient.
- Distinguished the still-supported generic `build_trading_folds()` utility from the current period-driven orchestration contract.

Updated specifications:

- `docs/superpowers/specs/2026-07-10-strong-pullback-self-evolution-design.md`
- `docs/superpowers/specs/2026-07-12-generic-strategy-evolution-core-design.md`

## RED Evidence

Command:

`python -m pytest -q tests/test_strong_pullback_evolution.py::EvolutionConfigTest::test_baseline_and_overrides_require_exact_parameter_runtime_types tests/test_strong_pullback_evolution.py::EvolutionOrchestrationTest::test_startup_committed_success_manifest_repairs_missing_metadata_only tests/test_strong_pullback_evolution.py::EvolutionOrchestrationTest::test_startup_committed_success_metadata_repair_is_repeatable_and_deduplicated tests/test_strong_pullback_evolution.py::EvolutionOrchestrationTest::test_final_accumulated_legacy_rejection_is_persisted_before_holdout`

Observed before production changes:

- Coercive `int()`/`float()` validation accepted bools, fractional integers, and numeric strings and did not consistently reject non-finite controls with field-specific errors.
- Both committed-success recovery cases returned without creating/repairing metadata.
- The cumulative candidate loaded and ran holdout before its direct legacy rejection, and the decision was absent from required artifacts.
- Pytest summary: `192 failed, 1 passed, 115 subtests passed in 27.32s`.

## GREEN Evidence

Targeted release regression:

- Initial GREEN: `4 passed, 304 subtests passed in 2.27s`.
- Final targeted set including the early evidence-failure regression: `5 passed, 306 subtests passed in 2.34s`.

Adjacent recovery/final-gate regression:

- `python -m pytest -q tests/test_strong_pullback_evolution.py -k "runtime_types or startup or final_accumulated"`
- Result: `9 passed, 99 deselected, 330 subtests passed in 3.40s`.

Evolution module:

- `python -m pytest -q tests/test_strong_pullback_evolution.py`
- Result: `108 passed, 498 subtests passed in 53.83s`.
- An intermediate full-module run exposed two early selection-bundle subtest failures because final-legacy audit defaults were initialized too late; moving them before the run `try` restored the original assertion and the final module run passed.

Focused evolution/execution regression:

- `python -m pytest -q tests/test_strategy_evolution_core.py tests/test_strong_pullback_evolution.py tests/test_strong_pullback_satellite.py tests/test_execution_rules.py`
- Result: `161 passed, 498 subtests passed in 69.18s`.

Final full regression on committed code content:

- `python -m pytest -q`
- Result: `251 passed, 498 subtests passed in 79.99s`.

Compilation:

- `python -m compileall strategy_evolution_core.py strong_pullback_evolution.py run_strong_pullback_evolution.py run_strong_pullback_satellite.py execution_rules.py train_next_open_rank_model.py run_backtest.py generate_strong_pullback_candidates.py -q`
- Result: exit code `0`.

Reduced paper-only real-engine smoke:

- `python -m pytest -q tests/test_strong_pullback_evolution.py::EvolutionCliTest::test_real_engine_evolution_writes_versioned_outputs`
- Result: `1 passed in 37.85s`.
- Default dry-run behavior remained in force.

Diff and stale-document hygiene:

- `git diff --check`: exit code `0`.
- Stale-design search result: `stale-doc check clean`.
- Removed patterns included the old 2025 full-year validation boundary, `120`/`126`, single-flag mutation statements, and the old two-stage ordering expression.

The explicitly excluded 4.8M-row replay was not run.

## Changed Files

- `strong_pullback_evolution.py`
- `run_strong_pullback_evolution.py`
- `tests/test_strong_pullback_evolution.py`
- `docs/superpowers/specs/2026-07-10-strong-pullback-self-evolution-design.md`
- `docs/superpowers/specs/2026-07-12-generic-strategy-evolution-core-design.md`
- `.superpowers/sdd/final-review-release-cleanup-report.md`

## Safety Boundaries

- CLI still defaults to dry-run and no shadow promotion.
- State mutation still requires explicit `--no-dry-run --promote-shadow` plus existing freshness/CAS authorization.
- Formal YAML, broker, order, and daily-pipeline paths remain untouched.
- Final legacy rejection now stops earlier and cannot expose core or holdout evidence.
- Metadata recovery rewrites only metadata under its lock; evolution state and completed run artifacts remain unchanged.
- Existing core and final holdout risk thresholds were not changed.

## Self-Review

- Verified the parameter category union exactly equals all strategy defaults and covers every allowed override.
- Verified type checks occur before all integer/float conversions.
- Verified committed-success recovery uses the existing locked deduplicating publisher and never calls state persistence.
- Verified repeated recovery preserves one registry entry and byte-stable metadata content.
- Verified final legacy identity/status/reasons appear in score, experiment, manifest, and decision artifacts.
- Verified the early rejection occurs before the first post-selection bundle load.
- Verified both design specs contain all three current periods, `100/63`, dual mutation flags, and explicit historical labeling.
- Verified the final changed-file list contains no prohibited operational trading file.

## Concerns

- No known correctness concern remains in the targeted release scope.
- Startup verification republishes a truthful committed-success manifest under the metadata lock even when metadata is already complete; content remains idempotent, but this intentionally performs small atomic metadata writes on each startup reconciliation.
- The expensive 4.8M-row replay remains unexecuted by explicit review-wave convention.
