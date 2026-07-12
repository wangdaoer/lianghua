import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from strategy_evolution_core import EvolutionState

from strong_pullback_evolution import (
    DEFAULT_STRATEGY_PARAMS,
    TradingFold,
    assess_test_result,
    SearchCandidate,
    SearchGroup,
    build_trading_folds,
    build_group_candidates,
    calculate_fold_metrics,
    calculate_segment_metrics,
    choose_group_winner,
    evaluate_promotion,
    load_evolution_config,
    parse_evolution_config,
    run_strong_pullback_folds,
)
from run_strong_pullback_evolution import (
    PriceBundle,
    StrategyRun,
    can_resume_trial,
    execute_strategy_trial,
    load_price_bundle,
    load_trial_artifacts,
    parse_args,
    persist_evolution_outcome,
    run_evolution,
    validate_input_schema,
    write_trial_artifacts,
)


def valid_raw_config() -> dict[str, object]:
    return {
        "strategy": "strong_pullback_satellite",
        "evolution_core": {
            "train_days": 504,
            "validation_days": 126,
            "test_days": 126,
            "step_days": 63,
            "max_candidates_per_group": 8,
            "random_seed": 20260712,
            "min_folds": 3,
            "min_filled_trades_per_fold": 5,
            "min_positive_fold_ratio": 0.6666666667,
            "min_mean_return_improvement": 0.01,
            "max_drawdown_floor": -0.40,
            "max_drawdown_worsening": 0.05,
            "max_turnover_ratio": 1.50,
            "max_pnl_concentration": 0.50,
        },
        "periods": {
            "research_start": "2022-01-01",
            "train_end": "2024-12-31",
            "validation_start": "2025-01-01",
            "validation_end": "2025-12-31",
            "test_start": "2026-01-01",
            "test_end": None,
        },
        "baseline": {"top_n": 8, "leverage": 0.60, "max_position_weight": 0.08},
        "search_groups": [
            {
                "id": "risk_budget",
                "hypothesis_cn": "扩大风险预算",
                "candidates": [
                    {"id": "risk_075", "overrides": {"leverage": 0.75}},
                    {"id": "risk_090", "overrides": {"leverage": 0.90}},
                ],
            }
        ],
        "selection": {
            "min_validation_days": 120,
            "min_test_days": 60,
            "max_drawdown_floor": -0.40,
            "min_annualized_return_delta": 0.01,
            "min_sharpe_delta": -0.10,
            "max_turnover_ratio": 1.50,
            "rolling_window_days": 126,
            "max_negative_window_rate": 0.60,
        },
    }


class EvolutionCliTest(unittest.TestCase):
    def test_cli_requires_explicit_data_path(self):
        with self.assertRaises(SystemExit):
            parse_args(["--config", "configs/evolution_strong_pullback.yaml"])

    def test_default_config_parses_and_has_three_hypothesis_groups(self):
        config = load_evolution_config(Path("configs/evolution_strong_pullback.yaml"))

        self.assertEqual(
            [group.group_id for group in config.search_groups],
            ["risk_budget", "entry_depth", "rebound_exit"],
        )
        self.assertEqual(config.selection.max_drawdown_floor, -0.40)
        self.assertEqual(config.evolution_core.train_days, 504)
        self.assertEqual(config.evolution_core.random_seed, 20260712)
        self.assertEqual(config.evolution_core.min_positive_fold_ratio, 0.6666666667)

    def test_cli_defaults_to_dry_run_and_requires_explicit_shadow_promotion(self):
        args = parse_args([
            "--config", "configs/evolution_strong_pullback.yaml",
            "--data", "panel.csv",
        ])

        self.assertTrue(args.dry_run)
        self.assertFalse(args.promote_shadow)
        explicit = parse_args([
            "--config", "configs/evolution_strong_pullback.yaml",
            "--data", "panel.csv",
            "--no-dry-run",
            "--promote-shadow",
        ])
        self.assertFalse(explicit.dry_run)
        self.assertTrue(explicit.promote_shadow)

    def test_real_engine_evolution_writes_versioned_outputs(self):
        dates = pd.bdate_range("2024-01-02", periods=150)
        raw = valid_raw_config()
        raw["periods"] = {
            "research_start": dates[0].strftime("%Y-%m-%d"),
            "train_end": dates[89].strftime("%Y-%m-%d"),
            "validation_start": dates[90].strftime("%Y-%m-%d"),
            "validation_end": dates[119].strftime("%Y-%m-%d"),
            "test_start": dates[120].strftime("%Y-%m-%d"),
            "test_end": dates[-1].strftime("%Y-%m-%d"),
        }
        raw["baseline"].update({
            "train_days": 65,
            "retrain_frequency": 10,
            "top_n": 4,
            "rebalance_frequency": 5,
            "max_position_weight": 0.20,
            "leverage": 0.60,
            "min_avg_amount_20d": 1.0,
            "min_pullback_5d": 0.0,
            "max_pullback_5d": 1.0,
            "min_prior_return_20": -1.0,
            "min_prior_return_60": -1.0,
            "min_return_20d": -1.0,
            "min_return_60d": -1.0,
            "min_distance_ma60": -1.0,
            "max_intraday_return": 1.0,
        })
        raw["search_groups"] = [{
            "id": "risk_budget",
            "hypothesis_cn": "合成样本风险预算测试",
            "candidates": [{"id": "risk_065", "overrides": {"leverage": 0.65}}],
        }]
        raw["selection"].update({
            "min_validation_days": 20,
            "min_test_days": 15,
            "max_drawdown_floor": -0.90,
            "min_annualized_return_delta": 0.0,
            "min_sharpe_delta": -10.0,
            "max_turnover_ratio": 10.0,
            "rolling_window_days": 10,
            "max_negative_window_rate": 1.0,
        })
        rows: list[dict[str, object]] = []
        for symbol_index in range(32):
            symbol = f"{symbol_index + 1:06d}"
            for day_index, date in enumerate(dates):
                base = 8.0 + symbol_index * 0.08
                close = base * (1.0 + day_index * 0.001) * (
                    1.0 + 0.03 * np.sin(day_index / 6.0 + symbol_index * 0.2)
                )
                open_price = close * (1.0 - 0.002 * np.cos(day_index / 5.0))
                volume = 1_000_000.0 + symbol_index * 1_000.0
                rows.append({
                    "date": date,
                    "symbol": symbol,
                    "open": open_price,
                    "high": max(open_price, close) * 1.01,
                    "low": min(open_price, close) * 0.99,
                    "close": close,
                    "volume": volume,
                    "amount": volume * close,
                })

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "panel.csv"
            config_path = root / "evolution.yaml"
            pd.DataFrame(rows).to_csv(data_path, index=False)
            config_path.write_text(
                yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8"
            )
            config = load_evolution_config(config_path)
            outcome = run_evolution(
                config=config,
                data_path=data_path,
                config_path=config_path,
                benchmark_path=None,
                asof_date=dates[-1],
                output_root=root / "runs",
                run_id="real-engine-smoke",
                resume=False,
                git_commit="test-commit",
            )

            self.assertTrue((outcome.run_dir / "manifest.json").exists())
            self.assertTrue((outcome.run_dir / "champion_candidate.yaml").exists())
            self.assertTrue((outcome.run_dir / "test_comparison.csv").exists())
            self.assertTrue((outcome.run_dir / "final" / "baseline" / "equity_curve.csv").exists())


class EvolutionConfigTest(unittest.TestCase):
    def test_rejects_missing_and_unknown_evolution_core_keys(self):
        raw = valid_raw_config()
        del raw["evolution_core"]["train_days"]
        with self.assertRaisesRegex(ValueError, "Missing evolution_core keys"):
            parse_evolution_config(raw)

        raw = valid_raw_config()
        raw["evolution_core"]["future_rule"] = 1
        with self.assertRaisesRegex(ValueError, "Unknown evolution_core keys"):
            parse_evolution_config(raw)

    def test_rejects_invalid_evolution_core_values_and_boolean_integers(self):
        invalid_values = {
            "train_days": (0, True),
            "validation_days": (0, True),
            "test_days": (0, True),
            "step_days": (0, True),
            "max_candidates_per_group": (0, True),
            "random_seed": (True,),
            "min_folds": (0, True),
            "min_filled_trades_per_fold": (-1, True),
            "min_positive_fold_ratio": (-0.01, 1.01, True),
            "min_mean_return_improvement": (-0.01, 1.01, True),
            "max_drawdown_floor": (-1.01, 0.01, True),
            "max_drawdown_worsening": (-0.01, 1.01, True),
            "max_turnover_ratio": (0.0, True),
            "max_pnl_concentration": (-0.01, 1.01, True),
        }
        for key, values in invalid_values.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    raw = valid_raw_config()
                    raw["evolution_core"][key] = value
                    with self.assertRaisesRegex(ValueError, key):
                        parse_evolution_config(raw)

    def test_rejects_missing_evolution_core_block(self):
        raw = valid_raw_config()
        del raw["evolution_core"]

        with self.assertRaisesRegex(ValueError, "evolution_core must be a mapping"):
            parse_evolution_config(raw)

    def test_rejects_overlapping_validation_and_test_periods(self):
        raw = valid_raw_config()
        raw["periods"]["test_start"] = "2025-12-31"

        with self.assertRaisesRegex(ValueError, "train_end < validation_start <= validation_end < test_start"):
            parse_evolution_config(raw)

    def test_rejects_unknown_strategy_override(self):
        raw = valid_raw_config()
        raw["search_groups"][0]["candidates"][0]["overrides"] = {"future_leak": 1}

        with self.assertRaisesRegex(ValueError, "Unknown strategy parameters"):
            parse_evolution_config(raw)

    def test_rejects_duplicate_candidate_ids_across_groups(self):
        raw = valid_raw_config()
        raw["search_groups"].append(copy.deepcopy(raw["search_groups"][0]))
        raw["search_groups"][1]["id"] = "entry_depth"

        with self.assertRaisesRegex(ValueError, "Duplicate candidate id"):
            parse_evolution_config(raw)

    def test_group_candidates_share_incumbent_but_not_each_other(self):
        config = parse_evolution_config(valid_raw_config())
        generated = build_group_candidates(config.baseline, config.search_groups[0])

        self.assertEqual(generated[0][1]["leverage"], 0.75)
        self.assertEqual(generated[1][1]["leverage"], 0.90)
        self.assertEqual(config.baseline["leverage"], 0.60)
        self.assertIsNot(generated[0][1], generated[1][1])

    def test_group_candidates_use_generic_generator_with_explicit_seed_and_limit(self):
        config = parse_evolution_config(valid_raw_config())
        group = config.search_groups[0]

        with patch(
            "strong_pullback_evolution.generate_parameter_candidates",
            return_value=({**config.baseline, "leverage": 0.90},),
        ) as generator:
            generated = build_group_candidates(
                config.baseline, group, max_candidates=1, seed=12345
            )

        generator.assert_called_once_with(
            config.baseline,
            [candidate.overrides for candidate in group.candidates],
            max_candidates=1,
            seed=12345,
        )
        self.assertEqual(generated[0][0], "risk_090")

    def test_rejects_null_empty_and_non_string_group_ids(self):
        for invalid_id in (None, "", 123):
            with self.subTest(invalid_id=invalid_id):
                raw = valid_raw_config()
                raw["search_groups"][0]["id"] = invalid_id

                with self.assertRaisesRegex(ValueError, "Group id must be a non-empty string"):
                    parse_evolution_config(raw)

    def test_rejects_null_empty_and_non_string_candidate_ids(self):
        for invalid_id in (None, "", 123):
            with self.subTest(invalid_id=invalid_id):
                raw = valid_raw_config()
                raw["search_groups"][0]["candidates"][0]["id"] = invalid_id

                with self.assertRaisesRegex(ValueError, "Candidate id must be a non-empty string"):
                    parse_evolution_config(raw)

    def test_group_candidates_are_deeply_independent(self):
        incumbent = {"nested": {"values": ["incumbent"]}}
        group = SearchGroup(
            "nested",
            "nested values",
            (
                SearchCandidate("first", {"nested": {"values": ["first"]}}),
                SearchCandidate("second", {"nested": {"values": ["second"]}}),
            ),
        )

        generated = build_group_candidates(incumbent, group)
        generated[0][1]["nested"]["values"].append("changed")

        self.assertEqual(generated[1][1]["nested"]["values"], ["second"])
        self.assertEqual(incumbent["nested"]["values"], ["incumbent"])

    def test_rejects_malformed_search_group_entries(self):
        raw = valid_raw_config()
        raw["search_groups"] = [None]

        with self.assertRaisesRegex(ValueError, "search_groups entries must be mappings"):
            parse_evolution_config(raw)

    def test_rejects_malformed_search_candidate_entries(self):
        raw = valid_raw_config()
        raw["search_groups"][0]["candidates"] = [None]

        with self.assertRaisesRegex(ValueError, "candidates entries must be mappings"):
            parse_evolution_config(raw)

    def test_rejects_unknown_selection_keys(self):
        raw = valid_raw_config()
        raw["selection"]["future_rule"] = True

        with self.assertRaisesRegex(ValueError, "Unknown selection keys"):
            parse_evolution_config(raw)

    def test_rejects_missing_selection_keys(self):
        raw = valid_raw_config()
        del raw["selection"]["min_test_days"]

        with self.assertRaisesRegex(ValueError, "Missing selection keys"):
            parse_evolution_config(raw)

    def test_rejects_unsafe_execution_and_exposure_parameters(self):
        invalid_values = {
            "commission_bps": -0.01,
            "impact_bps": -0.01,
            "limit_buffer": (-0.01, 1.01),
            "rebound_exit_scale": (-0.01, 1.01),
            "max_buy_open_gap": -0.01,
            "market_below_ma_exposure": (-0.01, 1.01),
            "market_crash_exposure": (-0.01, 1.01),
            "basket_guard_scale": (-0.01, 1.01),
            "rebound_exit_market_exposure_max": (-0.01, 1.01),
            "rebound_exit_market_exposure_min": (-0.01, 1.01),
        }

        for key, values in invalid_values.items():
            for value in values if isinstance(values, tuple) else (values,):
                with self.subTest(key=key, value=value):
                    raw = valid_raw_config()
                    raw["baseline"][key] = value

                    with self.assertRaisesRegex(ValueError, key):
                        parse_evolution_config(raw)

    def test_rejects_invalid_remaining_baseline_controls(self):
        invalid_values = {
            "market_ma_window": 0,
            "market_risk_off_drawdown_20d": (-1.01, 0.01),
            "max_abs_daily_return": (0.0, 1.01),
            "initial_capital": 0.0,
            "basket_guard_return_20d_min": (-1.01, 1.01),
            "basket_guard_distance_ma60_min": (-1.01, 1.01),
            "rebound_exit_return": (-0.01, 1.01),
        }

        for key, values in invalid_values.items():
            for value in values if isinstance(values, tuple) else (values,):
                with self.subTest(key=key, value=value):
                    raw = valid_raw_config()
                    raw["baseline"][key] = value

                    with self.assertRaisesRegex(ValueError, key):
                        parse_evolution_config(raw)


class EvolutionAdapterTest(unittest.TestCase):
    def test_dry_run_writes_snapshot_without_changing_global_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "global" / "strong_pullback.json"
            run_dir = root / "run"
            snapshot = EvolutionState.initial(
                "v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00"
            )

            changed = persist_evolution_outcome(
                run_dir=run_dir,
                snapshot=snapshot,
                candidate_scores=[{"candidate_id": "c1", "status": "rejected"}],
                experiments=[{"experiment_id": "e1", "status": "rejected"}],
                decision_markdown="# 影子决定\n\n纸面研究，未更新全局状态。\n",
                dry_run=True,
                promote_shadow=False,
                state_path=state_path,
                expected_previous_fingerprint=None,
            )

            self.assertTrue((run_dir / "evolution_state_snapshot.json").exists())
            self.assertTrue((run_dir / "candidate_scores.csv").exists())
            self.assertTrue((run_dir / "experiments" / "e1.json").exists())
            self.assertTrue((run_dir / "shadow_decision.md").exists())
            self.assertFalse(state_path.exists())
            self.assertFalse(changed)

    def test_shadow_state_changes_only_with_both_explicit_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = EvolutionState.initial(
                "v1", {"leverage": 0.6}, now="2026-07-12T00:00:00+00:00"
            )
            state_path = root / "state" / "strong_pullback.json"

            for index, flags in enumerate(((True, True), (False, False))):
                changed = persist_evolution_outcome(
                    root / f"run-{index}", snapshot, [], [], "# decision\n",
                    dry_run=flags[0], promote_shadow=flags[1], state_path=state_path,
                    expected_previous_fingerprint=None,
                )
                self.assertFalse(changed)
                self.assertFalse(state_path.exists())

            changed = persist_evolution_outcome(
                root / "run-promote", snapshot, [], [], "# decision\n",
                dry_run=False, promote_shadow=True, state_path=state_path,
                expected_previous_fingerprint=None,
            )
            self.assertTrue(changed)
            self.assertTrue(state_path.exists())

    def test_runtime_state_path_cannot_target_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = EvolutionState.initial("v1", {"leverage": 0.6})

            with self.assertRaisesRegex(ValueError, "configs"):
                persist_evolution_outcome(
                    root / "run", snapshot, [], [], "# decision\n",
                    dry_run=False, promote_shadow=True,
                    state_path=root / "configs" / "evolution_strong_pullback.yaml",
                    expected_previous_fingerprint=None,
                )

    def test_schema_requires_real_ohlcv_and_amount_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel.csv"
            pd.DataFrame({"date": ["2025-01-01"], "symbol": ["000001"], "close": [10.0]}).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "Missing evolution input columns"):
                validate_input_schema(path)

    def test_price_bundle_is_physically_truncated_at_requested_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel.csv"
            rows = []
            for date in pd.date_range("2025-01-01", periods=4, freq="B"):
                rows.append({
                    "date": date, "symbol": "000001", "open": 10.0, "high": 11.0,
                    "low": 9.0, "close": 10.0, "volume": 1000.0, "amount": 10_000.0,
                })
            pd.DataFrame(rows).to_csv(path, index=False)

            bundle = load_price_bundle(
                path, pd.Timestamp("2025-01-03"), None,
                {**DEFAULT_STRATEGY_PARAMS, "max_abs_daily_return": 0.22},
            )

            self.assertEqual(bundle.close.index.max(), pd.Timestamp("2025-01-03"))

    def test_trial_artifacts_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            trial_dir = Path(tmp) / "trial"
            run = StrategyRun(
                equity=pd.DataFrame({"date": ["2025-01-02"], "equity": [1_010_000.0], "gross_return": [0.01], "cost": [0.0], "turnover": [0.1], "gross_exposure": [0.6]}),
                weights=pd.DataFrame({"date": ["2025-01-01"], "momentum_20": [1.0]}),
                trades=pd.DataFrame({"signal_date": ["2025-01-01"], "gross_return": [0.01]}),
                candidates=pd.DataFrame({"signal_date": ["2025-01-01"], "symbol": ["000001"]}),
            )

            write_trial_artifacts(trial_dir, run, {"validation": {"total_return": 0.01}}, {"status": "completed"})
            loaded = load_trial_artifacts(trial_dir)

            self.assertEqual(float(loaded.equity.loc[0, "equity"]), 1_010_000.0)
            self.assertEqual(str(loaded.candidates.loc[0, "symbol"]), "000001")

    def test_execute_strategy_trial_forwards_engine_boundary(self):
        bundle = StrategyRun(
            equity=pd.DataFrame(), weights=pd.DataFrame(), trades=pd.DataFrame(), candidates=pd.DataFrame()
        )
        price_bundle = type("PriceBundleStub", (), {
            "close": pd.DataFrame({"000001": [10.0]}),
            "open_px": pd.DataFrame({"000001": [10.0]}),
            "high": pd.DataFrame({"000001": [10.0]}),
            "low": pd.DataFrame({"000001": [10.0]}),
            "amount": pd.DataFrame({"000001": [1_000.0]}),
            "market_exposure": pd.Series([0.6]),
        })()
        params = {
            **DEFAULT_STRATEGY_PARAMS,
            "train_days": 111,
            "top_n": 3,
            "leverage": 0.75,
            "min_score": 0.42,
            "commission_bps": 1.2,
            "impact_bps": 0.8,
            "initial_capital": 123_456.0,
            "basket_guard_return_20d_min": 0.05,
            "basket_guard_distance_ma60_min": -0.02,
            "basket_guard_scale": 0.7,
            "rebound_exit_return": 0.12,
            "rebound_exit_scale": 0.25,
            "rebound_exit_market_exposure_max": 0.4,
            "rebound_exit_market_exposure_min": 0.1,
        }
        captured = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return (
                pd.DataFrame({"date": ["2025-01-01"], "equity": [1.0]}),
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            )

        with patch("run_strong_pullback_evolution.run_satellite_walk_forward", side_effect=spy):
            result = execute_strategy_trial(price_bundle, params)

        self.assertIsInstance(result, StrategyRun)
        self.assertIs(captured["close"], price_bundle.close)
        self.assertIs(captured["open_px"], price_bundle.open_px)
        self.assertIs(captured["high"], price_bundle.high)
        self.assertIs(captured["low"], price_bundle.low)
        self.assertIs(captured["amount"], price_bundle.amount)
        self.assertEqual(captured["train_days"], 111)
        self.assertEqual(captured["retrain_frequency"], params["retrain_frequency"])
        self.assertEqual(captured["top_n"], 3)
        self.assertEqual(captured["rebalance_frequency"], params["rebalance_frequency"])
        self.assertEqual(captured["max_position_weight"], params["max_position_weight"])
        self.assertEqual(captured["leverage"], 0.75)
        self.assertEqual(captured["min_score"], 0.42)
        self.assertEqual(captured["commission_bps"], 1.2)
        self.assertEqual(captured["impact_bps"], 0.8)
        self.assertEqual(captured["max_buy_open_gap"], params["max_buy_open_gap"])
        self.assertEqual(captured["limit_buffer"], params["limit_buffer"])
        self.assertIs(captured["market_exposure"], price_bundle.market_exposure)
        self.assertEqual(captured["initial_capital"], 123_456.0)
        self.assertEqual(captured["filter_kwargs"], {key: float(params[key]) for key in (
            "min_close", "min_avg_amount_20d", "min_pullback_5d", "max_pullback_5d",
            "min_prior_return_20", "min_prior_return_60", "min_return_20d",
            "min_return_60d", "min_distance_ma60", "max_intraday_return",
        )})
        self.assertEqual(captured["basket_guard_return_20d_min"], 0.05)
        self.assertEqual(captured["basket_guard_distance_ma60_min"], -0.02)
        self.assertEqual(captured["basket_guard_scale"], 0.7)
        self.assertEqual(captured["rebound_exit_return"], 0.12)
        self.assertEqual(captured["rebound_exit_scale"], 0.25)
        self.assertEqual(captured["rebound_exit_market_exposure_max"], 0.4)
        self.assertEqual(captured["rebound_exit_market_exposure_min"], 0.1)

    def test_execute_strategy_trial_rejects_empty_equity(self):
        price_bundle = type("PriceBundleStub", (), {
            "close": pd.DataFrame(), "open_px": pd.DataFrame(), "high": pd.DataFrame(),
            "low": pd.DataFrame(), "amount": pd.DataFrame(), "market_exposure": pd.Series(dtype=float),
        })()

        with patch(
            "run_strong_pullback_evolution.run_satellite_walk_forward",
            return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
        ):
            with self.assertRaisesRegex(ValueError, "Trial generated no equity rows"):
                execute_strategy_trial(price_bundle, DEFAULT_STRATEGY_PARAMS)


class EvolutionOrchestrationTest(unittest.TestCase):
    @staticmethod
    def _flat_bundle(end_date: pd.Timestamp) -> PriceBundle:
        dates = pd.date_range("2022-01-03", end=end_date, freq="B")
        frame = pd.DataFrame({"000001": 10.0}, index=dates)
        return PriceBundle(
            frame, frame.copy(), frame.copy(), frame.copy(), frame * 1_000_000,
            pd.Series(1.0, index=dates),
        )

    @staticmethod
    def _run_for_params(bundle: PriceBundle, params: dict[str, object]) -> StrategyRun:
        dates = bundle.close.index[bundle.close.index >= pd.Timestamp("2024-01-01")]
        daily = 0.0002 + float(params["leverage"]) * 0.0001
        equity = pd.DataFrame({
            "date": dates,
            "equity": 1_000_000.0 * (1.0 + daily) ** pd.RangeIndex(1, len(dates) + 1),
            "gross_return": daily,
            "cost": 0.0,
            "turnover": 0.10,
            "gross_exposure": float(params["leverage"]),
        })
        return StrategyRun(equity, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    def test_research_load_finishes_at_validation_end_before_test_load(self):
        config = parse_evolution_config(valid_raw_config())
        requested_ends: list[pd.Timestamp] = []

        def loader(data_path, end_date, benchmark_path, params):
            requested_ends.append(pd.Timestamp(end_date))
            return self._flat_bundle(pd.Timestamp(end_date))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "panel.csv"
            config_path = root / "config.yaml"
            data.write_text("evidence", encoding="utf-8")
            config_path.write_text("evidence", encoding="utf-8")
            outcome = run_evolution(
                config=config,
                data_path=data,
                config_path=config_path,
                benchmark_path=None,
                asof_date=pd.Timestamp("2026-07-09"),
                output_root=root / "runs",
                run_id="isolation-test",
                resume=False,
                bundle_loader=loader,
                trial_executor=self._run_for_params,
                git_commit="test-commit",
            )

        self.assertEqual(requested_ends[0], pd.Timestamp("2025-12-31"))
        self.assertTrue(all(value <= pd.Timestamp("2025-12-31") for value in requested_ends[:-1]))
        self.assertEqual(requested_ends[-1], pd.Timestamp("2026-07-09"))
        self.assertIn(outcome.test_status, {"ready_for_manual_review", "rollback_recommended"})

    def test_research_bundle_rejects_holdout_dates_in_any_member(self):
        config = parse_evolution_config(valid_raw_config())

        for late_member in ("open_px", "market_exposure"):
            with self.subTest(late_member=late_member), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = root / "panel.csv"
                config_path = root / "config.yaml"
                data.write_text("evidence", encoding="utf-8")
                config_path.write_text("evidence", encoding="utf-8")

                def loader(data_path, end_date, benchmark_path, params):
                    bundle = self._flat_bundle(pd.Timestamp(end_date))
                    late_date = pd.Timestamp(end_date) + pd.offsets.BDay(1)
                    if late_member == "open_px":
                        open_px = pd.concat([
                            bundle.open_px,
                            pd.DataFrame({"000001": [10.0]}, index=[late_date]),
                        ])
                        return PriceBundle(
                            bundle.close, open_px, bundle.high, bundle.low, bundle.amount,
                            bundle.market_exposure,
                        )
                    exposure = pd.concat([
                        bundle.market_exposure,
                        pd.Series([1.0], index=[late_date]),
                    ])
                    return PriceBundle(
                        bundle.close, bundle.open_px, bundle.high, bundle.low, bundle.amount, exposure
                    )

                with self.assertRaisesRegex(AssertionError, "Research bundle contains holdout dates"):
                    run_evolution(
                        config, data, config_path, None, pd.Timestamp("2026-07-09"), root / "runs",
                        f"late-{late_member}", False, loader, self._run_for_params, "test-commit",
                    )

    def test_resume_requires_exact_trial_evidence_and_params(self):
        state = {
            "status": "completed",
            "trial_id": "risk_075",
            "evidence_fingerprint": "abc",
            "params_hash": "def",
        }

        self.assertTrue(can_resume_trial(state, "abc", "def", "risk_075"))
        self.assertFalse(can_resume_trial(state, "changed", "def", "risk_075"))
        self.assertFalse(can_resume_trial(state, "abc", "changed", "risk_075"))
        self.assertFalse(can_resume_trial(state, "abc", "def", "other_trial"))

    def test_resume_rejects_run_level_evidence_mismatch_before_executing_trials(self):
        config = parse_evolution_config(valid_raw_config())
        calls: list[float] = []

        def executor(bundle, params):
            calls.append(float(params["leverage"]))
            return self._run_for_params(bundle, params)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "panel.csv"
            config_path = root / "config.yaml"
            data.write_text("evidence", encoding="utf-8")
            config_path.write_text("evidence", encoding="utf-8")
            kwargs = dict(
                config=config,
                data_path=data,
                config_path=config_path,
                benchmark_path=None,
                asof_date=pd.Timestamp("2026-07-09"),
                output_root=root / "runs",
                run_id="evidence-mismatch",
                bundle_loader=lambda data_path, end_date, benchmark_path, params: self._flat_bundle(
                    pd.Timestamp(end_date)
                ),
                trial_executor=executor,
                git_commit="test-commit",
            )
            run_evolution(resume=False, **kwargs)
            calls.clear()
            data.write_text("changed evidence", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Resume evidence does not match"):
                run_evolution(resume=True, **kwargs)

        self.assertEqual(calls, [])

    def test_resume_rejects_changed_benchmark_evidence_before_executing_trials(self):
        config = parse_evolution_config(valid_raw_config())
        calls: list[float] = []

        def executor(bundle, params):
            calls.append(float(params["leverage"]))
            return self._run_for_params(bundle, params)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "panel.csv"
            config_path = root / "config.yaml"
            benchmark = root / "benchmark.csv"
            data.write_text("evidence", encoding="utf-8")
            config_path.write_text("evidence", encoding="utf-8")
            benchmark.write_text("benchmark evidence", encoding="utf-8")
            kwargs = dict(
                config=config,
                data_path=data,
                config_path=config_path,
                benchmark_path=benchmark,
                asof_date=pd.Timestamp("2026-07-09"),
                output_root=root / "runs",
                run_id="benchmark-evidence-mismatch",
                bundle_loader=lambda data_path, end_date, benchmark_path, params: self._flat_bundle(
                    pd.Timestamp(end_date)
                ),
                trial_executor=executor,
                git_commit="test-commit",
            )
            run_evolution(resume=False, **kwargs)
            calls.clear()
            benchmark.write_text("changed benchmark evidence", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Resume evidence does not match"):
                run_evolution(resume=True, **kwargs)

        self.assertEqual(calls, [])

    def test_resume_recomputes_trial_when_cached_metrics_or_csv_are_invalid(self):
        config = parse_evolution_config(valid_raw_config())
        mutations = ("missing_metric", "non_finite_metric", "missing_csv", "corrupt_csv")

        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = root / "panel.csv"
                config_path = root / "config.yaml"
                data.write_text("evidence", encoding="utf-8")
                config_path.write_text("evidence", encoding="utf-8")
                calls: list[tuple[pd.Timestamp, float]] = []

                def executor(bundle, params):
                    calls.append((bundle.close.index.max(), float(params["leverage"])))
                    return self._run_for_params(bundle, params)

                kwargs = dict(
                    config=config,
                    data_path=data,
                    config_path=config_path,
                    benchmark_path=None,
                    asof_date=pd.Timestamp("2026-07-09"),
                    output_root=root / "runs",
                    run_id=f"invalid-cache-{mutation}",
                    bundle_loader=lambda data_path, end_date, benchmark_path, params: self._flat_bundle(
                        pd.Timestamp(end_date)
                    ),
                    trial_executor=executor,
                    git_commit="test-commit",
                )
                outcome = run_evolution(resume=False, **kwargs)
                trial_dir = outcome.run_dir / "trials" / "baseline"
                if mutation == "missing_metric":
                    metrics = json.loads((trial_dir / "metrics.json").read_text(encoding="utf-8"))
                    del metrics["validation"]["worst_rolling_return"]
                    (trial_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                elif mutation == "non_finite_metric":
                    metrics = json.loads((trial_dir / "metrics.json").read_text(encoding="utf-8"))
                    metrics["train"]["annualized_return"] = "not-a-number"
                    (trial_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                elif mutation == "missing_csv":
                    (trial_dir / "trade_audit.csv").unlink()
                else:
                    (trial_dir / "equity_curve.csv").write_text("not,date\nvalid,rows\n", encoding="utf-8")

                calls.clear()
                run_evolution(resume=True, **kwargs)

                research_baseline_calls = [
                    call for call in calls
                    if call[0] <= config.periods.validation_end and call[1] == config.baseline["leverage"]
                ]
                self.assertEqual(len(research_baseline_calls), 1)

    def test_successful_run_writes_manifest_versioned_artifacts_and_chinese_report(self):
        config = parse_evolution_config(valid_raw_config())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "panel.csv"
            config_path = root / "config.yaml"
            data.write_text("evidence", encoding="utf-8")
            config_path.write_text("evidence", encoding="utf-8")
            outcome = run_evolution(
                config, data, config_path, None, pd.Timestamp("2026-07-09"), root / "runs",
                "successful-run", False,
                lambda data_path, end_date, benchmark_path, params: self._flat_bundle(pd.Timestamp(end_date)),
                self._run_for_params, "test-commit",
            )
            run_dir = outcome.run_dir
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["status"], "success")
            self.assertTrue((root / "runs" / "latest.json").exists())
            self.assertTrue((run_dir / "resolved_config.yaml").exists())
            self.assertTrue((run_dir / "trials.csv").exists())
            self.assertTrue((run_dir / "rounds.csv").exists())
            self.assertTrue((run_dir / "test_comparison.csv").exists())
            folds = json.loads((run_dir / "folds.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(folds), config.evolution_core.min_folds)
            scores = pd.read_csv(run_dir / "candidate_scores.csv")
            self.assertEqual(set(scores["candidate_id"]), {"risk_075", "risk_090"})
            self.assertIn("failed_gates", scores.columns)
            self.assertEqual(len(list((run_dir / "experiments").glob("*.json"))), 2)
            for experiment_path in (run_dir / "experiments").glob("*.json"):
                json.loads(
                    experiment_path.read_text(encoding="utf-8"),
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-standard JSON constant: {value}")
                    ),
                )
            snapshot = json.loads(
                (run_dir / "evolution_state_snapshot.json").read_text(encoding="utf-8")
            )
            self.assertEqual(snapshot["last_data_fingerprint"], manifest["evidence_fingerprint"])
            self.assertIn("纸面", (run_dir / "shadow_decision.md").read_text(encoding="utf-8"))
            self.assertFalse(manifest["global_state_changed"])
            self.assertTrue((run_dir / "final" / "baseline" / "metrics.json").exists())
            self.assertTrue((run_dir / "final" / "champion" / "metrics.json").exists())
            summary = Path(manifest["summary"])
            self.assertIn("风险提示", summary.read_text(encoding="utf-8"))

    def test_failed_run_does_not_update_latest_pointer_and_marks_manifest_failed(self):
        config = parse_evolution_config(valid_raw_config())

        def broken_executor(bundle, params):
            raise RuntimeError("trial failed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "panel.csv"
            config_path = root / "config.yaml"
            data.write_text("evidence", encoding="utf-8")
            config_path.write_text("evidence", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Baseline trial failed"):
                run_evolution(
                    config, data, config_path, None, pd.Timestamp("2026-07-09"), root / "runs",
                    "failed-run", False,
                    lambda data_path, end_date, benchmark_path, params: self._flat_bundle(pd.Timestamp(end_date)),
                    broken_executor, "test-commit",
                )
            self.assertFalse((root / "runs" / "latest.json").exists())
            manifest = json.loads(
                (root / "runs" / "failed-run" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertFalse(manifest["global_state_changed"])
            run_dir = root / "runs" / "failed-run"
            for relative in (
                "folds.json",
                "candidate_scores.csv",
                "evolution_state_snapshot.json",
                "shadow_decision.md",
            ):
                self.assertTrue((run_dir / relative).exists(), relative)
            self.assertTrue((run_dir / "experiments").is_dir())

    def test_all_candidate_failures_preserve_every_candidate_experiment(self):
        config = parse_evolution_config(valid_raw_config())

        def candidate_failure_executor(bundle, params):
            if float(params["leverage"]) != float(config.baseline["leverage"]):
                raise RuntimeError("candidate failed")
            return self._run_for_params(bundle, params)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "panel.csv"
            config_path = root / "config.yaml"
            data.write_text("evidence", encoding="utf-8")
            config_path.write_text("evidence", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "All candidates failed"):
                run_evolution(
                    config, data, config_path, None, pd.Timestamp("2026-07-09"),
                    root / "runs", "candidate-failures", False,
                    lambda data_path, end_date, benchmark_path, params: self._flat_bundle(
                        pd.Timestamp(end_date)
                    ),
                    candidate_failure_executor, "test-commit",
                )

            run_dir = root / "runs" / "candidate-failures"
            scores = pd.read_csv(run_dir / "candidate_scores.csv")
            self.assertEqual(set(scores["candidate_id"]), {"risk_075", "risk_090"})
            self.assertEqual(set(scores["status"]), {"trial_error"})
            experiments = list((run_dir / "experiments").glob("*.json"))
            self.assertEqual(len(experiments), 2)

    def test_run_refuses_output_root_under_configs_before_writing_yaml(self):
        config = parse_evolution_config(valid_raw_config())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "panel.csv"
            config_path = root / "config.yaml"
            data.write_text("evidence", encoding="utf-8")
            config_path.write_text("evidence", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "configs"):
                run_evolution(
                    config, data, config_path, None, pd.Timestamp("2026-07-09"),
                    root / "configs", "unsafe-run", False,
                    lambda data_path, end_date, benchmark_path, params: self._flat_bundle(
                        pd.Timestamp(end_date)
                    ),
                    self._run_for_params, "test-commit",
                )

            self.assertFalse((root / "configs").exists())

    def test_holdout_rollback_preserves_artifacts_but_does_not_publish(self):
        raw = valid_raw_config()
        raw["selection"]["min_annualized_return_delta"] = 0.0
        config = parse_evolution_config(raw)

        def rollback_executor(bundle, params):
            run = self._run_for_params(bundle, params)
            if bundle.close.index.max() > config.periods.validation_end:
                daily = 0.0002 if float(params["leverage"]) == config.baseline["leverage"] else -0.0002
                equity = run.equity.copy()
                equity["equity"] = 1_000_000.0 * (1.0 + daily) ** pd.RangeIndex(1, len(equity) + 1)
                equity["gross_return"] = daily
                return StrategyRun(equity, run.weights, run.trades, run.candidates)
            return run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "panel.csv"
            config_path = root / "config.yaml"
            data.write_text("evidence", encoding="utf-8")
            config_path.write_text("evidence", encoding="utf-8")
            common = dict(
                config=config,
                data_path=data,
                config_path=config_path,
                benchmark_path=None,
                asof_date=pd.Timestamp("2026-07-09"),
                output_root=root / "runs",
                bundle_loader=lambda data_path, end_date, benchmark_path, params: self._flat_bundle(
                    pd.Timestamp(end_date)
                ),
                git_commit="test-commit",
            )
            run_evolution(
                run_id="published-run",
                resume=False,
                trial_executor=self._run_for_params,
                **common,
            )
            latest_before = (root / "runs" / "latest.json").read_text(encoding="utf-8")
            registry_before = (root / "runs" / "evolution_registry.jsonl").read_text(encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Holdout test rollback_recommended"):
                run_evolution(
                    run_id="rollback-run",
                    resume=False,
                    trial_executor=rollback_executor,
                    **common,
                )

            run_dir = root / "runs" / "rollback-run"
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["test_status"], "rollback_recommended")
            self.assertTrue((run_dir / "test_comparison.csv").exists())
            self.assertTrue((run_dir / "final" / "baseline" / "metrics.json").exists())
            self.assertTrue((run_dir / "final" / "champion" / "metrics.json").exists())
            self.assertEqual((root / "runs" / "latest.json").read_text(encoding="utf-8"), latest_before)
            self.assertEqual(
                (root / "runs" / "evolution_registry.jsonl").read_text(encoding="utf-8"), registry_before
            )

    def test_holdout_warning_preserves_artifacts_but_does_not_publish(self):
        published_config = parse_evolution_config(valid_raw_config())
        warning_raw = valid_raw_config()
        warning_raw["selection"].update({
            "min_annualized_return_delta": 0.0,
            "min_test_days": 1_000,
        })
        warning_config = parse_evolution_config(warning_raw)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "panel.csv"
            config_path = root / "config.yaml"
            data.write_text("evidence", encoding="utf-8")
            config_path.write_text("evidence", encoding="utf-8")
            common = dict(
                data_path=data,
                config_path=config_path,
                benchmark_path=None,
                asof_date=pd.Timestamp("2026-07-09"),
                output_root=root / "runs",
                bundle_loader=lambda data_path, end_date, benchmark_path, params: self._flat_bundle(
                    pd.Timestamp(end_date)
                ),
                trial_executor=self._run_for_params,
                git_commit="test-commit",
            )
            run_evolution(
                config=published_config,
                run_id="published-run",
                resume=False,
                **common,
            )
            latest_before = (root / "runs" / "latest.json").read_text(encoding="utf-8")
            registry_before = (root / "runs" / "evolution_registry.jsonl").read_text(encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Holdout test test_warning"):
                run_evolution(
                    config=warning_config,
                    run_id="warning-run",
                    resume=False,
                    **common,
                )

            run_dir = root / "runs" / "warning-run"
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["test_status"], "test_warning")
            self.assertTrue((run_dir / "test_comparison.csv").exists())
            self.assertTrue((run_dir / "final" / "baseline" / "metrics.json").exists())
            self.assertTrue((run_dir / "final" / "champion" / "metrics.json").exists())
            self.assertEqual((root / "runs" / "latest.json").read_text(encoding="utf-8"), latest_before)
            self.assertEqual(
                (root / "runs" / "evolution_registry.jsonl").read_text(encoding="utf-8"), registry_before
            )


class EvolutionDecisionTest(unittest.TestCase):
    def test_build_trading_folds_uses_actual_index_positions(self):
        dates = pd.DatetimeIndex(
            ["2026-01-02", "2026-01-05", "2026-01-08", "2026-01-09", "2026-01-12", "2026-01-16"]
        )

        folds = build_trading_folds(
            dates, train_days=2, validation_days=2, test_days=1, step_days=1
        )

        self.assertEqual(
            folds[0].train_dates,
            (pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")),
        )
        self.assertEqual(
            folds[0].validation_dates,
            (pd.Timestamp("2026-01-08"), pd.Timestamp("2026-01-09")),
        )
        self.assertEqual(folds[0].test_dates, (pd.Timestamp("2026-01-12"),))

    def test_build_trading_folds_rejects_duplicate_dates(self):
        dates = pd.DatetimeIndex(["2026-01-02", "2026-01-02", "2026-01-05"])

        with pytest.raises(ValueError, match="unique"):
            build_trading_folds(dates, train_days=1, validation_days=1, test_days=1, step_days=1)

    def test_calculate_fold_metrics_uses_realized_symbol_pnl_concentration(self):
        equity = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=3),
                "gross_return": [0.0, 0.02, 0.01],
                "cost": [0.0, 0.0, 0.0],
                "turnover": [0.0, 0.2, 0.1],
                "gross_exposure": [0.0, 0.6, 0.6],
            }
        )
        trades = pd.DataFrame(
            {
                "symbol_contributions_json": [
                    '{"000001": 0.012, "000002": 0.008}',
                    '{"000001": 0.006, "000002": 0.004}',
                ]
            }
        )

        metrics = calculate_fold_metrics(equity, trades, fold_id="f1")

        self.assertAlmostEqual(metrics.pnl_concentration, 0.60)

    def test_fold_runner_never_receives_rows_after_test_end(self):
        dates = pd.date_range("2026-01-01", periods=6)
        panel = pd.DataFrame({"date": dates, "symbol": "000001"})
        folds = (
            TradingFold("f1", tuple(dates[:2]), tuple(dates[2:4]), (dates[4],)),
            TradingFold("f2", tuple(dates[:3]), tuple(dates[3:5]), (dates[5],)),
        )
        seen: list[pd.Timestamp] = []

        results = run_strong_pullback_folds(
            panel,
            folds,
            lambda sliced, params: seen.append(sliced["date"].max()) or {"params": params},
            {"leverage": 0.6},
        )

        self.assertEqual(seen, [fold.test_end for fold in folds])
        self.assertEqual([fold.fold_id for fold, _ in results], ["f1", "f2"])

    def test_segment_metrics_compound_net_returns_inside_requested_period(self):
        equity = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=4, freq="B"),
                "gross_return": [0.50, 0.10, -0.05, 0.02],
                "cost": [0.0, 0.01, 0.0, 0.0],
                "turnover": [0.0, 0.2, 0.1, 0.0],
                "gross_exposure": [0.0, 0.6, 0.6, 0.6],
            }
        )

        metrics = calculate_segment_metrics(
            equity, "2025-01-02", "2025-01-06", rolling_window_days=2
        )

        self.assertAlmostEqual(metrics["total_return"], 1.09 * 0.95 * 1.02 - 1.0)
        self.assertEqual(metrics["trade_days"], 3)
        self.assertAlmostEqual(metrics["avg_turnover"], 0.1)
        self.assertEqual(metrics["rolling_window_count"], 1)

    def test_promotion_accepts_exact_drawdown_and_sharpe_boundaries(self):
        rules = parse_evolution_config(valid_raw_config()).selection
        incumbent = {
            "annualized_return": 0.10, "max_drawdown": -0.30, "sharpe_like": 0.50,
            "avg_turnover": 0.10, "trade_days": 200, "negative_window_rate": 0.20,
            "rolling_window_count": 1,
        }
        candidate = {
            "annualized_return": 0.11, "max_drawdown": -0.40, "sharpe_like": 0.40,
            "avg_turnover": 0.15, "trade_days": 200, "negative_window_rate": 0.60,
            "rolling_window_count": 1,
        }

        decision = evaluate_promotion(candidate, incumbent, rules)

        self.assertTrue(decision.eligible)
        self.assertEqual(decision.reasons, ())

    def test_group_keeps_incumbent_when_no_candidate_passes(self):
        rules = parse_evolution_config(valid_raw_config()).selection
        incumbent = {
            "annualized_return": 0.10, "max_drawdown": -0.20, "sharpe_like": 0.50,
            "avg_turnover": 0.10, "trade_days": 200, "negative_window_rate": 0.20,
            "rolling_window_count": 1,
        }
        candidate = {**incumbent, "annualized_return": 0.105}

        winner, decisions = choose_group_winner(
            "incumbent", incumbent, (("weak_gain", candidate),), rules
        )

        self.assertEqual(winner, "incumbent")
        self.assertFalse(decisions[0].promotion.eligible)

    def test_promotion_rejects_candidate_without_completed_rolling_windows(self):
        rules = parse_evolution_config(valid_raw_config()).selection
        incumbent = {
            "annualized_return": 0.10, "max_drawdown": -0.20, "sharpe_like": 0.50,
            "avg_turnover": 0.10, "trade_days": 200, "negative_window_rate": 0.20,
            "rolling_window_count": 1,
        }
        candidate = {
            "annualized_return": 0.11, "max_drawdown": -0.20, "sharpe_like": 0.50,
            "avg_turnover": 0.10, "trade_days": 200, "negative_window_rate": 0.0,
            "rolling_window_count": 0,
        }

        decision = evaluate_promotion(candidate, incumbent, rules)

        self.assertFalse(decision.eligible)
        self.assertIn("滚动窗口", "；".join(decision.reasons))

    def test_holdout_recommends_rollback_when_champion_loses(self):
        rules = parse_evolution_config(valid_raw_config()).selection
        baseline = {"total_return": 0.20, "max_drawdown": -0.20, "sharpe_like": 0.50, "trade_days": 100}
        champion = {"total_return": 0.10, "max_drawdown": -0.25, "sharpe_like": 0.45, "trade_days": 100}

        status, reason = assess_test_result(baseline, champion, rules)

        self.assertEqual(status, "rollback_recommended")
        self.assertIn("收益", reason)

    def test_holdout_warns_when_required_metrics_are_missing_or_non_finite(self):
        rules = parse_evolution_config(valid_raw_config()).selection
        baseline = {"total_return": 0.20, "max_drawdown": -0.20, "sharpe_like": 0.50, "trade_days": 100}
        champion = {"total_return": 0.20, "max_drawdown": -0.20, "sharpe_like": 0.50, "trade_days": 100}

        for missing_key in ("total_return", "max_drawdown", "sharpe_like", "trade_days"):
            with self.subTest(case=f"missing {missing_key}"):
                metrics = dict(champion)
                del metrics[missing_key]
                status, _ = assess_test_result(baseline, metrics, rules)
                self.assertEqual(status, "test_warning")

        for non_finite_key in ("total_return", "max_drawdown", "sharpe_like"):
            with self.subTest(case=f"non-finite {non_finite_key}"):
                metrics = dict(champion)
                metrics[non_finite_key] = float("nan")
                status, _ = assess_test_result(baseline, metrics, rules)
                self.assertEqual(status, "test_warning")

        metrics = dict(champion)
        metrics["trade_days"] = float("nan")
        status, _ = assess_test_result(baseline, metrics, rules)
        self.assertEqual(status, "test_warning")
