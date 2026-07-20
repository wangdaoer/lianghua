import tempfile
import unittest
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

from run_daily_model_pipeline import (
    PipelineConfig,
    PipelineStep,
    PipelineStepExecutionError,
    _trend_ignition_shadow_state,
    append_daily_run_state,
    attach_daily_run_card,
    build_daily_pipeline_steps,
    build_daily_run_state,
    build_marketlens_export_step,
    default_stability_inputs_available,
    discover_base_panel,
    effective_daily_start,
    ensure_fetch_status_ok,
    execute_pipeline,
    factor_decay_monitor_status,
    latest_daily_date,
    latest_shadow_account_review,
    metrics_path_for_date,
    names_source_for_date,
    parse_args,
    pipeline_step_dependencies,
    run_steps,
)


class DailyModelPipelineTest(unittest.TestCase):
    def test_pipeline_dependencies_keep_database_sync_terminal(self):
        steps = [
            PipelineStep("update_panel", ("python", "panel.py")),
            PipelineStep("early_pattern_watchlist", ("python", "early.py")),
            PipelineStep("research_database_sync", ("python", "sync.py")),
        ]

        dependencies = pipeline_step_dependencies(steps)

        self.assertEqual(dependencies["update_panel"], ())
        self.assertEqual(dependencies["early_pattern_watchlist"], ("update_panel",))
        self.assertEqual(
            dependencies["research_database_sync"],
            ("update_panel", "early_pattern_watchlist"),
        )

    def test_run_steps_parallelizes_ready_steps_and_honors_dependencies(self):
        steps = [
            PipelineStep("benchmark_refresh", ("python", "benchmark.py")),
            PipelineStep("update_panel", ("python", "panel.py")),
            PipelineStep("trend_state", ("python", "trend.py")),
        ]
        lock = threading.Lock()
        active = 0
        maximum_active = 0
        completed: set[str] = set()

        def fake_run(command, **kwargs):
            nonlocal active, maximum_active
            name = Path(command[1]).stem
            with lock:
                if name == "trend":
                    self.assertIn("panel", completed)
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.03)
            with lock:
                active -= 1
                completed.add(name)
            return subprocess.CompletedProcess(command, 0)

        with patch("run_daily_model_pipeline.subprocess.run", side_effect=fake_run):
            executions = run_steps(
                steps,
                Path("C:/model"),
                max_parallel_steps=2,
            )

        self.assertEqual(maximum_active, 2)
        self.assertEqual([item["name"] for item in executions], [step.name for step in steps])
        self.assertTrue(all(item["status"] == "success" for item in executions))
        self.assertTrue(all(float(item["duration_seconds"]) > 0 for item in executions))

    def test_run_steps_failure_preserves_completed_execution_records(self):
        steps = [PipelineStep("benchmark_refresh", ("python", "benchmark.py"))]
        with patch(
            "run_daily_model_pipeline.subprocess.run",
            side_effect=subprocess.CalledProcessError(7, ["python", "benchmark.py"]),
        ):
            with self.assertRaises(PipelineStepExecutionError) as raised:
                run_steps(steps, Path("C:/model"), max_parallel_steps=2)

        self.assertEqual(raised.exception.returncode, 7)
        self.assertEqual(len(raised.exception.completed_executions), 1)
        self.assertEqual(raised.exception.completed_executions[0]["status"], "failed")

    def test_discover_base_panel_uses_latest_panel_before_asof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "data_panel_history_main_chinext_20220101_20260712.csv"
            latest_base = root / "data_panel_history_main_chinext_20220101_20260713.csv"
            same_day = root / "data_panel_history_main_chinext_20220101_20260714.csv"
            for path in (older, latest_base, same_day):
                path.write_text("date,symbol\n", encoding="utf-8")

            selected = discover_base_panel(root, "2026-07-14")

        self.assertEqual(selected, latest_base)

    def test_discover_base_panel_prefers_parquet_for_same_end_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / "data_panel_history_main_chinext_20220101_20260713.csv"
            parquet = root / "data_panel_history_main_chinext_20220101_20260713.parquet"
            csv.touch()
            parquet.touch()

            selected = discover_base_panel(root, "2026-07-14")

        self.assertEqual(selected, parquet)

    def test_effective_daily_start_advances_past_base_panel(self):
        config = PipelineConfig(
            asof_date="2026-07-14",
            base_panel=Path("data_panel_history_main_chinext_20220101_20260713.csv"),
            daily_start="2026-06-22",
        )
        self.assertEqual(effective_daily_start(config), "2026-07-14")

    def test_default_stability_inputs_are_optional_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(default_stability_inputs_available(Path(tmp)))

    def test_latest_daily_date_prefers_newest_normalized_daily_file(self):
        daily_dir = Path("D:/codex/daily-market-data/ths_exports/normalized")
        files = [
            daily_dir / "ths_hs_a_share_2026-06-22.csv",
            daily_dir / "ths_hs_a_share_2026-06-29.csv",
            daily_dir / "not_market_data.csv",
        ]

        self.assertEqual(latest_daily_date(files), "2026-06-29")

    def test_pipeline_runs_overlay_before_chinese_report(self):
        config = PipelineConfig(
            asof_date="2026-06-29",
            python_exe="python",
            project_root=Path("C:/model"),
            output_root=Path("C:/model/outputs/high_return_v2"),
        )

        steps = build_daily_pipeline_steps(config)
        names = [step.name for step in steps]

        self.assertEqual(names[:2], ["benchmark_refresh", "update_panel"])
        benchmark_step = steps[0]
        benchmark_args = " ".join(str(part) for part in benchmark_step.command)
        self.assertIn("update_benchmark_510300.py", benchmark_args)
        self.assertIn("benchmark_refresh_status_20260629.json", benchmark_args)
        self.assertIn("regime_shadow_compare", names)
        shadow_step = steps[names.index("regime_shadow_compare")]
        shadow_args = " ".join(str(part) for part in shadow_step.command)
        self.assertIn("run_daily_regime_shadow_compare.py", shadow_args)
        self.assertIn("regime_shadow_compare_20260629", shadow_args)
        self.assertIn(
            "daily-market-data/benchmarks/510300.csv",
            shadow_args.replace("\\", "/"),
        )
        self.assertIn("regime_shadow_tracking", names)
        tracking_step = steps[names.index("regime_shadow_tracking")]
        tracking_args = " ".join(str(part) for part in tracking_step.command)
        self.assertEqual(names[names.index("regime_shadow_compare") + 1], "regime_shadow_tracking")
        self.assertIn("update_regime_shadow_tracking.py", tracking_args)
        self.assertIn("regime_shadow_tracking.csv", tracking_args)
        self.assertIn("regime_shadow_tracking_summary.json", tracking_args)
        self.assertIn("regime_shadow_tracking_report.md", tracking_args)
        self.assertIn("--target-days 20", tracking_args)
        self.assertIn("train_breadth_guard_challenger", names)
        breadth_step = steps[names.index("train_breadth_guard_challenger")]
        breadth_args = " ".join(str(part) for part in breadth_step.command)
        self.assertIn("train_next_open_rank_model.py", breadth_args)
        self.assertIn("next_open_rank_model_breadth_guard_bench20260629", breadth_args)
        self.assertIn("--breadth-filter", breadth_args)
        self.assertIn("--breadth-threshold 0.45", breadth_args)

        self.assertEqual(
            names[-14:],
            [
                "personal_overlay",
                "early_pattern_watchlist",
                "main_net_volume_shadow",
                "institutional_accumulation_shadow",
                "institutional_accumulation_tracking",
                "czsc_structure_shadow",
                "hidden_accumulation_tracking",
                "factor_decay_monitor",
                "daily_chinese_report",
                "merged_daily_outputs",
                "strategy_family_forward_report",
                "strategy_arena",
                "strategy_stability_report",
                "research_database_sync",
            ],
        )
        watch_step = steps[names.index("early_pattern_watchlist")]
        watch_args = " ".join(str(part) for part in watch_step.command)
        self.assertIn("early_pattern_watchlist.py", watch_args)
        flow_step = steps[names.index("main_net_volume_shadow")]
        flow_args = " ".join(str(part) for part in flow_step.command)
        self.assertIn("main_net_volume_shadow.py", flow_args)
        self.assertIn("main_net_volume_shadow_20260629.csv", flow_args)
        self.assertIn("early_pattern_watchlist_20260629.csv", flow_args)
        self.assertIn("--minimum-history 5", flow_args)
        institutional_step = steps[names.index("institutional_accumulation_shadow")]
        institutional_args = " ".join(str(part) for part in institutional_step.command)
        self.assertIn("institutional_accumulation_shadow.py", institutional_args)
        self.assertIn("institutional_accumulation_shadow_20260629.csv", institutional_args)
        self.assertIn("configs/institutional_accumulation_shadow.yaml", institutional_args.replace("\\", "/"))
        institutional_tracking_step = steps[names.index("institutional_accumulation_tracking")]
        institutional_tracking_args = " ".join(
            str(part) for part in institutional_tracking_step.command
        )
        self.assertIn("track_institutional_accumulation_shadow.py", institutional_tracking_args)
        self.assertIn("institutional_accumulation_tracking_20260629.csv", institutional_tracking_args)
        self.assertIn("--horizons 1,3,5,10", institutional_tracking_args)
        czsc_step = steps[names.index("czsc_structure_shadow")]
        czsc_args = " ".join(str(part) for part in czsc_step.command)
        self.assertIn("czsc_structure_shadow.py", czsc_args)
        self.assertIn("czsc_structure_shadow_20260629.csv", czsc_args)
        self.assertIn("rank_model_candidates_trend_gated_personal_overlay_20260629.csv", czsc_args)
        self.assertIn("early_pattern_watchlist_20260629.csv", czsc_args)
        self.assertIn("--min-bars 100", czsc_args)
        tracking_step = steps[names.index("hidden_accumulation_tracking")]
        tracking_args = " ".join(str(part) for part in tracking_step.command)
        self.assertIn("track_hidden_accumulation_watch.py", tracking_args)
        self.assertIn("hidden_accumulation_trade_watch_tracking_20260629.csv", tracking_args)
        monitor_step = steps[names.index("factor_decay_monitor")]
        monitor_args = " ".join(str(part) for part in monitor_step.command)
        self.assertIn("monitor_factor_decay.py", monitor_args)
        self.assertIn("data_panel_history_main_chinext_20220101_20260629.parquet", monitor_args)
        self.assertIn(
            "configs/factor_replacement_preregistration.json",
            monitor_args.replace("\\", "/"),
        )
        overlay_step = steps[names.index("personal_overlay")]
        overlay_args = " ".join(str(part) for part in overlay_step.command)
        self.assertIn("--names-source", overlay_args)
        self.assertIn("ths_hs_a_share_2026-06-29.csv", overlay_args)
        report_step = steps[-6]
        report_args = " ".join(str(part) for part in report_step.command)
        self.assertIn("build_daily_personal_overlay_report.py", report_args)
        self.assertIn("rank_model_candidates_trend_gated_personal_overlay_20260629.csv", report_args)
        self.assertIn("early_pattern_watchlist_20260629.csv", report_args)
        self.assertIn("--names-source", report_args)
        merged_step = steps[-5]
        merged_args = " ".join(str(part) for part in merged_step.command)
        self.assertIn("merged_daily_outputs.py", merged_args)
        self.assertIn("trend_state_20260629", merged_args)
        self.assertIn("rank_model_candidates_trend_gated_bench20260629_20260629.csv", merged_args)
        self.assertIn("rank_model_candidates_trend_gated_personal_overlay_20260629.csv", merged_args)
        self.assertIn("early_pattern_watchlist_20260629.csv", merged_args)
        family_step = steps[-4]
        family_args = " ".join(str(part) for part in family_step.command)
        self.assertIn("evaluate_strategy_family_forward_returns.py", family_args)
        self.assertIn("--horizons 1,3,5,10", family_args)
        self.assertIn("--token 20260629", family_args)
        arena_step = steps[-3]
        arena_args = " ".join(str(part) for part in arena_step.command)
        self.assertIn("build_strategy_arena_report.py", arena_args)
        self.assertIn("strategy_arena_history.csv", arena_args)
        self.assertIn("next_open_rank_model_breadth_guard_bench20260629", arena_args)
        self.assertIn("regime_shadow_compare_20260629", arena_args)
        self.assertIn("strategy_family_health_20260629.csv", arena_args)
        stability_step = steps[-2]
        stability_args = " ".join(str(part) for part in stability_step.command)
        self.assertIn("summarize_core_risk_filter_stability.py", stability_args)
        self.assertIn("core_risk_filter_finalist_stability_20260629", stability_args)
        database_step = steps[-1]
        database_args = " ".join(database_step.command)
        self.assertIn("sync_daily_research_database.py", database_args)
        self.assertIn("research_database_sync_20260629.json", database_args)

    def test_pipeline_can_skip_research_database_sync(self):
        config = PipelineConfig(
            asof_date="2026-06-29",
            python_exe="python",
            project_root=Path("C:/model"),
            include_research_db_sync=False,
        )
        names = [step.name for step in build_daily_pipeline_steps(config)]
        self.assertNotIn("research_database_sync", names)

    def test_marketlens_export_step_is_a_post_state_publication_command(self):
        config = PipelineConfig(
            asof_date="2026-06-29",
            python_exe="python",
            project_root=Path("C:/model"),
            output_root=Path("outputs/high_return_v2"),
            include_marketlens_export=True,
            marketlens_dashboard_output=Path("C:/dashboard/data/quant-model3-latest.json"),
        )

        step = build_marketlens_export_step(config)

        self.assertIsNotNone(step)
        self.assertEqual(step.name, "marketlens_export")
        command = " ".join(step.command)
        self.assertIn("export_marketlens_model3_feed.py", command)
        self.assertIn("--state-record", command)
        self.assertIn(".marketlens_state_20260629.json", command)
        self.assertIn("--require-complete-inputs", command)
        self.assertIn("C:\\dashboard\\data\\quant-model3-latest.json", command)

    def test_trend_ignition_shadow_is_opt_in_and_runs_after_merged_outputs(self):
        default_config = PipelineConfig(
            asof_date="2026-06-29",
            python_exe="python",
            project_root=Path("C:/model"),
        )
        self.assertNotIn(
            "trend_ignition_shadow",
            [step.name for step in build_daily_pipeline_steps(default_config)],
        )

        config = PipelineConfig(
            asof_date="2026-06-29",
            python_exe="python",
            project_root=Path("C:/model"),
            include_trend_ignition_shadow=True,
        )
        steps = build_daily_pipeline_steps(config)
        names = [step.name for step in steps]
        shadow_index = names.index("trend_ignition_shadow")
        self.assertEqual(names[shadow_index - 1], "merged_daily_outputs")
        command = " ".join(str(part) for part in steps[shadow_index].command)
        self.assertIn("score_trend_ignition_shadow.py", command)
        self.assertIn("trend_ignition_shadow_20260629", command)
        self.assertIn("scorer_v3_shortlist_exploratory", command)
        self.assertIn("--start 2026-06-29 --end 2026-06-29", command)
        self.assertIn("--selection-status exploratory_posthoc", command)

    def test_pipeline_can_skip_factor_decay_monitor(self):
        config = PipelineConfig(
            asof_date="2026-06-29",
            python_exe="python",
            project_root=Path("C:/model"),
            include_factor_decay_monitor=False,
        )
        names = [step.name for step in build_daily_pipeline_steps(config)]
        self.assertNotIn("factor_decay_monitor", names)

    def test_parse_args_accepts_skip_research_database_sync_flag(self):
        with patch.object(sys, "argv", ["run_daily_model_pipeline.py", "--skip-research-db-sync"]):
            args = parse_args()
        self.assertTrue(args.skip_research_db_sync)

    def test_parse_args_accepts_skip_marketlens_export_flag(self):
        with patch.object(sys, "argv", ["run_daily_model_pipeline.py", "--skip-marketlens-export"]):
            args = parse_args()
        self.assertTrue(args.skip_marketlens_export)

    def test_parse_args_defaults_to_two_parallel_steps(self):
        with patch.object(sys, "argv", ["run_daily_model_pipeline.py"]):
            args = parse_args()
        self.assertEqual(args.max_parallel_steps, 2)

    def test_parse_args_accepts_skip_factor_decay_monitor_flag(self):
        with patch.object(sys, "argv", ["run_daily_model_pipeline.py", "--skip-factor-decay-monitor"]):
            args = parse_args()
        self.assertTrue(args.skip_factor_decay_monitor)

    def test_parse_args_accepts_skip_strategy_arena_flag(self):
        with patch.object(sys, "argv", ["run_daily_model_pipeline.py", "--skip-strategy-arena"]):
            args = parse_args()
        self.assertTrue(args.skip_strategy_arena)

    def test_parse_args_supports_auto_base_panel_and_explicit_stability(self):
        with patch.object(
            sys,
            "argv",
            ["run_daily_model_pipeline.py", "--enable-strategy-stability"],
        ):
            args = parse_args()
        self.assertIsNone(args.base_panel)
        self.assertTrue(args.enable_strategy_stability)

    def test_parse_args_accepts_trend_ignition_shadow_flag(self):
        with patch.object(
            sys,
            "argv",
            ["run_daily_model_pipeline.py", "--enable-trend-ignition-shadow"],
        ):
            args = parse_args()
        self.assertTrue(args.enable_trend_ignition_shadow)
        self.assertEqual(args.trend_ignition_selection_status, "exploratory_posthoc")

    def test_trend_ignition_shadow_state_rejects_stale_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "research_only": True,
                        "trade_instruction": False,
                        "inputs": {"start": "2026-06-28", "end": "2026-06-28"},
                    }
                ),
                encoding="utf-8",
            )
            config = PipelineConfig(
                asof_date="2026-06-29",
                include_trend_ignition_shadow=True,
            )
            steps = [PipelineStep("trend_ignition_shadow", ("python", "shadow.py"))]

            state, artifact = _trend_ignition_shadow_state(
                config,
                steps,
                {"trend_ignition_shadow_manifest": manifest},
                "success",
                None,
            )

        self.assertEqual(state["status"], "stale")
        self.assertIsNone(artifact)

    def test_factor_decay_status_preserves_observation_only_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "factor_decay_monitor_20260629.json"
            path.write_text(
                json.dumps(
                    {
                        "asof_date": "2026-06-29",
                        "factor": "liquidity_stability_20",
                        "overall_status": "direction_reversal",
                        "calculation_mode": "incremental",
                        "incremental_checkpoint": {
                            "recompute_start": "2026-05-29",
                            "parent": {"source_asof_date": "2026-06-26"},
                        },
                        "automatic_model_change": False,
                        "research_only": True,
                        "replacement_competition": {
                            "status": "complete",
                            "shortlist": ["momentum_60"],
                            "preregistered_candidates": ["momentum_60"],
                            "validation_start_date": "2026-06-30",
                            "tracking_status": "collecting",
                        },
                    }
                ),
                encoding="utf-8",
            )

            status = factor_decay_monitor_status(path, "2026-06-29")

            self.assertEqual(status["status"], "direction_reversal")
            self.assertFalse(status["automatic_model_change"])
            self.assertEqual(status["replacement_shortlist"], ["momentum_60"])
            self.assertEqual(status["replacement_tracking_status"], "collecting")
            self.assertEqual(status["calculation_mode"], "incremental")
            self.assertEqual(status["recompute_start"], "2026-05-29")
            self.assertEqual(status["checkpoint_parent_asof_date"], "2026-06-26")

    def test_pipeline_passes_latest_shadow_account_review_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_dir = (
                root
                / "outputs"
                / "personal_trade_review_20260629"
                / "shadow_account"
            )
            review_dir.mkdir(parents=True)
            review = review_dir / "shadow_account_review.json"
            review.write_text("{}", encoding="utf-8")
            future_dir = (
                root
                / "outputs"
                / "personal_trade_review_20260701"
                / "shadow_account"
            )
            future_dir.mkdir(parents=True)
            future = future_dir / "shadow_account_review.json"
            future.write_text("{}", encoding="utf-8")
            older_dir = (
                root
                / "outputs"
                / "personal_trade_review_20260620"
                / "shadow_account"
            )
            older_dir.mkdir(parents=True)
            older = older_dir / "shadow_account_review.json"
            older.write_text("{}", encoding="utf-8")
            config = PipelineConfig(
                asof_date="20260629",
                python_exe="python",
                project_root=root,
                output_root=Path("outputs/high_return_v2"),
            )

            steps = {step.name: step for step in build_daily_pipeline_steps(config)}
            report_args = " ".join(steps["daily_chinese_report"].command)
            merged_args = " ".join(steps["merged_daily_outputs"].command)
            detected = latest_shadow_account_review(root, "2026-06-29")

        self.assertEqual(detected, review)
        self.assertIn(f"--shadow-account-review {review}", report_args)
        self.assertIn(f"--shadow-account-review {review}", merged_args)

    def test_pipeline_can_skip_regime_shadow_compare(self):
        config = PipelineConfig(
            asof_date="2026-06-29",
            python_exe="python",
            project_root=Path("C:/model"),
            include_regime_shadow_compare=False,
            include_regime_shadow_tracking=False,
        )

        names = [step.name for step in build_daily_pipeline_steps(config)]

        self.assertNotIn("regime_shadow_compare", names)
        self.assertNotIn("strategy_arena", names)
        self.assertNotIn("train_breadth_guard_challenger", names)

    def test_pipeline_can_skip_strategy_arena_without_disabling_regime_shadow(self):
        config = PipelineConfig(
            asof_date="2026-06-29",
            python_exe="python",
            project_root=Path("C:/model"),
            include_strategy_arena=False,
        )

        names = [step.name for step in build_daily_pipeline_steps(config)]

        self.assertIn("regime_shadow_compare", names)
        self.assertNotIn("strategy_arena", names)
        self.assertNotIn("train_breadth_guard_challenger", names)

    def test_pipeline_requires_compare_when_tracking_enabled(self):
        config = PipelineConfig(
            asof_date="2026-06-29",
            python_exe="python",
            project_root=Path("C:/model"),
            include_regime_shadow_compare=False,
        )
        object.__setattr__(config, "include_regime_shadow_tracking", True)

        with self.assertRaisesRegex(ValueError, "comparison"):
            build_daily_pipeline_steps(config)

    def test_parse_args_accepts_skip_regime_shadow_tracking_flag(self):
        with patch.object(sys, "argv", ["run_daily_model_pipeline.py", "--skip-regime-shadow-tracking"]):
            try:
                args = parse_args()
            except SystemExit as exc:
                self.fail(f"parse_args should accept --skip-regime-shadow-tracking: {exc}")

        self.assertTrue(args.skip_regime_shadow_tracking)

    def test_pipeline_can_skip_benchmark_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = PipelineConfig(
                asof_date="2026-06-29",
                python_exe="python",
                project_root=Path(tmp),
                include_benchmark_refresh=False,
                daily_data_dir=Path(tmp),
                fetch_status_utils=Path(tmp) / "missing_utils.py",
            )
            stale_status = (
                Path(tmp)
                / "outputs"
                / "high_return_v2"
                / "benchmark_refresh_status_20260629.json"
            )
            stale_status.parent.mkdir(parents=True)
            stale_status.write_text('{"status":"old"}', encoding="utf-8")

            names = [step.name for step in build_daily_pipeline_steps(config)]
            record = build_daily_run_state(config, [])

            self.assertNotIn("benchmark_refresh", names)
            self.assertEqual(names[0], "update_panel")
            self.assertEqual(record["verification"]["benchmark_refresh_status"], "skipped")
            self.assertIsNone(record["artifacts"]["benchmark_refresh_status"])

    def test_failed_benchmark_step_is_appended_to_state_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_log = root / "daily_run_state.jsonl"
            config = PipelineConfig(
                asof_date="2026-06-29",
                python_exe="python",
                project_root=root,
                output_root=Path("outputs/high_return_v2"),
                daily_data_dir=root,
                fetch_status_utils=root / "missing_utils.py",
            )
            steps = [PipelineStep("benchmark_refresh", ("python", "refresh.py"))]

            with patch(
                "run_daily_model_pipeline.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["python", "refresh.py"]),
            ):
                with self.assertRaisesRegex(RuntimeError, "benchmark_refresh"):
                    execute_pipeline(
                        config,
                        steps,
                        state_log=state_log,
                        argv=["--asof-date", "2026-06-29"],
                        test_status="not_run",
                    )

            record = json.loads(state_log.read_text(encoding="utf-8").strip())
            self.assertEqual(record["run_status"], "failed")
            self.assertEqual(record["failure"]["step"], "benchmark_refresh")
            self.assertEqual(record["verification"]["benchmark_refresh_status"], "missing")

    def test_fetch_status_must_be_ok_before_real_run(self):
        ensure_fetch_status_ok({"status": "ok", "run_id": "run-1"})

        with self.assertRaisesRegex(RuntimeError, "not ready"):
            ensure_fetch_status_ok({"status": "status_error", "message": "offline"})

    def test_names_source_for_date_falls_back_to_xls_when_csv_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normalized = root / "ths_exports" / "normalized"
            normalized.mkdir(parents=True)
            xls = normalized / "ths_hs_a_share_2026-06-30.xls"
            xls.write_text("placeholder", encoding="utf-8")

            self.assertEqual(names_source_for_date("2026-06-30", root), xls)

    def test_metrics_path_for_date_uses_latest_existing_best_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            latest = output_root / "personal_behavior_overlay_best_20220101_20260629" / "metrics.json"
            latest.parent.mkdir(parents=True)
            latest.write_text("{}", encoding="utf-8")

            self.assertEqual(metrics_path_for_date("2026-06-30", output_root), latest)

    def test_build_daily_run_state_summarizes_daily_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs" / "high_return_v2"
            output.mkdir(parents=True)
            token = "20260629"
            self._write_csv(
                output / f"merged_priority_watchlist_{token}.csv",
                [
                    {"symbol": "000001", "stock_name": "Alpha", "priority_bucket": "model_focus"},
                    {"symbol": "000002", "stock_name": "", "priority_bucket": "risk_watch"},
                ],
            )
            self._write_csv(output / f"early_pattern_watchlist_{token}.csv", [{"pattern_type": "pattern_a"}])
            self._write_csv(
                output / f"main_net_volume_shadow_{token}.csv",
                [
                    {"symbol": "000001", "main_net_volume_ratio": 0.01},
                    {"symbol": "000002", "main_net_volume_ratio": -0.01},
                ],
            )
            (output / f"main_net_volume_shadow_{token}.json").write_text(
                json.dumps(
                    {
                        "status": "warmup",
                        "source_latest_date": "2026-06-29",
                        "available_source_sessions": 2,
                        "latest_source_coverage": 0.998,
                        "eligible_5d_rows": 0,
                        "early_pattern_overlap_count": 1,
                        "selection_effect": False,
                    }
                ),
                encoding="utf-8",
            )
            self._write_csv(
                output / f"institutional_accumulation_shadow_{token}.csv",
                [
                    {
                        "symbol": "000001",
                        "institutional_accumulation_level": "资金流预热观察",
                        "signal_active": False,
                    }
                ],
            )
            (output / f"institutional_accumulation_shadow_{token}.json").write_text(
                json.dumps(
                    {
                        "status": "warmup",
                        "active_signal_rows": 0,
                        "available_flow_sessions": 4,
                        "source_columns_available": {
                            "main_net_volume_ratio": True,
                            "main_net_inflow": False,
                        },
                        "source_column_latest_dates": {
                            "main_net_volume_ratio": "2026-06-29",
                            "main_net_inflow": "2026-06-28",
                        },
                        "selection_effect": False,
                        "registration_id": "institutional_accumulation_shadow_v1",
                        "validation_start_date": "2026-07-20",
                        "config_sha256": "a" * 64,
                        "implementation_sha256": "b" * 64,
                    }
                ),
                encoding="utf-8",
            )
            (output / f"factor_decay_monitor_{token}.json").write_text(
                json.dumps(
                    {
                        "asof_date": "2026-06-29",
                        "overall_status": "stable",
                        "factor_metadata": {
                            "factor_current_availability": {"flow_persistence": False},
                            "factor_current_coverage": {"flow_persistence": 0.0},
                            "factor_input_latest_dates": {"main_net_inflow": "2026-06-28"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            self._write_csv(
                output / f"institutional_accumulation_tracking_{token}.csv",
                [{"symbol": "000001", "tracking_status": "pending_entry"}],
            )
            (output / f"institutional_accumulation_tracking_{token}.json").write_text(
                json.dumps(
                    {
                        "status": "provisional",
                        "primary_completed_samples": 0,
                        "gate_evaluation_allowed": False,
                        "promotion_allowed": False,
                    }
                ),
                encoding="utf-8",
            )
            self._write_csv(
                output / f"czsc_structure_shadow_{token}.csv",
                [
                    {"symbol": "000001", "analysis_status": "analyzed"},
                    {"symbol": "000002", "analysis_status": "insufficient_history"},
                ],
            )
            (output / f"czsc_structure_shadow_{token}.json").write_text(
                json.dumps(
                    {
                        "status": "partial",
                        "analyzed_count": 1,
                        "pattern_confluence_count": 1,
                        "second_buy_count": 1,
                        "third_buy_zone_consistent_count": 0,
                        "selection_effect": False,
                    }
                ),
                encoding="utf-8",
            )
            self._write_csv(
                output / f"strategy_arena_portfolio_{token}.csv",
                [
                    {"entrant_id": "core_rank", "role": "production_champion"},
                    {"entrant_id": "core_breadth_guard", "role": "shadow_challenger"},
                    {"entrant_id": "pullback_baseline", "role": "league_reference"},
                    {"entrant_id": "pullback_dynamic", "role": "shadow_challenger"},
                ],
            )
            (output / f"strategy_arena_{token}.json").write_text(
                json.dumps(
                    {
                        "status": "research_ready",
                        "asof_date": "2026-06-29",
                        "production_champion": "core_rank",
                        "historical_pareto_by_league": {
                            "core_next_open": ["core_breadth_guard"]
                        },
                        "independent_observation_status": "collecting",
                        "independent_observation_count": 2,
                        "independent_observation_target": 20,
                        "automatic_promotion": False,
                    }
                ),
                encoding="utf-8",
            )
            (output / "strategy_arena_history.csv").write_text(
                "asof_date,entrant_id\n2026-06-29,core_rank\n", encoding="utf-8"
            )
            self._write_csv(
                output / f"hidden_accumulation_trade_watch_tracking_{token}.csv",
                [
                    {"symbol": "000001", "tracking_status": "complete"},
                    {"symbol": "000002", "tracking_status": "partial"},
                ],
            )
            self._write_csv(output / f"merged_model_decision_table_{token}.csv", [{"symbol": "000001"}])
            self._write_csv(output / f"daily_personal_overlay_selected_{token}.csv", [{"symbol": "000001"}])
            self._write_csv(output / f"daily_personal_overlay_changes_{token}.csv", [{"symbol": "000002"}])
            (output / f"daily_personal_overlay_report_{token}.md").write_text("report", encoding="utf-8")
            shadow_dir = output / f"regime_shadow_compare_{token}"
            shadow_dir.mkdir()
            (shadow_dir / "comparison.json").write_text(
                json.dumps(
                    {
                        "decision": "experimental_only",
                        "benchmark_last_date": "2026-07-10",
                        "benchmark_fresh": False,
                        "delta": {"total_return": 0.03, "max_drawdown": 0.01},
                        "latest_dynamic_state": {
                            "risk_regime": "strong",
                            "target_leverage": 0.75,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (shadow_dir / "report.md").write_text("shadow report", encoding="utf-8")
            (output / f"benchmark_refresh_status_{token}.json").write_text(
                json.dumps(
                    {
                        "status": "already_fresh",
                        "latest_date": "2026-06-29",
                        "source_agreement": True,
                        "rows_added": 0,
                    }
                ),
                encoding="utf-8",
            )
            config = PipelineConfig(
                asof_date="2026-06-29",
                python_exe="python",
                project_root=root,
                output_root=Path("outputs/high_return_v2"),
                daily_data_dir=root,
                fetch_status_utils=root / "missing_utils.py",
            )

            record = build_daily_run_state(
                config,
                [PipelineStep("example", ("python", "example.py"))],
                argv=["--asof-date", "2026-06-29"],
                test_status="not_run",
            )

            self.assertEqual(record["run_type"], "full_train")
            self.assertEqual(record["verification"]["priority_rows"], 2)
            self.assertEqual(record["verification"]["missing_stock_names"], 1)
            self.assertEqual(record["verification"]["priority_bucket_counts"], {"model_focus": 1, "risk_watch": 1})
            self.assertEqual(record["verification"]["early_pattern_counts"], {"pattern_a": 1})
            self.assertEqual(record["verification"]["main_net_volume_shadow_status"], "warmup")
            self.assertEqual(record["verification"]["main_net_volume_shadow_rows"], 2)
            self.assertEqual(record["verification"]["main_net_volume_source_sessions"], 2)
            self.assertFalse(record["verification"]["main_net_volume_selection_effect"])
            self.assertEqual(record["verification"]["institutional_accumulation_status"], "warmup")
            self.assertEqual(record["verification"]["institutional_accumulation_rows"], 1)
            self.assertEqual(record["verification"]["institutional_accumulation_active_signals"], 0)
            self.assertEqual(record["verification"]["institutional_accumulation_flow_sessions"], 4)
            self.assertEqual(
                record["verification"]["institutional_accumulation_source_columns_available"],
                {"main_net_volume_ratio": True, "main_net_inflow": False},
            )
            self.assertEqual(
                record["verification"]["institutional_accumulation_source_column_latest_dates"][
                    "main_net_inflow"
                ],
                "2026-06-28",
            )
            self.assertEqual(
                record["verification"]["institutional_accumulation_registration_id"],
                "institutional_accumulation_shadow_v1",
            )
            self.assertEqual(
                record["verification"]["institutional_accumulation_validation_start_date"],
                "2026-07-20",
            )
            self.assertEqual(
                record["verification"]["institutional_accumulation_config_sha256"],
                "a" * 64,
            )
            self.assertEqual(
                record["verification"]["institutional_accumulation_implementation_sha256"],
                "b" * 64,
            )
            self.assertFalse(record["verification"]["institutional_accumulation_selection_effect"])
            self.assertEqual(record["verification"]["institutional_accumulation_tracking_rows"], 1)
            self.assertEqual(record["verification"]["institutional_accumulation_tracking_status"], "provisional")
            self.assertEqual(record["verification"]["institutional_accumulation_completed_5d"], 0)
            self.assertFalse(record["verification"]["institutional_accumulation_gate_allowed"])
            self.assertFalse(record["verification"]["institutional_accumulation_promotion_allowed"])
            self.assertEqual(
                record["verification"]["factor_current_availability"],
                {"flow_persistence": False},
            )
            self.assertEqual(
                record["verification"]["factor_current_coverage"],
                {"flow_persistence": 0.0},
            )
            self.assertEqual(
                record["verification"]["factor_input_latest_dates"],
                {"main_net_inflow": "2026-06-28"},
            )
            self.assertTrue(
                str(record["artifacts"]["institutional_accumulation_report"]).endswith(".md")
            )
            self.assertEqual(record["verification"]["czsc_structure_shadow_status"], "partial")
            self.assertEqual(record["verification"]["czsc_structure_shadow_rows"], 2)
            self.assertEqual(record["verification"]["czsc_structure_analyzed_rows"], 1)
            self.assertEqual(record["verification"]["czsc_structure_pattern_confluence"], 1)
            self.assertFalse(record["verification"]["czsc_structure_selection_effect"])
            self.assertTrue(
                str(record["artifacts"]["czsc_structure_shadow_report"]).endswith(".md")
            )
            self.assertEqual(record["verification"]["strategy_arena_status"], "research_ready")
            self.assertEqual(record["verification"]["strategy_arena_portfolio_entrants"], 4)
            self.assertEqual(record["verification"]["strategy_arena_production_champion"], "core_rank")
            self.assertEqual(record["verification"]["strategy_arena_observation_count"], 2)
            self.assertFalse(record["verification"]["strategy_arena_automatic_promotion"])
            self.assertTrue(str(record["artifacts"]["strategy_arena_report"]).endswith(".md"))
            self.assertEqual(record["verification"]["hidden_trade_watch_tracking_rows"], 2)
            self.assertEqual(
                record["verification"]["hidden_trade_watch_tracking_status_counts"],
                {"complete": 1, "partial": 1},
            )
            self.assertEqual(record["top10"][0]["symbol"], "000001")
            self.assertEqual(record["fallback_fetch_status"]["status"], "missing_utils")
            self.assertEqual(record["verification"]["regime_shadow_decision"], "experimental_only")
            self.assertIsNone(record["verification"]["regime_shadow_risk_regime"])
            self.assertIsNone(record["verification"]["regime_shadow_target_leverage"])
            self.assertEqual(record["verification"]["regime_shadow_total_return_delta"], 0.03)
            self.assertFalse(record["verification"]["regime_shadow_benchmark_fresh"])
            self.assertTrue(str(record["artifacts"]["regime_shadow_report"]).endswith("report.md"))
            self.assertEqual(record["verification"]["benchmark_refresh_status"], "already_fresh")
            self.assertEqual(record["verification"]["benchmark_latest_date"], "2026-06-29")
            self.assertTrue(record["verification"]["benchmark_source_agreement"])
            self.assertTrue(str(record["artifacts"]["benchmark_refresh_status"]).endswith(".json"))

    def test_build_daily_run_state_includes_tracking_summary_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs" / "high_return_v2"
            output.mkdir(parents=True)
            token = "20260629"
            self._write_csv(
                output / f"merged_priority_watchlist_{token}.csv",
                [{"symbol": "000001", "stock_name": "Alpha", "priority_bucket": "model_focus"}],
            )
            self._write_csv(output / f"early_pattern_watchlist_{token}.csv", [])
            self._write_csv(output / f"merged_model_decision_table_{token}.csv", [])
            self._write_csv(output / f"daily_personal_overlay_selected_{token}.csv", [])
            self._write_csv(output / f"daily_personal_overlay_changes_{token}.csv", [])
            shadow_dir = output / f"regime_shadow_compare_{token}"
            shadow_dir.mkdir()
            (shadow_dir / "comparison.json").write_text(
                json.dumps({"decision": "experimental_only", "benchmark_fresh": True, "delta": {}, "latest_dynamic_state": {}}),
                encoding="utf-8",
            )
            (shadow_dir / "report.md").write_text("shadow report", encoding="utf-8")
            tracking_summary = output / "regime_shadow_tracking_summary.json"
            tracking_summary.write_text(
                json.dumps(
                    {
                        "status": "manual_review_ready",
                        "latest_asof_date": "2026-06-29",
                        "valid_observation_count": 20,
                        "target_days": 20,
                        "remaining_days": 0,
                        "invalid_observation_count": 0,
                        "cumulative_return_delta": 0.05,
                        "latest_benchmark_fresh": True,
                        "latest_risk_state": {"risk_regime": "strong", "target_leverage": 0.75},
                    }
                ),
                encoding="utf-8",
            )
            (output / "regime_shadow_tracking.csv").write_text("asof_date\n2026-06-29\n", encoding="utf-8")
            (output / "regime_shadow_tracking_report.md").write_text("tracking report", encoding="utf-8")
            config = PipelineConfig(
                asof_date="2026-06-29",
                python_exe="python",
                project_root=root,
                output_root=Path("outputs/high_return_v2"),
                daily_data_dir=root,
                fetch_status_utils=root / "missing_utils.py",
            )

            record = build_daily_run_state(config, [])

            self.assertEqual(record["verification"]["regime_shadow_tracking_status"], "manual_review_ready")
            self.assertEqual(record["verification"]["regime_shadow_tracking_valid_observations"], 20)
            self.assertEqual(record["verification"]["regime_shadow_tracking_target_days"], 20)
            self.assertEqual(record["verification"]["regime_shadow_tracking_remaining_days"], 0)
            self.assertEqual(record["verification"]["regime_shadow_tracking_cumulative_return_delta"], 0.05)
            self.assertTrue(record["verification"]["regime_shadow_tracking_benchmark_fresh"])
            self.assertEqual(record["verification"]["regime_shadow_tracking_risk_regime"], "strong")
            self.assertTrue(str(record["artifacts"]["regime_shadow_tracking_summary"]).endswith(".json"))

    def test_failed_pipeline_before_tracking_hides_stale_tracking_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs" / "high_return_v2"
            output.mkdir(parents=True)
            token = "20260629"
            self._seed_minimal_daily_state_inputs(output, token)
            self._write_tracking_outputs(
                output,
                status="manual_review_ready",
                latest_asof_date="2026-06-28",
            )
            config = PipelineConfig(
                asof_date="2026-06-29",
                python_exe="python",
                project_root=root,
                output_root=Path("outputs/high_return_v2"),
                daily_data_dir=root,
                fetch_status_utils=root / "missing_utils.py",
            )
            steps = build_daily_pipeline_steps(config)

            record = build_daily_run_state(
                config,
                steps,
                run_status="failed",
                failure={"step": "regime_shadow_compare", "returncode": 1},
            )

            self.assertEqual(record["verification"]["regime_shadow_tracking_status"], "not_run")
            self.assertIsNone(record["artifacts"]["regime_shadow_tracking_ledger"])
            self.assertIsNone(record["artifacts"]["regime_shadow_tracking_summary"])
            self.assertIsNone(record["artifacts"]["regime_shadow_tracking_report"])

    def test_failed_tracking_step_hides_stale_tracking_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs" / "high_return_v2"
            output.mkdir(parents=True)
            token = "20260629"
            self._seed_minimal_daily_state_inputs(output, token)
            self._write_tracking_outputs(
                output,
                status="manual_review_ready",
                latest_asof_date="2026-06-28",
            )
            config = PipelineConfig(
                asof_date="2026-06-29",
                python_exe="python",
                project_root=root,
                output_root=Path("outputs/high_return_v2"),
                daily_data_dir=root,
                fetch_status_utils=root / "missing_utils.py",
            )
            steps = build_daily_pipeline_steps(config)

            record = build_daily_run_state(
                config,
                steps,
                run_status="failed",
                failure={"step": "regime_shadow_tracking", "returncode": 1},
            )

            self.assertEqual(record["verification"]["regime_shadow_tracking_status"], "failed")
            self.assertIsNone(record["artifacts"]["regime_shadow_tracking_ledger"])
            self.assertIsNone(record["artifacts"]["regime_shadow_tracking_summary"])
            self.assertIsNone(record["artifacts"]["regime_shadow_tracking_report"])

    def test_failed_pipeline_after_tracking_requires_current_tracking_asof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs" / "high_return_v2"
            output.mkdir(parents=True)
            token = "20260629"
            self._seed_minimal_daily_state_inputs(output, token)
            self._write_tracking_outputs(
                output,
                status="manual_review_ready",
                latest_asof_date="2026-06-28",
            )
            config = PipelineConfig(
                asof_date="2026-06-29",
                python_exe="python",
                project_root=root,
                output_root=Path("outputs/high_return_v2"),
                daily_data_dir=root,
                fetch_status_utils=root / "missing_utils.py",
            )
            steps = build_daily_pipeline_steps(config)

            record = build_daily_run_state(
                config,
                steps,
                run_status="failed",
                failure={"step": "train_rank_model", "returncode": 1},
            )

            self.assertEqual(record["verification"]["regime_shadow_tracking_status"], "missing")
            self.assertIsNone(record["artifacts"]["regime_shadow_tracking_ledger"])
            self.assertIsNone(record["artifacts"]["regime_shadow_tracking_summary"])
            self.assertIsNone(record["artifacts"]["regime_shadow_tracking_report"])

    def test_build_daily_run_state_marks_tracking_skipped_and_hides_stale_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs" / "high_return_v2"
            output.mkdir(parents=True)
            token = "20260629"
            self._write_csv(
                output / f"merged_priority_watchlist_{token}.csv",
                [{"symbol": "000001", "stock_name": "Alpha", "priority_bucket": "model_focus"}],
            )
            self._write_csv(output / f"early_pattern_watchlist_{token}.csv", [])
            self._write_csv(output / f"merged_model_decision_table_{token}.csv", [])
            self._write_csv(output / f"daily_personal_overlay_selected_{token}.csv", [])
            self._write_csv(output / f"daily_personal_overlay_changes_{token}.csv", [])
            (output / "regime_shadow_tracking.csv").write_text("stale", encoding="utf-8")
            (output / "regime_shadow_tracking_summary.json").write_text('{"status": "stale"}', encoding="utf-8")
            (output / "regime_shadow_tracking_report.md").write_text("stale", encoding="utf-8")
            config = PipelineConfig(
                asof_date="2026-06-29",
                python_exe="python",
                project_root=root,
                output_root=Path("outputs/high_return_v2"),
                daily_data_dir=root,
                fetch_status_utils=root / "missing_utils.py",
            )
            object.__setattr__(config, "include_regime_shadow_tracking", False)

            record = build_daily_run_state(config, [])

            self.assertEqual(record["verification"]["regime_shadow_tracking_status"], "skipped")
            self.assertIsNone(record["verification"]["regime_shadow_tracking_valid_observations"])
            self.assertIsNone(record["artifacts"]["regime_shadow_tracking_ledger"])
            self.assertIsNone(record["artifacts"]["regime_shadow_tracking_summary"])
            self.assertIsNone(record["artifacts"]["regime_shadow_tracking_report"])

    def test_daily_run_state_uses_newest_pending_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs" / "high_return_v2"
            output.mkdir(parents=True)
            token = "20260629"
            self._write_csv(
                output / f"merged_priority_watchlist_{token}.csv",
                [{"symbol": "000001", "stock_name": "Old", "priority_bucket": "model_focus"}],
            )
            time.sleep(0.01)
            self._write_csv(
                output / f"merged_priority_watchlist_{token}_pending.csv",
                [
                    {"symbol": "000002", "stock_name": "New", "priority_bucket": "pattern_watch"},
                    {"symbol": "000003", "stock_name": "Newer", "priority_bucket": "pattern_watch"},
                ],
            )
            self._write_csv(output / f"early_pattern_watchlist_{token}.csv", [])
            self._write_csv(output / f"merged_model_decision_table_{token}.csv", [])
            self._write_csv(output / f"daily_personal_overlay_selected_{token}.csv", [])
            self._write_csv(output / f"daily_personal_overlay_changes_{token}.csv", [])
            config = PipelineConfig(
                asof_date="2026-06-29",
                python_exe="python",
                project_root=root,
                output_root=Path("outputs/high_return_v2"),
                daily_data_dir=root,
                fetch_status_utils=root / "missing_utils.py",
                train_model=False,
            )

            record = build_daily_run_state(config, [])

            self.assertEqual(record["run_type"], "skip_train")
            self.assertEqual(record["verification"]["priority_rows"], 2)
            self.assertTrue(str(record["artifacts"]["priority"]).endswith("_pending.csv"))

    def test_append_daily_run_state_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_log = Path(tmp) / "daily_run_state.jsonl"
            append_daily_run_state(state_log, {"asof_date": "2026-06-29", "value": 1})
            append_daily_run_state(state_log, {"asof_date": "2026-06-30", "value": 2})

            rows = [json.loads(line) for line in state_log.read_text(encoding="utf-8").splitlines()]

            self.assertEqual([row["asof_date"] for row in rows], ["2026-06-29", "2026-06-30"])

    def test_daily_run_state_records_research_database_sync_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs" / "high_return_v2"
            output.mkdir(parents=True)
            status_path = output / "research_database_sync_20260629.json"
            status_path.write_text(
                json.dumps({
                    "status": "success",
                    "asof_date": "2026-06-29",
                    "database_latest_date": "2026-06-29",
                    "source_latest_date": "2026-06-29",
                    "price_usable_latest_date": "2026-06-29",
                    "factor_usable_latest_date": "2026-06-29",
                    "price_usable_rows": 4990,
                    "price_usable_ratio": 0.998,
                    "factor_usable_rows": 4980,
                    "factor_usable_ratio": 0.996,
                    "factor_column_coverage": {"amount": 0.996},
                    "database_asof_rows": 5000,
                    "database_daily_rows": 100000,
                    "database_observation_rows": 200,
                    "observation_removed_source_alias_rows": 3,
                }),
                encoding="utf-8",
            )
            config = PipelineConfig(
                asof_date="2026-06-29",
                project_root=root,
                output_root=Path("outputs/high_return_v2"),
                include_benchmark_refresh=False,
                include_regime_shadow_compare=False,
                include_regime_shadow_tracking=False,
            )
            steps = build_daily_pipeline_steps(config)

            record = build_daily_run_state(config, steps)

        self.assertEqual(record["verification"]["research_database_sync_status"], "success")
        self.assertEqual(record["verification"]["research_database_asof_rows"], 5000)
        self.assertEqual(
            record["verification"]["research_database_source_latest_date"],
            "2026-06-29",
        )
        self.assertEqual(
            record["verification"]["research_database_price_usable_latest_date"],
            "2026-06-29",
        )
        self.assertEqual(
            record["verification"]["research_database_factor_usable_latest_date"],
            "2026-06-29",
        )
        self.assertEqual(
            record["verification"]["research_database_factor_column_coverage"],
            {"amount": 0.996},
        )
        self.assertEqual(
            record["verification"]["research_database_removed_source_alias_rows"],
            3,
        )
        self.assertTrue(str(record["artifacts"]["research_database_sync_status"]).endswith(".json"))

    def test_daily_run_state_marks_database_sync_not_run_after_earlier_failure(self):
        config = PipelineConfig(
            asof_date="2026-06-29",
            project_root=Path("C:/model"),
            include_benchmark_refresh=False,
            include_regime_shadow_compare=False,
            include_regime_shadow_tracking=False,
        )
        steps = build_daily_pipeline_steps(config)
        record = build_daily_run_state(
            config,
            steps,
            run_status="failed",
            failure={"step": "update_panel", "returncode": 1},
        )
        self.assertEqual(record["verification"]["research_database_sync_status"], "not_run")
        self.assertIsNone(record["artifacts"]["research_database_sync_status"])

    def test_attach_daily_run_card_writes_card_and_updates_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs" / "high_return_v2"
            output.mkdir(parents=True)
            self._seed_minimal_daily_state_inputs(output, "20260629")
            config = PipelineConfig(
                asof_date="2026-06-29",
                python_exe="python",
                project_root=root,
                output_root=Path("outputs/high_return_v2"),
                daily_data_dir=root,
                fetch_status_utils=root / "missing_utils.py",
            )
            record = build_daily_run_state(
                config, [PipelineStep("example", ("python", "example.py"))]
            )

            paths = attach_daily_run_card(
                config, record, generated_at="2026-06-29T16:00:00"
            )

            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["markdown"].exists())
            self.assertEqual(
                record["artifacts"]["daily_run_card_json"], str(paths["json"])
            )
            self.assertEqual(
                record["artifacts"]["daily_run_card_markdown"],
                str(paths["markdown"]),
            )

    def test_successful_pipeline_records_state_when_run_card_write_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_log = root / "daily_run_state.jsonl"
            config = PipelineConfig(
                asof_date="2026-06-29",
                python_exe="python",
                project_root=root,
                output_root=Path("outputs/high_return_v2"),
                daily_data_dir=root,
                fetch_status_utils=root / "missing_utils.py",
                include_benchmark_refresh=False,
                include_regime_shadow_compare=False,
                include_regime_shadow_tracking=False,
            )

            with patch(
                "run_daily_model_pipeline.attach_daily_run_card",
                side_effect=OSError("run card unavailable"),
            ):
                execute_pipeline(config, [], state_log=state_log)

            record = json.loads(state_log.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(record["run_status"], "success")
        self.assertEqual(record["daily_run_card_error"], "run card unavailable")

    def test_execute_pipeline_publishes_marketlens_from_current_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs" / "high_return_v2"
            dashboard_output = root / "dashboard" / "data" / "quant-model3-latest.json"
            state_log = root / "daily_run_state.jsonl"
            self._seed_minimal_daily_state_inputs(output, "20260629")
            config = PipelineConfig(
                asof_date="2026-06-29",
                python_exe=sys.executable,
                project_root=Path(__file__).resolve().parents[1],
                output_root=output,
                daily_data_dir=root,
                fetch_status_utils=root / "missing_utils.py",
                include_benchmark_refresh=False,
                include_regime_shadow_compare=False,
                include_regime_shadow_tracking=False,
                include_research_db_sync=False,
                include_factor_decay_monitor=False,
                include_strategy_family_forward_report=False,
                include_strategy_arena=False,
                include_strategy_stability_report=False,
                include_marketlens_export=True,
                marketlens_dashboard_output=dashboard_output,
            )

            execute_pipeline(
                config,
                [],
                state_log=state_log,
                test_status="123 passed",
            )

            model_feed = output / "marketlens_model3_latest.json"
            feed = json.loads(model_feed.read_text(encoding="utf-8"))
            state = json.loads(state_log.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(feed["asofDate"], "2026-06-29")
            self.assertEqual(feed["verification"]["tests"], "123 passed")
            self.assertEqual(json.loads(dashboard_output.read_text(encoding="utf-8")), feed)
            self.assertEqual(state["schema_version"], 2)
            self.assertEqual(state["run_status"], "success")
            self.assertEqual(state["verification"]["marketlens_export_status"], "success")
            self.assertEqual(state["artifacts"]["marketlens_feed"], str(model_feed))
            self.assertFalse((output / ".marketlens_state_20260629.json").exists())

    def test_execute_pipeline_records_marketlens_failure_when_current_input_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs" / "high_return_v2"
            dashboard_output = root / "dashboard" / "data" / "quant-model3-latest.json"
            state_log = root / "daily_run_state.jsonl"
            self._write_csv(
                output / "merged_priority_watchlist_20260629.csv",
                [{"symbol": "000001", "stock_name": "Alpha", "priority_bucket": "model_focus"}],
            )
            self._write_csv(output / "early_pattern_watchlist_20260629.csv", [])
            config = PipelineConfig(
                asof_date="2026-06-29",
                python_exe=sys.executable,
                project_root=Path(__file__).resolve().parents[1],
                output_root=output,
                daily_data_dir=root,
                fetch_status_utils=root / "missing_utils.py",
                include_benchmark_refresh=False,
                include_regime_shadow_compare=False,
                include_regime_shadow_tracking=False,
                include_research_db_sync=False,
                include_factor_decay_monitor=False,
                include_strategy_family_forward_report=False,
                include_strategy_arena=False,
                include_strategy_stability_report=False,
                include_marketlens_export=True,
                marketlens_dashboard_output=dashboard_output,
            )

            with self.assertRaises(PipelineStepExecutionError):
                execute_pipeline(config, [], state_log=state_log)

            state = json.loads(state_log.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(state["run_status"], "failed")
            self.assertEqual(state["failure"]["step"], "marketlens_export")
            self.assertEqual(state["verification"]["marketlens_export_status"], "failed")
            self.assertFalse((output / "marketlens_model3_latest.json").exists())
            self.assertFalse(dashboard_output.exists())
            self.assertFalse((output / ".marketlens_state_20260629.json").exists())

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        import csv

        path.parent.mkdir(parents=True, exist_ok=True)
        columns = sorted({column for row in rows for column in row})
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def _seed_minimal_daily_state_inputs(self, output: Path, token: str) -> None:
        self._write_csv(
            output / f"merged_priority_watchlist_{token}.csv",
            [{"symbol": "000001", "stock_name": "Alpha", "priority_bucket": "model_focus"}],
        )
        self._write_csv(output / f"early_pattern_watchlist_{token}.csv", [])
        self._write_csv(output / f"merged_model_decision_table_{token}.csv", [])
        self._write_csv(output / f"daily_personal_overlay_selected_{token}.csv", [])
        self._write_csv(output / f"daily_personal_overlay_changes_{token}.csv", [])
        shadow_dir = output / f"regime_shadow_compare_{token}"
        shadow_dir.mkdir(exist_ok=True)
        (shadow_dir / "comparison.json").write_text(
            json.dumps(
                {
                    "decision": "experimental_only",
                    "benchmark_last_date": "2026-06-29",
                    "benchmark_fresh": True,
                    "delta": {},
                    "latest_dynamic_state": {"risk_regime": "strong", "target_leverage": 0.75},
                }
            ),
            encoding="utf-8",
        )
        (shadow_dir / "report.md").write_text("shadow report", encoding="utf-8")

    def _write_tracking_outputs(self, output: Path, *, status: str, latest_asof_date: str) -> None:
        (output / "regime_shadow_tracking.csv").write_text("asof_date\n2026-06-28\n", encoding="utf-8")
        (output / "regime_shadow_tracking_summary.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "latest_asof_date": latest_asof_date,
                    "valid_observation_count": 20,
                    "target_days": 20,
                    "remaining_days": 0,
                    "invalid_observation_count": 0,
                    "cumulative_return_delta": 0.05,
                    "latest_benchmark_fresh": True,
                    "latest_risk_state": {"risk_regime": "strong", "target_leverage": 0.75},
                }
            ),
            encoding="utf-8",
        )
        (output / "regime_shadow_tracking_report.md").write_text("tracking report", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
