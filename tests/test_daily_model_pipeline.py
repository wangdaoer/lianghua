import tempfile
import unittest
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

from run_daily_model_pipeline import (
    PipelineConfig,
    PipelineStep,
    append_daily_run_state,
    attach_daily_run_card,
    build_daily_pipeline_steps,
    build_daily_run_state,
    ensure_fetch_status_ok,
    execute_pipeline,
    latest_daily_date,
    latest_shadow_account_review,
    metrics_path_for_date,
    names_source_for_date,
    parse_args,
)


class DailyModelPipelineTest(unittest.TestCase):
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
        self.assertIn("daily-market-data\\benchmarks\\510300.csv", shadow_args)
        self.assertIn("regime_shadow_tracking", names)
        tracking_step = steps[names.index("regime_shadow_tracking")]
        tracking_args = " ".join(str(part) for part in tracking_step.command)
        self.assertEqual(names[names.index("regime_shadow_compare") + 1], "regime_shadow_tracking")
        self.assertIn("update_regime_shadow_tracking.py", tracking_args)
        self.assertIn("regime_shadow_tracking.csv", tracking_args)
        self.assertIn("regime_shadow_tracking_summary.json", tracking_args)
        self.assertIn("regime_shadow_tracking_report.md", tracking_args)
        self.assertIn("--target-days 20", tracking_args)

        self.assertEqual(
            names[-8:],
            [
                "personal_overlay",
                "early_pattern_watchlist",
                "hidden_accumulation_tracking",
                "daily_chinese_report",
                "merged_daily_outputs",
                "strategy_family_forward_report",
                "strategy_stability_report",
                "research_database_sync",
            ],
        )
        watch_step = steps[-7]
        watch_args = " ".join(str(part) for part in watch_step.command)
        self.assertIn("early_pattern_watchlist.py", watch_args)
        tracking_step = steps[-6]
        tracking_args = " ".join(str(part) for part in tracking_step.command)
        self.assertIn("track_hidden_accumulation_watch.py", tracking_args)
        self.assertIn("hidden_accumulation_trade_watch_tracking_20260629.csv", tracking_args)
        report_step = steps[-5]
        report_args = " ".join(str(part) for part in report_step.command)
        self.assertIn("build_daily_personal_overlay_report.py", report_args)
        self.assertIn("rank_model_candidates_trend_gated_personal_overlay_20260629.csv", report_args)
        self.assertIn("early_pattern_watchlist_20260629.csv", report_args)
        self.assertIn("--names-source", report_args)
        merged_step = steps[-4]
        merged_args = " ".join(str(part) for part in merged_step.command)
        self.assertIn("merged_daily_outputs.py", merged_args)
        self.assertIn("trend_state_20260629", merged_args)
        self.assertIn("rank_model_candidates_trend_gated_bench20260629_20260629.csv", merged_args)
        self.assertIn("rank_model_candidates_trend_gated_personal_overlay_20260629.csv", merged_args)
        self.assertIn("early_pattern_watchlist_20260629.csv", merged_args)
        family_step = steps[-3]
        family_args = " ".join(str(part) for part in family_step.command)
        self.assertIn("evaluate_strategy_family_forward_returns.py", family_args)
        self.assertIn("--horizons 1,3,5,10", family_args)
        self.assertIn("--token 20260629", family_args)
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

    def test_parse_args_accepts_skip_research_database_sync_flag(self):
        with patch.object(sys, "argv", ["run_daily_model_pipeline.py", "--skip-research-db-sync"]):
            args = parse_args()
        self.assertTrue(args.skip_research_db_sync)

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
                    "database_asof_rows": 5000,
                    "database_daily_rows": 100000,
                    "database_observation_rows": 200,
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
