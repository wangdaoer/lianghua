import tempfile
import unittest
import json
import time
from pathlib import Path

from run_daily_model_pipeline import (
    PipelineConfig,
    PipelineStep,
    append_daily_run_state,
    build_daily_pipeline_steps,
    build_daily_run_state,
    latest_daily_date,
    metrics_path_for_date,
    names_source_for_date,
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

        self.assertEqual(
            names[-7:],
            [
                "personal_overlay",
                "early_pattern_watchlist",
                "hidden_accumulation_tracking",
                "daily_chinese_report",
                "merged_daily_outputs",
                "strategy_family_forward_report",
                "strategy_stability_report",
            ],
        )
        watch_step = steps[-6]
        watch_args = " ".join(str(part) for part in watch_step.command)
        self.assertIn("early_pattern_watchlist.py", watch_args)
        tracking_step = steps[-5]
        tracking_args = " ".join(str(part) for part in tracking_step.command)
        self.assertIn("track_hidden_accumulation_watch.py", tracking_args)
        self.assertIn("hidden_accumulation_trade_watch_tracking_20260629.csv", tracking_args)
        report_step = steps[-4]
        report_args = " ".join(str(part) for part in report_step.command)
        self.assertIn("build_daily_personal_overlay_report.py", report_args)
        self.assertIn("rank_model_candidates_trend_gated_personal_overlay_20260629.csv", report_args)
        self.assertIn("early_pattern_watchlist_20260629.csv", report_args)
        self.assertIn("--names-source", report_args)
        merged_step = steps[-3]
        merged_args = " ".join(str(part) for part in merged_step.command)
        self.assertIn("merged_daily_outputs.py", merged_args)
        self.assertIn("trend_state_20260629", merged_args)
        self.assertIn("rank_model_candidates_trend_gated_bench20260629_20260629.csv", merged_args)
        self.assertIn("rank_model_candidates_trend_gated_personal_overlay_20260629.csv", merged_args)
        self.assertIn("early_pattern_watchlist_20260629.csv", merged_args)
        family_step = steps[-2]
        family_args = " ".join(str(part) for part in family_step.command)
        self.assertIn("evaluate_strategy_family_forward_returns.py", family_args)
        self.assertIn("--horizons 1,3,5,10", family_args)
        self.assertIn("--token 20260629", family_args)
        stability_step = steps[-1]
        stability_args = " ".join(str(part) for part in stability_step.command)
        self.assertIn("summarize_core_risk_filter_stability.py", stability_args)
        self.assertIn("core_risk_filter_finalist_stability_20260629", stability_args)

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

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        import csv

        path.parent.mkdir(parents=True, exist_ok=True)
        columns = sorted({column for row in rows for column in row})
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
