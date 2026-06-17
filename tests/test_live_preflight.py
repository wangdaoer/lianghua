from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser
from quant_etf_lab.live_preflight import run_live_preflight


def write_dashboard_snapshot(path: Path, paper_dir: Path, **overrides: object) -> None:
    payload = {
        "generated_at": "2026-06-15T16:30:00",
        "dashboard_posture": "core_base_watch_allocator_gate",
        "data_freshness_status": "fresh_enough",
        "market_cache_status": "fresh_enough",
        "allocator_input_status": "fresh_enough",
        "paper_account_status": "ok",
        "paper_freshness_status": "fresh_enough",
        "sentiment_freshness_status": "fresh_enough",
        "trigger_freshness_status": "fresh_enough",
        "trading_day_gate_status": "trading_day_data_ready",
        "after_close_data_status": "ready",
        "paper_account_dir": str(paper_dir),
        "paper_account_metrics_path": str(paper_dir / "metrics.json"),
        "paper_latest_regime": "risk_off",
        "paper_latest_core_weight": 1.0,
        "paper_latest_satellite_weight": 0.0,
        "paper_latest_cash_weight": 0.0,
        "paper_latest_date": "2026-06-15",
        "paper_stock_target_review_action_count": 0,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_paper_account_fixture(paper_dir: Path, action_count: int = 0, satellite_weight: float = 0.0) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    target_holdings_path = paper_dir / "target_holdings.csv"
    actions_path = paper_dir / "stock_target_review_actions.csv"
    pd.DataFrame(
        {
            "layer": ["core", "satellite", "cash"],
            "target_weight": [1.0 - satellite_weight, satellite_weight, 0.0],
        }
    ).to_csv(target_holdings_path, index=False)
    pd.DataFrame({"action_code": [f"review_{index}" for index in range(action_count)]}).to_csv(actions_path, index=False)
    (paper_dir / "metrics.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-06-15T16:30:00",
                "latest_date": "2026-06-15",
                "latest_regime": "risk_on" if satellite_weight > 0 else "risk_off",
                "latest_core_weight": 1.0 - satellite_weight,
                "latest_satellite_weight": satellite_weight,
                "latest_cash_weight": 0.0,
                "target_holdings_path": str(target_holdings_path),
                "stock_target_review_actions_path": str(actions_path),
                "stock_target_review_action_count": action_count,
            }
        ),
        encoding="utf-8",
    )


class LivePreflightTests(unittest.TestCase):
    def test_live_preflight_cli_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "live-preflight",
                "--dashboard-snapshot",
                "outputs/research/latest_dashboard/latest_dashboard_snapshot.json",
                "--paper-account-dir",
                "outputs/research/paper_account_latest",
                "--pipeline-snapshot",
                "outputs/research/daily_pipeline_latest/daily_pipeline_snapshot.json",
                "--daily-run-status-snapshot",
                "outputs/research/daily_run_status_latest/daily_run_status_snapshot.json",
                "--live-shadow-review-decisions-file",
                "outputs/research/live_shadow_review_decisions_latest/live_shadow_review_decisions.csv",
                "--output-dir",
                "outputs/research/live_preflight_latest",
                "--fail-on-blocked",
            ]
        )

        self.assertEqual(args.command, "live-preflight")
        self.assertEqual(args.paper_account_dir, "outputs/research/paper_account_latest")
        self.assertEqual(args.pipeline_snapshot, "outputs/research/daily_pipeline_latest/daily_pipeline_snapshot.json")
        self.assertEqual(
            args.live_shadow_review_decisions_file,
            "outputs/research/live_shadow_review_decisions_latest/live_shadow_review_decisions.csv",
        )
        self.assertTrue(args.fail_on_blocked)

    def test_live_preflight_allows_watch_only_stage_without_broker_connection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_dir = root / "paper"
            dashboard_snapshot = root / "latest_dashboard_snapshot.json"
            write_paper_account_fixture(paper_dir, action_count=0, satellite_weight=0.0)
            write_dashboard_snapshot(dashboard_snapshot, paper_dir)

            result = run_live_preflight(
                dashboard_snapshot=dashboard_snapshot,
                output_dir=root / "live_preflight",
            )

            self.assertEqual(result.snapshot["decision"], "ready_for_watch_only_pre_stage")
            self.assertEqual(result.snapshot["broker_connection_status"], "not_connected")
            self.assertEqual(result.snapshot["blocking_items"], [])
            self.assertEqual(result.snapshot["stock_target_review_action_count"], 0)
            self.assertEqual(result.snapshot["active_target_count"], 1)
            self.assertTrue(result.checklist_path.exists())
            self.assertTrue(result.snapshot_path.exists())
            self.assertTrue(result.report_path.exists())
            self.assertIn("does not connect to brokers", result.report_path.read_text(encoding="utf-8"))

    def test_live_preflight_uses_pipeline_snapshot_for_pipeline_only_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_dir = root / "paper"
            dashboard_snapshot = root / "latest_dashboard_snapshot.json"
            pipeline_snapshot = root / "daily_pipeline_snapshot.json"
            write_paper_account_fixture(paper_dir, action_count=0, satellite_weight=0.0)
            write_dashboard_snapshot(dashboard_snapshot, paper_dir)
            dashboard_payload = json.loads(dashboard_snapshot.read_text(encoding="utf-8"))
            dashboard_payload.pop("trading_day_gate_status")
            dashboard_payload.pop("after_close_data_status")
            dashboard_snapshot.write_text(json.dumps(dashboard_payload), encoding="utf-8")
            pipeline_snapshot.write_text(
                json.dumps(
                    {
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                    }
                ),
                encoding="utf-8",
            )

            result = run_live_preflight(
                dashboard_snapshot=dashboard_snapshot,
                pipeline_snapshot=pipeline_snapshot,
                output_dir=root / "live_preflight",
            )

            self.assertEqual(result.snapshot["decision"], "ready_for_watch_only_pre_stage")
            self.assertEqual(result.snapshot["pipeline_snapshot_status"], "ok")
            self.assertEqual(result.snapshot["trading_day_gate_status"], "trading_day_data_ready")
            self.assertEqual(result.snapshot["after_close_data_status"], "ready")

    def test_live_preflight_prefers_pipeline_paper_account_paths_over_stale_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale_paper_dir = root / "stale_paper"
            fresh_paper_dir = root / "fresh_paper"
            dashboard_snapshot = root / "latest_dashboard_snapshot.json"
            pipeline_snapshot = root / "daily_pipeline_snapshot.json"
            write_paper_account_fixture(stale_paper_dir, action_count=0, satellite_weight=0.0)
            stale_metrics = json.loads((stale_paper_dir / "metrics.json").read_text(encoding="utf-8"))
            stale_metrics["latest_date"] = "2026-06-12"
            (stale_paper_dir / "metrics.json").write_text(json.dumps(stale_metrics), encoding="utf-8")
            write_paper_account_fixture(fresh_paper_dir, action_count=0, satellite_weight=0.0)
            fresh_metrics_path = fresh_paper_dir / "metrics.json"
            write_dashboard_snapshot(dashboard_snapshot, stale_paper_dir, paper_latest_date="2026-06-12")
            pipeline_snapshot.write_text(
                json.dumps(
                    {
                        "paper_account_output_dir": str(fresh_paper_dir),
                        "paper_latest_date": "2026-06-15",
                    }
                ),
                encoding="utf-8",
            )

            result = run_live_preflight(
                dashboard_snapshot=dashboard_snapshot,
                pipeline_snapshot=pipeline_snapshot,
                output_dir=root / "live_preflight",
            )

            self.assertEqual(result.snapshot["paper_account_dir"], str(fresh_paper_dir))
            self.assertEqual(result.snapshot["paper_metrics_path"], str(fresh_metrics_path))
            self.assertEqual(result.snapshot["paper_latest_date"], "2026-06-15")

    def test_live_preflight_blocks_on_stale_gates_review_queue_and_problem_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_dir = root / "paper"
            dashboard_snapshot = root / "latest_dashboard_snapshot.json"
            daily_status = root / "daily_run_status_snapshot.json"
            write_paper_account_fixture(paper_dir, action_count=2, satellite_weight=0.05)
            write_dashboard_snapshot(
                dashboard_snapshot,
                paper_dir,
                market_cache_status="stale",
                trigger_freshness_status="stale",
                trading_day_gate_status="trading_day_data_not_ready",
                after_close_data_status="not_ready",
            )
            daily_status.write_text(json.dumps({"problem_state": True, "run_state": "failed"}), encoding="utf-8")

            result = run_live_preflight(
                dashboard_snapshot=dashboard_snapshot,
                daily_run_status_snapshot=daily_status,
                output_dir=root / "live_preflight",
            )

            self.assertEqual(result.snapshot["decision"], "blocked")
            self.assertIn("market_cache_status=stale", result.snapshot["blocking_items"])
            self.assertIn("trigger_freshness_status=stale", result.snapshot["blocking_items"])
            self.assertIn("trading_day_gate_status=trading_day_data_not_ready", result.snapshot["blocking_items"])
            self.assertIn("daily_run_problem_state=true", result.snapshot["blocking_items"])
            self.assertEqual(result.snapshot["stock_target_review_action_count"], 2)
            self.assertEqual(result.snapshot["stock_target_review_blocking_action_count"], 2)
            self.assertEqual(result.snapshot["stock_target_review_unknown_action_count"], 2)
            self.assertIn("stock_target_review_blocking_action_count=2", result.snapshot["blocking_items"])

    def test_live_preflight_ignores_self_referential_daily_run_problem_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_dir = root / "paper"
            dashboard_snapshot = root / "latest_dashboard_snapshot.json"
            daily_status = root / "daily_run_status_snapshot.json"
            write_paper_account_fixture(paper_dir, action_count=0, satellite_weight=0.0)
            write_dashboard_snapshot(dashboard_snapshot, paper_dir)
            daily_status.write_text(
                json.dumps({"problem_state": True, "run_state": "live_preflight_blocked"}),
                encoding="utf-8",
            )

            result = run_live_preflight(
                dashboard_snapshot=dashboard_snapshot,
                daily_run_status_snapshot=daily_status,
                output_dir=root / "live_preflight",
            )

            self.assertEqual(result.snapshot["decision"], "ready_for_watch_only_pre_stage")
            self.assertNotIn("daily_run_problem_state=true", result.snapshot["blocking_items"])

    def test_live_preflight_blocks_on_live_shadow_review_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper_dir = root / "paper"
            dashboard_snapshot = root / "latest_dashboard_snapshot.json"
            decisions_path = root / "live_shadow_review_decisions.csv"
            write_paper_account_fixture(paper_dir, action_count=0, satellite_weight=0.05)
            write_dashboard_snapshot(
                dashboard_snapshot,
                paper_dir,
                paper_latest_regime="risk_on",
                paper_latest_satellite_weight=0.05,
            )
            pd.DataFrame(
                [
                    {"code": "002142", "review_decision": "blocked_by_tracking_rule"},
                    {"code": "002216", "review_decision": "manual_considered"},
                    {"code": "000895", "review_decision": "watch_only"},
                ]
            ).to_csv(decisions_path, index=False)

            result = run_live_preflight(
                dashboard_snapshot=dashboard_snapshot,
                live_shadow_review_decisions_file=decisions_path,
                output_dir=root / "live_preflight",
            )

            self.assertEqual(result.snapshot["decision"], "blocked")
            self.assertEqual(result.snapshot["live_shadow_review_decision_count"], 3)
            self.assertEqual(result.snapshot["live_shadow_review_blocking_decision_count"], 1)
            self.assertEqual(result.snapshot["live_shadow_review_monitor_decision_count"], 1)
            self.assertEqual(result.snapshot["live_shadow_review_unknown_decision_count"], 0)
            self.assertEqual(result.snapshot["live_shadow_review_decisions_path"], str(decisions_path))
            self.assertIn("live_shadow_review_blocking_decision_count=1", result.snapshot["blocking_items"])
            self.assertIn("live_shadow_review_monitor_decision_count=1", result.snapshot["monitor_items"])
            self.assertIn("Live-shadow review decisions", result.report_path.read_text(encoding="utf-8"))
