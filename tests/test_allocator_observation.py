from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_etf_lab.allocator_observation import run_allocator_observation
from quant_etf_lab.cli import build_parser


class AllocatorObservationTests(unittest.TestCase):
    def test_allocator_observation_cli(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "allocator-observation",
                "--pipeline-snapshot",
                "outputs/research/daily_pipeline/daily_pipeline_snapshot.json",
                "--output-dir",
                "outputs/research/allocator_observation",
                "--baseline-label",
                "Rollback quality-v2",
            ]
        )

        self.assertEqual(args.command, "allocator-observation")
        self.assertEqual(args.pipeline_snapshot, "outputs/research/daily_pipeline/daily_pipeline_snapshot.json")
        self.assertEqual(args.output_dir, "outputs/research/allocator_observation")
        self.assertEqual(args.baseline_label, "Rollback quality-v2")

    def test_allocator_observation_waits_for_outcome_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline_snapshot = root / "daily_pipeline_snapshot.json"
            output_dir = root / "allocator_observation"
            pipeline_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-12",
                        "pipeline_report_path": "outputs/research/daily_pipeline/daily_pipeline.md",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "info",
                        "alert_count": 3,
                        "alert_action_stage": "monitor",
                        "history_review_health_state": "ok",
                        "model_audit_status": "ok",
                        "model_audit_walk_forward_action_items": 0,
                        "promotion_decision": "promote_candidate",
                        "promotion_support_group_count": 2,
                        "promotion_min_sensitivity_support": 2,
                        "promotion_evidence_support_count": 1,
                        "paper_latest_regime": "risk_off",
                        "paper_latest_candidate": "quality__sat25_ma60_drop03_unfiltered",
                        "paper_latest_core_weight": 1.0,
                        "paper_latest_satellite_weight": 0.0,
                        "paper_total_return": 0.49474488,
                        "paper_max_drawdown": -0.0958,
                        "paper_sharpe": 1.165,
                        "paper_stock_target_review_required_count": 3,
                        "paper_stock_target_review_required_unreviewed_count": 0,
                        "paper_stock_target_review_action_count": 0,
                        "paper_stock_target_review_drawdown_count": 3,
                        "paper_stock_target_review_suppressed_layer_count": 1,
                        "paper_stock_target_review_outcome_analysis_status": "waiting_for_evaluable_returns",
                        "paper_stock_target_review_outcome_analysis_ready_horizon_count": 0,
                        "paper_stock_target_review_outcome_due_row_count": 0,
                        "paper_stock_target_review_outcome_calendar_next_action_date": "2026-06-15",
                        "paper_stock_target_review_outcome_calendar_next_action_horizon": "1d",
                        "paper_stock_target_review_outcome_maturity_next_1d_date": "2026-06-15",
                        "paper_stock_target_review_outcome_maturity_next_5d_date": "2026-06-19",
                        "paper_stock_target_review_outcome_maturity_next_10d_date": "2026-06-26",
                        "paper_stock_target_review_outcome_maturity_next_20d_date": "2026-07-10",
                    }
                ),
                encoding="utf-8",
            )

            result = run_allocator_observation(
                pipeline_snapshot=pipeline_snapshot,
                output_dir=output_dir,
                baseline_label="Rollback quality-v2",
            )

            self.assertEqual(result.snapshot["observation_status"], "waiting_for_outcome_samples")
            self.assertEqual(result.snapshot["next_action_stage"], "accumulate_outcome_samples")
            self.assertEqual(result.snapshot["next_review_date"], "2026-06-15")
            self.assertEqual(result.snapshot["next_review_horizon"], "1d")
            self.assertEqual(result.snapshot["blocking_items"], [])
            self.assertTrue(result.snapshot["rollback_baseline_retained"])
            self.assertTrue(result.checklist_path.exists())
            self.assertTrue(result.snapshot_path.exists())
            self.assertTrue(result.report_path.exists())
            checklist = pd.read_csv(result.checklist_path)
            self.assertIn("outcome_maturity", set(checklist["section"]))
            report_text = result.report_path.read_text(encoding="utf-8")
            self.assertIn("Post-Switch Allocator Observation", report_text)
            self.assertIn("waiting_for_outcome_samples", report_text)
            self.assertIn("2026-06-15", report_text)

    def test_allocator_observation_blocks_when_pipeline_gates_violate_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline_snapshot = root / "daily_pipeline_snapshot.json"
            output_dir = root / "allocator_observation"
            pipeline_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-12",
                        "pipeline_report_path": "outputs/research/daily_pipeline/daily_pipeline.md",
                        "dashboard_posture": "watchlist_only",
                        "trading_day_gate_status": "missing",
                        "after_close_data_status": "stale",
                        "alert_level": "warning",
                        "alert_count": 4,
                        "alert_action_stage": "monitor",
                        "history_review_health_state": "needs_attention",
                        "model_audit_status": "needs_attention",
                        "model_audit_walk_forward_action_items": 2,
                        "promotion_decision": "watch_candidate",
                        "promotion_support_group_count": 0,
                        "promotion_min_sensitivity_support": 0,
                        "promotion_evidence_support_count": 0,
                        "paper_latest_regime": "risk_on",
                        "paper_latest_candidate": "quality__sat25_ma60_drop03_unfiltered",
                        "paper_latest_core_weight": 0.7,
                        "paper_latest_satellite_weight": 0.3,
                        "paper_total_return": 0.12,
                        "paper_max_drawdown": -0.08,
                        "paper_sharpe": 1.08,
                        "paper_stock_target_review_required_count": 3,
                        "paper_stock_target_review_required_unreviewed_count": 1,
                        "paper_stock_target_review_action_count": 1,
                        "paper_stock_target_review_drawdown_count": 2,
                        "paper_stock_target_review_suppressed_layer_count": 1,
                        "paper_stock_target_review_manual_pending_count": 2,
                        "paper_stock_target_review_outcome_analysis_status": "waiting_for_evaluable_returns",
                        "paper_stock_target_review_outcome_analysis_ready_horizon_count": 0,
                        "paper_stock_target_review_outcome_due_row_count": 0,
                        "paper_stock_target_review_outcome_calendar_next_action_date": "2026-06-15",
                        "paper_stock_target_review_outcome_calendar_next_action_horizon": "1d",
                        "pipeline_user_intervention_required": True,
                    }
                ),
                encoding="utf-8",
            )

            result = run_allocator_observation(
                pipeline_snapshot=pipeline_snapshot,
                output_dir=output_dir,
                baseline_label="Rollback quality-v2",
            )

            self.assertEqual(result.snapshot["observation_status"], "review_blockers")
            self.assertEqual(result.snapshot["next_action_stage"], "resolve_blockers")
            self.assertEqual(result.snapshot["risk_budget_decision"], "hold_previous_risk_budget")
            self.assertIn("promotion_decision=watch_candidate", result.snapshot["blocking_items"])
            self.assertIn("dashboard_posture=watchlist_only", result.snapshot["blocking_items"])
            self.assertIn("trading_day_gate=missing", result.snapshot["blocking_items"])
            self.assertIn("after_close_data=stale", result.snapshot["blocking_items"])
            self.assertIn("alert_level=warning", result.snapshot["blocking_items"])
            self.assertIn("history_health=needs_attention", result.snapshot["blocking_items"])
            self.assertIn("pipeline_user_intervention_required=true", result.snapshot["blocking_items"])
            monitor_items = set(result.snapshot["monitor_items"])
            self.assertIn("reviewed_drawdown_targets=2", monitor_items)
            self.assertIn("suppressed_layer_targets=1", monitor_items)
            self.assertIn("manual_pending_rows=2", monitor_items)
            self.assertIn("outcome_samples_waiting", monitor_items)
            report_text = result.report_path.read_text(encoding="utf-8")
            self.assertIn("Blocking Items", report_text)
            self.assertIn("resolve_blockers", report_text)

    def test_allocator_observation_marks_outcomes_ready_for_review_when_counts_are_due(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline_snapshot = root / "daily_pipeline_snapshot.json"
            output_dir = root / "allocator_observation"
            pipeline_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-12",
                        "pipeline_report_path": "outputs/research/daily_pipeline/daily_pipeline.md",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "normal",
                        "history_review_health_state": "ok",
                        "promotion_decision": "promote_candidate",
                        "paper_latest_regime": "risk_on",
                        "paper_latest_core_weight": 0.8,
                        "paper_latest_satellite_weight": 0.2,
                        "paper_total_return": 0.15,
                        "paper_max_drawdown": -0.06,
                        "paper_sharpe": 1.20,
                        "paper_stock_target_review_outcome_analysis_status": "ready",
                        "paper_stock_target_review_outcome_analysis_ready_horizon_count": 2,
                        "paper_stock_target_review_outcome_due_row_count": 0,
                        "paper_stock_target_review_drawdown_count": 0,
                    }
                ),
                encoding="utf-8",
            )

            result = run_allocator_observation(
                pipeline_snapshot=pipeline_snapshot,
                output_dir=output_dir,
            )

            self.assertEqual(result.snapshot["observation_status"], "outcomes_ready_for_review")
            self.assertEqual(result.snapshot["next_action_stage"], "review_stock_target_outcomes")
            self.assertEqual(result.snapshot["risk_budget_decision"], "review_before_budget_change")

    def test_allocator_observation_goes_to_routine_when_risk_off_and_outcomes_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline_snapshot = root / "daily_pipeline_snapshot.json"
            output_dir = root / "allocator_observation"
            pipeline_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-14",
                        "pipeline_report_path": "outputs/research/daily_pipeline/daily_pipeline.md",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "info",
                        "history_review_health_state": "ok",
                        "model_audit_status": "ok",
                        "promotion_decision": "promote_candidate",
                        "paper_latest_regime": "risk_off",
                        "paper_latest_core_weight": 1.0,
                        "paper_latest_satellite_weight": 0.0,
                        "paper_stock_target_review_outcome_analysis_status": "ready",
                        "paper_stock_target_review_outcome_analysis_ready_horizon_count": 0,
                        "paper_stock_target_review_outcome_due_row_count": 0,
                    }
                ),
                encoding="utf-8",
            )

            result = run_allocator_observation(
                pipeline_snapshot=pipeline_snapshot,
                output_dir=output_dir,
            )

            self.assertEqual(result.snapshot["observation_status"], "routine_monitor")
            self.assertEqual(result.snapshot["next_action_stage"], "wait_for_risk_on_exposure")
            self.assertEqual(result.snapshot["risk_budget_decision"], "hold_default_core_base")
            self.assertIn("routine_monitor", result.report_path.read_text(encoding="utf-8"))

    def test_allocator_observation_becomes_eligible_for_budget_review_when_risk_on_and_satellite_active(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline_snapshot = root / "daily_pipeline_snapshot.json"
            output_dir = root / "allocator_observation"
            pipeline_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-14",
                        "pipeline_report_path": "outputs/research/daily_pipeline/daily_pipeline.md",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "normal",
                        "history_review_health_state": "ok",
                        "model_audit_status": "ok",
                        "promotion_decision": "promote_candidate",
                        "paper_latest_regime": "risk_on",
                        "paper_latest_core_weight": 0.85,
                        "paper_latest_satellite_weight": 0.15,
                        "paper_stock_target_review_outcome_analysis_status": "ready",
                        "paper_stock_target_review_outcome_analysis_ready_horizon_count": 0,
                        "paper_stock_target_review_outcome_due_row_count": 0,
                    }
                ),
                encoding="utf-8",
            )

            result = run_allocator_observation(
                pipeline_snapshot=pipeline_snapshot,
                output_dir=output_dir,
            )

            self.assertEqual(result.snapshot["observation_status"], "eligible_for_budget_review")
            self.assertEqual(result.snapshot["next_action_stage"], "review_satellite_budget")
            self.assertEqual(result.snapshot["risk_budget_decision"], "eligible_for_satellite_budget_review")
            report_text = result.report_path.read_text(encoding="utf-8")
            self.assertIn("eligible_for_budget_review", report_text)

    def test_allocator_observation_fails_when_pipeline_snapshot_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                run_allocator_observation(
                    pipeline_snapshot=root / "missing.json",
                    output_dir=root / "allocator_observation",
                )
