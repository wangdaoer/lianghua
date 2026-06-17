from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_etf_lab.allocator_switch import run_allocator_switch_readiness
from quant_etf_lab.cli import build_parser


class AllocatorSwitchTests(unittest.TestCase):
    def test_allocator_switch_readiness_cli(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "allocator-switch-readiness",
                "--default-snapshot",
                "outputs/default/daily_pipeline_snapshot.json",
                "--candidate-snapshot",
                "outputs/candidate/daily_pipeline_snapshot.json",
                "--output-dir",
                "outputs/research/allocator_switch",
                "--default-label",
                "Default quality-v2",
                "--candidate-label",
                "Candidate source-selection 6m",
                "--outcome-analysis-path",
                "outputs/research/paper/stock_target_review_outcome_analysis.json",
            ]
        )

        self.assertEqual(args.command, "allocator-switch-readiness")
        self.assertEqual(args.default_snapshot, "outputs/default/daily_pipeline_snapshot.json")
        self.assertEqual(args.candidate_snapshot, "outputs/candidate/daily_pipeline_snapshot.json")
        self.assertEqual(args.output_dir, "outputs/research/allocator_switch")
        self.assertEqual(args.default_label, "Default quality-v2")
        self.assertEqual(args.candidate_label, "Candidate source-selection 6m")
        self.assertEqual(
            args.outcome_analysis_path,
            "outputs/research/paper/stock_target_review_outcome_analysis.json",
        )

    def test_allocator_switch_readiness_compares_pipeline_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_snapshot = root / "default_snapshot.json"
            candidate_snapshot = root / "candidate_snapshot.json"
            output_dir = root / "switch_readiness"
            default_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-12",
                        "pipeline_report_path": "default/daily_pipeline.md",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "info",
                        "alert_count": 3,
                        "alert_action_stage": "monitor",
                        "pipeline_next_step_stage": "accumulate_outcome_samples",
                        "promotion_decision": "promote_candidate",
                        "promotion_support_group_count": 2,
                        "promotion_evidence_support_count": 1,
                        "paper_final_equity": 1432764.52,
                        "paper_total_return": 0.43276452,
                        "paper_max_drawdown": -0.1151,
                        "paper_sharpe": 1.046,
                        "paper_current_drawdown": -0.0267,
                        "paper_latest_regime": "risk_off",
                        "paper_latest_candidate": "sat20_ma60_drop03_unfiltered",
                        "paper_latest_core_weight": 1.0,
                        "paper_latest_satellite_weight": 0.0,
                        "paper_stock_target_count": 16,
                        "paper_stock_target_review_required_count": 3,
                        "paper_stock_target_review_drawdown_count": 3,
                        "paper_stock_target_review_action_count": 0,
                        "paper_stock_target_review_outcome_analysis_status": "waiting_for_evaluable_returns",
                        "paper_stock_target_review_outcome_analysis_ready_horizon_count": 0,
                        "paper_stock_target_review_outcome_calendar_next_action_date": "2026-06-15",
                        "paper_stock_target_review_outcome_calendar_next_action_horizon": "1d",
                        "history_review_health_state": "ok",
                        "model_audit_status": "ok",
                        "model_audit_walk_forward_action_items": 0,
                    }
                ),
                encoding="utf-8",
            )
            candidate_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-12",
                        "pipeline_report_path": "candidate/daily_pipeline.md",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "info",
                        "alert_count": 3,
                        "alert_action_stage": "monitor",
                        "pipeline_next_step_stage": "accumulate_outcome_samples",
                        "promotion_decision": "promote_candidate",
                        "promotion_support_group_count": 2,
                        "promotion_evidence_support_count": 1,
                        "paper_final_equity": 1494744.88,
                        "paper_total_return": 0.49474488,
                        "paper_max_drawdown": -0.0958,
                        "paper_sharpe": 1.165,
                        "paper_current_drawdown": -0.0242,
                        "paper_latest_regime": "risk_off",
                        "paper_latest_candidate": "quality__sat25_ma60_drop03_unfiltered",
                        "paper_latest_core_weight": 1.0,
                        "paper_latest_satellite_weight": 0.0,
                        "paper_stock_target_count": 16,
                        "paper_stock_target_review_required_count": 3,
                        "paper_stock_target_review_drawdown_count": 3,
                        "paper_stock_target_review_action_count": 0,
                        "paper_stock_target_review_outcome_analysis_status": "waiting_for_evaluable_returns",
                        "paper_stock_target_review_outcome_analysis_ready_horizon_count": 0,
                        "paper_stock_target_review_outcome_calendar_next_action_date": "2026-06-15",
                        "paper_stock_target_review_outcome_calendar_next_action_horizon": "1d",
                        "history_review_health_state": "ok",
                        "model_audit_status": "ok",
                        "model_audit_walk_forward_action_items": 0,
                    }
                ),
                encoding="utf-8",
            )

            result = run_allocator_switch_readiness(
                default_snapshot=default_snapshot,
                candidate_snapshot=candidate_snapshot,
                output_dir=output_dir,
                default_label="Default quality-v2",
                candidate_label="Candidate source-selection 6m",
            )

            self.assertEqual(result.snapshot["decision"], "ready_for_controlled_default_switch")
            self.assertAlmostEqual(result.snapshot["candidate_return_edge"], 0.06198036, places=6)
            self.assertAlmostEqual(result.snapshot["candidate_drawdown_change"], 0.0193, places=4)
            self.assertAlmostEqual(result.snapshot["candidate_sharpe_edge"], 0.119, places=6)
            self.assertEqual(result.snapshot["shared_alert_level"], "info")
            self.assertEqual(result.snapshot["blocking_items"], [])
            self.assertEqual(result.snapshot["risk_budget_decision"], "wait_for_outcome_samples")
            self.assertEqual(result.snapshot["next_outcome_review_date"], "2026-06-15")
            self.assertEqual(result.snapshot["next_outcome_review_horizon"], "1d")
            self.assertTrue(result.comparison_path.exists())
            self.assertTrue(result.snapshot_path.exists())
            self.assertTrue(result.report_path.exists())
            comparison = pd.read_csv(result.comparison_path)
            self.assertIn("Paper total return", set(comparison["item"]))
            report_text = result.report_path.read_text(encoding="utf-8")
            self.assertIn("Default vs Candidate", report_text)
            self.assertIn("ready_for_controlled_default_switch", report_text)
            self.assertIn("wait_for_outcome_samples", report_text)

    def test_allocator_switch_readiness_uses_outcome_analysis_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_snapshot = root / "default_snapshot.json"
            candidate_snapshot = root / "candidate_snapshot.json"
            outcome_analysis = root / "stock_target_review_outcome_analysis.json"
            output_dir = root / "switch_readiness"
            shared_snapshot = {
                "as_of_date": "2026-06-12",
                "dashboard_posture": "core_base_watch_allocator_gate",
                "trading_day_gate_status": "trading_day_data_ready",
                "after_close_data_status": "ready",
                "alert_level": "info",
                "alert_count": 3,
                "alert_action_stage": "monitor",
                "pipeline_next_step_stage": "review_outcome_analysis",
                "promotion_decision": "promote_candidate",
                "promotion_support_group_count": 2,
                "promotion_evidence_support_count": 1,
                "paper_stock_target_review_action_count": 0,
                "history_review_health_state": "ok",
                "model_audit_status": "ok",
                "model_audit_walk_forward_action_items": 0,
            }
            default_snapshot.write_text(
                json.dumps(
                    {
                        **shared_snapshot,
                        "pipeline_report_path": "default/daily_pipeline.md",
                        "paper_total_return": 0.40,
                        "paper_max_drawdown": -0.12,
                        "paper_sharpe": 1.00,
                        "paper_latest_regime": "risk_off",
                        "paper_latest_core_weight": 1.0,
                        "paper_latest_satellite_weight": 0.0,
                        "paper_stock_target_review_outcome_analysis_status": "waiting_for_evaluable_returns",
                        "paper_stock_target_review_outcome_analysis_ready_horizon_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            candidate_snapshot.write_text(
                json.dumps(
                    {
                        **shared_snapshot,
                        "pipeline_report_path": "candidate/daily_pipeline.md",
                        "paper_total_return": 0.48,
                        "paper_max_drawdown": -0.10,
                        "paper_sharpe": 1.12,
                        "paper_latest_regime": "risk_off",
                        "paper_latest_core_weight": 1.0,
                        "paper_latest_satellite_weight": 0.0,
                        "paper_stock_target_review_outcome_analysis_status": "waiting_for_evaluable_returns",
                        "paper_stock_target_review_outcome_analysis_ready_horizon_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            outcome_analysis.write_text(
                json.dumps(
                    {
                        "analysis_status": "ready_for_review",
                        "ready_horizon_count": 2,
                        "ready_horizons": ["1d", "5d"],
                        "horizon_readiness": {
                            "1d": {
                                "status": "ready_for_review",
                                "evaluable_count": 24,
                                "min_evaluable": 20,
                                "qualified_group_count": 6,
                                "min_group_evaluable": 5,
                            },
                            "5d": {
                                "status": "ready_for_review",
                                "evaluable_count": 22,
                                "min_evaluable": 20,
                                "qualified_group_count": 5,
                                "min_group_evaluable": 5,
                            },
                        },
                        "best_groups": {
                            "1d": {
                                "dimension": "review_status",
                                "group_value": "watch",
                                "evaluable_count_1d": 14,
                                "win_rate_1d": 0.64,
                                "avg_return_1d": 0.012,
                            }
                        },
                        "worst_groups": {
                            "1d": {
                                "dimension": "review_status",
                                "group_value": "resolved",
                                "evaluable_count_1d": 10,
                                "win_rate_1d": 0.30,
                                "avg_return_1d": -0.009,
                            }
                        },
                        "analysis_path": "paper/stock_target_review_outcome_analysis.csv",
                        "analysis_report_path": "paper/stock_target_review_outcome_analysis.md",
                    }
                ),
                encoding="utf-8",
            )

            result = run_allocator_switch_readiness(
                default_snapshot=default_snapshot,
                candidate_snapshot=candidate_snapshot,
                output_dir=output_dir,
                default_label="Default quality-v2",
                candidate_label="Candidate source-selection 6m",
                outcome_analysis_path=outcome_analysis,
            )

            self.assertEqual(result.snapshot["decision"], "ready_for_controlled_default_switch")
            self.assertEqual(result.snapshot["risk_budget_decision"], "review_outcome_samples")
            self.assertEqual(result.snapshot["outcome_analysis_status"], "ready_for_review")
            self.assertEqual(result.snapshot["outcome_ready_horizon_count"], 2)
            self.assertEqual(result.snapshot["outcome_ready_horizons"], ["1d", "5d"])
            self.assertEqual(result.snapshot["outcome_analysis_path"], str(outcome_analysis))
            self.assertEqual(result.snapshot["outcome_horizon_readiness"]["1d"]["evaluable_count"], 24)
            self.assertIn("Outcome Readiness", result.report_path.read_text(encoding="utf-8"))

    def test_allocator_switch_readiness_blocks_risk_budget_when_switch_readiness_blocks_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_snapshot = root / "default_snapshot.json"
            candidate_snapshot = root / "candidate_snapshot.json"
            shared = {
                "as_of_date": "2026-06-12",
                "pipeline_report_path": "candidate/daily_pipeline.md",
                "dashboard_posture": "core_base_watch_allocator_gate",
                "trading_day_gate_status": "trading_day_data_ready",
                "after_close_data_status": "ready",
                "alert_level": "warning",
                "alert_count": 3,
                "alert_action_stage": "monitor",
                "pipeline_next_step_stage": "accumulate_outcome_samples",
                "promotion_support_group_count": 2,
                "promotion_evidence_support_count": 1,
                "paper_final_equity": 1200000.0,
                "paper_total_return": 0.45,
                "paper_max_drawdown": -0.095,
                "paper_sharpe": 1.15,
                "paper_current_drawdown": -0.022,
                "paper_latest_regime": "risk_off",
                "paper_latest_candidate": "quality__sat25_ma60_drop03_unfiltered",
                "paper_latest_core_weight": 1.0,
                "paper_latest_satellite_weight": 0.0,
                "paper_stock_target_count": 16,
                "paper_stock_target_review_required_count": 3,
                "paper_stock_target_review_drawdown_count": 1,
                "paper_stock_target_review_action_count": 0,
                "paper_stock_target_review_outcome_analysis_status": "waiting_for_evaluable_returns",
                "paper_stock_target_review_outcome_analysis_ready_horizon_count": 0,
                "paper_stock_target_review_outcome_calendar_next_action_date": "2026-06-15",
                "paper_stock_target_review_outcome_calendar_next_action_horizon": "1d",
                "history_review_health_state": "ok",
                "model_audit_status": "ok",
                "model_audit_walk_forward_action_items": 0,
            }
            default_snapshot.write_text(
                json.dumps(
                    {
                        **shared,
                        "paper_total_return": 0.40,
                        "paper_max_drawdown": -0.12,
                        "paper_sharpe": 1.0,
                        "promotion_decision": "promote_candidate",
                        "pipeline_report_path": "default/daily_pipeline.md",
                    }
                ),
                encoding="utf-8",
            )
            candidate_snapshot.write_text(
                json.dumps(
                    {
                        **shared,
                        "paper_total_return": 0.48,
                        "paper_max_drawdown": -0.10,
                        "paper_sharpe": 1.17,
                        "promotion_decision": "watch_candidate",
                        "model_audit_walk_forward_action_items": 1,
                    }
                ),
                encoding="utf-8",
            )

            result = run_allocator_switch_readiness(
                default_snapshot=default_snapshot,
                candidate_snapshot=candidate_snapshot,
                output_dir=root / "switch_readiness",
            )

            self.assertEqual(result.snapshot["decision"], "keep_parallel_watch")
            self.assertEqual(result.snapshot["risk_budget_decision"], "blocked_by_switch_readiness")
            self.assertIn("promotion_decision=watch_candidate", result.snapshot["blocking_items"])
            self.assertIn("candidate_model_audit_actions=1", result.snapshot["blocking_items"])
            self.assertAlmostEqual(result.snapshot["candidate_return_edge"], 0.08, places=6)

    def test_allocator_switch_readiness_keeps_parallel_when_headline_metrics_do_not_improve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_snapshot = root / "default_snapshot.json"
            candidate_snapshot = root / "candidate_snapshot.json"
            default_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-12",
                        "pipeline_report_path": "default/daily_pipeline.md",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "info",
                        "paper_total_return": 0.45,
                        "paper_max_drawdown": -0.11,
                        "paper_sharpe": 1.02,
                        "promotion_decision": "promote_candidate",
                    }
                ),
                encoding="utf-8",
            )
            candidate_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-12",
                        "pipeline_report_path": "candidate/daily_pipeline.md",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "info",
                        "paper_total_return": 0.40,
                        "paper_max_drawdown": -0.12,
                        "paper_sharpe": 1.00,
                        "promotion_decision": "promote_candidate",
                        "history_review_health_state": "ok",
                        "model_audit_walk_forward_action_items": 0,
                    }
                ),
                encoding="utf-8",
            )

            result = run_allocator_switch_readiness(
                default_snapshot=default_snapshot,
                candidate_snapshot=candidate_snapshot,
                output_dir=root / "switch_readiness",
            )

            self.assertEqual(result.snapshot["decision"], "keep_parallel_watch")
            self.assertAlmostEqual(result.snapshot["candidate_return_edge"], -0.05, places=6)

    def test_allocator_switch_readiness_uses_candidate_embedded_outcome_analysis_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_snapshot = root / "default_snapshot.json"
            candidate_snapshot = root / "candidate_snapshot.json"
            outcome_analysis = root / "candidate_stock_target_review_outcome_analysis.json"
            default_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-12",
                        "pipeline_report_path": "default/daily_pipeline.md",
                        "promotion_decision": "promote_candidate",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "info",
                        "paper_total_return": 0.42,
                        "paper_max_drawdown": -0.12,
                        "paper_sharpe": 1.05,
                        "paper_stock_target_review_action_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            candidate_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-12",
                        "pipeline_report_path": "candidate/daily_pipeline.md",
                        "promotion_decision": "promote_candidate",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "info",
                        "paper_total_return": 0.50,
                        "paper_max_drawdown": -0.10,
                        "paper_sharpe": 1.15,
                        "paper_stock_target_review_action_count": 0,
                        "paper_stock_target_review_outcome_analysis_json_path": str(outcome_analysis),
                    }
                ),
                encoding="utf-8",
            )
            outcome_analysis.write_text(
                json.dumps(
                    {
                        "analysis_status": "ready_for_review",
                        "ready_horizon_count": 1,
                        "ready_horizons": ["1d"],
                        "horizon_readiness": {
                            "1d": {"status": "ready_for_review", "evaluable_count": 20, "min_evaluable": 20},
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = run_allocator_switch_readiness(
                default_snapshot=default_snapshot,
                candidate_snapshot=candidate_snapshot,
                output_dir=root / "switch_readiness",
            )

            self.assertEqual(result.snapshot["outcome_analysis_path"], str(outcome_analysis))
            self.assertEqual(result.snapshot["risk_budget_decision"], "review_outcome_samples")
            self.assertEqual(result.snapshot["outcome_ready_horizon_count"], 1)

    def test_allocator_switch_readiness_fails_when_snapshot_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_snapshot = root / "default_snapshot.json"
            candidate_snapshot = root / "candidate_snapshot.json"
            candidate_snapshot.write_text(json.dumps({"promotion_decision": "promote_candidate"}), encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                run_allocator_switch_readiness(
                    default_snapshot=default_snapshot,
                    candidate_snapshot=candidate_snapshot,
                    output_dir=root / "switch_readiness",
                )

    def test_allocator_switch_readiness_embedded_outcome_analysis_without_ready_horizons_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_snapshot = root / "default_snapshot.json"
            candidate_snapshot = root / "candidate_snapshot.json"
            outcome_analysis = root / "candidate_stock_target_review_outcome_analysis.json"
            shared = {
                "as_of_date": "2026-06-12",
                "pipeline_report_path": "candidate/daily_pipeline.md",
                "dashboard_posture": "core_base_watch_allocator_gate",
                "trading_day_gate_status": "trading_day_data_ready",
                "after_close_data_status": "ready",
                "alert_level": "info",
                "alert_count": 1,
                "alert_action_stage": "monitor",
                "pipeline_next_step_stage": "accumulate_outcome_samples",
                "promotion_decision": "promote_candidate",
                "history_review_health_state": "ok",
                "model_audit_status": "ok",
                "model_audit_walk_forward_action_items": 0,
                "paper_stock_target_review_action_count": 0,
            }
            default_snapshot.write_text(
                json.dumps(
                    {
                        **shared,
                        "pipeline_report_path": "default/daily_pipeline.md",
                        "paper_total_return": 0.41,
                        "paper_max_drawdown": -0.12,
                        "paper_sharpe": 1.0,
                        "paper_latest_regime": "risk_on",
                        "paper_latest_core_weight": 1.0,
                        "paper_latest_satellite_weight": 0.0,
                        "paper_stock_target_review_outcome_analysis_status": "ready_for_review",
                    }
                ),
                encoding="utf-8",
            )
            candidate_payload = {
                **shared,
                "paper_total_return": 0.48,
                "paper_max_drawdown": -0.09,
                "paper_sharpe": 1.1,
                "paper_latest_regime": "risk_on",
                "paper_latest_core_weight": 0.8,
                "paper_latest_satellite_weight": 0.2,
                "paper_stock_target_review_outcome_analysis_json_path": str(outcome_analysis),
                "paper_stock_target_review_outcome_calendar_next_action_date": "2026-06-15",
                "paper_stock_target_review_outcome_calendar_next_action_horizon": "1d",
                "paper_stock_target_review_outcome_analysis_status": "waiting_for_evaluable_returns",
                "paper_stock_target_review_outcome_analysis_ready_horizon_count": 0,
            }
            candidate_snapshot.write_text(json.dumps(candidate_payload), encoding="utf-8")
            outcome_analysis.write_text(
                json.dumps(
                    {
                        "analysis_status": "waiting_for_evaluable_returns",
                        "ready_horizon_count": 0,
                        "ready_horizons": [],
                        "horizon_readiness": {
                            "1d": {
                                "status": "waiting_for_evaluable_returns",
                                "evaluable_count": 14,
                                "min_evaluable": 20,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = run_allocator_switch_readiness(
                default_snapshot=default_snapshot,
                candidate_snapshot=candidate_snapshot,
                output_dir=root / "switch_readiness",
                default_label="Default quality-v2",
                candidate_label="Candidate source-selection 6m",
            )

            self.assertEqual(result.snapshot["outcome_analysis_path"], str(outcome_analysis))
            self.assertEqual(result.snapshot["outcome_ready_horizon_count"], 0)
            self.assertEqual(result.snapshot["outcome_ready_horizons"], [])
            self.assertEqual(result.snapshot["risk_budget_decision"], "wait_for_outcome_samples")
            self.assertEqual(result.snapshot["outcome_analysis_status"], "waiting_for_evaluable_returns")
            self.assertEqual(result.snapshot["blocking_items"], [])
            report_text = result.report_path.read_text(encoding="utf-8")
            self.assertIn("1d", report_text)

    def test_allocator_switch_readiness_blocks_candidate_intervention(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_snapshot = root / "default_snapshot.json"
            candidate_snapshot = root / "candidate_snapshot.json"
            baseline = {
                "as_of_date": "2026-06-12",
                "pipeline_report_path": "default/daily_pipeline.md",
                "dashboard_posture": "core_base_watch_allocator_gate",
                "trading_day_gate_status": "trading_day_data_ready",
                "after_close_data_status": "ready",
                "alert_level": "info",
                "alert_count": 1,
                "alert_action_stage": "monitor",
                "pipeline_next_step_stage": "review_outcome_analysis",
                "promotion_decision": "promote_candidate",
                "history_review_health_state": "ok",
                "model_audit_status": "ok",
                "model_audit_walk_forward_action_items": 0,
                "paper_total_return": 0.40,
                "paper_max_drawdown": -0.12,
                "paper_sharpe": 1.02,
                "paper_latest_candidate": "core",
                "paper_latest_regime": "risk_on",
                "paper_latest_core_weight": 1.0,
                "paper_latest_satellite_weight": 0.0,
                "paper_stock_target_review_action_count": 0,
                "paper_stock_target_review_outcome_analysis_status": "ready_for_review",
                "paper_stock_target_review_outcome_analysis_ready_horizon_count": 1,
            }
            candidate = {
                **baseline,
                "pipeline_report_path": "candidate/daily_pipeline.md",
                "paper_total_return": 0.46,
                "paper_max_drawdown": -0.10,
                "paper_sharpe": 1.15,
                "pipeline_user_intervention_required": True,
                "paper_stock_target_review_outcome_analysis_status": "ready_for_review",
                "paper_stock_target_review_outcome_analysis_ready_horizon_count": 2,
            }
            default_snapshot.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_snapshot.write_text(json.dumps(candidate), encoding="utf-8")

            result = run_allocator_switch_readiness(
                default_snapshot=default_snapshot,
                candidate_snapshot=candidate_snapshot,
                output_dir=root / "switch_readiness",
            )

            self.assertEqual(result.snapshot["decision"], "keep_parallel_watch")
            self.assertEqual(result.snapshot["risk_budget_decision"], "blocked_by_switch_readiness")
            self.assertIn("candidate_user_intervention_required=true", result.snapshot["blocking_items"])
