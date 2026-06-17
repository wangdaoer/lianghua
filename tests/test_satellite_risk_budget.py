from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser
from quant_etf_lab.satellite_risk_budget import run_satellite_risk_budget_review


class SatelliteRiskBudgetTests(unittest.TestCase):
    def test_satellite_risk_budget_review_cli(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "satellite-risk-budget-review",
                "--pipeline-snapshot",
                "outputs/research/daily_pipeline/daily_pipeline_snapshot.json",
                "--outcome-analysis-path",
                "outputs/research/paper/stock_target_review_outcome_analysis.json",
                "--network-lab-snapshot",
                "outputs/research/network_lab_snapshot.json",
                "--network-max-cluster-count-warning",
                "2",
                "--network-residual-mi-warning",
                "0.15",
                "--output-dir",
                "outputs/research/satellite_risk_budget_review",
                "--trial-satellite-budget",
                "0.05",
                "--max-satellite-budget",
                "0.20",
                "--min-overall-win-rate",
                "0.55",
                "--min-overall-avg-return",
                "0.0",
                "--max-worst-group-loss",
                "-0.05",
            ]
        )

        self.assertEqual(args.command, "satellite-risk-budget-review")
        self.assertEqual(args.pipeline_snapshot, "outputs/research/daily_pipeline/daily_pipeline_snapshot.json")
        self.assertEqual(args.outcome_analysis_path, "outputs/research/paper/stock_target_review_outcome_analysis.json")
        self.assertEqual(args.network_lab_snapshot, "outputs/research/network_lab_snapshot.json")
        self.assertEqual(args.network_max_cluster_count_warning, 2)
        self.assertAlmostEqual(args.network_residual_mi_warning, 0.15)
        self.assertEqual(args.output_dir, "outputs/research/satellite_risk_budget_review")
        self.assertEqual(args.trial_satellite_budget, 0.05)
        self.assertEqual(args.max_satellite_budget, 0.20)
        self.assertEqual(args.min_overall_win_rate, 0.55)
        self.assertEqual(args.min_overall_avg_return, 0.0)
        self.assertEqual(args.max_worst_group_loss, -0.05)

    def test_satellite_risk_budget_review_waits_for_outcome_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline_snapshot = root / "daily_pipeline_snapshot.json"
            outcome_analysis = root / "stock_target_review_outcome_analysis.json"
            output_dir = root / "satellite_risk_budget_review"
            pipeline_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-12",
                        "pipeline_report_path": "outputs/research/daily_pipeline/daily_pipeline.md",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "info",
                        "history_review_health_state": "ok",
                        "model_audit_walk_forward_action_items": 0,
                        "promotion_decision": "promote_candidate",
                        "paper_latest_regime": "risk_off",
                        "paper_latest_core_weight": 1.0,
                        "paper_latest_satellite_weight": 0.0,
                        "paper_stock_target_review_action_count": 0,
                        "paper_stock_target_review_outcome_calendar_next_action_date": "2026-06-15",
                        "paper_stock_target_review_outcome_calendar_next_action_horizon": "1d",
                    }
                ),
                encoding="utf-8",
            )
            outcome_analysis.write_text(
                json.dumps(
                    {
                        "analysis_status": "waiting_for_evaluable_returns",
                        "ready_horizon_count": 0,
                        "ready_horizons": [],
                        "horizon_readiness": {
                            "1d": {
                                "status": "waiting_for_evaluable_returns",
                                "evaluable_count": 0,
                                "min_evaluable": 20,
                                "qualified_group_count": 0,
                                "min_group_evaluable": 5,
                            }
                        },
                        "sample_warning": "No horizons have evaluable returns yet.",
                        "top_analysis_rows": [],
                    }
                ),
                encoding="utf-8",
            )

            result = run_satellite_risk_budget_review(
                pipeline_snapshot=pipeline_snapshot,
                outcome_analysis_path=outcome_analysis,
                output_dir=output_dir,
            )

            self.assertEqual(result.snapshot["risk_budget_decision"], "wait_for_outcome_samples")
            self.assertEqual(result.snapshot["recommended_satellite_budget"], 0.0)
            self.assertEqual(result.snapshot["selected_horizon"], None)
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.checklist_path.exists())
            self.assertTrue(result.snapshot_path.exists())

    def test_satellite_risk_budget_review_allows_small_trial_for_positive_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline_snapshot = root / "daily_pipeline_snapshot.json"
            outcome_analysis = root / "stock_target_review_outcome_analysis.json"
            output_dir = root / "satellite_risk_budget_review"
            pipeline_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-20",
                        "pipeline_report_path": "outputs/research/daily_pipeline/daily_pipeline.md",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "info",
                        "history_review_health_state": "ok",
                        "model_audit_walk_forward_action_items": 0,
                        "promotion_decision": "promote_candidate",
                        "paper_latest_regime": "risk_off",
                        "paper_latest_core_weight": 1.0,
                        "paper_latest_satellite_weight": 0.0,
                        "paper_stock_target_review_action_count": 0,
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
                            "1d": {
                                "status": "ready_for_review",
                                "evaluable_count": 24,
                                "min_evaluable": 20,
                                "qualified_group_count": 6,
                                "min_group_evaluable": 5,
                            }
                        },
                        "top_analysis_rows": [
                            {
                                "dimension": "overall",
                                "group_value": "all",
                                "evaluable_1d": 24,
                                "avg_return_1d": 0.012,
                                "win_rate_1d": 0.625,
                            }
                        ],
                        "best_groups": {
                            "1d": {
                                "dimension": "review_bucket",
                                "group_value": "routine_review",
                                "evaluable_count_1d": 16,
                                "avg_return_1d": 0.018,
                                "win_rate_1d": 0.6875,
                            }
                        },
                        "worst_groups": {
                            "1d": {
                                "dimension": "manual_status",
                                "group_value": "watch",
                                "evaluable_count_1d": 8,
                                "avg_return_1d": -0.012,
                                "win_rate_1d": 0.50,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = run_satellite_risk_budget_review(
                pipeline_snapshot=pipeline_snapshot,
                outcome_analysis_path=outcome_analysis,
                output_dir=output_dir,
                trial_satellite_budget=0.05,
                max_satellite_budget=0.20,
                min_overall_win_rate=0.55,
                min_overall_avg_return=0.0,
                max_worst_group_loss=-0.05,
            )

            self.assertEqual(result.snapshot["risk_budget_decision"], "eligible_for_small_satellite_trial")
            self.assertEqual(result.snapshot["recommended_satellite_budget"], 0.05)
            self.assertEqual(result.snapshot["selected_horizon"], "1d")
            self.assertEqual(result.snapshot["overall_win_rate"], 0.625)
            self.assertEqual(result.snapshot["overall_avg_return"], 0.012)
            checklist = pd.read_csv(result.checklist_path)
            self.assertIn("overall_edge", set(checklist["item"]))
            report_text = result.report_path.read_text(encoding="utf-8")
            self.assertIn("Satellite Risk Budget Review", report_text)
            self.assertIn("research-only", report_text)

    def test_satellite_risk_budget_review_holds_when_network_risk_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline_snapshot = root / "daily_pipeline_snapshot.json"
            outcome_analysis = root / "stock_target_review_outcome_analysis.json"
            network_snapshot = root / "network_lab_snapshot.json"
            pipeline_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-20",
                        "pipeline_report_path": "outputs/research/daily_pipeline/daily_pipeline.md",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "info",
                        "history_review_health_state": "ok",
                        "model_audit_walk_forward_action_items": 0,
                        "promotion_decision": "promote_candidate",
                        "paper_latest_regime": "risk_off",
                        "paper_latest_core_weight": 1.0,
                        "paper_latest_satellite_weight": 0.0,
                        "paper_stock_target_review_action_count": 0,
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
                            "1d": {
                                "status": "ready_for_review",
                                "evaluable_count": 24,
                                "min_evaluable": 20,
                                "qualified_group_count": 5,
                                "min_group_evaluable": 5,
                            }
                        },
                        "top_analysis_rows": [
                            {
                                "dimension": "overall",
                                "group_value": "all",
                                "evaluable_1d": 24,
                                "avg_return_1d": 0.012,
                                "win_rate_1d": 0.625,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            network_snapshot.write_text(
                json.dumps(
                    {
                        "network_snapshot_available": True,
                        "cluster_count": 1,
                        "loaded_symbol_count": 80,
                        "top_residual_mutual_information": 0.23,
                        "top_mutual_information": 0.40,
                        "snapshot_path": str(network_snapshot),
                    }
                ),
                encoding="utf-8",
            )

            result = run_satellite_risk_budget_review(
                pipeline_snapshot=pipeline_snapshot,
                outcome_analysis_path=outcome_analysis,
                network_lab_snapshot=network_snapshot,
                network_max_cluster_count_warning=1,
                network_residual_mi_warning=0.20,
                output_dir=root / "review",
            )

            self.assertEqual(result.snapshot["risk_budget_decision"], "hold_default_core_base")
            self.assertIn("Network monitor:", result.snapshot["risk_budget_reason"])
            self.assertIn("cluster_count=1 <= warning=1", result.snapshot["risk_budget_reason"])
            self.assertIn("top_residual_mi=0.2300 >= warning=0.2000", result.snapshot["risk_budget_reason"])

    def test_satellite_risk_budget_review_holds_for_negative_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline_snapshot = root / "daily_pipeline_snapshot.json"
            outcome_analysis = root / "stock_target_review_outcome_analysis.json"
            pipeline_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-20",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "info",
                        "history_review_health_state": "ok",
                        "model_audit_walk_forward_action_items": 0,
                        "promotion_decision": "promote_candidate",
                        "paper_latest_regime": "risk_off",
                        "paper_latest_core_weight": 1.0,
                        "paper_latest_satellite_weight": 0.0,
                        "paper_stock_target_review_action_count": 0,
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
                        "horizon_readiness": {"1d": {"status": "ready_for_review", "evaluable_count": 24}},
                        "top_analysis_rows": [
                            {
                                "dimension": "overall",
                                "group_value": "all",
                                "evaluable_1d": 24,
                                "avg_return_1d": -0.004,
                                "win_rate_1d": 0.4583,
                            }
                        ],
                        "worst_groups": {
                            "1d": {
                                "dimension": "manual_status",
                                "group_value": "watch",
                                "avg_return_1d": -0.025,
                                "win_rate_1d": 0.40,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = run_satellite_risk_budget_review(
                pipeline_snapshot=pipeline_snapshot,
                outcome_analysis_path=outcome_analysis,
                output_dir=root / "review",
            )

            self.assertEqual(result.snapshot["risk_budget_decision"], "hold_default_core_base")
            self.assertEqual(result.snapshot["risk_budget_reason"], "Overall outcome edge is not positive enough for a satellite budget trial.")
            self.assertEqual(result.snapshot["recommended_satellite_budget"], 0.0)

    def test_satellite_risk_budget_review_uses_pipeline_embedded_outcome_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline_snapshot = root / "daily_pipeline_snapshot.json"
            outcome_analysis = root / "stock_target_review_outcome_analysis.json"
            pipeline_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-20",
                        "pipeline_report_path": "outputs/research/daily_pipeline/daily_pipeline.md",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "info",
                        "history_review_health_state": "ok",
                        "model_audit_walk_forward_action_items": 0,
                        "promotion_decision": "promote_candidate",
                        "paper_latest_regime": "risk_off",
                        "paper_latest_core_weight": 1.0,
                        "paper_latest_satellite_weight": 0.0,
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
                            "1d": {
                                "status": "ready_for_review",
                                "evaluable_count": 24,
                                "min_evaluable": 20,
                                "qualified_group_count": 5,
                                "min_group_evaluable": 5,
                            }
                        },
                        "top_analysis_rows": [
                            {
                                "dimension": "overall",
                                "group_value": "all",
                                "evaluable_1d": 24,
                                "avg_return_1d": 0.009,
                                "win_rate_1d": 0.66,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = run_satellite_risk_budget_review(
                pipeline_snapshot=pipeline_snapshot,
                output_dir=root / "review",
                trial_satellite_budget=0.05,
            )

            self.assertEqual(result.snapshot["risk_budget_decision"], "eligible_for_small_satellite_trial")
            self.assertEqual(result.snapshot["outcome_analysis_path"], str(outcome_analysis))
            self.assertEqual(result.snapshot["selected_horizon"], "1d")
            self.assertAlmostEqual(result.snapshot["recommended_satellite_budget"], 0.05, places=6)

    def test_satellite_risk_budget_review_blocks_for_pipeline_gate_violations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline_snapshot = root / "daily_pipeline_snapshot.json"
            outcome_analysis = root / "stock_target_review_outcome_analysis.json"
            pipeline_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-20",
                        "pipeline_report_path": "outputs/research/daily_pipeline/daily_pipeline.md",
                        "dashboard_posture": "warning_posture",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "warning",
                        "history_review_health_state": "ok",
                        "model_audit_walk_forward_action_items": 1,
                        "promotion_decision": "promote_candidate",
                        "paper_latest_regime": "risk_off",
                        "paper_latest_core_weight": 1.0,
                        "paper_latest_satellite_weight": 0.07,
                        "paper_stock_target_review_action_count": 0,
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
                            "1d": {"status": "ready_for_review", "evaluable_count": 30, "min_evaluable": 20},
                        },
                        "top_analysis_rows": [
                            {
                                "dimension": "overall",
                                "group_value": "all",
                                "evaluable_1d": 30,
                                "avg_return_1d": 0.012,
                                "win_rate_1d": 0.67,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = run_satellite_risk_budget_review(
                pipeline_snapshot=pipeline_snapshot,
                outcome_analysis_path=outcome_analysis,
                output_dir=root / "review",
                trial_satellite_budget=0.10,
            )

            self.assertEqual(result.snapshot["risk_budget_decision"], "blocked_by_pipeline_gates")
            self.assertIn("dashboard_posture=warning_posture", result.snapshot["blocking_items"])
            self.assertIn("alert_level=warning", result.snapshot["blocking_items"])
            self.assertIn("model_audit_actions=1", result.snapshot["blocking_items"])
            self.assertEqual(result.snapshot["recommended_satellite_budget"], 0.07)

    def test_satellite_risk_budget_review_fails_when_outcome_analysis_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline_snapshot = root / "daily_pipeline_snapshot.json"
            pipeline_snapshot.write_text(
                json.dumps(
                    {
                        "as_of_date": "2026-06-20",
                        "pipeline_report_path": "outputs/research/daily_pipeline/daily_pipeline.md",
                        "dashboard_posture": "core_base_watch_allocator_gate",
                        "trading_day_gate_status": "trading_day_data_ready",
                        "after_close_data_status": "ready",
                        "alert_level": "info",
                        "history_review_health_state": "ok",
                        "model_audit_walk_forward_action_items": 0,
                        "promotion_decision": "promote_candidate",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                run_satellite_risk_budget_review(
                    pipeline_snapshot=pipeline_snapshot,
                    output_dir=root / "review",
                )
