from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser
from quant_etf_lab.source_decision import run_source_decision_review
from quant_etf_lab.source_guard_decomposition import run_source_guard_decomposition
from quant_etf_lab.source_risk_budget import run_source_risk_budget_review


class SourceDecisionTests(unittest.TestCase):
    def test_source_decision_review_cli(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "source-decision-review",
                "--source-dir",
                "outputs/portfolio_source_selection/candidate",
                "--baseline-dir",
                "outputs/portfolio_walk_forward/base",
                "--output-dir",
                "outputs/research/source_decision",
            ]
        )
        self.assertEqual(args.command, "source-decision-review")
        self.assertEqual(args.source_dir, "outputs/portfolio_source_selection/candidate")
        self.assertEqual(args.baseline_dir, "outputs/portfolio_walk_forward/base")
        self.assertEqual(args.output_dir, "outputs/research/source_decision")

    def test_source_guard_decomposition_cli(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "source-guard-decomposition",
                "--default-dir",
                "outputs/portfolio_source_selection/default",
                "--candidate-dir",
                "outputs/portfolio_source_selection/guarded",
                "--output-dir",
                "outputs/research/source_guard_decomposition",
            ]
        )
        self.assertEqual(args.command, "source-guard-decomposition")
        self.assertEqual(args.default_dir, "outputs/portfolio_source_selection/default")
        self.assertEqual(args.candidate_dir, "outputs/portfolio_source_selection/guarded")
        self.assertEqual(args.output_dir, "outputs/research/source_guard_decomposition")

    def test_source_risk_budget_review_cli(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "source-risk-budget-review",
                "--source-dir",
                "outputs/portfolio_source_selection/default",
                "--guard-decomposition-dir",
                "outputs/research/source_guard_decomposition",
                "--output-dir",
                "outputs/research/source_risk_budget",
                "--fragile-source-edge",
                "0.03",
                "--max-low-risk-satellite-weight",
                "0.05",
            ]
        )
        self.assertEqual(args.command, "source-risk-budget-review")
        self.assertEqual(args.source_dir, "outputs/portfolio_source_selection/default")
        self.assertEqual(args.guard_decomposition_dir, "outputs/research/source_guard_decomposition")
        self.assertEqual(args.output_dir, "outputs/research/source_risk_budget")
        self.assertAlmostEqual(args.fragile_source_edge, 0.03)
        self.assertAlmostEqual(args.max_low_risk_satellite_weight, 0.05)

    def test_source_decision_review_explains_selected_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            baseline_dir = root / "baseline"
            source_dir.mkdir()
            baseline_dir.mkdir()
            pd.DataFrame(
                {
                    "window": ["w1", "w2"],
                    "test_start": ["20240101", "20240701"],
                    "test_end": ["20240630", "20241231"],
                    "selected_candidate": ["reversal__sat20", "quality__sat20"],
                    "selected_source": ["reversal", "quality"],
                    "selected_allocation": ["sat20", "sat20"],
                    "selected_score": [0.80, 0.90],
                    "selection_score_type": ["validation", "validation"],
                    "validation_score": [0.80, 0.90],
                    "raw_best_candidate": ["reversal__sat20", "quality__sat20"],
                    "raw_best_source": ["reversal", "quality"],
                    "raw_best_score": [0.80, 0.90],
                    "source_validation_months": [6, 6],
                    "test_total_return": [0.05, -0.02],
                    "test_max_drawdown": [-0.03, -0.08],
                    "test_sharpe": [1.10, -0.20],
                }
            ).to_csv(source_dir / "portfolio_walk_forward_summary.csv", index=False)
            pd.DataFrame(
                {
                    "window": ["w1", "w1", "w1", "w2", "w2", "w2"],
                    "candidate": [
                        "quality__sat20",
                        "reversal__sat20",
                        "blend__sat20",
                        "quality__sat20",
                        "reversal__sat20",
                        "blend__sat20",
                    ],
                    "source_name": ["quality", "reversal", "blend", "quality", "reversal", "blend"],
                    "allocation_candidate": ["sat20", "sat20", "sat20", "sat20", "sat20", "sat20"],
                    "score": [0.72, 0.80, 0.72, 0.90, 0.65, 0.65],
                    "train_score": [0.70, 0.78, 0.70, 0.88, 0.60, 0.60],
                    "validation_score": [0.72, 0.80, 0.72, 0.90, 0.65, 0.65],
                }
            ).to_csv(source_dir / "portfolio_candidate_results.csv", index=False)
            pd.DataFrame(
                {
                    "window": ["w1", "w2"],
                    "test_total_return": [0.03, -0.01],
                    "test_max_drawdown": [-0.04, -0.06],
                    "test_sharpe": [0.80, -0.10],
                }
            ).to_csv(baseline_dir / "portfolio_walk_forward_summary.csv", index=False)

            result = run_source_decision_review(
                source_dir=source_dir,
                baseline_dir=baseline_dir,
                output_dir=root / "review",
            )

            self.assertTrue(result.report_path.exists())
            windows = pd.read_csv(result.windows_path)
            scores = pd.read_csv(result.source_scores_path)
            self.assertEqual(result.snapshot["window_count"], 2)
            self.assertEqual(len(windows), 2)
            self.assertEqual(windows.iloc[0]["selected_source"], "reversal")
            self.assertEqual(windows.iloc[0]["selection_reason"], "raw_best_source")
            self.assertAlmostEqual(windows.iloc[0]["score_edge_vs_second"], 0.08)
            self.assertAlmostEqual(windows.iloc[0]["return_edge_vs_baseline"], 0.02)
            self.assertIn("source_rank", scores.columns)
            self.assertIn("reversal", result.report_path.read_text(encoding="utf-8"))

    def test_source_guard_decomposition_identifies_source_switch_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_dir = root / "default"
            candidate_dir = root / "candidate"
            default_dir.mkdir()
            candidate_dir.mkdir()
            pd.DataFrame(
                {
                    "window": ["pf_04", "pf_05"],
                    "test_start": ["20250704", "20260104"],
                    "test_end": ["20260103", "20260615"],
                    "selected_candidate": ["blend__sat25", "quality__sat25"],
                    "selected_source": ["blend", "quality"],
                    "selected_score": [1.10, 0.60],
                    "raw_best_candidate": ["blend__sat25", "quality__sat25"],
                    "raw_best_source": ["blend", "quality"],
                    "raw_best_score": [1.10, 0.60],
                    "source_switch_margin": [0.0, 0.0],
                    "test_total_return": [0.05, 0.08],
                    "test_max_drawdown": [-0.09, -0.07],
                    "test_sharpe": [0.30, 1.20],
                }
            ).to_csv(default_dir / "portfolio_walk_forward_summary.csv", index=False)
            pd.DataFrame(
                {
                    "window": ["pf_04", "pf_05"],
                    "test_start": ["20250704", "20260104"],
                    "test_end": ["20260103", "20260615"],
                    "selected_candidate": ["core_only", "quality__sat25"],
                    "selected_source": ["core_only", "quality"],
                    "selected_score": [1.06, 0.60],
                    "raw_best_candidate": ["blend__sat25", "quality__sat25"],
                    "raw_best_source": ["blend", "quality"],
                    "raw_best_score": [1.08, 0.60],
                    "default_source": ["core_only", "core_only"],
                    "source_switch_margin": [0.03, 0.03],
                    "score_mode": ["capital_efficiency", "capital_efficiency"],
                    "score_mode_min_edge": [0.05, 0.05],
                    "score_mode_gate_applied": [False, False],
                    "test_total_return": [0.03, 0.08],
                    "test_max_drawdown": [-0.10, -0.07],
                    "test_sharpe": [0.10, 1.20],
                }
            ).to_csv(candidate_dir / "portfolio_walk_forward_summary.csv", index=False)

            result = run_source_guard_decomposition(
                default_dir=default_dir,
                candidate_dir=candidate_dir,
                output_dir=root / "decomposition",
            )

            windows = pd.read_csv(result.windows_path)
            pf04 = windows.loc[windows["window"] == "pf_04"].iloc[0]
            self.assertEqual(result.snapshot["changed_window_count"], 1)
            self.assertEqual(result.snapshot["source_switch_guard_count"], 1)
            self.assertEqual(result.snapshot["worsened_changed_window_count"], 1)
            self.assertEqual(pf04["changed_by"], "source_switch_guard")
            self.assertAlmostEqual(pf04["raw_edge_vs_candidate_selected"], 0.02)
            self.assertAlmostEqual(pf04["return_delta_vs_default"], -0.02)
            self.assertEqual(pf04["recommendation"], "review_risk_budget_not_default_source_switch")
            self.assertTrue(result.report_path.exists())

    def test_source_risk_budget_review_flags_low_risk_fragile_source_for_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            guard_dir = root / "guard"
            source_dir.mkdir()
            guard_dir.mkdir()
            pd.DataFrame(
                {
                    "window": ["pf_01", "pf_04", "pf_05"],
                    "test_start": ["20240104", "20250704", "20260104"],
                    "test_end": ["20240703", "20260103", "20260615"],
                    "selected_candidate": ["core_only", "blend__sat25", "quality__sat25"],
                    "selected_source": ["core_only", "blend", "quality"],
                    "selected_score": [0.10, 1.10, 0.80],
                    "raw_best_candidate": ["core_only", "blend__sat25", "quality__sat25"],
                    "raw_best_source": ["core_only", "blend", "quality"],
                    "raw_best_score": [0.10, 1.10, 0.80],
                    "test_total_return": [0.02, 0.05, 0.10],
                    "test_core_total_return": [0.02, 0.03, 0.04],
                    "test_excess_return_vs_core": [0.00, 0.02, 0.06],
                    "test_max_drawdown": [-0.02, -0.08, -0.05],
                    "test_core_max_drawdown": [-0.02, -0.09, -0.06],
                    "test_drawdown_change_vs_core": [0.00, 0.01, 0.01],
                    "test_sharpe": [0.20, 0.40, 1.20],
                    "test_average_satellite_weight": [0.00, 0.03, 0.12],
                    "test_satellite_active_day_ratio": [0.00, 0.12, 0.50],
                    "test_satellite_filter_off_day_ratio": [0.00, 0.85, 0.00],
                    "test_risk_on_day_ratio": [0.00, 0.30, 0.60],
                    "test_risk_off_day_ratio": [1.00, 0.70, 0.40],
                    "test_crash_day_ratio": [0.00, 0.00, 0.00],
                }
            ).to_csv(source_dir / "portfolio_walk_forward_summary.csv", index=False)
            pd.DataFrame(
                {
                    "window": ["pf_01", "pf_01", "pf_04", "pf_04", "pf_05", "pf_05"],
                    "candidate": ["core_only", "quality__sat25", "blend__sat25", "core_only", "quality__sat25", "core_only"],
                    "source_name": ["core_only", "quality", "blend", "core_only", "quality", "core_only"],
                    "allocation_candidate": ["core_only", "sat25", "sat25", "core_only", "sat25", "core_only"],
                    "score": [0.10, 0.10, 1.10, 1.085, 0.80, 0.60],
                    "train_score": [0.10, 0.08, 0.40, 0.39, 0.70, 0.50],
                    "validation_score": [0.10, 0.10, 1.10, 1.085, 0.80, 0.60],
                }
            ).to_csv(source_dir / "portfolio_candidate_results.csv", index=False)
            pd.DataFrame(
                {
                    "window": ["pf_04"],
                    "changed_by": ["source_switch_guard"],
                    "return_delta_vs_default": [-0.01],
                    "recommendation": ["review_risk_budget_not_default_source_switch"],
                }
            ).to_csv(guard_dir / "source_guard_decomposition_windows.csv", index=False)

            result = run_source_risk_budget_review(
                source_dir=source_dir,
                guard_decomposition_dir=guard_dir,
                output_dir=root / "risk_budget",
                fragile_source_edge=0.03,
                max_low_risk_satellite_weight=0.05,
                min_filter_off_ratio=0.50,
            )

            windows = pd.read_csv(result.windows_path)
            pf04 = windows.loc[windows["window"] == "pf_04"].iloc[0]
            pf01 = windows.loc[windows["window"] == "pf_01"].iloc[0]
            self.assertEqual(result.snapshot["fragile_window_count"], 1)
            self.assertEqual(result.snapshot["low_risk_fragile_window_count"], 1)
            self.assertEqual(result.snapshot["budget_cap_review_count"], 0)
            self.assertEqual(result.snapshot["decision"], "keep_default_review_market_state_budget")
            self.assertEqual(pf01["risk_budget_recommendation"], "no_fragility_action")
            self.assertAlmostEqual(pf04["source_edge_vs_second"], 0.015)
            self.assertTrue(bool(pf04["already_low_satellite_risk"]))
            self.assertEqual(pf04["risk_budget_recommendation"], "observe_without_source_switch")
            self.assertTrue(result.report_path.exists())
