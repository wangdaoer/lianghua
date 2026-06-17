from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_etf_lab.allocator_promotion import run_allocator_promotion_review
from quant_etf_lab.cli import build_parser


def write_allocator_run_fixture(
    run_dir: Path,
    equity_values: list[float],
    window_returns: list[float],
    window_drawdowns: list[float] | None = None,
    window_sharpes: list[float] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2024-01-01", periods=len(equity_values), freq="D")
    pd.DataFrame({"date": dates, "stitched_equity": equity_values}).to_csv(run_dir / "oos_equity_stitched.csv", index=False)
    count = len(window_returns)
    pd.DataFrame(
        {
            "window": [f"pf_{index:02d}" for index in range(1, count + 1)],
            "test_total_return": window_returns,
            "test_max_drawdown": window_drawdowns or [-0.05] * count,
            "test_sharpe": window_sharpes or [0.8] * count,
        }
    ).to_csv(run_dir / "portfolio_walk_forward_summary.csv", index=False)


class AllocatorPromotionTests(unittest.TestCase):
    def test_allocator_promotion_review_cli(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "allocator-promotion-review",
                "--baseline-dir",
                "outputs/portfolio_walk_forward/base",
                "--candidate-dir",
                "outputs/portfolio_source_selection/candidate",
                "--sensitivity-dir",
                "outputs/portfolio_source_selection/validation3",
                "--sensitivity-group",
                "validation_3m",
                "--sensitivity-dir",
                "outputs/portfolio_source_selection/validation9",
                "--sensitivity-group",
                "validation_9m",
                "--evidence-snapshot",
                "outputs/research/execution_cost_stress/snapshot.json",
                "--evidence-group",
                "execution_cost_stress",
                "--output-dir",
                "outputs/research/promotion",
            ]
        )
        self.assertEqual(args.command, "allocator-promotion-review")
        self.assertEqual(len(args.sensitivity_dir), 2)
        self.assertEqual(args.sensitivity_group, ["validation_3m", "validation_9m"])
        self.assertEqual(args.evidence_snapshot, ["outputs/research/execution_cost_stress/snapshot.json"])
        self.assertEqual(args.evidence_group, ["execution_cost_stress"])
        self.assertEqual(args.output_dir, "outputs/research/promotion")

    def test_allocator_promotion_review_marks_candidate_watch_without_sensitivity_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            sensitivity_3m = root / "sensitivity_3m"
            sensitivity_9m = root / "sensitivity_9m"
            write_allocator_run_fixture(baseline, [100, 108, 112, 118, 143], [0.08, -0.02, 0.13, 0.00, 0.20])
            write_allocator_run_fixture(candidate, [100, 112, 125, 135, 149], [0.08, 0.01, 0.13, 0.01, 0.21])
            write_allocator_run_fixture(sensitivity_3m, [100, 106, 110, 116, 138], [0.08, -0.03, 0.12, 0.00, 0.16])
            write_allocator_run_fixture(sensitivity_9m, [100, 107, 111, 118, 140], [0.07, -0.03, 0.13, 0.00, 0.18])

            result = run_allocator_promotion_review(
                baseline_dir=baseline,
                candidate_dir=candidate,
                sensitivity_dirs=[sensitivity_3m, sensitivity_9m],
                output_dir=root / "review",
                min_return_edge=0.03,
                min_sharpe_edge=0.0,
                min_sensitivity_support=1,
            )

            self.assertEqual(result.decision, "watch_candidate")
            self.assertEqual(int(result.snapshot["sensitivity_support_count"]), 0)
            self.assertTrue(result.report_path.exists())

    def test_allocator_promotion_review_groups_correlated_sensitivity_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            sensitivity_a = root / "sensitivity_a"
            sensitivity_b = root / "sensitivity_b"
            write_allocator_run_fixture(baseline, [100, 108, 112, 118, 143], [0.08, 0.02, 0.13, 0.01, 0.20])
            write_allocator_run_fixture(candidate, [100, 112, 125, 137, 151], [0.08, 0.03, 0.13, 0.02, 0.22])
            write_allocator_run_fixture(sensitivity_a, [100, 112, 122, 137, 148], [0.08, 0.03, 0.12, 0.02, 0.21])
            write_allocator_run_fixture(sensitivity_b, [100, 113, 123, 138, 149], [0.08, 0.03, 0.13, 0.02, 0.21])

            result = run_allocator_promotion_review(
                baseline_dir=baseline,
                candidate_dir=candidate,
                sensitivity_dirs=[sensitivity_a, sensitivity_b],
                sensitivity_groups=["guardrail", "guardrail"],
                output_dir=root / "review",
                min_return_edge=0.03,
                min_sharpe_edge=0.0,
                min_sensitivity_support=2,
            )

            self.assertEqual(result.decision, "watch_candidate")
            self.assertEqual(int(result.snapshot["sensitivity_run_support_count"]), 2)
            self.assertEqual(int(result.snapshot["sensitivity_group_support_count"]), 1)
            self.assertEqual(int(result.snapshot["sensitivity_support_count"]), 1)
            comparison = pd.read_csv(result.comparison_path)
            self.assertIn("sensitivity_group", comparison.columns)
            guardrail_rows = comparison[comparison["role"] == "sensitivity"]
            self.assertEqual(guardrail_rows["sensitivity_group"].tolist(), ["guardrail", "guardrail"])

    def test_allocator_promotion_review_counts_external_evidence_group_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            sensitivity_a = root / "sensitivity_a"
            sensitivity_b = root / "sensitivity_b"
            evidence_snapshot = root / "execution_cost_stress_snapshot.json"
            write_allocator_run_fixture(baseline, [100, 108, 112, 118, 143], [0.08, 0.02, 0.13, 0.01, 0.20])
            write_allocator_run_fixture(candidate, [100, 112, 125, 137, 151], [0.08, 0.03, 0.13, 0.02, 0.22])
            write_allocator_run_fixture(sensitivity_a, [100, 112, 122, 137, 148], [0.08, 0.03, 0.12, 0.02, 0.21])
            write_allocator_run_fixture(sensitivity_b, [100, 113, 123, 138, 149], [0.08, 0.03, 0.13, 0.02, 0.21])
            evidence_snapshot.write_text(
                json.dumps(
                    {
                        "status": "supports_candidate",
                        "all_cost_rates_support_candidate": True,
                        "min_return_edge": 0.054,
                        "min_sharpe_edge": 0.106,
                        "min_drawdown_change": 0.018,
                    }
                ),
                encoding="utf-8",
            )

            result = run_allocator_promotion_review(
                baseline_dir=baseline,
                candidate_dir=candidate,
                sensitivity_dirs=[sensitivity_a, sensitivity_b],
                sensitivity_groups=["guardrail", "guardrail"],
                evidence_snapshots=[evidence_snapshot],
                evidence_groups=["execution_cost_stress"],
                output_dir=root / "review",
                min_return_edge=0.03,
                min_sharpe_edge=0.0,
                min_sensitivity_support=2,
            )

            self.assertEqual(result.decision, "promote_candidate")
            self.assertEqual(int(result.snapshot["sensitivity_run_support_count"]), 2)
            self.assertEqual(int(result.snapshot["sensitivity_group_support_count"]), 1)
            self.assertEqual(int(result.snapshot["evidence_support_count"]), 1)
            self.assertEqual(int(result.snapshot["support_group_count"]), 2)
            self.assertEqual(int(result.snapshot["sensitivity_support_count"]), 2)
            self.assertEqual(result.snapshot["evidence_groups"], ["execution_cost_stress"])
            comparison = pd.read_csv(result.comparison_path)
            evidence_rows = comparison[comparison["role"] == "evidence"]
            self.assertEqual(evidence_rows["sensitivity_group"].tolist(), ["execution_cost_stress"])
            self.assertEqual(evidence_rows["supports_candidate"].tolist(), [True])

    def test_allocator_promotion_review_rejects_candidate_when_headline_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            write_allocator_run_fixture(
                baseline,
                [100, 110, 121, 132, 144],
                [0.08, -0.01, 0.10, 0.03, 0.09],
            )
            write_allocator_run_fixture(
                candidate,
                [100, 108, 116, 127, 137],
                [0.08, -0.01, 0.07, 0.02, 0.07],
            )

            result = run_allocator_promotion_review(
                baseline_dir=baseline,
                candidate_dir=candidate,
                output_dir=root / "review",
                min_return_edge=0.03,
                min_sharpe_edge=0.0,
                min_sensitivity_support=1,
            )

            self.assertEqual(result.decision, "reject_candidate")
            self.assertFalse(result.snapshot["candidate_passes_headline_gate"])
            self.assertEqual(int(result.snapshot["sensitivity_support_count"]), 0)

    def test_allocator_promotion_review_errors_on_group_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            sensitivity_3m = root / "sensitivity_3m"
            write_allocator_run_fixture(baseline, [100, 108, 112, 118, 143], [0.08, 0.02, 0.13, 0.01, 0.20])
            write_allocator_run_fixture(candidate, [100, 112, 125, 137, 151], [0.08, 0.03, 0.13, 0.02, 0.22])
            write_allocator_run_fixture(sensitivity_3m, [100, 112, 122, 137, 148], [0.08, 0.03, 0.12, 0.02, 0.21])

            with self.assertRaises(ValueError) as cm:
                run_allocator_promotion_review(
                    baseline_dir=baseline,
                    candidate_dir=candidate,
                    sensitivity_dirs=[sensitivity_3m],
                    sensitivity_groups=["grp_3m", "extra"],
                    output_dir=root / "review",
                    min_return_edge=0.03,
                    min_sharpe_edge=0.0,
                    min_sensitivity_support=1,
                )
            self.assertIn("sensitivity_groups must have the same length as sensitivity_dirs", str(cm.exception))

    def test_allocator_promotion_review_can_reject_due_to_low_positive_window_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            candidate = root / "candidate"
            write_allocator_run_fixture(
                baseline,
                [100, 110, 130, 150, 142],
                [0.10, 0.18, 0.15, -0.05],
            )
            write_allocator_run_fixture(
                candidate,
                [100, 110, 99, 109, 146],
                [0.10, -0.10, 0.10, 0.34],
            )

            result = run_allocator_promotion_review(
                baseline_dir=baseline,
                candidate_dir=candidate,
                output_dir=root / "review",
                min_return_edge=0.03,
                min_sharpe_edge=-1.0,
                max_drawdown_worsening=0.1,
            )

            self.assertEqual(result.decision, "reject_candidate")
            self.assertFalse(result.snapshot["candidate_passes_headline_gate"])
