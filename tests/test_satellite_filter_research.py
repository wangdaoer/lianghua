from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser


class SatelliteFilterResearchTests(unittest.TestCase):
    def test_satellite_filter_research_cli(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "satellite-filter-research",
                "--outcomes-history",
                "outputs/research/stock_target_review_outcomes_history.csv",
                "--output-dir",
                "outputs/research/satellite_filter_research",
                "--horizons",
                "5d,10d",
                "--dimensions",
                "review_bucket,action_code",
                "--confirmation-horizons",
                "5d,10d",
                "--min-sample-count",
                "3",
                "--min-win-rate",
                "0.60",
                "--min-avg-return",
                "0.0",
                "--max-worst-return",
                "-0.20",
            ]
        )

        self.assertEqual(args.command, "satellite-filter-research")
        self.assertEqual(args.outcomes_history, "outputs/research/stock_target_review_outcomes_history.csv")
        self.assertEqual(args.output_dir, "outputs/research/satellite_filter_research")
        self.assertEqual(args.horizons, "5d,10d")
        self.assertEqual(args.dimensions, "review_bucket,action_code")
        self.assertEqual(args.confirmation_horizons, "5d,10d")
        self.assertEqual(args.min_sample_count, 3)
        self.assertEqual(args.min_win_rate, 0.60)
        self.assertEqual(args.min_avg_return, 0.0)
        self.assertEqual(args.max_worst_return, -0.20)

    def test_satellite_filter_research_confirms_only_satellite_groups(self) -> None:
        from quant_etf_lab.satellite_filter_research import run_satellite_filter_research

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "outcomes.csv"
            output_dir = root / "satellite_filter_research"
            rows = []
            for idx, ret_5d, ret_10d in [
                (1, 0.05, 0.08),
                (2, 0.04, 0.06),
                (3, 0.03, 0.05),
                (4, -0.02, -0.01),
            ]:
                rows.append(
                    {
                        "date": "2026-06-10",
                        "layer": "satellite",
                        "code": f"00000{idx}",
                        "name": f"sat-good-{idx}",
                        "review_bucket": "good_bucket",
                        "action_code": "observe",
                        "manual_status_normalized": "reviewed",
                        "return_5d": ret_5d,
                        "return_10d": ret_10d,
                    }
                )
            for idx, ret_5d, ret_10d in [
                (5, -0.05, -0.08),
                (6, -0.04, -0.06),
                (7, 0.01, -0.03),
                (8, -0.03, 0.01),
            ]:
                rows.append(
                    {
                        "date": "2026-06-10",
                        "layer": "satellite",
                        "code": f"00000{idx}",
                        "name": f"sat-bad-{idx}",
                        "review_bucket": "bad_bucket",
                        "action_code": "observe",
                        "manual_status_normalized": "reviewed",
                        "return_5d": ret_5d,
                        "return_10d": ret_10d,
                    }
                )
            for idx in range(9, 13):
                rows.append(
                    {
                        "date": "2026-06-10",
                        "layer": "core",
                        "code": f"0000{idx}",
                        "name": f"core-noise-{idx}",
                        "review_bucket": "good_bucket",
                        "action_code": "observe",
                        "manual_status_normalized": "reviewed",
                        "return_5d": -0.10,
                        "return_10d": -0.10,
                    }
                )
            pd.DataFrame(rows).to_csv(history, index=False)

            result = run_satellite_filter_research(
                outcomes_history_path=history,
                output_dir=output_dir,
                horizons=("5d", "10d"),
                dimensions=("review_bucket",),
                confirmation_horizons=("5d", "10d"),
                min_sample_count=3,
                min_win_rate=0.60,
                min_avg_return=0.0,
                max_worst_return=-0.20,
            )

            self.assertEqual(result.snapshot["research_status"], "candidate_filters_found")
            self.assertEqual(result.snapshot["satellite_rows"], 8)
            self.assertEqual(result.snapshot["confirmed_candidate_count"], 1)
            candidates = pd.read_csv(result.candidates_path)
            confirmed = candidates[candidates["candidate_status"] == "confirmed"]
            self.assertEqual(confirmed["dimension"].tolist(), ["review_bucket"])
            self.assertEqual(confirmed["group_value"].tolist(), ["good_bucket"])
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.group_scores_path.exists())


if __name__ == "__main__":
    unittest.main()
