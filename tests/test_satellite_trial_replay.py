from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser
from quant_etf_lab.satellite_trial_replay import run_satellite_trial_replay


class SatelliteTrialReplayTests(unittest.TestCase):
    def test_satellite_trial_replay_cli(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "satellite-trial-replay",
                "--rules-path",
                "outputs/research/satellite_risk_budget_review_latest/satellite_trial_rules.csv",
                "--outcomes-history",
                "outputs/research/stock_target_review_outcomes_history.csv",
                "--output-dir",
                "outputs/research/satellite_trial_replay_latest",
                "--horizons",
                "1d,5d,10d",
            ]
        )

        self.assertEqual(args.command, "satellite-trial-replay")
        self.assertEqual(args.rules_path, "outputs/research/satellite_risk_budget_review_latest/satellite_trial_rules.csv")
        self.assertEqual(args.outcomes_history, "outputs/research/stock_target_review_outcomes_history.csv")
        self.assertEqual(args.output_dir, "outputs/research/satellite_trial_replay_latest")
        self.assertEqual(args.horizons, "1d,5d,10d")

    def test_satellite_trial_replay_compares_rule_union_with_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rules_path = root / "satellite_trial_rules.csv"
            history_path = root / "stock_target_review_outcomes_history.csv"
            output_dir = root / "replay"

            pd.DataFrame(
                [
                    {
                        "rule_id": "sat_trial_01",
                        "trial_rule_status": "eligible_when_risk_on",
                        "dimension": "layer",
                        "group_value": "satellite",
                        "horizon": "5d",
                        "reference_total_budget": 0.05,
                        "reference_group_weight_cap": 0.025,
                    },
                    {
                        "rule_id": "sat_trial_02",
                        "trial_rule_status": "eligible_when_risk_on",
                        "dimension": "review_bucket",
                        "group_value": "suppressed_layer_review",
                        "horizon": "5d",
                        "reference_total_budget": 0.05,
                        "reference_group_weight_cap": 0.025,
                    },
                ]
            ).to_csv(rules_path, index=False)

            rows = []
            for idx, ret_1d, ret_5d in [
                (1, 0.04, 0.10),
                (2, 0.02, 0.08),
                (3, -0.01, 0.06),
                (4, 0.03, 0.12),
            ]:
                rows.append(
                    {
                        "date": "2026-06-10",
                        "layer": "satellite",
                        "code": f"30000{idx}",
                        "name": f"sat-{idx}",
                        "review_bucket": "suppressed_layer_review" if idx in {1, 2} else "trigger_review",
                        "action_code": "review_required_pending",
                        "return_1d": ret_1d,
                        "outcome_status_1d": "available",
                        "return_5d": ret_5d,
                        "outcome_status_5d": "available",
                    }
                )
            for idx, ret_1d, ret_5d in [
                (5, -0.03, -0.06),
                (6, -0.02, -0.04),
                (7, 0.01, -0.02),
            ]:
                rows.append(
                    {
                        "date": "2026-06-10",
                        "layer": "core",
                        "code": f"60000{idx}",
                        "name": f"core-{idx}",
                        "review_bucket": "routine_review",
                        "action_code": "",
                        "return_1d": ret_1d,
                        "outcome_status_1d": "available",
                        "return_5d": ret_5d,
                        "outcome_status_5d": "available",
                    }
                )
            pd.DataFrame(rows).to_csv(history_path, index=False)

            result = run_satellite_trial_replay(
                rules_path=rules_path,
                outcomes_history_path=history_path,
                output_dir=output_dir,
                horizons=("1d", "5d"),
            )

            self.assertEqual(result.snapshot["status"], "completed")
            self.assertEqual(result.snapshot["rule_count"], 2)
            self.assertEqual(result.snapshot["matched_event_count"], 4)
            summary = pd.read_csv(result.summary_path)
            union_5d = summary[(summary["scope"] == "rule_union") & (summary["horizon"] == "5d")].iloc[0]
            baseline_5d = summary[(summary["scope"] == "baseline_all") & (summary["horizon"] == "5d")].iloc[0]
            self.assertEqual(int(union_5d["sample_count"]), 4)
            self.assertGreater(float(union_5d["avg_return"]), float(baseline_5d["avg_return"]))
            self.assertGreater(float(union_5d["win_rate"]), float(baseline_5d["win_rate"]))
            self.assertGreater(float(union_5d["avg_return_edge"]), 0.05)
            matches = pd.read_csv(result.matches_path)
            self.assertEqual(len(matches), 4)
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.snapshot_path.exists())
            self.assertIn("Satellite Trial Replay", result.report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
