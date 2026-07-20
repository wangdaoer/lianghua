from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from quant_etf_lab.cli import _publish_stock_outcome_gate_for_pipeline, build_parser
import pandas as pd

from quant_etf_lab.stock_outcome_gate import (
    build_stock_outcome_gate,
    build_stock_outcome_shadow_selection,
    build_stock_outcome_target_overlay,
    build_weak_sentiment_opportunity_trial,
    run_stock_outcome_gate,
)


def _ready_payload() -> dict:
    return {
        "generated_at": "2026-06-27T18:39:53",
        "analysis_status": "ready_for_review",
        "ready_horizons": ["1d", "5d", "10d"],
        "ready_horizon_count": 3,
        "horizon_readiness": {
            "1d": {"status": "ready_for_group_review", "evaluable_count": 156},
            "5d": {"status": "ready_for_group_review", "evaluable_count": 93},
            "10d": {"status": "ready_for_group_review", "evaluable_count": 31},
        },
        "analysis_path": "paper/stock_target_review_outcome_analysis.csv",
        "analysis_report_path": "paper/stock_target_review_outcome_analysis.md",
        "top_analysis_rows": [
            {
                "dimension": "overall",
                "group_value": "all",
                "evaluable_5d": 93,
                "avg_return_5d": 0.0047,
                "win_rate_5d": 0.3548,
            },
            {
                "dimension": "layer",
                "group_value": "satellite",
                "history_row_count": 19,
                "evaluable_5d": 19,
                "avg_return_5d": 0.1778,
                "win_rate_5d": 0.8947,
            },
            {
                "dimension": "layer",
                "group_value": "core",
                "history_row_count": 108,
                "evaluable_5d": 62,
                "avg_return_5d": -0.0389,
                "win_rate_5d": 0.2097,
            },
            {
                "dimension": "review_bucket",
                "group_value": "routine_review",
                "history_row_count": 48,
                "evaluable_5d": 35,
                "avg_return_5d": -0.0442,
                "win_rate_5d": 0.1714,
            },
            {
                "dimension": "review_stage",
                "group_value": "monitor",
                "history_row_count": 40,
                "evaluable_5d": 27,
                "avg_return_5d": 0.0629,
                "win_rate_5d": 0.5185,
            },
        ],
    }


class StockOutcomeGateTest(unittest.TestCase):
    def test_cli_exposes_stock_outcome_gate_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["stock-outcome-gate", "--outcome-analysis-path", "paper/outcomes.json"])

        self.assertEqual(args.command, "stock-outcome-gate")
        self.assertEqual(args.outcome_analysis_path, "paper/outcomes.json")
        self.assertEqual(args.primary_horizon, "5d")
        self.assertIsNone(args.stock_target_review_path)
        self.assertEqual(args.relaxed_caution_multiplier, 0.5)
        self.assertEqual(args.relaxed_prefer_trial_weight, 0.20)
        self.assertIsNone(args.pipeline_snapshot_path)
        self.assertIsNone(args.paper_ledger_path)
        self.assertIsNone(args.weak_sentiment_opportunity_output_dir)
        self.assertEqual(args.weak_sentiment_single_weight_cap, 0.05)
        self.assertEqual(args.weak_sentiment_total_weight_cap, 0.20)
        self.assertEqual(args.output_dir, "outputs/research/stock_outcome_gate_latest")

    def test_daily_pipeline_stock_outcome_gate_is_opt_in(self) -> None:
        defaults = build_parser().parse_args(["daily-pipeline"])
        enabled = build_parser().parse_args(
            [
                "daily-pipeline",
                "--publish-stock-outcome-gate",
                "--stock-outcome-gate-weak-output-dir",
                "outputs/research/weak_trial",
            ]
        )

        self.assertFalse(defaults.publish_stock_outcome_gate)
        self.assertEqual(defaults.stock_outcome_gate_primary_horizon, "5d")
        self.assertEqual(defaults.stock_outcome_gate_relaxed_caution_multiplier, 0.5)
        self.assertEqual(defaults.stock_outcome_gate_relaxed_prefer_trial_weight, 0.20)
        self.assertTrue(enabled.publish_stock_outcome_gate)
        self.assertEqual(enabled.stock_outcome_gate_weak_output_dir, "outputs/research/weak_trial")

    def test_stock_outcome_gate_prefers_satellite_and_cautions_weak_groups(self) -> None:
        gate, snapshot = build_stock_outcome_gate(_ready_payload(), primary_horizon="5d")

        self.assertEqual(snapshot["decision"], "prefer_positive_groups_with_caution")
        self.assertEqual(snapshot["selected_horizon"], "5d")
        self.assertEqual(snapshot["broker_action"], "none")
        self.assertTrue(snapshot["research_only"])
        self.assertIn("layer=satellite", snapshot["prefer_groups"])
        self.assertIn("review_stage=monitor", snapshot["prefer_groups"])
        self.assertIn("layer=core", snapshot["caution_groups"])
        self.assertIn("review_bucket=routine_review", snapshot["caution_groups"])
        self.assertEqual(set(gate["gate_action"]), {"prefer", "caution"})

    def test_stock_outcome_gate_waits_when_no_ready_horizon(self) -> None:
        gate, snapshot = build_stock_outcome_gate(
            {
                "analysis_status": "waiting_for_evaluable_returns",
                "ready_horizons": [],
                "horizon_readiness": {"5d": {"status": "waiting_for_evaluable_returns", "evaluable_count": 0}},
                "top_analysis_rows": [],
            },
            primary_horizon="5d",
        )

        self.assertTrue(gate.empty)
        self.assertEqual(snapshot["decision"], "wait_for_outcome_samples")
        self.assertIsNone(snapshot["selected_horizon"])
        self.assertEqual(snapshot["broker_action"], "none")

    def test_stock_outcome_gate_caution_reason_lists_only_triggered_conditions(self) -> None:
        gate, _ = build_stock_outcome_gate(
            {
                "analysis_status": "ready_for_review",
                "ready_horizons": ["5d"],
                "horizon_readiness": {"5d": {"status": "ready_for_group_review", "evaluable_count": 20}},
                "top_analysis_rows": [
                    {
                        "dimension": "review_bucket",
                        "group_value": "low_win_only",
                        "history_row_count": 20,
                        "evaluable_5d": 20,
                        "avg_return_5d": 0.001,
                        "win_rate_5d": 0.30,
                    }
                ],
            },
            primary_horizon="5d",
        )

        reason = str(gate.iloc[0]["gate_reason"])
        self.assertEqual(gate.iloc[0]["gate_action"], "caution")
        self.assertIn("win_rate=0.3000 <= 0.3500", reason)
        self.assertNotIn("avg_return=0.0010 <=", reason)

    def test_stock_outcome_gate_builds_target_overlay_without_changing_weights(self) -> None:
        gate, _ = build_stock_outcome_gate(_ready_payload(), primary_horizon="5d")
        targets = pd.DataFrame(
            [
                {
                    "code": "300001",
                    "name": "satellite target",
                    "layer": "satellite",
                    "review_bucket": "suppressed_layer_review",
                    "review_stage": "monitor",
                    "portfolio_target_weight": 0.12,
                },
                {
                    "code": "600001",
                    "name": "routine core",
                    "layer": "core",
                    "review_bucket": "routine_review",
                    "review_stage": "routine",
                    "portfolio_target_weight": 0.08,
                },
                {
                    "code": "600002",
                    "name": "drawdown core",
                    "layer": "core",
                    "review_bucket": "drawdown_review",
                    "review_stage": "review_required",
                    "portfolio_target_weight": 0.07,
                },
            ]
        )

        overlay, payload = build_stock_outcome_target_overlay(targets, gate)

        self.assertEqual(payload["target_overlay_row_count"], 3)
        self.assertEqual(payload["broker_action"], "none")
        self.assertEqual(list(overlay["portfolio_target_weight"]), [0.12, 0.08, 0.07])
        by_code = overlay.set_index("code")
        self.assertEqual(by_code.loc["300001", "outcome_gate_target_action"], "prefer")
        self.assertEqual(by_code.loc["300001", "outcome_gate_research_priority"], "increase_research_priority")
        self.assertEqual(by_code.loc["600001", "outcome_gate_target_action"], "caution")
        self.assertEqual(by_code.loc["600001", "outcome_gate_research_priority"], "reduce_research_priority")
        self.assertEqual(by_code.loc["600002", "outcome_gate_target_action"], "caution")
        self.assertIn("layer=core", by_code.loc["600002", "outcome_gate_matches"])

    def test_stock_outcome_gate_builds_shadow_selection_weight_sets(self) -> None:
        overlay = pd.DataFrame(
            [
                {
                    "code": "300001",
                    "name": "prefer active",
                    "portfolio_target_weight": 0.03,
                    "outcome_gate_research_priority": "increase_research_priority",
                    "outcome_gate_target_action": "prefer",
                },
                {
                    "code": "600001",
                    "name": "caution active",
                    "portfolio_target_weight": 0.08,
                    "outcome_gate_research_priority": "reduce_research_priority",
                    "outcome_gate_target_action": "caution",
                },
                {
                    "code": "600002",
                    "name": "manual active",
                    "portfolio_target_weight": 0.04,
                    "outcome_gate_research_priority": "manual_compare",
                    "outcome_gate_target_action": "mixed_review",
                },
                {
                    "code": "600003",
                    "name": "neutral active",
                    "portfolio_target_weight": 0.05,
                    "outcome_gate_research_priority": "maintain_research_priority",
                    "outcome_gate_target_action": "neutral",
                },
            ]
        )

        shadow, payload = build_stock_outcome_shadow_selection(overlay)

        self.assertEqual(payload["shadow_selection_row_count"], 4)
        self.assertAlmostEqual(payload["baseline_target_weight_total"], 0.20)
        self.assertAlmostEqual(payload["shadow_reduce_caution_raw_weight_total"], 0.12)
        self.assertAlmostEqual(payload["shadow_prefer_only_raw_weight_total"], 0.03)
        self.assertAlmostEqual(payload["shadow_relaxed_opportunity_raw_weight_total"], 0.20)
        by_code = shadow.set_index("code")
        self.assertEqual(by_code.loc["300001", "shadow_watch_action"], "shadow_prefer_watch")
        self.assertEqual(by_code.loc["600001", "shadow_watch_action"], "shadow_reduce_watch")
        self.assertAlmostEqual(float(by_code.loc["600001", "shadow_reduce_caution_raw_weight"]), 0.0)
        self.assertAlmostEqual(float(by_code.loc["600002", "shadow_reduce_caution_raw_weight"]), 0.04)
        self.assertAlmostEqual(float(by_code.loc["300001", "shadow_relaxed_opportunity_raw_weight"]), 0.07)
        self.assertAlmostEqual(float(by_code.loc["600001", "shadow_relaxed_opportunity_raw_weight"]), 0.04)

    def test_weak_sentiment_trial_uses_prefer_candidates_and_caps(self) -> None:
        shadow = pd.DataFrame(
            [
                {
                    "date": "2026-06-29",
                    "code": "300003",
                    "name": "prefer low score",
                    "layer": "satellite",
                    "review_priority_score": 60,
                    "baseline_target_weight": 0.0,
                    "shadow_relaxed_opportunity_raw_weight": 0.08,
                    "outcome_gate_target_action": "prefer",
                    "outcome_gate_research_priority": "increase_research_priority",
                    "shadow_watch_action": "shadow_prefer_watch",
                    "outcome_gate_matches": "layer=satellite",
                },
                {
                    "date": "2026-06-29",
                    "code": "300001",
                    "name": "prefer high score",
                    "layer": "satellite",
                    "review_priority_score": 90,
                    "baseline_target_weight": 0.0,
                    "shadow_relaxed_opportunity_raw_weight": 0.08,
                    "outcome_gate_target_action": "prefer",
                    "outcome_gate_research_priority": "increase_research_priority",
                    "shadow_watch_action": "shadow_prefer_watch",
                    "outcome_gate_matches": "layer=satellite",
                },
                {
                    "date": "2026-06-29",
                    "code": "600001",
                    "name": "caution",
                    "layer": "core",
                    "review_priority_score": 100,
                    "baseline_target_weight": 0.10,
                    "shadow_relaxed_opportunity_raw_weight": 0.05,
                    "outcome_gate_target_action": "caution",
                    "outcome_gate_research_priority": "reduce_research_priority",
                    "shadow_watch_action": "shadow_reduce_watch",
                    "outcome_gate_matches": "layer=core",
                },
            ]
        )

        trial, payload = build_weak_sentiment_opportunity_trial(
            shadow,
            {"selected_horizon": "5d", "prefer_groups": ["layer=satellite"], "caution_groups": ["layer=core"]},
            pipeline_snapshot={"sentiment_state": "weak", "dashboard_posture": "defensive_review_only"},
            single_weight_cap=0.05,
            total_weight_cap=0.10,
        )

        self.assertEqual(payload["weak_sentiment_trial_status"], "completed")
        self.assertEqual(list(trial["代码"]), ["300001", "300003"])
        self.assertEqual(list(trial["影子试运行权重"]), ["5.00%", "5.00%"])
        self.assertAlmostEqual(payload["weak_sentiment_trial_assigned_weight_total"], 0.10)

    def test_run_stock_outcome_gate_writes_report_snapshot_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_path = root / "outcome_analysis.json"
            output_dir = root / "gate"
            analysis_path.write_text(json.dumps(_ready_payload(), ensure_ascii=False), encoding="utf-8")

            result = run_stock_outcome_gate(
                outcome_analysis_path=analysis_path,
                output_dir=output_dir,
                primary_horizon="5d",
            )

            self.assertTrue(result.gate_path.exists())
            self.assertTrue(result.snapshot_path.exists())
            self.assertTrue(result.report_path.exists())
            self.assertEqual(result.snapshot["decision"], "prefer_positive_groups_with_caution")
            self.assertIn("layer=satellite", result.report_path.read_text(encoding="utf-8"))

    def test_run_stock_outcome_gate_accepts_in_memory_pipeline_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            targets = pd.DataFrame(
                [
                    {
                        "date": "2026-07-20",
                        "code": "300001",
                        "name": "candidate",
                        "layer": "satellite",
                        "review_bucket": "routine_review",
                        "review_stage": "monitor",
                        "portfolio_target_weight": 0.05,
                    }
                ]
            )
            with patch("quant_etf_lab.stock_outcome_gate._load_json", side_effect=AssertionError("disk JSON read")), patch(
                "quant_etf_lab.stock_outcome_gate._load_optional_json",
                side_effect=AssertionError("disk snapshot read"),
            ), patch(
                "quant_etf_lab.stock_outcome_gate._load_latest_csv_record",
                side_effect=AssertionError("disk ledger read"),
            ), patch("quant_etf_lab.stock_outcome_gate.pd.read_csv", side_effect=AssertionError("disk CSV read")):
                result = run_stock_outcome_gate(
                    outcome_analysis_path=root / "missing-analysis.json",
                    stock_target_review_path=root / "missing-review.csv",
                    output_dir=root / "gate",
                    weak_sentiment_opportunity_output_dir=root / "weak",
                    outcome_payload_override=_ready_payload(),
                    stock_target_review_frame=targets,
                    pipeline_snapshot_override={"sentiment_state": "weak"},
                    paper_ledger_record_override={"date": "2026-07-20", "equity": 10000.0},
                )

            self.assertTrue(result.report_path.exists())
            self.assertEqual(result.snapshot["source_stock_target_review_path"], str(root / "missing-review.csv"))
            self.assertIsNotNone(result.weak_sentiment_trial_csv_path)

    def test_pipeline_stock_outcome_gate_publisher_reuses_paper_objects(self) -> None:
        paper = SimpleNamespace(
            stock_target_review_outcome_analysis_json_path=Path("paper/analysis.json"),
            stock_target_review_path=Path("paper/review.csv"),
            ledger_path=Path("paper/ledger.csv"),
            stock_target_review_outcome_analysis_payload={"ready_horizons": []},
            stock_target_review=pd.DataFrame([{"code": "300001"}]),
            ledger=pd.DataFrame([{"date": "2026-07-20", "equity": 10000.0}]),
        )
        pipeline = SimpleNamespace(
            paper_account_result=paper,
            snapshot_path=Path("pipeline/snapshot.json"),
            snapshot={"as_of_date": "2026-07-20"},
        )
        args = SimpleNamespace(
            stock_outcome_gate_output_dir="outputs/gate",
            stock_outcome_gate_primary_horizon="5d",
            stock_outcome_gate_relaxed_caution_multiplier=0.5,
            stock_outcome_gate_relaxed_prefer_trial_weight=0.20,
            stock_outcome_gate_weak_output_dir="outputs/weak",
            stock_outcome_gate_weak_single_weight_cap=0.05,
            stock_outcome_gate_weak_total_weight_cap=0.20,
        )
        expected = SimpleNamespace(
            output_dir=Path("outputs/gate"),
            report_path=Path("outputs/gate/report.md"),
            snapshot={"decision": "wait_for_outcome_samples"},
        )

        with patch("quant_etf_lab.cli.process_lock", return_value=nullcontext()), patch(
            "quant_etf_lab.cli.run_stock_outcome_gate", return_value=expected
        ) as run_mock:
            self.assertIs(_publish_stock_outcome_gate_for_pipeline(args, pipeline), expected)

        kwargs = run_mock.call_args.kwargs
        self.assertIs(kwargs["outcome_payload_override"], paper.stock_target_review_outcome_analysis_payload)
        self.assertIs(kwargs["stock_target_review_frame"], paper.stock_target_review)
        self.assertIs(kwargs["pipeline_snapshot_override"], pipeline.snapshot)
        self.assertEqual(kwargs["paper_ledger_record_override"]["equity"], 10000.0)

    def test_run_stock_outcome_gate_writes_target_overlay_when_review_path_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_path = root / "outcome_analysis.json"
            review_path = root / "stock_target_review.csv"
            output_dir = root / "gate"
            analysis_path.write_text(json.dumps(_ready_payload(), ensure_ascii=False), encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "code": "300001",
                        "name": "satellite target",
                        "layer": "satellite",
                        "review_bucket": "suppressed_layer_review",
                        "review_stage": "monitor",
                        "portfolio_target_weight": 0.12,
                    }
                ]
            ).to_csv(review_path, index=False, encoding="utf-8")

            result = run_stock_outcome_gate(
                outcome_analysis_path=analysis_path,
                stock_target_review_path=review_path,
                output_dir=output_dir,
                primary_horizon="5d",
            )

            self.assertIsNotNone(result.target_overlay_path)
            self.assertTrue(result.target_overlay_path.exists())
            self.assertEqual(result.snapshot["target_overlay_status"], "completed")
            self.assertEqual(result.snapshot["target_overlay_row_count"], 1)
            self.assertIsNotNone(result.shadow_selection_path)
            self.assertTrue(result.shadow_selection_path.exists())
            self.assertEqual(result.snapshot["shadow_selection_status"], "completed")
            self.assertIn("increase_research_priority", result.report_path.read_text(encoding="utf-8"))

    def test_run_stock_outcome_gate_writes_weak_sentiment_trial_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_path = root / "outcome_analysis.json"
            review_path = root / "stock_target_review.csv"
            pipeline_path = root / "daily_pipeline_snapshot.json"
            ledger_path = root / "ledger.csv"
            output_dir = root / "gate"
            weak_dir = root / "weak_trial"
            analysis_path.write_text(json.dumps(_ready_payload(), ensure_ascii=False), encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "date": "2026-06-29",
                        "code": "300001",
                        "name": "satellite target",
                        "layer": "satellite",
                        "review_bucket": "suppressed_layer_review",
                        "review_stage": "monitor",
                        "review_priority_score": 80,
                        "portfolio_target_weight": 0.0,
                    },
                    {
                        "date": "2026-06-29",
                        "code": "600001",
                        "name": "routine core",
                        "layer": "core",
                        "review_bucket": "routine_review",
                        "review_stage": "routine",
                        "review_priority_score": 40,
                        "portfolio_target_weight": 0.10,
                    },
                ]
            ).to_csv(review_path, index=False, encoding="utf-8")
            pipeline_path.write_text(
                json.dumps(
                    {
                        "paper_latest_date": "2026-06-29",
                        "sentiment_state": "weak",
                        "dashboard_posture": "defensive_review_only",
                        "live_preflight_decision": "blocked",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "date": "2026-06-29",
                        "effective_regime": "risk_off",
                        "effective_reason": "benchmark_weak_drop",
                        "benchmark_drop": -0.06,
                    }
                ]
            ).to_csv(ledger_path, index=False, encoding="utf-8")

            result = run_stock_outcome_gate(
                outcome_analysis_path=analysis_path,
                stock_target_review_path=review_path,
                output_dir=output_dir,
                primary_horizon="5d",
                pipeline_snapshot_path=pipeline_path,
                paper_ledger_path=ledger_path,
                weak_sentiment_opportunity_output_dir=weak_dir,
            )

            self.assertIsNotNone(result.weak_sentiment_trial_csv_path)
            self.assertIsNotNone(result.weak_sentiment_trial_report_path)
            self.assertTrue(result.weak_sentiment_trial_csv_path.exists())
            self.assertTrue(result.weak_sentiment_trial_report_path.exists())
            self.assertEqual(result.snapshot["weak_sentiment_trial_status"], "completed")
            self.assertIn("弱情绪受限机会试运行", result.weak_sentiment_trial_report_path.read_text(encoding="utf-8"))
            self.assertIn("300001", result.weak_sentiment_trial_csv_path.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
