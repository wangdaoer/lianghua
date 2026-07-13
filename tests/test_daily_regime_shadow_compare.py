import argparse
import json
import tempfile
import unittest
from pathlib import Path

from run_daily_regime_shadow_compare import (
    build_comparison_payload,
    build_variant_command,
    ensure_panel_asof_date,
    load_variant_params,
    render_chinese_report,
    run_comparison,
)


class DailyRegimeShadowCompareTest(unittest.TestCase):
    def test_loads_baseline_and_named_dynamic_candidate(self):
        config = Path(__file__).resolve().parents[1] / "configs" / "evolution_strong_pullback.yaml"

        baseline, dynamic = load_variant_params(config, "regime_090_balanced")

        self.assertEqual(baseline["leverage"], 0.60)
        self.assertIsNone(baseline["regime_strong_leverage"])
        self.assertEqual(dynamic["leverage"], 0.60)
        self.assertEqual(dynamic["regime_strong_leverage"], 0.75)
        self.assertEqual(dynamic["regime_exceptional_leverage"], 0.90)
        self.assertEqual(dynamic["max_position_weight"], 0.12)

    def test_build_variant_command_includes_dynamic_regime_controls(self):
        command = build_variant_command(
            python_exe="python",
            script=Path("run_strong_pullback_satellite.py"),
            data=Path("panel.csv"),
            benchmark=Path("510300.csv"),
            output_dir=Path("dynamic"),
            params={
                "leverage": 0.60,
                "regime_strong_leverage": 0.75,
                "regime_exceptional_leverage": 0.90,
                "regime_strong_breadth_threshold": 0.55,
                "regime_exceptional_breadth_threshold": 0.70,
                "regime_strong_volatility_max": 0.28,
                "regime_exceptional_volatility_max": 0.20,
                "min_score": None,
            },
        )

        rendered = " ".join(command)
        self.assertIn("--regime-strong-leverage 0.75", rendered)
        self.assertIn("--regime-exceptional-leverage 0.9", rendered)
        self.assertNotIn("--min-score", rendered)

    def test_comparison_payload_and_report_label_dynamic_as_experimental(self):
        baseline = {
            "total_return": 0.20,
            "annualized_return": 0.08,
            "max_drawdown": -0.18,
            "sharpe_like": 0.60,
            "avg_turnover": 0.20,
            "avg_gross_exposure": 0.50,
        }
        dynamic = {
            "total_return": 0.23,
            "annualized_return": 0.09,
            "max_drawdown": -0.17,
            "sharpe_like": 0.63,
            "avg_turnover": 0.24,
            "avg_gross_exposure": 0.55,
            "risk_regime_day_share": {"base": 0.7, "strong": 0.2, "exceptional": 0.1},
        }

        payload = build_comparison_payload(
            asof_date="2026-07-13",
            candidate_id="regime_090_balanced",
            baseline=baseline,
            dynamic=dynamic,
            latest_dynamic_state={"risk_regime": "strong", "target_leverage": 0.75},
            benchmark_last_date="2026-07-13",
        )
        report = render_chinese_report(payload)

        self.assertEqual(payload["asof"], "2026-07-13")
        self.assertAlmostEqual(payload["delta"]["total_return"], 0.03)
        self.assertAlmostEqual(payload["delta"]["max_drawdown"], 0.01)
        self.assertEqual(payload["decision"], "experimental_only")
        self.assertTrue(payload["benchmark_fresh"])
        self.assertIn("实验策略", report)
        self.assertIn("strong", report)

    def test_stale_benchmark_masks_latest_dynamic_state(self):
        metrics = {
            "total_return": 0.20,
            "annualized_return": 0.08,
            "max_drawdown": -0.18,
            "sharpe_like": 0.60,
            "avg_turnover": 0.20,
            "avg_gross_exposure": 0.50,
        }

        payload = build_comparison_payload(
            asof_date="2026-07-13",
            candidate_id="regime_090_balanced",
            baseline=metrics,
            dynamic=metrics,
            latest_dynamic_state={"risk_regime": "strong", "target_leverage": 0.75},
            benchmark_last_date="2026-07-10",
        )
        report = render_chinese_report(payload)

        self.assertFalse(payload["benchmark_fresh"])
        self.assertEqual(
            payload["latest_dynamic_state"]["risk_regime"],
            "unknown_stale_benchmark",
        )
        self.assertIsNone(payload["latest_dynamic_state"]["target_leverage"])
        self.assertIn("基准滞后", report)
        self.assertIn("不可用", report)

    def test_panel_asof_date_must_match_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = Path(tmp) / "panel.csv"
            panel.write_text(
                "date,symbol,close\n2026-07-10,000001,10\n2026-07-13,000001,11\n",
                encoding="utf-8",
            )

            ensure_panel_asof_date(panel, "2026-07-13")
            with self.assertRaisesRegex(ValueError, "does not match"):
                ensure_panel_asof_date(panel, "2026-07-10")

    def test_run_comparison_rejects_panel_mismatch_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "panel.csv"
            panel.write_text(
                "date,symbol,close\n2026-07-13,000001,11\n",
                encoding="utf-8",
            )
            output = root / "shadow-output"
            args = argparse.Namespace(
                config=str(root / "missing.yaml"),
                data=str(panel),
                benchmark=str(root / "missing-benchmark.csv"),
                output_dir=str(output),
                asof_date="2026-07-10",
                candidate_id="regime_090_balanced",
                python_exe="python",
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                run_comparison(args)

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
