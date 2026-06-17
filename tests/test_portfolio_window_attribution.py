from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser


def _write_curve(path: Path, values: list[float]) -> None:
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D")
    pd.DataFrame({"date": dates, "equity": values}).to_csv(path, index=False)


class PortfolioWindowAttributionTests(unittest.TestCase):
    def test_portfolio_window_attribution_cli(self) -> None:
        parser = build_parser()
        try:
            args = parser.parse_args(
                [
                    "portfolio-window-attribution",
                    "--config",
                    "configs/portfolio_core_source_selection_quality_reversal_v1.yaml",
                    "--start",
                    "20240102",
                    "--end",
                    "20240105",
                    "--variant",
                    "default=outputs/default_selected_params.json",
                    "--variant",
                    "cap20=outputs/cap20_selected_params.json",
                    "--include-core-only",
                    "--output-dir",
                    "outputs/research/pf02_daily_attribution",
                ]
            )
        except SystemExit as exc:
            self.fail(f"portfolio-window-attribution CLI should parse without exiting: {exc}")

        self.assertEqual(args.command, "portfolio-window-attribution")
        self.assertEqual(args.config, "configs/portfolio_core_source_selection_quality_reversal_v1.yaml")
        self.assertEqual(args.start, "20240102")
        self.assertEqual(args.end, "20240105")
        self.assertEqual(
            args.variant,
            ["default=outputs/default_selected_params.json", "cap20=outputs/cap20_selected_params.json"],
        )
        self.assertTrue(args.include_core_only)
        self.assertEqual(args.output_dir, "outputs/research/pf02_daily_attribution")

    def test_portfolio_window_attribution_replays_selected_params_and_writes_daily_comparison(self) -> None:
        try:
            from quant_etf_lab.portfolio_window_attribution import run_portfolio_window_attribution
        except ImportError as exc:
            self.fail(f"portfolio_window_attribution module should exist: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core_path = root / "core.csv"
            satellite_path = root / "satellite.csv"
            benchmark_path = root / "benchmark.csv"
            config_path = root / "portfolio.yaml"
            default_params = root / "default_selected_params.json"
            cap_params = root / "cap20_selected_params.json"

            _write_curve(core_path, [100.0, 100.0, 100.0, 100.0, 100.0])
            _write_curve(satellite_path, [100.0, 100.0, 120.0, 120.0, 120.0])
            pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=5, freq="D"),
                    "close": [100.0, 100.0, 101.0, 102.0, 103.0],
                }
            ).to_csv(benchmark_path, index=False)
            config_path.write_text(
                f"""
project:
  name: attribution_fixture
  initial_cash: 100.0
  output_dir: "{(root / "portfolio_runs").as_posix()}"
curves:
  core:
    path: "{core_path.as_posix()}"
    equity_column: equity
  satellite:
    path: "{satellite_path.as_posix()}"
    equity_column: equity
regime:
  benchmark_path: "{benchmark_path.as_posix()}"
  benchmark_close_column: close
  ma_window: 1
  drop_window: 1
  risk_on_drop_threshold: -0.50
  crash_drop_threshold: -0.80
  default_regime: risk_on
weights:
  risk_on:
    core: 1.0
    satellite: 0.0
  risk_off:
    core: 1.0
    satellite: 0.0
  crash:
    core: 1.0
    satellite: 0.0
satellite_filter:
  enabled: false
satellite_sources:
  reversal:
    path: "{satellite_path.as_posix()}"
    equity_column: equity
""",
                encoding="utf-8",
            )

            base_allocation = {
                "regime_overrides": {"ma_window": 1, "risk_on_drop_threshold": -0.50, "crash_drop_threshold": -0.80},
                "satellite_filter": {"enabled": False},
            }
            default_params.write_text(
                json.dumps(
                    {
                        "source_name": "reversal",
                        "source_path": str(root / "stale_missing_satellite.csv"),
                        "allocation": {
                            **base_allocation,
                            "weights": {
                                "risk_on": {"core": 0.75, "satellite": 0.25, "cash": 0.0},
                                "risk_off": {"core": 1.0, "satellite": 0.0, "cash": 0.0},
                                "crash": {"core": 1.0, "satellite": 0.0, "cash": 0.0},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            cap_params.write_text(
                json.dumps(
                    {
                        "source_name": "reversal",
                        "source_path": str(root / "also_stale_missing_satellite.csv"),
                        "allocation": {
                            **base_allocation,
                            "weights": {
                                "risk_on": {"core": 0.80, "satellite": 0.20, "cash": 0.0},
                                "risk_off": {"core": 1.0, "satellite": 0.0, "cash": 0.0},
                                "crash": {"core": 1.0, "satellite": 0.0, "cash": 0.0},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = run_portfolio_window_attribution(
                config_path=config_path,
                start="20240101",
                end="20240105",
                variants=[f"default={default_params}", f"cap20={cap_params}"],
                include_core_only=True,
                output_dir=root / "attribution",
            )

            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.metrics_path.exists())
            self.assertTrue(result.daily_path.exists())
            self.assertTrue(result.snapshot_path.exists())
            metrics = pd.read_csv(result.metrics_path)
            daily = pd.read_csv(result.daily_path)
            self.assertEqual(result.snapshot["best_variant_by_total_return"], "default")
            self.assertEqual(set(metrics["variant"]), {"default", "cap20", "core_only"})
            self.assertIn("default_indexed_equity", daily.columns)
            self.assertIn("cap20_indexed_equity", daily.columns)
            self.assertIn("default_minus_cap20_indexed_equity", daily.columns)
            self.assertGreater(
                float(metrics.loc[metrics["variant"] == "default", "total_return"].iloc[0]),
                float(metrics.loc[metrics["variant"] == "cap20", "total_return"].iloc[0]),
            )
            self.assertIn("stale_source_path_ignored", result.snapshot["variant_notes"]["default"])

