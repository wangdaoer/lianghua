from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser
from quant_etf_lab.paper_account import apply_stock_market_cap_tracking_rule, build_stock_targets
from quant_etf_lab.portfolio import CurveConfig, PortfolioConfig


class PaperAccountStockTargetsTest(unittest.TestCase):
    def test_paper_account_cli_defaults_to_no_market_cap_upper_limit(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["paper-account"])

        self.assertIsNone(args.stock_tracking_max_market_cap_yi)

    def test_market_cap_tracking_rule_defaults_to_no_upper_limit(self) -> None:
        targets = pd.DataFrame({"code": ["600000"], "name": ["LargeCap"]})
        market_cap = pd.DataFrame({"code": ["600000"], "market_cap_yi": [2600.0]})

        tracked, payload = apply_stock_market_cap_tracking_rule(targets, market_cap)

        self.assertFalse(bool(tracked.iloc[0]["tracking_excluded"]))
        self.assertEqual(tracked.iloc[0]["tracking_rule_status"], "tracking_allowed")
        self.assertIsNone(payload["stock_tracking_max_market_cap_yi"])
        self.assertEqual(payload["stock_tracking_excluded_large_market_cap_count"], 0)

    def test_market_cap_tracking_rule_zero_disables_upper_limit(self) -> None:
        targets = pd.DataFrame({"code": ["600000"], "name": ["LargeCap"]})
        market_cap = pd.DataFrame({"code": ["600000"], "market_cap_yi": [2600.0]})

        tracked, payload = apply_stock_market_cap_tracking_rule(targets, market_cap, max_market_cap_yi=0)

        self.assertFalse(bool(tracked.iloc[0]["tracking_excluded"]))
        self.assertEqual(tracked.iloc[0]["tracking_rule_status"], "tracking_allowed")
        self.assertIsNone(payload["stock_tracking_max_market_cap_yi"])
        self.assertEqual(payload["stock_tracking_requested_max_market_cap_yi"], 0.0)
        self.assertEqual(payload["stock_tracking_excluded_large_market_cap_count"], 0)

    def test_fresh_external_trigger_is_added_to_stock_target_pool_for_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backtests = root / "outputs" / "backtests"
            core_run = backtests / "core_run"
            satellite_run = backtests / "satellite_run"
            core_run.mkdir(parents=True)
            satellite_run.mkdir(parents=True)

            core_curve = root / "core_curve.csv"
            satellite_curve = root / "satellite_curve.csv"
            pd.DataFrame({"date": ["2026-06-22"], "window": ["core_run"], "equity": [100000.0]}).to_csv(
                core_curve,
                index=False,
            )
            pd.DataFrame({"date": ["2026-06-22"], "window": ["satellite_run"], "equity": [100000.0]}).to_csv(
                satellite_curve,
                index=False,
            )
            pd.DataFrame(
                {
                    "date": ["2026-06-22"],
                    "code": ["000001"],
                    "name": ["目标一"],
                    "side": ["BUY"],
                    "price": [10.0],
                    "quantity": [1000.0],
                }
            ).to_csv(core_run / "trades.csv", index=False)
            pd.DataFrame(columns=["date", "code", "name", "side", "price", "quantity"]).to_csv(
                satellite_run / "trades.csv",
                index=False,
            )

            stock_dir = root / "data" / "processed" / "stocks"
            stock_dir.mkdir(parents=True)
            pd.DataFrame({"date": ["2026-06-22"], "close": [12.0]}).to_csv(stock_dir / "000001.csv", index=False)

            trigger_signal_path = root / "signals_latest.csv"
            pd.DataFrame(
                {
                    "run_time": ["2026-06-22_150000", "2026-06-22_150000"],
                    "code": ["000001", "600360"],
                    "name": ["目标一", "华微电子"],
                    "signal_type": ["观察", "突破候选"],
                    "action": ["继续观察", "等回踩不破再买"],
                    "reason": ["目标已有触发", "短线 fresh 触发"],
                    "score": [80, 97],
                    "score_level": ["A", "A"],
                    "pct": [1.2, 1.8],
                    "last": [12.0, 12.4],
                }
            ).to_csv(trigger_signal_path, index=False, encoding="utf-8")

            portfolio_config = PortfolioConfig(
                project_root=root,
                name="test",
                initial_cash=100000.0,
                output_dir=root / "portfolio",
                core=CurveConfig(name="core", path=core_curve, equity_column="equity"),
                satellite=CurveConfig(name="satellite", path=satellite_curve, equity_column="equity"),
                benchmark_path=root / "benchmark.csv",
                benchmark_close_column="close",
                ma_window=120,
                drop_window=20,
                risk_on_drop_threshold=-0.04,
                crash_drop_threshold=-0.08,
                default_regime="risk_on",
                weights={
                    "risk_on": {"core": 1.0, "satellite": 0.0, "cash": 0.0},
                    "risk_off": {"core": 1.0, "satellite": 0.0, "cash": 0.0},
                    "crash": {"core": 0.0, "satellite": 0.0, "cash": 1.0},
                },
            )

            targets, payload = build_stock_targets(
                project_root=root,
                portfolio_config=portfolio_config,
                target_holdings=pd.DataFrame(
                    {
                        "layer": ["core", "satellite", "cash"],
                        "target_weight": [1.0, 0.0, 0.0],
                        "target_value": [100000.0, 0.0, 0.0],
                    }
                ),
                latest_date="2026-06-22",
                trigger_signal_path=trigger_signal_path,
            )

            self.assertIn("600360", set(targets["code"]))
            trigger_row = targets[targets["code"] == "600360"].iloc[0]
            self.assertEqual(trigger_row["layer"], "trigger")
            self.assertEqual(trigger_row["target_action"], "trigger_watch_candidate")
            self.assertEqual(trigger_row["risk_filter_status"], "not_in_current_portfolio_target")
            self.assertEqual(trigger_row["trigger_signal_validity_status"], "fresh_trigger_signal")
            self.assertEqual(trigger_row["execution_gate_action"], "usable_for_candidate_review")
            self.assertAlmostEqual(float(trigger_row["portfolio_target_weight"]), 0.0)
            self.assertAlmostEqual(float(targets["portfolio_target_weight"].sum()), 1.0)
            self.assertEqual(int(payload["external_fresh_trigger_candidate_count"]), 1)
            self.assertEqual(int(payload["active_stock_target_count"]), 1)


if __name__ == "__main__":
    unittest.main()
