import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_strategy_family_forward_returns import (
    annotate_priority_watchlist_with_health,
    annotate_selected_table_with_health,
    attach_strategy_family_forward_returns,
    build_strategy_family_health,
    discover_watchlists,
    load_strategy_watchlist,
    summarize_strategy_family_forward_returns,
    write_strategy_family_forward_report,
)


class StrategyFamilyForwardReturnsTest(unittest.TestCase):
    def test_forward_returns_use_next_open_and_holding_period_low(self):
        dates = pd.date_range("2026-01-01", periods=7, freq="D")
        open_px = pd.DataFrame({"000001": [10.0, 11.0, 12.0, 15.0, 18.0, 20.0, 21.0]}, index=dates)
        low_px = pd.DataFrame({"000001": [9.8, 10.5, 11.4, 14.0, 17.5, 19.5, 20.5]}, index=dates)
        candidates = pd.DataFrame(
            [
                {
                    "asof_date": "2026-01-02",
                    "symbol": "1",
                    "stock_name": "Alpha",
                    "strategy_family": "trend_momentum",
                    "strategy_family_cn": "趋势动量",
                }
            ]
        )

        out = attach_strategy_family_forward_returns(candidates, open_px, low_px, horizons=(1, 3))

        self.assertEqual(out.loc[0, "symbol"], "000001")
        self.assertEqual(out.loc[0, "entry_date"], "2026-01-03")
        self.assertEqual(out.loc[0, "exit_date_1d"], "2026-01-04")
        self.assertAlmostEqual(out.loc[0, "forward_return_1d"], 15.0 / 12.0 - 1.0)
        self.assertAlmostEqual(out.loc[0, "max_adverse_return_1d"], 11.4 / 12.0 - 1.0)
        self.assertEqual(out.loc[0, "exit_date_3d"], "2026-01-06")
        self.assertAlmostEqual(out.loc[0, "forward_return_3d"], 20.0 / 12.0 - 1.0)
        self.assertAlmostEqual(out.loc[0, "max_adverse_return_3d"], 11.4 / 12.0 - 1.0)

    def test_discover_watchlists_filters_cn_and_date_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in [
                "merged_priority_watchlist_20260701.csv",
                "merged_priority_watchlist_20260702_cn.csv",
                "merged_priority_watchlist_20260703.csv",
                "other.csv",
            ]:
                (root / name).write_text("symbol\n000001\n", encoding="utf-8")

            paths = discover_watchlists(root, start="2026-07-02", end="2026-07-03")

            self.assertEqual([path.name for path in paths], ["merged_priority_watchlist_20260703.csv"])

    def test_load_watchlist_infers_missing_strategy_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "merged_priority_watchlist_20260703.csv"
            pd.DataFrame(
                [
                    {
                        "symbol": "000001",
                        "stock_name": "Alpha",
                        "pattern_type": "隐性吸筹观察",
                        "priority_bucket": "pattern_watch",
                    }
                ]
            ).to_csv(path, index=False, encoding="utf-8-sig")

            table = load_strategy_watchlist(path)

            self.assertEqual(table.loc[0, "asof_date"], "2026-07-03")
            self.assertEqual(table.loc[0, "strategy_family"], "hidden_accumulation")
            self.assertEqual(table.loc[0, "strategy_family_cn"], "隐性吸筹")

    def test_summarize_strategy_family_forward_returns_counts_pending(self):
        samples = pd.DataFrame(
            {
                "strategy_family": ["trend_momentum", "trend_momentum", "hidden_accumulation"],
                "strategy_family_cn": ["趋势动量", "趋势动量", "隐性吸筹"],
                "priority_bucket": ["model_focus", "model_focus", "pattern_watch"],
                "forward_return_1d": [0.10, np.nan, -0.05],
                "max_adverse_return_1d": [-0.02, np.nan, -0.08],
            }
        )

        summary = summarize_strategy_family_forward_returns(samples, horizons=(1,))
        trend = summary[summary["strategy_family"].eq("trend_momentum")].iloc[0]

        self.assertEqual(trend["signal_count"], 2)
        self.assertEqual(trend["completed_count"], 1)
        self.assertEqual(trend["pending_count"], 1)
        self.assertAlmostEqual(trend["avg_return"], 0.10)
        self.assertAlmostEqual(trend["worst_adverse_return"], -0.02)

    def test_strategy_family_health_marks_insufficient_samples(self):
        summary = pd.DataFrame(
            {
                "strategy_family": ["hidden_accumulation"],
                "strategy_family_cn": ["隐性吸筹"],
                "horizon_days": [1],
                "signal_count": [15],
                "completed_count": [0],
                "pending_count": [15],
                "avg_return": [np.nan],
                "win_rate": [np.nan],
                "avg_adverse_return": [-0.03],
                "worst_adverse_return": [-0.06],
            }
        )

        health = build_strategy_family_health(summary, min_completed=20)

        self.assertEqual(health.loc[0, "family_health_status"], "insufficient")
        self.assertEqual(health.loc[0, "family_health_status_cn"], "样本不足")

    def test_strategy_family_health_marks_cool_down_when_returns_are_weak(self):
        summary = pd.DataFrame(
            {
                "strategy_family": ["strong_pullback", "strong_pullback"],
                "strategy_family_cn": ["强势回调二波", "强势回调二波"],
                "horizon_days": [1, 3],
                "signal_count": [100, 100],
                "completed_count": [60, 40],
                "pending_count": [40, 60],
                "avg_return": [-0.01, -0.06],
                "win_rate": [0.40, 0.20],
                "avg_adverse_return": [-0.04, -0.08],
                "worst_adverse_return": [-0.12, -0.18],
            }
        )

        health = build_strategy_family_health(summary, min_completed=20)

        self.assertEqual(health.loc[0, "selected_horizon_days"], 3)
        self.assertEqual(health.loc[0, "family_health_status"], "cool_down")
        self.assertEqual(health.loc[0, "family_health_status_cn"], "降温观察")

    def test_annotate_priority_watchlist_adds_health_warning_and_risk_flag(self):
        priority = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "stock_name": "Alpha",
                    "strategy_family": "strong_pullback",
                    "strategy_family_cn": "强势回调二波",
                    "strategy_family_reason": "强势趋势后的回调观察",
                    "risk_flags": "liquidity_watch",
                }
            ]
        )
        family_health = pd.DataFrame(
            [
                {
                    "strategy_family": "strong_pullback",
                    "strategy_family_cn": "强势回调二波",
                    "selected_horizon_days": 3,
                    "completed_count": 40,
                    "avg_return": -0.06,
                    "win_rate": 0.20,
                    "family_health_status": "cool_down",
                    "family_health_status_cn": "降温观察",
                    "family_health_reason": "3日平均收益 -6.00%",
                }
            ]
        )

        annotated = annotate_priority_watchlist_with_health(priority, family_health)

        self.assertEqual(annotated.loc[0, "family_health_status_cn"], "降温观察")
        self.assertIn("不追高", annotated.loc[0, "strategy_family_warning"])
        self.assertIn("liquidity_watch", annotated.loc[0, "risk_flags"])
        self.assertIn("strategy_family_cool_down", annotated.loc[0, "risk_flags"])

    def test_annotate_selected_table_is_idempotent_and_appends_strategy_warning_once(self):
        selected = pd.DataFrame(
            [
                {
                    "股票代码": "000001",
                    "股票名称": "Alpha",
                    "调整原因": "个人交易习惯层保留",
                    "趋势状态": "回调可观察",
                }
            ]
        )
        priority = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "stock_name": "Alpha",
                    "strategy_family": "strong_pullback",
                    "strategy_family_cn": "强势回调二波",
                    "strategy_family_reason": "强势趋势后的回调观察",
                    "family_health_status": "cool_down",
                    "family_health_status_cn": "降温观察",
                    "family_health_reason": "3日平均收益 -6.00%",
                    "family_health_horizon_days": 3,
                    "family_health_completed_count": 40,
                    "family_health_avg_return": -0.06,
                    "family_health_win_rate": 0.20,
                    "strategy_family_warning": "降温观察：3日平均收益 -6.00% 不追高，优先等待回撤修复或下一轮确认。",
                }
            ]
        )
        family_health = pd.DataFrame(
            [
                {
                    "strategy_family": "strong_pullback",
                    "strategy_family_cn": "强势回调二波",
                    "selected_horizon_days": 3,
                    "completed_count": 40,
                    "avg_return": -0.06,
                    "win_rate": 0.20,
                    "family_health_status": "cool_down",
                    "family_health_status_cn": "降温观察",
                    "family_health_reason": "3日平均收益 -6.00%",
                }
            ]
        )

        once = annotate_selected_table_with_health(selected, priority, family_health)
        twice = annotate_selected_table_with_health(once, priority, family_health)

        self.assertEqual(twice.columns.tolist().count("策略族风险提示"), 1)
        self.assertEqual(twice.columns.tolist().count("策略族健康状态中文"), 1)
        self.assertEqual(twice.loc[0, "调整原因"].count("策略族提示："), 1)
        self.assertEqual(twice.loc[0, "策略族中文"], "强势回调二波")
        self.assertIn("不追高", twice.loc[0, "策略族风险提示"])

    def test_write_strategy_family_forward_report_outputs_csv_and_markdown(self):
        samples = pd.DataFrame(
            {
                "asof_date": ["2026-07-03"],
                "symbol": ["000001"],
                "stock_name": ["Alpha"],
                "strategy_family": ["trend_momentum"],
                "strategy_family_cn": ["趋势动量"],
                "priority_bucket": ["model_focus"],
                "entry_date": ["2026-07-06"],
                "forward_return_1d": [0.03],
                "max_adverse_return_1d": [-0.01],
            }
        )
        family_summary = summarize_strategy_family_forward_returns(samples, horizons=(1,))
        family_health = build_strategy_family_health(family_summary, min_completed=1)
        bucket_summary = summarize_strategy_family_forward_returns(
            samples,
            horizons=(1,),
            group_cols=("strategy_family", "strategy_family_cn", "priority_bucket"),
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = write_strategy_family_forward_report(
                samples,
                family_summary,
                bucket_summary,
                family_health,
                Path(tmp),
                "20260703",
            )

            self.assertTrue(paths["samples"].exists())
            self.assertTrue(paths["family_summary"].exists())
            self.assertTrue(paths["bucket_summary"].exists())
            self.assertTrue(paths["family_health"].exists())
            self.assertTrue(paths["family_health_cn"].exists())
            report = paths["report"].read_text(encoding="utf-8")
            self.assertIn("策略族前瞻表现报告", report)
            self.assertIn("按策略族汇总", report)
            self.assertIn("策略族健康状态", report)


if __name__ == "__main__":
    unittest.main()
