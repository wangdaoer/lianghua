import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from build_daily_personal_overlay_report import (
    _markdown_table,
    build_risk_warnings,
    build_watchlist_view,
    build_selected_view,
    load_name_map,
    summarize_overlay_changes,
)


class DailyPersonalOverlayReportTest(unittest.TestCase):
    def test_summarize_overlay_changes_counts_kept_added_removed_and_reduced(self):
        base = pd.DataFrame(
            [
                {"symbol": "000001", "selected": True, "target_weight": 0.02},
                {"symbol": "000002", "selected": True, "target_weight": 0.02},
                {"symbol": "000003", "selected": False, "target_weight": 0.0},
                {"symbol": "000004", "selected": False, "target_weight": 0.0},
            ]
        )
        overlay = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "personal_selected": True,
                    "target_weight_before_behavior": 0.02,
                    "personal_adjusted_target_weight": 0.01,
                    "personal_action": "reduce",
                },
                {
                    "symbol": "000002",
                    "personal_selected": False,
                    "target_weight_before_behavior": 0.02,
                    "personal_adjusted_target_weight": 0.0,
                    "personal_action": "watch_only",
                },
                {
                    "symbol": "000003",
                    "personal_selected": True,
                    "target_weight_before_behavior": 0.0,
                    "personal_adjusted_target_weight": 0.02,
                    "personal_action": "allow",
                },
            ]
        )

        summary = summarize_overlay_changes(base, overlay)

        self.assertEqual(summary["base_selected_count"], 2)
        self.assertEqual(summary["overlay_selected_count"], 2)
        self.assertEqual(summary["kept_count"], 1)
        self.assertEqual(summary["added_symbols"], ["000003"])
        self.assertEqual(summary["removed_symbols"], ["000002"])
        self.assertEqual(summary["reduced_selected_count"], 1)
        self.assertAlmostEqual(summary["base_gross_weight"], 0.04)
        self.assertAlmostEqual(summary["overlay_gross_weight"], 0.03)

    def test_build_selected_view_uses_chinese_headers_and_weight_delta(self):
        overlay = pd.DataFrame(
            [
                {
                    "symbol": "1",
                    "rank": 7,
                    "personal_rank": 2,
                    "personal_selected": True,
                    "target_weight_before_behavior": 0.02,
                    "personal_adjusted_target_weight": 0.012,
                    "personal_action_cn": "降仓",
                    "personal_reasons_cn": "20日动量偏弱，降低仓位",
                    "trend_state": "生命线健康",
                    "return_20d": -0.08,
                    "close_position": 0.33,
                    "score": 0.01,
                    "personal_adjusted_score": 0.0,
                }
            ]
        )

        name_map = {"000001": "平安银行"}

        view = build_selected_view(overlay, name_map=name_map)

        self.assertIn("股票代码", view.columns)
        self.assertIn("股票名称", view.columns)
        self.assertIn("原模型排名", view.columns)
        self.assertIn("个人习惯层权重", view.columns)
        self.assertIn("调整原因", view.columns)
        self.assertEqual(view.loc[0, "股票代码"], "000001")
        self.assertEqual(view.loc[0, "股票名称"], "平安银行")
        self.assertAlmostEqual(view.loc[0, "权重变化"], -0.008)

    def test_build_selected_view_includes_shadow_account_prompt_columns_when_present(self):
        overlay = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "personal_selected": True,
                    "personal_adjusted_target_weight": 0.02,
                    "shadow_account_signal_cn": "风险提示",
                    "shadow_account_notes": "high_quarter: 历史亏损显著",
                }
            ]
        )

        view = build_selected_view(overlay, name_map={"000001": "平安银行"})

        self.assertIn("影子账户提示", view.columns)
        self.assertIn("影子账户说明", view.columns)
        self.assertEqual(view.loc[0, "影子账户提示"], "风险提示")
        self.assertIn("high_quarter", view.loc[0, "影子账户说明"])

    def test_load_name_map_accepts_daily_market_data_headers(self):
        names = pd.DataFrame(
            [
                {"security_code": "1", "security_name": "平安银行"},
                {"security_code": "600000", "security_name": "浦发银行"},
            ]
        )

        name_map = load_name_map(names)

        self.assertEqual(name_map["000001"], "平安银行")
        self.assertEqual(name_map["600000"], "浦发银行")

    def test_load_name_map_accepts_tab_text_with_xls_suffix(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ths_hs_a_share_2026-06-30.xls"
            path.write_bytes(
                "代码\t    名称\t现价\nSZ300044\t*ST赛为\t2.95\nSH688728\t格科微\t12.80\n".encode(
                    "gb18030"
                )
            )

            name_map = load_name_map(path)

        self.assertEqual(name_map["300044"], "*ST赛为")
        self.assertEqual(name_map["688728"], "格科微")

    def test_markdown_table_keeps_leading_zero_stock_codes(self):
        table = pd.DataFrame(
            [
                {
                    "股票代码": "000001",
                    "原模型排名": 1,
                    "个人习惯层权重": 0.02,
                }
            ]
        )

        markdown = _markdown_table(table)

        self.assertIn("000001", markdown)

    def test_build_risk_warnings_mentions_watch_only_and_drawdown(self):
        summary = {
            "overlay_gross_weight": 0.72,
            "removed_count": 2,
            "reduced_selected_count": 25,
            "action_counts": {"watch_only": 8},
        }
        metrics = {"personal_overlay": {"max_drawdown": -0.217}}

        warnings = build_risk_warnings(summary, metrics)

        joined = "\n".join(warnings)
        self.assertIn("只观察", joined)
        self.assertIn("降仓", joined)
        self.assertIn("最大回撤", joined)

    def test_build_watchlist_view_keeps_pattern_reason_and_names(self):
        watchlist = pd.DataFrame(
            [
                {
                    "symbol": "1",
                    "pattern_type": "强势平台蓄势",
                    "pattern_score": 1.2,
                    "pattern_reason": "上涨后横向消化，均线托住",
                    "close": 12.3,
                    "return_20d": 0.08,
                    "return_60d": 0.35,
                    "range_15d": 0.12,
                }
            ]
        )

        view = build_watchlist_view(watchlist, {"000001": "平安银行"})

        self.assertEqual(view.loc[0, "股票代码"], "000001")
        self.assertEqual(view.loc[0, "股票名称"], "平安银行")
        self.assertEqual(view.loc[0, "观察模式"], "强势平台蓄势")
        self.assertIn("均线托住", view.loc[0, "观察理由"])


if __name__ == "__main__":
    unittest.main()
