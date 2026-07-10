import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from merged_daily_outputs import (
    _write_table,
    build_model_decision_table,
    build_priority_watchlist,
    build_state_pattern_scan,
    fill_missing_stock_names,
)


class MergedDailyOutputsTest(unittest.TestCase):
    def test_state_pattern_scan_joins_watchlist_with_trend_state(self):
        trend_state = pd.DataFrame(
            [
                {"symbol": "000001", "trend_state": "life_line_ok", "trend_score": 2.1},
                {"symbol": "000002", "trend_state": "trend_broken", "trend_score": -1.2},
            ]
        )
        watchlist = pd.DataFrame(
            [
                {
                    "symbol": "1",
                    "stock_name": "Alpha",
                    "pattern_type": "second_wave",
                    "pattern_score": 1.5,
                    "pattern_reason": "tight range",
                    "return_20d": 0.1,
                },
                {
                    "symbol": "000002",
                    "stock_name": "Beta",
                    "pattern_type": "platform",
                    "pattern_score": 0.8,
                    "pattern_reason": "cooling off",
                    "return_20d": -0.03,
                },
            ]
        )

        merged = build_state_pattern_scan(
            trend_state,
            watchlist,
            constructive_trend_states={"life_line_ok"},
        )

        self.assertEqual(list(merged["symbol"]), ["000001", "000002"])
        self.assertEqual(merged.loc[0, "trend_state"], "life_line_ok")
        self.assertEqual(merged.loc[0, "state_pattern_bucket"], "pattern_confirmed_by_trend")
        self.assertEqual(merged.loc[0, "strategy_family"], "strong_pullback")
        self.assertEqual(merged.loc[1, "state_pattern_bucket"], "pattern_needs_review")

    def test_state_pattern_scan_tags_hidden_accumulation_family(self):
        trend_state = pd.DataFrame(
            [{"symbol": "000001", "trend_state": "life_line_ok", "trend_score": 2.1}]
        )
        watchlist = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "stock_name": "Alpha",
                    "pattern_type": "隐性吸筹观察",
                    "pattern_score": 1.5,
                    "pattern_reason": "mild amount expansion",
                }
            ]
        )

        merged = build_state_pattern_scan(trend_state, watchlist)

        self.assertEqual(merged.loc[0, "strategy_family"], "hidden_accumulation")
        self.assertEqual(merged.loc[0, "strategy_family_cn"], "隐性吸筹")

    def test_model_decision_table_keeps_base_and_personal_decisions_together(self):
        base = pd.DataFrame(
            [
                {"symbol": "1", "rank": 1, "selected": True, "target_weight": 0.02, "score": 1.0},
                {"symbol": "000002", "rank": 2, "selected": True, "target_weight": 0.02, "score": 0.8},
                {"symbol": "000003", "rank": 3, "selected": False, "target_weight": 0.0, "score": 0.4},
            ]
        )
        overlay = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "personal_rank": 1,
                    "personal_selected": True,
                    "target_weight_before_behavior": 0.02,
                    "personal_adjusted_target_weight": 0.01,
                    "personal_adjusted_score": 0.7,
                    "personal_action": "reduce",
                    "personal_reasons": "position trimmed",
                },
                {
                    "symbol": "000002",
                    "personal_rank": 2,
                    "personal_selected": False,
                    "target_weight_before_behavior": 0.02,
                    "personal_adjusted_target_weight": 0.0,
                    "personal_adjusted_score": 0.2,
                    "personal_action": "watch_only",
                    "personal_reasons": "blocked",
                },
                {
                    "symbol": "000003",
                    "personal_rank": 3,
                    "personal_selected": True,
                    "target_weight_before_behavior": 0.0,
                    "personal_adjusted_target_weight": 0.02,
                    "personal_adjusted_score": 0.9,
                    "personal_action": "allow",
                    "personal_reasons": "filled slot",
                },
            ]
        )

        table = build_model_decision_table(base, overlay)

        self.assertEqual(list(table["symbol"]), ["000001", "000002", "000003"])
        self.assertEqual(table.loc[0, "base_rank"], 1)
        self.assertEqual(table.loc[0, "personal_rank"], 1)
        self.assertAlmostEqual(table.loc[0, "target_weight_delta"], -0.01)
        self.assertEqual(table.loc[0, "decision_layer"], "reduced_by_overlay")
        self.assertEqual(table.loc[1, "decision_layer"], "removed_by_overlay")
        self.assertEqual(table.loc[2, "decision_layer"], "added_by_overlay")

    def test_priority_watchlist_promotes_selected_confirmed_patterns(self):
        state_pattern_scan = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "stock_name": "Alpha",
                    "trend_state": "life_line_ok",
                    "pattern_type": "second_wave",
                    "pattern_score": 1.5,
                    "state_pattern_bucket": "pattern_confirmed_by_trend",
                    "pattern_reason": "confirmed",
                },
                {
                    "symbol": "000002",
                    "stock_name": "Beta",
                    "trend_state": "life_line_ok",
                    "pattern_type": "platform",
                    "pattern_score": 1.8,
                    "state_pattern_bucket": "pattern_confirmed_by_trend",
                    "pattern_reason": "watch",
                },
                {
                    "symbol": "000003",
                    "stock_name": "Gamma",
                    "trend_state": "trend_broken",
                    "pattern_type": "platform",
                    "pattern_score": 2.0,
                    "state_pattern_bucket": "pattern_needs_review",
                    "pattern_reason": "weak trend",
                },
            ]
        )
        model_decision_table = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "stock_name": "Alpha",
                    "personal_rank": 5,
                    "personal_selected": True,
                    "personal_target_weight": 0.02,
                    "decision_layer": "kept_by_overlay",
                },
                {
                    "symbol": "000002",
                    "stock_name": "Beta",
                    "personal_rank": 3,
                    "personal_selected": False,
                    "personal_target_weight": 0.0,
                    "decision_layer": "removed_by_overlay",
                    "personal_action": "watch_only",
                },
            ]
        )

        table = build_priority_watchlist(state_pattern_scan, model_decision_table, top_n=3)

        self.assertEqual(list(table["symbol"]), ["000001", "000002", "000003"])
        self.assertEqual(table.loc[0, "priority_bucket"], "action_focus")
        self.assertEqual(table.loc[0, "strategy_family"], "strong_pullback")
        self.assertEqual(table.loc[1, "priority_bucket"], "pattern_watch")
        self.assertEqual(table.loc[2, "priority_bucket"], "review_later")
        self.assertGreater(table.loc[0, "priority_score"], table.loc[1, "priority_score"])

    def test_priority_watchlist_prefers_pattern_strategy_family_over_model_family(self):
        state_pattern_scan = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "stock_name": "Alpha",
                    "strategy_family": "hidden_accumulation",
                    "strategy_family_cn": "隐性吸筹",
                    "strategy_family_reason": "pattern source",
                    "trend_state": "life_line_ok",
                    "pattern_type": "隐性吸筹观察",
                    "pattern_score": 1.5,
                    "state_pattern_bucket": "pattern_confirmed_by_trend",
                }
            ]
        )
        model_decision_table = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "stock_name": "Alpha",
                    "strategy_family": "trend_momentum",
                    "strategy_family_cn": "趋势动量",
                    "strategy_family_reason": "model source",
                    "personal_rank": 1,
                    "personal_selected": True,
                    "personal_target_weight": 0.02,
                    "decision_layer": "kept_by_overlay",
                }
            ]
        )

        table = build_priority_watchlist(state_pattern_scan, model_decision_table, top_n=1)

        self.assertEqual(table.loc[0, "strategy_family"], "hidden_accumulation")
        self.assertEqual(table.loc[0, "strategy_family_reason"], "pattern source")

    def test_priority_watchlist_ranks_model_focus_above_pure_pattern_watch(self):
        state_pattern_scan = pd.DataFrame(
            [
                {
                    "symbol": "000002",
                    "stock_name": "Beta",
                    "trend_state": "life_line_ok",
                    "pattern_type": "second_wave",
                    "pattern_score": 4.5,
                    "state_pattern_bucket": "pattern_confirmed_by_trend",
                }
            ]
        )
        model_decision_table = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "stock_name": "Alpha",
                    "personal_rank": 1,
                    "personal_selected": True,
                    "personal_target_weight": 0.02,
                    "decision_layer": "kept_by_overlay",
                }
            ]
        )

        table = build_priority_watchlist(state_pattern_scan, model_decision_table, top_n=2)

        self.assertEqual(list(table["symbol"]), ["000001", "000002"])
        self.assertEqual(table.loc[0, "priority_bucket"], "model_focus")
        self.assertEqual(table.loc[1, "priority_bucket"], "pattern_watch")

    def test_priority_watchlist_demotes_st_names_to_risk_watch(self):
        model_decision_table = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "stock_name": "*ST Sample",
                    "personal_rank": 1,
                    "personal_selected": True,
                    "personal_target_weight": 0.02,
                    "decision_layer": "kept_by_overlay",
                },
                {
                    "symbol": "000002",
                    "stock_name": "Normal",
                    "personal_rank": 2,
                    "personal_selected": True,
                    "personal_target_weight": 0.02,
                    "decision_layer": "kept_by_overlay",
                },
            ]
        )

        table = build_priority_watchlist(pd.DataFrame(), model_decision_table, top_n=2)

        self.assertEqual(list(table["symbol"]), ["000002", "000001"])
        self.assertEqual(table.loc[0, "priority_bucket"], "model_focus")
        self.assertEqual(table.loc[1, "priority_bucket"], "risk_watch")
        self.assertEqual(table.loc[1, "risk_flags"], "st_or_special_treatment")

    def test_priority_watchlist_keeps_risk_watch_visible_in_default_view(self):
        state_pattern_scan = pd.DataFrame(
            [
                {
                    "symbol": f"{index:06d}",
                    "stock_name": f"Pattern {index}",
                    "pattern_score": 1.0,
                    "state_pattern_bucket": "pattern_confirmed_by_trend",
                }
                for index in range(200001, 200021)
            ]
        )
        model_decision_table = pd.DataFrame(
            [
                {
                    "symbol": f"{index:06d}",
                    "stock_name": f"Model {index}",
                    "personal_rank": index,
                    "personal_selected": True,
                    "personal_target_weight": 0.02,
                    "decision_layer": "kept_by_overlay",
                }
                for index in range(1, 40)
            ]
            + [
                {
                    "symbol": "000040",
                    "stock_name": "*ST Risk",
                    "personal_rank": 40,
                    "personal_selected": True,
                    "personal_target_weight": 0.02,
                    "decision_layer": "kept_by_overlay",
                }
            ]
        )

        table = build_priority_watchlist(state_pattern_scan, model_decision_table)

        self.assertEqual(len(table), 50)
        self.assertEqual(
            table["priority_bucket"].value_counts().to_dict(),
            {"model_focus": 39, "risk_watch": 1, "pattern_watch": 10},
        )

    def test_priority_watchlist_caps_risk_watch_in_default_view(self):
        state_pattern_scan = pd.DataFrame(
            [
                {
                    "symbol": f"{index:06d}",
                    "stock_name": f"Pattern {index}",
                    "pattern_score": 1.0,
                    "state_pattern_bucket": "pattern_confirmed_by_trend",
                }
                for index in range(300001, 300021)
            ]
        )
        model_decision_table = pd.DataFrame(
            [
                {
                    "symbol": f"{index:06d}",
                    "stock_name": f"Model {index}",
                    "personal_rank": index,
                    "personal_selected": True,
                    "personal_target_weight": 0.02,
                    "decision_layer": "kept_by_overlay",
                }
                for index in range(1, 41)
            ]
            + [
                {
                    "symbol": f"{index:06d}",
                    "stock_name": f"*ST Risk {index}",
                    "personal_rank": index,
                    "personal_selected": False,
                    "personal_target_weight": 0.0,
                    "decision_layer": "unchanged_candidate",
                }
                for index in range(41, 61)
            ]
        )

        table = build_priority_watchlist(state_pattern_scan, model_decision_table)

        self.assertEqual(len(table), 50)
        self.assertEqual(
            table["priority_bucket"].value_counts().to_dict(),
            {"model_focus": 40, "risk_watch": 5, "pattern_watch": 5},
        )

    def test_priority_watchlist_default_keeps_model_focus_and_pattern_supplements(self):
        state_pattern_scan = pd.DataFrame(
            [
                {
                    "symbol": f"{index:06d}",
                    "stock_name": f"Pattern {index}",
                    "pattern_score": 1.0,
                    "state_pattern_bucket": "pattern_confirmed_by_trend",
                }
                for index in range(100001, 100021)
            ]
        )
        model_decision_table = pd.DataFrame(
            [
                {
                    "symbol": f"{index:06d}",
                    "stock_name": f"Model {index}",
                    "personal_rank": index,
                    "personal_selected": True,
                    "personal_target_weight": 0.02,
                    "decision_layer": "kept_by_overlay",
                }
                for index in range(1, 41)
            ]
        )

        table = build_priority_watchlist(state_pattern_scan, model_decision_table)

        self.assertEqual(len(table), 50)
        self.assertEqual(table["priority_bucket"].value_counts().to_dict(), {"model_focus": 40, "pattern_watch": 10})

    def test_fill_missing_stock_names_uses_name_map_without_overwriting_existing_names(self):
        table = pd.DataFrame(
            [
                {"symbol": "1", "stock_name": pd.NA},
                {"symbol": "000002", "stock_name": "Existing"},
            ]
        )

        filled = fill_missing_stock_names(table, {"000001": "Alpha", "000002": "Beta"})

        self.assertEqual(filled.loc[0, "stock_name"], "Alpha")
        self.assertEqual(filled.loc[1, "stock_name"], "Existing")

    def test_write_table_uses_pending_file_when_target_is_unwritable(self):
        table = pd.DataFrame([{"symbol": "000001", "score": 1.0}])
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "locked.csv"
            target.mkdir()

            actual = _write_table(target, table)

            self.assertEqual(actual.name, "locked_pending.csv")
            self.assertTrue(actual.exists())


if __name__ == "__main__":
    unittest.main()
