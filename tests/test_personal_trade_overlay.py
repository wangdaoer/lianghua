import unittest

import pandas as pd

from apply_personal_trade_overlay import DEFAULT_RULES, apply_overlay, decide_row


class PersonalTradeOverlayTest(unittest.TestCase):
    def test_low_entry_position_gets_bonus(self):
        row = pd.Series(
            {
                "close_position": 0.40,
                "return_20d": 0.03,
                "personal_trades": 0,
            }
        )

        decision = decide_row(row, DEFAULT_RULES)

        self.assertEqual(decision["personal_action"], "allow")
        self.assertGreater(decision["personal_score_delta"], 0.0)
        self.assertIn("entry_in_low_or_mid_low_zone", decision["personal_reasons"])

    def test_high_entry_position_reduces_weight(self):
        row = pd.Series(
            {
                "close_position": 0.92,
                "return_20d": 0.08,
                "personal_trades": 0,
            }
        )

        decision = decide_row(row, DEFAULT_RULES)

        self.assertEqual(decision["personal_action"], "reduce")
        self.assertLess(decision["personal_score_delta"], 0.0)
        self.assertLess(decision["personal_weight_multiplier"], 1.0)

    def test_damaged_20d_trend_becomes_watch_only(self):
        row = pd.Series(
            {
                "close_position": 0.35,
                "return_20d": -0.25,
                "personal_trades": 0,
            }
        )

        decision = decide_row(row, DEFAULT_RULES)

        self.assertEqual(decision["personal_action"], "watch_only")
        self.assertEqual(decision["personal_weight_multiplier"], 0.0)

    def test_reselect_excludes_watch_only_and_reweights_selected_rows(self):
        candidates = pd.DataFrame(
            [
                {
                    "symbol": "1",
                    "score": 0.09,
                    "selected": True,
                    "target_weight": 0.02,
                    "close_position": 0.90,
                    "return_20d": -0.30,
                },
                {
                    "symbol": "2",
                    "score": 0.05,
                    "selected": False,
                    "target_weight": 0.0,
                    "close_position": 0.40,
                    "return_20d": 0.02,
                },
                {
                    "symbol": "3",
                    "score": 0.04,
                    "selected": False,
                    "target_weight": 0.0,
                    "close_position": 0.80,
                    "return_20d": 0.04,
                },
            ]
        )
        history = pd.DataFrame(columns=["symbol"])

        out = apply_overlay(
            candidates,
            history,
            DEFAULT_RULES,
            reselect_top_n=2,
            base_target_weight=0.02,
        )
        by_symbol = out.set_index("symbol")

        self.assertFalse(bool(by_symbol.loc["000001", "personal_selected"]))
        self.assertEqual(by_symbol.loc["000001", "personal_action"], "watch_only")
        self.assertTrue(bool(by_symbol.loc["000002", "personal_selected"]))
        self.assertTrue(bool(by_symbol.loc["000003", "personal_selected"]))
        self.assertAlmostEqual(by_symbol.loc["000002", "personal_adjusted_target_weight"], 0.02)
        self.assertAlmostEqual(by_symbol.loc["000003", "personal_adjusted_target_weight"], 0.01)

    def test_original_adjust_keeps_original_selection_without_filling_new_names(self):
        candidates = pd.DataFrame(
            [
                {
                    "symbol": "1",
                    "score": 0.20,
                    "selected": True,
                    "target_weight": 0.03,
                    "close_position": 0.40,
                    "return_20d": 0.02,
                },
                {
                    "symbol": "2",
                    "score": 0.90,
                    "selected": False,
                    "target_weight": 0.0,
                    "close_position": 0.40,
                    "return_20d": 0.02,
                },
            ]
        )
        history = pd.DataFrame(columns=["symbol"])

        out = apply_overlay(
            candidates,
            history,
            DEFAULT_RULES,
            reselect_top_n=2,
            selection_mode="original_adjust",
        )
        by_symbol = out.set_index("symbol")

        self.assertTrue(bool(by_symbol.loc["000001", "personal_selected"]))
        self.assertFalse(bool(by_symbol.loc["000002", "personal_selected"]))
        self.assertAlmostEqual(by_symbol.loc["000001", "personal_adjusted_target_weight"], 0.03)
        self.assertAlmostEqual(by_symbol.loc["000002", "personal_adjusted_target_weight"], 0.0)

    def test_special_treatment_stock_is_watch_only(self):
        decision = decide_row(
            pd.Series(
                {
                    "stock_name": "*ST示例",
                    "close_position": 0.20,
                    "return_20d": 0.10,
                    "personal_trades": 0,
                }
            ),
            DEFAULT_RULES,
        )

        self.assertEqual(decision["personal_action"], "watch_only")
        self.assertEqual(decision["personal_weight_multiplier"], 0.0)
        self.assertIn(
            "special_treatment_execution_unsupported",
            decision["personal_reasons"],
        )

    def test_st_letters_outside_name_prefix_do_not_trigger_watch_only(self):
        decision = decide_row(
            pd.Series(
                {
                    "stock_name": "BEST示例",
                    "close_position": 0.20,
                    "return_20d": 0.10,
                    "personal_trades": 0,
                }
            ),
            DEFAULT_RULES,
        )

        self.assertEqual(decision["personal_action"], "allow")
        self.assertNotIn(
            "special_treatment_execution_unsupported",
            decision["personal_reasons"],
        )

    def test_reselect_fills_st_seat_with_non_st_candidate(self):
        candidates = pd.DataFrame(
            [
                {
                    "symbol": "1",
                    "stock_name": "*ST示例",
                    "score": 0.90,
                    "selected": True,
                    "target_weight": 0.02,
                    "close_position": 0.20,
                    "return_20d": 0.10,
                },
                {
                    "symbol": "2",
                    "stock_name": "普通股票",
                    "score": 0.80,
                    "selected": False,
                    "target_weight": 0.0,
                    "close_position": 0.20,
                    "return_20d": 0.10,
                },
            ]
        )

        out = apply_overlay(
            candidates,
            pd.DataFrame(columns=["symbol"]),
            DEFAULT_RULES,
            reselect_top_n=1,
            base_target_weight=0.02,
        ).set_index("symbol")

        self.assertFalse(bool(out.loc["000001", "personal_selected"]))
        self.assertTrue(bool(out.loc["000002", "personal_selected"]))


if __name__ == "__main__":
    unittest.main()
