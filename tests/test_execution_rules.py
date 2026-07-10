import unittest

import pandas as pd

from execution_rules import apply_open_constraints, limit_thresholds, next_open_return_label


class ExecutionRulesTest(unittest.TestCase):
    def test_next_open_return_label_uses_next_open_entry_and_following_open_exit(self):
        open_px = pd.DataFrame(
            {
                "000001": [10.0, 11.0, 12.1, 12.0],
                "300001": [20.0, 19.0, 18.05, 18.0],
            },
            index=pd.date_range("2026-01-01", periods=4, freq="D"),
        )

        label = next_open_return_label(open_px, max_abs_daily_return=0.20)

        self.assertAlmostEqual(label.iloc[0]["000001"], 12.1 / 11.0 - 1.0)
        self.assertAlmostEqual(label.iloc[0]["300001"], 18.05 / 19.0 - 1.0)
        self.assertTrue(pd.isna(label.iloc[2]["000001"]))
        self.assertTrue(pd.isna(label.iloc[3]["000001"]))

    def test_next_open_return_label_masks_extreme_returns(self):
        open_px = pd.DataFrame(
            {"000001": [10.0, 10.0, 15.0]},
            index=pd.date_range("2026-01-01", periods=3, freq="D"),
        )

        label = next_open_return_label(open_px, max_abs_daily_return=0.20)

        self.assertTrue(pd.isna(label.iloc[0]["000001"]))

    def test_limit_thresholds_uses_twenty_percent_for_chinext_and_ten_percent_elsewhere(self):
        thresholds = limit_thresholds(pd.Index(["000001", "300001", "301001", "600001"]))

        self.assertAlmostEqual(thresholds["000001"], 0.10)
        self.assertAlmostEqual(thresholds["300001"], 0.20)
        self.assertAlmostEqual(thresholds["301001"], 0.20)
        self.assertAlmostEqual(thresholds["600001"], 0.10)

    def test_apply_open_constraints_blocks_limit_up_buys_gap_buys_and_limit_down_sells(self):
        current = pd.Series({"000001": 0.0, "000002": 0.2, "300001": 0.1})
        target = pd.Series({"000001": 0.2, "000002": 0.0, "300001": 0.3})
        prev_close = pd.Series({"000001": 10.0, "000002": 10.0, "300001": 10.0})
        open_row = pd.Series({"000001": 10.0 * 1.10, "000002": 10.0 * 0.90, "300001": 10.0 * 1.16})

        adjusted = apply_open_constraints(
            current,
            target,
            open_row,
            prev_close,
            max_buy_open_gap=0.15,
            limit_buffer=0.995,
        )

        self.assertAlmostEqual(adjusted["000001"], current["000001"])
        self.assertAlmostEqual(adjusted["000002"], current["000002"])
        self.assertAlmostEqual(adjusted["300001"], current["300001"])

    def test_apply_open_constraints_can_disable_specific_blocks_for_backtest_configs(self):
        current = pd.Series({"000001": 0.0, "000002": 0.2})
        target = pd.Series({"000001": 0.2, "000002": 0.0})
        prev_close = pd.Series({"000001": 10.0, "000002": 10.0})
        open_row = pd.Series({"000001": 11.0, "000002": 9.0})

        adjusted = apply_open_constraints(
            current,
            target,
            open_row,
            prev_close,
            max_buy_open_gap=None,
            limit_buffer=0.995,
            block_limit_up_buys=False,
            block_limit_down_sells=False,
        )

        pd.testing.assert_series_equal(adjusted, target)


if __name__ == "__main__":
    unittest.main()
