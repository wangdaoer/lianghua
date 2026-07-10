import unittest

import pandas as pd

from run_strong_pullback_satellite import (
    apply_rebound_exit,
    select_target_weights,
    should_apply_rebound_exit,
)


class StrongPullbackSatelliteTest(unittest.TestCase):
    def test_selects_top_candidates_with_leverage_cap(self):
        candidates = pd.DataFrame(
            {
                "score": [0.20, 0.50, 0.10],
            },
            index=["000001", "000002", "000003"],
        )

        target = select_target_weights(
            candidates,
            all_symbols=pd.Index(["000001", "000002", "000003", "000004"]),
            top_n=2,
            leverage=0.60,
            max_position_weight=0.40,
        )

        self.assertAlmostEqual(target["000002"], 0.30)
        self.assertAlmostEqual(target["000001"], 0.30)
        self.assertAlmostEqual(target["000003"], 0.0)
        self.assertAlmostEqual(target.sum(), 0.60)

    def test_empty_candidates_return_zero_weights(self):
        target = select_target_weights(
            pd.DataFrame({"score": []}),
            all_symbols=pd.Index(["000001", "000002"]),
            top_n=5,
            leverage=0.80,
            max_position_weight=0.10,
        )

        self.assertEqual(target.to_dict(), {"000001": 0.0, "000002": 0.0})

    def test_min_score_filters_low_conviction_candidates(self):
        candidates = pd.DataFrame(
            {
                "score": [0.20, -0.01, 0.10],
            },
            index=["000001", "000002", "000003"],
        )

        target = select_target_weights(
            candidates,
            all_symbols=pd.Index(["000001", "000002", "000003"]),
            top_n=3,
            leverage=0.60,
            max_position_weight=0.40,
            min_score=0.0,
        )

        self.assertAlmostEqual(target["000001"], 0.30)
        self.assertAlmostEqual(target["000003"], 0.30)
        self.assertAlmostEqual(target["000002"], 0.0)

    def test_basket_guard_can_cut_exposure_for_broken_pullback_basket(self):
        candidates = pd.DataFrame(
            {
                "score": [0.20, 0.10, 0.05],
                "return_20d": [-0.10, -0.09, -0.11],
                "raw_distance_ma60": [-0.05, -0.04, -0.06],
            },
            index=["000001", "000002", "000003"],
        )

        target = select_target_weights(
            candidates,
            all_symbols=pd.Index(["000001", "000002", "000003"]),
            top_n=3,
            leverage=0.60,
            max_position_weight=0.40,
            basket_guard_return_20d_min=-0.08,
            basket_guard_distance_ma60_min=-0.03,
            basket_guard_scale=0.0,
        )

        self.assertAlmostEqual(target.sum(), 0.0)

    def test_rebound_exit_reduces_only_existing_positions_that_hit_profit_threshold(self):
        current = pd.Series({"000001": 0.08, "000002": 0.08, "000003": 0.0})
        target = pd.Series({"000001": 0.08, "000002": 0.08, "000003": 0.08})
        close_row = pd.Series({"000001": 10.7, "000002": 10.3, "000003": 12.0})
        entry_price = pd.Series({"000001": 10.0, "000002": 10.0, "000003": 11.0})

        adjusted, hits = apply_rebound_exit(
            current,
            target,
            close_row,
            entry_price,
            rebound_exit_return=0.05,
            rebound_exit_scale=0.0,
        )

        self.assertAlmostEqual(adjusted["000001"], 0.0)
        self.assertAlmostEqual(adjusted["000002"], 0.08)
        self.assertAlmostEqual(adjusted["000003"], 0.08)
        self.assertEqual(hits["000001"], "rebound_exit")
        self.assertNotIn("000002", hits)
        self.assertNotIn("000003", hits)

    def test_rebound_exit_market_exposure_gate_only_enables_in_weak_market(self):
        self.assertFalse(
            should_apply_rebound_exit(
                rebound_exit_return=0.095,
                market_exposure=1.0,
                market_exposure_max=0.99,
            )
        )
        self.assertTrue(
            should_apply_rebound_exit(
                rebound_exit_return=0.095,
                market_exposure=0.6,
                market_exposure_max=0.99,
            )
        )
        self.assertTrue(
            should_apply_rebound_exit(
                rebound_exit_return=0.095,
                market_exposure=1.0,
                market_exposure_max=None,
            )
        )

    def test_rebound_exit_market_exposure_floor_only_enables_in_strong_market(self):
        self.assertFalse(
            should_apply_rebound_exit(
                rebound_exit_return=0.095,
                market_exposure=0.6,
                market_exposure_min=0.99,
            )
        )
        self.assertTrue(
            should_apply_rebound_exit(
                rebound_exit_return=0.095,
                market_exposure=1.0,
                market_exposure_min=0.99,
            )
        )


if __name__ == "__main__":
    unittest.main()
