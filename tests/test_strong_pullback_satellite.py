import json
import unittest

import numpy as np
import pandas as pd

from run_strong_pullback_satellite import (
    apply_rebound_exit,
    run_satellite_walk_forward,
    select_target_weights,
    should_apply_rebound_exit,
)


class StrongPullbackSatelliteTest(unittest.TestCase):
    def test_trade_audit_symbol_contributions_sum_to_gross_return(self):
        dates = pd.bdate_range("2025-01-01", periods=90)
        symbols = pd.Index(["000001", "000002"])
        rising = 10.0 * np.power(1.02, np.arange(61))
        pullback = rising[-1] * np.power(0.99, np.arange(1, 30))
        close = pd.DataFrame(
            {"000001": np.concatenate((rising, pullback)), "000002": np.concatenate((rising * 1.01, pullback * 1.01))},
            index=dates,
        )
        open_px = close * 0.998
        high = close * 1.01
        low = close * 0.99
        amount = pd.DataFrame(50_000_000.0, index=dates, columns=symbols)
        market_exposure = pd.Series(1.0, index=dates)

        _, _, trades, _ = run_satellite_walk_forward(
            close, open_px, high, low, amount,
            train_days=65,
            retrain_frequency=5,
            top_n=1,
            rebalance_frequency=1,
            max_position_weight=0.6,
            leverage=0.6,
            min_score=None,
            commission_bps=0.0,
            impact_bps=0.0,
            max_buy_open_gap=0.05,
            limit_buffer=1.0,
            market_exposure=market_exposure,
            initial_capital=1_000_000.0,
            filter_kwargs={},
        )

        contributions = [json.loads(row.symbol_contributions_json) for row in trades.itertuples()]
        self.assertTrue(any(any(value != 0.0 for value in row.values()) for row in contributions))
        for row, contribution in zip(trades.itertuples(), contributions):
            self.assertAlmostEqual(sum(contribution.values()), row.gross_return)

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
