import unittest

import pandas as pd

from run_core_satellite_overlay import (
    apply_overlay_satellite_rebound_exit,
    build_core_eligibility_mask,
    combine_target_weights,
    select_rank_targets,
)


class CoreSatelliteOverlayTest(unittest.TestCase):
    def test_select_rank_targets_uses_top_scores_and_caps_position_weight(self):
        score = pd.Series(
            [0.1, 0.5, 0.3, -0.2],
            index=["000001", "000002", "000003", "000004"],
        )

        target = select_rank_targets(
            score,
            top_n=3,
            sleeve_leverage=0.75,
            max_position_weight=0.20,
        )

        self.assertAlmostEqual(target["000002"], 0.20)
        self.assertAlmostEqual(target["000003"], 0.20)
        self.assertAlmostEqual(target["000001"], 0.20)
        self.assertNotIn("000004", target.index)

    def test_combine_target_weights_caps_overlap_and_total_exposure(self):
        all_symbols = pd.Index(["000001", "000002", "000003"])
        core = pd.Series({"000001": 0.06, "000002": 0.06})
        satellite = pd.Series({"000002": 0.08, "000003": 0.08})

        combined = combine_target_weights(
            core,
            satellite,
            all_symbols=all_symbols,
            total_leverage=0.18,
            max_position_weight=0.10,
        )

        self.assertAlmostEqual(combined["000002"], 0.10)
        self.assertLessEqual(float(combined.sum()), 0.18)
        self.assertGreater(combined["000001"], 0.0)
        self.assertGreater(combined["000003"], 0.0)

    def test_satellite_rebound_exit_only_reduces_satellite_sleeve_when_gate_is_on(self):
        current_satellite = pd.Series({"000001": 0.08, "000002": 0.08})
        target_satellite = pd.Series({"000001": 0.08, "000002": 0.08})
        close_row = pd.Series({"000001": 10.9, "000002": 10.2})
        entry_price = pd.Series({"000001": 10.0, "000002": 10.0})

        adjusted, hits = apply_overlay_satellite_rebound_exit(
            current_satellite,
            target_satellite,
            close_row,
            entry_price,
            market_exposure=1.0,
            rebound_exit_return=0.085,
            rebound_exit_scale=0.0,
            rebound_exit_market_exposure_min=0.99,
        )

        self.assertAlmostEqual(adjusted["000001"], 0.0)
        self.assertAlmostEqual(adjusted["000002"], 0.08)
        self.assertEqual(hits["000001"], "rebound_exit")

        gated, gated_hits = apply_overlay_satellite_rebound_exit(
            current_satellite,
            target_satellite,
            close_row,
            entry_price,
            market_exposure=0.6,
            rebound_exit_return=0.085,
            rebound_exit_scale=0.0,
            rebound_exit_market_exposure_min=0.99,
        )

        self.assertAlmostEqual(gated["000001"], 0.08)
        self.assertEqual(gated_hits, {})

    def test_core_eligibility_mask_uses_only_known_price_liquidity_and_risk_filters(self):
        dates = pd.date_range("2024-01-01", periods=25)
        close = pd.DataFrame(
            {
                "000001": [10.0] * 25,
                "000002": [1.8] * 25,
                "000003": [9.0] * 25,
                "000004": [10.0] * 20 + [9.5, 9.0, 8.55, 8.12, 7.71],
                "000005": [11.0] * 25,
            },
            index=dates,
        )
        amount = pd.DataFrame(
            {
                "000001": [80_000_000.0] * 25,
                "000002": [80_000_000.0] * 25,
                "000003": [10_000_000.0] * 25,
                "000004": [80_000_000.0] * 25,
                "000005": [80_000_000.0] * 25,
            },
            index=dates,
        )

        mask = build_core_eligibility_mask(
            close,
            amount,
            row_index=24,
            min_close=2.0,
            min_avg_amount_20d=30_000_000.0,
            max_large_down_days_20=2,
            large_down_threshold=-0.045,
            exclude_symbols={"000005"},
        )

        self.assertTrue(mask["000001"])
        self.assertFalse(mask["000002"])
        self.assertFalse(mask["000003"])
        self.assertFalse(mask["000004"])
        self.assertFalse(mask["000005"])


if __name__ == "__main__":
    unittest.main()
