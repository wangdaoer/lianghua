import unittest

import numpy as np
import pandas as pd

from early_pattern_watchlist import (
    detect_pattern_row,
    is_high_quality_hidden_accumulation,
    scan_early_patterns,
)


def make_ohlc(close_values: list[float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2026-01-01", periods=len(close_values), freq="D")
    close = pd.DataFrame({"000001": close_values}, index=dates)
    open_px = close.shift(1).fillna(close.iloc[0]) * 0.995
    high = close * 1.025
    low = close * 0.975
    amount = pd.DataFrame({"000001": np.linspace(80_000_000, 130_000_000, len(close_values))}, index=dates)
    return close, open_px, high, low, amount


class EarlyPatternWatchlistTest(unittest.TestCase):
    def test_detects_strong_platform_digest_pattern(self):
        close_values = list(np.linspace(10, 18, 70)) + [
            18.5,
            19.2,
            19.8,
            19.4,
            19.7,
            19.3,
            19.6,
            19.9,
            19.5,
            19.8,
            20.1,
            19.7,
            20.0,
            20.2,
            20.4,
        ]
        close, open_px, high, low, amount = make_ohlc(close_values)

        table = scan_early_patterns(close, open_px, high, low, amount)

        self.assertEqual(table.loc[0, "pattern_type"], "强势平台蓄势")
        self.assertGreater(table.loc[0, "pattern_score"], 0.0)

    def test_detects_second_wave_breakout_setup(self):
        close_values = (
            list(np.linspace(8, 16, 45))
            + list(np.linspace(16, 11.5, 25))
            + list(np.linspace(11.5, 15.5, 25))
            + [15.3, 15.7, 16.2, 16.8, 17.5]
        )
        close, open_px, high, low, amount = make_ohlc(close_values)
        amount.iloc[-1, 0] = amount.iloc[-20:-1, 0].median() * 2.0

        decision = detect_pattern_row(close["000001"], open_px["000001"], high["000001"], low["000001"], amount["000001"])

        self.assertEqual(decision["pattern_type"], "趋势二波启动")
        self.assertIn("重新放量", decision["pattern_reason"])

    def test_excludes_one_way_downtrend(self):
        close_values = list(np.linspace(20, 8, 120))
        close, open_px, high, low, amount = make_ohlc(close_values)

        table = scan_early_patterns(close, open_px, high, low, amount)

        self.assertTrue(table.empty)

    def test_excludes_already_overheated_breakout(self):
        close_values = (
            list(np.linspace(8, 14, 60))
            + list(np.linspace(14, 11, 20))
            + list(np.linspace(11, 16, 15))
            + [18.5, 22.0, 26.0, 30.0, 34.0]
        )
        close, open_px, high, low, amount = make_ohlc(close_values)
        amount.iloc[-1, 0] = amount.iloc[-20:-1, 0].median() * 2.5

        table = scan_early_patterns(close, open_px, high, low, amount)

        self.assertTrue(table.empty)

    def test_detects_hidden_accumulation_with_steady_amount_and_price_lift(self):
        base = list(np.linspace(10.0, 12.0, 90))
        digest = [
            12.00,
            11.92,
            12.05,
            12.02,
            12.12,
            12.10,
            12.18,
            12.21,
            12.27,
            12.33,
            12.39,
            12.44,
            12.50,
            12.57,
            12.64,
            12.70,
            12.78,
            12.84,
            12.91,
            13.00,
        ]
        close, open_px, high, low, amount = make_ohlc(base + digest)
        amount.iloc[-20:-10, 0] = 50_000_000
        amount.iloc[-10:, 0] = 200_000_000

        decision = detect_pattern_row(close["000001"], open_px["000001"], high["000001"], low["000001"], amount["000001"])

        self.assertEqual(decision["pattern_type"], "隐性吸筹观察")
        self.assertGreater(decision["hidden_accumulation_score"], 0.0)
        self.assertLessEqual(decision["pattern_score"], 1.85)
        self.assertGreater(decision["up_day_amount_persistence"], 0.55)
        self.assertIn("连续温和放量", decision["pattern_reason"])

    def test_rejects_single_day_volume_spike_as_hidden_accumulation(self):
        base = list(np.linspace(10.0, 12.0, 90))
        digest = [12.0, 11.96, 12.03, 12.01, 12.05, 12.08, 12.1, 12.12, 12.15, 12.18]
        close, open_px, high, low, amount = make_ohlc(base + digest)
        amount.iloc[-10:-1, 0] = 90_000_000
        amount.iloc[-1, 0] = 500_000_000

        decision = detect_pattern_row(close["000001"], open_px["000001"], high["000001"], low["000001"], amount["000001"])

        self.assertNotEqual(decision["pattern_type"], "隐性吸筹观察")

    def test_high_quality_hidden_accumulation_requires_moderate_trend_and_volume(self):
        self.assertTrue(
            is_high_quality_hidden_accumulation(
                {
                    "return_20d": 0.10,
                    "drawdown_20d": -0.04,
                    "distance_ma20": 0.05,
                    "amount_ratio": 1.30,
                }
            )
        )
        self.assertFalse(
            is_high_quality_hidden_accumulation(
                {
                    "return_20d": 0.22,
                    "drawdown_20d": -0.04,
                    "distance_ma20": 0.05,
                    "amount_ratio": 1.30,
                }
            )
        )
        self.assertFalse(
            is_high_quality_hidden_accumulation(
                {
                    "return_20d": 0.10,
                    "drawdown_20d": -0.04,
                    "distance_ma20": 0.05,
                    "amount_ratio": 2.10,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
