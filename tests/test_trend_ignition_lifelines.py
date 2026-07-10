import unittest

import numpy as np
import pandas as pd

from analyze_trend_ignition_lifelines import (
    choose_lifeline,
    find_ignition_candidates,
    summarize_trends_for_symbol,
)


class TrendIgnitionLifelineTest(unittest.TestCase):
    def test_finds_volume_breakout_ignition_before_large_trend(self):
        dates = pd.date_range("2025-01-01", periods=180, freq="D")
        close = pd.Series(10.0, index=dates)
        high = close * 1.01
        amount = pd.Series(10_000_000.0, index=dates)
        close.iloc[90:] = np.linspace(11.5, 28.0, len(dates) - 90)
        high.iloc[90:] = close.iloc[90:] * 1.01
        amount.iloc[90] = 25_000_000.0

        ignitions = find_ignition_candidates(
            close,
            high,
            amount,
            breakout_window=60,
            amount_window=20,
            amount_multiplier=1.5,
            breakout_buffer=0.01,
        )

        self.assertIn(dates[90], ignitions.index)

    def test_choose_lifeline_prefers_shortest_ma_that_holds(self):
        dates = pd.date_range("2025-01-01", periods=180, freq="D")
        close = pd.Series(np.linspace(10.0, 30.0, len(dates)), index=dates)
        close.iloc[125:128] *= 0.92
        ignition_date = dates[80]
        peak_date = dates[-1]

        lifeline = choose_lifeline(
            close,
            ignition_date,
            peak_date,
            windows=(20, 60),
            breach_buffer=0.03,
            max_breach_days=0,
        )

        self.assertEqual(lifeline["lifeline_ma"], 60)

    def test_summarize_trends_keeps_only_segments_above_threshold(self):
        dates = pd.date_range("2025-01-01", periods=220, freq="D")
        close = pd.Series(10.0, index=dates)
        high = close * 1.01
        amount = pd.Series(10_000_000.0, index=dates)
        close.iloc[100:] = np.linspace(11.5, 25.0, len(dates) - 100)
        high.iloc[100:] = close.iloc[100:] * 1.01
        amount.iloc[100] = 30_000_000.0

        trends = summarize_trends_for_symbol(
            symbol="000001",
            close=close,
            high=high,
            amount=amount,
            start=pd.Timestamp("2025-06-01"),
            end=pd.Timestamp("2025-08-08"),
            min_trend_return=0.80,
        )

        self.assertEqual(len(trends), 1)
        self.assertEqual(trends[0]["symbol"], "000001")
        self.assertGreaterEqual(trends[0]["peak_return"], 0.80)


if __name__ == "__main__":
    unittest.main()
