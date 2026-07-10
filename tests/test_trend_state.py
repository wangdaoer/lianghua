import unittest

import numpy as np
import pandas as pd

from trend_state import classify_trend_state, choose_realtime_lifeline


class TrendStateTest(unittest.TestCase):
    def test_detects_ignition_observation_on_volume_breakout(self):
        dates = pd.date_range("2025-01-01", periods=140, freq="D")
        close = pd.Series(10.0, index=dates)
        high = close * 1.01
        low = close * 0.99
        amount = pd.Series(10_000_000.0, index=dates)
        close.iloc[-1] = 11.4
        high.iloc[-1] = 11.5
        low.iloc[-1] = 10.9
        amount.iloc[-1] = 25_000_000.0

        state = classify_trend_state(close, high, low, amount, dates[-1])

        self.assertEqual(state["trend_state"], "起爆观察")
        self.assertTrue(state["is_ignition"])

    def test_detects_pullback_observable_when_trend_holds_lifeline(self):
        dates = pd.date_range("2025-01-01", periods=220, freq="D")
        close = pd.Series(np.linspace(10.0, 30.0, len(dates)), index=dates)
        high = close * 1.02
        low = close * 0.98
        amount = pd.Series(50_000_000.0, index=dates)
        close.iloc[-6:] = [31.0, 30.2, 29.4, 28.8, 28.4, 28.0]
        high.iloc[-6:] = close.iloc[-6:] * 1.01
        low.iloc[-6:] = close.iloc[-6:] * 0.99

        state = classify_trend_state(close, high, low, amount, dates[-1])

        self.assertEqual(state["trend_state"], "回调可观察")
        self.assertGreaterEqual(state["distance_to_lifeline"], -0.03)

    def test_detects_broken_trend_after_three_lifeline_breaches(self):
        dates = pd.date_range("2025-01-01", periods=240, freq="D")
        close = pd.Series(np.linspace(10.0, 28.0, len(dates)), index=dates)
        high = close * 1.02
        low = close * 0.98
        amount = pd.Series(60_000_000.0, index=dates)
        lifeline = close.rolling(120, min_periods=60).mean()
        close.iloc[-3:] = lifeline.iloc[-3:] * 0.94
        high.iloc[-3:] = close.iloc[-3:] * 1.01
        low.iloc[-3:] = close.iloc[-3:] * 0.99

        state = classify_trend_state(close, high, low, amount, dates[-1])

        self.assertEqual(state["trend_state"], "趋势破坏")
        self.assertGreaterEqual(state["lifeline_breach_days_5"], 3)

    def test_realtime_lifeline_uses_only_past_data(self):
        dates = pd.date_range("2025-01-01", periods=200, freq="D")
        close = pd.Series(np.linspace(10.0, 25.0, len(dates)), index=dates)
        close.iloc[-1] = 5.0

        before_crash = choose_realtime_lifeline(close, dates[-2])
        after_crash = choose_realtime_lifeline(close, dates[-1])

        self.assertNotEqual(before_crash["distance_to_lifeline"], after_crash["distance_to_lifeline"])
        self.assertGreater(before_crash["distance_to_lifeline"], 0.0)


if __name__ == "__main__":
    unittest.main()
