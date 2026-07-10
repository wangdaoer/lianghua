import importlib
import unittest

import pandas as pd


class BehaviorOverlaySegmentsTest(unittest.TestCase):
    def _module(self):
        try:
            return importlib.import_module("summarize_behavior_overlay_segments")
        except ImportError as exc:
            self.fail(f"missing segment summary module: {exc}")

    def test_add_return_columns_uses_initial_capital_for_first_day(self):
        mod = self._module()
        frame = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "base_equity": [110.0, 121.0],
                "overlay_equity": [105.0, 115.5],
                "market_exposure": [1.0, 0.65],
            }
        )

        out = mod.add_return_columns(frame, initial_capital=100.0)

        self.assertAlmostEqual(out.loc[0, "base_daily_return"], 0.10)
        self.assertAlmostEqual(out.loc[0, "overlay_daily_return"], 0.05)
        self.assertAlmostEqual(out.loc[1, "base_daily_return"], 0.10)

    def test_segment_metrics_compounds_returns_by_year(self):
        mod = self._module()
        frame = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03", "2025-01-02", "2025-01-03"],
                "base_equity": [110.0, 121.0, 108.9, 119.79],
                "overlay_equity": [105.0, 115.5, 109.725, 120.6975],
                "market_exposure": [1.0, 0.65, 0.10, 1.0],
            }
        )
        enriched = mod.add_return_columns(frame, initial_capital=100.0)
        enriched["year"] = pd.to_datetime(enriched["date"]).dt.year.astype(str)

        summary = mod.segment_metrics(enriched, "year")
        by_year = summary.set_index("segment")

        self.assertAlmostEqual(by_year.loc["2024", "base_return"], 0.21)
        self.assertAlmostEqual(by_year.loc["2025", "base_return"], -0.01)
        self.assertEqual(by_year.loc["2024", "days"], 2)

    def test_classifies_market_regime_from_exposure(self):
        mod = self._module()

        self.assertEqual(mod.classify_market_regime(1.0), "risk_on")
        self.assertEqual(mod.classify_market_regime(0.65), "reduced")
        self.assertEqual(mod.classify_market_regime(0.10), "defensive")


if __name__ == "__main__":
    unittest.main()
