import unittest

import numpy as np
import pandas as pd

from generate_strong_pullback_candidates import (
    filter_strong_pullback_candidates,
    latest_raw_metrics,
)
from train_next_open_rank_model import build_features


class StrongPullbackCandidateFilterTest(unittest.TestCase):
    def test_excludes_one_way_downtrend_even_with_high_score(self):
        candidates = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "score": 0.99,
                    "close": 4.5,
                    "return_5d": -0.10,
                    "return_20d": -0.18,
                    "return_60d": -0.35,
                    "prior_return_20_before_pullback": -0.12,
                    "prior_return_60_before_pullback": -0.20,
                    "distance_ma60": -0.18,
                    "raw_intraday_return": -0.01,
                    "avg_amount_20d": 100_000_000,
                },
                {
                    "symbol": "000002",
                    "score": 0.50,
                    "close": 12.0,
                    "return_5d": -0.08,
                    "return_20d": 0.06,
                    "return_60d": 0.28,
                    "prior_return_20_before_pullback": 0.16,
                    "prior_return_60_before_pullback": 0.34,
                    "distance_ma60": 0.04,
                    "raw_intraday_return": -0.02,
                    "avg_amount_20d": 80_000_000,
                },
            ]
        )

        filtered = filter_strong_pullback_candidates(candidates)

        self.assertEqual(filtered["symbol"].tolist(), ["000002"])

    def test_excludes_illiquid_or_excessive_pullback_rows(self):
        candidates = pd.DataFrame(
            [
                {
                    "symbol": "000003",
                    "score": 0.70,
                    "close": 8.0,
                    "return_5d": -0.22,
                    "return_20d": 0.12,
                    "return_60d": 0.30,
                    "prior_return_20_before_pullback": 0.20,
                    "prior_return_60_before_pullback": 0.45,
                    "distance_ma60": 0.02,
                    "raw_intraday_return": -0.01,
                    "avg_amount_20d": 100_000_000,
                },
                {
                    "symbol": "000004",
                    "score": 0.65,
                    "close": 10.0,
                    "return_5d": -0.07,
                    "return_20d": 0.08,
                    "return_60d": 0.24,
                    "prior_return_20_before_pullback": 0.18,
                    "prior_return_60_before_pullback": 0.40,
                    "distance_ma60": 0.03,
                    "raw_intraday_return": -0.01,
                    "avg_amount_20d": 5_000_000,
                },
            ]
        )

        filtered = filter_strong_pullback_candidates(candidates)

        self.assertTrue(filtered.empty)

    def test_excludes_short_burst_inside_negative_sixty_day_trend(self):
        candidates = pd.DataFrame(
            [
                {
                    "symbol": "000005",
                    "score": 0.80,
                    "close": 9.0,
                    "return_5d": -0.08,
                    "return_20d": -0.02,
                    "return_60d": -0.05,
                    "prior_return_20_before_pullback": 0.22,
                    "prior_return_60_before_pullback": 0.02,
                    "distance_ma60": -0.03,
                    "raw_intraday_return": -0.01,
                    "avg_amount_20d": 120_000_000,
                }
            ]
        )

        filtered = filter_strong_pullback_candidates(candidates)

        self.assertTrue(filtered.empty)

    def test_excludes_st_named_rows_when_name_is_available(self):
        candidates = pd.DataFrame(
            [
                {
                    "symbol": "000006",
                    "stock_name": "*ST海华",
                    "score": 0.90,
                    "close": 9.0,
                    "return_5d": -0.08,
                    "return_20d": 0.08,
                    "return_60d": 0.24,
                    "prior_return_20_before_pullback": 0.18,
                    "prior_return_60_before_pullback": 0.40,
                    "distance_ma60": 0.03,
                    "raw_intraday_return": -0.01,
                    "avg_amount_20d": 120_000_000,
                },
                {
                    "symbol": "000007",
                    "stock_name": "正常股份",
                    "score": 0.80,
                    "close": 9.0,
                    "return_5d": -0.08,
                    "return_20d": 0.08,
                    "return_60d": 0.24,
                    "prior_return_20_before_pullback": 0.18,
                    "prior_return_60_before_pullback": 0.40,
                    "distance_ma60": 0.03,
                    "raw_intraday_return": -0.01,
                    "avg_amount_20d": 120_000_000,
                },
            ]
        )

        filtered = filter_strong_pullback_candidates(candidates)

        self.assertEqual(filtered["symbol"].tolist(), ["000007"])

    def test_raw_metric_names_do_not_overlap_rank_feature_names(self):
        dates = pd.date_range("2026-01-01", periods=80, freq="D")
        columns = ["000001", "000002"]
        base = pd.DataFrame(
            np.linspace(10, 20, len(dates) * len(columns)).reshape(len(dates), len(columns)),
            index=dates,
            columns=columns,
        )
        open_px = base * 0.99
        high = base * 1.02
        low = base * 0.98
        amount = base * 1_000_000

        feature_names = set(build_features(base, open_px, high, low, amount))
        raw_names = set(latest_raw_metrics(base, open_px, high, low, amount, dates[-1]).columns)

        self.assertFalse(feature_names & raw_names)


if __name__ == "__main__":
    unittest.main()
