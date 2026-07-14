import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_trend_ignition_lifelines import (
    choose_lifeline,
    find_ignition_candidates,
    load_tdx_history_prices,
    parse_args,
    parse_tdx_symbols,
    summarize_trends_for_symbol,
    to_chinese_columns,
)
from research_database import ResearchDatabase


class TrendIgnitionLifelineTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

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

    def test_ignition_features_do_not_change_when_future_prices_change(self):
        dates = pd.date_range("2025-01-01", periods=180, freq="D")
        close = pd.Series(np.linspace(10.0, 12.0, len(dates)), index=dates)
        high = close * 1.01
        amount = pd.Series(10_000_000.0, index=dates)
        close.iloc[90] = high.iloc[:90].max() * 1.03
        high.iloc[90] = close.iloc[90] * 1.01
        amount.iloc[90] = 30_000_000.0

        original = find_ignition_candidates(close, high, amount)
        changed_close = close.copy()
        changed_high = high.copy()
        changed_amount = amount.copy()
        changed_close.iloc[91:] *= 5.0
        changed_high.iloc[91:] *= 5.0
        changed_amount.iloc[91:] *= 10.0
        changed = find_ignition_candidates(changed_close, changed_high, changed_amount)

        feature_columns = [
            "breakout_pct",
            "amount_ratio",
            "return_20d",
            "return_60d",
            "volatility_20d",
            "ma20_over_ma60",
            "close_over_ma20",
            "drawdown_120d",
            "amount_trend_5_20",
            "breakout_count_20d",
        ]
        pd.testing.assert_series_equal(
            original.loc[dates[90], feature_columns],
            changed.loc[dates[90], feature_columns],
        )

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

    def test_summarize_trends_can_keep_below_threshold_training_controls(self):
        dates = pd.date_range("2025-01-01", periods=220, freq="D")
        close = pd.Series(10.0, index=dates)
        high = close * 1.01
        amount = pd.Series(10_000_000.0, index=dates)
        close.iloc[100:] = np.linspace(11.5, 15.0, len(dates) - 100)
        high.iloc[100:] = close.iloc[100:] * 1.01
        amount.iloc[100] = 30_000_000.0

        samples = summarize_trends_for_symbol(
            symbol="000001",
            close=close,
            high=high,
            amount=amount,
            start=pd.Timestamp("2025-06-01"),
            end=pd.Timestamp("2025-08-08"),
            min_trend_return=0.80,
            include_below_threshold=True,
        )

        self.assertEqual(len(samples), 1)
        self.assertLess(samples[0]["peak_return"], 0.80)
        self.assertEqual(samples[0]["analysis_end_date"], "2025-08-08")

    def test_training_sample_cooldown_does_not_depend_on_future_peak_date(self):
        dates = pd.date_range("2025-01-01", periods=250, freq="D")
        close = pd.Series(10.0, index=dates)
        high = close * 1.005
        amount = pd.Series(10_000_000.0, index=dates)
        close.iloc[100:] = 12.0
        close.iloc[125:] = np.linspace(14.0, 30.0, len(dates) - 125)
        high.iloc[100:] = close.iloc[100:] * 1.005
        amount.iloc[100] = 30_000_000.0
        amount.iloc[125] = 30_000_000.0

        samples = summarize_trends_for_symbol(
            symbol="000001",
            close=close,
            high=high,
            amount=amount,
            start=dates[80],
            end=dates[-1],
            min_trend_return=0.0,
            include_below_threshold=True,
            ignition_cooldown_bars=20,
        )

        ignition_dates = {row["ignition_date"] for row in samples}
        self.assertIn(dates[100].strftime("%Y-%m-%d"), ignition_dates)
        self.assertIn(dates[125].strftime("%Y-%m-%d"), ignition_dates)

    def test_training_mode_excludes_prestart_ignitions_and_caps_outcome_horizon(self):
        dates = pd.date_range("2024-01-01", periods=800, freq="D")
        close = pd.Series(10.0, index=dates)
        high = close * 1.01
        amount = pd.Series(10_000_000.0, index=dates)
        for index in (180, 430):
            close.iloc[index:] += np.linspace(1.5, 8.0, len(dates) - index)
            high.iloc[index:] = close.iloc[index:] * 1.01
            amount.iloc[index] = 30_000_000.0

        samples = summarize_trends_for_symbol(
            symbol="000001",
            close=close,
            high=high,
            amount=amount,
            start=pd.Timestamp("2025-01-01"),
            end=pd.Timestamp("2026-01-31"),
            min_trend_return=0.0,
            include_below_threshold=True,
            include_prestart_ignitions=False,
            outcome_horizon_days=365,
        )

        self.assertTrue(samples)
        self.assertTrue(all(pd.Timestamp(row["ignition_date"]) >= pd.Timestamp("2025-01-01") for row in samples))
        self.assertTrue(all((pd.Timestamp(row["outcome_end_date"]) - pd.Timestamp(row["ignition_date"])).days <= 365 for row in samples))

    def test_load_tdx_history_prices_returns_stock_history_in_panel_shape(self):
        main = ResearchDatabase(self.tmp_path / "research.sqlite3")
        history = ResearchDatabase(self.tmp_path / "tdx_history.sqlite3")
        history.import_tdx_prices(
            pd.DataFrame(
                [
                    {
                        "market": "SZ",
                        "symbol": "000001",
                        "date": "1991-04-03",
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.8,
                        "close": 10.2,
                        "volume": 100,
                        "amount": 1000,
                        "asset_type": "stock",
                        "source": "szlday.zip!sz000001.day",
                    },
                    {
                        "market": "SH",
                        "symbol": "000001",
                        "date": "1991-04-03",
                        "open": 99.0,
                        "high": 100.0,
                        "low": 98.0,
                        "close": 99.5,
                        "volume": 1,
                        "amount": 1,
                        "asset_type": "stock",
                        "source": "shlday.zip!sh000001.day",
                    },
                ]
            )
        )

        prices = load_tdx_history_prices(
            main.path,
            self.tmp_path / "tdx_history.sqlite3",
            symbols=["000001"],
            start="1991-01-01",
            end="1991-12-31",
        )

        self.assertEqual(prices["symbol"].tolist(), ["000001"])
        self.assertEqual(prices["close"].tolist(), [10.2])
        self.assertEqual(list(prices.columns), [
            "date", "symbol", "open", "high", "low", "close", "volume", "amount"
        ])

    def test_parse_args_allows_tdx_history_without_data_path(self):
        args = parse_args(["--use-tdx-history", "--tdx-symbols", "000001"])

        self.assertTrue(args.use_tdx_history)
        self.assertEqual(args.tdx_symbols, "000001")

    def test_parse_tdx_symbols_combines_cli_and_file(self):
        symbols_file = self.tmp_path / "symbols.txt"
        symbols_file.write_text("SZ000002\n000001\n", encoding="utf-8")

        symbols = parse_tdx_symbols("300001,600000.SH,000001", symbols_file)

        self.assertEqual(symbols, ["000001", "000002", "300001", "600000"])

    def test_parse_args_requires_data_outside_tdx_mode(self):
        with self.assertRaises(SystemExit):
            parse_args([])

    def test_chinese_report_translates_point_in_time_features(self):
        translated = to_chinese_columns(
            pd.DataFrame({"return_20d": [0.1], "close_over_ma20": [0.05]})
        )

        self.assertEqual(list(translated.columns), ["近20日收益", "收盘价相对MA20偏离"])


if __name__ == "__main__":
    unittest.main()
