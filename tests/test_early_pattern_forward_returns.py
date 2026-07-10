import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_early_pattern_forward_returns import (
    HIDDEN_ACCUMULATION_PATTERN,
    attach_forward_returns,
    evaluate_pattern_forward_returns,
    scan_hidden_accumulation_history,
    summarize_forward_returns,
    write_forward_return_report,
)


class EarlyPatternForwardReturnsTest(unittest.TestCase):
    def test_forward_returns_use_next_open_after_signal(self):
        dates = pd.date_range("2026-01-01", periods=6, freq="D")
        open_px = pd.DataFrame({"000001": [10.0, 11.0, 12.0, 15.0, 18.0, 20.0]}, index=dates)
        candidates = pd.DataFrame(
            [
                {
                    "symbol": "000001",
                    "pattern_type": HIDDEN_ACCUMULATION_PATTERN,
                    "pattern_score": 1.2,
                }
            ]
        )

        out = attach_forward_returns(candidates, open_px, dates[1], horizons=(1, 3))

        self.assertEqual(out.loc[0, "entry_date"], "2026-01-03")
        self.assertEqual(out.loc[0, "exit_date_1d"], "2026-01-04")
        self.assertAlmostEqual(out.loc[0, "forward_return_1d"], 15.0 / 12.0 - 1.0)
        self.assertEqual(out.loc[0, "exit_date_3d"], "2026-01-06")
        self.assertAlmostEqual(out.loc[0, "forward_return_3d"], 20.0 / 12.0 - 1.0)

    def test_evaluate_filters_pattern_type_and_passes_asof_to_scanner(self):
        dates = pd.date_range("2026-01-01", periods=6, freq="D")
        matrix = pd.DataFrame({"000001": np.linspace(10, 12, len(dates))}, index=dates)
        seen_asof_dates: list[pd.Timestamp] = []

        def scanner(close, open_px, high, low, amount, asof_date, top_n):
            seen_asof_dates.append(pd.Timestamp(asof_date))
            return pd.DataFrame(
                [
                    {
                        "symbol": "000001",
                        "pattern_type": HIDDEN_ACCUMULATION_PATTERN,
                        "pattern_score": 1.4,
                    },
                    {"symbol": "000002", "pattern_type": "other", "pattern_score": 2.0},
                ]
            )

        out = evaluate_pattern_forward_returns(
            matrix,
            matrix,
            matrix,
            matrix,
            matrix,
            asof_dates=[dates[1]],
            horizons=(1,),
            pattern_types=(HIDDEN_ACCUMULATION_PATTERN,),
            scanner=scanner,
        )

        self.assertEqual(seen_asof_dates, [dates[1]])
        self.assertEqual(out["symbol"].tolist(), ["000001"])
        self.assertEqual(out.loc[0, "asof_date"], "2026-01-02")

    def test_summarizes_returns_by_pattern_and_horizon(self):
        samples = pd.DataFrame(
            {
                "pattern_type": [HIDDEN_ACCUMULATION_PATTERN, HIDDEN_ACCUMULATION_PATTERN, "other"],
                "pattern_score": [1.2, 1.6, 0.8],
                "forward_return_1d": [0.10, -0.05, 0.20],
                "forward_return_3d": [0.30, 0.10, np.nan],
            }
        )

        summary = summarize_forward_returns(samples, horizons=(1, 3))
        hidden_1d = summary[
            (summary["pattern_type"] == HIDDEN_ACCUMULATION_PATTERN) & (summary["horizon_days"] == 1)
        ].iloc[0]

        self.assertEqual(hidden_1d["sample_count"], 2)
        self.assertAlmostEqual(hidden_1d["avg_return"], 0.025)
        self.assertAlmostEqual(hidden_1d["win_rate"], 0.5)
        self.assertAlmostEqual(hidden_1d["avg_pattern_score"], 1.4)

    def test_writes_csv_and_markdown_report(self):
        samples = pd.DataFrame(
            {
                "asof_date": ["2026-01-02"],
                "symbol": ["000001"],
                "pattern_type": [HIDDEN_ACCUMULATION_PATTERN],
                "pattern_score": [1.2],
                "hidden_accumulation_trade_watch": [True],
                "forward_return_1d": [0.10],
            }
        )
        summary = summarize_forward_returns(samples, horizons=(1,))

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_forward_return_report(samples, summary, output_dir, "20260102")

            self.assertTrue((output_dir / "early_pattern_forward_returns_20260102.csv").exists())
            self.assertTrue((output_dir / "early_pattern_forward_summary_20260102.csv").exists())
            self.assertTrue((output_dir / "early_pattern_forward_trade_watch_20260102.csv").exists())
            self.assertIn(
                "提前观察形态历史收益检验",
                (output_dir / "early_pattern_forward_report_20260102.md").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "高质量观察子集",
                (output_dir / "early_pattern_forward_report_20260102.md").read_text(encoding="utf-8"),
            )

    def test_vectorized_hidden_history_detects_steady_accumulation(self):
        dates = pd.date_range("2026-01-01", periods=130, freq="D")
        close_values = list(np.linspace(10.0, 12.0, 110)) + [
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
        close = pd.DataFrame({"000001": close_values}, index=dates)
        open_px = close.shift(1).fillna(close.iloc[0])
        high = close * 1.025
        low = close * 0.975
        amount = pd.DataFrame({"000001": np.linspace(80_000_000, 130_000_000, len(dates))}, index=dates)
        amount.iloc[-20:-10, 0] = 80_000_000
        amount.iloc[-10:, 0] = 112_000_000

        out = scan_hidden_accumulation_history(
            close,
            open_px,
            high,
            low,
            amount,
            asof_dates=[dates[-1]],
        )

        self.assertEqual(out.loc[0, "symbol"], "000001")
        self.assertEqual(out.loc[0, "pattern_type"], HIDDEN_ACCUMULATION_PATTERN)
        self.assertLessEqual(out.loc[0, "pattern_score"], 1.85)
        self.assertTrue(bool(out.loc[0, "hidden_accumulation_trade_watch"]))


if __name__ == "__main__":
    unittest.main()
