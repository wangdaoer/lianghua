import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from track_hidden_accumulation_watch import (
    build_tracking_table,
    load_high_quality_watchlists,
    write_tracking_outputs,
)


class HiddenAccumulationTrackingTest(unittest.TestCase):
    def test_tracking_table_marks_unfinished_horizons_pending(self):
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        open_px = pd.DataFrame({"000001": [10.0, 11.0, 12.0, 15.0, 18.0]}, index=dates)
        watch = pd.DataFrame(
            [
                {
                    "watch_date": "2026-01-02",
                    "symbol": "000001",
                    "stock_name": "Alpha",
                    "pattern_type": "隐性吸筹观察",
                    "hidden_accumulation_trade_watch": True,
                }
            ]
        )

        out = build_tracking_table(watch, open_px, horizons=(1, 3))

        self.assertEqual(out.loc[0, "entry_date"], "2026-01-03")
        self.assertAlmostEqual(out.loc[0, "forward_return_1d"], 15.0 / 12.0 - 1.0)
        self.assertTrue(np.isnan(out.loc[0, "forward_return_3d"]))
        self.assertEqual(out.loc[0, "completed_horizons"], "1")
        self.assertEqual(out.loc[0, "pending_horizons"], "3")
        self.assertEqual(out.loc[0, "tracking_status"], "partial")

    def test_tracking_table_marks_complete_when_all_horizons_available(self):
        dates = pd.date_range("2026-01-01", periods=6, freq="D")
        open_px = pd.DataFrame({"000001": [10.0, 11.0, 12.0, 15.0, 18.0, 20.0]}, index=dates)
        watch = pd.DataFrame(
            [
                {
                    "watch_date": "2026-01-02",
                    "symbol": "000001",
                    "stock_name": "Alpha",
                    "pattern_type": "隐性吸筹观察",
                    "hidden_accumulation_trade_watch": True,
                }
            ]
        )

        out = build_tracking_table(watch, open_px, horizons=(1, 3))

        self.assertAlmostEqual(out.loc[0, "forward_return_3d"], 20.0 / 12.0 - 1.0)
        self.assertEqual(out.loc[0, "completed_horizons"], "1,3")
        self.assertEqual(out.loc[0, "pending_horizons"], "")
        self.assertEqual(out.loc[0, "tracking_status"], "complete")

    def test_load_high_quality_watchlists_filters_and_uses_date_from_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "early_pattern_watchlist_20260708.csv"
            pd.DataFrame(
                [
                    {
                        "symbol": "000001",
                        "stock_name": "Alpha",
                        "hidden_accumulation_trade_watch": True,
                    },
                    {
                        "symbol": "000002",
                        "stock_name": "Beta",
                        "hidden_accumulation_trade_watch": False,
                    },
                ]
            ).to_csv(path, index=False, encoding="utf-8-sig")

            out = load_high_quality_watchlists([path])

            self.assertEqual(out["symbol"].tolist(), ["000001"])
            self.assertEqual(out.loc[0, "watch_date"], "2026-07-08")

    def test_load_high_quality_watchlists_handles_gb18030_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "early_pattern_watchlist_20260709.csv"
            content = (
                "symbol,stock_name,hidden_accumulation_trade_watch\n"
                "000001,测试股份,True\n"
            )
            path.write_text(content, encoding="gb18030")

            out = load_high_quality_watchlists([path])

            self.assertEqual(out.loc[0, "stock_name"], "测试股份")

    def test_write_tracking_outputs_creates_csv_cn_and_markdown(self):
        table = pd.DataFrame(
            [
                {
                    "watch_date": "2026-07-08",
                    "symbol": "000001",
                    "stock_name": "Alpha",
                    "tracking_status": "pending_entry",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "hidden_accumulation_trade_watch_tracking_20260708.csv"

            write_tracking_outputs(table, output)

            self.assertTrue(output.exists())
            self.assertTrue(output.with_name("hidden_accumulation_trade_watch_tracking_20260708_cn.csv").exists())
            report = output.with_suffix(".md")
            self.assertTrue(report.exists())
            self.assertIn("高质量隐性吸筹跟踪", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
