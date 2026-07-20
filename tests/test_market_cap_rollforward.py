from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from quant_etf_lab.market_cap import roll_forward_stock_market_cap_frame, update_stock_market_cap_cache


class MarketCapRollForwardTests(unittest.TestCase):
    def test_update_market_cap_cache_prefers_local_daily_snapshot(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily_dir = root / "daily-market-data"
            normalized_dir = daily_dir / "ths_exports" / "normalized"
            normalized_dir.mkdir(parents=True)
            ingest_dir = root / "exchange-ingest"
            scripts_dir = ingest_dir / "scripts"
            scripts_dir.mkdir(parents=True)
            (scripts_dir / "market_data_utils.py").write_text(
                "\n".join(
                    [
                        "class FetchStatus:",
                        "    status = 'ok'",
                        "",
                        "def get_latest_fetch_status(project_root=None):",
                        "    return FetchStatus()",
                        "",
                        "def get_latest_trade_date(project_root=None):",
                        "    return '2026-06-30'",
                        "",
                        "def load_snapshot_rows(**kwargs):",
                        "    return []",
                    ]
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "security_code": "000001",
                        "security_name": "One",
                        "trade_date": "2026-07-01",
                        "close_price": 12.3,
                        "change_ratio": 2.5,
                        "market_cap": 1234.0 * 100_000_000,
                        "float_market_cap": 900.0 * 100_000_000,
                        "source": "daily_market_data_csv",
                    }
                ]
            ).to_csv(normalized_dir / "ths_hs_a_share_2026-07-01.csv", index=False, encoding="utf-8")
            output_path = root / "stock_market_cap_yi.csv"

            with patch(
                "quant_etf_lab.market_cap.fetch_stock_market_cap_snapshot",
                side_effect=AssertionError("AKShare should not be called when local daily snapshot is usable"),
            ):
                frame = update_stock_market_cap_cache(
                    output_path=output_path,
                    as_of_date="2026-07-01",
                    daily_data_dir=daily_dir,
                    ingest_project_dir=ingest_dir,
                )

            self.assertTrue(output_path.exists())
            self.assertEqual(len(frame), 1)
            row = frame.iloc[0]
            self.assertEqual(row["code"], "000001")
            self.assertAlmostEqual(float(row["market_cap_yi"]), 1234.0)
            self.assertAlmostEqual(float(row["float_market_cap_yi"]), 900.0)
            self.assertEqual(row["snapshot_date"], "2026-07-01")
            self.assertEqual(row["source"], "daily_market_data_csv")

    def test_roll_forward_rebuilds_full_cache_when_snapshot_contains_market_cap(self) -> None:
        cache = pd.DataFrame(
            {
                "code": ["000001"],
                "name": ["Old One"],
                "market_cap_yi": [1000.0],
                "market_cap_yuan": [1000.0 * 100_000_000],
                "latest_price": [10.0],
                "snapshot_date": ["2026-06-17"],
            }
        )
        snapshot_rows = [
            {
                "security_code": "000001",
                "security_name": "One",
                "trade_date": "2026-06-22",
                "close_price": 12.0,
                "change_ratio": 1.5,
                "market_cap": 1200.0 * 100_000_000,
                "float_market_cap": 960.0 * 100_000_000,
                "source": "ths_hs_a_share_export",
            },
            {
                "security_code": "000002",
                "security_name": "Two",
                "trade_date": "2026-06-22",
                "close_price": 9.0,
                "change_ratio": -2.0,
                "market_cap": 900.0 * 100_000_000,
                "source": "ths_hs_a_share_export",
            },
        ]

        rolled = roll_forward_stock_market_cap_frame(
            cache,
            snapshot_rows,
            updated_at="2026-06-22T16:30:00",
        )

        self.assertEqual(set(rolled["code"]), {"000001", "000002"})
        one = rolled[rolled["code"] == "000001"].iloc[0]
        two = rolled[rolled["code"] == "000002"].iloc[0]
        self.assertAlmostEqual(float(one["market_cap_yi"]), 1200.0)
        self.assertAlmostEqual(float(one["float_market_cap_yi"]), 960.0)
        self.assertEqual(one["latest_price"], 12.0)
        self.assertEqual(one["snapshot_date"], "2026-06-22")
        self.assertEqual(one["updated_at"], "2026-06-22T16:30:00")
        self.assertEqual(one["source"], "ths_hs_a_share_export")
        self.assertAlmostEqual(float(two["market_cap_yi"]), 900.0)

    def test_roll_forward_uses_latest_snapshot_close_and_prior_share_base(self) -> None:
        cache = pd.DataFrame(
            {
                "code": ["000001", "600000"],
                "name": ["One", "Two"],
                "market_cap_yi": [1000.0, 900.0],
                "market_cap_yuan": [1000.0 * 100_000_000, 900.0 * 100_000_000],
                "float_market_cap_yi": [800.0, pd.NA],
                "float_market_cap_yuan": [800.0 * 100_000_000, pd.NA],
                "latest_price": [10.0, 9.0],
                "pct_change": [0.0, 0.0],
                "snapshot_date": ["2026-06-17", "2026-06-17"],
                "updated_at": ["2026-06-17T16:00:00", "2026-06-17T16:00:00"],
                "source": ["akshare", "akshare"],
            }
        )
        snapshot_rows = [
            {
                "security_code": "000001",
                "security_name": "One",
                "trade_date": "2026-06-18",
                "close_price": 12.0,
                "change_ratio": 1.5,
            },
            {
                "security_code": "sh600000",
                "security_name": "Two",
                "trade_date": "2026-06-18",
                "close_price": 8.1,
                "change_ratio": -2.0,
            },
        ]

        rolled = roll_forward_stock_market_cap_frame(
            cache,
            snapshot_rows,
            updated_at="2026-06-18T16:30:00",
        )

        one = rolled[rolled["code"] == "000001"].iloc[0]
        two = rolled[rolled["code"] == "600000"].iloc[0]
        self.assertAlmostEqual(float(one["market_cap_yi"]), 1200.0)
        self.assertAlmostEqual(float(one["float_market_cap_yi"]), 960.0)
        self.assertEqual(one["snapshot_date"], "2026-06-18")
        self.assertEqual(one["latest_price"], 12.0)
        self.assertEqual(one["pct_change"], 1.5)
        self.assertEqual(one["updated_at"], "2026-06-18T16:30:00")
        self.assertIn("close_rollforward", one["source"])
        self.assertAlmostEqual(float(two["market_cap_yi"]), 810.0)
        self.assertTrue(pd.isna(two["float_market_cap_yi"]))


if __name__ == "__main__":
    unittest.main()
