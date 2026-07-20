from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd


FAKE_UTILS = """
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class FetchStatus:
    run_id: str | None
    source: str | None
    market_filter: str | None
    row_count: int
    status: str
    message: str | None
    run_time: str | None

def ensure_latest_fetch_ok(db_path=None, project_root=None):
    Path(project_root, "status_checked.txt").write_text("checked", encoding="utf-8")

def get_latest_fetch_status(db_path=None, project_root=None):
    return FetchStatus("run-1", "fake", "all", 1, "ok", None, "2026-06-15T15:30:00")

def get_latest_trade_date(db_path=None, project_root=None):
    return "2026-06-15"

def load_snapshot_rows(trade_date=None, market="all", db_path=None, project_root=None, fallback_to_csv=True):
    return [
        {
            "market": "szse",
            "trade_date": trade_date or "2026-06-15",
            "security_code": "000002",
            "security_name": "Fallback",
            "close_price": "20.0",
            "change_ratio": "2.0",
            "source": "fake_ingest",
        }
    ]
"""


def _write_fake_ingest(root: Path) -> None:
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "market_data_utils.py").write_text(FAKE_UTILS, encoding="utf-8")


class MarketDataSourceTests(unittest.TestCase):
    def test_load_market_snapshot_prefers_daily_hub(self) -> None:
        try:
            from quant_etf_lab.market_data_source import load_market_snapshot_rows
        except ImportError as exc:
            self.fail(f"market_data_source module should exist: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily = root / "daily-market-data"
            ingest = root / "exchange-ingest"
            (daily / "snapshots").mkdir(parents=True)
            _write_fake_ingest(ingest)
            pd.DataFrame(
                [
                    {
                        "market": "sse",
                        "trade_date": "2026-06-15",
                        "security_code": "000001",
                        "security_name": "Daily",
                        "close_price": 10.0,
                        "change_ratio": 1.0,
                        "source": "daily_hub",
                    }
                ]
            ).to_csv(daily / "snapshots" / "2026-06-15_market_snapshot.csv", index=False)

            result = load_market_snapshot_rows(daily_data_dir=daily, ingest_project_dir=ingest)

            self.assertEqual(result.source_kind, "daily_market_data_csv")
            self.assertEqual(result.trade_date, "2026-06-15")
            self.assertEqual(result.rows[0]["security_code"], "000001")

    def test_load_market_snapshot_prefers_ths_normalized_exports(self) -> None:
        try:
            from quant_etf_lab.market_data_source import load_market_snapshot_rows
        except ImportError as exc:
            self.fail(f"market_data_source module should exist: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily = root / "daily-market-data"
            ingest = root / "exchange-ingest"
            (daily / "ths_exports" / "normalized").mkdir(parents=True)
            ths_path = daily / "ths_exports" / "normalized" / "ths_hs_a_share_2026-06-29.csv"
            _write_fake_ingest(ingest)
            pd.DataFrame(
                [
                    {
                        "market": "sse",
                        "trade_date": "2026-06-29",
                        "security_code": "688548",
                        "security_name": "THS",
                        "close_price": 46.49,
                        "change_ratio": 20.01,
                        "source": "ths_hs_a_share_export",
                    }
                ]
            ).to_csv(ths_path, index=False)

            result = load_market_snapshot_rows(daily_data_dir=daily, ingest_project_dir=ingest)

            self.assertEqual(result.source_kind, "daily_market_data_csv")
            self.assertEqual(result.trade_date, "2026-06-29")
            self.assertEqual(result.source_path, ths_path)
            self.assertEqual(result.rows[0]["security_code"], "688548")

    def test_load_market_snapshot_accepts_simplified_ths_normalized_exports(self) -> None:
        try:
            from quant_etf_lab.market_data_source import load_market_snapshot_rows
        except ImportError as exc:
            self.fail(f"market_data_source module should exist: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily = root / "daily-market-data"
            ingest = root / "exchange-ingest"
            (daily / "ths_exports" / "normalized").mkdir(parents=True)
            ths_path = daily / "ths_exports" / "normalized" / "ths_hs_a_share_2026-06-30.csv"
            _write_fake_ingest(ingest)
            pd.DataFrame(
                [
                    {
                        "date": "2026-06-30",
                        "code": "300044",
                        "name": "*ST赛为",
                        "open": 4.26,
                        "high": 5.14,
                        "low": 4.25,
                        "close": 5.14,
                        "volume": "",
                        "turnover_value": "",
                    }
                ]
            ).to_csv(ths_path, index=False)

            result = load_market_snapshot_rows(
                trade_date="2026-06-30",
                daily_data_dir=daily,
                ingest_project_dir=ingest,
            )

            self.assertEqual(result.source_kind, "daily_market_data_csv")
            self.assertEqual(result.trade_date, "2026-06-30")
            self.assertEqual(result.source_path, ths_path)
            self.assertEqual(result.rows[0]["market"], "szse")
            self.assertEqual(result.rows[0]["security_code"], "300044")
            self.assertEqual(result.rows[0]["security_name"], "*ST赛为")
            self.assertEqual(result.rows[0]["close_price"], 5.14)
            self.assertIsNone(result.rows[0]["turnover"])

    def test_load_market_snapshot_falls_back_to_exchange_ingest_when_daily_latest_missing(self) -> None:
        try:
            from quant_etf_lab.market_data_source import load_market_snapshot_rows
        except ImportError as exc:
            self.fail(f"market_data_source module should exist: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily = root / "daily-market-data"
            ingest = root / "exchange-ingest"
            daily.mkdir()
            _write_fake_ingest(ingest)

            result = load_market_snapshot_rows(daily_data_dir=daily, ingest_project_dir=ingest)

            self.assertEqual(result.source_kind, "exchange_ingest")
            self.assertEqual(result.rows[0]["security_code"], "000002")
            self.assertTrue((ingest / "status_checked.txt").exists())

    def test_load_market_snapshot_falls_back_to_exchange_ingest_when_daily_latest_is_stale(self) -> None:
        try:
            from quant_etf_lab.market_data_source import load_market_snapshot_rows
        except ImportError as exc:
            self.fail(f"market_data_source module should exist: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily = root / "daily-market-data"
            ingest = root / "exchange-ingest"
            (daily / "snapshots").mkdir(parents=True)
            _write_fake_ingest(ingest)
            pd.DataFrame(
                [
                    {
                        "market": "sse",
                        "trade_date": "2026-06-14",
                        "security_code": "600000",
                        "security_name": "DailyStale",
                        "close_price": 10.0,
                        "change_ratio": 0.5,
                        "source": "daily_hub",
                    }
                ]
            ).to_csv(daily / "snapshots" / "2026-06-14_market_snapshot.csv", index=False)

            result = load_market_snapshot_rows(
                trade_date="2026-06-15",
                daily_data_dir=daily,
                ingest_project_dir=ingest,
            )

            self.assertEqual(result.source_kind, "exchange_ingest")
            self.assertEqual(result.trade_date, "2026-06-15")
            self.assertEqual(result.rows[0]["security_code"], "000002")
            self.assertEqual(result.rows[0]["security_name"], "Fallback")
            self.assertTrue((ingest / "status_checked.txt").exists())

    def test_load_market_snapshot_falls_back_to_exchange_ingest_when_daily_data_is_stale(self) -> None:
        try:
            from quant_etf_lab.market_data_source import load_market_snapshot_rows
        except ImportError as exc:
            self.fail(f"market_data_source module should exist: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily = root / "daily-market-data"
            ingest = root / "exchange-ingest"
            (daily / "snapshots").mkdir(parents=True)
            _write_fake_ingest(ingest)
            pd.DataFrame(
                [
                    {
                        "market": "sse",
                        "trade_date": "2026-06-14",
                        "security_code": "600001",
                        "security_name": "DailyStale",
                        "close_price": 11.0,
                        "change_ratio": 0.5,
                        "source": "daily_hub",
                    }
                ]
            ).to_csv(daily / "snapshots" / "2026-06-14_market_snapshot.csv", index=False)

            result = load_market_snapshot_rows(daily_data_dir=daily, ingest_project_dir=ingest)

            self.assertEqual(result.source_kind, "exchange_ingest")
            self.assertEqual(result.trade_date, "2026-06-15")
            self.assertEqual(result.rows[0]["security_code"], "000002")
            self.assertTrue((ingest / "status_checked.txt").exists())

    def test_load_market_snapshot_uses_daily_latest_when_exchange_is_stale(self) -> None:
        try:
            from quant_etf_lab.market_data_source import load_market_snapshot_rows
        except ImportError as exc:
            self.fail(f"market_data_source module should exist: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily = root / "daily-market-data"
            ingest = root / "exchange-ingest"
            (daily / "snapshots").mkdir(parents=True)
            _write_fake_ingest(ingest)

            pd.DataFrame(
                [
                    {
                        "market": "sse",
                        "trade_date": "2026-06-16",
                        "security_code": "688888",
                        "security_name": "DailyFuture",
                        "close_price": 50.0,
                        "change_ratio": 1.5,
                        "source": "daily_hub",
                    }
                ]
            ).to_csv(daily / "snapshots" / "2026-06-16_market_snapshot.csv", index=False)

            result = load_market_snapshot_rows(daily_data_dir=daily, ingest_project_dir=ingest)

            self.assertEqual(result.source_kind, "daily_market_data_csv")
            self.assertEqual(result.trade_date, "2026-06-16")
            self.assertEqual(result.rows[0]["security_code"], "688888")

    def test_dashboard_market_cache_summary_uses_unified_source_fallback(self) -> None:
        from quant_etf_lab import market_data_source
        from quant_etf_lab.dashboard import _summarize_market_cache

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily = root / "daily-market-data"
            ingest = root / "exchange-ingest"
            daily.mkdir()
            _write_fake_ingest(ingest)
            original_ingest = market_data_source.DEFAULT_EXCHANGE_INGEST_DIR
            try:
                market_data_source.DEFAULT_EXCHANGE_INGEST_DIR = ingest
                summary = _summarize_market_cache(
                    project_root=root,
                    data_cache_dir=daily,
                    as_of=date(2026, 6, 15),
                    max_staleness_days=0,
                    min_cache_fresh_ratio=0.90,
                )
            finally:
                market_data_source.DEFAULT_EXCHANGE_INGEST_DIR = original_ingest

            self.assertEqual(summary["market_cache_status"], "fresh_enough")
            self.assertEqual(summary["market_cache_source_kind"], "exchange_ingest")
            self.assertEqual(summary["market_cache_latest_date"], "2026-06-15")
            self.assertTrue((ingest / "status_checked.txt").exists())
