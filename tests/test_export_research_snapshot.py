import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from export_research_snapshot import (
    build_research_snapshot,
    discover_latest_sources,
    write_research_snapshot,
)


class ResearchSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_sources(
        self,
        token: str = "20260714",
        *,
        symbols: tuple[str, ...] = ("603010", "603317"),
        card_symbols: tuple[str, ...] | None = None,
    ) -> tuple[Path, Path]:
        asof_date = f"{token[:4]}-{token[4:6]}-{token[6:]}"
        card_symbols = card_symbols or symbols
        run_card = {
            "schema_version": 1,
            "generated_at": f"{asof_date}T22:00:13",
            "asof_date": asof_date,
            "run_status": "success",
            "project_root": r"C:\private\lianghua",
            "commands": ["python secret_pipeline.py --target-weight 0.75"],
            "argv": ["--private-input", r"D:\secret.csv"],
            "artifacts": {"database": r"D:\private\research.sqlite3"},
            "verification": {
                "tests": "534 passed; 563 subtests passed",
                "priority_rows": len(symbols),
                "priority_bucket_counts": {"model_focus": len(symbols)},
                "priority_strategy_family_counts": {"trend_momentum": len(symbols)},
                "selected_rows": 2,
                "change_rows": 1,
                "model_decision_rows": 30,
                "early_pattern_rows": 9,
                "missing_stock_names": 0,
                "research_database_sync_status": "success",
                "research_database_latest_date": asof_date,
                "research_database_asof_rows": 4012,
                "research_database_daily_rows": 4_825_661,
                "research_database_observation_rows": 2528,
                "benchmark_refresh_status": "already_fresh",
                "benchmark_latest_date": asof_date,
                "benchmark_source_agreement": True,
                "regime_shadow_target_leverage": 0.75,
                "factor_decay_monitor": {"path": r"D:\private\factor.json"},
            },
            "top10": [
                {
                    "symbol": symbol,
                    "stock_name": f"name-{symbol}",
                    "strategy_family": "trend_momentum",
                    "priority_bucket": "model_focus",
                }
                for symbol in card_symbols
            ],
            "warnings": [
                "artifact_missing:stability_report",
                "tests:534 passed; 563 subtests passed",
                r"unsafe:C:\private\file.txt",
            ],
        }
        run_card_path = self.root / f"daily_run_card_{token}.json"
        run_card_path.write_text(
            json.dumps(run_card, ensure_ascii=False), encoding="utf-8"
        )

        watchlist_path = self.root / f"merged_priority_watchlist_{token}.csv"
        fieldnames = [
            "symbol",
            "stock_name",
            "strategy_family",
            "strategy_family_cn",
            "priority_bucket",
            "priority_score",
            "trend_state",
            "pattern_type",
            "pattern_score",
            "risk_flags",
            "family_health_status",
            "personal_target_weight",
            "personal_action",
            "shadow_account_notes",
        ]
        with watchlist_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, symbol in enumerate(symbols):
                writer.writerow(
                    {
                        "symbol": symbol,
                        "stock_name": f"name-{symbol}",
                        "strategy_family": "trend_momentum",
                        "strategy_family_cn": "趋势动量",
                        "priority_bucket": "model_focus",
                        "priority_score": str(212.325 - index),
                        "trend_state": "生命线健康",
                        "pattern_type": "",
                        "pattern_score": "",
                        "risk_flags": "strategy_family_insufficient",
                        "family_health_status": "insufficient",
                        "personal_target_weight": "0.75",
                        "personal_action": "buy_now",
                        "shadow_account_notes": "private-note",
                    }
                )
        return run_card_path, watchlist_path

    def test_builds_sanitized_snapshot_from_matching_sources(self) -> None:
        run_card, watchlist = self._write_sources()

        snapshot = build_research_snapshot(
            run_card,
            watchlist,
            published_at=datetime(2026, 7, 15, 3, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(snapshot["schema_version"], 1)
        self.assertEqual(snapshot["asof_date"], "2026-07-14")
        self.assertEqual(snapshot["freshness"]["status"], "fresh")
        self.assertEqual(snapshot["freshness"]["age_days"], 1)
        self.assertEqual(snapshot["summary"]["priority_rows"], 2)
        self.assertEqual(snapshot["coverage"]["database_daily_rows"], 4_825_661)
        self.assertEqual(snapshot["watchlist"]["top10"][0]["symbol"], "603010")
        self.assertEqual(snapshot["watchlist"]["top10"][0]["priority_score"], 212.325)
        self.assertTrue(snapshot["research_only"])
        self.assertFalse(snapshot["trade_instruction"])

        serialized = json.dumps(snapshot, ensure_ascii=False)
        for forbidden in (
            r"C:\private",
            r"D:\secret.csv",
            "secret_pipeline.py",
            "personal_target_weight",
            "personal_action",
            "shadow_account_notes",
            "buy_now",
            "private-note",
            "target_leverage",
            "factor.json",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_marks_old_snapshot_stale(self) -> None:
        run_card, watchlist = self._write_sources()

        snapshot = build_research_snapshot(
            run_card,
            watchlist,
            published_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
            stale_after_days=3,
        )

        self.assertEqual(snapshot["freshness"]["status"], "stale")
        self.assertIn("data_stale", snapshot["quality"]["warnings"])

    def test_rejects_top10_mismatch(self) -> None:
        run_card, watchlist = self._write_sources(card_symbols=("000001",))

        with self.assertRaisesRegex(ValueError, "top rows do not match"):
            build_research_snapshot(run_card, watchlist)

    def test_discovers_latest_complete_source_pair(self) -> None:
        old_card, old_watchlist = self._write_sources(token="20260713")
        new_card, new_watchlist = self._write_sources(token="20260714")
        stray_card = self.root / "daily_run_card_20260715.json"
        stray_card.write_text("{}", encoding="utf-8")

        discovered = discover_latest_sources(self.root)

        self.assertEqual(discovered, (new_card, new_watchlist))
        self.assertNotEqual(discovered, (old_card, old_watchlist))

    def test_writes_round_trip_json(self) -> None:
        run_card, watchlist = self._write_sources()
        snapshot = build_research_snapshot(
            run_card,
            watchlist,
            published_at=datetime(2026, 7, 15, tzinfo=timezone.utc),
        )
        output = self.root / "nested" / "research_latest.json"

        written = write_research_snapshot(snapshot, output)

        self.assertEqual(written, output.resolve())
        self.assertEqual(
            json.loads(output.read_text(encoding="utf-8")), snapshot
        )


if __name__ == "__main__":
    unittest.main()
