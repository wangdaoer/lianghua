from __future__ import annotations

import importlib.util
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sync_daily_market_snapshot_to_cache.py"


def load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_daily_market_snapshot_to_cache", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_cache(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=["date", "code", "name", "open", "high", "low", "close", "volume", "amount"],
    ).to_csv(path, index=False, encoding="utf-8")


def test_snapshot_row_to_quote_normalizes_prefixed_code_and_prices() -> None:
    module = load_sync_module()

    quote = module.snapshot_row_to_quote(
        {
            "security_code": "sz000001",
            "security_name": "Daily Name",
            "trade_date": "2026-06-17",
            "open_price": 10.1,
            "high_price": 10.5,
            "low_price": 9.9,
            "close_price": 10.2,
            "last_price": 0.0,
            "volume": 1000,
            "turnover": 10200,
        }
    )

    assert quote is not None
    assert quote.code == "000001"
    assert quote.name == "Daily Name"
    assert quote.date == "2026-06-17"
    assert quote.open == 10.1
    assert quote.high == 10.5
    assert quote.low == 9.9
    assert quote.close == 10.2
    assert quote.volume == 1000.0
    assert quote.amount == 10200.0


def test_append_snapshot_quote_to_csv_is_idempotent_and_preserves_existing_name(tmp_path: Path) -> None:
    module = load_sync_module()
    path = tmp_path / "000001.csv"
    _write_cache(
        path,
        [
            {
                "date": "2026-06-16",
                "code": "000001",
                "name": "Existing Name",
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "volume": 100.0,
                "amount": 1010.0,
            }
        ],
    )
    quote = module.SnapshotQuote(
        code="000001",
        name="Daily Name",
        date="2026-06-17",
        open=10.1,
        high=10.5,
        low=9.9,
        close=10.2,
        volume=1000.0,
        amount=10200.0,
    )

    first = module.append_snapshot_quote_to_csv(path, quote, target_date="2026-06-17")
    second = module.append_snapshot_quote_to_csv(path, quote, target_date="2026-06-17")

    frame = pd.read_csv(path)
    assert first["status"] == "appended"
    assert second["status"] == "already_present"
    assert len(frame) == 2
    assert frame.iloc[-1]["date"] == "2026-06-17"
    assert frame.iloc[-1]["name"] == "Existing Name"
    assert frame.iloc[-1]["close"] == 10.2


def test_append_snapshot_quote_to_csv_skips_stale_duplicate_quote(tmp_path: Path) -> None:
    module = load_sync_module()
    path = tmp_path / "000001.csv"
    _write_cache(
        path,
        [
            {
                "date": "2026-06-18",
                "code": "000001",
                "name": "Existing Name",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "volume": 1000.0,
                "amount": 10200.0,
            }
        ],
    )
    quote = module.SnapshotQuote(
        code="000001",
        name="Daily Name",
        date="2026-06-19",
        open=10.0,
        high=10.5,
        low=9.8,
        close=10.2,
        volume=1000.0,
        amount=10200.0,
    )

    result = module.append_snapshot_quote_to_csv(path, quote, target_date="2026-06-19")

    frame = pd.read_csv(path)
    assert result["status"] == "stale_duplicate_quote"
    assert result["previous_latest_date"] == "2026-06-18"
    assert result["latest_date"] == "2026-06-18"
    assert len(frame) == 1
    assert frame.iloc[-1]["date"] == "2026-06-18"


def test_append_snapshot_quote_to_csv_backfills_missing_middle_date(tmp_path: Path) -> None:
    module = load_sync_module()
    path = tmp_path / "000001.csv"
    _write_cache(
        path,
        [
            {
                "date": "2026-06-24",
                "code": "000001",
                "name": "Existing Name",
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "volume": 100.0,
                "amount": 1010.0,
            },
            {
                "date": "2026-06-26",
                "code": "000001",
                "name": "Existing Name",
                "open": 10.4,
                "high": 10.8,
                "low": 10.2,
                "close": 10.7,
                "volume": 120.0,
                "amount": 1284.0,
            },
        ],
    )
    quote = module.SnapshotQuote(
        code="000001",
        name="Daily Name",
        date="2026-06-25",
        open=10.2,
        high=10.6,
        low=10.1,
        close=10.5,
        volume=110.0,
        amount=1155.0,
    )

    result = module.append_snapshot_quote_to_csv(path, quote, target_date="2026-06-25")

    frame = pd.read_csv(path)
    assert result["status"] == "backfilled"
    assert result["previous_latest_date"] == "2026-06-26"
    assert list(frame["date"]) == ["2026-06-24", "2026-06-25", "2026-06-26"]
    assert frame.loc[frame["date"] == "2026-06-25", "close"].iloc[0] == 10.5


def test_append_snapshot_quotes_to_csv_appends_range_once_and_is_idempotent(
    tmp_path: Path,
) -> None:
    module = load_sync_module()
    path = tmp_path / "000001.csv"
    _write_cache(
        path,
        [
            {
                "date": "2026-07-03",
                "code": "000001",
                "name": "Existing Name",
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "volume": 100.0,
                "amount": 1010.0,
            }
        ],
    )
    quotes = [
        module.SnapshotQuote(
            code="000001",
            name="Daily Name",
            date="2026-07-06",
            open=10.1,
            high=10.5,
            low=10.0,
            close=10.4,
            volume=1000.0,
            amount=10400.0,
        ),
        module.SnapshotQuote(
            code="000001",
            name="Daily Name",
            date="2026-07-07",
            open=10.4,
            high=10.8,
            low=10.3,
            close=10.7,
            volume=1100.0,
            amount=11770.0,
        ),
    ]

    first = module.append_snapshot_quotes_to_csv(path, quotes)
    second = module.append_snapshot_quotes_to_csv(path, quotes)

    frame = pd.read_csv(path)
    assert [row["status"] for row in first] == ["appended", "appended"]
    assert [row["status"] for row in second] == ["already_present", "already_present"]
    assert list(frame["date"]) == ["2026-07-03", "2026-07-06", "2026-07-07"]
    assert set(frame["name"]) == {"Existing Name"}


def test_append_snapshot_quotes_to_csv_can_replace_existing_zero_volume(
    tmp_path: Path,
) -> None:
    module = load_sync_module()
    path = tmp_path / "000001.csv"
    _write_cache(
        path,
        [
            {
                "date": "2026-07-20",
                "code": "000001",
                "name": "Existing Name",
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "volume": 0.0,
                "amount": 0.0,
            }
        ],
    )
    quote = module.SnapshotQuote(
        code="000001",
        name="Daily Name",
        date="2026-07-20",
        open=10.0,
        high=10.2,
        low=9.9,
        close=10.1,
        volume=1234.0,
        amount=12463.4,
    )

    first = module.append_snapshot_quotes_to_csv(
        path,
        [quote],
        replace_existing=True,
    )
    second = module.append_snapshot_quotes_to_csv(
        path,
        [quote],
        replace_existing=True,
    )

    frame = pd.read_csv(path)
    assert first[0]["status"] == "replaced"
    assert second[0]["status"] == "already_present"
    assert len(frame) == 1
    assert frame.loc[0, "name"] == "Existing Name"
    assert frame.loc[0, "volume"] == 1234.0
    assert frame.loc[0, "amount"] == 12463.4


def test_refresh_from_rows_updates_processed_and_raw_mirror(tmp_path: Path) -> None:
    module = load_sync_module()
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    for root in [data_dir / "processed" / "stocks", data_dir / "raw" / "stocks"]:
        _write_cache(
            root / "000001.csv",
            [
                {
                    "date": "2026-06-16",
                    "code": "000001",
                    "name": "Existing Name",
                    "open": 10.0,
                    "high": 10.2,
                    "low": 9.9,
                    "close": 10.1,
                    "volume": 100.0,
                    "amount": 1010.0,
                }
            ],
        )

    summary = module.refresh_from_rows(
        data_dir=data_dir,
        output_dir=output_dir,
        target_date="2026-06-17",
        rows=[
            {
                "security_code": "000001",
                "security_name": "Daily Name",
                "trade_date": "2026-06-17",
                "open_price": 10.1,
                "high_price": 10.5,
                "low_price": 9.9,
                "close_price": 10.2,
                "volume": 1000,
                "turnover": 10200,
            }
        ],
        source_kind="daily_market_data_sqlite",
        source_path=Path("market.db"),
        max_symbols=None,
    )

    assert summary["statuses"] == {"appended": 2}
    assert summary["total_quotes"] == 1
    assert summary["total_target_files"] == 2
    assert (output_dir / "daily_market_snapshot_sync_audit.csv").exists()
    assert (output_dir / "daily_market_snapshot_sync_summary.json").exists()
    for cache_path in [
        data_dir / "processed" / "stocks" / "000001.csv",
        data_dir / "raw" / "stocks" / "000001.csv",
    ]:
        frame = pd.read_csv(cache_path)
        assert list(frame["date"]) == ["2026-06-16", "2026-06-17"]


def test_refresh_from_rows_dry_run_reports_without_writing(tmp_path: Path) -> None:
    module = load_sync_module()
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    for root in [data_dir / "processed" / "stocks", data_dir / "raw" / "stocks"]:
        _write_cache(
            root / "000001.csv",
            [
                {
                    "date": "2026-06-16",
                    "code": "000001",
                    "name": "Existing Name",
                    "open": 10.0,
                    "high": 10.2,
                    "low": 9.9,
                    "close": 10.1,
                    "volume": 100.0,
                    "amount": 1010.0,
                }
            ],
        )

    summary = module.refresh_from_rows(
        data_dir=data_dir,
        output_dir=output_dir,
        target_date="2026-06-17",
        rows=[
            {
                "security_code": "000001",
                "security_name": "Daily Name",
                "trade_date": "2026-06-17",
                "open_price": 10.1,
                "high_price": 10.5,
                "low_price": 9.9,
                "close_price": 10.2,
                "volume": 1000,
                "turnover": 10200,
            }
        ],
        source_kind="daily_market_data_sqlite",
        source_path=Path("market.db"),
        max_symbols=None,
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["statuses"] == {"would_append": 2}
    for cache_path in [
        data_dir / "processed" / "stocks" / "000001.csv",
        data_dir / "raw" / "stocks" / "000001.csv",
    ]:
        frame = pd.read_csv(cache_path)
        assert list(frame["date"]) == ["2026-06-16"]


def test_refresh_from_rows_blocks_intraday_write_before_after_close(tmp_path: Path) -> None:
    module = load_sync_module()
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    for root in [data_dir / "processed" / "stocks", data_dir / "raw" / "stocks"]:
        _write_cache(
            root / "000001.csv",
            [
                {
                    "date": "2026-06-19",
                    "code": "000001",
                    "name": "Existing Name",
                    "open": 10.0,
                    "high": 10.2,
                    "low": 9.9,
                    "close": 10.1,
                    "volume": 100.0,
                    "amount": 1010.0,
                }
            ],
        )

    summary = module.refresh_from_rows(
        data_dir=data_dir,
        output_dir=output_dir,
        target_date="2026-06-22",
        rows=[
            {
                "security_code": "000001",
                "security_name": "Daily Name",
                "trade_date": "2026-06-22",
                "open_price": 10.1,
                "high_price": 10.5,
                "low_price": 9.9,
                "close_price": 10.2,
                "volume": 1000,
                "turnover": 10200,
            }
        ],
        source_kind="daily_market_data_sqlite",
        source_path=Path("market.db"),
        max_symbols=None,
        current_dt=datetime(2026, 6, 22, 10, 0, 0),
    )

    assert summary["intraday_blocked"] is True
    assert summary["statuses"] == {"intraday_target_blocked": 2}
    for cache_path in [
        data_dir / "processed" / "stocks" / "000001.csv",
        data_dir / "raw" / "stocks" / "000001.csv",
    ]:
        frame = pd.read_csv(cache_path)
        assert list(frame["date"]) == ["2026-06-19"]


def test_refresh_from_rows_allows_intraday_write_when_explicitly_enabled(tmp_path: Path) -> None:
    module = load_sync_module()
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    _write_cache(
        data_dir / "processed" / "stocks" / "000001.csv",
        [
            {
                "date": "2026-06-19",
                "code": "000001",
                "name": "Existing Name",
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "volume": 100.0,
                "amount": 1010.0,
            }
        ],
    )

    summary = module.refresh_from_rows(
        data_dir=data_dir,
        output_dir=output_dir,
        target_date="2026-06-22",
        rows=[
            {
                "security_code": "000001",
                "security_name": "Daily Name",
                "trade_date": "2026-06-22",
                "open_price": 10.1,
                "high_price": 10.5,
                "low_price": 9.9,
                "close_price": 10.2,
                "volume": 1000,
                "turnover": 10200,
            }
        ],
        source_kind="daily_market_data_sqlite",
        source_path=Path("market.db"),
        max_symbols=None,
        current_dt=datetime(2026, 6, 22, 10, 0, 0),
        allow_intraday=True,
    )

    frame = pd.read_csv(data_dir / "processed" / "stocks" / "000001.csv")
    assert summary["allow_intraday"] is True
    assert summary["intraday_blocked"] is False
    assert summary["statuses"] == {"appended": 1}
    assert list(frame["date"]) == ["2026-06-19", "2026-06-22"]


def test_refresh_from_rows_blocks_today_when_source_updated_before_after_close(tmp_path: Path) -> None:
    module = load_sync_module()
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    _write_cache(
        data_dir / "processed" / "stocks" / "000001.csv",
        [
            {
                "date": "2026-06-19",
                "code": "000001",
                "name": "Existing Name",
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "volume": 100.0,
                "amount": 1010.0,
            }
        ],
    )

    summary = module.refresh_from_rows(
        data_dir=data_dir,
        output_dir=output_dir,
        target_date="2026-06-22",
        rows=[
            {
                "security_code": "000001",
                "security_name": "Daily Name",
                "trade_date": "2026-06-22",
                "open_price": 10.1,
                "high_price": 10.5,
                "low_price": 9.9,
                "close_price": 10.2,
                "volume": 1000,
                "turnover": 10200,
            }
        ],
        source_kind="daily_market_data_sqlite",
        source_path=Path("market.db"),
        max_symbols=None,
        current_dt=datetime(2026, 6, 22, 15, 45, 0),
        source_updated_at=datetime(2026, 6, 22, 9, 37, 0),
    )

    frame = pd.read_csv(data_dir / "processed" / "stocks" / "000001.csv")
    assert summary["intraday_blocked"] is True
    assert summary["intraday_block_reason"] == "source_before_after_close"
    assert summary["statuses"] == {"intraday_target_blocked": 1}
    assert list(frame["date"]) == ["2026-06-19"]


def test_script_can_be_invoked_directly_for_help() -> None:
    result = subprocess.run(
        ["python", str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Sync unified daily market snapshots" in result.stdout
    assert "--dates" in result.stdout
    assert "--replace-existing" in result.stdout


def test_script_market_choices_include_bse_for_ths_exports() -> None:
    text = SCRIPT.read_text(encoding="utf-8-sig")
    assert 'choices=["all", "sse", "szse", "bse"]' in text
