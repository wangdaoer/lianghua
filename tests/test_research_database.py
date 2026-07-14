import sqlite3
import struct
from zipfile import ZipFile

import pandas as pd
import pytest

from research_database import (
    ResearchDatabase,
    classify_tdx_asset_type,
    is_a_share_stock_symbol,
    normalize_a_share_symbol,
    normalize_a_share_symbols,
)
from tdx_day_source import TdxArchiveSpec, parse_tdx_day_bytes, read_tdx_day_archive


def test_import_prices_is_idempotent_and_queryable(tmp_path):
    db = ResearchDatabase(tmp_path / "research.sqlite3")
    prices = pd.DataFrame(
        {
            "symbol": ["000001", "000001"],
            "date": ["2026-07-10", "2026-07-11"],
            "open": [10.0, 10.2],
            "high": [10.5, 10.4],
            "low": [9.8, 10.0],
            "close": [10.2, 10.1],
            "volume": [100, 120],
            "amount": [1000, 1200],
        }
    )
    assert db.import_prices(prices) == 2
    assert db.import_prices(prices) == 0
    result = db.query("SELECT symbol, date, close FROM daily_prices ORDER BY date")
    assert result.to_dict("records") == [
        {"symbol": "000001", "date": "2026-07-10", "close": 10.2},
        {"symbol": "000001", "date": "2026-07-11", "close": 10.1},
    ]


def test_normalize_a_share_symbols_preserves_leading_zero_codes():
    values = pd.Series([1, 1.0, "000001", "000001.SZ", "SZ000001", "300001", None, "not-a-code", "1.5"])
    assert normalize_a_share_symbols(values).tolist()[:6] == [
        "000001", "000001", "000001", "000001", "000001", "300001"
    ]
    assert normalize_a_share_symbols(values).isna().tolist()[6:] == [True, True, True]


def test_normalize_a_share_symbol_matches_batch_contract():
    assert normalize_a_share_symbol(1.0) == "000001"
    assert normalize_a_share_symbol("SZ000001") == "000001"
    assert normalize_a_share_symbol("000001.SZ") == "000001"
    assert normalize_a_share_symbol("1.5") is None


def test_import_prices_accepts_numeric_symbols_and_zero_pads_them(tmp_path):
    db = ResearchDatabase(tmp_path / "research.sqlite3")
    prices = pd.DataFrame({
        "symbol": [1], "date": ["2026-07-13"], "open": [10], "high": [11],
        "low": [9], "close": [10.5], "volume": [100], "amount": [1000],
    })
    assert db.import_prices(prices) == 1
    assert db.query("SELECT symbol FROM daily_prices").iloc[0]["symbol"] == "000001"


def test_import_prices_can_update_corrected_daily_values(tmp_path):
    db = ResearchDatabase(tmp_path / "research.sqlite3")
    prices = pd.DataFrame({
        "symbol": ["000001"], "date": ["2026-07-13"], "open": [10], "high": [11],
        "low": [9], "close": [10.5], "volume": [100], "amount": [1000],
    })
    assert db.import_prices(prices, "first", update_existing=True) == 1
    corrected = prices.assign(close=10.8)
    assert db.import_prices(corrected, "corrected", update_existing=True) == 1
    assert db.import_prices(corrected, "corrected", update_existing=True) == 0
    assert db.query("SELECT close, source FROM daily_prices").to_dict("records") == [
        {"close": 10.8, "source": "corrected"}
    ]


def test_database_connections_release_file_handles(tmp_path):
    path = tmp_path / "research.sqlite3"
    db = ResearchDatabase(path)
    db.query("SELECT 1 AS value")
    path.unlink()
    assert not path.exists()


def test_import_observations_preserves_source_and_payload(tmp_path):
    db = ResearchDatabase(tmp_path / "research.sqlite3")
    observations = pd.DataFrame(
        {"symbol": ["600000"], "date": ["2026-07-11"], "score": [0.8]}
    )
    assert db.import_observations(observations, "candidate", "daily_candidates.csv") == 1
    row = db.query("SELECT source, kind, payload FROM observations").iloc[0]
    assert row["source"] == "daily_candidates.csv"
    assert row["kind"] == "candidate"
    assert '"score":0.8' in row["payload"]


def test_import_observations_normalizes_numeric_symbols(tmp_path):
    db = ResearchDatabase(tmp_path / "research.sqlite3")
    observations = pd.DataFrame({"symbol": [1.0], "date": ["2026-07-13"], "score": [0.8]})
    assert db.import_observations(observations, "candidate", "daily.csv") == 1
    assert db.query("SELECT symbol FROM observations").iloc[0]["symbol"] == "000001"


def test_import_observations_without_symbol_is_idempotent(tmp_path):
    db = ResearchDatabase(tmp_path / "research.sqlite3")
    observations = pd.DataFrame({"date": ["2026-07-13"], "score": [0.8]})
    assert db.import_observations(observations, "portfolio", "daily.csv") == 1
    assert db.import_observations(observations, "portfolio", "daily.csv") == 0


def test_import_observations_updates_changed_payload(tmp_path):
    db = ResearchDatabase(tmp_path / "research.sqlite3")
    first = pd.DataFrame({"symbol": ["000001"], "date": ["2026-07-13"], "score": [0.8]})
    corrected = first.assign(score=0.9)
    assert db.import_observations(first, "candidate", "daily.csv") == 1
    assert db.import_observations(corrected, "candidate", "daily.csv") == 1
    assert db.import_observations(corrected, "candidate", "daily.csv") == 0
    assert '"score":0.9' in db.query("SELECT payload FROM observations").iloc[0]["payload"]


def test_import_observations_preserves_multiple_rows_for_same_symbol_and_date(tmp_path):
    db = ResearchDatabase(tmp_path / "research.sqlite3")
    observations = pd.DataFrame({
        "symbol": ["000001", "000001"],
        "date": ["2026-07-13", "2026-07-13"],
        "horizon": [1, 5],
        "return": [0.01, 0.05],
    })
    assert db.import_observations(observations, "forward_return", "daily.csv") == 2
    assert db.import_observations(observations, "forward_return", "daily.csv") == 0
    row_keys = db.query("SELECT row_key FROM observations ORDER BY row_key")["row_key"].tolist()
    assert len(row_keys) == 2
    assert all(len(row_key) == 64 for row_key in row_keys)


def test_observation_snapshot_handles_reordering_and_deleted_rows(tmp_path):
    db = ResearchDatabase(tmp_path / "research.sqlite3")
    observations = pd.DataFrame({
        "symbol": ["000001", "000001"],
        "date": ["2026-07-13", "2026-07-13"],
        "horizon": [1, 5],
    })
    assert db.import_observations(observations, "forward", "daily.csv") == 2
    assert db.import_observations(observations.iloc[::-1], "forward", "daily.csv") == 0
    assert db.import_observations(observations.iloc[:1], "forward", "daily.csv") == 2
    assert db.query("SELECT COUNT(*) AS rows FROM observations").iloc[0]["rows"] == 1


def test_existing_observations_schema_is_migrated_without_data_loss(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                date TEXT NOT NULL,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                payload TEXT NOT NULL,
                UNIQUE(symbol, date, kind, source)
            );
            INSERT INTO observations (symbol,date,kind,source,payload)
            VALUES ('000001','2026-07-13','candidate','daily.csv','{"score": 0.8}');
            """
        )

    db = ResearchDatabase(path)

    assert "row_key" in db.query("PRAGMA table_info(observations)")["name"].tolist()
    assert db.query("SELECT symbol, row_key FROM observations").to_dict("records") == [
        {"symbol": "000001", "row_key": "00000000"}
    ]


def test_failed_observation_migration_rolls_back_original_table(tmp_path):
    path = tmp_path / "malformed.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE observations (
            id INTEGER PRIMARY KEY, symbol TEXT, date TEXT, kind TEXT, source TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO observations VALUES (1, '000001', '2026-07-13', 'candidate', 'daily.csv')"
        )

    with pytest.raises(sqlite3.OperationalError):
        ResearchDatabase(path)

    with sqlite3.connect(path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "observations" in tables
        assert "observations_v2_new" not in tables
        assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 1


def _tdx_record(date, open_px, high_px, low_px, close_px, amount, volume):
    return struct.pack(
        "<iiiiifII",
        date,
        int(open_px * 100),
        int(high_px * 100),
        int(low_px * 100),
        int(close_px * 100),
        float(amount),
        int(volume),
        0,
    )


def test_parse_tdx_day_bytes_normalizes_records():
    data = b"".join(
        [
            _tdx_record(20260710, 10.1, 10.8, 9.9, 10.5, 12345.0, 888),
            _tdx_record(20260711, 10.5, 11.0, 10.2, 10.9, 22345.0, 999),
        ]
    )
    frame = parse_tdx_day_bytes(data, symbol="000001", market="SH", asset_type="stock", source="sample")
    assert frame[["market", "symbol", "date", "open", "close", "volume"]].to_dict("records") == [
        {"market": "SH", "symbol": "000001", "date": "2026-07-10", "open": 10.1, "close": 10.5, "volume": 888.0},
        {"market": "SH", "symbol": "000001", "date": "2026-07-11", "open": 10.5, "close": 10.9, "volume": 999.0},
    ]


def test_parse_tdx_day_bytes_uses_native_cent_scale():
    data = struct.pack("<iiiiifII", 20260612, 1124, 1130, 1100, 1124, 1.0, 1, 0)
    frame = parse_tdx_day_bytes(data, symbol="000001", market="SZ", asset_type="stock", source="sample")
    assert frame.iloc[0]["close"] == 11.24


def test_parse_tdx_day_bytes_skips_invalid_calendar_dates():
    invalid = struct.pack("<iiiiifII", 20260230, 100, 100, 100, 100, 1.0, 1, 0)
    valid = struct.pack("<iiiiifII", 20260228, 100, 100, 100, 100, 1.0, 1, 0)
    frame = parse_tdx_day_bytes(invalid + valid, symbol="000001", market="SZ", asset_type="stock", source="sample")
    assert frame["date"].tolist() == ["2026-02-28"]


def test_import_tdx_prices_keeps_market_separate_and_idempotent(tmp_path):
    db = ResearchDatabase(tmp_path / "research.sqlite3")
    frame = pd.DataFrame(
        [
            {
                "market": "SH",
                "symbol": "000001",
                "date": "2026-07-10",
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "volume": 100,
                "amount": 1000,
                "asset_type": "index",
                "source": "shzsday.zip!sh000001.day",
            },
            {
                "market": "SZ",
                "symbol": "000001",
                "date": "2026-07-10",
                "open": 20.0,
                "high": 20.2,
                "low": 19.9,
                "close": 20.1,
                "volume": 200,
                "amount": 2000,
                "asset_type": "stock",
                "source": "szlday.zip!sz000001.day",
            },
        ]
    )
    assert db.import_tdx_prices(frame) == 2
    assert db.import_tdx_prices(frame) == 0
    result = db.query("SELECT market, symbol, close FROM tdx_daily_prices ORDER BY market")
    assert result.to_dict("records") == [
        {"market": "SH", "symbol": "000001", "close": 10.1},
        {"market": "SZ", "symbol": "000001", "close": 20.1},
    ]


def test_query_tdx_history_attaches_separate_database(tmp_path):
    main = ResearchDatabase(tmp_path / "research.sqlite3")
    history = ResearchDatabase(tmp_path / "tdx_history.sqlite3")
    frame = pd.DataFrame(
        [
            {
                "market": "SZ",
                "symbol": "000001",
                "date": "1991-01-02",
                "open": 10.0,
                "high": 10.2,
                "low": 9.9,
                "close": 10.1,
                "volume": 100,
                "amount": 1000,
                "asset_type": "stock",
                "source": "szlday.zip!sz000001.day",
            }
        ]
    )
    history.import_tdx_prices(frame)

    result = main.query_tdx_history(
        "SELECT market, symbol, date, close FROM tdx_history.tdx_daily_prices WHERE symbol = ?",
        tmp_path / "tdx_history.sqlite3",
        params=("000001",),
    )

    assert result.to_dict("records") == [
        {"market": "SZ", "symbol": "000001", "date": "1991-01-02", "close": 10.1}
    ]
    assert main.query("SELECT COUNT(*) AS rows FROM tdx_daily_prices").iloc[0]["rows"] == 0


def test_classify_tdx_asset_type_corrects_common_index_codes():
    assert classify_tdx_asset_type("SH", "000001", "stock") == "index"
    assert classify_tdx_asset_type("SH", "000300", "stock") == "index"
    assert classify_tdx_asset_type("SZ", "399001", "stock") == "index"
    assert classify_tdx_asset_type("SZ", "399006", "stock") == "index"
    assert classify_tdx_asset_type("SH", "600000", "stock") == "stock"
    assert classify_tdx_asset_type("SZ", "000001", "stock") == "stock"


def test_is_a_share_stock_symbol_respects_market_prefixes():
    assert is_a_share_stock_symbol("SH", "600000")
    assert is_a_share_stock_symbol("SZ", "000001")
    assert not is_a_share_stock_symbol("SH", "000004")
    assert not is_a_share_stock_symbol("SZ", "600000")


def test_normalized_stock_query_filters_misclassified_cross_market_codes(tmp_path):
    main = ResearchDatabase(tmp_path / "research.sqlite3")
    history = ResearchDatabase(tmp_path / "tdx_history.sqlite3")
    rows = []
    for market, close in [("SH", 99.0), ("SZ", 10.0)]:
        rows.append(
            {
                "market": market,
                "symbol": "000004",
                "date": "2025-01-02",
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1,
                "amount": 1,
                "asset_type": "stock",
                "source": f"{market.lower()}lday.zip!{market.lower()}000004.day",
            }
        )
    history.import_tdx_prices(pd.DataFrame(rows))

    result = main.query_tdx_history_normalized(
        tmp_path / "tdx_history.sqlite3",
        symbols=["000004"],
        asset_type="stock",
    )

    assert result[["market", "symbol", "close"]].to_dict("records") == [
        {"market": "SZ", "symbol": "000004", "close": 10.0}
    ]


def test_query_tdx_history_normalized_exposes_clean_asset_type(tmp_path):
    main = ResearchDatabase(tmp_path / "research.sqlite3")
    history = ResearchDatabase(tmp_path / "tdx_history.sqlite3")
    history.import_tdx_prices(
        pd.DataFrame(
            [
                {
                    "market": "SH",
                    "symbol": "000001",
                    "date": "1990-12-19",
                    "open": 96.0,
                    "high": 99.0,
                    "low": 95.0,
                    "close": 99.0,
                    "volume": 1,
                    "amount": 1,
                    "asset_type": "stock",
                    "source": "shlday.zip!sh000001.day",
                }
            ]
        )
    )

    result = main.query_tdx_history_normalized(
        tmp_path / "tdx_history.sqlite3",
        symbol="000001",
        market="SH",
    )

    assert result[["market", "symbol", "asset_type", "raw_asset_type"]].to_dict("records") == [
        {"market": "SH", "symbol": "000001", "asset_type": "index", "raw_asset_type": "stock"}
    ]


def test_query_tdx_history_normalized_prefers_raw_index_duplicates(tmp_path):
    main = ResearchDatabase(tmp_path / "research.sqlite3")
    history = ResearchDatabase(tmp_path / "tdx_history.sqlite3")
    rows = []
    for raw_asset_type, close in [("stock", 99.0), ("index", 100.0)]:
        rows.append(
            {
                "market": "SH",
                "symbol": "000001",
                "date": "1990-12-19",
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1,
                "amount": 1,
                "asset_type": raw_asset_type,
                "source": f"{raw_asset_type}.zip!sh000001.day",
            }
        )
    history.import_tdx_prices(pd.DataFrame(rows))

    result = main.query_tdx_history_normalized(tmp_path / "tdx_history.sqlite3", symbol="000001", market="SH")

    assert result[["asset_type", "raw_asset_type", "close"]].to_dict("records") == [
        {"asset_type": "index", "raw_asset_type": "index", "close": 100.0}
    ]


def test_tdx_imported_file_marker_is_queryable(tmp_path):
    db = ResearchDatabase(tmp_path / "research.sqlite3")
    assert not db.has_tdx_imported_file("archive.zip", "sh600000.day")
    db.mark_tdx_imported_file("archive.zip", "sh600000.day", 10)
    assert db.has_tdx_imported_file("archive.zip", "sh600000.day")


def test_import_tdx_member_frames_marks_files_and_skips_on_rerun(tmp_path):
    db = ResearchDatabase(tmp_path / "research.sqlite3")
    frame = pd.DataFrame(
        [
            {
                "market": "SH",
                "symbol": "600000",
                "date": "2026-07-10",
                "open": 7.0,
                "high": 7.3,
                "low": 6.9,
                "close": 7.2,
                "volume": 100,
                "amount": 1000,
                "asset_type": "stock",
                "source": "archive.zip!sh600000.day",
            }
        ]
    )
    result = db.import_tdx_member_frames("archive.zip", [("sh600000.day", frame)])
    assert result == {"read_rows": 1, "inserted_rows": 1, "skipped_files": 0, "imported_files": 1}
    second = db.import_tdx_member_frames("archive.zip", [("sh600000.day", frame)])
    assert second == {"read_rows": 0, "inserted_rows": 0, "skipped_files": 1, "imported_files": 0}


def test_import_tdx_member_frames_resumes_partially_imported_archive(tmp_path):
    db = ResearchDatabase(tmp_path / "research.sqlite3")
    first = pd.DataFrame([{
        "market": "SH", "symbol": "600000", "date": "2026-07-10",
        "open": 7.0, "high": 7.3, "low": 6.9, "close": 7.2,
        "volume": 100, "amount": 1000, "asset_type": "stock", "source": "archive.zip!sh600000.day",
    }])
    second = pd.DataFrame([{
        "market": "SH", "symbol": "600001", "date": "2026-07-10",
        "open": 8.0, "high": 8.3, "low": 7.9, "close": 8.2,
        "volume": 200, "amount": 2000, "asset_type": "stock", "source": "archive.zip!sh600001.day",
    }])
    db.import_tdx_member_frames("archive.zip", [("sh600000.day", first)])

    result = db.import_tdx_member_frames(
        "archive.zip",
        [("sh600000.day", first), ("sh600001.day", second)],
    )

    assert result == {"read_rows": 1, "inserted_rows": 1, "skipped_files": 1, "imported_files": 1}
    assert db.query("SELECT symbol FROM tdx_daily_prices ORDER BY symbol")["symbol"].tolist() == ["600000", "600001"]


def test_read_tdx_day_archive_from_zip(tmp_path):
    archive_path = tmp_path / "shlday.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("sh600000.day", _tdx_record(20260710, 7.1, 7.3, 7.0, 7.2, 1000.0, 300))
    frame = read_tdx_day_archive(TdxArchiveSpec(archive_path, market="SH", asset_type="stock"))
    assert frame[["market", "symbol", "date", "close", "asset_type"]].to_dict("records") == [
        {"market": "SH", "symbol": "600000", "date": "2026-07-10", "close": 7.2, "asset_type": "stock"}
    ]


def test_read_tdx_day_archive_respects_symbol_filter(tmp_path):
    archive_path = tmp_path / "shlday.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("sh600000.day", _tdx_record(20260710, 7.1, 7.3, 7.0, 7.2, 1000.0, 300))
        archive.writestr("sh600001.day", _tdx_record(20260710, 8.1, 8.3, 8.0, 8.2, 2000.0, 400))
    frame = read_tdx_day_archive(
        TdxArchiveSpec(archive_path, market="SH", asset_type="stock"),
        symbol_filter={"600001"},
    )
    assert frame["symbol"].tolist() == ["600001"]
