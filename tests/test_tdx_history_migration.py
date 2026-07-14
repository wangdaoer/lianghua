import pandas as pd
import pytest

from migrate_tdx_history import migrate_tdx_history, rebuild_main_without_tdx
from research_database import ResearchDatabase


def test_migrate_tdx_history_copies_rows_without_deleting_source_by_default(tmp_path):
    source = ResearchDatabase(tmp_path / "research.sqlite3")
    tdx = pd.DataFrame(
        [
            {
                "market": "SZ",
                "symbol": "000001",
                "date": "1991-01-02",
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "volume": 1000,
                "amount": 2000,
                "asset_type": "stock",
                "source": "szlday.zip!sz000001.day",
            }
        ]
    )
    source.import_tdx_member_frames("szlday.zip", [("sz000001.day", tdx)])

    result = migrate_tdx_history(source.path, tmp_path / "tdx_history.sqlite3", vacuum=False)

    assert result == {"price_rows": 1, "file_rows": 1}
    assert source.query("SELECT COUNT(*) AS rows FROM tdx_daily_prices").iloc[0]["rows"] == 1
    assert source.query("SELECT COUNT(*) AS rows FROM tdx_imported_files").iloc[0]["rows"] == 1

    target = ResearchDatabase(tmp_path / "tdx_history.sqlite3")
    assert target.query("SELECT market, symbol, close FROM tdx_daily_prices").to_dict("records") == [
        {"market": "SZ", "symbol": "000001", "close": 10.2}
    ]
    assert target.query("SELECT archive, member, rows FROM tdx_imported_files").to_dict("records") == [
        {"archive": "szlday.zip", "member": "sz000001.day", "rows": 1}
    ]


def test_migrate_tdx_history_deletes_source_only_when_explicit(tmp_path):
    source = ResearchDatabase(tmp_path / "research.sqlite3")
    source.import_tdx_prices(pd.DataFrame([{
        "market": "SZ", "symbol": "000001", "date": "1991-01-02",
        "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2,
        "volume": 1000, "amount": 2000, "asset_type": "stock", "source": "sample",
    }]))

    migrate_tdx_history(source.path, tmp_path / "tdx.sqlite3", delete_source=True, vacuum=False)

    assert source.query("SELECT COUNT(*) AS rows FROM tdx_daily_prices").iloc[0]["rows"] == 0


def test_migrate_tdx_history_replaces_stale_target_row(tmp_path):
    source = ResearchDatabase(tmp_path / "research.sqlite3")
    target = ResearchDatabase(tmp_path / "tdx.sqlite3")
    row = {
        "market": "SZ", "symbol": "000001", "date": "1991-01-02",
        "open": 10.0, "high": 10.5, "low": 9.8, "close": 10.2,
        "volume": 1000, "amount": 2000, "asset_type": "stock", "source": "sample",
    }
    source.import_tdx_prices(pd.DataFrame([row]))
    target.import_tdx_prices(pd.DataFrame([{**row, "close": 1.02, "source": "stale"}]))

    migrate_tdx_history(source.path, target.path, vacuum=False)

    assert target.query("SELECT close, source FROM tdx_daily_prices").to_dict("records") == [
        {"close": 10.2, "source": "sample"}
    ]


def test_migrate_tdx_history_preserves_unrelated_target_history(tmp_path):
    source = ResearchDatabase(tmp_path / "research.sqlite3")
    target = ResearchDatabase(tmp_path / "tdx.sqlite3")
    common = {
        "market": "SZ", "date": "1991-01-02", "open": 10.0, "high": 10.5,
        "low": 9.8, "close": 10.2, "volume": 1000, "amount": 2000,
        "asset_type": "stock", "source": "sample",
    }
    source.import_tdx_prices(pd.DataFrame([{**common, "symbol": "000001"}]))
    target.import_tdx_prices(pd.DataFrame([{**common, "symbol": "000002"}]))

    migrate_tdx_history(source.path, target.path, delete_source=True, vacuum=False)

    assert target.query("SELECT symbol FROM tdx_daily_prices ORDER BY symbol")["symbol"].tolist() == ["000001", "000002"]
    assert source.query("SELECT COUNT(*) AS rows FROM tdx_daily_prices").iloc[0]["rows"] == 0


def test_migrate_tdx_history_rejects_missing_or_same_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        migrate_tdx_history(tmp_path / "missing.sqlite3", tmp_path / "target.sqlite3")
    source = ResearchDatabase(tmp_path / "research.sqlite3")
    with pytest.raises(ValueError):
        migrate_tdx_history(source.path, source.path)


def test_rebuild_main_without_tdx_preserves_research_tables(tmp_path):
    source = ResearchDatabase(tmp_path / "research.sqlite3")
    source.import_prices(
        pd.DataFrame(
            {
                "symbol": ["000001"],
                "date": ["2026-07-10"],
                "open": [10.0],
                "high": [10.2],
                "low": [9.9],
                "close": [10.1],
                "volume": [100],
                "amount": [1000],
            }
        ),
        source="panel",
    )
    source.import_observations(pd.DataFrame({"symbol": ["000001"], "date": ["2026-07-10"], "score": [0.9]}), "candidate", "daily")

    rebuilt_path = tmp_path / "research_rebuilt.sqlite3"
    result = rebuild_main_without_tdx(source.path, rebuilt_path)

    assert result == {"daily_price_rows": 1, "observation_rows": 1, "instrument_rows": 0}
    rebuilt = ResearchDatabase(rebuilt_path)
    assert rebuilt.query("SELECT symbol, close, source FROM daily_prices").to_dict("records") == [
        {"symbol": "000001", "close": 10.1, "source": "panel"}
    ]
    rebuilt_observation = rebuilt.query("SELECT symbol, kind, row_key FROM observations").iloc[0]
    assert rebuilt_observation["symbol"] == "000001"
    assert rebuilt_observation["kind"] == "candidate"
    assert len(rebuilt_observation["row_key"]) == 64
    assert rebuilt.query("SELECT COUNT(*) AS rows FROM tdx_daily_prices").iloc[0]["rows"] == 0


def test_rebuild_main_without_tdx_refuses_implicit_overwrite(tmp_path):
    source = ResearchDatabase(tmp_path / "research.sqlite3")
    rebuilt = ResearchDatabase(tmp_path / "rebuilt.sqlite3")
    with pytest.raises(FileExistsError):
        rebuild_main_without_tdx(source.path, rebuilt.path)
