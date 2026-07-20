import json
from pathlib import Path

import pandas as pd
import pytest

from research_database import ResearchDatabase
from sync_daily_research_database import sync_daily_research_database, write_status_atomic
from update_model_panel_from_daily_data import read_daily_xls


def test_sync_daily_research_database_is_incremental_and_auditable(tmp_path):
    daily_dir = tmp_path / "daily"
    output_dir = tmp_path / "outputs"
    daily_dir.mkdir()
    output_dir.mkdir()
    daily_path = daily_dir / "ths_hs_a_share_2026-07-13.xls"
    daily_path.write_text(
        "代码\t现价\t开盘\t最高\t最低\t成交量\t成交额\n"
        "000001\t11.24\t11.00\t11.30\t10.90\t100\t1124\n"
        "300001\t20.00\t19.50\t20.20\t19.40\t200\t4000\n",
        encoding="gb18030",
    )
    pd.DataFrame({"symbol": ["000001"], "score": [0.9]}).to_csv(
        output_dir / "merged_priority_watchlist_20260713.csv", index=False
    )
    db_path = tmp_path / "research.sqlite3"

    first = sync_daily_research_database(db_path, daily_dir, output_dir, "2026-07-13")
    second = sync_daily_research_database(db_path, daily_dir, output_dir, "20260713")

    assert first["daily_changed_rows"] == 2
    assert first["observation_changed_rows"] == 1
    assert first["database_asof_rows"] == 2
    assert first["source_latest_date"] == "2026-07-13"
    assert first["price_usable_latest_date"] == "2026-07-13"
    assert first["factor_usable_latest_date"] == "2026-07-13"
    assert first["price_usable_rows"] == 2
    assert first["factor_usable_rows"] == 2
    assert first["price_usable_ratio"] == 1.0
    assert first["factor_column_coverage"]["amount"] == 1.0
    assert second["daily_changed_rows"] == 0
    assert second["observation_changed_rows"] == 0
    assert second["observation_removed_source_alias_rows"] == 0
    db = ResearchDatabase(db_path)
    stored = db.query("SELECT symbol, volume, amount FROM daily_prices ORDER BY symbol")
    assert stored["symbol"].tolist() == ["000001", "300001"]
    assert stored["volume"].tolist() == [100.0, 200.0]
    assert stored["amount"].tolist() == [1124.0, 4000.0]


def test_panel_update_and_database_sync_share_the_same_ths_contract(tmp_path):
    daily_dir = tmp_path / "daily"
    output_dir = tmp_path / "outputs"
    daily_dir.mkdir()
    output_dir.mkdir()
    daily_path = daily_dir / "ths_hs_a_share_2026-07-13.xls"
    daily_path.write_text(
        "代码\t现价\t开盘\t最高\t最低\t总手\t总金额\t换手\t总市值\t大单净额\t主力净量\n"
        "000001\t11.24\t11.00\t11.30\t10.90\t100\t1124\t1.2\t10亿\t1.5万\t0.77\n",
        encoding="gb18030",
    )
    pd.DataFrame({"symbol": ["000001"], "score": [0.9]}).to_csv(
        output_dir / "merged_priority_watchlist_20260713.csv", index=False
    )

    panel_frame, _ = read_daily_xls(daily_path, "2026-07-13", ("000",))
    db_path = tmp_path / "research.sqlite3"
    status = sync_daily_research_database(
        db_path, daily_dir, output_dir, "2026-07-13"
    )
    stored = ResearchDatabase(db_path).query(
        "SELECT symbol, open, high, low, close, volume, amount FROM daily_prices"
    )

    expected = panel_frame.loc[:, stored.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(stored, expected, check_dtype=False)
    assert status["factor_sources"]["main_net_inflow"] == "大单净额"
    assert status["factor_sources"]["main_net_volume_ratio"] == "主力净量"


def test_sync_canonicalizes_relative_and_absolute_observation_sources(tmp_path, monkeypatch):
    daily_dir = tmp_path / "daily"
    output_dir = tmp_path / "outputs"
    daily_dir.mkdir()
    output_dir.mkdir()
    (daily_dir / "ths_hs_a_share_2026-07-13.xls").write_text(
        "代码\t现价\t开盘\t最高\t最低\n000001\t11.24\t11.00\t11.30\t10.90\n",
        encoding="gb18030",
    )
    pd.DataFrame({"symbol": ["000001"], "score": [0.9]}).to_csv(
        output_dir / "merged_priority_watchlist_20260713.csv", index=False
    )
    db_path = tmp_path / "research.sqlite3"
    monkeypatch.chdir(tmp_path)

    first = sync_daily_research_database(
        db_path, Path("daily"), Path("outputs"), "2026-07-13"
    )
    second = sync_daily_research_database(
        db_path, daily_dir.resolve(), output_dir.resolve(), "2026-07-13"
    )

    db = ResearchDatabase(db_path)
    assert first["observation_changed_rows"] == 1
    assert second["observation_changed_rows"] == 0
    assert second["observation_removed_source_alias_rows"] == 0
    assert db.query("SELECT COUNT(*) AS rows FROM observations").iloc[0]["rows"] == 1
    source = db.query("SELECT source FROM observations").iloc[0]["source"]
    assert source == str((output_dir / "merged_priority_watchlist_20260713.csv").resolve())


def test_sync_removes_legacy_relative_source_aliases(tmp_path, monkeypatch):
    daily_dir = tmp_path / "daily"
    output_dir = tmp_path / "outputs"
    daily_dir.mkdir()
    output_dir.mkdir()
    (daily_dir / "ths_hs_a_share_2026-07-13.xls").write_text(
        "代码\t现价\t开盘\t最高\t最低\n000001\t11.24\t11.00\t11.30\t10.90\n",
        encoding="gb18030",
    )
    observation = output_dir / "merged_priority_watchlist_20260713.csv"
    pd.DataFrame({"symbol": ["000001"], "score": [0.9]}).to_csv(observation, index=False)
    db_path = tmp_path / "research.sqlite3"
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame(
        {"date": ["2026-07-13"], "symbol": ["000001"], "score": [0.9]}
    )
    db = ResearchDatabase(db_path)
    db.import_observations(
        frame,
        observation.stem,
        str(Path("outputs") / observation.name),
    )

    status = sync_daily_research_database(
        db_path, daily_dir, output_dir, "2026-07-13"
    )

    assert status["observation_removed_source_alias_rows"] == 1
    assert db.query("SELECT COUNT(*) AS rows FROM observations").iloc[0]["rows"] == 1
    assert db.query("SELECT source FROM observations").iloc[0]["source"] == str(observation.resolve())


def test_sync_rejects_mismatched_daily_dates_without_changing_database(tmp_path):
    daily_dir = tmp_path / "daily"
    output_dir = tmp_path / "outputs"
    daily_dir.mkdir()
    output_dir.mkdir()
    (daily_dir / "ths_hs_a_share_2026-07-13.xls").write_text(
        "symbol\tdate\tclose\n000001\t2026-07-14\t11.24\n",
        encoding="gb18030",
    )
    pd.DataFrame({"symbol": ["000001"], "score": [0.9]}).to_csv(
        output_dir / "merged_priority_watchlist_20260713.csv", index=False
    )
    db_path = tmp_path / "research.sqlite3"
    db = ResearchDatabase(db_path)

    with pytest.raises(ValueError, match="must contain only"):
        sync_daily_research_database(db_path, daily_dir, output_dir, "2026-07-13")

    assert db.query("SELECT COUNT(*) AS rows FROM daily_prices").iloc[0]["rows"] == 0
    assert db.query("SELECT COUNT(*) AS rows FROM observations").iloc[0]["rows"] == 0


def test_sync_ignores_unapproved_dated_csv_files(tmp_path):
    daily_dir = tmp_path / "daily"
    output_dir = tmp_path / "outputs"
    daily_dir.mkdir()
    output_dir.mkdir()
    (daily_dir / "ths_hs_a_share_2026-07-13.xls").write_text(
        "代码\t现价\t开盘\t最高\t最低\n"
        "000001\t11.24\t11.00\t11.30\t10.90\n",
        encoding="gb18030",
    )
    pd.DataFrame({"symbol": ["000001"], "score": [0.9]}).to_csv(
        output_dir / "merged_priority_watchlist_20260713.csv", index=False
    )
    pd.DataFrame({"secret": ["ignore"]}).to_csv(
        output_dir / "unrelated_export_20260713.csv", index=False
    )

    status = sync_daily_research_database(
        tmp_path / "research.sqlite3", daily_dir, output_dir, "2026-07-13"
    )

    assert status["observation_file_count"] == 1
    assert status["observation_files"][0].endswith("merged_priority_watchlist_20260713.csv")
    assert status["price_usable_latest_date"] == "2026-07-13"
    assert status["factor_usable_latest_date"] is None


def test_sync_excludes_unusable_ohlc_and_records_source_coverage(tmp_path):
    daily_dir = tmp_path / "daily"
    output_dir = tmp_path / "outputs"
    daily_dir.mkdir()
    output_dir.mkdir()
    (daily_dir / "ths_hs_a_share_2026-07-13.xls").write_text(
        "代码\t现价\t开盘\t最高\t最低\n"
        "000001\t11.24\t11.00\t11.30\t10.90\n"
        "000002\t12.00\t--\t12.10\t11.90\n",
        encoding="gb18030",
    )
    pd.DataFrame({"symbol": ["000001"], "score": [0.9]}).to_csv(
        output_dir / "merged_priority_watchlist_20260713.csv", index=False
    )
    db_path = tmp_path / "research.sqlite3"

    status = sync_daily_research_database(db_path, daily_dir, output_dir, "2026-07-13")

    db = ResearchDatabase(db_path)
    assert status["source_unique_symbols"] == 2
    assert status["price_usable_rows"] == 1
    assert status["price_unusable_rows"] == 1
    assert status["price_unusable_symbols"] == ["000002"]
    assert status["price_usable_ratio"] == 0.5
    assert status["daily_import_rows"] == 1
    assert db.query("SELECT symbol FROM daily_prices")["symbol"].tolist() == ["000001"]


def test_sync_removes_prior_asof_rows_that_are_no_longer_price_usable(tmp_path):
    daily_dir = tmp_path / "daily"
    output_dir = tmp_path / "outputs"
    daily_dir.mkdir()
    output_dir.mkdir()
    daily_path = daily_dir / "ths_hs_a_share_2026-07-13.xls"
    daily_path.write_text(
        "代码\t现价\t开盘\t最高\t最低\n"
        "000001\t11.24\t11.00\t11.30\t10.90\n"
        "000002\t12.00\t11.90\t12.10\t11.80\n",
        encoding="gb18030",
    )
    pd.DataFrame({"symbol": ["000001"], "score": [0.9]}).to_csv(
        output_dir / "merged_priority_watchlist_20260713.csv", index=False
    )
    db_path = tmp_path / "research.sqlite3"
    sync_daily_research_database(db_path, daily_dir, output_dir, "2026-07-13")
    daily_path.write_text(
        "代码\t现价\t开盘\t最高\t最低\n"
        "000001\t11.24\t11.00\t11.30\t10.90\n"
        "000002\t--\t--\t--\t--\n",
        encoding="gb18030",
    )

    status = sync_daily_research_database(db_path, daily_dir, output_dir, "2026-07-13")

    assert status["daily_removed_unusable_rows"] == 1
    assert status["database_asof_rows"] == 1
    db = ResearchDatabase(db_path)
    assert db.query("SELECT symbol FROM daily_prices")["symbol"].tolist() == ["000001"]


def test_sync_rejects_duplicate_date_symbol_rows(tmp_path):
    daily_dir = tmp_path / "daily"
    output_dir = tmp_path / "outputs"
    daily_dir.mkdir()
    output_dir.mkdir()
    (daily_dir / "ths_hs_a_share_2026-07-13.xls").write_text(
        "代码\t现价\t开盘\t最高\t最低\n"
        "000001\t11.24\t11.00\t11.30\t10.90\n"
        "000001\t11.25\t11.00\t11.30\t10.90\n",
        encoding="gb18030",
    )
    pd.DataFrame({"symbol": ["000001"], "score": [0.9]}).to_csv(
        output_dir / "merged_priority_watchlist_20260713.csv", index=False
    )

    with pytest.raises(ValueError, match="duplicate date-symbol"):
        sync_daily_research_database(
            tmp_path / "research.sqlite3", daily_dir, output_dir, "2026-07-13"
        )


def test_write_status_atomic_replaces_existing_file(tmp_path):
    path = tmp_path / "status.json"
    write_status_atomic(path, {"status": "old"})
    write_status_atomic(path, {"status": "success", "rows": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "success", "rows": 2}
