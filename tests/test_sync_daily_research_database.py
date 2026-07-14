import json

import pandas as pd
import pytest

from research_database import ResearchDatabase
from sync_daily_research_database import sync_daily_research_database, write_status_atomic


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
    assert second["daily_changed_rows"] == 0
    assert second["observation_changed_rows"] == 0
    db = ResearchDatabase(db_path)
    assert db.query("SELECT symbol FROM daily_prices ORDER BY symbol")["symbol"].tolist() == ["000001", "300001"]


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
        "代码\t现价\n000001\t11.24\n", encoding="gb18030"
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


def test_write_status_atomic_replaces_existing_file(tmp_path):
    path = tmp_path / "status.json"
    write_status_atomic(path, {"status": "old"})
    write_status_atomic(path, {"status": "success", "rows": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "success", "rows": 2}
