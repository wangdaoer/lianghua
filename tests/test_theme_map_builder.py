from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser
from quant_etf_lab.theme_map_builder import parse_theme_source, run_theme_map_build


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
    return FetchStatus("run-1", "fake", "all", 3, "ok", None, "2026-06-15T15:30:00")

def get_latest_trade_date(db_path=None, project_root=None):
    return "2026-06-15"

def load_snapshot_rows(trade_date=None, market="all", db_path=None, project_root=None, fallback_to_csv=True):
    return []
"""


def _write_fake_ingest(root: Path) -> None:
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "market_data_utils.py").write_text(FAKE_UTILS, encoding="utf-8")


def _write_daily_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"market": "sse", "trade_date": "2026-06-15", "security_code": "000001", "security_name": "PingAn"},
            {"market": "szse", "trade_date": "2026-06-15", "security_code": "300001", "security_name": "ChinextA"},
            {"market": "sse", "trade_date": "2026-06-15", "security_code": "688001", "security_name": "StarA"},
        ]
    ).to_csv(path, index=False)


def test_theme_map_cli_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["theme-map", "build"])

    assert args.command == "theme-map"
    assert args.theme_map_command == "build"
    assert args.output_dir == "outputs/research/theme_map_latest"
    assert args.include_board_fallback is True
    assert args.require_success is True


def test_parse_csv_source_expands_multi_theme_memberships(tmp_path: Path) -> None:
    source = tmp_path / "concepts.csv"
    pd.DataFrame(
        [
            {"code": "000001", "concept": "AI;robot", "name": "A"},
            {"code": "sz300001", "concept": "robot", "name": "B"},
        ]
    ).to_csv(source, index=False)

    frame, audit = parse_theme_source(source)

    assert audit["status"] == "ok"
    assert set(frame.loc[frame["code"] == "000001", "theme"]) == {"AI", "robot"}
    assert set(frame.loc[frame["code"] == "300001", "theme"]) == {"robot"}


def test_parse_ini_source_uses_section_as_theme(tmp_path: Path) -> None:
    source = tmp_path / "stockblock.ini"
    source.write_text("[watch_alpha]\n000001,sz300001\n[watch_beta]\nSH600000\n", encoding="utf-8")

    frame, audit = parse_theme_source(source)

    assert audit["status"] == "ok"
    assert set(frame.loc[frame["code"] == "000001", "theme"]) == {"watch_alpha"}
    assert set(frame.loc[frame["code"] == "300001", "theme"]) == {"watch_alpha"}
    assert set(frame.loc[frame["code"] == "600000", "theme"]) == {"watch_beta"}


def test_run_theme_map_build_prefers_explicit_sources_and_fills_board_fallback(tmp_path: Path) -> None:
    daily = tmp_path / "daily-market-data"
    ingest = tmp_path / "exchange-ingest"
    _write_fake_ingest(ingest)
    _write_daily_snapshot(daily / "snapshots" / "2026-06-15_market_snapshot.csv")
    source = tmp_path / "concepts.csv"
    pd.DataFrame(
        [
            {"code": "000001", "concept": "AI;robot", "name": "A"},
            {"code": "300001", "concept": "robot", "name": "B"},
        ]
    ).to_csv(source, index=False)

    result = run_theme_map_build(
        project_root=tmp_path,
        output_dir=tmp_path / "theme_map",
        sources=[source],
        include_default_scan_roots=False,
        daily_data_dir=daily,
        ingest_project_dir=ingest,
    )

    assert result.snapshot["market_source_kind"] == "daily_market_data_csv"
    assert result.snapshot["explicit_source_rows"] == 3
    assert result.snapshot["board_fallback_rows"] == 1
    assert set(result.theme_map.loc[result.theme_map["code"] == "000001", "theme"]) == {"AI", "robot"}
    assert set(result.theme_map.loc[result.theme_map["code"] == "688001", "theme"]) == {"star"}
    assert result.theme_map_path.exists()
    assert result.snapshot_path.exists()
    assert result.report_path.exists()
