from pathlib import Path

import pytest

from build_research_database import (
    discover_latest_panel,
    extract_date_tokens,
    infer_file_date,
    prepare_observation_frame,
    select_observation_files,
    select_supported_observation_files,
    unresolved_observation_files,
)


def test_extract_date_tokens_rejects_invalid_calendar_dates():
    assert extract_date_tokens(Path("candidate_20260710.csv")) == ["20260710"]
    assert extract_date_tokens(Path("ths_hs_a_share_2026-07-10.xls")) == ["20260710"]
    assert extract_date_tokens(Path("candidate_20261340.csv")) == []
    assert extract_date_tokens(Path("candidate_2026-13-40.csv")) == []


def test_infer_file_date_supports_compact_and_dashed_names():
    assert infer_file_date(Path("candidate_20260710.csv")) == "2026-07-10"
    assert infer_file_date(Path("ths_hs_a_share_2026-07-10.xls")) == "2026-07-10"
    assert infer_file_date(Path("candidate_without_date.csv")) is None


def test_discover_latest_panel_uses_latest_end_date(tmp_path):
    older = tmp_path / "data_panel_history_main_chinext_20220101_20260709.csv"
    latest = tmp_path / "data_panel_history_main_chinext_20220101_20260710.csv"
    older.touch()
    latest.touch()
    assert discover_latest_panel(tmp_path) == latest


def test_select_observation_files_uses_latest_date_or_explicit_date(tmp_path):
    older = tmp_path / "candidates_20260709.csv"
    latest_a = tmp_path / "candidates_20260710.csv"
    latest_b = tmp_path / "risk_20260710.csv"
    for path in (older, latest_a, latest_b):
        path.touch()
    assert select_observation_files(tmp_path) == [latest_a, latest_b]
    assert select_observation_files(tmp_path, "20260709") == [older]
    assert select_observation_files(tmp_path, "2026-07-09") == [older]


def test_select_observation_files_rejects_invalid_asof_date(tmp_path):
    (tmp_path / "candidates_20260710.csv").touch()
    with pytest.raises(ValueError):
        select_observation_files(tmp_path, "20261340")
    with pytest.raises(ValueError):
        select_observation_files(tmp_path, "2026--07-10")
    with pytest.raises(ValueError):
        select_observation_files(tmp_path, "2026-0710")


def test_select_observation_files_validates_asof_date_even_without_candidates(tmp_path):
    with pytest.raises(ValueError):
        select_observation_files(tmp_path, "not-a-date")


def test_unresolved_observation_files_exposes_undated_and_invalid_names(tmp_path):
    valid = tmp_path / "candidates_20260710.csv"
    undated = tmp_path / "candidates_latest.csv"
    invalid = tmp_path / "candidates_2026-13-40.csv"
    for path in (valid, undated, invalid):
        path.touch()
    assert unresolved_observation_files(tmp_path) == [invalid, undated]


def test_select_supported_observation_files_rejects_unrelated_dated_csv(tmp_path):
    supported = tmp_path / "merged_priority_watchlist_20260713.csv"
    flow_shadow = tmp_path / "main_net_volume_shadow_20260713.csv"
    institutional_shadow = tmp_path / "institutional_accumulation_shadow_20260713.csv"
    institutional_tracking = tmp_path / "institutional_accumulation_tracking_20260713.csv"
    czsc_shadow = tmp_path / "czsc_structure_shadow_20260713.csv"
    arena = tmp_path / "strategy_arena_portfolio_20260713.csv"
    chinese_copy = tmp_path / "czsc_structure_shadow_20260713_cn.csv"
    unrelated = tmp_path / "unrelated_export_20260713.csv"
    supported.touch()
    flow_shadow.touch()
    institutional_shadow.touch()
    institutional_tracking.touch()
    czsc_shadow.touch()
    arena.touch()
    chinese_copy.touch()
    unrelated.touch()

    assert select_supported_observation_files(tmp_path, "20260713") == [
        czsc_shadow,
        institutional_shadow,
        institutional_tracking,
        flow_shadow,
        supported,
        arena,
    ]


def test_prepare_observation_frame_rejects_future_dates(tmp_path):
    path = tmp_path / "merged_priority_watchlist_20260713.csv"
    path.write_text("symbol,date\n000001,2026-07-14\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dates after 2026-07-13"):
        prepare_observation_frame(path, "20260713")


def test_prepare_observation_frame_infers_and_normalizes_filename_date(tmp_path):
    path = tmp_path / "merged_priority_watchlist_20260713.csv"
    path.write_text("symbol,score\n000001,0.9\n", encoding="utf-8")

    frame = prepare_observation_frame(path, "2026-07-13")

    assert frame["date"].tolist() == ["2026-07-13"]
