from pathlib import Path

import pytest

from build_research_database import discover_latest_panel, extract_date_tokens, select_observation_files


def test_extract_date_tokens_rejects_invalid_calendar_dates():
    assert extract_date_tokens(Path("candidate_20260710.csv")) == ["20260710"]
    assert extract_date_tokens(Path("candidate_20261340.csv")) == []


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


def test_select_observation_files_rejects_invalid_asof_date(tmp_path):
    (tmp_path / "candidates_20260710.csv").touch()
    with pytest.raises(ValueError):
        select_observation_files(tmp_path, "20261340")
