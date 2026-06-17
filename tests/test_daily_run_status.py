import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser, main
from quant_etf_lab.daily_run_status import run_daily_run_status


FAKE_MARKET_DATA_UTILS = """
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
            "source": "fake_ingest",
        }
    ]
"""


def _write_fake_market_data_utils(root: Path) -> None:
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "market_data_utils.py").write_text(FAKE_MARKET_DATA_UTILS, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_daily_run_status_summarizes_ready_observation(tmp_path: Path) -> None:
    logs = tmp_path / "outputs" / "logs"
    research = tmp_path / "outputs" / "research"
    stocks = tmp_path / "data" / "processed" / "stocks"
    stocks.mkdir(parents=True)
    (stocks / "000001.csv").write_text("date,open,close\n2026-06-12,1,2\n2026-06-15,2,3\n", encoding="utf-8")
    _write_json(
        logs / "daily_research_refresh_20260615_161000.status.json",
        {
            "started_at": "2026-06-15T16:10:00",
            "finished_at": "2026-06-15T16:20:00",
            "exit_code": 0,
            "stdout_log": "refresh.out.log",
            "stderr_log": "refresh.err.log",
        },
    )
    _write_json(
        logs / "allocator_observation_20260615_162001.status.json",
        {
            "started_at": "2026-06-15T16:20:01",
            "finished_at": "2026-06-15T16:20:05",
            "exit_code": 0,
            "observation_dir": str(research / "allocator_observation_20260615"),
            "pipeline_snapshot": str(research / "daily_pipeline_20260615" / "daily_pipeline_snapshot.json"),
        },
    )
    _write_json(
        research / "allocator_observation_20260615" / "allocator_observation_snapshot.json",
        {
            "status": "allocator_observation_completed",
            "as_of_date": "2026-06-15",
            "next_action_stage": "review_outcome_analysis",
            "risk_budget_decision": "trial_small_satellite",
            "outcome_ready_horizon_count": 1,
            "outcome_analysis_status": "ready",
            "paper_stock_target_review_outcome_calendar_next_action_date": "2026-06-16",
            "paper_stock_target_review_outcome_calendar_next_action_horizon": "5d",
            "paper_stock_target_review_outcome_maturity_next_1d_date": "2026-06-15",
        },
    )

    result = run_daily_run_status(
        project_root=tmp_path,
        output_dir=tmp_path / "status",
        logs_dir=logs,
        research_dir=research,
        data_cache_dir=stocks,
    )

    assert result.snapshot["run_state"] == "outcome_ready"
    assert result.snapshot["run_state_severity"] == "ok"
    assert result.snapshot["problem_state"] is False
    assert result.snapshot["latest_refresh_exit_code"] == 0
    assert result.snapshot["latest_observation_exit_code"] == 0
    assert result.snapshot["outcome_ready_horizon_count"] == 1
    assert result.snapshot["latest_stock_cache_date"] == "2026-06-15"
    assert result.snapshot["stock_cache_file_count"] == 1
    assert result.snapshot["stock_cache_source_kind"] == "legacy_stock_csv"
    assert result.snapshot["stock_cache_source_path"] == str(stocks)
    assert result.snapshot["next_review_date"] == "2026-06-16"
    assert result.snapshot["next_review_horizon"] == "5d"
    assert result.snapshot_path.exists()
    assert "Outcome ready horizons" in result.report_path.read_text(encoding="utf-8")


def test_daily_run_status_reads_unified_daily_market_data_hub(tmp_path: Path, monkeypatch) -> None:
    from quant_etf_lab import daily_run_status, market_data_source

    daily = tmp_path / "daily-market-data"
    ingest = tmp_path / "exchange-ingest"
    (daily / "snapshots").mkdir(parents=True)
    _write_fake_market_data_utils(ingest)
    pd.DataFrame(
        [
            {
                "market": "sse",
                "trade_date": "2026-06-15",
                "security_code": "600000",
                "security_name": "Daily",
                "close_price": 10.0,
                "source": "daily_hub",
            }
        ]
    ).to_csv(daily / "snapshots" / "2026-06-15_market_snapshot.csv", index=False)
    monkeypatch.setattr(market_data_source, "DEFAULT_EXCHANGE_INGEST_DIR", ingest)
    monkeypatch.setattr(daily_run_status, "DEFAULT_EXCHANGE_INGEST_DIR", ingest)

    result = run_daily_run_status(
        project_root=tmp_path,
        output_dir=tmp_path / "status",
        logs_dir=tmp_path / "missing_logs",
        research_dir=tmp_path / "missing_research",
        data_cache_dir=daily,
        current_dt=datetime(2026, 6, 15, 15, 0, 0),
    )

    assert result.snapshot["latest_stock_cache_date"] == "2026-06-15"
    assert result.snapshot["stock_cache_file_count"] == 1
    assert result.snapshot["stock_cache_source_kind"] == "daily_market_data_csv"
    assert result.snapshot["stock_cache_snapshot_row_count"] == 1
    assert result.snapshot["stock_cache_fetch_status"] == "ok"
    assert result.snapshot["stock_cache_source_path"].endswith("2026-06-15_market_snapshot.csv")
    assert (ingest / "status_checked.txt").exists()


def test_daily_run_status_falls_back_to_exchange_ingest_when_daily_hub_missing_latest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from quant_etf_lab import daily_run_status, market_data_source

    daily = tmp_path / "daily-market-data"
    ingest = tmp_path / "exchange-ingest"
    daily.mkdir(parents=True)
    _write_fake_market_data_utils(ingest)
    monkeypatch.setattr(market_data_source, "DEFAULT_EXCHANGE_INGEST_DIR", ingest)
    monkeypatch.setattr(daily_run_status, "DEFAULT_EXCHANGE_INGEST_DIR", ingest)

    result = run_daily_run_status(
        project_root=tmp_path,
        output_dir=tmp_path / "status",
        logs_dir=tmp_path / "missing_logs",
        research_dir=tmp_path / "missing_research",
        data_cache_dir=daily,
        current_dt=datetime(2026, 6, 15, 15, 0, 0),
    )

    assert result.snapshot["latest_stock_cache_date"] == "2026-06-15"
    assert result.snapshot["stock_cache_file_count"] == 1
    assert result.snapshot["stock_cache_source_kind"] == "exchange_ingest"
    assert result.snapshot["stock_cache_fetch_status"] == "ok"
    assert result.snapshot["stock_cache_snapshot_row_count"] == 1
    assert (ingest / "status_checked.txt").exists()


def test_daily_run_status_flags_failed_refresh_before_observation(tmp_path: Path) -> None:
    logs = tmp_path / "outputs" / "logs"
    research = tmp_path / "outputs" / "research"
    _write_json(
        logs / "daily_research_refresh_20260615_161000.status.json",
        {
            "started_at": "2026-06-15T16:10:00",
            "finished_at": "2026-06-15T16:10:30",
            "exit_code": 1,
            "stderr_log": "refresh.err.log",
        },
    )

    result = run_daily_run_status(
        project_root=tmp_path,
        output_dir=tmp_path / "status",
        logs_dir=logs,
        research_dir=research,
        data_cache_dir=tmp_path / "missing",
    )

    assert result.snapshot["run_state"] == "refresh_failed"
    assert result.snapshot["run_state_severity"] == "problem"
    assert result.snapshot["problem_state"] is True
    assert result.snapshot["latest_refresh_exit_code"] == 1
    assert result.snapshot["latest_observation_status_path"] is None
    assert result.snapshot["latest_stock_cache_date"] is None


def test_daily_run_status_reads_powershell_utf8_bom_status_json(tmp_path: Path) -> None:
    logs = tmp_path / "outputs" / "logs"
    research = tmp_path / "outputs" / "research"
    logs.mkdir(parents=True)
    (logs / "daily_research_refresh_20260615_161000.status.json").write_text(
        json.dumps({"exit_code": 0, "finished_at": "2026-06-15T16:20:00"}),
        encoding="utf-8-sig",
    )

    result = run_daily_run_status(
        project_root=tmp_path,
        output_dir=tmp_path / "status",
        logs_dir=logs,
        research_dir=research,
        data_cache_dir=tmp_path / "missing",
    )

    assert result.snapshot["latest_refresh_exit_code"] == 0
    assert result.snapshot["run_state"] == "observation_missing"
    assert result.snapshot["run_state_severity"] == "problem"
    assert result.snapshot["problem_state"] is True


def test_daily_run_status_waits_for_scheduled_run_before_today_review_time(tmp_path: Path) -> None:
    logs = tmp_path / "outputs" / "logs"
    research = tmp_path / "outputs" / "research"
    _write_json(
        logs / "daily_research_refresh_20260614_161000.status.json",
        {"exit_code": 0, "finished_at": "2026-06-14T16:20:00"},
    )
    _write_json(
        research / "allocator_observation_precheck_20260614" / "allocator_observation_snapshot.json",
        {
            "outcome_ready_horizon_count": 0,
            "paper_stock_target_review_outcome_calendar_next_action_date": "2026-06-15",
            "paper_stock_target_review_outcome_calendar_next_action_horizon": "1d",
        },
    )

    result = run_daily_run_status(
        project_root=tmp_path,
        output_dir=tmp_path / "status",
        logs_dir=logs,
        research_dir=research,
        data_cache_dir=tmp_path / "missing",
        current_dt=datetime(2026, 6, 15, 4, 30),
        scheduled_run_time="16:10",
    )

    assert result.snapshot["run_state"] == "waiting_for_scheduled_run"
    assert result.snapshot["run_state_severity"] == "waiting"
    assert result.snapshot["problem_state"] is False
    assert result.snapshot["scheduled_run_time"] == "16:10"


def test_daily_run_status_flags_missing_observation_after_today_schedule(tmp_path: Path) -> None:
    logs = tmp_path / "outputs" / "logs"
    research = tmp_path / "outputs" / "research"
    _write_json(
        logs / "daily_research_refresh_20260614_161000.status.json",
        {"exit_code": 0, "finished_at": "2026-06-14T16:20:00"},
    )
    _write_json(
        research / "allocator_observation_precheck_20260614" / "allocator_observation_snapshot.json",
        {
            "outcome_ready_horizon_count": 0,
            "paper_stock_target_review_outcome_calendar_next_action_date": "2026-06-15",
            "paper_stock_target_review_outcome_calendar_next_action_horizon": "1d",
        },
    )

    result = run_daily_run_status(
        project_root=tmp_path,
        output_dir=tmp_path / "status",
        logs_dir=logs,
        research_dir=research,
        data_cache_dir=tmp_path / "missing",
        current_dt=datetime(2026, 6, 15, 16, 30),
        scheduled_run_time="16:10",
    )

    assert result.snapshot["run_state"] == "observation_missing"
    assert result.snapshot["run_state_severity"] == "problem"
    assert result.snapshot["problem_state"] is True


def test_daily_run_status_includes_latest_wrapper_status(tmp_path: Path) -> None:
    logs = tmp_path / "outputs" / "logs"
    research = tmp_path / "outputs" / "research"
    _write_json(
        logs / "daily_research_refresh_with_observation_20260615_162100.status.json",
        {
            "exit_code": 0,
            "final_stage": "completed",
            "refresh_exit_code": 0,
            "observation_exit_code": 0,
            "daily_run_status_exit_code": 0,
            "wrapper_status_path": "wrapper.status.json",
        },
    )
    _write_json(
        logs / "daily_research_refresh_20260615_161000.status.json",
        {"exit_code": 0, "finished_at": "2026-06-15T16:20:00"},
    )
    _write_json(
        logs / "allocator_observation_20260615_162001.status.json",
        {
            "exit_code": 0,
            "observation_dir": str(research / "allocator_observation_20260615"),
        },
    )
    _write_json(
        research / "allocator_observation_20260615" / "allocator_observation_snapshot.json",
        {"outcome_ready_horizon_count": 0},
    )

    result = run_daily_run_status(
        project_root=tmp_path,
        output_dir=tmp_path / "status",
        logs_dir=logs,
        research_dir=research,
        data_cache_dir=tmp_path / "missing",
    )

    assert result.snapshot["latest_wrapper_exit_code"] == 0
    assert result.snapshot["latest_wrapper_final_stage"] == "completed"
    assert result.snapshot["latest_wrapper_refresh_exit_code"] == 0
    assert result.snapshot["latest_wrapper_observation_exit_code"] == 0
    assert result.snapshot["latest_wrapper_daily_run_status_exit_code"] == 0
    assert result.snapshot["run_state_severity"] == "waiting"
    assert result.snapshot["problem_state"] is False
    assert result.snapshot["latest_wrapper_status_path"].endswith("daily_research_refresh_with_observation_20260615_162100.status.json")
    assert "Wrapper final stage" in result.report_path.read_text(encoding="utf-8")


def test_daily_run_status_includes_live_preflight_status(tmp_path: Path) -> None:
    logs = tmp_path / "outputs" / "logs"
    research = tmp_path / "outputs" / "research"
    logs.mkdir(parents=True)
    preflight_output = tmp_path / "outputs" / "research" / "live_preflight_20260615"
    preflight_output.mkdir(parents=True)
    _write_json(
        preflight_output / "live_preflight_snapshot.json",
        {
            "status": "live_preflight_completed",
            "decision": "blocked",
            "blocking_items": ["paper_account_stale", "market_cache_stale"],
            "monitor_items": ["low_exposure"],
            "broker_connection_status": "not_connected",
            "active_target_count": 3,
            "stock_target_review_action_count": 0,
            "live_shadow_review_decisions_path": str(preflight_output / "live_shadow_review_decisions.csv"),
            "live_shadow_review_decision_count": 4,
            "live_shadow_review_blocking_decision_count": 1,
            "live_shadow_review_monitor_decision_count": 2,
            "live_shadow_review_unknown_decision_count": 1,
        },
    )
    preflight_report = preflight_output / "live_preflight.md"
    preflight_report.write_text("# Live Preflight Readiness", encoding="utf-8")
    preflight_status_file = logs / "live_preflight_20260615_163000.status.json"
    preflight_status_path = preflight_status_file
    preflight_status_file.parent.mkdir(parents=True, exist_ok=True)
    preflight_status_file.write_text("{}", encoding="utf-8")
    preflight_output.mkdir(parents=True, exist_ok=True)
    preflight_report.write_text("# Live Preflight Readiness", encoding="utf-8")
    _write_json(
        logs / "live_preflight_20260615_163000.status.json",
        {
            "started_at": "2026-06-15T16:30:00",
            "finished_at": "2026-06-15T16:30:10",
            "exit_code": 1,
            "output_dir": str(preflight_output),
            "stdout_log": "live_preflight.out.log",
            "stderr_log": "live_preflight.err.log",
            "command": "python -m quant_etf_lab live-preflight ...",
        },
    )
    _write_json(
        logs / "daily_research_refresh_with_observation_20260615_163500.status.json",
        {
            "exit_code": 1,
            "final_stage": "live_preflight_blocked",
            "refresh_exit_code": 0,
            "observation_exit_code": 0,
            "daily_run_status_exit_code": 0,
            "live_preflight_exit_code": 1,
            "live_preflight_status_path": str(logs / "live_preflight_20260615_163000.status.json"),
        },
    )
    _write_json(
        logs / "daily_research_refresh_20260615_161000.status.json",
        {
            "started_at": "2026-06-15T16:10:00",
            "finished_at": "2026-06-15T16:20:00",
            "exit_code": 0,
            "stdout_log": "refresh.out.log",
            "stderr_log": "refresh.err.log",
        },
    )
    _write_json(
        logs / "allocator_observation_20260615_162001.status.json",
        {
            "started_at": "2026-06-15T16:20:01",
            "finished_at": "2026-06-15T16:20:05",
            "exit_code": 0,
            "observation_dir": str(research / "allocator_observation_20260615"),
        },
    )

    result = run_daily_run_status(
        project_root=tmp_path,
        output_dir=tmp_path / "status",
        logs_dir=logs,
        research_dir=research,
        data_cache_dir=tmp_path / "missing",
    )

    assert result.snapshot["run_state"] == "live_preflight_blocked"
    assert result.snapshot["run_state_severity"] == "problem"
    assert result.snapshot["problem_state"] is True
    assert result.snapshot["latest_live_preflight_status_path"].endswith("live_preflight_20260615_163000.status.json")
    assert result.snapshot["latest_live_preflight_exit_code"] == 1
    assert result.snapshot["latest_live_preflight_status"] == "live_preflight_completed"
    assert result.snapshot["latest_live_preflight_decision"] == "blocked"
    assert result.snapshot["latest_live_preflight_blocking_items_count"] == 2
    assert result.snapshot["latest_live_preflight_monitor_items_count"] == 1
    assert result.snapshot["latest_live_preflight_live_shadow_review_decisions_path"] == str(
        preflight_output / "live_shadow_review_decisions.csv"
    )
    assert result.snapshot["latest_live_preflight_live_shadow_review_decision_count"] == 4
    assert result.snapshot["latest_live_preflight_live_shadow_review_blocking_decision_count"] == 1
    assert result.snapshot["latest_live_preflight_live_shadow_review_monitor_decision_count"] == 2
    assert result.snapshot["latest_live_preflight_live_shadow_review_unknown_decision_count"] == 1
    assert result.snapshot["latest_live_preflight_output_dir"] == str(preflight_output)
    assert result.snapshot["latest_live_preflight_snapshot_path"] == str(preflight_output / "live_preflight_snapshot.json")
    assert "Live preflight status" in result.report_path.read_text(encoding="utf-8")


def test_daily_run_status_ignores_live_preflight_block_when_wrapper_skips_live_preflight(tmp_path: Path) -> None:
    logs = tmp_path / "outputs" / "logs"
    research = tmp_path / "outputs" / "research"
    logs.mkdir(parents=True)
    stale_output = tmp_path / "outputs" / "research" / "live_preflight_20260614"
    stale_output.mkdir(parents=True)
    _write_json(
        stale_output / "live_preflight_snapshot.json",
        {
            "status": "live_preflight_completed",
            "decision": "blocked",
            "blocking_items": ["paper_account_stale"],
            "monitor_items": [],
            "broker_connection_status": "not_connected",
            "active_target_count": 2,
            "stock_target_review_action_count": 0,
        },
    )
    stale_status = logs / "live_preflight_20260614_163000.status.json"
    _write_json(
        stale_status,
        {
            "output_dir": str(stale_output),
            "snapshot_path": str(stale_output / "live_preflight_snapshot.json"),
            "exit_code": 0,
            "stdout_log": "stale.out.log",
            "stderr_log": "stale.err.log",
        },
    )

    _write_json(
        logs / "daily_research_refresh_20260615_161000.status.json",
        {"exit_code": 0, "finished_at": "2026-06-15T16:20:00"},
    )
    _write_json(
        logs / "daily_research_refresh_with_observation_20260615_162100.status.json",
        {
            "exit_code": 0,
            "final_stage": "completed",
            "refresh_exit_code": 0,
            "observation_exit_code": 0,
            "daily_run_status_exit_code": 0,
            "skip_live_preflight": True,
        },
    )
    _write_json(
        logs / "allocator_observation_20260615_162001.status.json",
        {
            "exit_code": 0,
            "observation_dir": str(research / "allocator_observation_20260615"),
        },
    )
    _write_json(
        research / "allocator_observation_20260615" / "allocator_observation_snapshot.json",
        {
            "status": "allocator_observation_completed",
            "as_of_date": "2026-06-15",
            "outcome_ready_horizon_count": 0,
            "outcome_analysis_status": "waiting_for_evaluable_returns",
            "paper_stock_target_review_outcome_calendar_next_action_date": "2026-06-16",
            "paper_stock_target_review_outcome_calendar_next_action_horizon": "1d",
        },
    )

    result = run_daily_run_status(
        project_root=tmp_path,
        output_dir=tmp_path / "status",
        logs_dir=logs,
        research_dir=research,
        data_cache_dir=tmp_path / "missing",
    )

    assert result.snapshot["run_state"] == "waiting_for_outcome_samples"
    assert result.snapshot["run_state_severity"] == "waiting"
    assert result.snapshot["problem_state"] is False
    assert result.snapshot["latest_live_preflight_status_path"] is None
    assert result.snapshot["latest_live_preflight_snapshot_path"] is None
    assert result.snapshot["latest_live_preflight_status"] is None
    assert result.snapshot["skip_live_preflight"] is True


def test_daily_run_status_includes_pipeline_preflight_smoke_status(tmp_path: Path) -> None:
    logs = tmp_path / "outputs" / "logs"
    logs.mkdir(parents=True)
    _write_json(
        logs / "daily_research_refresh_20260615_161000.status.json",
        {
            "exit_code": 0,
            "finished_at": "2026-06-15T16:20:00",
        },
    )
    _write_json(
        logs / "pipeline_preflight_smoke_20260615_162000.status.json",
        {
            "exit_code": 4,
            "finished_at": "2026-06-15T16:20:05",
            "stdout_log": "pipeline_preflight_smoke_20260615_162000.out.log",
        },
    )
    _write_json(
        logs / "daily_research_refresh_with_observation_20260615_163000.status.json",
        {
            "exit_code": 4,
            "final_stage": "daily_pipeline_preflight_smoke_failed",
            "refresh_exit_code": 0,
            "observation_exit_code": None,
            "daily_run_status_exit_code": None,
            "pipeline_preflight_smoke_exit_code": 4,
            "pipeline_preflight_smoke_status_path": str(logs / "pipeline_preflight_smoke_20260615_162000.status.json"),
        },
    )

    result = run_daily_run_status(
        project_root=tmp_path,
        output_dir=tmp_path / "status",
        logs_dir=logs,
        research_dir=tmp_path / "outputs" / "research",
        data_cache_dir=tmp_path / "missing",
    )

    assert result.snapshot["run_state"] == "daily_pipeline_preflight_smoke_failed"
    assert result.snapshot["run_state_severity"] == "problem"
    assert result.snapshot["problem_state"] is True
    assert result.snapshot["latest_pipeline_preflight_smoke_exit_code"] == 4
    assert result.snapshot["latest_pipeline_preflight_smoke_status_path"] == str(
        logs / "pipeline_preflight_smoke_20260615_162000.status.json"
    )
    assert "Pipeline preflight smoke exit code" in result.report_path.read_text(encoding="utf-8")


def test_daily_run_status_prioritizes_latest_wrapper_failure_over_stale_refresh_success(tmp_path: Path) -> None:
    logs = tmp_path / "outputs" / "logs"
    research = tmp_path / "outputs" / "research"
    _write_json(
        logs / "daily_research_refresh_20260614_161000.status.json",
        {"exit_code": 0, "finished_at": "2026-06-14T16:20:00"},
    )
    _write_json(
        logs / "daily_research_refresh_with_observation_20260615_161000.status.json",
        {
            "exit_code": 75,
            "final_stage": "lock_conflict",
            "refresh_exit_code": None,
            "observation_exit_code": None,
            "daily_run_status_exit_code": None,
        },
    )

    result = run_daily_run_status(
        project_root=tmp_path,
        output_dir=tmp_path / "status",
        logs_dir=logs,
        research_dir=research,
        data_cache_dir=tmp_path / "missing",
    )

    assert result.snapshot["run_state"] == "lock_conflict"
    assert result.snapshot["run_state_severity"] == "problem"
    assert result.snapshot["problem_state"] is True
    assert result.snapshot["latest_wrapper_exit_code"] == 75
    assert result.snapshot["latest_refresh_exit_code"] == 0


def test_daily_run_status_uses_newer_wrapper_after_older_lock_conflict(tmp_path: Path) -> None:
    logs = tmp_path / "outputs" / "logs"
    research = tmp_path / "outputs" / "research"
    _write_json(
        logs / "daily_research_refresh_20260615_162031.status.json",
        {"exit_code": 0, "finished_at": "2026-06-15T16:21:19"},
    )
    _write_json(
        logs / "daily_research_refresh_with_observation_20260615_161007.status.json",
        {
            "started_at": "2026-06-15T16:10:07",
            "finished_at": "2026-06-15T16:10:07",
            "exit_code": 75,
            "final_stage": "lock_conflict",
            "refresh_exit_code": None,
            "observation_exit_code": None,
            "daily_run_status_exit_code": None,
        },
    )
    _write_json(
        logs / "daily_research_refresh_with_observation_20260615_162030.status.json",
        {
            "started_at": "2026-06-15T16:20:30",
            "finished_at": "2026-06-15T16:22:03",
            "exit_code": 7,
            "final_stage": "live_preflight_blocked",
            "refresh_exit_code": 0,
            "observation_exit_code": 0,
            "daily_run_status_exit_code": 0,
            "live_preflight_status_path": str(logs / "live_preflight_20260615_162201.status.json"),
        },
    )
    _write_json(
        logs / "live_preflight_20260615_162201.status.json",
        {
            "exit_code": 7,
            "started_at": "2026-06-15T16:22:01",
            "finished_at": "2026-06-15T16:22:03",
            "output_dir": str(research / "live_preflight_20260615"),
        },
    )

    result = run_daily_run_status(
        project_root=tmp_path,
        output_dir=tmp_path / "status",
        logs_dir=logs,
        research_dir=research,
        data_cache_dir=tmp_path / "missing",
    )

    assert result.snapshot["latest_wrapper_final_stage"] == "live_preflight_blocked"
    assert result.snapshot["latest_wrapper_status_path"].endswith(
        "daily_research_refresh_with_observation_20260615_162030.status.json"
    )
    assert result.snapshot["latest_wrapper_exit_code"] == 7
    assert result.snapshot["run_state"] == "live_preflight_blocked"


def test_daily_run_status_cli_gate_returns_nonzero_for_problem_state(tmp_path: Path, monkeypatch) -> None:
    logs = tmp_path / "outputs" / "logs"
    research = tmp_path / "outputs" / "research"
    output = tmp_path / "status"
    _write_json(
        logs / "daily_research_refresh_20260615_161000.status.json",
        {"exit_code": 0, "finished_at": "2026-06-15T16:20:00"},
    )
    _write_json(
        logs / "daily_research_refresh_with_observation_20260615_161000.status.json",
        {
            "exit_code": 75,
            "final_stage": "lock_conflict",
            "refresh_exit_code": None,
            "observation_exit_code": None,
            "daily_run_status_exit_code": None,
        },
    )
    monkeypatch.chdir(tmp_path)

    base_args = [
        "daily-run-status",
        "--logs-dir",
        str(logs),
        "--research-dir",
        str(research),
        "--data-cache-dir",
        str(tmp_path / "missing"),
        "--output-dir",
        str(output),
    ]

    assert main(base_args) == 0
    assert main([*base_args, "--fail-on-problem-state"]) == 6


def test_daily_run_status_cli_prints_live_preflight_and_wrapper_lines(tmp_path: Path, monkeypatch, capsys) -> None:
    logs = tmp_path / "outputs" / "logs"
    research = tmp_path / "outputs" / "research"
    output = tmp_path / "status"
    preflight_output = tmp_path / "outputs" / "research" / "live_preflight_20260615"
    preflight_status_path = logs / "live_preflight_20260615_163000.status.json"
    preflight_output.mkdir(parents=True)
    _write_json(
        preflight_output / "live_preflight_snapshot.json",
        {
            "status": "live_preflight_completed",
            "decision": "ready_for_manual_broker_reconciliation",
            "blocking_items": [],
            "monitor_items": ["paper_target_age"],
            "broker_connection_status": "not_connected",
            "active_target_count": 4,
            "stock_target_review_action_count": 0,
        },
    )
    preflight_report = preflight_output / "live_preflight.md"
    preflight_report.write_text("# Live Preflight Readiness", encoding="utf-8")
    _write_json(
        logs / "live_preflight_20260615_163000.status.json",
        {
            "started_at": "2026-06-15T16:30:00",
            "finished_at": "2026-06-15T16:30:10",
            "exit_code": 0,
            "output_dir": str(preflight_output),
            "snapshot_path": str(preflight_output / "live_preflight_snapshot.json"),
            "report_path": str(preflight_report),
            "command": "python -m quant_etf_lab live-preflight --pipeline-snapshot example.json",
        },
    )
    _write_json(
        logs / "daily_research_refresh_with_observation_20260615_163500.status.json",
        {
            "exit_code": 0,
            "final_stage": "completed",
            "refresh_exit_code": 0,
            "observation_exit_code": 0,
            "daily_run_status_exit_code": 0,
            "live_preflight_exit_code": 0,
            "live_preflight_status_path": str(logs / "live_preflight_20260615_163000.status.json"),
        },
    )
    _write_json(
        logs / "daily_research_refresh_20260615_161000.status.json",
        {
            "started_at": "2026-06-15T16:10:00",
            "finished_at": "2026-06-15T16:20:00",
            "exit_code": 0,
            "stdout_log": "refresh.out.log",
            "stderr_log": "refresh.err.log",
        },
    )
    _write_json(
        logs / "allocator_observation_20260615_162001.status.json",
        {
            "exit_code": 0,
            "started_at": "2026-06-15T16:20:01",
            "finished_at": "2026-06-15T16:20:05",
            "observation_dir": str(research / "allocator_observation_20260615"),
        },
    )
    _write_json(
        research / "allocator_observation_20260615" / "allocator_observation_snapshot.json",
        {
            "outcome_ready_horizon_count": 2,
            "status": "allocator_observation_completed",
            "as_of_date": "2026-06-15",
        },
    )

    monkeypatch.chdir(tmp_path)
    args = [
        "daily-run-status",
        "--logs-dir",
        str(logs),
        "--research-dir",
        str(research),
        "--data-cache-dir",
        str(tmp_path / "missing"),
        "--output-dir",
        str(output),
    ]
    assert main(args) == 0

    output_text = capsys.readouterr().out
    assert "Wrapper final stage: completed" in output_text
    assert "Live preflight status: live_preflight_completed" in output_text
    assert "Live preflight decision: ready_for_manual_broker_reconciliation" in output_text
    assert "Live preflight blockers/monitors: 0 / 1" in output_text
    assert f"Live preflight status path: {logs / 'live_preflight_20260615_163000.status.json'}" in output_text
    assert f"Live preflight report path: {preflight_report}" in output_text
    assert f"Live preflight snapshot path: {preflight_output / 'live_preflight_snapshot.json'}" in output_text
    assert preflight_status_path.exists()
    assert preflight_report.exists()
    assert (preflight_output / "live_preflight_snapshot.json").exists()


def test_daily_run_status_cli_print_json_and_write_path(tmp_path: Path, monkeypatch, capsys) -> None:
    logs = tmp_path / "outputs" / "logs"
    research = tmp_path / "outputs" / "research"
    output = tmp_path / "status"
    logs.mkdir(parents=True)
    research.mkdir(parents=True)
    _write_json(
        logs / "daily_research_refresh_20260615_161000.status.json",
        {"exit_code": 0, "finished_at": "2026-06-15T16:20:00"},
    )
    _write_json(
        logs / "allocator_observation_20260615_162001.status.json",
        {"exit_code": 0, "observation_dir": str(research / "allocator_observation_20260615")},
    )
    _write_json(
        research / "allocator_observation_20260615" / "allocator_observation_snapshot.json",
        {
            "status": "allocator_observation_completed",
            "as_of_date": "2026-06-15",
            "outcome_ready_horizon_count": 1,
        },
    )

    monkeypatch.chdir(tmp_path)
    json_path = tmp_path / "monitor_payload.json"
    args = [
        "daily-run-status",
        "--logs-dir",
        str(logs),
        "--research-dir",
        str(research),
        "--data-cache-dir",
        str(tmp_path / "missing"),
        "--output-dir",
        str(output),
        "--print-json",
        "--json-path",
        str(json_path),
    ]
    assert main(args) == 0
    output_text = capsys.readouterr().out
    json_lines = [line for line in output_text.splitlines() if line.startswith("{") and line.endswith("}")]
    assert len(json_lines) >= 1
    payload = json.loads(json_lines[-1])
    assert payload["run_state"] == "outcome_ready"
    assert payload["latest_observation_exit_code"] == 0

    json_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_payload["run_state"] == "outcome_ready"
    assert json_payload["latest_observation_exit_code"] == 0


def test_daily_run_status_cli_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["daily-run-status"])

    assert args.command == "daily-run-status"
    assert args.output_dir == "outputs/research/daily_run_status_latest"
    assert args.logs_dir == "outputs/logs"
    assert args.research_dir == "outputs/research"
    assert args.data_cache_dir == "D:/codex/daily-market-data"
    assert args.scheduled_run_time == "16:10"
    assert args.fail_on_problem_state is False
    assert args.print_json is False
    assert args.json_path is None


def test_daily_run_status_cli_parser_enables_problem_state_gate() -> None:
    parser = build_parser()
    args = parser.parse_args(["daily-run-status", "--fail-on-problem-state"])

    assert args.command == "daily-run-status"
    assert args.fail_on_problem_state is True
