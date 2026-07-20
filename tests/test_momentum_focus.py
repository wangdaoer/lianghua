from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from quant_etf_lab.cli import build_parser
from quant_etf_lab.momentum_focus import build_momentum_focus_candidates, run_momentum_focus


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
    return FetchStatus("run-1", "fake", "all", 6, "ok", None, "2026-06-15T15:30:00")

def get_latest_trade_date(db_path=None, project_root=None):
    return "2026-06-15"

def load_snapshot_rows(trade_date=None, market="all", db_path=None, project_root=None, fallback_to_csv=True):
    return []
"""


def _write_fake_ingest(root: Path) -> None:
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "market_data_utils.py").write_text(FAKE_MARKET_DATA_UTILS, encoding="utf-8")


def test_build_momentum_focus_candidates_keeps_5pct_strong_gainers_main_chinext_only() -> None:
    rows = [
        {"trade_date": "2026-06-15", "security_code": "000001", "security_name": "MainLimit", "change_ratio": 10.01, "close_price": 11.0, "turnover": 300_000_000},
        {"trade_date": "2026-06-15", "security_code": "300001", "security_name": "ChiLimit", "change_ratio": 20.02, "close_price": 24.0, "turnover": 800_000_000},
        {"trade_date": "2026-06-15", "security_code": "002001", "security_name": "0", "change_ratio": 7.2, "close_price": 8.0, "turnover": 120_000_000},
        {"trade_date": "2026-06-15", "security_code": "600001", "security_name": "Weak", "change_ratio": 6.9, "close_price": 9.0, "turnover": 90_000_000},
        {"trade_date": "2026-06-15", "security_code": "000333", "security_name": "Tiny", "change_ratio": 0.549, "close_price": 53.0, "turnover": 400_000_000},
        {"trade_date": "2026-06-15", "security_code": "688001", "security_name": "Star", "change_ratio": 20.1, "close_price": 40.0, "turnover": 500_000_000},
        {"trade_date": "2026-06-15", "security_code": "bj920126", "security_name": "BSE", "change_ratio": 126.9, "close_price": 17.0, "turnover": 700_000_000},
    ]

    candidates, payload = build_momentum_focus_candidates(rows, as_of_date="2026-06-15")

    assert candidates["code"].tolist() == ["300001", "000001", "002001", "600001"]
    assert payload["candidate_count"] == 4
    assert payload["limit_up_count"] == 2
    assert payload["strong_gain_count"] == 4
    assert payload["excluded_non_main_chinext_count"] == 2
    assert payload["below_threshold_count"] == 1
    assert payload["volume_positive_count"] == 0
    assert payload["turnover_positive_count"] == len(rows)
    assert payload["volume_coverage_ratio"] == 0.0
    assert payload["turnover_coverage_ratio"] == 1.0
    assert payload["market_data_quality_status"] == "partial"
    by_code = candidates.set_index("code")
    assert by_code.loc["300001", "signal_type"] == "strong_gain_5"
    assert by_code.loc["002001", "signal_type"] == "strong_gain_5"
    assert by_code.loc["002001", "name"] == "未知"
    assert bool(by_code.loc["002001", "limit_up"]) is False


def test_build_momentum_focus_candidates_excludes_review_required_codes() -> None:
    rows = [
        {"trade_date": "2026-07-20", "security_code": "300001", "security_name": "Excluded", "change_ratio": 20.0, "turnover": 800_000_000},
        {"trade_date": "2026-07-20", "security_code": "300002", "security_name": "Kept", "change_ratio": 10.0, "turnover": 500_000_000},
    ]

    candidates, payload = build_momentum_focus_candidates(
        rows,
        as_of_date="2026-07-20",
        excluded_codes={"300001"},
    )

    assert candidates["code"].tolist() == ["300002"]
    assert payload["excluded_by_review_required_count"] == 1
    assert payload["excluded_by_review_required_codes"] == ["300001"]


def test_build_momentum_focus_candidates_enriches_with_outcome_priors() -> None:
    rows = [
        {"trade_date": "2026-06-15", "security_code": "300001", "security_name": "HighPrior", "change_ratio": 10.0, "turnover": 50_000_000},
        {"trade_date": "2026-06-15", "security_code": "000001", "security_name": "LowPrior", "change_ratio": 10.0, "turnover": 50_000_000},
    ]
    outcome_summary = pd.DataFrame(
        [
            {
                "signal_type": "strong_gain_7",
                "board": "chinext",
                "amount_bucket": "lt_1y",
                "horizon": 5,
                "event_count": 500,
                "win_rate": 0.80,
                "avg_return": 0.18,
                "profit_factor": 3.5,
                "avg_amount_yi": 0.6,
            },
            {
                "signal_type": "strong_gain_7",
                "board": "main",
                "amount_bucket": "lt_1y",
                "horizon": 5,
                "event_count": 80,
                "win_rate": 0.42,
                "avg_return": 0.03,
                "profit_factor": 1.2,
                "avg_amount_yi": 0.8,
            },
        ]
    )

    candidates, payload = build_momentum_focus_candidates(
        rows,
        as_of_date="2026-06-15",
        strong_gain_threshold_pct=7.0,
        board_scope="main_chinext",
        outcome_summary=outcome_summary,
        target_horizon=5,
    )

    by_code = candidates.set_index("code")
    assert by_code.loc["300001", "outcome_prior_match_level"] == "exact"
    assert by_code.loc["000001", "outcome_prior_match_level"] == "exact"
    assert by_code.loc["300001", "outcome_prior_win_rate"] == 0.80
    assert by_code.loc["000001", "outcome_prior_win_rate"] == 0.42
    assert by_code.loc["300001", "outcome_prior_event_count"] == 500
    assert by_code.loc["000001", "outcome_prior_event_count"] == 80
    assert candidates.iloc[0]["code"] == "300001"
    assert payload["outcome_prior_rows"] == 2
    assert payload["outcome_prior_matched_count"] == 2


def test_build_momentum_focus_candidates_uses_name_map_for_unknown_snapshot_names() -> None:
    rows = [
        {
            "trade_date": "2026-06-15",
            "security_code": "300001",
            "security_name": "0",
            "change_ratio": 20.0,
            "close_price": 24.0,
            "turnover": 800_000_000,
        }
    ]

    candidates, payload = build_momentum_focus_candidates(
        rows,
        as_of_date="2026-06-15",
        name_map={"300001": "MappedName"},
    )

    assert payload["candidate_count"] == 1
    assert candidates.iloc[0]["name"] == "MappedName"


def test_build_momentum_focus_candidates_excludes_special_treatment_names() -> None:
    rows = [
        {
            "trade_date": "2026-06-15",
            "security_code": "300001",
            "security_name": "NormalName",
            "change_ratio": 20.0,
            "close_price": 24.0,
            "turnover": 800_000_000,
        },
        {
            "trade_date": "2026-06-15",
            "security_code": "300002",
            "security_name": "STRisk",
            "change_ratio": 20.0,
            "close_price": 12.0,
            "turnover": 300_000_000,
        },
        {
            "trade_date": "2026-06-15",
            "security_code": "600001",
            "security_name": "\u9000\u5e02Risk",
            "change_ratio": 10.0,
            "close_price": 8.0,
            "turnover": 200_000_000,
        },
    ]

    candidates, payload = build_momentum_focus_candidates(rows, as_of_date="2026-06-15")

    assert candidates["code"].tolist() == ["300001"]
    assert payload["excluded_special_treatment_count"] == 2


def test_run_momentum_focus_reads_daily_hub_after_ingest_status_check(tmp_path: Path) -> None:
    daily = tmp_path / "daily-market-data"
    ingest = tmp_path / "exchange-ingest"
    (daily / "snapshots").mkdir(parents=True)
    _write_fake_ingest(ingest)
    pd.DataFrame(
        [
            {"market": "szse", "trade_date": "2026-06-15", "security_code": "000001", "security_name": "MainLimit", "change_ratio": 10.01, "close_price": 11.0, "turnover": 300_000_000},
            {"market": "szse", "trade_date": "2026-06-15", "security_code": "300001", "security_name": "ChiLimit", "change_ratio": 20.02, "close_price": 24.0, "turnover": 800_000_000},
            {"market": "sse", "trade_date": "2026-06-15", "security_code": "600001", "security_name": "Weak", "change_ratio": 6.9, "close_price": 9.0, "turnover": 90_000_000},
        ]
    ).to_csv(daily / "snapshots" / "2026-06-15_market_snapshot.csv", index=False)

    result = run_momentum_focus(
        project_root=tmp_path,
        output_dir=tmp_path / "momentum_focus",
        daily_data_dir=daily,
        ingest_project_dir=ingest,
        as_of_date="2026-06-15",
    )

    assert result.snapshot["status"] == "ok"
    assert result.snapshot["source_kind"] == "daily_market_data_csv"
    assert result.snapshot["candidate_count"] == 3
    assert result.snapshot["limit_up_count"] == 2
    assert result.candidates_path.exists()
    assert result.report_path.exists()
    payload = json.loads(result.snapshot_path.read_text(encoding="utf-8"))
    assert payload["research_only"] is True


def test_run_momentum_focus_loads_default_name_map(tmp_path: Path) -> None:
    daily = tmp_path / "daily-market-data"
    ingest = tmp_path / "exchange-ingest"
    (daily / "snapshots").mkdir(parents=True)
    _write_fake_ingest(ingest)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    pd.DataFrame([{"code": "300001", "name": "MappedName"}]).to_csv(
        tmp_path / "data" / "processed" / "stock_name_map.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "market": "szse",
                "trade_date": "2026-06-15",
                "security_code": "300001",
                "security_name": "0",
                "change_ratio": 20.02,
                "close_price": 24.0,
                "turnover": 800_000_000,
            },
        ]
    ).to_csv(daily / "snapshots" / "2026-06-15_market_snapshot.csv", index=False)

    result = run_momentum_focus(
        project_root=tmp_path,
        output_dir=tmp_path / "momentum_focus",
        daily_data_dir=daily,
        ingest_project_dir=ingest,
        as_of_date="2026-06-15",
    )

    assert result.candidates.iloc[0]["name"] == "MappedName"
    assert result.snapshot["name_map_status"] == "ok"
    assert result.snapshot["name_map_rows"] == 1


def test_run_momentum_focus_can_separate_report_date_from_trade_date(tmp_path: Path) -> None:
    daily = tmp_path / "daily-market-data"
    ingest = tmp_path / "exchange-ingest"
    (daily / "snapshots").mkdir(parents=True)
    _write_fake_ingest(ingest)
    pd.DataFrame(
        [
            {"market": "szse", "trade_date": "2026-06-15", "security_code": "000001", "security_name": "MainLimit", "change_ratio": 10.01, "close_price": 11.0, "turnover": 300_000_000},
        ]
    ).to_csv(daily / "snapshots" / "2026-06-15_market_snapshot.csv", index=False)

    result = run_momentum_focus(
        project_root=tmp_path,
        output_dir=tmp_path / "momentum_focus",
        daily_data_dir=daily,
        ingest_project_dir=ingest,
        as_of_date="2026-06-16",
        trade_date="2026-06-15",
    )

    assert result.snapshot["as_of_date"] == "2026-06-16"
    assert result.snapshot["trade_date"] == "2026-06-15"
    assert result.snapshot["source_kind"] == "daily_market_data_csv"
    assert result.candidates.iloc[0]["as_of_date"] == "2026-06-16"
    assert result.candidates.iloc[0]["trade_date"] == "2026-06-15"


def test_momentum_focus_cli_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["momentum-focus"])

    assert args.command == "momentum-focus"
    assert args.data_cache_dir == "D:/codex/daily-market-data"
    assert args.ingest_project_dir == "D:/codex/2026-06-15-exchange-data-ingest"
    assert args.trade_date is None
    assert args.strong_gain_threshold_pct == 5.0
    assert args.board_scope == "main_chinext"
    assert args.outcome_summary_path == "outputs/research/momentum_outcomes_latest/momentum_outcome_summary.csv"
    assert args.name_map_path == "data/processed/stock_name_map.csv"
    assert args.target_horizon == 5


def test_daily_pipeline_records_momentum_focus_when_daily_hub_is_explicit(tmp_path: Path) -> None:
    from quant_etf_lab.daily_pipeline import run_daily_pipeline

    daily_data_dir = tmp_path / "daily-market-data"
    daily_data_dir.mkdir()
    fake_daily_result = SimpleNamespace(
        output_dir=tmp_path / "daily",
        report_path=tmp_path / "daily" / "daily_model_check.md",
        snapshot_path=tmp_path / "daily" / "daily_model_check_snapshot.json",
        snapshot={
            "data_freshness_status": "fresh_enough",
            "action_posture": "monitor",
            "phase2_posture": "core_base_allocator_gate",
            "latest_date": "2026-06-15",
        },
    )
    fake_dashboard_result = SimpleNamespace(
        output_dir=tmp_path / "dashboard",
        snapshot_path=tmp_path / "dashboard" / "latest_dashboard_snapshot.json",
        report_path=tmp_path / "dashboard" / "latest_dashboard.md",
        snapshot={
            "dashboard_posture": "core_base_watch_allocator_gate",
            "sentiment_state": "neutral",
            "sentiment_freshness_status": "fresh_enough",
            "market_cache_status": "fresh_enough",
            "market_cache_latest_date": "2026-06-15",
            "allocator_input_status": "fresh_enough",
            "paper_freshness_status": "missing",
            "trigger_freshness_status": "missing",
            "model_audit_status": "ok",
            "model_audit_walk_forward_action_items": 0,
            "model_audit_walk_forward_resume_candidates": 0,
            "model_audit_walk_forward_archive_review_candidates": 0,
            "promotion_status": "ok",
            "promotion_decision": "wait",
            "promotion_sensitivity_support_count": 0,
            "promotion_sensitivity_group_support_count": 0,
            "promotion_support_group_count": 0,
            "promotion_evidence_support_count": 0,
            "promotion_sensitivity_run_support_count": 0,
            "promotion_min_sensitivity_support": 0,
            "promotion_return_edge": 0.0,
            "promotion_sharpe_edge": 0.0,
            "promotion_drawdown_change": 0.0,
            "promotion_report_path": str(tmp_path / "promotion" / "allocator_promotion_review.md"),
            "pipeline_history_status": "ok",
            "pipeline_history_health_state": "ok",
            "pipeline_history_alert_count": 0,
            "pipeline_history_latest_as_of_date": "2026-06-15",
        },
    )
    fake_risk_budget_result = SimpleNamespace(
        output_dir=tmp_path / "risk",
        snapshot_path=tmp_path / "risk" / "satellite_risk_budget_snapshot.json",
        report_path=tmp_path / "risk" / "satellite_risk_budget_review.md",
        checklist_path=tmp_path / "risk" / "satellite_risk_budget_checklist.csv",
        snapshot={
            "risk_budget_decision": "wait_for_outcome_samples",
            "risk_budget_reason": "No outcome samples yet.",
            "next_action_stage": "monitor",
            "recommended_satellite_budget": 0.0,
            "selected_horizon": "",
            "outcome_ready_horizon_count": 0,
        },
    )
    fake_live_preflight = SimpleNamespace(
        output_dir=tmp_path / "live_preflight",
        snapshot_path=tmp_path / "live_preflight" / "live_preflight_snapshot.json",
        report_path=tmp_path / "live_preflight" / "live_preflight.md",
        checklist_path=tmp_path / "live_preflight" / "live_preflight_checklist.csv",
        snapshot={
            "decision": "hold",
            "broker_connection_status": "not_connected",
            "blocking_items": [],
            "monitor_items": [],
            "pipeline_snapshot_status": "present",
            "pipeline_snapshot_path": str(tmp_path / "pipeline" / "daily_pipeline_snapshot.json"),
            "active_target_count": 0,
            "stock_target_review_action_count": 0,
        },
    )
    fake_alerts = SimpleNamespace(
        output_dir=tmp_path / "alerts",
        report_path=tmp_path / "alerts" / "alerts.md",
        json_path=tmp_path / "alerts" / "alerts.json",
        latest_report_path=tmp_path / "alerts" / "alerts_latest.md",
        payload={
            "alert_level": "info",
            "alert_count": 0,
            "action_stage": "monitor",
            "action_summary": "No blocking alert.",
            "action_item_count": 0,
            "action_required": False,
            "critical_count": 0,
            "warning_count": 0,
            "info_count": 0,
            "alerts": [],
        },
    )
    fake_momentum_result = SimpleNamespace(
        output_dir=tmp_path / "momentum_focus",
        candidates_path=tmp_path / "momentum_focus" / "momentum_focus_candidates.csv",
        snapshot_path=tmp_path / "momentum_focus" / "momentum_focus_snapshot.json",
        report_path=tmp_path / "momentum_focus" / "momentum_focus.md",
        snapshot={
            "status": "ok",
            "candidate_count": 11,
            "limit_up_count": 4,
            "strong_gain_count": 11,
            "excluded_special_treatment_count": 2,
            "source_kind": "daily_market_data_sqlite",
            "trade_date": "2026-06-15",
            "board_scope": "main_chinext",
            "strong_gain_threshold_pct": 7.0,
            "outcome_prior_rows": 30,
            "outcome_prior_matched_count": 11,
            "outcome_summary_rows": 30,
            "name_map_status": "ok",
            "name_map_rows": 120,
            "name_map_path": "data/processed/stock_name_map.csv",
            "broker_action": "none",
            "research_only": True,
        },
    )

    with patch("quant_etf_lab.daily_pipeline.run_daily_model_check", return_value=fake_daily_result), patch(
        "quant_etf_lab.daily_pipeline.run_paper_account", side_effect=RuntimeError("paper offline")
    ), patch("quant_etf_lab.daily_pipeline.run_daily_dashboard", return_value=fake_dashboard_result), patch(
        "quant_etf_lab.daily_pipeline.run_momentum_focus", return_value=fake_momentum_result
    ) as momentum_mock, patch(
        "quant_etf_lab.daily_pipeline.run_satellite_risk_budget_review", return_value=fake_risk_budget_result
    ), patch(
        "quant_etf_lab.daily_pipeline.run_live_preflight", return_value=fake_live_preflight
    ), patch(
        "quant_etf_lab.daily_pipeline.write_daily_alerts", return_value=fake_alerts
    ):
        result = run_daily_pipeline(
            project_root=tmp_path,
            output_dir=tmp_path / "pipeline",
            data_cache_dir=daily_data_dir,
            history_path=None,
            run_history_review=False,
            as_of_date="2026-06-16",
            date_stamp=False,
        )

    momentum_mock.assert_called_once()
    assert momentum_mock.call_args.kwargs["daily_data_dir"] == daily_data_dir
    assert str(momentum_mock.call_args.kwargs["as_of_date"]) == "2026-06-16"
    assert str(momentum_mock.call_args.kwargs["trade_date"]) == "2026-06-15"
    assert result.momentum_focus_result is fake_momentum_result
    assert result.snapshot["momentum_focus_status"] == "ok"
    assert result.snapshot["momentum_focus_candidate_count"] == 11
    assert result.snapshot["momentum_focus_limit_up_count"] == 4
    assert result.snapshot["momentum_focus_excluded_special_treatment_count"] == 2
    assert result.snapshot["momentum_focus_outcome_prior_rows"] == 30
    assert result.snapshot["momentum_focus_outcome_prior_matched_count"] == 11
    assert result.snapshot["momentum_focus_outcome_summary_rows"] == 30
    assert result.snapshot["momentum_focus_name_map_status"] == "ok"
    assert result.snapshot["momentum_focus_name_map_rows"] == 120
    assert result.snapshot["momentum_focus_research_only"] is True
    assert "涨停与强势股研究池" in result.report_path.read_text(encoding="utf-8")
