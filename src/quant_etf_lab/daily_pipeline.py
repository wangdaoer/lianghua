"""After-close daily research pipeline orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .chip_reversal_candidate_outcomes import (
    ChipReversalCandidateOutcomesResult,
    DEFAULT_CANDIDATES_PATH as DEFAULT_CHIP_REVERSAL_CANDIDATES_PATH,
    DEFAULT_DATA_DIR as DEFAULT_CHIP_REVERSAL_OUTCOMES_DATA_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_CHIP_REVERSAL_OUTCOMES_OUTPUT,
    run_chip_reversal_candidate_outcomes,
)
from .dashboard import DailyDashboardResult, run_daily_dashboard
from .daily_alerts import DailyAlertsResult, write_daily_alerts
from .daily_check import DailyModelCheckResult, run_daily_model_check
from .live_shadow import LiveShadowPreflightResult, LiveShadowResult, run_live_shadow, run_live_shadow_preflight
from .market_data_source import DEFAULT_DAILY_MARKET_DATA_DIR, load_market_snapshot_rows
from .momentum_focus import DEFAULT_OUTPUT_DIR as DEFAULT_MOMENTUM_FOCUS_OUTPUT
from .momentum_focus import DEFAULT_OUTCOME_SUMMARY_PATH, MomentumFocusResult, run_momentum_focus
from .paper_account import PaperAccountResult, run_paper_account, write_stock_target_review_decision_template_xlsx
from .pipeline_history import PipelineHistoryReviewResult, run_pipeline_history_review
from .satellite_risk_budget import SatelliteRiskBudgetReviewResult, run_satellite_risk_budget_review
from .satellite_trial_replay import (
    DEFAULT_HORIZONS as DEFAULT_SATELLITE_TRIAL_REPLAY_HORIZONS,
)
from .satellite_trial_replay import DEFAULT_OUTPUT_DIR as DEFAULT_SATELLITE_TRIAL_REPLAY_OUTPUT
from .satellite_trial_replay import SatelliteTrialReplayResult, run_satellite_trial_replay
from .live_preflight import LivePreflightResult, run_live_preflight
from .trading_gate import build_a_share_trading_gate


DEFAULT_PIPELINE_OUTPUT = Path("outputs/research/daily_pipeline_latest")
DEFAULT_PIPELINE_BASE = Path("outputs/research/daily_pipeline")
DEFAULT_DAILY_CHECK_BASE = Path("outputs/research/daily_model_check")
DEFAULT_PAPER_ACCOUNT_BASE = Path("outputs/research/paper_account")
DEFAULT_DASHBOARD_BASE = Path("outputs/research/daily_dashboard")
DEFAULT_MOMENTUM_FOCUS_BASE = Path("outputs/research/momentum_focus")
DEFAULT_CHIP_REVERSAL_OUTCOMES_BASE = Path("outputs/research/chip_reversal_candidate_outcomes")
DEFAULT_PIPELINE_HISTORY = Path("outputs/research/daily_pipeline_history.csv")
DEFAULT_HISTORY_REVIEW_OUTPUT = Path("outputs/research/pipeline_history_review_latest")
DEFAULT_HISTORY_REVIEW_BASE = Path("outputs/research/pipeline_history_review")
DEFAULT_SATELLITE_RISK_BUDGET_OUTPUT = Path("outputs/research/satellite_risk_budget_review_latest")
DEFAULT_SATELLITE_RISK_BUDGET_BASE = Path("outputs/research/satellite_risk_budget_review")
DEFAULT_SATELLITE_TRIAL_REPLAY_BASE = Path("outputs/research/satellite_trial_replay")
DEFAULT_LIVE_SHADOW_OUTPUT = Path("outputs/research/live_shadow_latest")
DEFAULT_LIVE_SHADOW_BASE = Path("outputs/research/live_shadow")
DEFAULT_LIVE_PRE_FLIGHT_OUTPUT = Path("outputs/research/live_preflight_latest")
DEFAULT_LIVE_PRE_FLIGHT_BASE = Path("outputs/research/live_preflight")
DEFAULT_LIVE_SHADOW_PREFLIGHT_OUTPUT = Path("outputs/research/live_shadow_preflight_latest")
DEFAULT_LIVE_SHADOW_PREFLIGHT_BASE = Path("outputs/research/live_shadow_preflight")

HISTORY_COLUMNS = [
    "generated_at",
    "as_of_date",
    "local_equity_latest_date",
    "data_freshness_status",
    "trading_day_gate_status",
    "after_close_data_status",
    "trading_day_gate_action",
    "is_a_share_trading_day",
    "phase2_posture",
    "action_posture",
    "dashboard_posture",
    "market_cache_status",
    "allocator_input_status",
    "paper_freshness_status",
    "sentiment_state",
    "sentiment_freshness_status",
    "trigger_freshness_status",
    "momentum_focus_status",
    "momentum_focus_market_data_quality_status",
    "momentum_focus_volume_coverage_ratio",
    "momentum_focus_turnover_coverage_ratio",
    "momentum_focus_candidate_count",
    "momentum_focus_limit_up_count",
    "momentum_focus_strong_gain_count",
    "momentum_focus_excluded_special_treatment_count",
    "momentum_focus_excluded_by_review_required_count",
    "momentum_focus_source_kind",
    "momentum_focus_trade_date",
    "momentum_focus_outcome_prior_rows",
    "momentum_focus_outcome_prior_matched_count",
    "momentum_focus_outcome_summary_rows",
    "momentum_focus_name_map_status",
    "momentum_focus_name_map_rows",
    "chip_reversal_candidate_outcomes_status",
    "chip_reversal_candidate_outcomes_candidate_count",
    "chip_reversal_candidate_outcomes_ready_count",
    "chip_reversal_candidate_outcomes_pending_count",
    "chip_reversal_candidate_outcomes_readiness_status",
    "chip_reversal_candidate_outcomes_analysis_status",
    "chip_reversal_candidate_outcomes_next_review_horizon",
    "chip_reversal_candidate_outcomes_next_review_reason",
    "chip_reversal_candidate_outcomes_sample_warning",
    "chip_reversal_candidate_outcomes_latest_market_trade_date",
    "chip_reversal_candidate_outcomes_promotion_gate_status",
    "chip_reversal_candidate_outcomes_promotion_gate_reasons",
    "chip_reversal_candidate_outcomes_market_source_kind",
    "chip_reversal_candidate_outcomes_market_trade_date",
    "chip_reversal_candidate_outcomes_broker_action",
    "chip_reversal_candidate_outcomes_research_only",
    "chip_reversal_candidate_outcomes_report_path",
    "chip_reversal_candidate_outcomes_snapshot_path",
    "chip_reversal_candidate_outcomes_error",
    "model_audit_status",
    "model_audit_duplicate_config_groups",
    "model_audit_root_configs_without_extends",
    "model_audit_walk_forward_action_items",
    "model_audit_walk_forward_resume_candidates",
    "model_audit_walk_forward_archive_review_candidates",
    "model_audit_top_action",
    "promotion_status",
    "promotion_decision",
    "promotion_candidate_passes_headline_gate",
    "promotion_sensitivity_support_count",
    "promotion_sensitivity_group_support_count",
    "promotion_support_group_count",
    "promotion_evidence_support_count",
    "promotion_sensitivity_run_support_count",
    "promotion_min_sensitivity_support",
    "promotion_return_edge",
    "promotion_sharpe_edge",
    "promotion_drawdown_change",
    "promotion_report_path",
    "paper_latest_date",
    "paper_latest_regime",
    "paper_latest_candidate",
    "paper_latest_core_weight",
    "paper_latest_satellite_weight",
    "paper_latest_cash_weight",
    "paper_target_holdings_report_path",
    "paper_target_holdings_path",
    "paper_stock_targets_report_path",
    "paper_stock_targets_path",
    "paper_stock_target_review_report_path",
    "paper_stock_target_review_path",
    "paper_stock_target_review_actions_report_path",
    "paper_stock_target_review_actions_path",
    "paper_stock_target_review_assistant_report_path",
    "paper_stock_target_review_assistant_path",
    "paper_stock_target_review_decision_template_report_path",
    "paper_stock_target_review_decision_template_path",
    "paper_stock_target_review_decision_template_xlsx_path",
    "paper_stock_target_review_outcomes_report_path",
    "paper_stock_target_review_outcomes_path",
    "paper_stock_target_review_outcomes_history_report_path",
    "paper_stock_target_review_outcomes_history_path",
    "paper_stock_target_review_outcomes_history_snapshot_path",
    "paper_stock_target_review_outcome_analysis_report_path",
    "paper_stock_target_review_outcome_analysis_path",
    "paper_stock_target_review_outcome_calendar_report_path",
    "paper_stock_target_review_outcome_calendar_path",
    "paper_stock_target_review_outcome_due_report_path",
    "paper_stock_target_review_outcome_due_path",
    "paper_stock_target_review_notes_path",
    "paper_stock_target_review_notes_snapshot_path",
    "paper_stock_target_review_required_count",
    "paper_stock_target_review_monitor_count",
    "paper_stock_target_review_trigger_count",
    "paper_stock_target_review_drawdown_count",
    "paper_stock_target_review_suppressed_layer_count",
    "paper_stock_target_review_watch_count",
    "paper_stock_target_review_manual_note_count",
    "paper_stock_target_review_unreviewed_count",
    "paper_stock_target_review_required_unreviewed_count",
    "paper_stock_target_review_manual_pending_count",
    "paper_stock_target_review_manual_reviewed_count",
    "paper_stock_target_review_manual_watch_count",
    "paper_stock_target_review_manual_resolved_count",
    "paper_stock_target_review_manual_exclude_candidate_count",
    "paper_stock_target_review_manual_other_status_count",
    "paper_stock_target_review_action_count",
    "paper_stock_target_review_action_pending_model_count",
    "paper_stock_target_review_action_manual_watch_count",
    "paper_stock_target_review_action_manual_exclusion_count",
    "paper_stock_target_review_action_status_normalization_count",
    "paper_stock_target_review_action_next_review_due_count",
    "paper_stock_target_review_assistant_count",
    "paper_stock_target_review_assistant_pending_count",
    "paper_stock_target_review_assistant_price_ok_count",
    "paper_stock_target_review_assistant_price_stale_count",
    "paper_stock_target_review_assistant_price_missing_count",
    "paper_stock_target_review_decision_template_count",
    "paper_stock_target_review_decision_template_blank_status_count",
    "paper_stock_target_review_outcome_row_count",
    "paper_stock_target_review_outcome_complete_count",
    "paper_stock_target_review_outcome_partial_count",
    "paper_stock_target_review_outcome_pending_count",
    "paper_stock_target_review_outcome_missing_entry_price_count",
    "paper_stock_target_review_outcomes_history_row_count",
    "paper_stock_target_review_outcomes_history_updated_row_count",
    "paper_stock_target_review_outcomes_history_complete_count",
    "paper_stock_target_review_outcomes_history_partial_count",
    "paper_stock_target_review_outcomes_history_pending_count",
    "paper_stock_target_review_outcomes_history_missing_entry_price_count",
    "paper_stock_target_review_outcomes_history_latest_review_date",
    "paper_stock_target_review_outcome_analysis_status",
    "paper_stock_target_review_outcome_analysis_row_count",
    "paper_stock_target_review_outcome_analysis_min_evaluable",
    "paper_stock_target_review_outcome_analysis_min_group_evaluable",
    "paper_stock_target_review_outcome_analysis_ready_horizon_count",
    "paper_stock_target_review_outcome_analysis_sample_warning",
    "paper_stock_target_review_outcome_calendar_row_count",
    "paper_stock_target_review_outcome_calendar_ready_count",
    "paper_stock_target_review_outcome_calendar_pending_count",
    "paper_stock_target_review_outcome_calendar_due_count",
    "paper_stock_target_review_outcome_calendar_due_pending_count",
    "paper_stock_target_review_outcome_calendar_next_action_date",
    "paper_stock_target_review_outcome_calendar_next_action_horizon",
    "paper_stock_target_review_outcome_calendar_next_due_date",
    "paper_stock_target_review_outcome_calendar_next_due_horizon",
    "paper_stock_target_review_outcome_due_status",
    "paper_stock_target_review_outcome_due_row_count",
    "paper_stock_target_review_outcome_due_pending_count",
    "paper_stock_target_review_outcome_due_next_date",
    "paper_stock_target_review_outcome_due_next_horizon",
    "paper_stock_target_review_outcome_maturity_next_1d_date",
    "paper_stock_target_review_outcome_maturity_next_5d_date",
    "paper_stock_target_review_outcome_maturity_next_10d_date",
    "paper_stock_target_review_outcome_maturity_next_20d_date",
    "paper_stock_target_review_outcome_analysis_evaluable_1d_count",
    "paper_stock_target_review_outcome_analysis_evaluable_5d_count",
    "paper_stock_target_review_outcome_analysis_evaluable_10d_count",
    "paper_stock_target_review_outcome_analysis_evaluable_20d_count",
    "satellite_risk_budget_status",
    "satellite_risk_budget_decision",
    "satellite_risk_budget_reason",
    "satellite_risk_budget_next_action_stage",
    "satellite_risk_budget_recommended_satellite_weight",
    "satellite_risk_budget_selected_horizon",
    "satellite_risk_budget_ready_horizon_count",
    "satellite_risk_budget_report_path",
    "satellite_risk_budget_snapshot_path",
    "satellite_risk_budget_checklist_path",
    "satellite_trial_rule_count",
    "satellite_trial_rules_path",
    "satellite_trial_rules_json_path",
    "satellite_trial_replay_status",
    "satellite_trial_replay_matched_event_count",
    "satellite_trial_replay_best_horizon",
    "satellite_trial_replay_best_avg_return_edge",
    "satellite_trial_replay_best_win_rate_edge",
    "satellite_trial_replay_report_path",
    "satellite_trial_replay_summary_path",
    "satellite_trial_replay_matches_path",
    "satellite_trial_replay_snapshot_path",
    "live_shadow_status",
    "live_shadow_error",
    "live_shadow_output_dir",
    "live_shadow_snapshot_path",
    "live_shadow_report_path",
    "live_shadow_orders_path",
    "live_shadow_reconcile_path",
    "live_shadow_holdings_path",
    "live_shadow_prices_path",
    "live_shadow_trade_plan_status",
    "live_shadow_broker_action",
    "live_shadow_research_only",
    "live_shadow_current_equity",
    "live_shadow_cash",
    "live_shadow_order_count",
    "live_shadow_buy_count",
    "live_shadow_sell_count",
    "live_shadow_planned_buy_amount",
    "live_shadow_planned_sell_amount",
    "live_shadow_estimated_cash_after_orders",
    "live_shadow_target_gross_weight",
    "live_shadow_warning_count",
    "live_shadow_preflight_status",
    "live_shadow_preflight_output_dir",
    "live_shadow_preflight_decision",
    "live_shadow_preflight_snapshot_path",
    "live_shadow_preflight_report_path",
    "live_shadow_preflight_blockers_count",
    "live_shadow_preflight_error",
    "live_preflight_status",
    "live_preflight_output_dir",
    "live_preflight_decision",
    "live_preflight_broker_connection_status",
    "live_preflight_snapshot_path",
    "live_preflight_report_path",
    "live_preflight_checklist_path",
    "live_preflight_blocking_items_count",
    "live_preflight_monitor_items_count",
    "live_preflight_pipeline_snapshot_status",
    "live_preflight_pipeline_snapshot_path",
    "live_preflight_active_target_count",
    "live_preflight_stock_target_review_action_count",
    "live_preflight_live_shadow_review_decisions_path",
    "live_preflight_live_shadow_review_decision_count",
    "live_preflight_live_shadow_review_blocking_decision_count",
    "live_preflight_live_shadow_review_monitor_decision_count",
    "live_preflight_live_shadow_review_unknown_decision_count",
    "live_preflight_error",
    "daily_preflight_skipped",
    "paper_stock_target_review_drawdown_threshold",
    "paper_stock_target_review_watch_drawdown_threshold",
    "paper_stock_target_review_loss_attention_threshold",
    "paper_stock_target_review_warning_only_after_close",
    "paper_account_status",
    "paper_account_error",
    "paper_final_equity",
    "paper_total_return",
    "paper_max_drawdown",
    "paper_current_drawdown",
    "paper_sharpe",
    "paper_audit_event_count",
    "pipeline_next_step_stage",
    "pipeline_next_step_action",
    "pipeline_next_step_reason",
    "pipeline_blocker",
    "pipeline_user_intervention_required",
    "pipeline_report_path",
    "daily_check_report_path",
    "paper_account_report_path",
    "dashboard_report_path",
    "alert_level",
    "alert_count",
    "alert_action_stage",
    "alert_action_summary",
    "alert_action_item_count",
    "alert_action_required",
    "alert_critical_count",
    "alert_warning_count",
    "alert_info_count",
    "alerts_report_path",
    "alerts_json_path",
    "latest_alerts_path",
    "history_review_alert_refresh_status",
    "history_review_alert_refresh_error",
    "history_review_latest_alert_level",
    "history_review_latest_alert_action_stage",
    "history_review_latest_alert_count",
    "dashboard_alert_refresh_status",
    "dashboard_alert_refresh_error",
    "dashboard_history_refresh_status",
    "dashboard_history_refresh_error",
    "dashboard_history_refresh_report_path",
    "dashboard_history_refresh_snapshot_path",
    "dashboard_pipeline_history_status",
    "dashboard_pipeline_history_health_state",
    "dashboard_pipeline_history_alert_count",
    "dashboard_pipeline_history_latest_as_of_date",
    "dashboard_pipeline_history_latest_satellite_risk_budget_decision",
    "dashboard_pipeline_history_latest_satellite_risk_budget_recommended_weight",
]


@dataclass(frozen=True)
class DailyPipelineResult:
    output_dir: Path
    snapshot_path: Path
    report_path: Path
    history_path: Path | None
    history_review_result: PipelineHistoryReviewResult | None
    satellite_risk_budget_result: SatelliteRiskBudgetReviewResult | None
    satellite_trial_replay_result: SatelliteTrialReplayResult | None
    live_shadow_result: LiveShadowResult | None
    live_shadow_preflight_result: LiveShadowPreflightResult | None
    live_preflight_result: LivePreflightResult | None
    momentum_focus_result: MomentumFocusResult | None
    chip_reversal_candidate_outcomes_result: ChipReversalCandidateOutcomesResult | None
    alerts_result: DailyAlertsResult
    daily_check_result: DailyModelCheckResult
    paper_account_result: PaperAccountResult
    dashboard_result: DailyDashboardResult
    snapshot: dict[str, Any]


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = Path(path) if path is not None else default
    return raw if raw.is_absolute() else project_root / raw


def _parse_int_values(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, int):
        raw_values = [value]
    else:
        raw_values = list(value)
    parsed = [int(item) for item in raw_values]
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"Expected positive integer values, got: {value!r}")
    return parsed


def _should_run_momentum_focus(data_cache_dir: str | Path | None) -> bool:
    if data_cache_dir is None:
        return False
    path = Path(data_cache_dir)
    normalized = path.as_posix().rstrip("/").lower()
    default = DEFAULT_DAILY_MARKET_DATA_DIR.as_posix().rstrip("/").lower()
    return path.name.lower() == "daily-market-data" or normalized == default


def _momentum_focus_context(result: MomentumFocusResult) -> dict[str, Any]:
    snapshot = result.snapshot
    return {
        "momentum_focus_status": snapshot.get("status"),
        "momentum_focus_market_data_quality_status": snapshot.get("market_data_quality_status"),
        "momentum_focus_volume_positive_count": snapshot.get("volume_positive_count"),
        "momentum_focus_turnover_positive_count": snapshot.get("turnover_positive_count"),
        "momentum_focus_volume_coverage_ratio": snapshot.get("volume_coverage_ratio"),
        "momentum_focus_turnover_coverage_ratio": snapshot.get("turnover_coverage_ratio"),
        "momentum_focus_output_dir": str(result.output_dir),
        "momentum_focus_report_path": str(result.report_path),
        "momentum_focus_candidates_path": str(result.candidates_path),
        "momentum_focus_snapshot_path": str(result.snapshot_path),
        "momentum_focus_candidate_count": snapshot.get("candidate_count"),
        "momentum_focus_limit_up_count": snapshot.get("limit_up_count"),
        "momentum_focus_strong_gain_count": snapshot.get("strong_gain_count"),
        "momentum_focus_excluded_special_treatment_count": snapshot.get("excluded_special_treatment_count"),
        "momentum_focus_excluded_by_review_required_count": snapshot.get("excluded_by_review_required_count"),
        "momentum_focus_source_kind": snapshot.get("source_kind"),
        "momentum_focus_trade_date": snapshot.get("trade_date"),
        "momentum_focus_board_scope": snapshot.get("board_scope"),
        "momentum_focus_target_horizon": snapshot.get("target_horizon"),
        "momentum_focus_strong_gain_threshold_pct": snapshot.get("strong_gain_threshold_pct"),
        "momentum_focus_outcome_summary_path": snapshot.get("outcome_summary_path"),
        "momentum_focus_outcome_prior_rows": snapshot.get("outcome_prior_rows"),
        "momentum_focus_outcome_prior_matched_count": snapshot.get("outcome_prior_matched_count"),
        "momentum_focus_outcome_summary_rows": snapshot.get("outcome_summary_rows"),
        "momentum_focus_name_map_status": snapshot.get("name_map_status"),
        "momentum_focus_name_map_rows": snapshot.get("name_map_rows"),
        "momentum_focus_name_map_path": snapshot.get("name_map_path"),
        "momentum_focus_broker_action": snapshot.get("broker_action"),
        "momentum_focus_research_only": snapshot.get("research_only"),
        "momentum_focus_error": None,
    }


def _chip_reversal_candidate_outcomes_context(result: ChipReversalCandidateOutcomesResult) -> dict[str, Any]:
    snapshot = result.snapshot
    return {
        "chip_reversal_candidate_outcomes_status": snapshot.get("status"),
        "chip_reversal_candidate_outcomes_output_dir": str(result.output_dir),
        "chip_reversal_candidate_outcomes_outcomes_path": str(result.outcomes_path),
        "chip_reversal_candidate_outcomes_summary_path": str(result.summary_path),
        "chip_reversal_candidate_outcomes_group_summary_path": str(result.group_summary_path),
        "chip_reversal_candidate_outcomes_snapshot_path": str(result.snapshot_path),
        "chip_reversal_candidate_outcomes_report_path": str(result.report_path),
        "chip_reversal_candidate_outcomes_candidate_count": snapshot.get("candidate_count"),
        "chip_reversal_candidate_outcomes_ready_count": snapshot.get("ready_outcome_count"),
        "chip_reversal_candidate_outcomes_pending_count": snapshot.get("pending_outcome_count"),
        "chip_reversal_candidate_outcomes_readiness_status": snapshot.get("outcome_readiness_status"),
        "chip_reversal_candidate_outcomes_analysis_status": snapshot.get("outcome_analysis_status"),
        "chip_reversal_candidate_outcomes_ready_horizons": snapshot.get("outcome_ready_horizons"),
        "chip_reversal_candidate_outcomes_pending_horizons": snapshot.get("outcome_pending_horizons"),
        "chip_reversal_candidate_outcomes_next_review_horizon": snapshot.get("next_outcome_review_horizon"),
        "chip_reversal_candidate_outcomes_next_review_reason": snapshot.get("next_outcome_review_reason"),
        "chip_reversal_candidate_outcomes_sample_warning": snapshot.get("outcome_sample_warning"),
        "chip_reversal_candidate_outcomes_latest_market_trade_date": snapshot.get("latest_available_market_trade_date"),
        "chip_reversal_candidate_outcomes_market_source_kind": snapshot.get("market_source_kind"),
        "chip_reversal_candidate_outcomes_market_trade_date": snapshot.get("market_trade_date"),
        "chip_reversal_candidate_outcomes_promotion_gate_status": snapshot.get("promotion_gate_status"),
        "chip_reversal_candidate_outcomes_promotion_gate_reasons": snapshot.get("promotion_gate_reasons"),
        "chip_reversal_candidate_outcomes_broker_action": snapshot.get("broker_action"),
        "chip_reversal_candidate_outcomes_research_only": snapshot.get("research_only"),
        "chip_reversal_candidate_outcomes_error": None,
    }


def _momentum_focus_trade_date(
    as_of: date,
    daily_snapshot: dict[str, Any],
    dashboard_snapshot: dict[str, Any],
) -> date:
    for key in ("market_cache_latest_date", "local_equity_latest_date"):
        parsed = _parse_date(dashboard_snapshot.get(key))
        if parsed is not None:
            return parsed
    for key in ("local_equity_latest_date", "latest_date"):
        parsed = _parse_date(daily_snapshot.get(key))
        if parsed is not None:
            return parsed
    return as_of


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) == 8 and text.isdigit():
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        if pd.isna(parsed):
            return None
        return pd.Timestamp(parsed).date()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).date()


def _failed_paper_account_result(output_dir: Path, as_of: date | None, error: str) -> PaperAccountResult:
    output_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = output_dir / "ledger.csv"
    audit_path = output_dir / "rebalance_audit.csv"
    monthly_returns_path = output_dir / "monthly_returns.csv"
    target_holdings_path = output_dir / "target_holdings.csv"
    target_holdings_json_path = output_dir / "target_holdings.json"
    target_holdings_report_path = output_dir / "target_holdings.md"
    stock_targets_path = output_dir / "stock_targets.csv"
    stock_targets_json_path = output_dir / "stock_targets.json"
    stock_targets_report_path = output_dir / "stock_targets.md"
    stock_target_review_path = output_dir / "stock_target_review.csv"
    stock_target_review_json_path = output_dir / "stock_target_review.json"
    stock_target_review_report_path = output_dir / "stock_target_review.md"
    stock_target_review_notes_path = output_dir / "stock_target_review_notes.csv"
    stock_target_review_notes_snapshot_path = output_dir / "stock_target_review_notes.csv"
    stock_target_review_actions_path = output_dir / "stock_target_review_actions.csv"
    stock_target_review_actions_json_path = output_dir / "stock_target_review_actions.json"
    stock_target_review_actions_report_path = output_dir / "stock_target_review_actions.md"
    stock_target_review_assistant_path = output_dir / "stock_target_review_assistant.csv"
    stock_target_review_assistant_json_path = output_dir / "stock_target_review_assistant.json"
    stock_target_review_assistant_report_path = output_dir / "stock_target_review_assistant.md"
    stock_target_review_decision_template_path = output_dir / "stock_target_review_decision_template.csv"
    stock_target_review_decision_template_json_path = output_dir / "stock_target_review_decision_template.json"
    stock_target_review_decision_template_report_path = output_dir / "stock_target_review_decision_template.md"
    stock_target_review_decision_template_xlsx_path = output_dir / "stock_target_review_decision_template.xlsx"
    stock_target_review_outcomes_path = output_dir / "stock_target_review_outcomes.csv"
    stock_target_review_outcomes_json_path = output_dir / "stock_target_review_outcomes.json"
    stock_target_review_outcomes_report_path = output_dir / "stock_target_review_outcomes.md"
    stock_target_review_outcomes_history_path = output_dir / "stock_target_review_outcomes_history.csv"
    stock_target_review_outcomes_history_snapshot_path = output_dir / "stock_target_review_outcomes_history.csv"
    stock_target_review_outcomes_history_json_path = output_dir / "stock_target_review_outcomes_history.json"
    stock_target_review_outcomes_history_report_path = output_dir / "stock_target_review_outcomes_history.md"
    stock_target_review_outcome_analysis_path = output_dir / "stock_target_review_outcome_analysis.csv"
    stock_target_review_outcome_analysis_json_path = output_dir / "stock_target_review_outcome_analysis.json"
    stock_target_review_outcome_analysis_report_path = output_dir / "stock_target_review_outcome_analysis.md"
    stock_target_review_outcome_calendar_path = output_dir / "stock_target_review_outcome_calendar.csv"
    stock_target_review_outcome_calendar_json_path = output_dir / "stock_target_review_outcome_calendar.json"
    stock_target_review_outcome_calendar_report_path = output_dir / "stock_target_review_outcome_calendar.md"
    stock_target_review_outcome_due_path = output_dir / "stock_target_review_outcome_due.csv"
    stock_target_review_outcome_due_json_path = output_dir / "stock_target_review_outcome_due.json"
    stock_target_review_outcome_due_report_path = output_dir / "stock_target_review_outcome_due.md"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "paper_account.md"

    empty_frame = pd.DataFrame()
    for path in [
        ledger_path,
        audit_path,
        monthly_returns_path,
        target_holdings_path,
        stock_targets_path,
        stock_target_review_path,
        stock_target_review_notes_path,
        stock_target_review_actions_path,
        stock_target_review_assistant_path,
        stock_target_review_decision_template_path,
        stock_target_review_outcomes_path,
        stock_target_review_outcomes_history_path,
        stock_target_review_outcome_analysis_path,
        stock_target_review_outcome_calendar_path,
        stock_target_review_outcome_due_path,
    ]:
        empty_frame.to_csv(path, index=False, encoding="utf-8")

    failure_message = (
        "# Paper account step failed\n\n"
        "Daily pipeline fallback output was generated so later stages can still run.\n\n"
        f"Error: {error}\n"
    )
    report_payload = "# Paper account unavailable\n\nFallback execution mode.\n"
    for path in [
        report_path,
        target_holdings_report_path,
        stock_targets_report_path,
        stock_target_review_report_path,
        stock_target_review_actions_report_path,
        stock_target_review_assistant_report_path,
        stock_target_review_decision_template_report_path,
        stock_target_review_outcomes_report_path,
        stock_target_review_outcomes_history_report_path,
        stock_target_review_outcome_analysis_report_path,
        stock_target_review_outcome_calendar_report_path,
        stock_target_review_outcome_due_report_path,
    ]:
        path.write_text(f"{report_payload}{failure_message}", encoding="utf-8")

    for path in [
        target_holdings_json_path,
        stock_targets_json_path,
        stock_target_review_json_path,
        stock_target_review_actions_json_path,
        stock_target_review_assistant_json_path,
        stock_target_review_decision_template_json_path,
        stock_target_review_outcomes_json_path,
        stock_target_review_outcomes_history_json_path,
        stock_target_review_outcome_analysis_json_path,
        stock_target_review_outcome_calendar_json_path,
        stock_target_review_outcome_due_json_path,
    ]:
        path.write_text(json.dumps({"status": "unavailable", "error": error}, ensure_ascii=False, indent=2), encoding="utf-8")

    base_payload: dict[str, Any] = {
        "latest_date": as_of.isoformat() if as_of is not None else None,
        "latest_regime": "failed",
        "latest_candidate": None,
        "latest_core_weight": 0.0,
        "latest_satellite_weight": 0.0,
        "latest_cash_weight": 1.0,
        "final_equity": None,
        "total_return": None,
        "cagr": None,
        "max_drawdown": None,
        "current_drawdown": 0.0,
        "sharpe": None,
        "audit_event_count": 0,
        "total_estimated_fee": 0.0,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paper_account_error": str(error),
        "paper_account_status": "failed",
        "stock_target_review_decision_template_count": 0,
        "stock_target_review_decision_template_blank_status_count": 0,
    }
    target_holdings_payload = {
        "target_holding_count": 0,
        "active_target_count": 0,
        "broker_action": "failed",
    }
    stock_targets_payload = {
        "stock_target_count": 0,
        "source_stock_target_count": 0,
        "active_stock_target_count": 0,
        "suppressed_stock_count": 0,
        "review_required_excluded_count": 0,
        "stock_tracking_max_market_cap_yi": None,
        "stock_tracking_excluded_large_market_cap_count": 0,
        "stock_tracking_allowed_count": 0,
        "stock_tracking_market_cap_missing_count": 0,
        "stock_market_cap_cache_status": "missing",
        "stock_market_cap_cache_row_count": 0,
        "stock_market_cap_cache_latest_snapshot_date": None,
        "stock_market_cap_cache_updated_at": None,
        "stock_market_cap_path": None,
        "stock_tracking_merged_count": 0,
    }
    stock_target_review_payload = {
        "review_required_count": 0,
        "monitor_count": 0,
        "trigger_review_count": 0,
        "drawdown_review_count": 0,
        "suppressed_layer_review_count": 0,
        "watch_review_count": 0,
        "manual_note_count": 0,
        "unreviewed_count": 0,
        "review_required_unreviewed_count": 0,
        "manual_pending_count": 0,
        "manual_reviewed_count": 0,
        "manual_watch_count": 0,
        "manual_resolved_count": 0,
        "manual_exclude_candidate_count": 0,
        "manual_other_status_count": 0,
        "drawdown_threshold": None,
        "watch_drawdown_threshold": None,
        "loss_attention_threshold": None,
        "gain_attention_threshold": None,
    }
    stock_target_review_actions_payload = {
        "action_count": 0,
        "review_required_pending_count": 0,
        "manual_watch_followup_count": 0,
        "manual_exclusion_candidate_count": 0,
        "manual_status_normalization_count": 0,
        "manual_next_review_due_count": 0,
    }
    stock_target_review_assistant_payload = {
        "assistant_row_count": 0,
        "review_required_pending_count": 0,
        "price_history_ok_count": 0,
        "price_history_stale_count": 0,
        "price_history_missing_count": 0,
    }
    stock_target_review_decision_template_payload = {
        "status": "failed",
        "latest_date": as_of.isoformat() if as_of is not None else None,
        "decision_template_row_count": 0,
        "blank_manual_status_count": 0,
        "source_assistant_row_count": 0,
        "allowed_manual_statuses": ["reviewed", "watch", "resolved", "exclude_candidate", "unreviewed"],
        "research_only": True,
        "broker_action": "none",
        "paper_account_error": str(error),
        "decision_template_xlsx_path": str(stock_target_review_decision_template_xlsx_path),
    }
    write_stock_target_review_decision_template_xlsx(
        pd.DataFrame(columns=[]),
        stock_target_review_decision_template_payload,
        stock_target_review_decision_template_xlsx_path,
    )
    stock_target_review_outcomes_payload = {
        "outcome_row_count": 0,
        "complete_count": 0,
        "partial_count": 0,
        "pending_count": 0,
        "missing_entry_price_count": 0,
    }
    stock_target_review_outcomes_history_payload = {
        "history_row_count": 0,
        "history_updated_row_count": 0,
        "history_complete_count": 0,
        "history_partial_count": 0,
        "history_pending_count": 0,
        "history_missing_entry_price_count": 0,
        "history_latest_review_date": None,
    }
    stock_target_review_outcome_analysis_payload = {
        "analysis_status": "failed",
        "analysis_row_count": 0,
        "min_evaluable": None,
        "min_group_evaluable": None,
        "ready_horizon_count": 0,
        "sample_warning": "paper-account-step-failed",
        "maturity_forecast": {},
        "total_evaluable_by_horizon": {},
    }
    stock_target_review_outcome_calendar_payload = {
        "calendar_row_count": 0,
        "calendar_ready_count": 0,
        "calendar_pending_count": 0,
        "calendar_due_count": 0,
        "calendar_due_pending_count": 0,
        "next_action_date": None,
        "next_action_horizon": None,
        "next_due_date": None,
        "next_due_horizon": None,
    }
    stock_target_review_outcome_due_payload = {
        "due_status": "failed",
        "due_row_count": 0,
        "due_pending_count": 0,
        "next_due_date": None,
        "next_due_horizon": None,
    }
    metrics_path.write_text(json.dumps(base_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return PaperAccountResult(
        output_dir=output_dir,
        ledger_path=ledger_path,
        audit_path=audit_path,
        monthly_returns_path=monthly_returns_path,
        target_holdings_path=target_holdings_path,
        target_holdings_json_path=target_holdings_json_path,
        target_holdings_report_path=target_holdings_report_path,
        stock_targets_path=stock_targets_path,
        stock_targets_json_path=stock_targets_json_path,
        stock_targets_report_path=stock_targets_report_path,
        stock_target_review_path=stock_target_review_path,
        stock_target_review_json_path=stock_target_review_json_path,
        stock_target_review_report_path=stock_target_review_report_path,
        stock_target_review_notes_path=stock_target_review_notes_path,
        stock_target_review_notes_snapshot_path=stock_target_review_notes_snapshot_path,
        stock_target_review_actions_path=stock_target_review_actions_path,
        stock_target_review_actions_json_path=stock_target_review_actions_json_path,
        stock_target_review_actions_report_path=stock_target_review_actions_report_path,
        stock_target_review_assistant_path=stock_target_review_assistant_path,
        stock_target_review_assistant_json_path=stock_target_review_assistant_json_path,
        stock_target_review_assistant_report_path=stock_target_review_assistant_report_path,
        stock_target_review_decision_template_path=stock_target_review_decision_template_path,
        stock_target_review_decision_template_json_path=stock_target_review_decision_template_json_path,
        stock_target_review_decision_template_report_path=stock_target_review_decision_template_report_path,
        stock_target_review_decision_template_xlsx_path=stock_target_review_decision_template_xlsx_path,
        stock_target_review_outcomes_path=stock_target_review_outcomes_path,
        stock_target_review_outcomes_json_path=stock_target_review_outcomes_json_path,
        stock_target_review_outcomes_report_path=stock_target_review_outcomes_report_path,
        stock_target_review_outcomes_history_path=stock_target_review_outcomes_history_path,
        stock_target_review_outcomes_history_snapshot_path=stock_target_review_outcomes_history_snapshot_path,
        stock_target_review_outcomes_history_json_path=stock_target_review_outcomes_history_json_path,
        stock_target_review_outcomes_history_report_path=stock_target_review_outcomes_history_report_path,
        stock_target_review_outcome_analysis_path=stock_target_review_outcome_analysis_path,
        stock_target_review_outcome_analysis_json_path=stock_target_review_outcome_analysis_json_path,
        stock_target_review_outcome_analysis_report_path=stock_target_review_outcome_analysis_report_path,
        stock_target_review_outcome_calendar_path=stock_target_review_outcome_calendar_path,
        stock_target_review_outcome_calendar_json_path=stock_target_review_outcome_calendar_json_path,
        stock_target_review_outcome_calendar_report_path=stock_target_review_outcome_calendar_report_path,
        stock_target_review_outcome_due_path=stock_target_review_outcome_due_path,
        stock_target_review_outcome_due_json_path=stock_target_review_outcome_due_json_path,
        stock_target_review_outcome_due_report_path=stock_target_review_outcome_due_report_path,
        metrics_path=metrics_path,
        report_path=report_path,
        ledger=empty_frame,
        audit=empty_frame,
        target_holdings=empty_frame,
        target_holdings_payload=target_holdings_payload,
        stock_targets=empty_frame,
        stock_targets_payload=stock_targets_payload,
        stock_target_review=empty_frame,
        stock_target_review_payload=stock_target_review_payload,
        stock_target_review_actions=empty_frame,
        stock_target_review_actions_payload=stock_target_review_actions_payload,
        stock_target_review_assistant=empty_frame,
        stock_target_review_assistant_payload=stock_target_review_assistant_payload,
        stock_target_review_decision_template=empty_frame,
        stock_target_review_decision_template_payload=stock_target_review_decision_template_payload,
        stock_target_review_outcomes=empty_frame,
        stock_target_review_outcomes_payload=stock_target_review_outcomes_payload,
        stock_target_review_outcomes_history=empty_frame,
        stock_target_review_outcomes_history_payload=stock_target_review_outcomes_history_payload,
        stock_target_review_outcome_analysis=empty_frame,
        stock_target_review_outcome_analysis_payload=stock_target_review_outcome_analysis_payload,
        stock_target_review_outcome_calendar=empty_frame,
        stock_target_review_outcome_calendar_payload=stock_target_review_outcome_calendar_payload,
        stock_target_review_outcome_due=empty_frame,
        stock_target_review_outcome_due_payload=stock_target_review_outcome_due_payload,
        monthly_returns=empty_frame,
        metrics=base_payload,
    )


def _resolve_dated_output(
    project_root: Path,
    output_dir: str | Path | None,
    default_latest: Path,
    default_base: Path,
    as_of: date,
    date_stamp: bool,
) -> Path:
    raw = Path(output_dir) if output_dir is not None else (default_base if date_stamp else default_latest)
    if date_stamp:
        stamp = as_of.strftime("%Y%m%d")
        if not raw.name.endswith(f"_{stamp}"):
            raw = raw.with_name(f"{raw.name}_{stamp}")
    return raw if raw.is_absolute() else project_root / raw


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not pd.notna(number):
        return "N/A"
    return f"{number * 100:.2f}%"


def _num(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not pd.notna(number):
        return "N/A"
    return f"{number:.{digits}f}"


def _signed_num(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not pd.notna(number):
        return "N/A"
    return f"{number:+.{digits}f}"


def _signed_pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not pd.notna(number):
        return "N/A"
    return f"{number * 100:+.2f} pp"


def _read_history(history_path: Path) -> pd.DataFrame:
    if not history_path.exists() or history_path.stat().st_size == 0:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    try:
        history = pd.read_csv(history_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    missing_columns = [column for column in HISTORY_COLUMNS if column not in history.columns]
    if missing_columns:
        missing_frame = pd.DataFrame({column: pd.NA for column in missing_columns}, index=history.index)
        history = pd.concat([history, missing_frame], axis=1)
    return history


def _previous_history_row(history: pd.DataFrame, as_of: date) -> dict[str, Any] | None:
    if history.empty or "as_of_date" not in history.columns:
        return None
    data = history.copy()
    data["_as_of_sort"] = pd.to_datetime(data["as_of_date"], errors="coerce")
    data["_generated_sort"] = pd.to_datetime(data.get("generated_at"), errors="coerce")
    prior = data.dropna(subset=["_as_of_sort"])
    prior = prior[prior["_as_of_sort"].dt.date <= as_of]
    if prior.empty:
        return None
    prior = prior.sort_values(["_as_of_sort", "_generated_sort"], na_position="first")
    return prior.iloc[-1].drop(labels=["_as_of_sort", "_generated_sort"], errors="ignore").to_dict()


def _as_float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(number):
        return None
    return number


def _same_scalar(left: Any, right: Any) -> bool:
    left_number = _as_float_or_none(left)
    right_number = _as_float_or_none(right)
    if left_number is not None and right_number is not None:
        return abs(left_number - right_number) < 1e-12
    left_text = "" if left is None or pd.isna(left) else str(left)
    right_text = "" if right is None or pd.isna(right) else str(right)
    return left_text == right_text


def _history_cell_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _add_history_context(snapshot: dict[str, Any], previous: dict[str, Any] | None) -> None:
    if previous is None:
        snapshot.update(
            {
                "history_previous_as_of_date": None,
                "history_previous_generated_at": None,
                "history_previous_dashboard_posture": None,
                "history_previous_action_posture": None,
                "history_dashboard_posture_changed": False,
                "history_action_posture_changed": False,
                "history_market_cache_status_changed": False,
                "history_allocator_input_status_changed": False,
                "history_paper_final_equity_change": None,
                "history_paper_total_return_change": None,
                "history_paper_satellite_weight_change": None,
                "history_change_count": 0,
                "history_status_change_summary": "no_previous_history",
            }
        )
        return

    status_fields = [
        "dashboard_posture",
        "action_posture",
        "data_freshness_status",
        "market_cache_status",
        "allocator_input_status",
        "paper_freshness_status",
        "sentiment_state",
        "trigger_freshness_status",
        "promotion_status",
        "promotion_decision",
        "paper_latest_regime",
        "paper_latest_candidate",
    ]
    changed = [field for field in status_fields if not _same_scalar(snapshot.get(field), previous.get(field))]
    changes_text = "; ".join(f"{field}:{previous.get(field)}->{snapshot.get(field)}" for field in changed)

    final_equity = _as_float_or_none(snapshot.get("paper_final_equity"))
    previous_final_equity = _as_float_or_none(previous.get("paper_final_equity"))
    total_return = _as_float_or_none(snapshot.get("paper_total_return"))
    previous_total_return = _as_float_or_none(previous.get("paper_total_return"))
    satellite_weight = _as_float_or_none(snapshot.get("paper_latest_satellite_weight"))
    previous_satellite_weight = _as_float_or_none(previous.get("paper_latest_satellite_weight"))

    snapshot.update(
        {
            "history_previous_as_of_date": previous.get("as_of_date"),
            "history_previous_generated_at": previous.get("generated_at"),
            "history_previous_dashboard_posture": previous.get("dashboard_posture"),
            "history_previous_action_posture": previous.get("action_posture"),
            "history_dashboard_posture_changed": "dashboard_posture" in changed,
            "history_action_posture_changed": "action_posture" in changed,
            "history_market_cache_status_changed": "market_cache_status" in changed,
            "history_allocator_input_status_changed": "allocator_input_status" in changed,
            "history_paper_final_equity_change": (
                final_equity - previous_final_equity if final_equity is not None and previous_final_equity is not None else None
            ),
            "history_paper_total_return_change": (
                total_return - previous_total_return if total_return is not None and previous_total_return is not None else None
            ),
            "history_paper_satellite_weight_change": (
                satellite_weight - previous_satellite_weight if satellite_weight is not None and previous_satellite_weight is not None else None
            ),
            "history_change_count": len(changed),
            "history_status_change_summary": changes_text or "no_status_change",
        }
    )


def _append_history(snapshot: dict[str, Any], history_path: Path, as_of: date) -> pd.DataFrame:
    history = _read_history(history_path)
    previous = _previous_history_row(history, as_of)
    _add_history_context(snapshot, previous)

    row = {column: _history_cell_value(snapshot.get(column)) for column in HISTORY_COLUMNS}
    combined = history[HISTORY_COLUMNS].astype("object").copy()
    records = combined.where(pd.notna(combined), None).to_dict(orient="records")
    records.append(row)
    combined = pd.DataFrame.from_records(records, columns=HISTORY_COLUMNS, coerce_float=False)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(history_path, index=False)

    snapshot["history_status"] = "appended"
    snapshot["history_path"] = str(history_path)
    snapshot["history_row_count"] = int(len(combined))
    return combined


def _update_history_frame(snapshot: dict[str, Any], history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    mask = (
        (history["generated_at"].astype(str) == str(snapshot.get("generated_at")))
        & (history["as_of_date"].astype(str) == str(snapshot.get("as_of_date")))
    )
    if not mask.any():
        mask = history.index == history.index[-1]
    current = history.loc[mask].iloc[-1]
    changed_columns: list[str] = []
    changed_values: list[Any] = []
    for column in HISTORY_COLUMNS:
        value = _history_cell_value(snapshot.get(column))
        existing = current.get(column)
        existing_missing = bool(pd.isna(existing))
        value_missing = value is None or bool(pd.isna(value))
        if (existing_missing and value_missing) or (not existing_missing and not value_missing and existing == value):
            continue
        changed_columns.append(column)
        changed_values.append(value)
    if changed_columns:
        history[changed_columns] = history[changed_columns].astype("object")
        history.loc[mask, changed_columns] = [changed_values] * int(mask.sum())
    return history


def _write_history(history: pd.DataFrame, history_path: Path) -> None:
    history.to_csv(history_path, index=False)


def _history_review_context(result: PipelineHistoryReviewResult) -> dict[str, Any]:
    return {
        "history_review_status": "completed",
        "history_review_output_dir": str(result.output_dir),
        "history_review_snapshot_path": str(result.snapshot_path),
        "history_review_report_path": str(result.report_path),
        "history_review_health_state": result.snapshot.get("health_state"),
        "history_review_alert_count": result.snapshot.get("alert_count"),
        "history_review_latest_as_of_date": result.snapshot.get("latest_as_of_date"),
        "history_review_latest_alert_level": result.snapshot.get("latest_alert_level"),
        "history_review_latest_alert_action_stage": result.snapshot.get("latest_alert_action_stage"),
        "history_review_latest_alert_count": result.snapshot.get("latest_alert_count"),
    }


def _dashboard_history_context(result: DailyDashboardResult) -> dict[str, Any]:
    return {
        "dashboard_alert_refresh_status": "completed",
        "dashboard_alert_refresh_error": None,
        "dashboard_history_refresh_status": "completed",
        "dashboard_history_refresh_error": None,
        "dashboard_history_refresh_report_path": str(result.report_path),
        "dashboard_history_refresh_snapshot_path": str(result.snapshot_path),
        "dashboard_pipeline_history_status": result.snapshot.get("pipeline_history_status"),
        "dashboard_pipeline_history_health_state": result.snapshot.get("pipeline_history_health_state"),
        "dashboard_pipeline_history_alert_count": result.snapshot.get("pipeline_history_alert_count"),
        "dashboard_pipeline_history_latest_as_of_date": result.snapshot.get("pipeline_history_latest_as_of_date"),
        "dashboard_pipeline_history_latest_satellite_risk_budget_decision": result.snapshot.get(
            "pipeline_history_latest_satellite_risk_budget_decision"
        ),
        "dashboard_pipeline_history_latest_satellite_risk_budget_recommended_weight": result.snapshot.get(
            "pipeline_history_latest_satellite_risk_budget_recommended_satellite_weight"
        ),
        "daily_preflight_skipped": bool(result.snapshot.get("daily_preflight_skipped", False)),
    }


def _live_shadow_context(result: LiveShadowResult) -> dict[str, Any]:
    return {
        "live_shadow_status": "completed",
        "live_shadow_error": None,
        "live_shadow_output_dir": str(result.output_dir),
        "live_shadow_snapshot_path": str(result.snapshot_path),
        "live_shadow_report_path": str(result.report_path),
        "live_shadow_orders_path": str(result.orders_path),
        "live_shadow_reconcile_path": str(result.reconcile_path),
        "live_shadow_holdings_path": str(result.holdings_path),
        "live_shadow_prices_path": str(result.prices_path) if result.prices_path is not None else None,
        "live_shadow_trade_plan_status": result.snapshot.get("trade_plan_status"),
        "live_shadow_broker_action": result.snapshot.get("broker_action"),
        "live_shadow_research_only": result.snapshot.get("research_only"),
        "live_shadow_current_equity": result.snapshot.get("current_equity"),
        "live_shadow_cash": result.snapshot.get("cash"),
        "live_shadow_order_count": result.snapshot.get("order_count"),
        "live_shadow_buy_count": result.snapshot.get("buy_count"),
        "live_shadow_sell_count": result.snapshot.get("sell_count"),
        "live_shadow_planned_buy_amount": result.snapshot.get("planned_buy_amount"),
        "live_shadow_planned_sell_amount": result.snapshot.get("planned_sell_amount"),
        "live_shadow_estimated_cash_after_orders": result.snapshot.get("estimated_cash_after_orders"),
        "live_shadow_target_gross_weight": result.snapshot.get("target_gross_weight"),
        "live_shadow_warning_count": result.snapshot.get("warning_count"),
    }


def _live_shadow_preflight_context(result: LiveShadowPreflightResult) -> dict[str, Any]:
    return {
        "live_shadow_preflight_status": "completed",
        "live_shadow_preflight_output_dir": str(result.output_dir),
        "live_shadow_preflight_snapshot_path": str(result.snapshot_path),
        "live_shadow_preflight_report_path": str(result.report_path),
        "live_shadow_preflight_blockers_count": len(result.snapshot.get("blockers") or []),
        "live_shadow_preflight_decision": result.snapshot.get("status"),
        "live_shadow_preflight_error": None,
    }


def _live_preflight_context(result: LivePreflightResult) -> dict[str, Any]:
    return {
        "live_preflight_status": "completed",
        "live_preflight_output_dir": str(result.output_dir),
        "live_preflight_snapshot_path": str(result.snapshot_path),
        "live_preflight_report_path": str(result.report_path),
        "live_preflight_checklist_path": str(result.checklist_path),
        "live_preflight_decision": result.snapshot.get("decision"),
        "live_preflight_broker_connection_status": result.snapshot.get("broker_connection_status"),
        "live_preflight_blocking_items_count": len(result.snapshot.get("blocking_items") or []),
        "live_preflight_monitor_items_count": len(result.snapshot.get("monitor_items") or []),
        "live_preflight_pipeline_snapshot_status": result.snapshot.get("pipeline_snapshot_status"),
        "live_preflight_pipeline_snapshot_path": result.snapshot.get("pipeline_snapshot_path"),
        "live_preflight_active_target_count": result.snapshot.get("active_target_count"),
        "live_preflight_stock_target_review_action_count": result.snapshot.get("stock_target_review_action_count"),
        "live_preflight_live_shadow_review_decisions_path": result.snapshot.get("live_shadow_review_decisions_path"),
        "live_preflight_live_shadow_review_decision_count": result.snapshot.get("live_shadow_review_decision_count"),
        "live_preflight_live_shadow_review_blocking_decision_count": result.snapshot.get(
            "live_shadow_review_blocking_decision_count"
        ),
        "live_preflight_live_shadow_review_monitor_decision_count": result.snapshot.get(
            "live_shadow_review_monitor_decision_count"
        ),
        "live_preflight_live_shadow_review_unknown_decision_count": result.snapshot.get(
            "live_shadow_review_unknown_decision_count"
        ),
        "live_preflight_error": None,
    }


def _alerts_context(result: DailyAlertsResult) -> dict[str, Any]:
    return {
        "alert_level": result.payload.get("alert_level"),
        "alert_count": result.payload.get("alert_count"),
        "alert_action_stage": result.payload.get("action_stage"),
        "alert_action_summary": result.payload.get("action_summary"),
        "alert_action_item_count": len(result.payload.get("action_items") or []),
        "alert_action_required": result.payload.get("action_required"),
        "alert_critical_count": result.payload.get("critical_count"),
        "alert_warning_count": result.payload.get("warning_count"),
        "alert_info_count": result.payload.get("info_count"),
        "alerts_report_path": str(result.report_path),
        "alerts_json_path": str(result.json_path),
        "latest_alerts_path": str(result.latest_report_path),
    }


def _next_step_context(snapshot: dict[str, Any]) -> dict[str, Any]:
    gate_status = str(snapshot.get("trading_day_gate_status") or "")
    alert_level = str(snapshot.get("alert_level") or "normal")
    review_required_unreviewed = int(_as_float_or_none(snapshot.get("paper_stock_target_review_required_unreviewed_count")) or 0)
    action_count = int(_as_float_or_none(snapshot.get("paper_stock_target_review_action_count")) or 0)
    outcome_status = str(snapshot.get("paper_stock_target_review_outcome_analysis_status") or "")
    ready_horizons = int(_as_float_or_none(snapshot.get("paper_stock_target_review_outcome_analysis_ready_horizon_count")) or 0)
    due_horizons = int(_as_float_or_none(snapshot.get("paper_stock_target_review_outcome_due_row_count")) or 0)
    chip_outcome_status = str(snapshot.get("chip_reversal_candidate_outcomes_readiness_status") or "")
    chip_ready_horizons = snapshot.get("chip_reversal_candidate_outcomes_ready_horizons") or []
    chip_pending_horizons = snapshot.get("chip_reversal_candidate_outcomes_pending_horizons") or []
    chip_next_horizon = snapshot.get("chip_reversal_candidate_outcomes_next_review_horizon")
    chip_next_reason = snapshot.get("chip_reversal_candidate_outcomes_next_review_reason")
    chip_report_path = snapshot.get("chip_reversal_candidate_outcomes_report_path") or "chip_reversal_candidate_outcomes.md"
    promotion_group_support = _as_float_or_none(
        snapshot.get("promotion_support_group_count", snapshot.get("promotion_sensitivity_group_support_count"))
    )
    promotion_run_support = _as_float_or_none(snapshot.get("promotion_sensitivity_run_support_count"))
    promotion_min_support = _as_float_or_none(snapshot.get("promotion_min_sensitivity_support"))
    promotion_evidence_correlated = (
        str(snapshot.get("promotion_decision") or "") == "watch_candidate"
        and bool(snapshot.get("promotion_candidate_passes_headline_gate"))
        and promotion_group_support is not None
        and promotion_run_support is not None
        and promotion_min_support is not None
        and promotion_group_support < promotion_min_support
        and promotion_run_support >= promotion_min_support
        and promotion_run_support > promotion_group_support
    )
    live_shadow_preflight_status = str(snapshot.get("live_shadow_preflight_status") or "")
    live_shadow_preflight_decision = str(snapshot.get("live_shadow_preflight_decision") or "")
    live_shadow_status = str(snapshot.get("live_shadow_status") or "")

    if gate_status == "trading_day_data_not_ready":
        return {
            "pipeline_next_step_stage": "refresh_data",
            "pipeline_next_step_action": "Refresh local A-share data after close and rerun daily-pipeline.",
            "pipeline_next_step_reason": "Trading-day gate says after-close data is not ready.",
            "pipeline_blocker": "after_close_data_not_ready",
            "pipeline_user_intervention_required": False,
        }
    if gate_status == "future_dated_data":
        return {
            "pipeline_next_step_stage": "inspect_data_dates",
            "pipeline_next_step_action": "Inspect local data dates before relying on today's report.",
            "pipeline_next_step_reason": "Local equity data is dated after the requested as-of date.",
            "pipeline_blocker": "future_dated_data",
            "pipeline_user_intervention_required": True,
        }
    if str(snapshot.get("paper_account_status") or "") == "failed":
        return {
            "pipeline_next_step_stage": "retry_paper_account",
            "pipeline_next_step_action": "Re-run paper-account with corrected paper data or config, then rerun daily-pipeline.",
            "pipeline_next_step_reason": "Paper-account step failed and fallback outputs were generated.",
            "pipeline_blocker": "paper_account_step_failed",
            "pipeline_user_intervention_required": True,
        }
    if (
        live_shadow_status == "blocked_by_preflight"
        or (live_shadow_preflight_status == "completed" and live_shadow_preflight_decision == "blocked")
        or live_shadow_preflight_status == "failed"
    ):
        reason = f"Live-shadow preflight status is {live_shadow_preflight_status or 'unknown'}."
        blockers = snapshot.get("live_shadow_preflight_blockers_count")
        if blockers is not None:
            reason = f"{reason} Blockers: {blockers}."
        return {
            "pipeline_next_step_stage": "review_live_shadow_preflight",
            "pipeline_next_step_action": "Review the live-shadow preflight report, clear blockers or investigate failure, then rerun daily-pipeline and live-shadow.",
            "pipeline_next_step_reason": reason,
            "pipeline_blocker": "live_shadow_preflight_blocked",
            "pipeline_user_intervention_required": True,
        }
    if alert_level == "critical":
        return {
            "pipeline_next_step_stage": "resolve_critical_alerts",
            "pipeline_next_step_action": "Open alerts.md and resolve critical freshness, drawdown, or posture issues before reading new signals.",
            "pipeline_next_step_reason": "Daily alerts are critical.",
            "pipeline_blocker": "critical_alerts",
            "pipeline_user_intervention_required": True,
        }
    if review_required_unreviewed > 0:
        return {
            "pipeline_next_step_stage": "update_manual_review_notes",
            "pipeline_next_step_action": "Open stock_target_review_assistant.md, then update manual_status/manual_note in stock_target_review_notes.csv for open review-required rows.",
            "pipeline_next_step_reason": f"{review_required_unreviewed} review-required row(s) have no manual review note yet.",
            "pipeline_blocker": "manual_review_notes_open",
            "pipeline_user_intervention_required": True,
        }
    if action_count > 0:
        return {
            "pipeline_next_step_stage": "review_stock_target_actions",
            "pipeline_next_step_action": "Open stock_target_review_assistant.md or stock_target_review_actions.md and process the research-only review queue.",
            "pipeline_next_step_reason": f"{action_count} stock-target review action row(s) are open.",
            "pipeline_blocker": "stock_target_actions_open",
            "pipeline_user_intervention_required": True,
        }
    if due_horizons > 0:
        return {
            "pipeline_next_step_stage": "refresh_outcome_due_queue",
            "pipeline_next_step_action": "Open stock_target_review_outcome_due.md, refresh local OHLCV if needed, then rerun paper-account or daily-pipeline so due outcomes can be recorded.",
            "pipeline_next_step_reason": f"{due_horizons} outcome horizon(s) are due but still pending.",
            "pipeline_blocker": "outcome_due_queue_open",
            "pipeline_user_intervention_required": False,
        }
    if ready_horizons > 0:
        return {
            "pipeline_next_step_stage": "review_outcome_analysis",
            "pipeline_next_step_action": "Review stock_target_review_outcome_analysis.md for mature groups before changing strategy parameters.",
            "pipeline_next_step_reason": f"{ready_horizons} outcome horizon(s) passed the configured sample-readiness gates.",
            "pipeline_blocker": "",
            "pipeline_user_intervention_required": False,
        }
    if chip_outcome_status in {"ready", "partial"} and chip_ready_horizons:
        horizons_text = ", ".join(str(item) for item in chip_ready_horizons)
        return {
            "pipeline_next_step_stage": "review_chip_reversal_outcomes",
            "pipeline_next_step_action": f"Open {chip_report_path} and review mature chip-reversal outcome groups before promoting rules.",
            "pipeline_next_step_reason": f"Chip-reversal outcome horizon(s) are ready: {horizons_text}.",
            "pipeline_blocker": "",
            "pipeline_user_intervention_required": False,
        }
    if gate_status == "non_trading_day":
        return {
            "pipeline_next_step_stage": "wait_for_next_trading_close",
            "pipeline_next_step_action": "Wait for the next A-share trading day close, then refresh data and rerun daily-pipeline.",
            "pipeline_next_step_reason": "The as-of date is not an A-share trading day.",
            "pipeline_blocker": "non_trading_day",
            "pipeline_user_intervention_required": False,
        }
    if outcome_status in {"waiting_for_evaluable_returns", "sample_insufficient"}:
        return {
            "pipeline_next_step_stage": "accumulate_outcome_samples",
            "pipeline_next_step_action": "Keep collecting post-review OHLCV outcomes until sample-readiness gates are met.",
            "pipeline_next_step_reason": str(snapshot.get("paper_stock_target_review_outcome_analysis_sample_warning") or outcome_status),
            "pipeline_blocker": "insufficient_outcome_samples",
            "pipeline_user_intervention_required": False,
        }
    if chip_outcome_status in {"waiting_for_future_bar", "partial"} and chip_pending_horizons:
        pending_text = ", ".join(str(item) for item in chip_pending_horizons)
        horizon_text = str(chip_next_horizon or (chip_pending_horizons[0] if chip_pending_horizons else "N/A"))
        reason_text = str(chip_next_reason or chip_outcome_status)
        return {
            "pipeline_next_step_stage": "accumulate_chip_reversal_outcome_samples",
            "pipeline_next_step_action": "Keep refreshing unified daily market data and rerun chip-reversal candidate outcomes after the next close.",
            "pipeline_next_step_reason": f"Chip-reversal outcome horizon {horizon_text} is not evaluable yet; pending horizons: {pending_text}; reason: {reason_text}.",
            "pipeline_blocker": "chip_reversal_outcomes_waiting_for_future_bar",
            "pipeline_user_intervention_required": False,
        }
    if alert_level == "warning":
        return {
            "pipeline_next_step_stage": "inspect_warning_alerts",
            "pipeline_next_step_action": "Open alerts.md and inspect warning-level model, freshness, or review items.",
            "pipeline_next_step_reason": "Daily alerts are warning-level.",
            "pipeline_blocker": "warning_alerts",
            "pipeline_user_intervention_required": True,
        }
    if promotion_evidence_correlated:
        return {
            "pipeline_next_step_stage": "monitor_allocator_promotion_evidence",
            "pipeline_next_step_action": "Keep the allocator candidate on watch and wait for independent sensitivity-group support to improve.",
            "pipeline_next_step_reason": (
                f"Raw sensitivity-run support is {promotion_run_support:.0f}, "
                f"but independent group support is {promotion_group_support:.0f} / {promotion_min_support:.0f}."
            ),
            "pipeline_blocker": "",
            "pipeline_user_intervention_required": False,
        }
    return {
        "pipeline_next_step_stage": "routine_monitor",
        "pipeline_next_step_action": "Continue the after-close daily pipeline and monitor paper-account drift.",
        "pipeline_next_step_reason": "No blocking gate or mature outcome-review trigger is present.",
        "pipeline_blocker": "",
        "pipeline_user_intervention_required": False,
    }


def _render_report(snapshot: dict[str, Any]) -> str:
    return f"""# Daily Research Pipeline

Generated at: `{snapshot.get("generated_at")}`

This pipeline is a research-only after-close workflow. It does not connect to brokers, place orders, or provide investment advice.

## Run State

| Item | Value |
| --- | ---: |
| As-of date | {snapshot.get("as_of_date", "N/A")} |
| Data freshness | {snapshot.get("data_freshness_status", "N/A")} |
| Trading-day gate | `{snapshot.get("trading_day_gate_status", "N/A")}` |
| After-close data | `{snapshot.get("after_close_data_status", "N/A")}` |
| Dashboard posture | `{snapshot.get("dashboard_posture", "N/A")}` |
| Daily action posture | `{snapshot.get("action_posture", "N/A")}` |
| Paper latest regime | `{snapshot.get("paper_latest_regime", "N/A")}` |
| Paper latest candidate | `{snapshot.get("paper_latest_candidate", "N/A")}` |
| Paper core weight | {_pct(snapshot.get("paper_latest_core_weight"))} |
| Paper satellite weight | {_pct(snapshot.get("paper_latest_satellite_weight"))} |
| Paper cash weight | {_pct(snapshot.get("paper_latest_cash_weight"))} |
| Satellite risk-budget decision | `{snapshot.get("satellite_risk_budget_decision", "N/A")}` |
| Satellite budget recommendation | {_pct(snapshot.get("satellite_risk_budget_recommended_satellite_weight"))} |
| Alert level | `{snapshot.get("alert_level", "N/A")}` |
| Market cache | `{snapshot.get("market_cache_status", "N/A")}` |
| Market cache source | `{snapshot.get("market_cache_source_kind", "N/A")}` |
| Market cache latest date | {snapshot.get("market_cache_latest_date", "N/A")} |
| Allocator inputs | `{snapshot.get("allocator_input_status", "N/A")}` |

## Next Step

| Item | Value |
| --- | ---: |
| Stage | `{snapshot.get("pipeline_next_step_stage", "N/A")}` |
| Action | {snapshot.get("pipeline_next_step_action", "N/A")} |
| Reason | {snapshot.get("pipeline_next_step_reason", "N/A")} |
| Blocker | `{snapshot.get("pipeline_blocker", "N/A")}` |
| User intervention required | `{snapshot.get("pipeline_user_intervention_required", "N/A")}` |

## Key Metrics

| Metric | Value |
| --- | ---: |
| Local equity latest date | {snapshot.get("local_equity_latest_date", "N/A")} |
| Paper final equity | {_num(snapshot.get("paper_final_equity"), 2)} |
| Paper total return | {_pct(snapshot.get("paper_total_return"))} |
| Paper max drawdown | {_pct(snapshot.get("paper_max_drawdown"))} |
| Paper current drawdown | {_pct(snapshot.get("paper_current_drawdown"))} |
| Paper Sharpe | {_num(snapshot.get("paper_sharpe"), 3)} |
| Paper audit events | {snapshot.get("paper_audit_event_count", "N/A")} |
| Alert count | {snapshot.get("alert_count", "N/A")} |
| Action stage | `{snapshot.get("alert_action_stage", "N/A")}` |
| Sentiment state | `{snapshot.get("sentiment_state", "N/A")}` |
| Trigger freshness | `{snapshot.get("trigger_freshness_status", "N/A")}` |

## Model Build Hygiene

| Metric | Value |
| --- | ---: |
| Audit status | `{snapshot.get("model_audit_status", "N/A")}` |
| Duplicate config groups | {snapshot.get("model_audit_duplicate_config_groups", "N/A")} |
| Root configs without extends | {snapshot.get("model_audit_root_configs_without_extends", "N/A")} |
| Walk-forward action items | {snapshot.get("model_audit_walk_forward_action_items", "N/A")} |
| Resume/finalize candidates | {snapshot.get("model_audit_walk_forward_resume_candidates", "N/A")} |
| Archive-review candidates | {snapshot.get("model_audit_walk_forward_archive_review_candidates", "N/A")} |
| Top action | `{snapshot.get("model_audit_top_action", "N/A")}` |
| Audit report | `{snapshot.get("model_audit_report_path", "N/A")}` |
| Action CSV | `{snapshot.get("model_audit_actions_path", "N/A")}` |

## Allocator Promotion Watch

| Item | Value |
| --- | ---: |
| Status | `{snapshot.get("promotion_status", "N/A")}` |
| Decision | `{snapshot.get("promotion_decision", "N/A")}` |
| Headline gate | `{snapshot.get("promotion_candidate_passes_headline_gate", "N/A")}` |
| Independent support groups | {snapshot.get("promotion_support_group_count", snapshot.get("promotion_sensitivity_group_support_count", snapshot.get("promotion_sensitivity_support_count", "N/A")))} / {snapshot.get("promotion_min_sensitivity_support", "N/A")} |
| Sensitivity group support | {snapshot.get("promotion_sensitivity_group_support_count", snapshot.get("promotion_sensitivity_support_count", "N/A"))} |
| Sensitivity run support | {snapshot.get("promotion_sensitivity_run_support_count", snapshot.get("promotion_sensitivity_support_count", "N/A"))} |
| Evidence group support | {snapshot.get("promotion_evidence_support_count", "N/A")} |
| Return edge | {_signed_pct(snapshot.get("promotion_return_edge"))} |
| Sharpe edge | {_signed_num(snapshot.get("promotion_sharpe_edge"), 3)} |
| Drawdown change | {_signed_pct(snapshot.get("promotion_drawdown_change"))} |
| Review report | `{snapshot.get("promotion_report_path", "N/A")}` |

## Satellite Risk Budget

| Item | Value |
| --- | ---: |
| Status | `{snapshot.get("satellite_risk_budget_status", "N/A")}` |
| Decision | `{snapshot.get("satellite_risk_budget_decision", "N/A")}` |
| Reason | {snapshot.get("satellite_risk_budget_reason", "N/A")} |
| Next action stage | `{snapshot.get("satellite_risk_budget_next_action_stage", "N/A")}` |
| Recommended satellite budget | {_pct(snapshot.get("satellite_risk_budget_recommended_satellite_weight"))} |
| Selected outcome horizon | `{snapshot.get("satellite_risk_budget_selected_horizon", "N/A")}` |
| Ready outcome horizons | {snapshot.get("satellite_risk_budget_ready_horizon_count", "N/A")} |
| Satellite trial rules | {snapshot.get("satellite_trial_rule_count", "N/A")} |
| Review report | `{snapshot.get("satellite_risk_budget_report_path", "N/A")}` |
| Checklist CSV | `{snapshot.get("satellite_risk_budget_checklist_path", "N/A")}` |
| Trial rules CSV | `{snapshot.get("satellite_trial_rules_path", "N/A")}` |
| Trial rules JSON | `{snapshot.get("satellite_trial_rules_json_path", "N/A")}` |
| Trial replay status | `{snapshot.get("satellite_trial_replay_status", "N/A")}` |
| Trial replay matched events | {snapshot.get("satellite_trial_replay_matched_event_count", "N/A")} |
| Trial replay best horizon | `{snapshot.get("satellite_trial_replay_best_horizon", "N/A")}` |
| Trial replay avg-return edge | {_signed_pct(snapshot.get("satellite_trial_replay_best_avg_return_edge"))} |
| Trial replay win-rate edge | {_signed_pct(snapshot.get("satellite_trial_replay_best_win_rate_edge"))} |
| Trial replay report | `{snapshot.get("satellite_trial_replay_report_path", "N/A")}` |

## 涨停与强势股研究池

| 项目 | 值 |
| --- | ---: |
| 状态 | `{snapshot.get("momentum_focus_status", "N/A")}` |
| 行情日期 | {snapshot.get("momentum_focus_trade_date", "N/A")} |
| 数据来源 | `{snapshot.get("momentum_focus_source_kind", "N/A")}` |
| 量额数据质量 | `{snapshot.get("momentum_focus_market_data_quality_status", "N/A")}` |
| 成交量有效行数/覆盖率 | {snapshot.get("momentum_focus_volume_positive_count", "N/A")} / {_pct(snapshot.get("momentum_focus_volume_coverage_ratio"))} |
| 成交额有效行数/覆盖率 | {snapshot.get("momentum_focus_turnover_positive_count", "N/A")} / {_pct(snapshot.get("momentum_focus_turnover_coverage_ratio"))} |
| 候选数量 | {snapshot.get("momentum_focus_candidate_count", "N/A")} |
| 涨停数量 | {snapshot.get("momentum_focus_limit_up_count", "N/A")} |
| 涨幅大于等于7%数量 | {snapshot.get("momentum_focus_strong_gain_count", "N/A")} |
| ST/退市排除数量 | {snapshot.get("momentum_focus_excluded_special_treatment_count", "N/A")} |
| 复核硬排除数量 | {snapshot.get("momentum_focus_excluded_by_review_required_count", "N/A")} |
| 研究范围 | `{snapshot.get("momentum_focus_board_scope", "N/A")}` |
| 涨幅阈值 | {_num(snapshot.get("momentum_focus_strong_gain_threshold_pct"), 2)}% |
| 先验样本行数 | {snapshot.get("momentum_focus_outcome_prior_rows", "N/A")} |
| 先验匹配候选数 | {snapshot.get("momentum_focus_outcome_prior_matched_count", "N/A")} |
| 先验汇总行数 | {snapshot.get("momentum_focus_outcome_summary_rows", "N/A")} |
| 名称映射状态 | `{snapshot.get("momentum_focus_name_map_status", "N/A")}` |
| 名称映射行数 | {snapshot.get("momentum_focus_name_map_rows", "N/A")} |
| 券商动作 | `{snapshot.get("momentum_focus_broker_action", "N/A")}` |
| 研究专用 | `{snapshot.get("momentum_focus_research_only", "N/A")}` |
| 报告 | `{snapshot.get("momentum_focus_report_path", "N/A")}` |
| 候选CSV | `{snapshot.get("momentum_focus_candidates_path", "N/A")}` |
| 错误 | `{snapshot.get("momentum_focus_error", "N/A")}` |

## 筹码反转候选结果回看

| 项目 | 值 |
| --- | ---: |
| 状态 | `{snapshot.get("chip_reversal_candidate_outcomes_status", "N/A")}` |
| 结果成熟度 | `{snapshot.get("chip_reversal_candidate_outcomes_readiness_status", "N/A")}` |
| 候选数量 | {snapshot.get("chip_reversal_candidate_outcomes_candidate_count", "N/A")} |
| 已成熟结果 | {snapshot.get("chip_reversal_candidate_outcomes_ready_count", "N/A")} |
| 待成熟结果 | {snapshot.get("chip_reversal_candidate_outcomes_pending_count", "N/A")} |
| 已成熟周期 | `{snapshot.get("chip_reversal_candidate_outcomes_ready_horizons", "N/A")}` |
| 待成熟周期 | `{snapshot.get("chip_reversal_candidate_outcomes_pending_horizons", "N/A")}` |
| 下一回看周期 | `{snapshot.get("chip_reversal_candidate_outcomes_next_review_horizon", "N/A")}` |
| 下一回看原因 | `{snapshot.get("chip_reversal_candidate_outcomes_next_review_reason", "N/A")}` |
| 最新行情日期 | {snapshot.get("chip_reversal_candidate_outcomes_latest_market_trade_date", "N/A")} |
| 行情来源 | `{snapshot.get("chip_reversal_candidate_outcomes_market_source_kind", "N/A")}` |
| 晋级门槛 | `{snapshot.get("chip_reversal_candidate_outcomes_promotion_gate_status", "N/A")}` |
| 券商动作 | `{snapshot.get("chip_reversal_candidate_outcomes_broker_action", "N/A")}` |
| 报告 | `{snapshot.get("chip_reversal_candidate_outcomes_report_path", "N/A")}` |
| 错误 | `{snapshot.get("chip_reversal_candidate_outcomes_error", "N/A")}` |

## Paper Target Holdings

| Item | Value |
| --- | ---: |
| Target holdings report | `{snapshot.get("paper_target_holdings_report_path", "N/A")}` |
| Target holdings CSV | `{snapshot.get("paper_target_holdings_path", "N/A")}` |
| Target holdings JSON | `{snapshot.get("paper_target_holdings_json_path", "N/A")}` |
| Target holding count | {snapshot.get("paper_target_holding_count", "N/A")} |
| Active target count | {snapshot.get("paper_active_target_count", "N/A")} |
| Stock targets report | `{snapshot.get("paper_stock_targets_report_path", "N/A")}` |
| Stock targets CSV | `{snapshot.get("paper_stock_targets_path", "N/A")}` |
| Stock target rows | {snapshot.get("paper_stock_target_count", "N/A")} |
| Source rows before review exclusion | {snapshot.get("paper_stock_target_source_count", "N/A")} |
| Active stock targets | {snapshot.get("paper_active_stock_target_count", "N/A")} |
| Suppressed stock rows | {snapshot.get("paper_suppressed_stock_target_count", "N/A")} |
| Review-required rows excluded | {snapshot.get("paper_stock_target_review_excluded_count", "N/A")} |
| Large-cap tracking exclusions | {snapshot.get("paper_stock_tracking_excluded_large_market_cap_count", "N/A")} |
| Max tracking market cap | {snapshot.get("paper_stock_tracking_max_market_cap_yi", "N/A")} Yi Yuan |
| Market-cap missing rows | {snapshot.get("paper_stock_tracking_market_cap_missing_count", "N/A")} |
| Market-cap cache status | `{snapshot.get("paper_stock_market_cap_cache_status", "N/A")}` |
| Market-cap cache latest snapshot | {snapshot.get("paper_stock_market_cap_cache_latest_snapshot_date", "N/A")} |
| Market-cap cache path | `{snapshot.get("paper_stock_market_cap_path", "N/A")}` |
| Stock target review report | `{snapshot.get("paper_stock_target_review_report_path", "N/A")}` |
| Stock target review CSV | `{snapshot.get("paper_stock_target_review_path", "N/A")}` |
| Stock target action report | `{snapshot.get("paper_stock_target_review_actions_report_path", "N/A")}` |
| Stock target action CSV | `{snapshot.get("paper_stock_target_review_actions_path", "N/A")}` |
| Stock target assistant report | `{snapshot.get("paper_stock_target_review_assistant_report_path", "N/A")}` |
| Stock target assistant CSV | `{snapshot.get("paper_stock_target_review_assistant_path", "N/A")}` |
| Stock target decision template report | `{snapshot.get("paper_stock_target_review_decision_template_report_path", "N/A")}` |
| Stock target decision template CSV | `{snapshot.get("paper_stock_target_review_decision_template_path", "N/A")}` |
| Stock target decision template XLSX | `{snapshot.get("paper_stock_target_review_decision_template_xlsx_path", "N/A")}` |
| Stock target outcome report | `{snapshot.get("paper_stock_target_review_outcomes_report_path", "N/A")}` |
| Stock target outcome CSV | `{snapshot.get("paper_stock_target_review_outcomes_path", "N/A")}` |
| Stock target outcome history report | `{snapshot.get("paper_stock_target_review_outcomes_history_report_path", "N/A")}` |
| Stock target outcome history CSV | `{snapshot.get("paper_stock_target_review_outcomes_history_path", "N/A")}` |
| Stock target outcome history snapshot | `{snapshot.get("paper_stock_target_review_outcomes_history_snapshot_path", "N/A")}` |
| Stock target outcome analysis report | `{snapshot.get("paper_stock_target_review_outcome_analysis_report_path", "N/A")}` |
| Stock target outcome analysis CSV | `{snapshot.get("paper_stock_target_review_outcome_analysis_path", "N/A")}` |
| Stock target outcome calendar report | `{snapshot.get("paper_stock_target_review_outcome_calendar_report_path", "N/A")}` |
| Stock target outcome calendar CSV | `{snapshot.get("paper_stock_target_review_outcome_calendar_path", "N/A")}` |
| Stock target outcome due report | `{snapshot.get("paper_stock_target_review_outcome_due_report_path", "N/A")}` |
| Stock target outcome due CSV | `{snapshot.get("paper_stock_target_review_outcome_due_path", "N/A")}` |
| Review required | {snapshot.get("paper_stock_target_review_required_count", "N/A")} |
| Monitor reviews | {snapshot.get("paper_stock_target_review_monitor_count", "N/A")} |
| Trigger reviews | {snapshot.get("paper_stock_target_review_trigger_count", "N/A")} |
| Drawdown reviews | {snapshot.get("paper_stock_target_review_drawdown_count", "N/A")} |
| Suppressed-layer reviews | {snapshot.get("paper_stock_target_review_suppressed_layer_count", "N/A")} |
| Watch reviews | {snapshot.get("paper_stock_target_review_watch_count", "N/A")} |
| Review notes CSV | `{snapshot.get("paper_stock_target_review_notes_path", "N/A")}` |
| Review notes snapshot | `{snapshot.get("paper_stock_target_review_notes_snapshot_path", "N/A")}` |
| Manual reviewed current rows | {snapshot.get("paper_stock_target_review_manual_note_count", "N/A")} |
| Unreviewed current rows | {snapshot.get("paper_stock_target_review_unreviewed_count", "N/A")} |
| Review-required unreviewed rows | {snapshot.get("paper_stock_target_review_required_unreviewed_count", "N/A")} |
| Manual status pending / reviewed / watch | {snapshot.get("paper_stock_target_review_manual_pending_count", "N/A")} / {snapshot.get("paper_stock_target_review_manual_reviewed_count", "N/A")} / {snapshot.get("paper_stock_target_review_manual_watch_count", "N/A")} |
| Manual status resolved / exclude-candidate / other | {snapshot.get("paper_stock_target_review_manual_resolved_count", "N/A")} / {snapshot.get("paper_stock_target_review_manual_exclude_candidate_count", "N/A")} / {snapshot.get("paper_stock_target_review_manual_other_status_count", "N/A")} |
| Open manual action rows | {snapshot.get("paper_stock_target_review_action_count", "N/A")} |
| Action rows: pending / watch / exclusion | {snapshot.get("paper_stock_target_review_action_pending_model_count", "N/A")} / {snapshot.get("paper_stock_target_review_action_manual_watch_count", "N/A")} / {snapshot.get("paper_stock_target_review_action_manual_exclusion_count", "N/A")} |
| Action rows: normalize / due | {snapshot.get("paper_stock_target_review_action_status_normalization_count", "N/A")} / {snapshot.get("paper_stock_target_review_action_next_review_due_count", "N/A")} |
| Assistant rows / pending | {snapshot.get("paper_stock_target_review_assistant_count", "N/A")} / {snapshot.get("paper_stock_target_review_assistant_pending_count", "N/A")} |
| Assistant price ok / stale / missing | {snapshot.get("paper_stock_target_review_assistant_price_ok_count", "N/A")} / {snapshot.get("paper_stock_target_review_assistant_price_stale_count", "N/A")} / {snapshot.get("paper_stock_target_review_assistant_price_missing_count", "N/A")} |
| Decision template rows / blank status rows | {snapshot.get("paper_stock_target_review_decision_template_count", "N/A")} / {snapshot.get("paper_stock_target_review_decision_template_blank_status_count", "N/A")} |
| Outcome rows complete / partial / pending | {snapshot.get("paper_stock_target_review_outcome_complete_count", "N/A")} / {snapshot.get("paper_stock_target_review_outcome_partial_count", "N/A")} / {snapshot.get("paper_stock_target_review_outcome_pending_count", "N/A")} |
| Outcome rows missing entry price | {snapshot.get("paper_stock_target_review_outcome_missing_entry_price_count", "N/A")} |
| Outcome history rows complete / partial / pending | {snapshot.get("paper_stock_target_review_outcomes_history_complete_count", "N/A")} / {snapshot.get("paper_stock_target_review_outcomes_history_partial_count", "N/A")} / {snapshot.get("paper_stock_target_review_outcomes_history_pending_count", "N/A")} |
| Outcome history rows | {snapshot.get("paper_stock_target_review_outcomes_history_row_count", "N/A")} |
| Outcome history latest review date | {snapshot.get("paper_stock_target_review_outcomes_history_latest_review_date", "N/A")} |
| Outcome analysis status | `{snapshot.get("paper_stock_target_review_outcome_analysis_status", "N/A")}` |
| Outcome analysis rows | {snapshot.get("paper_stock_target_review_outcome_analysis_row_count", "N/A")} |
| Outcome analysis ready horizons | {snapshot.get("paper_stock_target_review_outcome_analysis_ready_horizon_count", "N/A")} |
| Outcome analysis min rows / group rows | {snapshot.get("paper_stock_target_review_outcome_analysis_min_evaluable", "N/A")} / {snapshot.get("paper_stock_target_review_outcome_analysis_min_group_evaluable", "N/A")} |
| Outcome calendar rows / pending | {snapshot.get("paper_stock_target_review_outcome_calendar_row_count", "N/A")} / {snapshot.get("paper_stock_target_review_outcome_calendar_pending_count", "N/A")} |
| Outcome due horizons / pending rows | {snapshot.get("paper_stock_target_review_outcome_due_row_count", "N/A")} / {snapshot.get("paper_stock_target_review_outcome_due_pending_count", "N/A")} |
| Outcome calendar next action | {snapshot.get("paper_stock_target_review_outcome_calendar_next_action_date", "N/A")} / `{snapshot.get("paper_stock_target_review_outcome_calendar_next_action_horizon", "N/A")}` |
| Outcome due next action | {snapshot.get("paper_stock_target_review_outcome_due_next_date", "N/A")} / `{snapshot.get("paper_stock_target_review_outcome_due_next_horizon", "N/A")}` |
| Outcome analysis evaluable 1D / 5D / 20D | {snapshot.get("paper_stock_target_review_outcome_analysis_evaluable_1d_count", "N/A")} / {snapshot.get("paper_stock_target_review_outcome_analysis_evaluable_5d_count", "N/A")} / {snapshot.get("paper_stock_target_review_outcome_analysis_evaluable_20d_count", "N/A")} |
| Outcome maturity next 1D / 5D / 10D / 20D | {snapshot.get("paper_stock_target_review_outcome_maturity_next_1d_date", "N/A")} / {snapshot.get("paper_stock_target_review_outcome_maturity_next_5d_date", "N/A")} / {snapshot.get("paper_stock_target_review_outcome_maturity_next_10d_date", "N/A")} / {snapshot.get("paper_stock_target_review_outcome_maturity_next_20d_date", "N/A")} |
| Outcome analysis warning | {snapshot.get("paper_stock_target_review_outcome_analysis_sample_warning", "N/A")} |
| Drawdown threshold | {_pct(snapshot.get("paper_stock_target_review_drawdown_threshold"))} |
| Watch drawdown threshold | {_pct(snapshot.get("paper_stock_target_review_watch_drawdown_threshold"))} |
| Loss attention threshold | {_pct(snapshot.get("paper_stock_target_review_loss_attention_threshold"))} |
| Warning only after close | `{snapshot.get("paper_stock_target_review_warning_only_after_close", "N/A")}` |
| Broker action | `{snapshot.get("paper_target_broker_action", "N/A")}` |

## Live Shadow

| Item | Value |
| --- | ---: |
| Status | `{snapshot.get("live_shadow_status", "N/A")}` |
| Trade plan status | `{snapshot.get("live_shadow_trade_plan_status", "N/A")}` |
| Broker action | `{snapshot.get("live_shadow_broker_action", "N/A")}` |
| Research only | `{snapshot.get("live_shadow_research_only", "N/A")}` |
| Current equity | {_num(snapshot.get("live_shadow_current_equity"), 2)} |
| Cash | {_num(snapshot.get("live_shadow_cash"), 2)} |
| Target gross weight | {_pct(snapshot.get("live_shadow_target_gross_weight"))} |
| Orders buy / sell | {snapshot.get("live_shadow_buy_count", "N/A")} / {snapshot.get("live_shadow_sell_count", "N/A")} |
| Order count | {snapshot.get("live_shadow_order_count", "N/A")} |
| Planned buy / sell amount | {_num(snapshot.get("live_shadow_planned_buy_amount"), 2)} / {_num(snapshot.get("live_shadow_planned_sell_amount"), 2)} |
| Estimated cash after shadow orders | {_num(snapshot.get("live_shadow_estimated_cash_after_orders"), 2)} |
| Warnings | {snapshot.get("live_shadow_warning_count", "N/A")} |
| Plan report | `{snapshot.get("live_shadow_report_path", "N/A")}` |
| Orders CSV | `{snapshot.get("live_shadow_orders_path", "N/A")}` |
| Reconcile CSV | `{snapshot.get("live_shadow_reconcile_path", "N/A")}` |
| Error | `{snapshot.get("live_shadow_error", "N/A")}` |

## Live Shadow Preflight

| Item | Value |
| --- | ---: |
| Status | `{snapshot.get("live_shadow_preflight_status", "N/A")}` |
| Decision | `{snapshot.get("live_shadow_preflight_decision", "N/A")}` |
| Blockers | `{snapshot.get("live_shadow_preflight_blockers_count", "N/A")}` |
| Report | `{snapshot.get("live_shadow_preflight_report_path", "N/A")}` |
| Error | `{snapshot.get("live_shadow_preflight_error", "N/A")}` |

## Live Preflight

| Item | Value |
| --- | ---: |
| Status | `{snapshot.get("live_preflight_status", "N/A")}` |
| Decision | `{snapshot.get("live_preflight_decision", "N/A")}` |
| Daily preflight skipped | `{snapshot.get("daily_preflight_skipped", "N/A")}` |
| Broker connection | `{snapshot.get("live_preflight_broker_connection_status", "N/A")}` |
| Pipeline snapshot status | `{snapshot.get("live_preflight_pipeline_snapshot_status", "N/A")}` |
| Blocking items | `{snapshot.get("live_preflight_blocking_items_count", "N/A")}` |
| Monitor items | `{snapshot.get("live_preflight_monitor_items_count", "N/A")}` |
| Active target holdings | `{snapshot.get("live_preflight_active_target_count", "N/A")}` |
| Review action rows | `{snapshot.get("live_preflight_stock_target_review_action_count", "N/A")}` |
| Report | `{snapshot.get("live_preflight_report_path", "N/A")}` |
| Checklist | `{snapshot.get("live_preflight_checklist_path", "N/A")}` |
| Error | `{snapshot.get("live_preflight_error", "N/A")}` |

## Trading-Day Gate

| Item | Value |
| --- | ---: |
| Gate status | `{snapshot.get("trading_day_gate_status", "N/A")}` |
| A-share trading day | `{snapshot.get("is_a_share_trading_day", "N/A")}` |
| Evidence | `{snapshot.get("trading_day_evidence", "N/A")}` |
| After-close cutoff reached | `{snapshot.get("is_after_close_cutoff_reached", "N/A")}` |
| After-close data status | `{snapshot.get("after_close_data_status", "N/A")}` |
| Latest local equity date | {snapshot.get("trading_day_gate_latest_local_equity_date", "N/A")} |
| Gate reason | `{snapshot.get("trading_day_gate_reason", "N/A")}` |
| Gate action | `{snapshot.get("trading_day_gate_action", "N/A")}` |

## History

| Item | Value |
| --- | ---: |
| History status | `{snapshot.get("history_status", "N/A")}` |
| History file | `{snapshot.get("history_path", "N/A")}` |
| History rows | {snapshot.get("history_row_count", "N/A")} |
| Previous as-of date | {snapshot.get("history_previous_as_of_date", "N/A")} |
| Posture changes | {snapshot.get("history_change_count", "N/A")} |
| Status change summary | `{snapshot.get("history_status_change_summary", "N/A")}` |
| Final equity change | {_signed_num(snapshot.get("history_paper_final_equity_change"), 2)} |
| Total return change | {_signed_pct(snapshot.get("history_paper_total_return_change"))} |
| Satellite weight change | {_signed_pct(snapshot.get("history_paper_satellite_weight_change"))} |

## History Review

| Item | Value |
| --- | ---: |
| Review status | `{snapshot.get("history_review_status", "N/A")}` |
| Health state | `{snapshot.get("history_review_health_state", "N/A")}` |
| Alerts | {snapshot.get("history_review_alert_count", "N/A")} |
| Latest history as-of | {snapshot.get("history_review_latest_as_of_date", "N/A")} |
| Review report | `{snapshot.get("history_review_report_path", "N/A")}` |
| Alert refresh | `{snapshot.get("history_review_alert_refresh_status", "N/A")}` |
| Dashboard alert refresh | `{snapshot.get("dashboard_alert_refresh_status", "N/A")}` |
| Dashboard history refresh | `{snapshot.get("dashboard_history_refresh_status", "N/A")}` |
| Dashboard pipeline-history state | `{snapshot.get("dashboard_pipeline_history_health_state", "N/A")}` |

## Alerts

| Item | Value |
| --- | ---: |
| Alert level | `{snapshot.get("alert_level", "N/A")}` |
| Action stage | `{snapshot.get("alert_action_stage", "N/A")}` |
| Action required | `{snapshot.get("alert_action_required", "N/A")}` |
| Action summary | `{snapshot.get("alert_action_summary", "N/A")}` |
| Critical / warning / info | {snapshot.get("alert_critical_count", "N/A")} / {snapshot.get("alert_warning_count", "N/A")} / {snapshot.get("alert_info_count", "N/A")} |
| Alert report | `{snapshot.get("alerts_report_path", "N/A")}` |

## Output Files

- Daily check: `{snapshot.get("daily_check_report_path")}`
- Paper account: `{snapshot.get("paper_account_report_path")}`
- Paper target holdings: `{snapshot.get("paper_target_holdings_report_path")}`
- Paper stock targets: `{snapshot.get("paper_stock_targets_report_path")}`
- Paper stock target review: `{snapshot.get("paper_stock_target_review_report_path")}`
- Paper stock target review actions: `{snapshot.get("paper_stock_target_review_actions_report_path")}`
- Paper stock target review assistant: `{snapshot.get("paper_stock_target_review_assistant_report_path")}`
- Paper stock target review decision template: `{snapshot.get("paper_stock_target_review_decision_template_report_path")}`
- Paper stock target review decision template XLSX: `{snapshot.get("paper_stock_target_review_decision_template_xlsx_path")}`
- Paper stock target review outcomes: `{snapshot.get("paper_stock_target_review_outcomes_report_path")}`
- Paper stock target review outcome history: `{snapshot.get("paper_stock_target_review_outcomes_history_report_path")}`
- Paper stock target review outcome analysis: `{snapshot.get("paper_stock_target_review_outcome_analysis_report_path")}`
- Paper stock target review outcome calendar: `{snapshot.get("paper_stock_target_review_outcome_calendar_report_path")}`
- Paper stock target review outcome due queue: `{snapshot.get("paper_stock_target_review_outcome_due_report_path")}`
- Paper stock target review notes: `{snapshot.get("paper_stock_target_review_notes_path")}`
- Dashboard: `{snapshot.get("dashboard_report_path")}`
- Model build audit: `{snapshot.get("model_audit_report_path")}`
- Allocator promotion review: `{snapshot.get("promotion_report_path")}`
- Satellite risk-budget review: `{snapshot.get("satellite_risk_budget_report_path")}`
- Satellite trial replay: `{snapshot.get("satellite_trial_replay_report_path")}`
- Momentum focus pool: `{snapshot.get("momentum_focus_report_path")}`
- Chip-reversal candidate outcomes: `{snapshot.get("chip_reversal_candidate_outcomes_report_path")}`
- Live shadow plan: `{snapshot.get("live_shadow_report_path")}`
- Live shadow preflight: `{snapshot.get("live_shadow_preflight_report_path")}`
- Live preflight: `{snapshot.get("live_preflight_report_path")}`
- Daily preflight skipped: `{snapshot.get("daily_preflight_skipped")}`
- Alerts: `{snapshot.get("alerts_report_path")}`
- Pipeline snapshot: `{snapshot.get("pipeline_snapshot_path")}`
"""


def run_daily_pipeline(
    project_root: str | Path = Path("."),
    output_dir: str | Path | None = None,
    daily_check_output_dir: str | Path | None = None,
    paper_account_output_dir: str | Path | None = None,
    dashboard_output_dir: str | Path | None = None,
    momentum_focus_output_dir: str | Path | None = None,
    components_path: str | Path | None = None,
    allocator_dir: str | Path | None = None,
    portfolio_config_path: str | Path | None = None,
    sentiment_dir: str | Path | None = None,
    data_cache_dir: str | Path | None = None,
    trigger_report: str | Path | None = None,
    model_audit_dir: str | Path | None = None,
    history_path: str | Path | None = DEFAULT_PIPELINE_HISTORY,
    history_review_output_dir: str | Path | None = None,
    satellite_risk_budget_output_dir: str | Path | None = None,
    satellite_trial_replay_output_dir: str | Path | None = None,
    satellite_trial_replay_horizons: str | Iterable[str] = DEFAULT_SATELLITE_TRIAL_REPLAY_HORIZONS,
    network_lab_snapshot: str | Path | None = None,
    network_max_cluster_count_warning: int = 1,
    network_residual_mi_warning: float = 0.20,
    live_shadow_output_dir: str | Path | None = None,
    live_shadow_holdings_file: str | Path | None = None,
    live_shadow_prices_file: str | Path | None = None,
    live_shadow_cash: float | None = None,
    live_shadow_lot_size: int = 100,
    live_shadow_min_trade_value: float = 1000.0,
    live_shadow_max_position_weight: float = 0.20,
    live_shadow_max_gross_exposure: float = 1.0,
    live_shadow_preflight: bool = False,
    live_shadow_preflight_fail_on_blocked: bool = True,
    phase2_review_dir: str | Path | None = None,
    promotion_review_dir: str | Path | None = None,
    max_staleness_days: int = 3,
    min_cache_fresh_ratio: float = 0.90,
    rebalance_cost_rate: float = 0.0,
    stock_market_cap_path: str | Path | None = None,
    stock_tracking_max_market_cap_yi: float = 0.0,
    stock_review_notes_path: str | Path | None = None,
    stock_review_outcomes_history_path: str | Path | None = None,
    stock_review_drawdown_threshold: float = -0.10,
    stock_review_watch_drawdown_threshold: float = -0.07,
    stock_review_loss_attention_threshold: float = -0.05,
    stock_review_gain_attention_threshold: float = 0.10,
    stock_review_watch_score_threshold: float = 30.0,
    stock_review_outcome_min_evaluable: int = 20,
    stock_review_outcome_min_group_evaluable: int = 5,
    stock_review_warning_only_after_close: bool = False,
    live_preflight_output_dir: str | Path | None = None,
    live_shadow_review_decisions_file: str | Path | None = None,
    run_history_review: bool = True,
    history_review_lookback_runs: int = 20,
    history_review_drawdown_watch_threshold: float = -0.08,
    history_review_min_sharpe_watch: float = 0.5,
    as_of_date: str | date | None = None,
    date_stamp: bool = True,
    momentum_focus_board_scope: str = "main_chinext",
    momentum_focus_strong_gain_threshold_pct: float = 5.0,
    momentum_focus_outcome_summary_path: str | Path | None = None,
    momentum_focus_target_horizon: int = 5,
    chip_reversal_candidate_outcomes: bool = False,
    chip_reversal_candidates_path: str | Path | None = None,
    chip_reversal_outcomes_data_dir: str | Path | None = None,
    chip_reversal_outcomes_output_dir: str | Path | None = None,
    chip_reversal_outcome_horizons: str | Iterable[int] = (1, 2),
    chip_reversal_outcome_success_threshold: float = 0.03,
    chip_reversal_outcome_min_ready_per_horizon: int = 30,
    chip_reversal_outcome_min_success_rate: float = 0.55,
) -> DailyPipelineResult:
    root = Path(project_root).resolve()
    as_of = _parse_date(as_of_date) if as_of_date is not None else datetime.now().date()
    if as_of is None:
        raise ValueError(f"Invalid as_of_date: {as_of_date}")

    resolved_output = _resolve_dated_output(root, output_dir, DEFAULT_PIPELINE_OUTPUT, DEFAULT_PIPELINE_BASE, as_of, date_stamp)
    resolved_daily_output = _resolve_dated_output(
        root,
        daily_check_output_dir,
        Path("outputs/research/daily_model_check_latest"),
        DEFAULT_DAILY_CHECK_BASE,
        as_of,
        date_stamp,
    )
    resolved_paper_output = _resolve_dated_output(
        root,
        paper_account_output_dir,
        Path("outputs/research/paper_account_latest"),
        DEFAULT_PAPER_ACCOUNT_BASE,
        as_of,
        date_stamp,
    )
    resolved_dashboard_output = _resolve_dated_output(
        root,
        dashboard_output_dir,
        Path("outputs/research/latest_dashboard"),
        DEFAULT_DASHBOARD_BASE,
        as_of,
        date_stamp,
    )
    resolved_momentum_focus_output = _resolve_dated_output(
        root,
        momentum_focus_output_dir,
        DEFAULT_MOMENTUM_FOCUS_OUTPUT,
        DEFAULT_MOMENTUM_FOCUS_BASE,
        as_of,
        date_stamp,
    )
    resolved_chip_reversal_outcomes_output = _resolve_dated_output(
        root,
        chip_reversal_outcomes_output_dir,
        DEFAULT_CHIP_REVERSAL_OUTCOMES_OUTPUT,
        DEFAULT_CHIP_REVERSAL_OUTCOMES_BASE,
        as_of,
        date_stamp,
    )
    resolved_chip_reversal_candidates_path = _resolve(
        root,
        chip_reversal_candidates_path,
        DEFAULT_CHIP_REVERSAL_CANDIDATES_PATH,
    )
    resolved_chip_reversal_outcomes_data_dir = _resolve(
        root,
        chip_reversal_outcomes_data_dir,
        DEFAULT_CHIP_REVERSAL_OUTCOMES_DATA_DIR,
    )
    resolved_satellite_risk_budget_output = _resolve_dated_output(
        root,
        satellite_risk_budget_output_dir,
        DEFAULT_SATELLITE_RISK_BUDGET_OUTPUT,
        DEFAULT_SATELLITE_RISK_BUDGET_BASE,
        as_of,
        date_stamp,
    )
    resolved_satellite_trial_replay_output = _resolve_dated_output(
        root,
        satellite_trial_replay_output_dir,
        DEFAULT_SATELLITE_TRIAL_REPLAY_OUTPUT,
        DEFAULT_SATELLITE_TRIAL_REPLAY_BASE,
        as_of,
        date_stamp,
    )
    resolved_live_shadow_output = _resolve_dated_output(
        root,
        live_shadow_output_dir,
        DEFAULT_LIVE_SHADOW_OUTPUT,
        DEFAULT_LIVE_SHADOW_BASE,
        as_of,
        date_stamp,
    )
    resolved_live_shadow_preflight_output = _resolve_dated_output(
        root,
        None,
        DEFAULT_LIVE_SHADOW_PREFLIGHT_OUTPUT,
        DEFAULT_LIVE_SHADOW_PREFLIGHT_BASE,
        as_of,
        date_stamp,
    )
    resolved_live_preflight_output = _resolve_dated_output(
        root,
        live_preflight_output_dir,
        DEFAULT_LIVE_PRE_FLIGHT_OUTPUT,
        DEFAULT_LIVE_PRE_FLIGHT_BASE,
        as_of,
        date_stamp,
    )
    resolved_momentum_focus_summary_path = _resolve(
        root,
        momentum_focus_outcome_summary_path,
        DEFAULT_OUTCOME_SUMMARY_PATH,
    )
    resolved_live_shadow_review_decisions_file = (
        _resolve(root, live_shadow_review_decisions_file, Path(str(live_shadow_review_decisions_file)))
        if live_shadow_review_decisions_file is not None
        else None
    )

    daily_result = run_daily_model_check(
        project_root=root,
        output_dir=resolved_daily_output,
        phase2_review_dir=phase2_review_dir,
        promotion_review_dir=promotion_review_dir,
        components_path=components_path,
        allocator_dir=allocator_dir,
        max_staleness_days=max_staleness_days,
        as_of_date=as_of,
        date_stamp=False,
    )
    dashboard_summary_cache: dict[str, dict[str, Any]] = {}
    shared_market_snapshot = None
    if _should_run_momentum_focus(data_cache_dir):
        try:
            shared_market_snapshot = load_market_snapshot_rows(
                daily_data_dir=data_cache_dir,
                require_success=True,
            )
        except (FileNotFoundError, ImportError, RuntimeError, ValueError):
            shared_market_snapshot = None
    try:
        paper_result = run_paper_account(
            project_root=root,
            config_path=portfolio_config_path,
            allocator_dir=allocator_dir,
            output_dir=resolved_paper_output,
            rebalance_cost_rate=rebalance_cost_rate,
            stock_market_cap_path=stock_market_cap_path,
            stock_tracking_max_market_cap_yi=stock_tracking_max_market_cap_yi,
            stock_review_notes_path=stock_review_notes_path,
            stock_review_outcomes_history_path=stock_review_outcomes_history_path,
            stock_review_drawdown_threshold=stock_review_drawdown_threshold,
            stock_review_watch_drawdown_threshold=stock_review_watch_drawdown_threshold,
            stock_review_loss_attention_threshold=stock_review_loss_attention_threshold,
            stock_review_gain_attention_threshold=stock_review_gain_attention_threshold,
            stock_review_watch_score_threshold=stock_review_watch_score_threshold,
            stock_review_outcome_min_evaluable=stock_review_outcome_min_evaluable,
            stock_review_outcome_min_group_evaluable=stock_review_outcome_min_group_evaluable,
            market_snapshot=shared_market_snapshot,
        )
        paper_account_status = "ok"
        paper_account_error = None
    except Exception as error:  # Paper-account failures should not block the daily research report.
        paper_account_status = "failed"
        paper_account_error = str(error)
        paper_result = _failed_paper_account_result(resolved_paper_output, as_of, paper_account_error)
    dashboard_artifacts_published = not run_history_review
    dashboard_publish_attempted = dashboard_artifacts_published
    dashboard_result = run_daily_dashboard(
        project_root=root,
        output_dir=resolved_dashboard_output,
        daily_check_dir=daily_result.output_dir,
        paper_account_dir=paper_result.output_dir,
        sentiment_dir=sentiment_dir,
        data_cache_dir=data_cache_dir,
        allocator_dir=allocator_dir,
        trigger_report=trigger_report,
        model_audit_dir=model_audit_dir,
        as_of_date=as_of,
        max_staleness_days=max_staleness_days,
        min_cache_fresh_ratio=min_cache_fresh_ratio,
        date_stamp=False,
        summary_cache=dashboard_summary_cache,
        market_snapshot=shared_market_snapshot,
        publish_artifacts=dashboard_artifacts_published,
    )

    momentum_focus_result: MomentumFocusResult | None = None
    if _should_run_momentum_focus(data_cache_dir):
        try:
            momentum_trade_date = _momentum_focus_trade_date(as_of, daily_result.snapshot, dashboard_result.snapshot)
            momentum_focus_result = run_momentum_focus(
                project_root=root,
                output_dir=resolved_momentum_focus_output,
                daily_data_dir=data_cache_dir,
                as_of_date=as_of,
                trade_date=momentum_trade_date,
                board_scope=momentum_focus_board_scope,
                strong_gain_threshold_pct=momentum_focus_strong_gain_threshold_pct,
                outcome_summary_path=resolved_momentum_focus_summary_path,
                target_horizon=momentum_focus_target_horizon,
                excluded_codes=set(
                    paper_result.stock_target_review.loc[
                        paper_result.stock_target_review.get("observation_excluded", False).astype(bool),
                        "code",
                    ].astype(str)
                )
                if not paper_result.stock_target_review.empty
                and "observation_excluded" in paper_result.stock_target_review.columns
                else set(),
                market_snapshot=shared_market_snapshot,
            )
            momentum_focus_snapshot = _momentum_focus_context(momentum_focus_result)
        except Exception as error:  # Momentum focus is research context and should not block the core daily report.
            momentum_focus_snapshot = {
                "momentum_focus_status": "failed",
                "momentum_focus_output_dir": str(resolved_momentum_focus_output),
                "momentum_focus_report_path": str(resolved_momentum_focus_output / "momentum_focus.md"),
                "momentum_focus_candidates_path": str(resolved_momentum_focus_output / "momentum_focus_candidates.csv"),
                "momentum_focus_snapshot_path": str(resolved_momentum_focus_output / "momentum_focus_snapshot.json"),
                "momentum_focus_candidate_count": 0,
                "momentum_focus_market_data_quality_status": "error",
                "momentum_focus_volume_positive_count": 0,
                "momentum_focus_turnover_positive_count": 0,
                "momentum_focus_volume_coverage_ratio": 0.0,
                "momentum_focus_turnover_coverage_ratio": 0.0,
                "momentum_focus_limit_up_count": 0,
                "momentum_focus_strong_gain_count": 0,
                "momentum_focus_excluded_special_treatment_count": 0,
                "momentum_focus_excluded_by_review_required_count": 0,
                "momentum_focus_source_kind": None,
                "momentum_focus_trade_date": None,
                "momentum_focus_board_scope": momentum_focus_board_scope,
                "momentum_focus_strong_gain_threshold_pct": momentum_focus_strong_gain_threshold_pct,
                "momentum_focus_target_horizon": momentum_focus_target_horizon,
                "momentum_focus_outcome_summary_path": str(resolved_momentum_focus_summary_path),
                "momentum_focus_outcome_prior_rows": 0,
                "momentum_focus_outcome_prior_matched_count": 0,
                "momentum_focus_outcome_summary_rows": 0,
                "momentum_focus_name_map_status": "unknown",
                "momentum_focus_name_map_rows": 0,
                "momentum_focus_name_map_path": None,
                "momentum_focus_broker_action": "none",
                "momentum_focus_research_only": True,
                "momentum_focus_error": str(error),
            }
    else:
        momentum_focus_snapshot = {
            "momentum_focus_status": "skipped_non_daily_market_data",
            "momentum_focus_output_dir": str(resolved_momentum_focus_output),
            "momentum_focus_report_path": str(resolved_momentum_focus_output / "momentum_focus.md"),
            "momentum_focus_candidates_path": str(resolved_momentum_focus_output / "momentum_focus_candidates.csv"),
            "momentum_focus_snapshot_path": str(resolved_momentum_focus_output / "momentum_focus_snapshot.json"),
            "momentum_focus_candidate_count": 0,
            "momentum_focus_market_data_quality_status": "not_run",
            "momentum_focus_volume_positive_count": 0,
            "momentum_focus_turnover_positive_count": 0,
            "momentum_focus_volume_coverage_ratio": 0.0,
            "momentum_focus_turnover_coverage_ratio": 0.0,
            "momentum_focus_limit_up_count": 0,
            "momentum_focus_strong_gain_count": 0,
            "momentum_focus_excluded_special_treatment_count": 0,
            "momentum_focus_excluded_by_review_required_count": 0,
            "momentum_focus_source_kind": None,
            "momentum_focus_trade_date": None,
            "momentum_focus_board_scope": momentum_focus_board_scope,
            "momentum_focus_strong_gain_threshold_pct": momentum_focus_strong_gain_threshold_pct,
            "momentum_focus_target_horizon": momentum_focus_target_horizon,
            "momentum_focus_outcome_summary_path": str(resolved_momentum_focus_summary_path),
            "momentum_focus_outcome_prior_rows": 0,
            "momentum_focus_outcome_prior_matched_count": 0,
            "momentum_focus_outcome_summary_rows": 0,
            "momentum_focus_name_map_status": "unknown",
            "momentum_focus_name_map_rows": 0,
            "momentum_focus_name_map_path": None,
            "momentum_focus_broker_action": "none",
            "momentum_focus_research_only": True,
            "momentum_focus_error": None,
        }

    chip_reversal_candidate_outcomes_result: ChipReversalCandidateOutcomesResult | None = None
    if chip_reversal_candidate_outcomes:
        try:
            chip_reversal_outcome_horizon_values = _parse_int_values(chip_reversal_outcome_horizons)
            chip_reversal_candidate_outcomes_result = run_chip_reversal_candidate_outcomes(
                project_root=root,
                candidates_path=resolved_chip_reversal_candidates_path,
                data_dir=resolved_chip_reversal_outcomes_data_dir,
                output_dir=resolved_chip_reversal_outcomes_output,
                horizons=chip_reversal_outcome_horizon_values,
                success_threshold=chip_reversal_outcome_success_threshold,
                min_ready_per_horizon=chip_reversal_outcome_min_ready_per_horizon,
                min_success_rate=chip_reversal_outcome_min_success_rate,
                market_snapshot_overlay=True,
                daily_data_dir=data_cache_dir,
            )
            chip_reversal_candidate_outcomes_snapshot = _chip_reversal_candidate_outcomes_context(
                chip_reversal_candidate_outcomes_result
            )
        except (OSError, ValueError, RuntimeError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            chip_reversal_candidate_outcomes_snapshot = {
                "chip_reversal_candidate_outcomes_status": "failed",
                "chip_reversal_candidate_outcomes_output_dir": str(resolved_chip_reversal_outcomes_output),
                "chip_reversal_candidate_outcomes_outcomes_path": str(
                    resolved_chip_reversal_outcomes_output / "chip_reversal_candidate_outcomes.csv"
                ),
                "chip_reversal_candidate_outcomes_summary_path": str(
                    resolved_chip_reversal_outcomes_output / "chip_reversal_candidate_outcome_summary.csv"
                ),
                "chip_reversal_candidate_outcomes_group_summary_path": str(
                    resolved_chip_reversal_outcomes_output / "chip_reversal_candidate_outcome_group_summary.csv"
                ),
                "chip_reversal_candidate_outcomes_snapshot_path": str(
                    resolved_chip_reversal_outcomes_output / "chip_reversal_candidate_outcomes_snapshot.json"
                ),
                "chip_reversal_candidate_outcomes_report_path": str(
                    resolved_chip_reversal_outcomes_output / "chip_reversal_candidate_outcomes.md"
                ),
                "chip_reversal_candidate_outcomes_candidate_count": 0,
                "chip_reversal_candidate_outcomes_ready_count": 0,
                "chip_reversal_candidate_outcomes_pending_count": 0,
                "chip_reversal_candidate_outcomes_readiness_status": "failed",
                "chip_reversal_candidate_outcomes_analysis_status": "failed",
                "chip_reversal_candidate_outcomes_ready_horizons": [],
                "chip_reversal_candidate_outcomes_pending_horizons": [],
                "chip_reversal_candidate_outcomes_next_review_horizon": None,
                "chip_reversal_candidate_outcomes_next_review_reason": str(error),
                "chip_reversal_candidate_outcomes_sample_warning": "failed",
                "chip_reversal_candidate_outcomes_latest_market_trade_date": None,
                "chip_reversal_candidate_outcomes_market_source_kind": None,
                "chip_reversal_candidate_outcomes_market_trade_date": None,
                "chip_reversal_candidate_outcomes_promotion_gate_status": None,
                "chip_reversal_candidate_outcomes_promotion_gate_reasons": [],
                "chip_reversal_candidate_outcomes_broker_action": "none",
                "chip_reversal_candidate_outcomes_research_only": True,
                "chip_reversal_candidate_outcomes_error": str(error),
            }
    else:
        chip_reversal_candidate_outcomes_snapshot = {
            "chip_reversal_candidate_outcomes_status": "disabled",
            "chip_reversal_candidate_outcomes_output_dir": str(resolved_chip_reversal_outcomes_output),
            "chip_reversal_candidate_outcomes_outcomes_path": str(
                resolved_chip_reversal_outcomes_output / "chip_reversal_candidate_outcomes.csv"
            ),
            "chip_reversal_candidate_outcomes_summary_path": str(
                resolved_chip_reversal_outcomes_output / "chip_reversal_candidate_outcome_summary.csv"
            ),
            "chip_reversal_candidate_outcomes_group_summary_path": str(
                resolved_chip_reversal_outcomes_output / "chip_reversal_candidate_outcome_group_summary.csv"
            ),
            "chip_reversal_candidate_outcomes_snapshot_path": str(
                resolved_chip_reversal_outcomes_output / "chip_reversal_candidate_outcomes_snapshot.json"
            ),
            "chip_reversal_candidate_outcomes_report_path": str(
                resolved_chip_reversal_outcomes_output / "chip_reversal_candidate_outcomes.md"
            ),
            "chip_reversal_candidate_outcomes_candidate_count": 0,
            "chip_reversal_candidate_outcomes_ready_count": 0,
            "chip_reversal_candidate_outcomes_pending_count": 0,
            "chip_reversal_candidate_outcomes_readiness_status": "disabled",
            "chip_reversal_candidate_outcomes_analysis_status": "disabled",
            "chip_reversal_candidate_outcomes_ready_horizons": [],
            "chip_reversal_candidate_outcomes_pending_horizons": [],
            "chip_reversal_candidate_outcomes_next_review_horizon": None,
            "chip_reversal_candidate_outcomes_next_review_reason": None,
            "chip_reversal_candidate_outcomes_sample_warning": None,
            "chip_reversal_candidate_outcomes_latest_market_trade_date": None,
            "chip_reversal_candidate_outcomes_market_source_kind": None,
            "chip_reversal_candidate_outcomes_market_trade_date": None,
            "chip_reversal_candidate_outcomes_promotion_gate_status": None,
            "chip_reversal_candidate_outcomes_promotion_gate_reasons": [],
            "chip_reversal_candidate_outcomes_broker_action": "none",
            "chip_reversal_candidate_outcomes_research_only": True,
            "chip_reversal_candidate_outcomes_error": None,
        }

    snapshot: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": as_of.isoformat(),
        "max_staleness_days": int(max_staleness_days),
        "daily_check_output_dir": str(daily_result.output_dir),
        "paper_account_output_dir": str(paper_result.output_dir),
        "dashboard_output_dir": str(dashboard_result.output_dir),
        "daily_check_report_path": str(daily_result.report_path),
        "paper_account_report_path": str(paper_result.report_path),
        "dashboard_report_path": str(dashboard_result.report_path),
        "data_freshness_status": daily_result.snapshot.get("data_freshness_status"),
        "action_posture": daily_result.snapshot.get("action_posture"),
        "phase2_posture": daily_result.snapshot.get("phase2_posture"),
        "local_equity_latest_date": daily_result.snapshot.get("latest_date"),
        "paper_latest_date": paper_result.metrics.get("latest_date"),
        "paper_latest_regime": paper_result.metrics.get("latest_regime"),
        "paper_latest_candidate": paper_result.metrics.get("latest_candidate"),
        "paper_latest_core_weight": paper_result.metrics.get("latest_core_weight"),
        "paper_latest_satellite_weight": paper_result.metrics.get("latest_satellite_weight"),
        "paper_latest_cash_weight": paper_result.metrics.get("latest_cash_weight"),
        "paper_target_holdings_path": str(paper_result.target_holdings_path),
        "paper_target_holdings_json_path": str(paper_result.target_holdings_json_path),
        "paper_target_holdings_report_path": str(paper_result.target_holdings_report_path),
        "paper_target_holding_count": paper_result.target_holdings_payload.get("target_holding_count"),
        "paper_active_target_count": paper_result.target_holdings_payload.get("active_target_count"),
        "paper_target_broker_action": paper_result.target_holdings_payload.get("broker_action"),
        "paper_stock_targets_path": str(paper_result.stock_targets_path),
        "paper_stock_targets_json_path": str(paper_result.stock_targets_json_path),
        "paper_stock_targets_report_path": str(paper_result.stock_targets_report_path),
        "paper_stock_target_count": paper_result.stock_targets_payload.get("stock_target_count"),
        "paper_stock_target_source_count": paper_result.stock_targets_payload.get("source_stock_target_count"),
        "paper_active_stock_target_count": paper_result.stock_targets_payload.get("active_stock_target_count"),
        "paper_suppressed_stock_target_count": paper_result.stock_targets_payload.get("suppressed_stock_count"),
        "paper_stock_target_review_excluded_count": paper_result.stock_targets_payload.get("review_required_excluded_count"),
        "paper_stock_tracking_max_market_cap_yi": paper_result.stock_targets_payload.get("stock_tracking_max_market_cap_yi"),
        "paper_stock_tracking_excluded_large_market_cap_count": paper_result.stock_targets_payload.get("stock_tracking_excluded_large_market_cap_count"),
        "paper_stock_tracking_allowed_count": paper_result.stock_targets_payload.get("stock_tracking_allowed_count"),
        "paper_stock_tracking_market_cap_missing_count": paper_result.stock_targets_payload.get("stock_tracking_market_cap_missing_count"),
        "paper_stock_market_cap_cache_status": paper_result.stock_targets_payload.get("stock_market_cap_cache_status"),
        "paper_stock_market_cap_path": paper_result.stock_targets_payload.get("stock_market_cap_path"),
        "paper_stock_market_cap_cache_row_count": paper_result.stock_targets_payload.get("stock_market_cap_cache_row_count"),
        "paper_stock_market_cap_cache_latest_snapshot_date": paper_result.stock_targets_payload.get("stock_market_cap_cache_latest_snapshot_date"),
        "paper_stock_market_cap_cache_updated_at": paper_result.stock_targets_payload.get("stock_market_cap_cache_updated_at"),
        "paper_stock_target_review_path": str(paper_result.stock_target_review_path),
        "paper_stock_target_review_json_path": str(paper_result.stock_target_review_json_path),
        "paper_stock_target_review_report_path": str(paper_result.stock_target_review_report_path),
        "paper_stock_target_review_actions_path": str(paper_result.stock_target_review_actions_path),
        "paper_stock_target_review_actions_json_path": str(paper_result.stock_target_review_actions_json_path),
        "paper_stock_target_review_actions_report_path": str(paper_result.stock_target_review_actions_report_path),
        "paper_stock_target_review_assistant_path": str(paper_result.stock_target_review_assistant_path),
        "paper_stock_target_review_assistant_json_path": str(paper_result.stock_target_review_assistant_json_path),
        "paper_stock_target_review_assistant_report_path": str(paper_result.stock_target_review_assistant_report_path),
        "paper_stock_target_review_decision_template_path": str(paper_result.stock_target_review_decision_template_path),
        "paper_stock_target_review_decision_template_json_path": str(paper_result.stock_target_review_decision_template_json_path),
        "paper_stock_target_review_decision_template_report_path": str(paper_result.stock_target_review_decision_template_report_path),
        "paper_stock_target_review_decision_template_xlsx_path": str(paper_result.stock_target_review_decision_template_xlsx_path),
        "paper_stock_target_review_outcomes_path": str(paper_result.stock_target_review_outcomes_path),
        "paper_stock_target_review_outcomes_json_path": str(paper_result.stock_target_review_outcomes_json_path),
        "paper_stock_target_review_outcomes_report_path": str(paper_result.stock_target_review_outcomes_report_path),
        "paper_stock_target_review_outcomes_history_path": str(paper_result.stock_target_review_outcomes_history_path),
        "paper_stock_target_review_outcomes_history_snapshot_path": str(paper_result.stock_target_review_outcomes_history_snapshot_path),
        "paper_stock_target_review_outcomes_history_json_path": str(paper_result.stock_target_review_outcomes_history_json_path),
        "paper_stock_target_review_outcomes_history_report_path": str(paper_result.stock_target_review_outcomes_history_report_path),
        "paper_stock_target_review_outcome_analysis_path": str(paper_result.stock_target_review_outcome_analysis_path),
        "paper_stock_target_review_outcome_analysis_json_path": str(paper_result.stock_target_review_outcome_analysis_json_path),
        "paper_stock_target_review_outcome_analysis_report_path": str(paper_result.stock_target_review_outcome_analysis_report_path),
        "paper_stock_target_review_outcome_calendar_path": str(paper_result.stock_target_review_outcome_calendar_path),
        "paper_stock_target_review_outcome_calendar_json_path": str(paper_result.stock_target_review_outcome_calendar_json_path),
        "paper_stock_target_review_outcome_calendar_report_path": str(paper_result.stock_target_review_outcome_calendar_report_path),
        "paper_stock_target_review_outcome_due_path": str(paper_result.stock_target_review_outcome_due_path),
        "paper_stock_target_review_outcome_due_json_path": str(paper_result.stock_target_review_outcome_due_json_path),
        "paper_stock_target_review_outcome_due_report_path": str(paper_result.stock_target_review_outcome_due_report_path),
        "paper_stock_target_review_notes_path": str(paper_result.stock_target_review_notes_path),
        "paper_stock_target_review_notes_snapshot_path": str(paper_result.stock_target_review_notes_snapshot_path),
        "paper_stock_target_review_required_count": paper_result.stock_target_review_payload.get("review_required_count"),
        "paper_stock_target_review_monitor_count": paper_result.stock_target_review_payload.get("monitor_count"),
        "paper_stock_target_review_trigger_count": paper_result.stock_target_review_payload.get("trigger_review_count"),
        "paper_stock_target_review_drawdown_count": paper_result.stock_target_review_payload.get("drawdown_review_count"),
        "paper_stock_target_review_suppressed_layer_count": paper_result.stock_target_review_payload.get("suppressed_layer_review_count"),
        "paper_stock_target_review_watch_count": paper_result.stock_target_review_payload.get("watch_review_count"),
        "paper_stock_target_review_manual_note_count": paper_result.stock_target_review_payload.get("manual_note_count"),
        "paper_stock_target_review_unreviewed_count": paper_result.stock_target_review_payload.get("unreviewed_count"),
        "paper_stock_target_review_required_unreviewed_count": paper_result.stock_target_review_payload.get("review_required_unreviewed_count"),
        "paper_stock_target_review_manual_pending_count": paper_result.stock_target_review_payload.get("manual_pending_count"),
        "paper_stock_target_review_manual_reviewed_count": paper_result.stock_target_review_payload.get("manual_reviewed_count"),
        "paper_stock_target_review_manual_watch_count": paper_result.stock_target_review_payload.get("manual_watch_count"),
        "paper_stock_target_review_manual_resolved_count": paper_result.stock_target_review_payload.get("manual_resolved_count"),
        "paper_stock_target_review_manual_exclude_candidate_count": paper_result.stock_target_review_payload.get("manual_exclude_candidate_count"),
        "paper_stock_target_review_manual_other_status_count": paper_result.stock_target_review_payload.get("manual_other_status_count"),
        "paper_stock_target_review_action_count": paper_result.stock_target_review_actions_payload.get("action_count"),
        "paper_stock_target_review_action_pending_model_count": paper_result.stock_target_review_actions_payload.get("review_required_pending_count"),
        "paper_stock_target_review_action_manual_watch_count": paper_result.stock_target_review_actions_payload.get("manual_watch_followup_count"),
        "paper_stock_target_review_action_manual_exclusion_count": paper_result.stock_target_review_actions_payload.get("manual_exclusion_candidate_count"),
        "paper_stock_target_review_action_status_normalization_count": paper_result.stock_target_review_actions_payload.get("manual_status_normalization_count"),
        "paper_stock_target_review_action_next_review_due_count": paper_result.stock_target_review_actions_payload.get("manual_next_review_due_count"),
        "paper_stock_target_review_assistant_count": paper_result.stock_target_review_assistant_payload.get("assistant_row_count"),
        "paper_stock_target_review_assistant_pending_count": paper_result.stock_target_review_assistant_payload.get("review_required_pending_count"),
        "paper_stock_target_review_assistant_price_ok_count": paper_result.stock_target_review_assistant_payload.get("price_history_ok_count"),
        "paper_stock_target_review_assistant_price_stale_count": paper_result.stock_target_review_assistant_payload.get("price_history_stale_count"),
        "paper_stock_target_review_assistant_price_missing_count": paper_result.stock_target_review_assistant_payload.get("price_history_missing_count"),
        "paper_stock_target_review_decision_template_count": paper_result.stock_target_review_decision_template_payload.get("decision_template_row_count"),
        "paper_stock_target_review_decision_template_blank_status_count": paper_result.stock_target_review_decision_template_payload.get("blank_manual_status_count"),
        "paper_stock_target_review_outcome_row_count": paper_result.stock_target_review_outcomes_payload.get("outcome_row_count"),
        "paper_stock_target_review_outcome_complete_count": paper_result.stock_target_review_outcomes_payload.get("complete_count"),
        "paper_stock_target_review_outcome_partial_count": paper_result.stock_target_review_outcomes_payload.get("partial_count"),
        "paper_stock_target_review_outcome_pending_count": paper_result.stock_target_review_outcomes_payload.get("pending_count"),
        "paper_stock_target_review_outcome_missing_entry_price_count": paper_result.stock_target_review_outcomes_payload.get("missing_entry_price_count"),
        "paper_stock_target_review_outcomes_history_row_count": paper_result.stock_target_review_outcomes_history_payload.get("history_row_count"),
        "paper_stock_target_review_outcomes_history_updated_row_count": paper_result.stock_target_review_outcomes_history_payload.get("history_updated_row_count"),
        "paper_stock_target_review_outcomes_history_complete_count": paper_result.stock_target_review_outcomes_history_payload.get("history_complete_count"),
        "paper_stock_target_review_outcomes_history_partial_count": paper_result.stock_target_review_outcomes_history_payload.get("history_partial_count"),
        "paper_stock_target_review_outcomes_history_pending_count": paper_result.stock_target_review_outcomes_history_payload.get("history_pending_count"),
        "paper_stock_target_review_outcomes_history_missing_entry_price_count": paper_result.stock_target_review_outcomes_history_payload.get("history_missing_entry_price_count"),
        "paper_stock_target_review_outcomes_history_latest_review_date": paper_result.stock_target_review_outcomes_history_payload.get("history_latest_review_date"),
        "paper_stock_target_review_outcome_analysis_status": paper_result.stock_target_review_outcome_analysis_payload.get("analysis_status"),
        "paper_stock_target_review_outcome_analysis_row_count": paper_result.stock_target_review_outcome_analysis_payload.get("analysis_row_count"),
        "paper_stock_target_review_outcome_analysis_min_evaluable": paper_result.stock_target_review_outcome_analysis_payload.get("min_evaluable"),
        "paper_stock_target_review_outcome_analysis_min_group_evaluable": paper_result.stock_target_review_outcome_analysis_payload.get("min_group_evaluable"),
        "paper_stock_target_review_outcome_analysis_ready_horizon_count": paper_result.stock_target_review_outcome_analysis_payload.get("ready_horizon_count"),
        "paper_stock_target_review_outcome_analysis_sample_warning": paper_result.stock_target_review_outcome_analysis_payload.get("sample_warning"),
        "paper_stock_target_review_outcome_calendar_row_count": paper_result.stock_target_review_outcome_calendar_payload.get("calendar_row_count"),
        "paper_stock_target_review_outcome_calendar_ready_count": paper_result.stock_target_review_outcome_calendar_payload.get("calendar_ready_count"),
        "paper_stock_target_review_outcome_calendar_pending_count": paper_result.stock_target_review_outcome_calendar_payload.get("calendar_pending_count"),
        "paper_stock_target_review_outcome_calendar_due_count": paper_result.stock_target_review_outcome_calendar_payload.get("calendar_due_count"),
        "paper_stock_target_review_outcome_calendar_due_pending_count": paper_result.stock_target_review_outcome_calendar_payload.get("calendar_due_pending_count"),
        "paper_stock_target_review_outcome_calendar_next_action_date": paper_result.stock_target_review_outcome_calendar_payload.get("next_action_date"),
        "paper_stock_target_review_outcome_calendar_next_action_horizon": paper_result.stock_target_review_outcome_calendar_payload.get("next_action_horizon"),
        "paper_stock_target_review_outcome_calendar_next_due_date": paper_result.stock_target_review_outcome_calendar_payload.get("next_due_date"),
        "paper_stock_target_review_outcome_calendar_next_due_horizon": paper_result.stock_target_review_outcome_calendar_payload.get("next_due_horizon"),
        "paper_stock_target_review_outcome_due_status": paper_result.stock_target_review_outcome_due_payload.get("due_status"),
        "paper_stock_target_review_outcome_due_row_count": paper_result.stock_target_review_outcome_due_payload.get("due_row_count"),
        "paper_stock_target_review_outcome_due_pending_count": paper_result.stock_target_review_outcome_due_payload.get("due_pending_count"),
        "paper_stock_target_review_outcome_due_next_date": paper_result.stock_target_review_outcome_due_payload.get("next_due_date"),
        "paper_stock_target_review_outcome_due_next_horizon": paper_result.stock_target_review_outcome_due_payload.get("next_due_horizon"),
        "paper_stock_target_review_outcome_maturity_next_1d_date": (
            paper_result.stock_target_review_outcome_analysis_payload.get("maturity_forecast") or {}
        ).get("1d", {}).get("estimated_next_evaluable_date"),
        "paper_stock_target_review_outcome_maturity_next_5d_date": (
            paper_result.stock_target_review_outcome_analysis_payload.get("maturity_forecast") or {}
        ).get("5d", {}).get("estimated_next_evaluable_date"),
        "paper_stock_target_review_outcome_maturity_next_10d_date": (
            paper_result.stock_target_review_outcome_analysis_payload.get("maturity_forecast") or {}
        ).get("10d", {}).get("estimated_next_evaluable_date"),
        "paper_stock_target_review_outcome_maturity_next_20d_date": (
            paper_result.stock_target_review_outcome_analysis_payload.get("maturity_forecast") or {}
        ).get("20d", {}).get("estimated_next_evaluable_date"),
        "paper_stock_target_review_outcome_analysis_evaluable_1d_count": (
            paper_result.stock_target_review_outcome_analysis_payload.get("total_evaluable_by_horizon") or {}
        ).get("1d"),
        "paper_stock_target_review_outcome_analysis_evaluable_5d_count": (
            paper_result.stock_target_review_outcome_analysis_payload.get("total_evaluable_by_horizon") or {}
        ).get("5d"),
        "paper_stock_target_review_outcome_analysis_evaluable_10d_count": (
            paper_result.stock_target_review_outcome_analysis_payload.get("total_evaluable_by_horizon") or {}
        ).get("10d"),
        "paper_stock_target_review_outcome_analysis_evaluable_20d_count": (
            paper_result.stock_target_review_outcome_analysis_payload.get("total_evaluable_by_horizon") or {}
        ).get("20d"),
        "paper_stock_target_review_drawdown_threshold": paper_result.stock_target_review_payload.get("drawdown_threshold"),
        "paper_stock_target_review_watch_drawdown_threshold": paper_result.stock_target_review_payload.get("watch_drawdown_threshold"),
        "paper_stock_target_review_loss_attention_threshold": paper_result.stock_target_review_payload.get("loss_attention_threshold"),
        "paper_stock_target_review_gain_attention_threshold": paper_result.stock_target_review_payload.get("gain_attention_threshold"),
        "paper_stock_target_review_warning_only_after_close": bool(stock_review_warning_only_after_close),
        "paper_final_equity": paper_result.metrics.get("final_equity"),
        "paper_total_return": paper_result.metrics.get("total_return"),
        "paper_cagr": paper_result.metrics.get("cagr"),
        "paper_max_drawdown": paper_result.metrics.get("max_drawdown"),
        "paper_current_drawdown": paper_result.metrics.get("current_drawdown"),
        "paper_sharpe": paper_result.metrics.get("sharpe"),
        "paper_account_status": paper_account_status,
        "paper_account_error": paper_account_error,
        "paper_audit_event_count": paper_result.metrics.get("audit_event_count"),
        "paper_total_estimated_fee": paper_result.metrics.get("total_estimated_fee"),
        "dashboard_posture": dashboard_result.snapshot.get("dashboard_posture"),
        "sentiment_state": dashboard_result.snapshot.get("sentiment_state"),
        "sentiment_freshness_status": dashboard_result.snapshot.get("sentiment_freshness_status"),
        "market_cache_status": dashboard_result.snapshot.get("market_cache_status"),
        "market_cache_source_kind": dashboard_result.snapshot.get("market_cache_source_kind"),
        "market_cache_source_path": dashboard_result.snapshot.get("market_cache_source_path"),
        "market_cache_latest_date": dashboard_result.snapshot.get("market_cache_latest_date"),
        "market_cache_snapshot_row_count": dashboard_result.snapshot.get("market_cache_snapshot_row_count"),
        "market_cache_fetch_status": dashboard_result.snapshot.get("market_cache_fetch_status"),
        "market_cache_fetch_run_id": dashboard_result.snapshot.get("market_cache_fetch_run_id"),
        "allocator_input_status": dashboard_result.snapshot.get("allocator_input_status"),
        "paper_freshness_status": dashboard_result.snapshot.get("paper_freshness_status"),
        "trigger_freshness_status": dashboard_result.snapshot.get("trigger_freshness_status"),
        "model_audit_status": dashboard_result.snapshot.get("model_audit_status"),
        "model_audit_dir": dashboard_result.snapshot.get("model_audit_dir"),
        "model_audit_snapshot_path": dashboard_result.snapshot.get("model_audit_snapshot_path"),
        "model_audit_report_path": dashboard_result.snapshot.get("model_audit_report_path"),
        "model_audit_actions_path": dashboard_result.snapshot.get("model_audit_actions_path"),
        "model_audit_config_map_path": dashboard_result.snapshot.get("model_audit_config_map_path"),
        "model_audit_generated_at": dashboard_result.snapshot.get("model_audit_generated_at"),
        "model_audit_days_since_generated": dashboard_result.snapshot.get("model_audit_days_since_generated"),
        "model_audit_duplicate_config_groups": dashboard_result.snapshot.get("model_audit_duplicate_config_groups"),
        "model_audit_root_configs_without_extends": dashboard_result.snapshot.get("model_audit_root_configs_without_extends"),
        "model_audit_walk_forward_action_items": dashboard_result.snapshot.get("model_audit_walk_forward_action_items"),
        "model_audit_walk_forward_resume_candidates": dashboard_result.snapshot.get("model_audit_walk_forward_resume_candidates"),
        "model_audit_walk_forward_archive_review_candidates": dashboard_result.snapshot.get("model_audit_walk_forward_archive_review_candidates"),
        "model_audit_top_action": dashboard_result.snapshot.get("model_audit_top_action"),
        "promotion_status": dashboard_result.snapshot.get("promotion_status"),
        "promotion_decision": dashboard_result.snapshot.get("promotion_decision"),
        "promotion_candidate_passes_headline_gate": dashboard_result.snapshot.get("promotion_candidate_passes_headline_gate"),
        "promotion_sensitivity_support_count": dashboard_result.snapshot.get("promotion_sensitivity_support_count"),
        "promotion_sensitivity_group_support_count": dashboard_result.snapshot.get("promotion_sensitivity_group_support_count"),
        "promotion_support_group_count": dashboard_result.snapshot.get("promotion_support_group_count"),
        "promotion_evidence_support_count": dashboard_result.snapshot.get("promotion_evidence_support_count"),
        "promotion_sensitivity_run_support_count": dashboard_result.snapshot.get("promotion_sensitivity_run_support_count"),
        "promotion_min_sensitivity_support": dashboard_result.snapshot.get("promotion_min_sensitivity_support"),
        "promotion_return_edge": dashboard_result.snapshot.get("promotion_return_edge"),
        "promotion_sharpe_edge": dashboard_result.snapshot.get("promotion_sharpe_edge"),
        "promotion_drawdown_change": dashboard_result.snapshot.get("promotion_drawdown_change"),
        "promotion_report_path": dashboard_result.snapshot.get("promotion_report_path"),
        "daily_preflight_skipped": bool(dashboard_result.snapshot.get("daily_preflight_skipped", False)),
        **momentum_focus_snapshot,
        **chip_reversal_candidate_outcomes_snapshot,
    }

    live_shadow_result: LiveShadowResult | None = None
    live_shadow_preflight_result: LiveShadowPreflightResult | None = None
    live_shadow_blocked_by_preflight = False

    if live_shadow_preflight:
        snapshot.update(
            {
                "live_shadow_preflight_status": "skipped",
                "live_shadow_preflight_output_dir": str(resolved_live_shadow_preflight_output),
                "live_shadow_preflight_snapshot_path": str(resolved_live_shadow_preflight_output / "live_shadow_preflight_snapshot.json"),
                "live_shadow_preflight_report_path": str(resolved_live_shadow_preflight_output / "live_shadow_preflight_report.md"),
                "live_shadow_preflight_blockers_count": 0,
                "live_shadow_preflight_decision": None,
                "live_shadow_preflight_error": None,
            }
        )
    else:
        snapshot.update(
            {
                "live_shadow_preflight_status": "disabled",
                "live_shadow_preflight_output_dir": str(resolved_live_shadow_preflight_output),
                "live_shadow_preflight_snapshot_path": str(resolved_live_shadow_preflight_output / "live_shadow_preflight_snapshot.json"),
                "live_shadow_preflight_report_path": str(resolved_live_shadow_preflight_output / "live_shadow_preflight_report.md"),
                "live_shadow_preflight_blockers_count": 0,
                "live_shadow_preflight_decision": None,
                "live_shadow_preflight_error": None,
            }
        )

    if (
        live_shadow_preflight
        and live_shadow_holdings_file is not None
        and live_shadow_cash is not None
    ):
        try:
            live_shadow_preflight_result = run_live_shadow_preflight(
                project_root=root,
                holdings_file=live_shadow_holdings_file,
                targets_file=paper_result.stock_targets_path,
                prices_file=live_shadow_prices_file,
                output_dir=resolved_live_shadow_preflight_output,
                cash=float(live_shadow_cash),
                as_of_date=as_of,
                lot_size=live_shadow_lot_size,
                min_trade_value=live_shadow_min_trade_value,
                max_position_weight=live_shadow_max_position_weight,
                max_gross_exposure=live_shadow_max_gross_exposure,
            )
            snapshot.update(_live_shadow_preflight_context(live_shadow_preflight_result))
            live_shadow_blocked_by_preflight = (
                live_shadow_preflight_fail_on_blocked
                and str(live_shadow_preflight_result.snapshot.get("status")) == "blocked"
            )
        except Exception as error:  # Preflight is a hard stop for the next stage when enabled.
            snapshot.update(
                {
                    "live_shadow_preflight_status": "failed",
                    "live_shadow_preflight_error": str(error),
                    "live_shadow_preflight_decision": "blocked" if live_shadow_preflight_fail_on_blocked else None,
                    "live_shadow_preflight_blockers_count": 0,
                }
            )
            live_shadow_blocked_by_preflight = live_shadow_preflight_fail_on_blocked

    if live_shadow_holdings_file is not None and live_shadow_cash is not None:
        if live_shadow_blocked_by_preflight:
            snapshot.update(
                {
                    "live_shadow_status": "blocked_by_preflight",
                    "live_shadow_error": "live_shadow_preflight decision did not pass.",
                    "live_shadow_output_dir": str(resolved_live_shadow_output),
                    "live_shadow_trade_plan_status": "blocked_by_preflight",
                    "live_shadow_broker_action": "none",
                    "live_shadow_research_only": True,
                    "live_shadow_order_count": 0,
                    "live_shadow_buy_count": 0,
                    "live_shadow_sell_count": 0,
                    "live_shadow_planned_buy_amount": 0.0,
                    "live_shadow_planned_sell_amount": 0.0,
                    "live_shadow_estimated_cash_after_orders": float(live_shadow_cash),
                }
            )
        else:
            try:
                live_shadow_result = run_live_shadow(
                    project_root=root,
                    holdings_file=live_shadow_holdings_file,
                    targets_file=paper_result.stock_targets_path,
                    prices_file=live_shadow_prices_file,
                    output_dir=resolved_live_shadow_output,
                    cash=float(live_shadow_cash),
                    as_of_date=as_of,
                    lot_size=live_shadow_lot_size,
                    min_trade_value=live_shadow_min_trade_value,
                    max_position_weight=live_shadow_max_position_weight,
                    max_gross_exposure=live_shadow_max_gross_exposure,
                )
                snapshot.update(_live_shadow_context(live_shadow_result))
            except Exception as error:  # Live-shadow planning is pre-trade review only and must not block research reporting.
                snapshot.update(
                    {
                        "live_shadow_status": "failed",
                        "live_shadow_error": str(error),
                        "live_shadow_output_dir": str(resolved_live_shadow_output),
                        "live_shadow_trade_plan_status": "unavailable",
                        "live_shadow_broker_action": "none",
                        "live_shadow_research_only": True,
                    }
                )
    elif live_shadow_holdings_file is not None or live_shadow_cash is not None:
        snapshot.update(
            {
                "live_shadow_status": "skipped_missing_inputs",
                "live_shadow_error": "live_shadow_holdings_file and live_shadow_cash are both required.",
                "live_shadow_output_dir": str(resolved_live_shadow_output),
                "live_shadow_trade_plan_status": "unavailable",
                "live_shadow_broker_action": "none",
                "live_shadow_research_only": True,
            }
        )
    else:
        snapshot.update(
            {
                "live_shadow_status": "disabled",
                "live_shadow_error": None,
                "live_shadow_output_dir": str(resolved_live_shadow_output),
                "live_shadow_trade_plan_status": "N/A",
                "live_shadow_broker_action": "none",
                "live_shadow_research_only": True,
                "live_shadow_order_count": 0,
                "live_shadow_buy_count": 0,
                "live_shadow_sell_count": 0,
            }
        )
    snapshot.update(build_a_share_trading_gate(snapshot, as_of, snapshot.get("generated_at")))

    resolved_output.mkdir(parents=True, exist_ok=True)
    snapshot_path = resolved_output / "daily_pipeline_snapshot.json"
    report_path = resolved_output / "daily_pipeline.md"
    snapshot["pipeline_snapshot_path"] = str(snapshot_path)
    snapshot["pipeline_report_path"] = str(report_path)

    satellite_risk_budget_result: SatelliteRiskBudgetReviewResult | None = None
    satellite_trial_replay_result: SatelliteTrialReplayResult | None = None
    try:
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        satellite_risk_budget_result = run_satellite_risk_budget_review(
            pipeline_snapshot=snapshot_path,
            outcome_analysis_path=paper_result.stock_target_review_outcome_analysis_json_path,
            output_dir=resolved_satellite_risk_budget_output,
            network_lab_snapshot=network_lab_snapshot,
            network_max_cluster_count_warning=network_max_cluster_count_warning,
            network_residual_mi_warning=network_residual_mi_warning,
        )
        snapshot.update(
            {
                "satellite_risk_budget_status": "completed",
                "satellite_risk_budget_output_dir": str(satellite_risk_budget_result.output_dir),
                "satellite_risk_budget_report_path": str(satellite_risk_budget_result.report_path),
                "satellite_risk_budget_snapshot_path": str(satellite_risk_budget_result.snapshot_path),
                "satellite_risk_budget_checklist_path": str(satellite_risk_budget_result.checklist_path),
                "satellite_risk_budget_decision": satellite_risk_budget_result.snapshot.get("risk_budget_decision"),
                "satellite_risk_budget_reason": satellite_risk_budget_result.snapshot.get("risk_budget_reason"),
                "satellite_risk_budget_next_action_stage": satellite_risk_budget_result.snapshot.get("next_action_stage"),
                "satellite_risk_budget_recommended_satellite_weight": satellite_risk_budget_result.snapshot.get(
                    "recommended_satellite_budget"
                ),
                "satellite_risk_budget_selected_horizon": satellite_risk_budget_result.snapshot.get("selected_horizon"),
                "satellite_risk_budget_ready_horizon_count": satellite_risk_budget_result.snapshot.get(
                    "outcome_ready_horizon_count"
                ),
                "satellite_trial_rule_count": satellite_risk_budget_result.snapshot.get("satellite_trial_rule_count"),
                "satellite_trial_rules_path": satellite_risk_budget_result.snapshot.get("satellite_trial_rules_path"),
                "satellite_trial_rules_json_path": satellite_risk_budget_result.snapshot.get(
                    "satellite_trial_rules_json_path"
                ),
            }
        )
    except Exception as error:  # Keep the main daily report available if optional budget review fails.
        snapshot.update(
            {
                "satellite_risk_budget_status": "failed",
                "satellite_risk_budget_output_dir": str(resolved_satellite_risk_budget_output),
                "satellite_risk_budget_error": str(error),
                "satellite_risk_budget_decision": "review_failed",
                "satellite_risk_budget_reason": str(error),
                "satellite_risk_budget_next_action_stage": "inspect_risk_budget_review",
                "satellite_risk_budget_recommended_satellite_weight": None,
                "satellite_risk_budget_selected_horizon": None,
                "satellite_risk_budget_ready_horizon_count": None,
                "satellite_risk_budget_report_path": None,
                "satellite_risk_budget_snapshot_path": None,
                "satellite_risk_budget_checklist_path": None,
                "satellite_trial_rule_count": 0,
                "satellite_trial_rules_path": None,
                "satellite_trial_rules_json_path": None,
            }
        )

    if satellite_risk_budget_result is not None:
        try:
            satellite_trial_replay_result = run_satellite_trial_replay(
                rules_path=Path(str(satellite_risk_budget_result.snapshot.get("satellite_trial_rules_path"))),
                outcomes_history_path=paper_result.stock_target_review_outcomes_history_path,
                output_dir=resolved_satellite_trial_replay_output,
                horizons=satellite_trial_replay_horizons,
            )
            snapshot.update(
                {
                    "satellite_trial_replay_status": satellite_trial_replay_result.snapshot.get("status"),
                    "satellite_trial_replay_output_dir": str(satellite_trial_replay_result.output_dir),
                    "satellite_trial_replay_matched_event_count": satellite_trial_replay_result.snapshot.get(
                        "matched_event_count"
                    ),
                    "satellite_trial_replay_best_horizon": satellite_trial_replay_result.snapshot.get(
                        "union_best_horizon"
                    ),
                    "satellite_trial_replay_best_avg_return_edge": satellite_trial_replay_result.snapshot.get(
                        "union_best_avg_return_edge"
                    ),
                    "satellite_trial_replay_best_win_rate_edge": satellite_trial_replay_result.snapshot.get(
                        "union_best_win_rate_edge"
                    ),
                    "satellite_trial_replay_report_path": str(satellite_trial_replay_result.report_path),
                    "satellite_trial_replay_summary_path": str(satellite_trial_replay_result.summary_path),
                    "satellite_trial_replay_matches_path": str(satellite_trial_replay_result.matches_path),
                    "satellite_trial_replay_snapshot_path": str(satellite_trial_replay_result.snapshot_path),
                    "satellite_trial_replay_error": None,
                }
            )
        except Exception as error:  # Optional replay evidence must not block the daily research report.
            snapshot.update(
                {
                    "satellite_trial_replay_status": "failed",
                    "satellite_trial_replay_output_dir": str(resolved_satellite_trial_replay_output),
                    "satellite_trial_replay_matched_event_count": 0,
                    "satellite_trial_replay_best_horizon": None,
                    "satellite_trial_replay_best_avg_return_edge": None,
                    "satellite_trial_replay_best_win_rate_edge": None,
                    "satellite_trial_replay_report_path": None,
                    "satellite_trial_replay_summary_path": None,
                    "satellite_trial_replay_matches_path": None,
                    "satellite_trial_replay_snapshot_path": None,
                    "satellite_trial_replay_error": str(error),
                }
            )
    else:
        snapshot.update(
            {
                "satellite_trial_replay_status": "skipped_risk_budget_failed",
                "satellite_trial_replay_output_dir": str(resolved_satellite_trial_replay_output),
                "satellite_trial_replay_matched_event_count": 0,
                "satellite_trial_replay_best_horizon": None,
                "satellite_trial_replay_best_avg_return_edge": None,
                "satellite_trial_replay_best_win_rate_edge": None,
                "satellite_trial_replay_report_path": None,
                "satellite_trial_replay_summary_path": None,
                "satellite_trial_replay_matches_path": None,
                "satellite_trial_replay_snapshot_path": None,
                "satellite_trial_replay_error": "satellite_risk_budget_review_failed",
            }
        )

    resolved_history_path = _resolve(root, history_path, DEFAULT_PIPELINE_HISTORY) if history_path is not None else None
    history_frame: pd.DataFrame | None = None
    if resolved_history_path is not None:
        try:
            history_frame = _append_history(snapshot, resolved_history_path, as_of)
        except OSError as error:
            _add_history_context(snapshot, None)
            snapshot["history_status"] = "failed"
            snapshot["history_path"] = str(resolved_history_path)
            snapshot["history_error"] = str(error)
            snapshot["history_row_count"] = "N/A"
    else:
        _add_history_context(snapshot, None)
        snapshot["history_status"] = "disabled"
        snapshot["history_path"] = None
        snapshot["history_row_count"] = "N/A"

    history_review_result: PipelineHistoryReviewResult | None = None
    resolved_history_review_output: Path | None = None
    if run_history_review and resolved_history_path is not None and snapshot.get("history_status") == "appended":
        resolved_history_review_output = _resolve_dated_output(
            root,
            history_review_output_dir,
            DEFAULT_HISTORY_REVIEW_OUTPUT,
            DEFAULT_HISTORY_REVIEW_BASE,
            as_of,
            date_stamp,
        )
        try:
            history_review_result = run_pipeline_history_review(
                project_root=root,
                history_file=resolved_history_path,
                output_dir=resolved_history_review_output,
                as_of_date=as_of,
                lookback_runs=history_review_lookback_runs,
                max_staleness_days=max_staleness_days,
                drawdown_watch_threshold=history_review_drawdown_watch_threshold,
                min_sharpe_watch=history_review_min_sharpe_watch,
                history_frame=history_frame,
            )
            snapshot.update(_history_review_context(history_review_result))
        except Exception as error:  # Keep the main daily report available if optional review fails.
            snapshot.update(
                {
                    "history_review_status": "failed",
                    "history_review_output_dir": str(resolved_history_review_output),
                    "history_review_error": str(error),
                    "history_review_health_state": "unknown",
                    "history_review_alert_count": "N/A",
                    "history_review_latest_as_of_date": "N/A",
                }
            )
    elif run_history_review:
        snapshot.update(
            {
                "history_review_status": "skipped_no_history",
                "history_review_health_state": "unknown",
                "history_review_alert_count": "N/A",
                "history_review_latest_as_of_date": "N/A",
            }
        )
    else:
        snapshot.update(
            {
                "history_review_status": "disabled",
                "history_review_health_state": "N/A",
                "history_review_alert_count": "N/A",
                "history_review_latest_as_of_date": "N/A",
            }
        )

    if history_review_result is not None:
        try:
            dashboard_result = run_daily_dashboard(
                project_root=root,
                output_dir=resolved_dashboard_output,
                daily_check_dir=daily_result.output_dir,
                paper_account_dir=paper_result.output_dir,
                sentiment_dir=sentiment_dir,
                data_cache_dir=data_cache_dir,
                allocator_dir=allocator_dir,
                trigger_report=trigger_report,
                model_audit_dir=model_audit_dir,
                pipeline_history_dir=history_review_result.output_dir,
                as_of_date=as_of,
                max_staleness_days=max_staleness_days,
                min_cache_fresh_ratio=min_cache_fresh_ratio,
                date_stamp=False,
                summary_cache=dashboard_summary_cache,
                market_snapshot=shared_market_snapshot,
                publish_artifacts=False,
            )
            snapshot.update(_dashboard_history_context(dashboard_result))
        except Exception as error:  # Dashboard history context is useful but should not block the daily report.
            snapshot.update(
                {
                    "dashboard_history_refresh_status": "failed",
                    "dashboard_history_refresh_error": str(error),
                    "dashboard_pipeline_history_health_state": "unknown",
                }
            )
    elif run_history_review:
        snapshot.update(
            {
                "dashboard_history_refresh_status": "skipped_no_completed_history_review",
                "dashboard_pipeline_history_health_state": "unknown",
            }
        )
    else:
        snapshot.update(
            {
                "dashboard_history_refresh_status": "disabled",
                "dashboard_pipeline_history_health_state": "N/A",
            }
        )

    alerts_result = write_daily_alerts(
        snapshot,
        resolved_output,
        stock_target_review_warning_only_after_close=stock_review_warning_only_after_close,
        publish_artifacts=False,
    )
    snapshot.update(_alerts_context(alerts_result))
    snapshot.update(_next_step_context(snapshot))
    if resolved_history_path is not None and snapshot.get("history_status") == "appended":
        if history_frame is not None:
            history_frame = _update_history_frame(snapshot, history_frame)
        if run_history_review and history_review_result is not None and resolved_history_review_output is not None:
            try:
                history_review_result = run_pipeline_history_review(
                    project_root=root,
                    history_file=resolved_history_path,
                    output_dir=resolved_history_review_output,
                    as_of_date=as_of,
                    lookback_runs=history_review_lookback_runs,
                    max_staleness_days=max_staleness_days,
                    drawdown_watch_threshold=history_review_drawdown_watch_threshold,
                    min_sharpe_watch=history_review_min_sharpe_watch,
                    history_frame=history_frame,
                )
                snapshot.update(_history_review_context(history_review_result))
                snapshot["history_review_alert_refresh_status"] = "completed"
                snapshot["history_review_alert_refresh_error"] = None
            except Exception as error:  # Preserve the already generated daily report if final review refresh fails.
                snapshot["history_review_alert_refresh_status"] = "failed"
                snapshot["history_review_alert_refresh_error"] = str(error)
                snapshot["dashboard_alert_refresh_status"] = "skipped_history_alert_refresh_failed"
                snapshot["dashboard_alert_refresh_error"] = None
            else:
                try:
                    dashboard_publish_attempted = True
                    dashboard_result = run_daily_dashboard(
                        project_root=root,
                        output_dir=resolved_dashboard_output,
                        daily_check_dir=daily_result.output_dir,
                        paper_account_dir=paper_result.output_dir,
                        sentiment_dir=sentiment_dir,
                        data_cache_dir=data_cache_dir,
                        allocator_dir=allocator_dir,
                        trigger_report=trigger_report,
                        model_audit_dir=model_audit_dir,
                        pipeline_history_dir=history_review_result.output_dir,
                        as_of_date=as_of,
                        max_staleness_days=max_staleness_days,
                        min_cache_fresh_ratio=min_cache_fresh_ratio,
                        date_stamp=False,
                        summary_cache=dashboard_summary_cache,
                        market_snapshot=shared_market_snapshot,
                        publish_artifacts=True,
                    )
                    dashboard_artifacts_published = True
                    snapshot.update(_dashboard_history_context(dashboard_result))
                except Exception as error:  # Keep the final history review even if dashboard rewrite fails.
                    snapshot["dashboard_alert_refresh_status"] = "failed"
                    snapshot["dashboard_alert_refresh_error"] = str(error)
                    snapshot["dashboard_history_refresh_status"] = "failed"
                    snapshot["dashboard_history_refresh_error"] = str(error)
                    snapshot["dashboard_pipeline_history_health_state"] = "unknown"
            snapshot.update(_next_step_context(snapshot))
            if history_frame is not None:
                history_frame = _update_history_frame(snapshot, history_frame)
        elif run_history_review:
            snapshot["history_review_alert_refresh_status"] = "skipped_no_completed_history_review"
            snapshot["dashboard_alert_refresh_status"] = "skipped_no_completed_history_review"
    elif run_history_review:
        snapshot["history_review_alert_refresh_status"] = "skipped_no_history"
        snapshot["dashboard_alert_refresh_status"] = "skipped_no_history"
    else:
        snapshot["history_review_alert_refresh_status"] = "disabled"
        snapshot["dashboard_alert_refresh_status"] = "disabled"

    if not dashboard_artifacts_published and not dashboard_publish_attempted:
        dashboard_publish_attempted = True
        try:
            dashboard_result = run_daily_dashboard(
                project_root=root,
                output_dir=resolved_dashboard_output,
                daily_check_dir=daily_result.output_dir,
                paper_account_dir=paper_result.output_dir,
                sentiment_dir=sentiment_dir,
                data_cache_dir=data_cache_dir,
                allocator_dir=allocator_dir,
                trigger_report=trigger_report,
                model_audit_dir=model_audit_dir,
                pipeline_history_dir=(history_review_result.output_dir if history_review_result is not None else None),
                as_of_date=as_of,
                max_staleness_days=max_staleness_days,
                min_cache_fresh_ratio=min_cache_fresh_ratio,
                date_stamp=False,
                summary_cache=dashboard_summary_cache,
                market_snapshot=shared_market_snapshot,
                publish_artifacts=True,
            )
            dashboard_artifacts_published = True
            if history_review_result is not None:
                snapshot.update(_dashboard_history_context(dashboard_result))
        except Exception as error:
            snapshot["dashboard_history_refresh_status"] = "failed"
            snapshot["dashboard_history_refresh_error"] = str(error)
            snapshot["dashboard_pipeline_history_health_state"] = "unknown"

    preflight_result: LivePreflightResult | None = None
    snapshot["live_preflight_output_dir"] = str(resolved_live_preflight_output)
    snapshot["live_shadow_review_decisions_path"] = (
        str(resolved_live_shadow_review_decisions_file) if resolved_live_shadow_review_decisions_file is not None else None
    )
    try:
        snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        preflight_result = run_live_preflight(
            dashboard_snapshot=dashboard_result.snapshot_path,
            paper_account_dir=paper_result.output_dir,
            pipeline_snapshot=snapshot_path,
            live_shadow_review_decisions_file=resolved_live_shadow_review_decisions_file,
            output_dir=resolved_live_preflight_output,
        )
        snapshot.update(_live_preflight_context(preflight_result))
    except Exception as error:
        snapshot["live_preflight_status"] = "failed"
        snapshot["live_preflight_error"] = str(error)
        snapshot["live_preflight_output_dir"] = str(resolved_live_preflight_output)
        snapshot["live_preflight_snapshot_path"] = str(resolved_live_preflight_output / "live_preflight_snapshot.json")
        snapshot["live_preflight_report_path"] = str(resolved_live_preflight_output / "live_preflight.md")
        snapshot["live_preflight_checklist_path"] = str(resolved_live_preflight_output / "live_preflight_checklist.csv")
        snapshot["live_preflight_decision"] = None
        snapshot["live_preflight_broker_connection_status"] = "not_connected"
        snapshot["live_preflight_blocking_items_count"] = 0
        snapshot["live_preflight_monitor_items_count"] = 0
        snapshot["live_preflight_pipeline_snapshot_status"] = "missing"
        snapshot["live_preflight_pipeline_snapshot_path"] = str(snapshot_path)
        snapshot["live_preflight_active_target_count"] = None
        snapshot["live_preflight_stock_target_review_action_count"] = None
        snapshot["live_preflight_live_shadow_review_decisions_path"] = (
            str(resolved_live_shadow_review_decisions_file) if resolved_live_shadow_review_decisions_file is not None else None
        )
        snapshot["live_preflight_live_shadow_review_decision_count"] = None
        snapshot["live_preflight_live_shadow_review_blocking_decision_count"] = None
        snapshot["live_preflight_live_shadow_review_monitor_decision_count"] = None
        snapshot["live_preflight_live_shadow_review_unknown_decision_count"] = None
    else:
        if snapshot.get("live_preflight_status") is None:
            snapshot["live_preflight_status"] = "completed"

    alerts_result = write_daily_alerts(
        snapshot,
        resolved_output,
        stock_target_review_warning_only_after_close=stock_review_warning_only_after_close,
    )
    snapshot.update(_alerts_context(alerts_result))
    snapshot.update(_next_step_context(snapshot))
    if resolved_history_path is not None and snapshot.get("history_status") == "appended":
        if history_frame is not None:
            history_frame = _update_history_frame(snapshot, history_frame)
            _write_history(history_frame, resolved_history_path)

    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot), encoding="utf-8")

    return DailyPipelineResult(
        output_dir=resolved_output,
        snapshot_path=snapshot_path,
        report_path=report_path,
        history_path=resolved_history_path,
        history_review_result=history_review_result,
        satellite_risk_budget_result=satellite_risk_budget_result,
        satellite_trial_replay_result=satellite_trial_replay_result,
        live_shadow_result=live_shadow_result,
        live_shadow_preflight_result=live_shadow_preflight_result,
        live_preflight_result=preflight_result,
        momentum_focus_result=momentum_focus_result,
        chip_reversal_candidate_outcomes_result=chip_reversal_candidate_outcomes_result,
        alerts_result=alerts_result,
        daily_check_result=daily_result,
        paper_account_result=paper_result,
        dashboard_result=dashboard_result,
        snapshot=snapshot,
    )
