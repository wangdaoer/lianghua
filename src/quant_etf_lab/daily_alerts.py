"""Alert rules for the daily research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .artifact_io import publish_json_if_semantically_changed, write_text_if_changed


SEVERITY_RANK = {"normal": 0, "info": 1, "warning": 2, "critical": 3}

ACTION_RULES = {
    "data_freshness_status_not_fresh": (
        "refresh_model_data",
        "Refresh local model data",
        "Refresh or inspect local equity data before using today's model posture.",
    ),
    "market_cache_status_not_fresh": (
        "refresh_market_cache",
        "Refresh cached market data",
        "Refresh processed OHLCV cache and rerun the daily pipeline.",
    ),
    "allocator_input_status_not_fresh": (
        "refresh_allocator_inputs",
        "Refresh allocator inputs",
        "Rerun or inspect allocator walk-forward outputs before reading target weights.",
    ),
    "paper_freshness_status_not_fresh": (
        "rebuild_paper_account",
        "Rebuild paper account",
        "Rerun the paper-account step and inspect the latest ledger date.",
    ),
    "sentiment_freshness_status_not_fresh": (
        "refresh_sentiment",
        "Refresh market sentiment",
        "Rerun market sentiment or review why the latest sentiment file is stale.",
    ),
    "trigger_freshness_status_not_fresh": (
        "refresh_trigger_monitor",
        "Refresh trigger monitor",
        "Inspect the latest trigger-monitor report before using stock-level signals.",
    ),
    "model_audit_needs_attention": (
        "review_model_build_audit",
        "Review model-build audit",
        "Open the model-build audit action list and resolve or explicitly record any incomplete walk-forward evidence.",
    ),
    "model_audit_resolved_future_dated": (
        "monitor_model_build_audit",
        "Monitor model-build audit",
        "Model-build audit has no open action rows; keep the future-dated status visible during routine review.",
    ),
    "model_audit_resolved_attention": (
        "monitor_model_build_audit",
        "Monitor model-build audit",
        "Model-build audit reports attention status but has no open action rows; keep it visible during routine review.",
    ),
    "promotion_evidence_correlated": (
        "monitor_allocator_promotion_evidence",
        "Monitor allocator promotion evidence",
        "Keep the allocator candidate on watch until independent sensitivity-group support improves.",
    ),
    "trading_day_data_missing": (
        "refresh_local_market_data",
        "Refresh local market data",
        "Local equity latest date is missing; refresh local data before interpreting the run.",
    ),
    "trading_day_data_not_ready": (
        "verify_trading_day_or_refresh_data",
        "Verify trading day or refresh data",
        "After-close local equity data is not current; verify whether today traded, then refresh data if needed.",
    ),
    "before_close_wait": (
        "wait_until_after_close",
        "Wait for after-close data",
        "Rerun after the A-share after-close cutoff when complete daily data is expected.",
    ),
    "future_dated_data": (
        "inspect_local_data_dates",
        "Inspect local data dates",
        "Local data is dated after the requested as-of date; inspect inputs before relying on the report.",
    ),
    "current_drawdown_critical": (
        "review_drawdown",
        "Review portfolio drawdown",
        "Treat current drawdown as a risk review item before promoting new exposure.",
    ),
    "current_drawdown_warning": (
        "review_drawdown",
        "Review portfolio drawdown",
        "Check whether current drawdown is still inside the accepted risk envelope.",
    ),
    "max_drawdown_limit_breached": (
        "review_risk_assumptions",
        "Review risk assumptions",
        "Do not promote the setup until the max-drawdown breach is understood.",
    ),
    "daily_return_drop_critical": (
        "inspect_paper_ledger",
        "Inspect paper ledger",
        "Inspect the paper ledger and market state before accepting today's output.",
    ),
    "daily_return_drop_warning": (
        "inspect_paper_ledger",
        "Inspect paper ledger",
        "Review the paper ledger and latest market state.",
    ),
    "dashboard_wait_state": (
        "resolve_dashboard_gate",
        "Resolve dashboard gate",
        "Open the dashboard freshness section and resolve the blocked gate first.",
    ),
    "dashboard_defensive_or_wait": (
        "review_dashboard_posture",
        "Review dashboard posture",
        "Review dashboard posture before interpreting allocator output.",
    ),
    "dashboard_posture_changed": (
        "compare_dashboard_changes",
        "Compare dashboard changes",
        "Compare freshness gates and market state with the previous run.",
    ),
    "dashboard_history_refresh_failed": (
        "inspect_dashboard_history_refresh",
        "Inspect dashboard history refresh",
        "Open the daily pipeline snapshot and rerun or inspect the dashboard history refresh step.",
    ),
    "history_review_alert_refresh_failed": (
        "inspect_history_review_alert_refresh",
        "Inspect alert-aware history refresh",
        "Open the daily pipeline snapshot and rerun or inspect the alert-aware pipeline-history refresh step.",
    ),
    "daily_action_posture_changed": (
        "compare_daily_action",
        "Compare daily action posture",
        "Review phase-2 posture and paper-account target weights.",
    ),
    "satellite_weight_opened": (
        "review_satellite_activation",
        "Review satellite activation",
        "Review why the allocator gate opened before treating the signal as usable.",
    ),
    "satellite_weight_reduced": (
        "confirm_satellite_reduction",
        "Confirm satellite reduction",
        "Confirm the risk-off or filter reason in the paper-account audit.",
    ),
    "satellite_weight_active": (
        "review_active_satellite",
        "Review active satellite",
        "Review allocator gate and paper-account audit before manual interpretation.",
    ),
    "satellite_risk_budget_waiting": (
        "wait_satellite_risk_budget_samples",
        "Wait for satellite budget samples",
        "Keep collecting outcome samples before changing the satellite risk budget.",
    ),
    "satellite_risk_budget_hold": (
        "hold_satellite_risk_budget",
        "Hold satellite risk budget",
        "Keep the current core-base risk budget until outcome evidence improves.",
    ),
    "satellite_risk_budget_blocked": (
        "resolve_satellite_risk_budget_blockers",
        "Resolve satellite budget blockers",
        "Resolve the upstream research gates before reviewing satellite risk budget.",
    ),
    "satellite_risk_budget_review_samples": (
        "review_satellite_risk_budget_samples",
        "Review satellite budget samples",
        "Open the satellite risk-budget review before changing the research budget.",
    ),
    "satellite_risk_budget_trial_eligible": (
        "review_satellite_risk_budget_trial",
        "Review small satellite trial",
        "Review the research-only trial budget and confirm no broker action is implied.",
    ),
    "satellite_risk_budget_failed": (
        "inspect_satellite_risk_budget_review",
        "Inspect satellite budget review",
        "Open the daily pipeline and inspect why the satellite risk-budget review failed.",
    ),
    "live_shadow_preflight_blocked": (
        "review_live_shadow_preflight",
        "Review live-shadow preflight blockers",
        "Open the live-shadow preflight report and clear blockers or inspect failure details before rebuilding the live-shadow step.",
    ),
    "stock_target_trigger_review": (
        "review_stock_target_trigger",
        "Review target-stock trigger matches",
        "Open the stock-target review panel and inspect matched trigger-monitor context before relying on the target list.",
    ),
    "stock_target_drawdown_review": (
        "review_stock_target_drawdown",
        "Review target-stock drawdowns",
        "Open the stock-target review panel and inspect loss source, source risk gate, and next-session gap risk.",
    ),
    "stock_target_drawdown_review_monitored": (
        "monitor_reviewed_drawdown_targets",
        "Monitor reviewed target-stock drawdowns",
        "Drawdown rows already have review coverage; keep them visible during routine follow-up.",
    ),
    "stock_target_review_required": (
        "review_stock_target_queue",
        "Review target-stock queue",
        "Open the stock-target review panel and inspect rows marked review_required.",
    ),
    "stock_target_review_notes_open": (
        "update_stock_target_review_notes",
        "Update stock-target review notes",
        "After reviewing rows marked review_required, update manual_status/manual_note in the persistent notes CSV.",
    ),
    "stock_target_manual_watch": (
        "review_manual_watch_targets",
        "Review manual watch target rows",
        "Open the notes CSV and review rows marked manual_status=watch during routine model review.",
    ),
    "stock_target_manual_exclude_candidate": (
        "review_manual_exclusion_candidates",
        "Review manual exclusion candidates",
        "Inspect rows marked manual_status=exclude_candidate; model targets are not changed automatically.",
    ),
    "stock_target_manual_other_status": (
        "normalize_manual_review_status",
        "Normalize manual review statuses",
        "Use a supported manual_status value so the daily pipeline can classify the row consistently.",
    ),
    "stock_target_suppressed_layer_review": (
        "monitor_suppressed_layer_targets",
        "Monitor suppressed source targets",
        "Keep suppressed source positions visible for audit; current portfolio target weight remains controlled by the layer gate.",
    ),
    "stock_target_watch_review": (
        "monitor_stock_target_watchlist",
        "Monitor stock-target watch rows",
        "Review target-stock watch rows during routine model review.",
    ),
    "stock_market_cap_cache_unavailable": (
        "refresh_stock_market_cap_cache",
        "Refresh stock market-cap cache",
        "Refresh the A-share market-cap cache before relying on the 1500 Yi Yuan tracking filter.",
    ),
    "stock_market_cap_cache_empty": (
        "refresh_stock_market_cap_cache",
        "Refresh stock market-cap cache",
        "Refresh the A-share market-cap cache before relying on the 1500 Yi Yuan tracking filter.",
    ),
    "stock_market_cap_cache_stale": (
        "refresh_stock_market_cap_cache",
        "Refresh stock market-cap cache",
        "Refresh the A-share market-cap cache so the tracking filter uses the latest local market date.",
    ),
    "stock_market_cap_target_missing": (
        "review_stock_market_cap_coverage",
        "Review market-cap coverage",
        "Inspect target rows with missing market cap; large-cap tracking exclusions may be incomplete for those rows.",
    ),
    "stock_target_outcome_analysis_pending": (
        "wait_stock_target_outcome_samples",
        "Wait for outcome sample maturity",
        "Rerun the daily pipeline after the next estimated outcome maturity date to refresh the research-only effectiveness sample.",
    ),
    "stock_target_outcome_due_queue": (
        "refresh_due_outcome_samples",
        "Refresh due outcome samples",
        "Open the due queue, refresh local OHLCV if needed, then rerun paper-account or daily-pipeline.",
    ),
    "stock_target_outcome_analysis_ready": (
        "review_stock_target_outcomes",
        "Review mature stock-target outcomes",
        "Open the outcome analysis report and review mature horizons before changing strategy parameters.",
    ),
    "non_trading_day": (
        "skip_new_signal_interpretation",
        "Skip new signal interpretation",
        "No same-day A-share close is expected; keep the run as a record only.",
    ),
}


@dataclass(frozen=True)
class DailyAlertsResult:
    output_dir: Path
    report_path: Path
    json_path: Path
    latest_report_path: Path
    payload: dict[str, Any]


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(number):
        return None
    return number


def _pct(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "N/A"
    return f"{number * 100:.2f}%"


def _num(value: Any, digits: int = 2) -> str:
    number = _as_float(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _sentence(value: Any) -> str:
    text = str(value or "N/A").strip()
    return text.rstrip(".。!！?？")


def _date_or_none(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _alert(
    alerts: list[dict[str, Any]],
    severity: str,
    code: str,
    title: str,
    detail: str,
    recommended_review: str,
) -> None:
    alerts.append(
        {
            "severity": severity,
            "code": code,
            "title": title,
            "detail": detail,
            "recommended_review": recommended_review,
        }
    )


def _action_stage(highest: str) -> str:
    if highest == "critical":
        return "blocked"
    if highest == "warning":
        return "review_required"
    if highest == "info":
        return "monitor"
    return "routine"


def _action_summary(stage: str) -> str:
    if stage == "blocked":
        return "Pause new manual interpretation until critical gates are resolved."
    if stage == "review_required":
        return "Review warning items before using today's model posture."
    if stage == "monitor":
        return "No blocking issue; note informational changes during routine review."
    return "No alert rules fired; continue routine review only."


def _build_action_items(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for alert in alerts:
        code = str(alert.get("code") or "")
        action_code, title, next_step = ACTION_RULES.get(
            code,
            (
                f"review_{code}" if code else "review_alert",
                str(alert.get("title") or "Review alert"),
                str(alert.get("recommended_review") or "Review this alert before relying on today's output."),
            ),
        )
        current = items.get(action_code)
        severity = str(alert.get("severity") or "info")
        if current is None or SEVERITY_RANK.get(severity, 0) > SEVERITY_RANK.get(current["severity"], 0):
            items[action_code] = {
                "severity": severity,
                "action_code": action_code,
                "title": title,
                "next_step": next_step,
                "source_alert_codes": [code],
            }
        elif code not in current["source_alert_codes"]:
            current["source_alert_codes"].append(code)

    return sorted(items.values(), key=lambda item: (-SEVERITY_RANK.get(item["severity"], 0), item["action_code"]))


def build_daily_alert_payload(
    snapshot: dict[str, Any],
    current_drawdown_warning: float = -0.08,
    current_drawdown_critical: float = -0.12,
    max_drawdown_limit: float = -0.16,
    daily_return_drop_warning: float = -0.02,
    daily_return_drop_critical: float = -0.05,
    stock_target_review_warning_only_after_close: bool = False,
) -> dict[str, Any]:
    alerts: list[dict[str, Any]] = []

    gate_status = snapshot.get("trading_day_gate_status")
    if gate_status == "non_trading_day":
        _alert(
            alerts,
            "info",
            "non_trading_day",
            "A-share non-trading day",
            f"As-of {snapshot.get('as_of_date')} is classified as non-trading by `{snapshot.get('trading_day_evidence')}`.",
            "Keep this run as a record only; do not interpret it as a new after-close trading signal.",
        )
    elif gate_status in {"trading_day_data_missing", "trading_day_data_not_ready"}:
        _alert(
            alerts,
            "critical",
            str(gate_status),
            "After-close data gate is blocked",
            f"Gate status is `{gate_status}` and after-close data status is `{snapshot.get('after_close_data_status')}`.",
            "Verify the trading day and refresh local market data before using today's model posture.",
        )
    elif gate_status == "before_close_wait":
        _alert(
            alerts,
            "warning",
            "before_close_wait",
            "Run happened before after-close cutoff",
            f"After-close cutoff `{snapshot.get('after_close_cutoff')}` had not been reached.",
            "Rerun after close when complete daily data is available.",
        )
    elif gate_status == "future_dated_data":
        _alert(
            alerts,
            "warning",
            "future_dated_data",
            "Local data is future-dated",
            "Local equity data is dated after the requested as-of date.",
            "Inspect local input dates before relying on this report.",
        )

    freshness_rules = {
        "data_freshness_status": "critical",
        "market_cache_status": "warning",
        "allocator_input_status": "critical",
        "paper_freshness_status": "critical",
        "sentiment_freshness_status": "warning",
        "trigger_freshness_status": "warning",
    }
    for field, severity in freshness_rules.items():
        value = snapshot.get(field)
        if value not in {None, "fresh_enough"}:
            _alert(
                alerts,
                severity,
                f"{field}_not_fresh",
                f"{field} is {value}",
                f"Freshness gate `{field}` is `{value}` for as-of {snapshot.get('as_of_date')}.",
                "Refresh or inspect the related source file before relying on today's model posture.",
            )

    model_audit_status = str(snapshot.get("model_audit_status") or "")
    model_audit_action_items = int(_as_float(snapshot.get("model_audit_walk_forward_action_items")) or 0)
    model_audit_resume_candidates = int(_as_float(snapshot.get("model_audit_walk_forward_resume_candidates")) or 0)
    model_audit_archive_candidates = int(_as_float(snapshot.get("model_audit_walk_forward_archive_review_candidates")) or 0)
    model_audit_report = snapshot.get("model_audit_report_path")
    model_audit_actions = snapshot.get("model_audit_actions_path")
    model_audit_top_action = snapshot.get("model_audit_top_action")
    if model_audit_status not in {"", "ok", "missing"}:
        model_audit_has_open_actions = (
            model_audit_action_items > 0
            or model_audit_resume_candidates > 0
            or model_audit_archive_candidates > 0
        )
        if model_audit_status == "future_dated" and not model_audit_has_open_actions:
            _alert(
                alerts,
                "info",
                "model_audit_resolved_future_dated",
                "Model-build audit has no open action rows",
                (
                    f"Model audit status is `{model_audit_status}`, but the action list is empty: "
                    f"{model_audit_action_items} action item(s), {model_audit_resume_candidates} resume/finalize candidate(s), "
                    f"{model_audit_archive_candidates} archive-review candidate(s). "
                    f"Audit report: `{model_audit_report}`. Action CSV: `{model_audit_actions}`."
                ),
                "Keep the audit visible for provenance; no model-build action is blocking this pipeline run.",
            )
        elif not model_audit_has_open_actions:
            _alert(
                alerts,
                "info",
                "model_audit_resolved_attention",
                "Model-build audit has no open action rows",
                (
                    f"Model audit status is `{model_audit_status}`, but the action list is empty: "
                    f"{model_audit_action_items} action item(s), {model_audit_resume_candidates} resume/finalize candidate(s), "
                    f"{model_audit_archive_candidates} archive-review candidate(s). "
                    f"Audit report: `{model_audit_report}`. Action CSV: `{model_audit_actions}`."
                ),
                "Keep the audit visible for provenance; no model-build action is blocking this pipeline run.",
            )
        else:
            _alert(
                alerts,
                "warning",
                "model_audit_needs_attention",
                "Model-build audit needs attention",
                (
                    f"Model audit status is `{model_audit_status}` with {model_audit_action_items} action item(s), "
                    f"{model_audit_resume_candidates} resume/finalize candidate(s), "
                    f"{model_audit_archive_candidates} archive-review candidate(s), top action `{model_audit_top_action or 'N/A'}`. "
                    f"Audit report: `{model_audit_report}`. Action CSV: `{model_audit_actions}`."
                ),
                "Open the model-build audit action list and resolve or record stale walk-forward evidence before promoting new model results.",
            )

    promotion_status = str(snapshot.get("promotion_status") or "")
    promotion_decision = str(snapshot.get("promotion_decision") or "")
    promotion_group_support = _as_float(snapshot.get("promotion_sensitivity_group_support_count"))
    promotion_run_support = _as_float(snapshot.get("promotion_sensitivity_run_support_count"))
    promotion_min_support = _as_float(snapshot.get("promotion_min_sensitivity_support"))
    promotion_report = snapshot.get("promotion_report_path")
    if (
        promotion_status == "ok"
        and promotion_decision == "watch_candidate"
        and _bool(snapshot.get("promotion_candidate_passes_headline_gate"))
        and promotion_group_support is not None
        and promotion_run_support is not None
        and promotion_min_support is not None
        and promotion_group_support < promotion_min_support
        and promotion_run_support >= promotion_min_support
        and promotion_run_support > promotion_group_support
    ):
        _alert(
            alerts,
            "info",
            "promotion_evidence_correlated",
            "Allocator promotion evidence is correlated",
            (
                f"Candidate passed the headline gate, but independent sensitivity-group support is "
                f"{promotion_group_support:.0f} / {promotion_min_support:.0f} while raw run support is {promotion_run_support:.0f}. "
                f"Promotion review: `{promotion_report}`."
            ),
            "Keep the candidate on watch; do not read correlated guardrail variants as independent promotion evidence.",
        )

    posture = str(snapshot.get("dashboard_posture") or "")
    if posture.startswith("wait_for_"):
        severity = "critical" if posture in {"wait_for_fresh_model_data", "wait_for_fresh_allocator_inputs", "wait_for_fresh_paper_account"} else "warning"
        _alert(
            alerts,
            severity,
            "dashboard_wait_state",
            f"Dashboard posture is {posture}",
            "The dashboard is not accepting the daily review posture because at least one gate is blocked.",
            "Open the dashboard freshness section and resolve the blocked gate first.",
        )
    elif posture in {"defensive_review_only", "model_ok_wait_for_current_trigger", "model_ok_wait_for_fresh_sentiment"}:
        _alert(
            alerts,
            "warning",
            "dashboard_defensive_or_wait",
            f"Dashboard posture is {posture}",
            "The model is not in the normal core-base watch state.",
            "Review dashboard posture before interpreting allocator output.",
        )

    dashboard_history_refresh_status = str(snapshot.get("dashboard_history_refresh_status") or "")
    dashboard_history_refresh_error = snapshot.get("dashboard_history_refresh_error")
    if dashboard_history_refresh_status == "failed":
        _alert(
            alerts,
            "warning",
            "dashboard_history_refresh_failed",
            "Dashboard history refresh failed",
            (
                "The daily pipeline could not refresh the dashboard with same-run pipeline-history context. "
                f"Error: {dashboard_history_refresh_error or 'N/A'}. "
                f"Dashboard report: `{snapshot.get('dashboard_report_path')}`. "
                f"History review: `{snapshot.get('history_review_report_path')}`."
            ),
            "Inspect the dashboard refresh error and rerun dashboard with the completed pipeline-history review directory.",
        )

    history_review_alert_refresh_status = str(snapshot.get("history_review_alert_refresh_status") or "")
    history_review_alert_refresh_error = snapshot.get("history_review_alert_refresh_error")
    if history_review_alert_refresh_status == "failed":
        _alert(
            alerts,
            "warning",
            "history_review_alert_refresh_failed",
            "Alert-aware history review refresh failed",
            (
                "The daily pipeline could not rerun pipeline-history after writing same-run daily alerts. "
                f"Error: {history_review_alert_refresh_error or 'N/A'}. "
                f"History review: `{snapshot.get('history_review_report_path')}`."
            ),
            "Inspect the alert-aware history refresh error and rerun daily-pipeline or pipeline-history after alerts are written.",
        )

    current_drawdown = _as_float(snapshot.get("paper_current_drawdown"))
    if current_drawdown is not None:
        if current_drawdown <= current_drawdown_critical:
            _alert(
                alerts,
                "critical",
                "current_drawdown_critical",
                "Paper account current drawdown is critical",
                f"Current drawdown is {_pct(current_drawdown)}, below critical threshold {_pct(current_drawdown_critical)}.",
                "Review whether the defensive core remains within the accepted risk envelope.",
            )
        elif current_drawdown <= current_drawdown_warning:
            _alert(
                alerts,
                "warning",
                "current_drawdown_warning",
                "Paper account current drawdown is elevated",
                f"Current drawdown is {_pct(current_drawdown)}, below warning threshold {_pct(current_drawdown_warning)}.",
                "Review drawdown trend and compare with allocator risk state.",
            )

    max_drawdown = _as_float(snapshot.get("paper_max_drawdown"))
    if max_drawdown is not None and max_drawdown <= max_drawdown_limit:
        _alert(
            alerts,
            "warning",
            "max_drawdown_limit_breached",
            "Paper account max drawdown exceeded limit",
            f"Max drawdown is {_pct(max_drawdown)}, below limit {_pct(max_drawdown_limit)}.",
            "Do not promote the setup without rechecking risk assumptions.",
        )

    return_change = _as_float(snapshot.get("history_paper_total_return_change"))
    if return_change is not None:
        if return_change <= daily_return_drop_critical:
            _alert(
                alerts,
                "critical",
                "daily_return_drop_critical",
                "Paper total return dropped sharply",
                f"Total return changed by {_pct(return_change)} versus previous pipeline snapshot.",
                "Inspect the paper ledger and market state before accepting today's output.",
            )
        elif return_change <= daily_return_drop_warning:
            _alert(
                alerts,
                "warning",
                "daily_return_drop_warning",
                "Paper total return dropped",
                f"Total return changed by {_pct(return_change)} versus previous pipeline snapshot.",
                "Review the paper ledger and latest market state.",
            )

    if _bool(snapshot.get("history_dashboard_posture_changed")):
        _alert(
            alerts,
            "warning",
            "dashboard_posture_changed",
            "Dashboard posture changed",
            f"Dashboard posture changed from `{snapshot.get('history_previous_dashboard_posture')}` to `{snapshot.get('dashboard_posture')}`.",
            "Open the dashboard and compare freshness gates and market state.",
        )
    if _bool(snapshot.get("history_action_posture_changed")):
        _alert(
            alerts,
            "warning",
            "daily_action_posture_changed",
            "Daily action posture changed",
            f"Daily action posture changed from `{snapshot.get('history_previous_action_posture')}` to `{snapshot.get('action_posture')}`.",
            "Review phase-2 posture and paper account target weights.",
        )

    satellite_change = _as_float(snapshot.get("history_paper_satellite_weight_change"))
    if satellite_change is not None:
        if satellite_change > 1e-12:
            _alert(
                alerts,
                "warning",
                "satellite_weight_opened",
                "Satellite target weight increased",
                f"Satellite target weight changed by {_pct(satellite_change)}; latest satellite weight is {_pct(snapshot.get('paper_latest_satellite_weight'))}.",
                "Review why the allocator gate opened before treating the signal as usable.",
            )
        elif satellite_change < -1e-12:
            _alert(
                alerts,
                "info",
                "satellite_weight_reduced",
                "Satellite target weight decreased",
                f"Satellite target weight changed by {_pct(satellite_change)}; latest satellite weight is {_pct(snapshot.get('paper_latest_satellite_weight'))}.",
                "Confirm the risk-off or filter reason in the paper account audit.",
            )

    latest_satellite_weight = _as_float(snapshot.get("paper_latest_satellite_weight"))
    if latest_satellite_weight is not None and latest_satellite_weight > 0:
        _alert(
            alerts,
            "info",
            "satellite_weight_active",
            "Satellite target weight is active",
            f"Latest satellite target weight is {_pct(latest_satellite_weight)}.",
            "Review the allocator gate and paper-account audit before any manual interpretation.",
        )

    satellite_risk_budget_status = str(snapshot.get("satellite_risk_budget_status") or "")
    satellite_risk_budget_decision = str(snapshot.get("satellite_risk_budget_decision") or "")
    satellite_risk_budget_reason = snapshot.get("satellite_risk_budget_reason")
    satellite_risk_budget_report = snapshot.get("satellite_risk_budget_report_path")
    satellite_risk_budget_checklist = snapshot.get("satellite_risk_budget_checklist_path")
    satellite_risk_budget_recommended_weight = snapshot.get("satellite_risk_budget_recommended_satellite_weight")
    satellite_risk_budget_selected_horizon = snapshot.get("satellite_risk_budget_selected_horizon")
    satellite_risk_budget_ready_horizons = int(_as_float(snapshot.get("satellite_risk_budget_ready_horizon_count")) or 0)
    satellite_risk_budget_reason_text = _sentence(satellite_risk_budget_reason)
    if satellite_risk_budget_status == "failed" or satellite_risk_budget_decision == "review_failed":
        _alert(
            alerts,
            "warning",
            "satellite_risk_budget_failed",
            "Satellite risk-budget review failed",
            f"Risk-budget review status is `{satellite_risk_budget_status or 'N/A'}` with reason: {satellite_risk_budget_reason or 'N/A'}.",
            "Inspect the satellite risk-budget review setup before changing any research budget.",
        )
    elif satellite_risk_budget_decision:
        if satellite_risk_budget_decision == "eligible_for_small_satellite_trial":
            _alert(
                alerts,
                "info",
                "satellite_risk_budget_trial_eligible",
                "Satellite risk budget is eligible for small research trial",
                (
                    f"Recommended research-only satellite budget is {_pct(satellite_risk_budget_recommended_weight)} "
                    f"on horizon `{satellite_risk_budget_selected_horizon or 'N/A'}`. "
                    f"Review report: `{satellite_risk_budget_report}`. Checklist: `{satellite_risk_budget_checklist}`."
                ),
                "Open the risk-budget review before changing any research parameters; this is not a broker action.",
            )
        elif satellite_risk_budget_decision == "wait_for_outcome_samples":
            _alert(
                alerts,
                "info",
                "satellite_risk_budget_waiting",
                "Satellite risk budget is waiting for outcome samples",
                (
                    f"Risk-budget decision is `{satellite_risk_budget_decision}` with {satellite_risk_budget_ready_horizons} ready horizon(s). "
                    f"Reason: {satellite_risk_budget_reason_text}. Review report: `{satellite_risk_budget_report}`."
                ),
                "Keep collecting local outcome samples; do not increase satellite budget before the review gates mature.",
            )
        elif satellite_risk_budget_decision == "hold_default_core_base":
            _alert(
                alerts,
                "info",
                "satellite_risk_budget_hold",
                "Satellite risk budget remains core-base",
                (
                    f"Risk-budget decision is `{satellite_risk_budget_decision}`; recommended satellite budget is "
                    f"{_pct(satellite_risk_budget_recommended_weight)}. Reason: {satellite_risk_budget_reason_text}. "
                    f"Review report: `{satellite_risk_budget_report}`."
                ),
                "Keep the core-base posture until outcome evidence improves.",
            )
        elif satellite_risk_budget_decision == "blocked_by_pipeline_gates":
            _alert(
                alerts,
                "info",
                "satellite_risk_budget_blocked",
                "Satellite risk budget is blocked by upstream gates",
                (
                    f"Risk-budget decision is `{satellite_risk_budget_decision}`. "
                    f"Reason: {satellite_risk_budget_reason_text}. Review report: `{satellite_risk_budget_report}`."
                ),
                "Resolve upstream research gates before reviewing satellite budget.",
            )
        elif satellite_risk_budget_decision == "review_outcome_samples":
            _alert(
                alerts,
                "info",
                "satellite_risk_budget_review_samples",
                "Satellite risk budget needs outcome sample review",
                (
                    f"Risk-budget decision is `{satellite_risk_budget_decision}` with {satellite_risk_budget_ready_horizons} ready horizon(s). "
                    f"Review report: `{satellite_risk_budget_report}`."
                ),
                "Open the satellite risk-budget review before changing research parameters.",
            )

    trigger_review_count = int(_as_float(snapshot.get("paper_stock_target_review_trigger_count")) or 0)
    drawdown_review_count = int(_as_float(snapshot.get("paper_stock_target_review_drawdown_count")) or 0)
    review_required_count = int(_as_float(snapshot.get("paper_stock_target_review_required_count")) or 0)
    suppressed_layer_review_count = int(_as_float(snapshot.get("paper_stock_target_review_suppressed_layer_count")) or 0)
    watch_review_count = int(_as_float(snapshot.get("paper_stock_target_review_watch_count")) or 0)
    review_report = snapshot.get("paper_stock_target_review_report_path")
    action_report = snapshot.get("paper_stock_target_review_actions_report_path")
    assistant_report = snapshot.get("paper_stock_target_review_assistant_report_path")
    outcome_report = snapshot.get("paper_stock_target_review_outcomes_report_path")
    outcome_history_report = snapshot.get("paper_stock_target_review_outcomes_history_report_path")
    outcome_analysis_report = snapshot.get("paper_stock_target_review_outcome_analysis_report_path")
    outcome_calendar_report = snapshot.get("paper_stock_target_review_outcome_calendar_report_path")
    outcome_due_report = snapshot.get("paper_stock_target_review_outcome_due_report_path")
    notes_path = snapshot.get("paper_stock_target_review_notes_path")
    review_required_unreviewed_count = int(_as_float(snapshot.get("paper_stock_target_review_required_unreviewed_count")) or 0)
    stock_target_action_count = int(_as_float(snapshot.get("paper_stock_target_review_action_count")) or 0)
    manual_watch_count = int(_as_float(snapshot.get("paper_stock_target_review_manual_watch_count")) or 0)
    manual_exclude_candidate_count = int(_as_float(snapshot.get("paper_stock_target_review_manual_exclude_candidate_count")) or 0)
    manual_other_status_count = int(_as_float(snapshot.get("paper_stock_target_review_manual_other_status_count")) or 0)
    outcome_analysis_status = str(snapshot.get("paper_stock_target_review_outcome_analysis_status") or "")
    outcome_ready_horizon_count = int(_as_float(snapshot.get("paper_stock_target_review_outcome_analysis_ready_horizon_count")) or 0)
    outcome_next_1d = snapshot.get("paper_stock_target_review_outcome_maturity_next_1d_date")
    outcome_next_5d = snapshot.get("paper_stock_target_review_outcome_maturity_next_5d_date")
    outcome_next_10d = snapshot.get("paper_stock_target_review_outcome_maturity_next_10d_date")
    outcome_next_20d = snapshot.get("paper_stock_target_review_outcome_maturity_next_20d_date")
    outcome_calendar_next_date = snapshot.get("paper_stock_target_review_outcome_calendar_next_action_date")
    outcome_calendar_next_horizon = snapshot.get("paper_stock_target_review_outcome_calendar_next_action_horizon")
    outcome_due_count = int(_as_float(snapshot.get("paper_stock_target_review_outcome_due_row_count")) or 0)
    outcome_due_pending_count = int(_as_float(snapshot.get("paper_stock_target_review_outcome_due_pending_count")) or 0)
    outcome_due_next_date = snapshot.get("paper_stock_target_review_outcome_due_next_date")
    outcome_due_next_horizon = snapshot.get("paper_stock_target_review_outcome_due_next_horizon")
    market_cap_fields_present = any(
        key in snapshot
        for key in {
            "paper_stock_market_cap_cache_status",
            "paper_stock_market_cap_cache_row_count",
            "paper_stock_tracking_market_cap_missing_count",
        }
    )
    market_cap_cache_status = str(snapshot.get("paper_stock_market_cap_cache_status") or "")
    market_cap_row_count = int(_as_float(snapshot.get("paper_stock_market_cap_cache_row_count")) or 0)
    market_cap_missing_count = int(_as_float(snapshot.get("paper_stock_tracking_market_cap_missing_count")) or 0)
    market_cap_path = snapshot.get("paper_stock_market_cap_path")
    market_cap_latest_date = snapshot.get("paper_stock_market_cap_cache_latest_snapshot_date")
    market_cap_updated_at = snapshot.get("paper_stock_market_cap_cache_updated_at")
    market_cap_threshold = snapshot.get("paper_stock_tracking_max_market_cap_yi")
    stock_target_count = int(_as_float(snapshot.get("paper_stock_target_count")) or 0)
    market_cap_reference_date = _date_or_none(snapshot.get("paper_latest_date") or snapshot.get("local_equity_latest_date"))
    market_cap_snapshot_date = _date_or_none(market_cap_latest_date)
    stock_review_warning_severity = "warning"
    if stock_target_review_warning_only_after_close and snapshot.get("trading_day_gate_status") != "trading_day_data_ready":
        stock_review_warning_severity = "info"
    if trigger_review_count > 0:
        _alert(
            alerts,
            stock_review_warning_severity,
            "stock_target_trigger_review",
            "Target stocks matched trigger monitor",
            f"{trigger_review_count} target stock(s) matched the latest trigger-monitor signal file. Review panel: `{review_report}`.",
            "Open the stock-target review panel and inspect trigger context before relying on the target list.",
        )
    if drawdown_review_count > 0:
        if review_required_unreviewed_count > 0 or stock_target_action_count > 0:
            _alert(
                alerts,
                stock_review_warning_severity,
                "stock_target_drawdown_review",
                "Target stocks require drawdown review",
                f"{drawdown_review_count} target stock(s) are in drawdown_review. Review panel: `{review_report}`.",
                "Review loss source, source risk gate, and next-session gap risk; this is not a sell instruction.",
            )
        else:
            _alert(
                alerts,
                "info",
                "stock_target_drawdown_review_monitored",
                "Reviewed target-stock drawdowns remain visible",
                f"{drawdown_review_count} target stock(s) remain in drawdown_review, but no unreviewed row or open action is present. Review panel: `{review_report}`.",
                "Keep the reviewed drawdown rows visible for routine follow-up; this is not a broker action.",
            )
    if review_required_count > 0 and trigger_review_count == 0 and drawdown_review_count == 0:
        _alert(
            alerts,
            stock_review_warning_severity,
            "stock_target_review_required",
            "Target-stock review queue requires attention",
            f"{review_required_count} target stock(s) are marked review_required. Review panel: `{review_report}`.",
            "Open the stock-target review panel before using today's target-stock list.",
        )
    if suppressed_layer_review_count > 0:
        _alert(
            alerts,
            "info",
            "stock_target_suppressed_layer_review",
            "Suppressed source positions are visible",
            f"{suppressed_layer_review_count} source position(s) are suppressed by current layer weight. Review panel: `{review_report}`.",
            "Keep these rows visible for audit; portfolio target weight remains controlled by the layer gate.",
        )
    if watch_review_count > 0:
        _alert(
            alerts,
            "info",
            "stock_target_watch_review",
            "Target-stock watch rows are present",
            f"{watch_review_count} target stock(s) are in watch_review. Review panel: `{review_report}`.",
            "Review these rows during routine model review.",
        )
    if market_cap_fields_present:
        if market_cap_cache_status and market_cap_cache_status != "ok":
            _alert(
                alerts,
                "warning",
                "stock_market_cap_cache_unavailable",
                "Stock market-cap cache is unavailable",
                f"Market-cap cache status is `{market_cap_cache_status}` at `{market_cap_path}`; max tracking threshold is {market_cap_threshold} Yi Yuan.",
                "Refresh the market-cap cache, then rerun the daily pipeline before relying on large-cap tracking exclusions.",
            )
        elif market_cap_row_count <= 0:
            _alert(
                alerts,
                "warning",
                "stock_market_cap_cache_empty",
                "Stock market-cap cache is empty",
                f"Market-cap cache has {market_cap_row_count} row(s) at `{market_cap_path}`; max tracking threshold is {market_cap_threshold} Yi Yuan.",
                "Refresh the market-cap cache, then rerun the daily pipeline before relying on large-cap tracking exclusions.",
            )
        elif market_cap_snapshot_date is not None and market_cap_reference_date is not None and market_cap_snapshot_date < market_cap_reference_date:
            _alert(
                alerts,
                "warning",
                "stock_market_cap_cache_stale",
                "Stock market-cap cache is older than local model data",
                f"Market-cap snapshot date is {market_cap_latest_date}, while local paper/latest date is {market_cap_reference_date.date()}. Cache updated at `{market_cap_updated_at}`.",
                "Refresh the market-cap cache so large-cap tracking exclusions use the same local market date as the daily report.",
            )
        if market_cap_missing_count > 0:
            severity = "warning" if stock_target_count and market_cap_missing_count >= stock_target_count else "info"
            _alert(
                alerts,
                severity,
                "stock_market_cap_target_missing",
                "Some target stocks are missing market-cap data",
                f"{market_cap_missing_count} target row(s) have missing market cap out of {stock_target_count or 'N/A'} target rows. Cache status is `{market_cap_cache_status or 'N/A'}`.",
                "Inspect missing market-cap rows; they remain visible, but the 1500 Yi Yuan exclusion cannot be applied to those rows.",
            )
    if review_required_unreviewed_count > 0:
        _alert(
            alerts,
            "info",
            "stock_target_review_notes_open",
            "Target-stock review notes need update",
            f"{review_required_unreviewed_count} review_required row(s) have no manual review note yet. Assistant: `{assistant_report}`. Notes CSV: `{notes_path}`.",
            "Open the assistant report, then update manual_status/manual_note after human review; this is a research audit note, not a broker action.",
        )
    if manual_watch_count > 0:
        _alert(
            alerts,
            "info",
            "stock_target_manual_watch",
            "Manual review has watch rows",
            f"{manual_watch_count} current target-review row(s) are marked manual_status=watch. Notes CSV: `{notes_path}`.",
            "Keep these rows in routine follow-up; model targets and portfolio weights are unchanged.",
        )
    if manual_exclude_candidate_count > 0:
        _alert(
            alerts,
            "info",
            "stock_target_manual_exclude_candidate",
            "Manual review has exclusion candidates",
            f"{manual_exclude_candidate_count} current row(s) are marked manual_status=exclude_candidate. Notes CSV: `{notes_path}`.",
            "Review exclusion candidates separately; this status does not alter model targets or generate broker orders.",
        )
    if manual_other_status_count > 0:
        _alert(
            alerts,
            "info",
            "stock_target_manual_other_status",
            "Manual review has unrecognized statuses",
            f"{manual_other_status_count} current row(s) use manual_status values outside the supported set. Notes CSV: `{notes_path}`.",
            "Normalize manual_status to reviewed, watch, resolved, exclude_candidate, or unreviewed for consistent tracking.",
        )
    if outcome_due_count > 0:
        _alert(
            alerts,
            "info",
            "stock_target_outcome_due_queue",
            "Stock-target outcome due queue is open",
            f"{outcome_due_count} outcome horizon(s) with {outcome_due_pending_count} pending row(s) are due for recheck. Next due: {outcome_due_next_date or 'N/A'} / {outcome_due_next_horizon or 'N/A'}. Due report: `{outcome_due_report}`.",
            "Open the due queue, refresh local OHLCV if needed, then rerun paper-account or daily-pipeline; this is a research sample update, not a trade signal.",
        )
    if outcome_ready_horizon_count > 0:
        _alert(
            alerts,
            "info",
            "stock_target_outcome_analysis_ready",
            "Stock-target outcome analysis has mature horizons",
            f"{outcome_ready_horizon_count} outcome horizon(s) passed the readiness gates. Analysis report: `{outcome_analysis_report}`. Calendar: `{outcome_calendar_report}`.",
            "Open the outcome analysis and calendar reports before changing strategy parameters; this is a research audit, not a trade signal.",
        )
    elif outcome_analysis_status in {"waiting_for_evaluable_returns", "sample_insufficient"}:
        next_dates = " / ".join(
            [
                f"1D {outcome_next_1d or 'N/A'}",
                f"5D {outcome_next_5d or 'N/A'}",
                f"10D {outcome_next_10d or 'N/A'}",
                f"20D {outcome_next_20d or 'N/A'}",
            ]
        )
        _alert(
            alerts,
            "info",
            "stock_target_outcome_analysis_pending",
            "Stock-target outcome samples are still maturing",
            f"Outcome analysis status is `{outcome_analysis_status}`. Next estimated maturity dates: {next_dates}. Calendar next action: {outcome_calendar_next_date or 'N/A'} / {outcome_calendar_next_horizon or 'N/A'}. Analysis report: `{outcome_analysis_report}`. Calendar: `{outcome_calendar_report}`.",
            "Keep collecting local OHLCV outcomes and rerun the daily pipeline after the next estimated maturity date.",
        )

    live_shadow_preflight_status = str(snapshot.get("live_shadow_preflight_status") or "")
    live_shadow_preflight_decision = str(snapshot.get("live_shadow_preflight_decision") or "")
    live_shadow_preflight_blockers_count = int(_as_float(snapshot.get("live_shadow_preflight_blockers_count")) or 0)
    live_shadow_preflight_report = snapshot.get("live_shadow_preflight_report_path")
    live_shadow_status = str(snapshot.get("live_shadow_status") or "")
    if (
        live_shadow_status == "blocked_by_preflight"
        or live_shadow_preflight_status == "failed"
        or (live_shadow_preflight_status == "completed" and live_shadow_preflight_decision == "blocked")
    ):
        _alert(
            alerts,
            "warning",
            "live_shadow_preflight_blocked",
            "Live-shadow preflight is blocked",
            (
                f"Live-shadow preflight status is `{live_shadow_preflight_status}` (decision `{live_shadow_preflight_decision or 'N/A'}`), "
                f"with {live_shadow_preflight_blockers_count} blocker(s). "
                f"Live-shadow status: `{live_shadow_status or 'unknown'}`. Preflight report: `{live_shadow_preflight_report}`."
            ),
            (
                "Open the live-shadow preflight report and clear blockers or inspect the failure reason; "
                "then rerun daily-pipeline to refresh the live-shadow stage."
            ),
        )

    highest = "normal"
    for item in alerts:
        if SEVERITY_RANK[item["severity"]] > SEVERITY_RANK[highest]:
            highest = item["severity"]
    action_required = highest in {"warning", "critical"}
    action_stage = _action_stage(highest)
    action_items = _build_action_items(alerts)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": snapshot.get("as_of_date"),
        "alert_level": highest,
        "action_stage": action_stage,
        "action_summary": _action_summary(action_stage),
        "action_items": action_items,
        "alert_count": len(alerts),
        "action_required": action_required,
        "critical_count": sum(item["severity"] == "critical" for item in alerts),
        "warning_count": sum(item["severity"] == "warning" for item in alerts),
        "info_count": sum(item["severity"] == "info" for item in alerts),
        "alerts": alerts,
        "source_pipeline_report_path": snapshot.get("pipeline_report_path"),
        "source_pipeline_snapshot_path": snapshot.get("pipeline_snapshot_path"),
        "stock_target_review_warning_only_after_close": bool(stock_target_review_warning_only_after_close),
        "daily_check_report_path": snapshot.get("daily_check_report_path"),
        "paper_account_report_path": snapshot.get("paper_account_report_path"),
        "model_audit_status": model_audit_status,
        "model_audit_report_path": model_audit_report,
        "model_audit_actions_path": model_audit_actions,
        "model_audit_walk_forward_action_items": model_audit_action_items,
        "model_audit_walk_forward_resume_candidates": model_audit_resume_candidates,
        "model_audit_walk_forward_archive_review_candidates": model_audit_archive_candidates,
        "model_audit_top_action": model_audit_top_action,
        "promotion_status": promotion_status,
        "promotion_decision": promotion_decision,
        "promotion_sensitivity_group_support_count": int(promotion_group_support) if promotion_group_support is not None else None,
        "promotion_sensitivity_run_support_count": int(promotion_run_support) if promotion_run_support is not None else None,
        "promotion_min_sensitivity_support": int(promotion_min_support) if promotion_min_support is not None else None,
        "promotion_report_path": promotion_report,
        "satellite_risk_budget_status": satellite_risk_budget_status,
        "satellite_risk_budget_decision": satellite_risk_budget_decision,
        "satellite_risk_budget_reason": satellite_risk_budget_reason,
        "satellite_risk_budget_recommended_satellite_weight": _as_float(satellite_risk_budget_recommended_weight),
        "satellite_risk_budget_selected_horizon": satellite_risk_budget_selected_horizon,
        "satellite_risk_budget_ready_horizon_count": satellite_risk_budget_ready_horizons,
        "satellite_risk_budget_report_path": satellite_risk_budget_report,
        "satellite_risk_budget_snapshot_path": snapshot.get("satellite_risk_budget_snapshot_path"),
        "satellite_risk_budget_checklist_path": satellite_risk_budget_checklist,
        "live_shadow_preflight_status": live_shadow_preflight_status,
        "live_shadow_preflight_decision": live_shadow_preflight_decision,
        "live_shadow_preflight_report_path": live_shadow_preflight_report,
        "live_shadow_preflight_blockers_count": live_shadow_preflight_blockers_count,
        "live_shadow_status": live_shadow_status,
        "stock_target_review_report_path": snapshot.get("paper_stock_target_review_report_path"),
        "stock_target_review_actions_report_path": action_report,
        "stock_target_review_assistant_report_path": assistant_report,
        "stock_target_review_outcomes_report_path": outcome_report,
        "stock_target_review_outcomes_history_report_path": outcome_history_report,
        "stock_target_review_outcome_analysis_report_path": outcome_analysis_report,
        "stock_target_review_outcome_calendar_report_path": outcome_calendar_report,
        "stock_target_review_outcome_due_report_path": outcome_due_report,
        "stock_target_review_outcome_analysis_status": snapshot.get("paper_stock_target_review_outcome_analysis_status"),
        "stock_target_review_outcome_analysis_sample_warning": snapshot.get("paper_stock_target_review_outcome_analysis_sample_warning"),
        "stock_target_review_outcome_analysis_ready_horizon_count": outcome_ready_horizon_count,
        "stock_target_review_outcome_maturity_next_1d_date": outcome_next_1d,
        "stock_target_review_outcome_maturity_next_5d_date": outcome_next_5d,
        "stock_target_review_outcome_maturity_next_10d_date": outcome_next_10d,
        "stock_target_review_outcome_maturity_next_20d_date": outcome_next_20d,
        "stock_target_review_outcome_calendar_next_action_date": outcome_calendar_next_date,
        "stock_target_review_outcome_calendar_next_action_horizon": outcome_calendar_next_horizon,
        "stock_target_review_outcome_due_row_count": outcome_due_count,
        "stock_target_review_outcome_due_pending_count": outcome_due_pending_count,
        "stock_target_review_outcome_due_next_date": outcome_due_next_date,
        "stock_target_review_outcome_due_next_horizon": outcome_due_next_horizon,
        "stock_market_cap_cache_status": market_cap_cache_status,
        "stock_market_cap_cache_row_count": market_cap_row_count,
        "stock_market_cap_cache_latest_snapshot_date": market_cap_latest_date,
        "stock_market_cap_cache_updated_at": market_cap_updated_at,
        "stock_market_cap_path": market_cap_path,
        "stock_tracking_market_cap_missing_count": market_cap_missing_count,
        "stock_tracking_max_market_cap_yi": market_cap_threshold,
        "stock_target_review_notes_path": snapshot.get("paper_stock_target_review_notes_path"),
        "stock_target_review_required_unreviewed_count": review_required_unreviewed_count,
        "stock_target_review_manual_watch_count": manual_watch_count,
        "stock_target_review_manual_exclude_candidate_count": manual_exclude_candidate_count,
        "stock_target_review_manual_other_status_count": manual_other_status_count,
        "daily_preflight_skipped": bool(snapshot.get("daily_preflight_skipped", False)),
        "dashboard_report_path": snapshot.get("dashboard_report_path"),
        "dashboard_history_refresh_status": dashboard_history_refresh_status,
        "dashboard_history_refresh_error": dashboard_history_refresh_error,
        "dashboard_history_refresh_report_path": snapshot.get("dashboard_history_refresh_report_path"),
        "dashboard_history_refresh_snapshot_path": snapshot.get("dashboard_history_refresh_snapshot_path"),
        "history_review_report_path": snapshot.get("history_review_report_path"),
        "history_review_alert_refresh_status": history_review_alert_refresh_status,
        "history_review_alert_refresh_error": history_review_alert_refresh_error,
        "history_review_latest_alert_level": snapshot.get("history_review_latest_alert_level"),
        "history_review_latest_alert_action_stage": snapshot.get("history_review_latest_alert_action_stage"),
        "history_review_latest_alert_count": snapshot.get("history_review_latest_alert_count"),
        "dashboard_alert_refresh_status": snapshot.get("dashboard_alert_refresh_status"),
        "dashboard_alert_refresh_error": snapshot.get("dashboard_alert_refresh_error"),
    }


def _render_alert_report(payload: dict[str, Any]) -> str:
    alerts = payload.get("alerts") or []
    action_items = payload.get("action_items") or []
    if alerts:
        rows = "\n".join(
            f"| `{item['severity']}` | `{item['code']}` | {item['title']} | {item['recommended_review']} |"
            for item in alerts
        )
    else:
        rows = "| `normal` | `none` | No alert rules fired | Continue routine review only |"
    if action_items:
        action_rows = "\n".join(
            f"| `{item['severity']}` | `{item['action_code']}` | {item['title']} | {item['next_step']} |"
            for item in action_items
        )
    else:
        action_rows = "| `normal` | `routine_review` | Routine review | No alert rules fired; continue routine review only. |"
    details = "\n\n".join(
        f"### {item['title']}\n\n- Severity: `{item['severity']}`\n- Code: `{item['code']}`\n- Detail: {item['detail']}\n- Review: {item['recommended_review']}"
        for item in alerts
    )
    if not details:
        details = "No warning or critical alert was generated."
    return f"""# Daily Pipeline Alerts

Generated at: `{payload.get("generated_at")}`

This alert file is research-only. It does not connect to brokers, place orders, or provide investment advice.

## Summary

| Item | Value |
| --- | ---: |
| As-of date | {payload.get("as_of_date", "N/A")} |
| Alert level | `{payload.get("alert_level")}` |
| Action stage | `{payload.get("action_stage")}` |
| Action required | `{payload.get("action_required")}` |
| Daily preflight skipped | `{payload.get("daily_preflight_skipped")}` |
| Action summary | {payload.get("action_summary")} |
| Critical / warning / info | {payload.get("critical_count", 0)} / {payload.get("warning_count", 0)} / {payload.get("info_count", 0)} |
| Total alerts | {payload.get("alert_count", 0)} |

## Action Playbook

| Severity | Action | Title | Next step |
| --- | --- | --- | --- |
{action_rows}

## Alerts

| Severity | Code | Title | Review |
| --- | --- | --- | --- |
{rows}

## Details

{details}

## Source Files

- Pipeline report: `{payload.get("source_pipeline_report_path")}`
- Pipeline snapshot: `{payload.get("source_pipeline_snapshot_path")}`
- Daily check: `{payload.get("daily_check_report_path")}`
- Paper account: `{payload.get("paper_account_report_path")}`
- Model build audit: `{payload.get("model_audit_report_path")}`
- Model build audit actions: `{payload.get("model_audit_actions_path")}`
- Daily preflight skipped: `{payload.get("daily_preflight_skipped")}`
- Satellite risk-budget review: `{payload.get("satellite_risk_budget_report_path")}`
- Satellite risk-budget decision / recommendation: `{payload.get("satellite_risk_budget_decision")}` / `{_pct(payload.get("satellite_risk_budget_recommended_satellite_weight"))}`
- Satellite risk-budget checklist: `{payload.get("satellite_risk_budget_checklist_path")}`
- Stock target review: `{payload.get("stock_target_review_report_path")}`
- Stock target review actions: `{payload.get("stock_target_review_actions_report_path")}`
- Stock target review assistant: `{payload.get("stock_target_review_assistant_report_path")}`
- Stock target review outcomes: `{payload.get("stock_target_review_outcomes_report_path")}`
- Stock target review outcome history: `{payload.get("stock_target_review_outcomes_history_report_path")}`
- Stock target review outcome analysis: `{payload.get("stock_target_review_outcome_analysis_report_path")}`
- Stock target review outcome calendar: `{payload.get("stock_target_review_outcome_calendar_report_path")}`
- Stock target review outcome due queue: `{payload.get("stock_target_review_outcome_due_report_path")}`
- Stock target review outcome analysis status: `{payload.get("stock_target_review_outcome_analysis_status")}`
- Stock target review outcome ready horizons: `{payload.get("stock_target_review_outcome_analysis_ready_horizon_count")}`
- Stock target review outcome next 1D / 5D maturity: `{payload.get("stock_target_review_outcome_maturity_next_1d_date")}` / `{payload.get("stock_target_review_outcome_maturity_next_5d_date")}`
- Stock target review outcome calendar next action: `{payload.get("stock_target_review_outcome_calendar_next_action_date")}` / `{payload.get("stock_target_review_outcome_calendar_next_action_horizon")}`
- Stock target review outcome due count / next: `{payload.get("stock_target_review_outcome_due_row_count")}` / `{payload.get("stock_target_review_outcome_due_next_date")}` / `{payload.get("stock_target_review_outcome_due_next_horizon")}`
- Stock market-cap cache: status `{payload.get("stock_market_cap_cache_status")}`, rows `{payload.get("stock_market_cap_cache_row_count")}`, latest `{payload.get("stock_market_cap_cache_latest_snapshot_date")}`, missing target rows `{payload.get("stock_tracking_market_cap_missing_count")}`
- Stock market-cap cache path: `{payload.get("stock_market_cap_path")}`
- Stock target review notes: `{payload.get("stock_target_review_notes_path")}`
- Dashboard: `{payload.get("dashboard_report_path")}`
- Dashboard history refresh: `{payload.get("dashboard_history_refresh_status")}` / `{payload.get("dashboard_history_refresh_report_path")}`
- History review: `{payload.get("history_review_report_path")}`
"""


def write_daily_alerts(
    snapshot: dict[str, Any],
    output_dir: str | Path,
    latest_report_path: str | Path | None = None,
    stock_target_review_warning_only_after_close: bool = False,
    publish_artifacts: bool = True,
) -> DailyAlertsResult:
    resolved_output = Path(output_dir)
    report_path = resolved_output / "alerts.md"
    json_path = resolved_output / "alerts.json"
    latest_path = Path(latest_report_path) if latest_report_path is not None else resolved_output.parent / "latest_alerts.md"
    payload = build_daily_alert_payload(
        snapshot,
        stock_target_review_warning_only_after_close=stock_target_review_warning_only_after_close,
    )
    payload["alerts_report_path"] = str(report_path)
    payload["alerts_json_path"] = str(json_path)
    payload["latest_alerts_path"] = str(latest_path)
    if publish_artifacts:
        payload, changed = publish_json_if_semantically_changed(json_path, payload)
        if changed or not report_path.exists() or not latest_path.exists():
            report_text = _render_alert_report(payload)
            write_text_if_changed(report_path, report_text)
            write_text_if_changed(latest_path, report_text)
    return DailyAlertsResult(
        output_dir=resolved_output,
        report_path=report_path,
        json_path=json_path,
        latest_report_path=latest_path,
        payload=payload,
    )
