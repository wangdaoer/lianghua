"""Daily pipeline history review and drift checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_HISTORY_FILE = Path("outputs/research/daily_pipeline_history.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/pipeline_history_review_latest")
EXPECTED_DASHBOARD_POSTURE = "core_base_watch_allocator_gate"
EXPECTED_FRESHNESS = "fresh_enough"
HARD_FRESHNESS_FIELDS = [
    "data_freshness_status",
    "market_cache_status",
    "allocator_input_status",
    "paper_freshness_status",
]
SOFT_FRESHNESS_FIELDS = [
    "sentiment_freshness_status",
    "trigger_freshness_status",
]
NUMERIC_COLUMNS = [
    "promotion_sensitivity_group_support_count",
    "promotion_sensitivity_run_support_count",
    "promotion_min_sensitivity_support",
    "paper_latest_core_weight",
    "paper_latest_satellite_weight",
    "paper_latest_cash_weight",
    "paper_final_equity",
    "paper_total_return",
    "paper_max_drawdown",
    "paper_current_drawdown",
    "paper_sharpe",
    "paper_audit_event_count",
    "satellite_risk_budget_recommended_satellite_weight",
    "dashboard_pipeline_history_alert_count",
    "dashboard_pipeline_history_latest_satellite_risk_budget_recommended_weight",
    "alert_count",
    "alert_action_item_count",
    "alert_critical_count",
    "alert_warning_count",
    "alert_info_count",
]


@dataclass(frozen=True)
class PipelineHistoryReviewResult:
    output_dir: Path
    snapshot_path: Path
    report_path: Path
    history_path: Path
    snapshot: dict[str, Any]
    history: pd.DataFrame


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = Path(path) if path is not None else default
    return raw if raw.is_absolute() else project_root / raw


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) == 8 and text.isdigit():
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).date()


def _read_history(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return pd.DataFrame(), "missing"
    if path.stat().st_size == 0:
        return pd.DataFrame(), "empty"
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), "empty"
    if frame.empty:
        return pd.DataFrame(), "empty"
    return frame, "ok"


def _clean_history(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in NUMERIC_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["_as_of_sort"] = pd.to_datetime(data.get("as_of_date"), errors="coerce")
    data["_generated_sort"] = pd.to_datetime(data.get("generated_at"), errors="coerce")
    data = data.dropna(subset=["_as_of_sort"]).sort_values(["_as_of_sort", "_generated_sort"], na_position="first")
    return data.reset_index(drop=True)


def _scalar(row: pd.Series, key: str, default: Any = None) -> Any:
    if key not in row:
        return default
    value = row.get(key)
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        return value
    return value


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


def _pp(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "N/A"
    return f"{number * 100:+.2f} pp"


def _num(value: Any, digits: int = 2) -> str:
    number = _as_float(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def _text(value: Any, default: str = "") -> str:
    if value in (None, ""):
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return str(value).strip()


def _count_changes(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns or len(frame) <= 1:
        return 0
    values = frame[column].astype(str)
    return int(values.ne(values.shift()).iloc[1:].sum())


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _freshness_failures(latest: pd.Series, fields: list[str]) -> list[str]:
    failures: list[str] = []
    for field in fields:
        value = _scalar(latest, field, "")
        if str(value) != EXPECTED_FRESHNESS:
            failures.append(f"{field}={value}")
    return failures


def _trend_label(final_equity_delta: float | None, current_drawdown_delta: float | None) -> str:
    if final_equity_delta is None:
        return "insufficient_history"
    if final_equity_delta > 0 and (current_drawdown_delta is None or current_drawdown_delta >= 0):
        return "improving"
    if final_equity_delta < 0 or (current_drawdown_delta is not None and current_drawdown_delta < 0):
        return "deteriorating"
    return "flat"


def _render_report(snapshot: dict[str, Any]) -> str:
    alerts = snapshot.get("alerts", [])
    if alerts:
        alerts_text = "\n".join(f"- {item}" for item in alerts)
    else:
        alerts_text = "- No history review alerts."

    return f"""# Pipeline History Review

Generated at: `{snapshot.get("generated_at")}`

This is a local research operations report. It does not connect to brokers, place orders, or provide investment advice.

## Health

| Item | Value |
| --- | ---: |
| Health state | `{snapshot.get("health_state")}` |
| Alert count | {snapshot.get("alert_count")} |
| History status | `{snapshot.get("history_status")}` |
| History freshness | `{snapshot.get("history_freshness_status")}` |
| As-of date | {snapshot.get("as_of_date")} |
| Latest history as-of | {snapshot.get("latest_as_of_date", "N/A")} |
| Days since latest history | {snapshot.get("days_since_latest_history", "N/A")} |

## Latest Run

| Item | Value |
| --- | ---: |
| Dashboard posture | `{snapshot.get("latest_dashboard_posture", "N/A")}` |
| Daily action posture | `{snapshot.get("latest_action_posture", "N/A")}` |
| Data freshness | `{snapshot.get("latest_data_freshness_status", "N/A")}` |
| Market cache | `{snapshot.get("latest_market_cache_status", "N/A")}` |
| Allocator inputs | `{snapshot.get("latest_allocator_input_status", "N/A")}` |
| Paper regime | `{snapshot.get("latest_paper_regime", "N/A")}` |
| Paper candidate | `{snapshot.get("latest_paper_candidate", "N/A")}` |
| Core weight | {_pct(snapshot.get("latest_core_weight"))} |
| Satellite weight | {_pct(snapshot.get("latest_satellite_weight"))} |
| Paper final equity | {_num(snapshot.get("latest_final_equity"), 2)} |
| Paper total return | {_pct(snapshot.get("latest_total_return"))} |
| Paper current drawdown | {_pct(snapshot.get("latest_current_drawdown"))} |
| Paper Sharpe | {_num(snapshot.get("latest_sharpe"), 3)} |
| Promotion decision | `{snapshot.get("latest_promotion_decision", "N/A")}` |
| Promotion group/run support | {_num(snapshot.get("latest_promotion_group_support_count"), 0)} / {_num(snapshot.get("latest_promotion_min_support"), 0)} / run {_num(snapshot.get("latest_promotion_run_support_count"), 0)} |

## Dashboard History Refresh

| Item | Value |
| --- | ---: |
| Refresh status | `{snapshot.get("latest_dashboard_history_refresh_status", "N/A")}` |
| Refresh error | {snapshot.get("latest_dashboard_history_refresh_error", "N/A")} |
| Alert refresh status | `{snapshot.get("latest_dashboard_alert_refresh_status", "N/A")}` |
| Alert refresh error | {snapshot.get("latest_dashboard_alert_refresh_error", "N/A")} |
| Refreshed dashboard | `{snapshot.get("latest_dashboard_history_refresh_report_path", "N/A")}` |
| Dashboard pipeline-history status | `{snapshot.get("latest_dashboard_pipeline_history_status", "N/A")}` |
| Dashboard pipeline-history health | `{snapshot.get("latest_dashboard_pipeline_history_health_state", "N/A")}` |
| Dashboard pipeline-history alerts | {_num(snapshot.get("latest_dashboard_pipeline_history_alert_count"), 0)} |
| Dashboard pipeline-history as-of | {snapshot.get("latest_dashboard_pipeline_history_latest_as_of_date", "N/A")} |
| Dashboard risk-budget decision | `{snapshot.get("latest_dashboard_pipeline_history_latest_satellite_risk_budget_decision", "N/A")}` |
| Dashboard risk-budget recommendation | {_pct(snapshot.get("latest_dashboard_pipeline_history_latest_satellite_risk_budget_recommended_weight"))} |

## Daily Alerts

| Item | Value |
| --- | ---: |
| Alert level | `{snapshot.get("latest_alert_level", "N/A")}` |
| Alert count | {_num(snapshot.get("latest_alert_count"), 0)} |
| Critical / warning / info | {_num(snapshot.get("latest_alert_critical_count"), 0)} / {_num(snapshot.get("latest_alert_warning_count"), 0)} / {_num(snapshot.get("latest_alert_info_count"), 0)} |
| Action stage | `{snapshot.get("latest_alert_action_stage", "N/A")}` |
| Action required | `{snapshot.get("latest_alert_action_required", "N/A")}` |
| Action items | {_num(snapshot.get("latest_alert_action_item_count"), 0)} |
| Action summary | {snapshot.get("latest_alert_action_summary", "N/A")} |
| Alert report | `{snapshot.get("latest_alerts_report_path", "N/A")}` |

## Satellite Risk Budget

| Item | Value |
| --- | ---: |
| Status | `{snapshot.get("latest_satellite_risk_budget_status", "N/A")}` |
| Decision | `{snapshot.get("latest_satellite_risk_budget_decision", "N/A")}` |
| Previous decision | `{snapshot.get("previous_satellite_risk_budget_decision", "N/A")}` |
| Decision changed | `{snapshot.get("satellite_risk_budget_decision_changed", False)}` |
| Reason | {snapshot.get("latest_satellite_risk_budget_reason", "N/A")} |
| Recommended satellite budget | {_pct(snapshot.get("latest_satellite_risk_budget_recommended_satellite_weight"))} |
| Recommended budget change | {_pp(snapshot.get("satellite_risk_budget_recommended_weight_change"))} |
| Selected outcome horizon | `{snapshot.get("latest_satellite_risk_budget_selected_horizon", "N/A")}` |
| Review report | `{snapshot.get("latest_satellite_risk_budget_report_path", "N/A")}` |

## Lookback

| Item | Value |
| --- | ---: |
| Lookback runs | {snapshot.get("lookback_runs")} |
| Final equity change | {_num(snapshot.get("lookback_final_equity_delta"), 2)} |
| Total return change | {_pp(snapshot.get("lookback_total_return_delta"))} |
| Current drawdown change | {_pp(snapshot.get("lookback_current_drawdown_delta"))} |
| Worst current drawdown | {_pct(snapshot.get("lookback_worst_current_drawdown"))} |
| Minimum Sharpe | {_num(snapshot.get("lookback_min_sharpe"), 3)} |
| Dashboard posture changes | {snapshot.get("lookback_dashboard_posture_changes")} |
| Action posture changes | {snapshot.get("lookback_action_posture_changes")} |
| Satellite active runs | {snapshot.get("lookback_satellite_active_runs")} |
| Dashboard history refresh failures | {snapshot.get("lookback_dashboard_history_refresh_failed_runs")} |
| Dashboard history refresh status changes | {snapshot.get("lookback_dashboard_history_refresh_status_changes")} |
| Dashboard alert refresh failures | {snapshot.get("lookback_dashboard_alert_refresh_failed_runs")} |
| Dashboard alert refresh status changes | {snapshot.get("lookback_dashboard_alert_refresh_status_changes")} |
| Warning/critical daily alert runs | {snapshot.get("lookback_alert_warning_or_critical_runs")} |
| Daily alert level changes | {snapshot.get("lookback_alert_level_changes")} |
| Satellite risk-budget decision changes | {snapshot.get("lookback_satellite_risk_budget_decision_changes")} |
| Satellite risk-budget trial-eligible runs | {snapshot.get("lookback_satellite_risk_budget_trial_eligible_runs")} |
| Trend label | `{snapshot.get("lookback_trend_label")}` |

## Alerts

{alerts_text}

## Files

- History file: `{snapshot.get("history_path")}`
- Snapshot: `{snapshot.get("snapshot_path")}`
- Report: `{snapshot.get("report_path")}`
"""


def run_pipeline_history_review(
    project_root: str | Path = Path("."),
    history_file: str | Path | None = DEFAULT_HISTORY_FILE,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    as_of_date: str | date | None = None,
    lookback_runs: int = 20,
    max_staleness_days: int = 3,
    drawdown_watch_threshold: float = -0.08,
    min_sharpe_watch: float = 0.5,
    history_frame: pd.DataFrame | None = None,
) -> PipelineHistoryReviewResult:
    root = Path(project_root).resolve()
    as_of = _parse_date(as_of_date) if as_of_date is not None else datetime.now().date()
    if as_of is None:
        raise ValueError(f"Invalid as_of_date: {as_of_date}")
    if lookback_runs <= 0:
        raise ValueError("lookback_runs must be positive.")

    resolved_history = _resolve(root, history_file, DEFAULT_HISTORY_FILE)
    resolved_output = _resolve(root, output_dir, DEFAULT_OUTPUT_DIR)
    if history_frame is None:
        history, history_status = _read_history(resolved_history)
    else:
        history = history_frame.copy()
        history_status = "empty" if history.empty else "ok"
    clean = _clean_history(history) if history_status == "ok" else pd.DataFrame()

    generated_at = datetime.now().isoformat(timespec="seconds")
    snapshot_path = resolved_output / "pipeline_history_review_snapshot.json"
    report_path = resolved_output / "pipeline_history_review.md"
    snapshot: dict[str, Any] = {
        "generated_at": generated_at,
        "as_of_date": as_of.isoformat(),
        "history_path": str(resolved_history),
        "output_dir": str(resolved_output),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "history_status": history_status,
        "history_row_count": int(len(history)),
        "clean_history_row_count": int(len(clean)),
        "lookback_runs": int(lookback_runs),
        "max_staleness_days": int(max_staleness_days),
        "drawdown_watch_threshold": float(drawdown_watch_threshold),
        "min_sharpe_watch": float(min_sharpe_watch),
        "alerts": [],
    }

    alerts: list[str] = []
    if history_status != "ok" or clean.empty:
        alerts.append(f"History is {history_status}; run daily-pipeline first.")
        snapshot.update(
            {
                "health_state": "blocked",
                "history_freshness_status": "missing",
                "alert_count": len(alerts),
                "alerts": alerts,
            }
        )
    else:
        latest = clean.iloc[-1]
        latest_as_of = pd.Timestamp(latest["_as_of_sort"]).date()
        days_since = (as_of - latest_as_of).days
        history_freshness = "fresh_enough" if days_since <= max_staleness_days else "stale"
        if days_since < 0:
            history_freshness = "future_dated"

        lookback = clean.tail(lookback_runs)
        first = lookback.iloc[0]
        final_equity_delta = None
        first_equity = _as_float(_scalar(first, "paper_final_equity"))
        latest_equity = _as_float(_scalar(latest, "paper_final_equity"))
        if first_equity is not None and latest_equity is not None:
            final_equity_delta = latest_equity - first_equity

        total_return_delta = None
        first_return = _as_float(_scalar(first, "paper_total_return"))
        latest_return = _as_float(_scalar(latest, "paper_total_return"))
        if first_return is not None and latest_return is not None:
            total_return_delta = latest_return - first_return

        current_drawdown_delta = None
        first_drawdown = _as_float(_scalar(first, "paper_current_drawdown"))
        latest_drawdown = _as_float(_scalar(latest, "paper_current_drawdown"))
        if first_drawdown is not None and latest_drawdown is not None:
            current_drawdown_delta = latest_drawdown - first_drawdown

        hard_failures = _freshness_failures(latest, HARD_FRESHNESS_FIELDS)
        soft_failures = _freshness_failures(latest, SOFT_FRESHNESS_FIELDS)
        latest_posture = str(_scalar(latest, "dashboard_posture", ""))
        latest_sharpe = _as_float(_scalar(latest, "paper_sharpe"))
        latest_current_drawdown = _as_float(_scalar(latest, "paper_current_drawdown"))
        latest_promotion_decision = str(_scalar(latest, "promotion_decision", ""))
        latest_promotion_headline_gate = str(_scalar(latest, "promotion_candidate_passes_headline_gate", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        latest_promotion_group_support = _as_float(_scalar(latest, "promotion_sensitivity_group_support_count"))
        latest_promotion_run_support = _as_float(_scalar(latest, "promotion_sensitivity_run_support_count"))
        latest_promotion_min_support = _as_float(_scalar(latest, "promotion_min_sensitivity_support"))
        latest_dashboard_history_refresh_status = _text(_scalar(latest, "dashboard_history_refresh_status"))
        latest_dashboard_history_refresh_error = _text(_scalar(latest, "dashboard_history_refresh_error"))
        latest_dashboard_history_refresh_report_path = _text(_scalar(latest, "dashboard_history_refresh_report_path"))
        latest_dashboard_history_refresh_snapshot_path = _text(_scalar(latest, "dashboard_history_refresh_snapshot_path"))
        latest_dashboard_pipeline_history_status = _text(_scalar(latest, "dashboard_pipeline_history_status"))
        latest_dashboard_pipeline_history_health_state = _text(_scalar(latest, "dashboard_pipeline_history_health_state"))
        latest_dashboard_alert_refresh_status = _text(_scalar(latest, "dashboard_alert_refresh_status"))
        latest_dashboard_alert_refresh_error = _text(_scalar(latest, "dashboard_alert_refresh_error"))
        latest_dashboard_pipeline_history_alert_count = _as_float(
            _scalar(latest, "dashboard_pipeline_history_alert_count")
        )
        latest_dashboard_pipeline_history_latest_as_of_date = _text(
            _scalar(latest, "dashboard_pipeline_history_latest_as_of_date")
        )
        latest_dashboard_pipeline_history_risk_budget_decision = _text(
            _scalar(latest, "dashboard_pipeline_history_latest_satellite_risk_budget_decision")
        )
        latest_dashboard_pipeline_history_risk_budget_weight = _as_float(
            _scalar(latest, "dashboard_pipeline_history_latest_satellite_risk_budget_recommended_weight")
        )
        latest_alert_level = _text(_scalar(latest, "alert_level"))
        latest_alert_count = _as_float(_scalar(latest, "alert_count"))
        latest_alert_action_stage = _text(_scalar(latest, "alert_action_stage"))
        latest_alert_action_summary = _text(_scalar(latest, "alert_action_summary"))
        latest_alert_action_item_count = _as_float(_scalar(latest, "alert_action_item_count"))
        latest_alert_action_required = _text(_scalar(latest, "alert_action_required"))
        latest_alert_critical_count = _as_float(_scalar(latest, "alert_critical_count"))
        latest_alert_warning_count = _as_float(_scalar(latest, "alert_warning_count"))
        latest_alert_info_count = _as_float(_scalar(latest, "alert_info_count"))
        latest_alerts_report_path = _text(_scalar(latest, "alerts_report_path"))
        latest_alerts_json_path = _text(_scalar(latest, "alerts_json_path"))
        latest_alerts_latest_report_path = _text(_scalar(latest, "latest_alerts_path"))
        latest_risk_budget_status = _text(_scalar(latest, "satellite_risk_budget_status"))
        latest_risk_budget_decision = _text(_scalar(latest, "satellite_risk_budget_decision"))
        latest_risk_budget_reason = _text(_scalar(latest, "satellite_risk_budget_reason"))
        latest_risk_budget_recommended_weight = _as_float(
            _scalar(latest, "satellite_risk_budget_recommended_satellite_weight")
        )
        latest_risk_budget_selected_horizon = _text(_scalar(latest, "satellite_risk_budget_selected_horizon"))
        latest_risk_budget_report_path = _text(_scalar(latest, "satellite_risk_budget_report_path"))
        latest_risk_budget_snapshot_path = _text(_scalar(latest, "satellite_risk_budget_snapshot_path"))
        latest_risk_budget_checklist_path = _text(_scalar(latest, "satellite_risk_budget_checklist_path"))
        previous_risk_budget_decision = ""
        previous_risk_budget_recommended_weight: float | None = None
        if len(clean) >= 2:
            previous = clean.iloc[-2]
            previous_risk_budget_decision = _text(_scalar(previous, "satellite_risk_budget_decision"))
            previous_risk_budget_recommended_weight = _as_float(
                _scalar(previous, "satellite_risk_budget_recommended_satellite_weight")
            )
        risk_budget_decision_changed = bool(
            latest_risk_budget_decision
            and previous_risk_budget_decision
            and latest_risk_budget_decision != previous_risk_budget_decision
        )
        risk_budget_recommended_weight_change = None
        if latest_risk_budget_recommended_weight is not None and previous_risk_budget_recommended_weight is not None:
            risk_budget_recommended_weight_change = (
                latest_risk_budget_recommended_weight - previous_risk_budget_recommended_weight
            )

        if "dashboard_alert_refresh_status" in lookback.columns:
            dashboard_alert_refresh_failed_runs = int(
                lookback["dashboard_alert_refresh_status"].astype(str).eq("failed").sum()
            )
        else:
            dashboard_alert_refresh_failed_runs = 0

        if history_freshness != "fresh_enough":
            alerts.append(f"History freshness is {history_freshness}; latest history as-of is {latest_as_of.isoformat()}.")
        if hard_failures:
            alerts.append("Hard freshness gates failed: " + ", ".join(hard_failures))
        if soft_failures:
            alerts.append("Reference freshness gates need review: " + ", ".join(soft_failures))
        if latest_posture != EXPECTED_DASHBOARD_POSTURE:
            alerts.append(f"Dashboard posture is {latest_posture}, expected {EXPECTED_DASHBOARD_POSTURE}.")
        if latest_current_drawdown is not None and latest_current_drawdown <= drawdown_watch_threshold:
            alerts.append(
                f"Paper current drawdown {_pct(latest_current_drawdown)} breached watch threshold {_pct(drawdown_watch_threshold)}."
            )
        if latest_sharpe is not None and latest_sharpe < min_sharpe_watch:
            alerts.append(f"Paper Sharpe {_num(latest_sharpe, 3)} is below watch threshold {_num(min_sharpe_watch, 3)}.")
        if (
            latest_promotion_decision == "watch_candidate"
            and latest_promotion_headline_gate
            and latest_promotion_group_support is not None
            and latest_promotion_run_support is not None
            and latest_promotion_min_support is not None
            and latest_promotion_group_support < latest_promotion_min_support
            and latest_promotion_run_support >= latest_promotion_min_support
            and latest_promotion_run_support > latest_promotion_group_support
        ):
            alerts.append(
                "Allocator promotion evidence is correlated: "
                f"group support {_num(latest_promotion_group_support, 0)} / {_num(latest_promotion_min_support, 0)}, "
                f"run support {_num(latest_promotion_run_support, 0)}."
            )
        if latest_dashboard_history_refresh_status == "failed":
            alerts.append(
                "Dashboard history refresh failed: "
                f"{latest_dashboard_history_refresh_error or 'no error recorded'}. "
                f"Dashboard report `{latest_dashboard_history_refresh_report_path or 'N/A'}`."
            )
        if latest_dashboard_alert_refresh_status == "failed":
            alerts.append(
                "Dashboard alert refresh failed: "
                f"{latest_dashboard_alert_refresh_error or 'no error recorded'}. "
                f"Dashboard history report `{latest_dashboard_history_refresh_report_path or 'N/A'}`."
            )
        if dashboard_alert_refresh_failed_runs >= 2:
            alerts.append(
                f"Dashboard alert refresh has failed {dashboard_alert_refresh_failed_runs} times in the "
                f"lookback window of {lookback_runs} runs."
            )
        if latest_alert_level.lower() in {"warning", "critical"}:
            alerts.append(
                f"Daily alert level is {latest_alert_level}: "
                f"stage `{latest_alert_action_stage or 'N/A'}`, "
                f"{_num(latest_alert_count, 0)} alert(s). "
                f"Report `{latest_alerts_report_path or 'N/A'}`."
            )
        if latest_risk_budget_status == "failed" or latest_risk_budget_decision == "review_failed":
            alerts.append(
                "Satellite risk-budget review failed: "
                f"{latest_risk_budget_reason or 'no reason recorded'}."
            )
        elif latest_risk_budget_decision == "eligible_for_small_satellite_trial":
            alerts.append(
                "Satellite risk-budget is eligible for a small satellite trial: "
                f"recommended budget {_pct(latest_risk_budget_recommended_weight)} "
                f"on horizon `{latest_risk_budget_selected_horizon or 'N/A'}`."
            )
        if risk_budget_decision_changed:
            alerts.append(
                "Satellite risk-budget decision changed from "
                f"`{previous_risk_budget_decision}` to `{latest_risk_budget_decision}`."
            )
        if risk_budget_recommended_weight_change is not None and risk_budget_recommended_weight_change > 1e-9:
            alerts.append(
                "Satellite risk-budget recommended weight increased by "
                f"{_pp(risk_budget_recommended_weight_change)} to {_pct(latest_risk_budget_recommended_weight)}."
            )

        hard_blocked = history_freshness not in {"fresh_enough", "future_dated"} or bool(hard_failures)
        if hard_blocked:
            health_state = "blocked"
        elif alerts:
            health_state = "watch"
        else:
            health_state = "ok"

        satellite_weights = _numeric_column(lookback, "paper_latest_satellite_weight")
        current_drawdowns = _numeric_column(lookback, "paper_current_drawdown")
        sharpes = _numeric_column(lookback, "paper_sharpe")
        if "dashboard_history_refresh_status" in lookback.columns:
            dashboard_refresh_failed_runs = int(
                lookback["dashboard_history_refresh_status"].astype(str).eq("failed").sum()
            )
        else:
            dashboard_refresh_failed_runs = 0
        if "alert_level" in lookback.columns:
            alert_levels = lookback["alert_level"].astype(str).str.lower()
            warning_or_critical_alert_runs = int(alert_levels.isin(["warning", "critical"]).sum())
        else:
            warning_or_critical_alert_runs = 0
        if "satellite_risk_budget_decision" in lookback.columns:
            risk_budget_trial_runs = int(
                lookback["satellite_risk_budget_decision"].astype(str).eq("eligible_for_small_satellite_trial").sum()
            )
        else:
            risk_budget_trial_runs = 0

        snapshot.update(
            {
                "health_state": health_state,
                "history_freshness_status": history_freshness,
                "latest_as_of_date": latest_as_of.isoformat(),
                "days_since_latest_history": int(days_since),
                "latest_generated_at": _scalar(latest, "generated_at"),
                "latest_dashboard_posture": latest_posture,
                "latest_action_posture": _scalar(latest, "action_posture"),
                "latest_data_freshness_status": _scalar(latest, "data_freshness_status"),
                "latest_market_cache_status": _scalar(latest, "market_cache_status"),
                "latest_allocator_input_status": _scalar(latest, "allocator_input_status"),
                "latest_paper_regime": _scalar(latest, "paper_latest_regime"),
                "latest_paper_candidate": _scalar(latest, "paper_latest_candidate"),
                "latest_core_weight": _as_float(_scalar(latest, "paper_latest_core_weight")),
                "latest_satellite_weight": _as_float(_scalar(latest, "paper_latest_satellite_weight")),
                "latest_cash_weight": _as_float(_scalar(latest, "paper_latest_cash_weight")),
                "latest_final_equity": latest_equity,
                "latest_total_return": latest_return,
                "latest_current_drawdown": latest_current_drawdown,
                "latest_max_drawdown": _as_float(_scalar(latest, "paper_max_drawdown")),
                "latest_sharpe": latest_sharpe,
                "latest_promotion_status": _scalar(latest, "promotion_status"),
                "latest_promotion_decision": latest_promotion_decision,
                "latest_promotion_candidate_passes_headline_gate": latest_promotion_headline_gate,
                "latest_promotion_group_support_count": latest_promotion_group_support,
                "latest_promotion_run_support_count": latest_promotion_run_support,
                "latest_promotion_min_support": latest_promotion_min_support,
                "latest_dashboard_history_refresh_status": latest_dashboard_history_refresh_status,
                "latest_dashboard_history_refresh_error": latest_dashboard_history_refresh_error,
                "latest_dashboard_history_refresh_report_path": latest_dashboard_history_refresh_report_path,
                "latest_dashboard_history_refresh_snapshot_path": latest_dashboard_history_refresh_snapshot_path,
                "latest_dashboard_pipeline_history_status": latest_dashboard_pipeline_history_status,
                "latest_dashboard_pipeline_history_health_state": latest_dashboard_pipeline_history_health_state,
                "latest_dashboard_pipeline_history_alert_count": latest_dashboard_pipeline_history_alert_count,
                "latest_dashboard_pipeline_history_latest_as_of_date": latest_dashboard_pipeline_history_latest_as_of_date,
                "latest_dashboard_alert_refresh_status": latest_dashboard_alert_refresh_status,
                "latest_dashboard_alert_refresh_error": latest_dashboard_alert_refresh_error,
                "latest_dashboard_pipeline_history_latest_satellite_risk_budget_decision": (
                    latest_dashboard_pipeline_history_risk_budget_decision
                ),
                "latest_dashboard_pipeline_history_latest_satellite_risk_budget_recommended_weight": (
                    latest_dashboard_pipeline_history_risk_budget_weight
                ),
                "latest_alert_level": latest_alert_level,
                "latest_alert_count": latest_alert_count,
                "latest_alert_action_stage": latest_alert_action_stage,
                "latest_alert_action_summary": latest_alert_action_summary,
                "latest_alert_action_item_count": latest_alert_action_item_count,
                "latest_alert_action_required": latest_alert_action_required,
                "latest_alert_critical_count": latest_alert_critical_count,
                "latest_alert_warning_count": latest_alert_warning_count,
                "latest_alert_info_count": latest_alert_info_count,
                "latest_alerts_report_path": latest_alerts_report_path,
                "latest_alerts_json_path": latest_alerts_json_path,
                "latest_alerts_latest_report_path": latest_alerts_latest_report_path,
                "latest_satellite_risk_budget_status": latest_risk_budget_status,
                "latest_satellite_risk_budget_decision": latest_risk_budget_decision,
                "latest_satellite_risk_budget_reason": latest_risk_budget_reason,
                "latest_satellite_risk_budget_recommended_satellite_weight": latest_risk_budget_recommended_weight,
                "latest_satellite_risk_budget_selected_horizon": latest_risk_budget_selected_horizon,
                "latest_satellite_risk_budget_report_path": latest_risk_budget_report_path,
                "latest_satellite_risk_budget_snapshot_path": latest_risk_budget_snapshot_path,
                "latest_satellite_risk_budget_checklist_path": latest_risk_budget_checklist_path,
                "previous_satellite_risk_budget_decision": previous_risk_budget_decision,
                "previous_satellite_risk_budget_recommended_satellite_weight": previous_risk_budget_recommended_weight,
                "satellite_risk_budget_decision_changed": risk_budget_decision_changed,
                "satellite_risk_budget_recommended_weight_change": risk_budget_recommended_weight_change,
                "hard_freshness_failures": hard_failures,
                "soft_freshness_failures": soft_failures,
                "lookback_actual_runs": int(len(lookback)),
                "lookback_final_equity_delta": final_equity_delta,
                "lookback_total_return_delta": total_return_delta,
                "lookback_current_drawdown_delta": current_drawdown_delta,
                "lookback_worst_current_drawdown": _as_float(current_drawdowns.min()),
                "lookback_min_sharpe": _as_float(sharpes.min()),
                "lookback_dashboard_posture_changes": _count_changes(lookback, "dashboard_posture"),
                "lookback_action_posture_changes": _count_changes(lookback, "action_posture"),
                "lookback_satellite_active_runs": int((satellite_weights.fillna(0.0) > 0.0).sum()),
                "lookback_dashboard_history_refresh_failed_runs": dashboard_refresh_failed_runs,
                "lookback_dashboard_history_refresh_status_changes": _count_changes(
                    lookback, "dashboard_history_refresh_status"
                ),
                "lookback_dashboard_alert_refresh_failed_runs": dashboard_alert_refresh_failed_runs,
                "lookback_dashboard_alert_refresh_status_changes": _count_changes(
                    lookback, "dashboard_alert_refresh_status"
                ),
                "lookback_alert_warning_or_critical_runs": warning_or_critical_alert_runs,
                "lookback_alert_level_changes": _count_changes(lookback, "alert_level"),
                "lookback_satellite_risk_budget_decision_changes": _count_changes(
                    lookback, "satellite_risk_budget_decision"
                ),
                "lookback_satellite_risk_budget_trial_eligible_runs": risk_budget_trial_runs,
                "lookback_trend_label": _trend_label(final_equity_delta, current_drawdown_delta),
                "alert_count": len(alerts),
                "alerts": alerts,
            }
        )

    resolved_output.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot), encoding="utf-8")
    return PipelineHistoryReviewResult(
        output_dir=resolved_output,
        snapshot_path=snapshot_path,
        report_path=report_path,
        history_path=resolved_history,
        snapshot=snapshot,
        history=clean,
    )
