"""Read-only pre-live readiness checks for research outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .trading_gate import build_a_share_trading_gate


DEFAULT_DASHBOARD_SNAPSHOT = Path("outputs/research/latest_dashboard/latest_dashboard_snapshot.json")
DEFAULT_PAPER_ACCOUNT_DIR = Path("outputs/research/paper_account_latest")
DEFAULT_LIVE_PREFLIGHT_OUTPUT = Path("outputs/research/live_preflight_latest")
STOCK_REVIEW_BLOCKING_ACTION_CODES = {
    "review_required_pending",
    "manual_status_normalization",
    "manual_exclusion_candidate",
}
STOCK_REVIEW_MONITOR_ACTION_CODES = {
    "manual_watch_followup",
    "manual_next_review_due",
}
LIVE_SHADOW_REVIEW_BLOCKING_DECISIONS = {
    "blocked_by_tracking_rule",
    "incomplete_review",
    "invalid_review_status",
    "needs_data",
}
LIVE_SHADOW_REVIEW_MONITOR_DECISIONS = {
    "watch_only",
    "skipped",
}
LIVE_SHADOW_REVIEW_OK_DECISIONS = {
    "manual_considered",
}
LIVE_SHADOW_REVIEW_KNOWN_DECISIONS = (
    LIVE_SHADOW_REVIEW_BLOCKING_DECISIONS | LIVE_SHADOW_REVIEW_MONITOR_DECISIONS | LIVE_SHADOW_REVIEW_OK_DECISIONS
)


@dataclass(frozen=True)
class LivePreflightResult:
    output_dir: Path
    checklist_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _as_int(value: Any, default: int = 0) -> int:
    return int(_as_float(value, default))


def _pct(value: Any) -> str:
    return f"{_as_float(value) * 100:.2f}%"


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = Path(path) if path is not None else default
    return raw if raw.is_absolute() else project_root / raw


def _recorded_path(project_root: Path, base_dir: Path, raw: Any, default_name: str) -> Path:
    if raw not in (None, ""):
        candidate = Path(str(raw))
        if candidate.is_absolute():
            return candidate
        project_candidate = project_root / candidate
        if project_candidate.exists():
            return project_candidate
        return base_dir / candidate
    return base_dir / default_name


def _read_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return 0
    return int(len(frame))


def _stock_review_action_counts_from_file(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return {"total": 0, "blocking": 0, "monitor": 0, "unknown": 0}
    if frame.empty:
        return {"total": 0, "blocking": 0, "monitor": 0, "unknown": 0}
    if "action_code" not in frame.columns:
        total = int(len(frame))
        return {"total": total, "blocking": total, "monitor": 0, "unknown": total}
    codes = frame["action_code"].fillna("").astype(str).str.strip()
    total = int(len(codes))
    blocking = int(codes.isin(STOCK_REVIEW_BLOCKING_ACTION_CODES).sum())
    monitor = int(codes.isin(STOCK_REVIEW_MONITOR_ACTION_CODES).sum())
    unknown = int((~codes.isin(STOCK_REVIEW_BLOCKING_ACTION_CODES | STOCK_REVIEW_MONITOR_ACTION_CODES)).sum())
    return {"total": total, "blocking": blocking + unknown, "monitor": monitor, "unknown": unknown}


def _stock_review_action_counts_from_metrics(metrics: dict[str, Any], source_snapshot: dict[str, Any]) -> dict[str, int]:
    total = _as_int(metrics.get("stock_target_review_action_count", source_snapshot.get("paper_stock_target_review_action_count")), 0)
    blocking = sum(
        _as_int(metrics.get(key, source_snapshot.get(f"paper_{key}")), 0)
        for key in [
            "stock_target_review_action_pending_model_count",
            "stock_target_review_action_manual_exclusion_count",
            "stock_target_review_action_status_normalization_count",
        ]
    )
    monitor = sum(
        _as_int(metrics.get(key, source_snapshot.get(f"paper_{key}")), 0)
        for key in [
            "stock_target_review_action_manual_watch_count",
            "stock_target_review_action_next_review_due_count",
        ]
    )
    unknown = max(total - blocking - monitor, 0)
    return {"total": total, "blocking": blocking + unknown, "monitor": monitor, "unknown": unknown}


def _live_shadow_review_decision_counts_from_file(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, dtype={"code": str})
    except (OSError, pd.errors.EmptyDataError):
        return {
            "total": 0,
            "blocking": 0,
            "monitor": 0,
            "unknown": 0,
            "tracking_blocked": 0,
            "incomplete": 0,
            "invalid": 0,
            "needs_data": 0,
            "manual_considered": 0,
            "watch_only": 0,
            "skipped": 0,
        }
    if frame.empty:
        return {
            "total": 0,
            "blocking": 0,
            "monitor": 0,
            "unknown": 0,
            "tracking_blocked": 0,
            "incomplete": 0,
            "invalid": 0,
            "needs_data": 0,
            "manual_considered": 0,
            "watch_only": 0,
            "skipped": 0,
        }
    if "review_decision" not in frame.columns:
        total = int(len(frame))
        return {
            "total": total,
            "blocking": total,
            "monitor": 0,
            "unknown": total,
            "tracking_blocked": 0,
            "incomplete": 0,
            "invalid": 0,
            "needs_data": 0,
            "manual_considered": 0,
            "watch_only": 0,
            "skipped": 0,
        }
    decisions = frame["review_decision"].fillna("").astype(str).str.strip()
    total = int(len(decisions))
    unknown = int((~decisions.isin(LIVE_SHADOW_REVIEW_KNOWN_DECISIONS)).sum())
    blocking_known = int(decisions.isin(LIVE_SHADOW_REVIEW_BLOCKING_DECISIONS).sum())
    monitor = int(decisions.isin(LIVE_SHADOW_REVIEW_MONITOR_DECISIONS).sum())
    return {
        "total": total,
        "blocking": blocking_known + unknown,
        "monitor": monitor,
        "unknown": unknown,
        "tracking_blocked": int((decisions == "blocked_by_tracking_rule").sum()),
        "incomplete": int((decisions == "incomplete_review").sum()),
        "invalid": int((decisions == "invalid_review_status").sum()),
        "needs_data": int((decisions == "needs_data").sum()),
        "manual_considered": int((decisions == "manual_considered").sum()),
        "watch_only": int((decisions == "watch_only").sum()),
        "skipped": int((decisions == "skipped").sum()),
    }


def _active_target_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return 0
    if frame.empty:
        return 0
    weight_column = "target_weight" if "target_weight" in frame.columns else "portfolio_target_weight"
    if weight_column not in frame.columns:
        return int(len(frame))
    weights = pd.to_numeric(frame[weight_column], errors="coerce").fillna(0.0)
    return int((weights.abs() > 1e-12).sum())


def _ensure_trading_gate_fields(source_snapshot: dict[str, Any]) -> None:
    if (
        source_snapshot.get("trading_day_gate_status")
        and source_snapshot.get("after_close_data_status")
    ):
        return
    as_of = source_snapshot.get("as_of_date")
    generated_at = source_snapshot.get("generated_at")
    gate_snapshot = build_a_share_trading_gate(source_snapshot, as_of=as_of, generated_at=generated_at)
    if not source_snapshot.get("trading_day_gate_status"):
        source_snapshot["trading_day_gate_status"] = gate_snapshot.get("trading_day_gate_status")
    if not source_snapshot.get("after_close_data_status"):
        source_snapshot["after_close_data_status"] = gate_snapshot.get("after_close_data_status")
    if not source_snapshot.get("trading_day_gate_action"):
        source_snapshot["trading_day_gate_action"] = gate_snapshot.get("trading_day_gate_action")


def _check_row(
    rows: list[dict[str, Any]],
    blockers: list[str],
    section: str,
    item: str,
    ok: bool,
    value: Any,
    required: Any,
    next_action: str,
    blocker: str | None = None,
) -> None:
    rows.append(
        {
            "section": section,
            "item": item,
            "status": "ok" if ok else "blocked",
            "value": value,
            "required": required,
            "next_action": "No action needed." if ok else next_action,
        }
    )
    if not ok:
        blockers.append(blocker or f"{item}={value}")


def _monitor_row(
    rows: list[dict[str, Any]],
    monitors: list[str],
    section: str,
    item: str,
    value: Any,
    next_action: str,
    monitor: str | None = None,
) -> None:
    rows.append(
        {
            "section": section,
            "item": item,
            "status": "monitor",
            "value": value,
            "required": "manual_review",
            "next_action": next_action,
        }
    )
    monitors.append(monitor or f"{item}={value}")


def _decision(blockers: list[str], monitors: list[str], latest_regime: str, satellite_weight: float) -> str:
    if blockers:
        return "blocked"
    if latest_regime != "risk_on" or satellite_weight <= 0:
        return "ready_for_watch_only_pre_stage"
    if monitors:
        return "ready_for_manual_review_pre_stage"
    return "ready_for_manual_broker_reconciliation"


def _render_report(snapshot: dict[str, Any], checklist: pd.DataFrame) -> str:
    rows = [
        f"| {row.section} | {row.item} | `{row.status}` | {row.value} | {row.required} | {row.next_action} |"
        for row in checklist.itertuples(index=False)
    ]
    blockers = snapshot.get("blocking_items") or []
    monitors = snapshot.get("monitor_items") or []
    blocker_text = "\n".join(f"- `{item}`" for item in blockers) if blockers else "- None"
    monitor_text = "\n".join(f"- `{item}`" for item in monitors) if monitors else "- None"
    return f"""# Live Preflight Readiness

Generated at: `{snapshot.get("generated_at")}`

This is a read-only pre-live readiness report. It does not connect to brokers, place orders, cancel orders, or create executable order instructions.

## Summary

| Item | Value |
| --- | ---: |
| Decision | `{snapshot.get("decision")}` |
| Broker connection status | `{snapshot.get("broker_connection_status")}` |
| Dashboard posture | `{snapshot.get("dashboard_posture")}` |
| Paper latest date | `{snapshot.get("paper_latest_date") or "N/A"}` |
| Latest regime | `{snapshot.get("paper_latest_regime") or "N/A"}` |
| Core / satellite / cash | {_pct(snapshot.get("paper_latest_core_weight"))} / {_pct(snapshot.get("paper_latest_satellite_weight"))} / {_pct(snapshot.get("paper_latest_cash_weight"))} |
| Active target holdings | `{snapshot.get("active_target_count")}` |
| Review action count | `{snapshot.get("stock_target_review_action_count")}` |
| Blocking review actions | `{snapshot.get("stock_target_review_blocking_action_count")}` |
| Monitor review actions | `{snapshot.get("stock_target_review_monitor_action_count")}` |
| Live-shadow review decisions | `{snapshot.get("live_shadow_review_decision_count") if snapshot.get("live_shadow_review_decision_count") is not None else "N/A"}` |
| Live-shadow blocking decisions | `{snapshot.get("live_shadow_review_blocking_decision_count") if snapshot.get("live_shadow_review_blocking_decision_count") is not None else "N/A"}` |
| Live-shadow monitor decisions | `{snapshot.get("live_shadow_review_monitor_decision_count") if snapshot.get("live_shadow_review_monitor_decision_count") is not None else "N/A"}` |
| Blocking items | `{len(blockers)}` |
| Monitor items | `{len(monitors)}` |

## Checklist

| Section | Item | Status | Value | Required | Next action |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## Blocking Items

{blocker_text}

## Monitor Items

{monitor_text}

## Operating Boundary

- This stage is allowed to prepare an evidence checklist for human review.
- This stage is allowed to compare paper targets with broker exports supplied by the user in a later step.
- This stage is not allowed to log in to a broker, submit orders, cancel orders, or auto-change live positions.

## Source Files

- Dashboard snapshot: `{snapshot.get("dashboard_snapshot_path")}`
- Pipeline snapshot: `{snapshot.get("pipeline_snapshot_path") or "N/A"}`
- Paper metrics: `{snapshot.get("paper_metrics_path")}`
- Target holdings: `{snapshot.get("target_holdings_path")}`
- Review actions: `{snapshot.get("stock_target_review_actions_path")}`
- Live-shadow review decisions: `{snapshot.get("live_shadow_review_decisions_path") or "N/A"}`
- Daily run status: `{snapshot.get("daily_run_status_snapshot_path") or "N/A"}`
"""


def run_live_preflight(
    dashboard_snapshot: str | Path = DEFAULT_DASHBOARD_SNAPSHOT,
    paper_account_dir: str | Path | None = None,
    pipeline_snapshot: str | Path | None = None,
    daily_run_status_snapshot: str | Path | None = None,
    live_shadow_review_decisions_file: str | Path | None = None,
    output_dir: str | Path = DEFAULT_LIVE_PREFLIGHT_OUTPUT,
) -> LivePreflightResult:
    project_root = Path.cwd()
    dashboard_path = _resolve(project_root, dashboard_snapshot, DEFAULT_DASHBOARD_SNAPSHOT)
    dashboard = _load_json(dashboard_path)
    raw_pipeline_path = pipeline_snapshot or dashboard.get("pipeline_snapshot_path")
    pipeline_path = _resolve(project_root, raw_pipeline_path, Path(str(raw_pipeline_path))) if raw_pipeline_path else None
    pipeline = _load_json(pipeline_path) if pipeline_path else {}
    source_snapshot = {**dashboard, **pipeline}
    _ensure_trading_gate_fields(source_snapshot)
    raw_paper_dir = (
        paper_account_dir
        or source_snapshot.get("paper_account_output_dir")
        or source_snapshot.get("paper_account_dir")
        or DEFAULT_PAPER_ACCOUNT_DIR
    )
    paper_dir = _resolve(project_root, raw_paper_dir, DEFAULT_PAPER_ACCOUNT_DIR)
    raw_metrics_path = pipeline.get("paper_account_metrics_path")
    if raw_metrics_path in (None, "") and pipeline.get("paper_account_output_dir"):
        raw_metrics_path = None
    elif raw_metrics_path in (None, ""):
        raw_metrics_path = source_snapshot.get("paper_account_metrics_path")
    paper_metrics_path = _recorded_path(
        project_root,
        paper_dir,
        raw_metrics_path,
        "metrics.json",
    )
    metrics = _load_json(paper_metrics_path)
    target_holdings_path = _recorded_path(project_root, paper_dir, metrics.get("target_holdings_path"), "target_holdings.csv")
    actions_path = _recorded_path(
        project_root,
        paper_dir,
        metrics.get("stock_target_review_actions_path"),
        "stock_target_review_actions.csv",
    )
    status_ref = daily_run_status_snapshot or dashboard.get("daily_run_status_snapshot_path")
    status_path = (
        _resolve(project_root, status_ref, Path("outputs/research/daily_run_status_latest/daily_run_status_snapshot.json"))
        if status_ref
        else None
    )
    status_payload = _load_json(status_path) if status_path is not None else {}
    if status_path is not None and not status_payload:
        fallback_status_path = project_root / "outputs" / "research" / "daily_run_status_latest" / "daily_run_status_snapshot.json"
        fallback_payload = _load_json(fallback_status_path)
        if fallback_payload:
            status_path = fallback_status_path
            status_payload = fallback_payload
    raw_live_shadow_review_decisions = (
        live_shadow_review_decisions_file
        or source_snapshot.get("live_shadow_review_decisions_path")
        or source_snapshot.get("live_shadow_review_decisions_file")
        or source_snapshot.get("live_preflight_live_shadow_review_decisions_path")
    )
    live_shadow_review_decisions_path = (
        _resolve(project_root, raw_live_shadow_review_decisions, Path(str(raw_live_shadow_review_decisions)))
        if raw_live_shadow_review_decisions
        else None
    )

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    monitors: list[str] = []
    _check_row(rows, blockers, "source", "dashboard_snapshot", bool(dashboard), str(dashboard_path), "exists", "Run dashboard before live preflight.")
    _check_row(rows, blockers, "source", "paper_metrics", bool(metrics), str(paper_metrics_path), "exists", "Run paper-account before live preflight.")

    required_statuses = {
        "dashboard_posture": "core_base_watch_allocator_gate",
        "data_freshness_status": "fresh_enough",
        "market_cache_status": "fresh_enough",
        "allocator_input_status": "fresh_enough",
        "paper_account_status": "ok",
        "paper_freshness_status": "fresh_enough",
        "sentiment_freshness_status": "fresh_enough",
        "trigger_freshness_status": "fresh_enough",
        "trading_day_gate_status": "trading_day_data_ready",
        "after_close_data_status": "ready",
    }
    for key, expected in required_statuses.items():
        value = source_snapshot.get(key)
        _check_row(
            rows,
            blockers,
            "dashboard",
            key,
            value == expected,
            value,
            expected,
            f"Refresh the daily research pipeline until {key} is {expected}.",
            f"{key}={value or 'missing'}",
        )

    status_run_state = str(status_payload.get("run_state") or "")
    problem_state = bool(status_payload.get("problem_state") or dashboard.get("daily_run_problem_state"))
    if status_run_state == "live_preflight_blocked":
        problem_state = False
    if status_payload or "daily_run_problem_state" in dashboard:
        _check_row(
            rows,
            blockers,
            "daily_run",
            "problem_state",
            not problem_state,
            problem_state,
            False,
            "Resolve daily-run-status problem state before entering pre-live review.",
            "daily_run_problem_state=true",
        )

    active_count = _active_target_count(target_holdings_path)
    _check_row(
        rows,
        blockers,
        "paper_account",
        "target_holdings",
        active_count is not None,
        str(target_holdings_path),
        "exists",
        "Regenerate paper-account target holdings.",
    )
    if active_count is not None:
        _monitor_row(rows, monitors, "paper_account", "active_target_count", active_count, "Use only for human comparison, not auto-order sizing.")

    action_counts = _stock_review_action_counts_from_file(actions_path)
    if action_counts is None:
        action_counts = _stock_review_action_counts_from_metrics(metrics, source_snapshot)
    action_count = int(action_counts["total"])
    blocking_action_count = int(action_counts["blocking"])
    monitor_action_count = int(action_counts["monitor"])
    unknown_action_count = int(action_counts["unknown"])
    _check_row(
        rows,
        blockers,
        "paper_account",
        "stock_target_review_blocking_action_count",
        blocking_action_count == 0,
        blocking_action_count,
        0,
        "Resolve blocking stock-target review actions before pre-live reconciliation.",
        f"stock_target_review_blocking_action_count={blocking_action_count}",
    )
    if monitor_action_count > 0:
        _monitor_row(
            rows,
            monitors,
            "paper_account",
            "stock_target_review_monitor_action_count",
            monitor_action_count,
            "Keep manually watched stock-target rows in routine follow-up.",
            f"stock_target_review_monitor_action_count={monitor_action_count}",
        )

    live_shadow_review_counts = None
    if live_shadow_review_decisions_path is not None:
        live_shadow_review_counts = _live_shadow_review_decision_counts_from_file(live_shadow_review_decisions_path)
        _check_row(
            rows,
            blockers,
            "live_shadow_review",
            "live_shadow_review_decisions_file",
            live_shadow_review_counts is not None,
            str(live_shadow_review_decisions_path),
            "exists",
            "Run live-shadow-review-apply or provide the generated live_shadow_review_decisions.csv before pre-live reconciliation.",
            "live_shadow_review_decisions_file=missing",
        )
        if live_shadow_review_counts is not None:
            blocking_review_decision_count = int(live_shadow_review_counts["blocking"])
            monitor_review_decision_count = int(live_shadow_review_counts["monitor"])
            _check_row(
                rows,
                blockers,
                "live_shadow_review",
                "live_shadow_review_blocking_decision_count",
                blocking_review_decision_count == 0,
                blocking_review_decision_count,
                0,
                "Resolve blocked, incomplete, invalid, needs-data, or unknown live-shadow review decisions first.",
                f"live_shadow_review_blocking_decision_count={blocking_review_decision_count}",
            )
            if monitor_review_decision_count > 0:
                _monitor_row(
                    rows,
                    monitors,
                    "live_shadow_review",
                    "live_shadow_review_monitor_decision_count",
                    monitor_review_decision_count,
                    "Keep watch-only or skipped live-shadow rows out of execution planning.",
                    f"live_shadow_review_monitor_decision_count={monitor_review_decision_count}",
                )

    latest_regime = str(metrics.get("latest_regime") or source_snapshot.get("paper_latest_regime") or "")
    satellite_weight = _as_float(metrics.get("latest_satellite_weight", source_snapshot.get("paper_latest_satellite_weight")))
    if latest_regime != "risk_on" or satellite_weight <= 0:
        _monitor_row(
            rows,
            monitors,
            "risk",
            "risk_exposure_state",
            f"regime={latest_regime or 'missing'}, satellite={_pct(satellite_weight)}",
            "Pre-live stage should stay watch-only until risk-on exposure appears.",
            "watch_only_risk_state",
        )

    decision = _decision(blockers, monitors, latest_regime, satellite_weight)
    resolved_output = _resolve(project_root, output_dir, DEFAULT_LIVE_PREFLIGHT_OUTPUT)
    resolved_output.mkdir(parents=True, exist_ok=True)
    checklist = pd.DataFrame(rows)
    checklist_path = resolved_output / "live_preflight_checklist.csv"
    snapshot_path = resolved_output / "live_preflight_snapshot.json"
    report_path = resolved_output / "live_preflight.md"
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "live_preflight_completed",
        "decision": decision,
        "broker_connection_status": "not_connected",
        "dashboard_snapshot_path": str(dashboard_path),
        "pipeline_snapshot_path": str(pipeline_path) if pipeline_path else None,
        "pipeline_snapshot_status": "ok" if pipeline else ("not_configured" if pipeline_path is None else "missing"),
        "paper_account_dir": str(paper_dir),
        "paper_metrics_path": str(paper_metrics_path),
        "target_holdings_path": str(target_holdings_path),
        "stock_target_review_actions_path": str(actions_path),
        "live_shadow_review_decisions_path": str(live_shadow_review_decisions_path)
        if live_shadow_review_decisions_path is not None
        else None,
        "daily_run_status_snapshot_path": str(status_path) if status_payload and status_path is not None else None,
        "blocking_items": blockers,
        "monitor_items": monitors,
        "active_target_count": active_count,
        "stock_target_review_action_count": action_count,
        "stock_target_review_blocking_action_count": blocking_action_count,
        "stock_target_review_monitor_action_count": monitor_action_count,
        "stock_target_review_unknown_action_count": unknown_action_count,
        "live_shadow_review_decision_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["total"]),
        "live_shadow_review_blocking_decision_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["blocking"]),
        "live_shadow_review_monitor_decision_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["monitor"]),
        "live_shadow_review_unknown_decision_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["unknown"]),
        "live_shadow_review_tracking_blocked_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["tracking_blocked"]),
        "live_shadow_review_incomplete_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["incomplete"]),
        "live_shadow_review_invalid_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["invalid"]),
        "live_shadow_review_needs_data_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["needs_data"]),
        "live_shadow_review_manual_considered_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["manual_considered"]),
        "live_shadow_review_watch_only_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["watch_only"]),
        "live_shadow_review_skipped_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["skipped"]),
        "research_only": True,
        **source_snapshot,
        "paper_account_dir": str(paper_dir),
        "paper_metrics_path": str(paper_metrics_path),
        "target_holdings_path": str(target_holdings_path),
        "stock_target_review_actions_path": str(actions_path),
        "live_shadow_review_decisions_path": str(live_shadow_review_decisions_path)
        if live_shadow_review_decisions_path is not None
        else None,
        "live_shadow_review_decision_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["total"]),
        "live_shadow_review_blocking_decision_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["blocking"]),
        "live_shadow_review_monitor_decision_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["monitor"]),
        "live_shadow_review_unknown_decision_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["unknown"]),
        "live_shadow_review_tracking_blocked_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["tracking_blocked"]),
        "live_shadow_review_incomplete_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["incomplete"]),
        "live_shadow_review_invalid_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["invalid"]),
        "live_shadow_review_needs_data_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["needs_data"]),
        "live_shadow_review_manual_considered_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["manual_considered"]),
        "live_shadow_review_watch_only_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["watch_only"]),
        "live_shadow_review_skipped_count": None
        if live_shadow_review_counts is None
        else int(live_shadow_review_counts["skipped"]),
        "paper_latest_regime": latest_regime or source_snapshot.get("paper_latest_regime"),
        "paper_latest_core_weight": metrics.get("latest_core_weight", source_snapshot.get("paper_latest_core_weight")),
        "paper_latest_satellite_weight": satellite_weight,
        "paper_latest_cash_weight": metrics.get("latest_cash_weight", source_snapshot.get("paper_latest_cash_weight")),
        "paper_latest_date": metrics.get("latest_date", source_snapshot.get("paper_latest_date")),
    }
    checklist.to_csv(checklist_path, index=False, encoding="utf-8-sig")
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, checklist), encoding="utf-8")
    return LivePreflightResult(
        output_dir=resolved_output,
        checklist_path=checklist_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
    )
