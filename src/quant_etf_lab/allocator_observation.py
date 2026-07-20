"""Post-switch allocator observation reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


from ._compat import read_text


@dataclass(frozen=True)
class AllocatorObservationResult:
    output_dir: Path
    report_path: Path
    checklist_path: Path
    snapshot_path: Path
    snapshot: dict[str, Any]


ALERT_RANK = {"normal": 0, "info": 1, "warning": 2, "critical": 3}


def _load_snapshot(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing daily-pipeline snapshot: {resolved}")
    return json.loads(read_text(resolved))


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    return int(number)


def _pct(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "N/A"
    return f"{number * 100:.2f}%"


def _num(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "N/A"
    return f"{number:.3f}"


def _text(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    return str(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if value is pd.NA:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _blocking_items(snapshot: dict[str, Any]) -> list[str]:
    items: list[str] = []
    if str(snapshot.get("promotion_decision") or "") != "promote_candidate":
        items.append(f"promotion_decision={snapshot.get('promotion_decision') or 'missing'}")
    if str(snapshot.get("dashboard_posture") or "") != "core_base_watch_allocator_gate":
        items.append(f"dashboard_posture={snapshot.get('dashboard_posture') or 'missing'}")
    if str(snapshot.get("trading_day_gate_status") or "") != "trading_day_data_ready":
        items.append(f"trading_day_gate={snapshot.get('trading_day_gate_status') or 'missing'}")
    if str(snapshot.get("after_close_data_status") or "") != "ready":
        items.append(f"after_close_data={snapshot.get('after_close_data_status') or 'missing'}")
    if ALERT_RANK.get(str(snapshot.get("alert_level") or "normal"), 0) >= ALERT_RANK["warning"]:
        items.append(f"alert_level={snapshot.get('alert_level')}")
    if str(snapshot.get("history_review_health_state") or "ok") not in {"ok", "missing"}:
        items.append(f"history_health={snapshot.get('history_review_health_state')}")
    if _as_int(snapshot.get("model_audit_walk_forward_action_items")) not in {None, 0}:
        items.append(f"model_audit_actions={snapshot.get('model_audit_walk_forward_action_items')}")
    if _as_int(snapshot.get("paper_stock_target_review_action_count")) not in {None, 0}:
        items.append(f"stock_target_actions={snapshot.get('paper_stock_target_review_action_count')}")
    if _as_int(snapshot.get("paper_stock_target_review_required_unreviewed_count")) not in {None, 0}:
        items.append(f"unreviewed_required_targets={snapshot.get('paper_stock_target_review_required_unreviewed_count')}")
    if bool(snapshot.get("pipeline_user_intervention_required")):
        items.append("pipeline_user_intervention_required=true")
    return items


def _monitor_items(snapshot: dict[str, Any]) -> list[str]:
    items: list[str] = []
    drawdown_count = _as_int(snapshot.get("paper_stock_target_review_drawdown_count"))
    if drawdown_count:
        items.append(f"reviewed_drawdown_targets={drawdown_count}")
    suppressed_count = _as_int(snapshot.get("paper_stock_target_review_suppressed_layer_count"))
    if suppressed_count:
        items.append(f"suppressed_layer_targets={suppressed_count}")
    manual_pending = _as_int(snapshot.get("paper_stock_target_review_manual_pending_count"))
    if manual_pending:
        items.append(f"manual_pending_rows={manual_pending}")
    outcome_status = str(snapshot.get("paper_stock_target_review_outcome_analysis_status") or "")
    if outcome_status in {"waiting_for_evaluable_returns", "sample_insufficient"}:
        items.append("outcome_samples_waiting")
    return items


def _status_context(snapshot: dict[str, Any], blockers: list[str]) -> dict[str, Any]:
    outcome_status = str(snapshot.get("paper_stock_target_review_outcome_analysis_status") or "")
    ready_horizons = _as_int(snapshot.get("paper_stock_target_review_outcome_analysis_ready_horizon_count")) or 0
    due_count = _as_int(snapshot.get("paper_stock_target_review_outcome_due_row_count")) or 0
    next_date = snapshot.get("paper_stock_target_review_outcome_calendar_next_action_date")
    next_horizon = snapshot.get("paper_stock_target_review_outcome_calendar_next_action_horizon")
    latest_regime = str(snapshot.get("paper_latest_regime") or "")
    latest_satellite_weight = _as_float(snapshot.get("paper_latest_satellite_weight")) or 0.0

    if blockers:
        status = "review_blockers"
        stage = "resolve_blockers"
        reason = "One or more research gates need attention before continuing the post-switch observation."
        risk_budget = "hold_previous_risk_budget"
    elif due_count > 0 or ready_horizons > 0:
        status = "outcomes_ready_for_review"
        stage = "review_stock_target_outcomes"
        reason = "At least one stock-target outcome horizon is ready or due for review."
        risk_budget = "review_before_budget_change"
    elif outcome_status in {"waiting_for_evaluable_returns", "sample_insufficient"}:
        status = "waiting_for_outcome_samples"
        stage = "accumulate_outcome_samples"
        reason = "Stock-target outcomes have not passed the configured sample-readiness gates."
        risk_budget = "hold_default_core_base"
    elif latest_regime != "risk_on" or latest_satellite_weight <= 0:
        status = "routine_monitor"
        stage = "wait_for_risk_on_exposure"
        reason = "Outcome gates are clear, but the latest paper account is not carrying active satellite exposure."
        risk_budget = "hold_default_core_base"
    else:
        status = "eligible_for_budget_review"
        stage = "review_satellite_budget"
        reason = "Outcome gates are clear and satellite exposure is active."
        risk_budget = "eligible_for_satellite_budget_review"

    return {
        "observation_status": status,
        "next_action_stage": stage,
        "next_action_reason": reason,
        "risk_budget_decision": risk_budget,
        "next_review_date": next_date,
        "next_review_horizon": next_horizon,
        "outcome_analysis_status": outcome_status,
        "outcome_ready_horizon_count": ready_horizons,
        "outcome_due_count": due_count,
    }


def _checklist_rows(snapshot: dict[str, Any], observation: dict[str, Any], blockers: list[str], monitors: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "section": "governance",
            "item": "promotion_decision",
            "value": snapshot.get("promotion_decision"),
            "status": "ok" if snapshot.get("promotion_decision") == "promote_candidate" else "review",
            "next_action": "Keep promotion evidence visible in daily pipeline reports.",
        },
        {
            "section": "governance",
            "item": "support_groups",
            "value": snapshot.get("promotion_support_group_count"),
            "status": "ok",
            "next_action": "Retain execution-cost and conservative-guard evidence as the promotion basis.",
        },
        {
            "section": "pipeline_gates",
            "item": "blocking_items",
            "value": len(blockers),
            "status": "ok" if not blockers else "review",
            "next_action": "; ".join(blockers) if blockers else "No blocking gate is open.",
        },
        {
            "section": "pipeline_gates",
            "item": "monitor_items",
            "value": len(monitors),
            "status": "monitor" if monitors else "ok",
            "next_action": "; ".join(monitors) if monitors else "No monitor-only item is open.",
        },
        {
            "section": "outcome_maturity",
            "item": "analysis_status",
            "value": observation.get("outcome_analysis_status"),
            "status": observation.get("observation_status"),
            "next_action": observation.get("next_action_reason"),
        },
        {
            "section": "outcome_maturity",
            "item": "next_review",
            "value": f"{observation.get('next_review_date') or 'N/A'} / {observation.get('next_review_horizon') or 'N/A'}",
            "status": observation.get("next_action_stage"),
            "next_action": "Rerun daily-pipeline after this maturity date when local OHLCV is refreshed.",
        },
        {
            "section": "risk_budget",
            "item": "latest_weights",
            "value": f"core={_pct(snapshot.get('paper_latest_core_weight'))}, satellite={_pct(snapshot.get('paper_latest_satellite_weight'))}",
            "status": observation.get("risk_budget_decision"),
            "next_action": "Do not increase satellite budget before outcome review gates are ready.",
        },
        {
            "section": "rollback",
            "item": "baseline_retained",
            "value": True,
            "status": "ok",
            "next_action": "Keep the previous allocator path available for baseline and rollback comparison.",
        },
    ]


def _render_report(snapshot: dict[str, Any], checklist: pd.DataFrame) -> str:
    blockers = snapshot.get("blocking_items") or []
    monitors = snapshot.get("monitor_items") or []
    blocker_text = "\n".join(f"- `{item}`" for item in blockers) if blockers else "- None"
    monitor_text = "\n".join(f"- `{item}`" for item in monitors) if monitors else "- None"
    rows = [
        f"| {row.section} | {row.item} | `{row.status}` | {row.value} | {row.next_action} |"
        for row in checklist.itertuples(index=False)
    ]
    return f"""# Post-Switch Allocator Observation

Generated at: `{snapshot.get("generated_at")}`

This report is research and review only. It does not connect to brokers, place orders, change live positions, or provide investment advice.

## Summary

| Item | Value |
| --- | ---: |
| Observation status | `{snapshot.get("observation_status")}` |
| Next action stage | `{snapshot.get("next_action_stage")}` |
| Next review | {snapshot.get("next_review_date") or "N/A"} / `{snapshot.get("next_review_horizon") or "N/A"}` |
| Risk budget decision | `{snapshot.get("risk_budget_decision")}` |
| As-of date | {snapshot.get("as_of_date", "N/A")} |
| Latest candidate | `{snapshot.get("paper_latest_candidate", "N/A")}` |
| Latest regime | `{snapshot.get("paper_latest_regime", "N/A")}` |
| Core / satellite weight | {_pct(snapshot.get("paper_latest_core_weight"))} / {_pct(snapshot.get("paper_latest_satellite_weight"))} |
| Paper total return | {_pct(snapshot.get("paper_total_return"))} |
| Paper max drawdown | {_pct(snapshot.get("paper_max_drawdown"))} |
| Paper Sharpe | {_num(snapshot.get("paper_sharpe"))} |
| Promotion decision | `{snapshot.get("promotion_decision", "N/A")}` |
| Alert level | `{snapshot.get("alert_level", "N/A")}` |
| History health | `{snapshot.get("history_review_health_state", "N/A")}` |
| Model audit | `{snapshot.get("model_audit_status", "N/A")}` |

## Checklist

| Section | Item | Status | Value | Next action |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## Blocking Items

{blocker_text}

## Monitor Items

{monitor_text}

## Outcome Dates

- 1D: {snapshot.get("paper_stock_target_review_outcome_maturity_next_1d_date") or "N/A"}
- 5D: {snapshot.get("paper_stock_target_review_outcome_maturity_next_5d_date") or "N/A"}
- 10D: {snapshot.get("paper_stock_target_review_outcome_maturity_next_10d_date") or "N/A"}
- 20D: {snapshot.get("paper_stock_target_review_outcome_maturity_next_20d_date") or "N/A"}

## Source Files

- Pipeline snapshot: `{snapshot.get("pipeline_snapshot")}`
- Pipeline report: `{snapshot.get("pipeline_report_path")}`
- Alerts: `{snapshot.get("alerts_report_path")}`
- Outcome analysis: `{snapshot.get("paper_stock_target_review_outcome_analysis_report_path")}`
- Outcome calendar: `{snapshot.get("paper_stock_target_review_outcome_calendar_report_path")}`
- Outcome due queue: `{snapshot.get("paper_stock_target_review_outcome_due_report_path")}`
- Rollback baseline: `{snapshot.get("baseline_label")}`
"""


def run_allocator_observation(
    pipeline_snapshot: str | Path,
    output_dir: str | Path = "outputs/research/allocator_observation_latest",
    baseline_label: str = "Rollback quality-v2",
) -> AllocatorObservationResult:
    pipeline_path = Path(pipeline_snapshot)
    source = _load_snapshot(pipeline_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blockers = _blocking_items(source)
    monitors = _monitor_items(source)
    observation = _status_context(source, blockers)
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "allocator_observation_completed",
        "pipeline_snapshot": str(pipeline_path),
        "baseline_label": baseline_label,
        "rollback_baseline_retained": True,
        "blocking_items": blockers,
        "monitor_items": monitors,
        **source,
        **observation,
    }
    checklist = pd.DataFrame(_checklist_rows(snapshot, observation, blockers, monitors))
    checklist_path = out_dir / "allocator_observation_checklist.csv"
    snapshot_path = out_dir / "allocator_observation_snapshot.json"
    report_path = out_dir / "allocator_observation.md"

    checklist.to_csv(checklist_path, index=False, encoding="utf-8-sig")
    snapshot_path.write_text(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, checklist), encoding="utf-8")

    return AllocatorObservationResult(
        output_dir=out_dir,
        report_path=report_path,
        checklist_path=checklist_path,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )
