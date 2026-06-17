"""Research-only satellite risk-budget review."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SatelliteRiskBudgetReviewResult:
    output_dir: Path
    report_path: Path
    checklist_path: Path
    snapshot_path: Path
    snapshot: dict[str, Any]


ALERT_RANK = {"normal": 0, "info": 1, "warning": 2, "critical": 3}
NETWORK_SIGNAL_MONITOR = "network_signal_monitor"
NETWORK_SIGNAL_HOLD = "network_signal_hold"


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


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


def _horizon_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return int(digits), text
    return 9999, text


def _snapshot_outcome_analysis_path(snapshot: dict[str, Any]) -> Path | None:
    raw_path = snapshot.get("paper_stock_target_review_outcome_analysis_json_path")
    if raw_path in (None, ""):
        return None
    resolved = Path(str(raw_path))
    return resolved if resolved.exists() else None


def _load_network_snapshot(path: str | Path | None) -> dict[str, Any]:
    if path in (None, ""):
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {"network_snapshot_available": False, "network_snapshot_path": str(resolved), "network_error": "missing_file"}
    try:
        return _load_json(resolved, "network-lab snapshot")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return {
            "network_snapshot_available": False,
            "network_snapshot_path": str(resolved),
            "network_error": str(error),
        }


def _network_signal_context(
    network_lab_snapshot: str | Path | None,
    max_cluster_count_warning: int,
    residual_mi_warning: float,
) -> dict[str, Any]:
    payload = _load_network_snapshot(network_lab_snapshot)
    if not payload:
        return {
            "network_signal_level": "missing",
            "network_signal_monitors": [],
            "network_snapshot_available": False,
            "network_snapshot_path": None,
        }
    if payload.get("network_snapshot_available") is False:
        return {
            "network_signal_level": "missing",
            "network_signal_monitors": [
                f"network_snapshot_error={payload.get('network_error') or 'unknown'}",
            ],
            "network_snapshot_available": False,
            "network_snapshot_path": payload.get("network_snapshot_path"),
        }

    loaded_symbols = _as_int(payload.get("loaded_symbol_count"))
    cluster_count = _as_int(payload.get("cluster_count"))
    top_residual_mi = _as_float(payload.get("top_residual_mutual_information"))
    top_residual_pair = payload.get("top_residual_mutual_information_pair")
    top_mi = _as_float(payload.get("top_mutual_information"))

    monitors: list[str] = []
    if cluster_count is not None and cluster_count <= max_cluster_count_warning:
        monitors.append(f"cluster_count={cluster_count} <= warning={max_cluster_count_warning}")
    if top_residual_mi is not None and top_residual_mi >= residual_mi_warning:
        monitors.append(
            f"top_residual_mi={top_residual_mi:.4f} >= warning={residual_mi_warning:.4f}"
        )

    signal_level = NETWORK_SIGNAL_MONITOR if monitors else NETWORK_SIGNAL_HOLD
    if len(monitors) >= 2:
        signal_level = NETWORK_SIGNAL_HOLD

    return {
        "network_snapshot_available": True,
        "network_signal_level": signal_level,
        "network_signal_monitors": monitors,
        "network_snapshot_path": payload.get("snapshot_path") or payload.get("report_path") or str(resolved_path(network_lab_snapshot)),
        "network_cluster_count": cluster_count,
        "network_loaded_symbol_count": loaded_symbols,
        "network_max_cluster_count_warning": max_cluster_count_warning,
        "network_residual_mi_warning": residual_mi_warning,
        "network_top_residual_mi": top_residual_mi,
        "network_top_residual_mi_pair": top_residual_pair,
        "network_top_mutual_information": top_mi,
    }


def resolved_path(path: str | Path | None) -> str | None:
    if path in (None, ""):
        return None
    return str(Path(path))


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
    if bool(snapshot.get("pipeline_user_intervention_required")):
        items.append("pipeline_user_intervention_required=true")
    return items


def _ready_horizons(outcome_payload: dict[str, Any], snapshot: dict[str, Any]) -> list[str]:
    raw_horizons = outcome_payload.get("ready_horizons")
    if isinstance(raw_horizons, list) and raw_horizons:
        return sorted([str(item) for item in raw_horizons if item not in (None, "")], key=_horizon_sort_key)
    readiness = outcome_payload.get("horizon_readiness") or {}
    if isinstance(readiness, dict):
        ready = [
            str(horizon)
            for horizon, details in readiness.items()
            if isinstance(details, dict) and str(details.get("status") or "") == "ready_for_review"
        ]
        if ready:
            return sorted(ready, key=_horizon_sort_key)
    count = _as_int(snapshot.get("paper_stock_target_review_outcome_analysis_ready_horizon_count")) or 0
    return ["unknown"] if count > 0 else []


def _overall_row(outcome_payload: dict[str, Any]) -> dict[str, Any]:
    rows = outcome_payload.get("top_analysis_rows") or []
    if not isinstance(rows, list):
        return {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("dimension") or "") == "overall" and str(row.get("group_value") or "") == "all":
            return row
    return {}


def _group_for_horizon(outcome_payload: dict[str, Any], key: str, horizon: str) -> dict[str, Any]:
    groups = outcome_payload.get(key) or {}
    if not isinstance(groups, dict):
        return {}
    value = groups.get(horizon)
    return value if isinstance(value, dict) else {}


def _horizon_metric(row: dict[str, Any], metric: str, horizon: str) -> float | None:
    return _as_float(row.get(f"{metric}_{horizon}"))


def _budget_context(
    snapshot: dict[str, Any],
    outcome_payload: dict[str, Any],
    blockers: list[str],
    trial_satellite_budget: float,
    max_satellite_budget: float,
    min_overall_win_rate: float,
    min_overall_avg_return: float,
    max_worst_group_loss: float,
    network_signal_level: str = NETWORK_SIGNAL_MONITOR,
    network_signal_monitors: list[str] | None = None,
) -> dict[str, Any]:
    ready = _ready_horizons(outcome_payload, snapshot)
    selected_horizon = ready[0] if ready else None
    overall = _overall_row(outcome_payload)
    best_group = _group_for_horizon(outcome_payload, "best_groups", selected_horizon or "")
    worst_group = _group_for_horizon(outcome_payload, "worst_groups", selected_horizon or "")
    overall_win_rate = _horizon_metric(overall, "win_rate", selected_horizon or "") if selected_horizon else None
    overall_avg_return = _horizon_metric(overall, "avg_return", selected_horizon or "") if selected_horizon else None
    overall_evaluable = _horizon_metric(overall, "evaluable", selected_horizon or "") if selected_horizon else None
    best_avg_return = _horizon_metric(best_group, "avg_return", selected_horizon or "") if selected_horizon else None
    best_win_rate = _horizon_metric(best_group, "win_rate", selected_horizon or "") if selected_horizon else None
    worst_avg_return = _horizon_metric(worst_group, "avg_return", selected_horizon or "") if selected_horizon else None
    worst_win_rate = _horizon_metric(worst_group, "win_rate", selected_horizon or "") if selected_horizon else None
    current_satellite = _as_float(snapshot.get("paper_latest_satellite_weight")) or 0.0
    capped_trial = max(0.0, min(float(trial_satellite_budget), float(max_satellite_budget)))

    network_monitors = list(network_signal_monitors or [])
    if blockers:
        decision = "blocked_by_pipeline_gates"
        reason = "Resolve research pipeline blockers before changing satellite risk budget."
        action = "resolve_blockers"
        recommended_budget = current_satellite
    elif not selected_horizon:
        decision = "wait_for_outcome_samples"
        reason = "Outcome samples have not reached a ready review horizon."
        action = "accumulate_outcome_samples"
        recommended_budget = 0.0
    elif overall_win_rate is None or overall_avg_return is None:
        decision = "review_outcome_samples"
        reason = "A ready horizon exists, but the overall outcome row is incomplete."
        action = "manual_review_required"
        recommended_budget = current_satellite
    elif overall_win_rate < min_overall_win_rate or overall_avg_return < min_overall_avg_return:
        decision = "hold_default_core_base"
        reason = "Overall outcome edge is not positive enough for a satellite budget trial."
        action = "hold_core_base"
        recommended_budget = 0.0
    elif worst_avg_return is not None and worst_avg_return < max_worst_group_loss:
        decision = "review_outcome_samples"
        reason = "Worst outcome group breaches the configured loss guard; review filters before adding risk budget."
        action = "review_group_filters"
        recommended_budget = current_satellite
    elif network_signal_level == NETWORK_SIGNAL_HOLD:
        decision = "hold_default_core_base"
        reason = (
            "Network-lab evidence indicates elevated hidden linkage risk. "
            "Pause satellite budget expansion and monitor structural clustering separately."
        )
        action = "hold_core_base"
        recommended_budget = current_satellite
    else:
        decision = "eligible_for_small_satellite_trial"
        reason = "Ready outcome horizon passes the initial win-rate, average-return, and worst-group loss guards."
        action = "allow_research_trial_when_regime_risk_on"
        recommended_budget = max(current_satellite, capped_trial)
        recommended_budget = min(recommended_budget, float(max_satellite_budget))

    if network_monitors:
        monitor_text = "; ".join(network_monitors)
        reason = f"{reason} Network monitor: {monitor_text}"

    return {
        "risk_budget_decision": decision,
        "risk_budget_reason": reason,
        "next_action_stage": action,
        "recommended_satellite_budget": float(recommended_budget),
        "selected_horizon": selected_horizon,
        "ready_horizons": ready,
        "outcome_analysis_status": outcome_payload.get("analysis_status"),
        "outcome_ready_horizon_count": len(ready),
        "overall_evaluable": int(overall_evaluable) if overall_evaluable is not None else None,
        "overall_win_rate": overall_win_rate,
        "overall_avg_return": overall_avg_return,
        "best_group": best_group,
        "best_group_win_rate": best_win_rate,
        "best_group_avg_return": best_avg_return,
        "worst_group": worst_group,
        "worst_group_win_rate": worst_win_rate,
        "worst_group_avg_return": worst_avg_return,
        "trial_satellite_budget": float(trial_satellite_budget),
        "max_satellite_budget": float(max_satellite_budget),
        "min_overall_win_rate": float(min_overall_win_rate),
        "min_overall_avg_return": float(min_overall_avg_return),
        "max_worst_group_loss": float(max_worst_group_loss),
        "next_review_date": snapshot.get("paper_stock_target_review_outcome_calendar_next_action_date"),
        "next_review_horizon": snapshot.get("paper_stock_target_review_outcome_calendar_next_action_horizon"),
        "sample_warning": outcome_payload.get("sample_warning"),
        "network_signal_level": network_signal_level,
        "network_signal_monitors": network_monitors,
    }


def _checklist_rows(snapshot: dict[str, Any], blockers: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "section": "evidence",
            "item": "network_signal",
            "value": (
                f"level={snapshot.get('network_signal_level')}, monitors={'; '.join(snapshot.get('network_signal_monitors') or []) or 'none'}"
            ),
            "status": "ok"
            if snapshot.get("network_signal_level") == NETWORK_SIGNAL_MONITOR
            else "review",
            "next_action": "Monitor network_lab_snapshot evidence for cross-sectional crowding."
            if snapshot.get("network_signal_level") == NETWORK_SIGNAL_HOLD
            else "No blocking network evidence.",
        },
        {
            "section": "pipeline",
            "item": "blocking_items",
            "value": len(blockers),
            "status": "ok" if not blockers else "blocked",
            "next_action": "; ".join(blockers) if blockers else "Pipeline gates allow risk-budget review.",
        },
        {
            "section": "outcome",
            "item": "ready_horizons",
            "value": ", ".join(snapshot.get("ready_horizons") or []) or "none",
            "status": snapshot.get("outcome_analysis_status") or "unknown",
            "next_action": snapshot.get("sample_warning") or snapshot.get("risk_budget_reason"),
        },
        {
            "section": "outcome",
            "item": "overall_edge",
            "value": f"win_rate={_pct(snapshot.get('overall_win_rate'))}, avg_return={_pct(snapshot.get('overall_avg_return'))}",
            "status": "ok" if snapshot.get("risk_budget_decision") == "eligible_for_small_satellite_trial" else "review",
            "next_action": snapshot.get("risk_budget_reason"),
        },
        {
            "section": "outcome",
            "item": "worst_group_guard",
            "value": f"worst_avg_return={_pct(snapshot.get('worst_group_avg_return'))}, guard={_pct(snapshot.get('max_worst_group_loss'))}",
            "status": "ok"
            if snapshot.get("worst_group_avg_return") is None
            or float(snapshot.get("worst_group_avg_return")) >= float(snapshot.get("max_worst_group_loss"))
            else "review",
            "next_action": "Review worst group before increasing budget if the guard is breached.",
        },
        {
            "section": "risk_budget",
            "item": "recommended_satellite_budget",
            "value": _pct(snapshot.get("recommended_satellite_budget")),
            "status": snapshot.get("risk_budget_decision"),
            "next_action": "Research-only recommendation; do not place broker orders from this file.",
        },
    ]


def _group_label(group: dict[str, Any]) -> str:
    if not group:
        return "N/A"
    return f"{group.get('dimension') or 'N/A'}={group.get('group_value') or 'N/A'}"


def _render_report(snapshot: dict[str, Any], checklist: pd.DataFrame) -> str:
    blockers = snapshot.get("blocking_items") or []
    blocker_text = "\n".join(f"- `{item}`" for item in blockers) if blockers else "- None"
    rows = [
        f"| {row.section} | {row.item} | `{row.status}` | {row.value} | {row.next_action} |"
        for row in checklist.itertuples(index=False)
    ]
    return f"""# Satellite Risk Budget Review

Generated at: `{snapshot.get("generated_at")}`

This is a research-only satellite budget review. It does not connect to brokers, place orders, change live positions, or provide investment advice.

## Summary

| Item | Value |
| --- | ---: |
| Decision | `{snapshot.get("risk_budget_decision")}` |
| Reason | {snapshot.get("risk_budget_reason")} |
| Next action stage | `{snapshot.get("next_action_stage")}` |
| As-of date | {snapshot.get("as_of_date", "N/A")} |
| Selected horizon | `{snapshot.get("selected_horizon") or "N/A"}` |
| Ready horizons | {", ".join(snapshot.get("ready_horizons") or []) or "None"} |
| Recommended satellite budget | {_pct(snapshot.get("recommended_satellite_budget"))} |
| Current core / satellite weight | {_pct(snapshot.get("paper_latest_core_weight"))} / {_pct(snapshot.get("paper_latest_satellite_weight"))} |
| Latest regime | `{snapshot.get("paper_latest_regime", "N/A")}` |
| Next review | {snapshot.get("next_review_date") or "N/A"} / `{snapshot.get("next_review_horizon") or "N/A"}` |

## Outcome Evidence

| Item | Value |
| --- | ---: |
| Overall evaluable | {snapshot.get("overall_evaluable") if snapshot.get("overall_evaluable") is not None else "N/A"} |
| Overall win rate | {_pct(snapshot.get("overall_win_rate"))} |
| Overall avg return | {_pct(snapshot.get("overall_avg_return"))} |
| Best group | `{_group_label(snapshot.get("best_group") or {})}` |
| Best group win rate / avg return | {_pct(snapshot.get("best_group_win_rate"))} / {_pct(snapshot.get("best_group_avg_return"))} |
| Worst group | `{_group_label(snapshot.get("worst_group") or {})}` |
| Worst group win rate / avg return | {_pct(snapshot.get("worst_group_win_rate"))} / {_pct(snapshot.get("worst_group_avg_return"))} |
| Worst group loss guard | {_pct(snapshot.get("max_worst_group_loss"))} |

## Checklist

| Section | Item | Status | Value | Next action |
| --- | --- | --- | --- | --- |
{chr(10).join(rows)}

## Blocking Items

{blocker_text}

## Source Files

- Pipeline snapshot: `{snapshot.get("pipeline_snapshot")}`
- Pipeline report: `{snapshot.get("pipeline_report_path")}`
- Outcome analysis JSON: `{snapshot.get("outcome_analysis_path")}`
- Outcome analysis report: `{snapshot.get("outcome_analysis_report_path")}`
- Network lab snapshot: `{snapshot.get("network_snapshot_path")}`
- Checklist CSV: `satellite_risk_budget_checklist.csv`
- Snapshot JSON: `satellite_risk_budget_snapshot.json`
"""


def run_satellite_risk_budget_review(
    pipeline_snapshot: str | Path,
    outcome_analysis_path: str | Path | None = None,
    output_dir: str | Path = "outputs/research/satellite_risk_budget_review_latest",
    trial_satellite_budget: float = 0.05,
    max_satellite_budget: float = 0.20,
    min_overall_win_rate: float = 0.55,
    min_overall_avg_return: float = 0.0,
    max_worst_group_loss: float = -0.05,
    network_lab_snapshot: str | Path | None = None,
    network_max_cluster_count_warning: int = 1,
    network_residual_mi_warning: float = 0.20,
) -> SatelliteRiskBudgetReviewResult:
    pipeline_path = Path(pipeline_snapshot)
    source = _load_json(pipeline_path, "daily-pipeline snapshot")
    resolved_outcome_path = Path(outcome_analysis_path) if outcome_analysis_path else _snapshot_outcome_analysis_path(source)
    if resolved_outcome_path is None:
        raise FileNotFoundError("Missing outcome analysis JSON path; pass --outcome-analysis-path.")
    outcome_payload = _load_json(resolved_outcome_path, "outcome analysis JSON")
    network_payload = _network_signal_context(
        network_lab_snapshot,
        max_cluster_count_warning=network_max_cluster_count_warning,
        residual_mi_warning=network_residual_mi_warning,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blockers = _blocking_items(source)
    budget = _budget_context(
        source,
        outcome_payload,
        blockers,
        trial_satellite_budget,
        max_satellite_budget,
        min_overall_win_rate,
        min_overall_avg_return,
        max_worst_group_loss,
        network_signal_level=str(network_payload.get("network_signal_level") or NETWORK_SIGNAL_MONITOR),
        network_signal_monitors=network_payload.get("network_signal_monitors"),
    )
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "satellite_risk_budget_review_completed",
        "pipeline_snapshot": str(pipeline_path),
        "outcome_analysis_path": str(resolved_outcome_path),
        "outcome_analysis_report_path": outcome_payload.get("analysis_report_path"),
        "blocking_items": blockers,
        **source,
        **budget,
        **network_payload,
    }
    checklist = pd.DataFrame(_checklist_rows(snapshot, blockers))
    checklist_path = out_dir / "satellite_risk_budget_checklist.csv"
    snapshot_path = out_dir / "satellite_risk_budget_snapshot.json"
    report_path = out_dir / "satellite_risk_budget_review.md"

    checklist.to_csv(checklist_path, index=False, encoding="utf-8-sig")
    snapshot_path.write_text(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, checklist), encoding="utf-8")

    return SatelliteRiskBudgetReviewResult(
        output_dir=out_dir,
        report_path=report_path,
        checklist_path=checklist_path,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )
