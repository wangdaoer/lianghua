"""Compare default and candidate allocator daily-pipeline probes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AllocatorSwitchReadinessResult:
    output_dir: Path
    report_path: Path
    comparison_path: Path
    snapshot_path: Path
    snapshot: dict[str, Any]


ALERT_RANK = {"normal": 0, "info": 1, "warning": 2, "critical": 3}


def _load_snapshot(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing pipeline snapshot: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _load_outcome_analysis(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing outcome analysis JSON: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def _candidate_outcome_analysis_path(candidate: dict[str, Any]) -> Path | None:
    raw_path = candidate.get("paper_stock_target_review_outcome_analysis_json_path")
    if raw_path in (None, ""):
        return None
    resolved = Path(str(raw_path))
    return resolved if resolved.exists() else None


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


def _delta(candidate: Any, default: Any) -> float | None:
    candidate_number = _as_float(candidate)
    default_number = _as_float(default)
    if candidate_number is None or default_number is None:
        return None
    return candidate_number - default_number


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


def _pp(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    sign = "+" if number >= 0 else ""
    return f"{sign}{number * 100:.2f} pp"


def _money(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "N/A"
    return f"{number:,.2f}"


def _number(value: Any, digits: int = 3) -> str:
    number = _as_float(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def _text(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    return f"`{value}`"


def _shared_alert_level(default: dict[str, Any], candidate: dict[str, Any]) -> str:
    default_level = str(default.get("alert_level") or "normal")
    candidate_level = str(candidate.get("alert_level") or "normal")
    if ALERT_RANK.get(candidate_level, 0) >= ALERT_RANK.get(default_level, 0):
        return candidate_level
    return default_level


def _blocking_items(default: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    items: list[str] = []
    if str(candidate.get("promotion_decision") or "") != "promote_candidate":
        items.append(f"promotion_decision={candidate.get('promotion_decision') or 'missing'}")
    for label, snapshot in (("default", default), ("candidate", candidate)):
        if str(snapshot.get("dashboard_posture") or "") != "core_base_watch_allocator_gate":
            items.append(f"{label}_dashboard_posture={snapshot.get('dashboard_posture') or 'missing'}")
        if str(snapshot.get("trading_day_gate_status") or "") != "trading_day_data_ready":
            items.append(f"{label}_trading_day_gate={snapshot.get('trading_day_gate_status') or 'missing'}")
        if str(snapshot.get("after_close_data_status") or "") != "ready":
            items.append(f"{label}_after_close_data={snapshot.get('after_close_data_status') or 'missing'}")
        if ALERT_RANK.get(str(snapshot.get("alert_level") or "normal"), 0) >= ALERT_RANK["warning"]:
            items.append(f"{label}_alert_level={snapshot.get('alert_level')}")
        if str(snapshot.get("history_review_health_state") or "ok") not in {"ok", "missing"}:
            items.append(f"{label}_history_health={snapshot.get('history_review_health_state')}")
        if _as_int(snapshot.get("model_audit_walk_forward_action_items")) not in {None, 0}:
            items.append(f"{label}_model_audit_actions={snapshot.get('model_audit_walk_forward_action_items')}")
        if _as_int(snapshot.get("paper_stock_target_review_action_count")) not in {None, 0}:
            items.append(f"{label}_stock_target_actions={snapshot.get('paper_stock_target_review_action_count')}")
        if bool(snapshot.get("pipeline_user_intervention_required")):
            items.append(f"{label}_user_intervention_required=true")
    return items


def _monitor_items(candidate: dict[str, Any]) -> list[str]:
    items: list[str] = []
    drawdown_count = _as_int(candidate.get("paper_stock_target_review_drawdown_count"))
    if drawdown_count:
        items.append(f"reviewed_drawdown_targets={drawdown_count}")
    suppressed_count = _as_int(candidate.get("paper_stock_target_review_suppressed_layer_count"))
    if suppressed_count:
        items.append(f"suppressed_layer_targets={suppressed_count}")
    outcome_status = str(candidate.get("paper_stock_target_review_outcome_analysis_status") or "")
    if outcome_status in {"waiting_for_evaluable_returns", "sample_insufficient"}:
        items.append("outcome_samples_waiting")
    return items


def _outcome_ready_horizons(outcome_payload: dict[str, Any] | None) -> list[str]:
    if not outcome_payload:
        return []
    raw_horizons = outcome_payload.get("ready_horizons")
    if isinstance(raw_horizons, list):
        return [str(horizon) for horizon in raw_horizons if horizon not in (None, "")]
    readiness = outcome_payload.get("horizon_readiness") or {}
    if not isinstance(readiness, dict):
        return []
    return [
        str(horizon)
        for horizon, details in readiness.items()
        if isinstance(details, dict) and str(details.get("status") or "") == "ready_for_review"
    ]


def _risk_budget_context(
    blockers: list[str],
    candidate: dict[str, Any],
    outcome_payload: dict[str, Any] | None = None,
    outcome_analysis_path: Path | None = None,
) -> dict[str, Any]:
    outcome_status = str(
        (outcome_payload or {}).get("analysis_status")
        or candidate.get("paper_stock_target_review_outcome_analysis_status")
        or ""
    )
    ready_horizon_count = _as_int((outcome_payload or {}).get("ready_horizon_count"))
    ready_horizons = _outcome_ready_horizons(outcome_payload)
    if ready_horizon_count is None:
        ready_horizon_count = len(ready_horizons) if ready_horizons else None
    if ready_horizon_count is None:
        ready_horizon_count = _as_int(candidate.get("paper_stock_target_review_outcome_analysis_ready_horizon_count")) or 0
    next_date = candidate.get("paper_stock_target_review_outcome_calendar_next_action_date")
    next_horizon = candidate.get("paper_stock_target_review_outcome_calendar_next_action_horizon")
    latest_regime = str(candidate.get("paper_latest_regime") or "")
    latest_satellite_weight = _as_float(candidate.get("paper_latest_satellite_weight")) or 0.0

    if blockers:
        decision = "blocked_by_switch_readiness"
        reason = "Resolve allocator switch-readiness blocking items before reviewing satellite risk budget."
    elif outcome_status in {"waiting_for_evaluable_returns", "sample_insufficient"} or ready_horizon_count <= 0:
        decision = "wait_for_outcome_samples"
        reason = "Stock-target outcome samples have not passed the configured readiness gates yet."
    elif ready_horizon_count > 0:
        decision = "review_outcome_samples"
        reason = "At least one stock-target outcome horizon is ready for review before changing satellite risk budget."
    elif latest_satellite_weight <= 0 or latest_regime != "risk_on":
        decision = "wait_for_risk_on_regime"
        reason = "The latest paper account is not carrying active satellite exposure."
    else:
        decision = "eligible_for_satellite_budget_review"
        reason = "Switch-readiness is clean, outcome samples are ready, and satellite exposure is active."

    return {
        "risk_budget_decision": decision,
        "risk_budget_reason": reason,
        "outcome_analysis_status": outcome_status,
        "outcome_ready_horizon_count": ready_horizon_count,
        "outcome_ready_horizons": ready_horizons,
        "next_outcome_review_date": next_date,
        "next_outcome_review_horizon": next_horizon,
        "outcome_analysis_path": str(outcome_analysis_path) if outcome_analysis_path else None,
        "outcome_horizon_readiness": (outcome_payload or {}).get("horizon_readiness") or {},
        "outcome_best_groups": (outcome_payload or {}).get("best_groups") or {},
        "outcome_worst_groups": (outcome_payload or {}).get("worst_groups") or {},
        "outcome_analysis_report_path": (outcome_payload or {}).get("analysis_report_path"),
        "outcome_analysis_csv_path": (outcome_payload or {}).get("analysis_path"),
        "outcome_sample_warning": (outcome_payload or {}).get("sample_warning"),
    }


def _comparison_rows(default: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [
        ("Promotion decision", "promotion_decision", "text"),
        ("Independent support groups", "promotion_support_group_count", "int"),
        ("Evidence support groups", "promotion_evidence_support_count", "int"),
        ("Dashboard posture", "dashboard_posture", "text"),
        ("Trading-day gate", "trading_day_gate_status", "text"),
        ("After-close data", "after_close_data_status", "text"),
        ("Alert level", "alert_level", "text"),
        ("Alert count", "alert_count", "int"),
        ("Action stage", "alert_action_stage", "text"),
        ("Pipeline next step", "pipeline_next_step_stage", "text"),
        ("Paper final equity", "paper_final_equity", "money"),
        ("Paper total return", "paper_total_return", "pct"),
        ("Paper max drawdown", "paper_max_drawdown", "pct"),
        ("Paper Sharpe", "paper_sharpe", "number"),
        ("Paper current drawdown", "paper_current_drawdown", "pct"),
        ("Latest regime", "paper_latest_regime", "text"),
        ("Latest candidate", "paper_latest_candidate", "text"),
        ("Core weight", "paper_latest_core_weight", "pct"),
        ("Satellite weight", "paper_latest_satellite_weight", "pct"),
        ("Stock target count", "paper_stock_target_count", "int"),
        ("Review-required targets", "paper_stock_target_review_required_count", "int"),
        ("Drawdown-review targets", "paper_stock_target_review_drawdown_count", "int"),
        ("Action queue count", "paper_stock_target_review_action_count", "int"),
        ("Outcome analysis status", "paper_stock_target_review_outcome_analysis_status", "text"),
        ("History health", "history_review_health_state", "text"),
        ("Model audit status", "model_audit_status", "text"),
        ("Model audit actions", "model_audit_walk_forward_action_items", "int"),
    ]
    rows: list[dict[str, Any]] = []
    for item, key, value_type in specs:
        default_value = default.get(key)
        candidate_value = candidate.get(key)
        rows.append(
            {
                "item": item,
                "field": key,
                "default_value": default_value,
                "candidate_value": candidate_value,
                "delta": _delta(candidate_value, default_value),
                "value_type": value_type,
            }
        )
    return rows


def _render_value(value: Any, value_type: str) -> str:
    if value_type == "pct":
        return _pct(value)
    if value_type == "money":
        return _money(value)
    if value_type == "number":
        return _number(value)
    if value_type == "int":
        number = _as_int(value)
        return "N/A" if number is None else f"`{number}`"
    return _text(value)


def _render_delta(value: Any, value_type: str) -> str:
    if value is None:
        return ""
    if value_type == "pct":
        return _pp(value)
    if value_type == "money":
        number = _as_float(value)
        if number is None:
            return ""
        sign = "+" if number >= 0 else ""
        return f"{sign}{number:,.2f}"
    if value_type == "number":
        number = _as_float(value)
        if number is None:
            return ""
        sign = "+" if number >= 0 else ""
        return f"{sign}{number:.3f}"
    if value_type == "int":
        number = _as_int(value)
        if number is None:
            return ""
        sign = "+" if number >= 0 else ""
        return f"{sign}{number}"
    return ""


def _horizon_sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return int(digits), text
    return 9999, text


def _render_outcome_horizon_rows(snapshot: dict[str, Any]) -> str:
    readiness = snapshot.get("outcome_horizon_readiness") or {}
    if not isinstance(readiness, dict) or not readiness:
        return "| N/A | N/A | N/A | N/A | N/A |"
    rows: list[str] = []
    for horizon in sorted(readiness, key=_horizon_sort_key):
        details = readiness.get(horizon) or {}
        if not isinstance(details, dict):
            details = {}
        rows.append(
            "| "
            f"`{horizon}` | "
            f"`{details.get('status') or 'N/A'}` | "
            f"{details.get('evaluable_count', 'N/A')} / {details.get('min_evaluable', 'N/A')} | "
            f"{details.get('qualified_group_count', 'N/A')} / {details.get('min_group_evaluable', 'N/A')} | "
            f"{'yes' if str(horizon) in set(snapshot.get('outcome_ready_horizons') or []) else 'no'} |"
        )
    return "\n".join(rows)


def _render_outcome_group_rows(snapshot: dict[str, Any], group_key: str) -> str:
    groups = snapshot.get(group_key) or {}
    if not isinstance(groups, dict) or not groups:
        return "| N/A | N/A | N/A | N/A |"
    rows: list[str] = []
    for horizon in sorted(groups, key=_horizon_sort_key):
        details = groups.get(horizon) or {}
        if not isinstance(details, dict):
            details = {}
        win_rate = details.get(f"win_rate_{horizon}")
        avg_return = details.get(f"avg_return_{horizon}")
        count = details.get(f"evaluable_count_{horizon}") or details.get("evaluable_count")
        rows.append(
            "| "
            f"`{horizon}` | "
            f"`{details.get('dimension') or 'N/A'}={details.get('group_value') or 'N/A'}` | "
            f"{count if count not in (None, '') else 'N/A'} | "
            f"{_pct(win_rate)} / {_pct(avg_return)} |"
        )
    return "\n".join(rows)


def _render_report(
    snapshot: dict[str, Any],
    comparison: pd.DataFrame,
    default_label: str,
    candidate_label: str,
) -> str:
    rows = [
        f"| {row.item} | {_render_value(row.default_value, row.value_type)} | {_render_value(row.candidate_value, row.value_type)} | {_render_delta(row.delta, row.value_type)} |"
        for row in comparison.itertuples(index=False)
    ]
    blockers = snapshot.get("blocking_items") or []
    monitors = snapshot.get("monitor_items") or []
    blocker_text = "\n".join(f"- `{item}`" for item in blockers) if blockers else "- None"
    monitor_text = "\n".join(f"- `{item}`" for item in monitors) if monitors else "- None"
    ready_horizons = snapshot.get("outcome_ready_horizons") or []
    ready_horizon_text = ", ".join(f"`{item}`" for item in ready_horizons) if ready_horizons else "None"
    return f"""# Allocator Switch Readiness

Generated at: `{snapshot["generated_at"]}`

This is a research-only comparison. It does not connect to brokers, place orders, change target weights, or provide investment advice.

## Summary

| Item | Value |
| --- | ---: |
| Decision | `{snapshot["decision"]}` |
| As-of date | {snapshot.get("as_of_date", "N/A")} |
| Candidate return edge | {_pp(snapshot.get("candidate_return_edge"))} |
| Candidate drawdown change | {_pp(snapshot.get("candidate_drawdown_change"))} |
| Candidate Sharpe edge | {_number(snapshot.get("candidate_sharpe_edge"))} |
| Shared alert level | `{snapshot.get("shared_alert_level")}` |
| Risk budget decision | `{snapshot.get("risk_budget_decision", "N/A")}` |
| Next outcome review | {snapshot.get("next_outcome_review_date") or "N/A"} / `{snapshot.get("next_outcome_review_horizon") or "N/A"}` |
| Blocking items | {len(blockers)} |
| Monitor items | {len(monitors)} |

## Default vs Candidate

| Item | {default_label} | {candidate_label} | Candidate - Default |
| --- | ---: | ---: | ---: |
{chr(10).join(rows)}

## Blocking Items

{blocker_text}

## Monitor Items

{monitor_text}

## Outcome Readiness

| Item | Value |
| --- | ---: |
| Outcome analysis status | `{snapshot.get("outcome_analysis_status") or "N/A"}` |
| Ready horizon count | `{snapshot.get("outcome_ready_horizon_count", "N/A")}` |
| Ready horizons | {ready_horizon_text} |
| Outcome sample warning | {snapshot.get("outcome_sample_warning") or "N/A"} |
| Outcome analysis JSON | `{snapshot.get("outcome_analysis_path") or "N/A"}` |
| Outcome analysis CSV | `{snapshot.get("outcome_analysis_csv_path") or "N/A"}` |
| Outcome analysis report | `{snapshot.get("outcome_analysis_report_path") or "N/A"}` |
| Risk budget reason | {snapshot.get("risk_budget_reason") or "N/A"} |

### Horizon Gates

| Horizon | Status | Evaluable / Minimum | Qualified groups / Minimum | Ready |
| --- | ---: | ---: | ---: | ---: |
{_render_outcome_horizon_rows(snapshot)}

### Best Outcome Groups

| Horizon | Group | Evaluable | Win rate / Avg return |
| --- | ---: | ---: | ---: |
{_render_outcome_group_rows(snapshot, "outcome_best_groups")}

### Worst Outcome Groups

| Horizon | Group | Evaluable | Win rate / Avg return |
| --- | ---: | ---: | ---: |
{_render_outcome_group_rows(snapshot, "outcome_worst_groups")}

## Interpretation

- `ready_for_controlled_default_switch` means the candidate passed the research-governance and pipeline-readiness checks in this comparison; it is not an automatic operational switch.
- `risk_budget_decision` is a separate guard for whether to consider increasing satellite exposure.
- A positive drawdown change means the candidate max drawdown is less negative than the default.
- Keep the candidate in parallel paper verification until outcome samples mature enough for stock-level effectiveness review.

## Source Files

- Default snapshot: `{snapshot.get("default_snapshot")}`
- Candidate snapshot: `{snapshot.get("candidate_snapshot")}`
- Default pipeline: `{snapshot.get("default_pipeline_report_path")}`
- Candidate pipeline: `{snapshot.get("candidate_pipeline_report_path")}`
- Comparison CSV: `allocator_switch_readiness_comparison.csv`
- Snapshot JSON: `allocator_switch_readiness_snapshot.json`
"""


def run_allocator_switch_readiness(
    default_snapshot: str | Path,
    candidate_snapshot: str | Path,
    output_dir: str | Path = "outputs/research/allocator_switch_readiness_latest",
    default_label: str = "Default quality-v2",
    candidate_label: str = "Candidate allocator",
    outcome_analysis_path: str | Path | None = None,
) -> AllocatorSwitchReadinessResult:
    default_path = Path(default_snapshot)
    candidate_path = Path(candidate_snapshot)
    default = _load_snapshot(default_path)
    candidate = _load_snapshot(candidate_path)
    resolved_outcome_path = Path(outcome_analysis_path) if outcome_analysis_path else _candidate_outcome_analysis_path(candidate)
    outcome_payload = _load_outcome_analysis(resolved_outcome_path) if resolved_outcome_path else None
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = pd.DataFrame(_comparison_rows(default, candidate))
    comparison_path = out_dir / "allocator_switch_readiness_comparison.csv"
    snapshot_path = out_dir / "allocator_switch_readiness_snapshot.json"
    report_path = out_dir / "allocator_switch_readiness.md"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    return_edge = _delta(candidate.get("paper_total_return"), default.get("paper_total_return"))
    drawdown_change = _delta(candidate.get("paper_max_drawdown"), default.get("paper_max_drawdown"))
    sharpe_edge = _delta(candidate.get("paper_sharpe"), default.get("paper_sharpe"))
    blockers = _blocking_items(default, candidate)
    risk_budget = _risk_budget_context(blockers, candidate, outcome_payload, resolved_outcome_path)
    decision = (
        "ready_for_controlled_default_switch"
        if not blockers
        and (return_edge is not None and return_edge > 0)
        and (drawdown_change is not None and drawdown_change >= 0)
        and (sharpe_edge is not None and sharpe_edge > 0)
        else "keep_parallel_watch"
    )
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "allocator_switch_readiness_completed",
        "decision": decision,
        "as_of_date": candidate.get("as_of_date") or default.get("as_of_date"),
        "default_label": default_label,
        "candidate_label": candidate_label,
        "default_snapshot": str(default_path),
        "candidate_snapshot": str(candidate_path),
        "default_pipeline_report_path": default.get("pipeline_report_path"),
        "candidate_pipeline_report_path": candidate.get("pipeline_report_path"),
        "candidate_return_edge": return_edge,
        "candidate_drawdown_change": drawdown_change,
        "candidate_sharpe_edge": sharpe_edge,
        "shared_alert_level": _shared_alert_level(default, candidate),
        "default_alert_count": _as_int(default.get("alert_count")),
        "candidate_alert_count": _as_int(candidate.get("alert_count")),
        "blocking_items": blockers,
        "monitor_items": _monitor_items(candidate),
        **risk_budget,
        "comparison_path": str(comparison_path),
        "report_path": str(report_path),
    }
    snapshot_path.write_text(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, comparison, default_label, candidate_label), encoding="utf-8")

    return AllocatorSwitchReadinessResult(
        output_dir=out_dir,
        report_path=report_path,
        comparison_path=comparison_path,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )
