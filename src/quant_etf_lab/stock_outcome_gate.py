"""Research-only gate for stock-target outcome evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ._compat import read_text


DEFAULT_OUTPUT_DIR = Path("outputs/research/stock_outcome_gate_latest")
DEFAULT_PRIMARY_HORIZON = "5d"
READY_STATUSES = {"ready_for_group_review", "ready_for_review"}
DEFAULT_FOCUS_DIMENSIONS = ("layer", "review_bucket", "review_stage")


@dataclass(frozen=True)
class StockOutcomeGateResult:
    output_dir: Path
    gate_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    gate: pd.DataFrame
    target_overlay_path: Path | None = None
    target_overlay: pd.DataFrame | None = None
    shadow_selection_path: Path | None = None
    shadow_selection: pd.DataFrame | None = None
    weak_sentiment_trial_csv_path: Path | None = None
    weak_sentiment_trial_report_path: Path | None = None
    weak_sentiment_trial: pd.DataFrame | None = None


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


def _horizon_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return text if text.endswith("d") else f"{text}d"


def _horizon_sort_key(value: Any) -> tuple[int, str]:
    text = _horizon_key(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits), text) if digits else (9999, text)


def _ready_horizons(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("ready_horizons")
    if isinstance(raw, list):
        ready = [_horizon_key(item) for item in raw if _horizon_key(item)]
        if ready:
            return sorted(set(ready), key=_horizon_sort_key)

    readiness = payload.get("horizon_readiness") or {}
    if not isinstance(readiness, dict):
        return []
    ready = []
    for horizon, details in readiness.items():
        if not isinstance(details, dict):
            continue
        if str(details.get("status") or "") in READY_STATUSES:
            key = _horizon_key(horizon)
            if key:
                ready.append(key)
    return sorted(set(ready), key=_horizon_sort_key)


def _selected_horizon(payload: dict[str, Any], primary_horizon: str) -> str | None:
    ready = _ready_horizons(payload)
    if not ready:
        return None
    primary = _horizon_key(primary_horizon)
    return primary if primary in ready else ready[0]


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing outcome analysis JSON: {resolved}")
    return json.loads(read_text(resolved))


def _load_optional_json(path: str | Path | None) -> dict[str, Any]:
    if path in (None, ""):
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    return json.loads(read_text(resolved))


def _load_latest_csv_record(path: str | Path | None) -> dict[str, Any]:
    if path in (None, ""):
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    frame = pd.read_csv(resolved)
    if frame.empty:
        return {}
    return frame.tail(1).iloc[0].to_dict()


def _analysis_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("top_analysis_rows")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _classify_group(
    row: dict[str, Any],
    horizon: str,
    min_group_evaluable: int,
    prefer_min_avg_return: float,
    prefer_min_win_rate: float,
    caution_max_avg_return: float,
    caution_max_win_rate: float,
) -> dict[str, Any] | None:
    dimension = str(row.get("dimension") or "").strip()
    group_value = str(row.get("group_value") or "").strip()
    if not dimension or dimension == "overall" or not group_value:
        return None

    evaluable = _as_int(row.get(f"evaluable_{horizon}")) or 0
    avg_return = _as_float(row.get(f"avg_return_{horizon}"))
    win_rate = _as_float(row.get(f"win_rate_{horizon}"))
    label = f"{dimension}={group_value}"

    if evaluable < min_group_evaluable:
        action = "insufficient_sample"
        reason = f"evaluable={evaluable} below min_group_evaluable={min_group_evaluable}"
    elif avg_return is None or win_rate is None:
        action = "insufficient_metrics"
        reason = "avg_return or win_rate is missing"
    elif avg_return >= prefer_min_avg_return and win_rate >= prefer_min_win_rate:
        action = "prefer"
        reason = (
            f"avg_return={avg_return:.4f} >= {prefer_min_avg_return:.4f} "
            f"and win_rate={win_rate:.4f} >= {prefer_min_win_rate:.4f}"
        )
    elif avg_return <= caution_max_avg_return or win_rate <= caution_max_win_rate:
        triggers = []
        if avg_return <= caution_max_avg_return:
            triggers.append(f"avg_return={avg_return:.4f} <= {caution_max_avg_return:.4f}")
        if win_rate <= caution_max_win_rate:
            triggers.append(f"win_rate={win_rate:.4f} <= {caution_max_win_rate:.4f}")
        action = "caution"
        reason = " or ".join(triggers)
    else:
        action = "neutral"
        reason = "group does not pass prefer or caution thresholds"

    return {
        "dimension": dimension,
        "group_value": group_value,
        "group_label": label,
        "selected_horizon": horizon,
        "history_row_count": _as_int(row.get("history_row_count")),
        "evaluable_count": evaluable,
        "avg_return": avg_return,
        "win_rate": win_rate,
        "gate_action": action,
        "gate_reason": reason,
        "research_only": True,
        "broker_action": "none",
    }


def _decision(prefer_groups: list[str], caution_groups: list[str], selected_horizon: str | None, actionable_rows: int) -> str:
    if not selected_horizon:
        return "wait_for_outcome_samples"
    if actionable_rows <= 0:
        return "wait_for_group_samples"
    if prefer_groups and caution_groups:
        return "prefer_positive_groups_with_caution"
    if prefer_groups:
        return "prefer_positive_groups"
    if caution_groups:
        return "caution_on_weak_groups"
    return "neutral_monitor"


def build_stock_outcome_gate(
    outcome_payload: dict[str, Any],
    primary_horizon: str = DEFAULT_PRIMARY_HORIZON,
    focus_dimensions: tuple[str, ...] = DEFAULT_FOCUS_DIMENSIONS,
    min_group_evaluable: int = 5,
    prefer_min_avg_return: float = 0.03,
    prefer_min_win_rate: float = 0.50,
    caution_max_avg_return: float = -0.02,
    caution_max_win_rate: float = 0.35,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert mature outcome-analysis groups into a research-only evidence gate."""
    selected = _selected_horizon(outcome_payload, primary_horizon)
    ready = _ready_horizons(outcome_payload)
    rows: list[dict[str, Any]] = []
    if selected:
        focus = set(focus_dimensions)
        for raw in _analysis_rows(outcome_payload):
            if str(raw.get("dimension") or "") not in focus:
                continue
            classified = _classify_group(
                raw,
                selected,
                min_group_evaluable=min_group_evaluable,
                prefer_min_avg_return=prefer_min_avg_return,
                prefer_min_win_rate=prefer_min_win_rate,
                caution_max_avg_return=caution_max_avg_return,
                caution_max_win_rate=caution_max_win_rate,
            )
            if classified is not None:
                rows.append(classified)

    gate = pd.DataFrame(rows)
    if not gate.empty:
        order = {"prefer": 0, "caution": 1, "neutral": 2, "insufficient_sample": 3, "insufficient_metrics": 4}
        gate["_action_order"] = gate["gate_action"].map(order).fillna(99)
        gate = gate.sort_values(
            ["_action_order", "dimension", "group_value"],
            ascending=[True, True, True],
        ).drop(columns=["_action_order"]).reset_index(drop=True)

    prefer_groups = gate.loc[gate["gate_action"] == "prefer", "group_label"].tolist() if not gate.empty else []
    caution_groups = gate.loc[gate["gate_action"] == "caution", "group_label"].tolist() if not gate.empty else []
    neutral_groups = gate.loc[gate["gate_action"] == "neutral", "group_label"].tolist() if not gate.empty else []
    insufficient_groups = (
        gate.loc[gate["gate_action"].isin(["insufficient_sample", "insufficient_metrics"]), "group_label"].tolist()
        if not gate.empty
        else []
    )
    actionable_rows = len(prefer_groups) + len(caution_groups) + len(neutral_groups)
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "stock_outcome_gate_completed",
        "analysis_status": outcome_payload.get("analysis_status"),
        "selected_horizon": selected,
        "requested_primary_horizon": _horizon_key(primary_horizon),
        "ready_horizons": ready,
        "decision": _decision(prefer_groups, caution_groups, selected, actionable_rows),
        "prefer_group_count": len(prefer_groups),
        "caution_group_count": len(caution_groups),
        "neutral_group_count": len(neutral_groups),
        "insufficient_group_count": len(insufficient_groups),
        "prefer_groups": prefer_groups,
        "caution_groups": caution_groups,
        "neutral_groups": neutral_groups,
        "insufficient_groups": insufficient_groups,
        "min_group_evaluable": int(min_group_evaluable),
        "prefer_min_avg_return": float(prefer_min_avg_return),
        "prefer_min_win_rate": float(prefer_min_win_rate),
        "caution_max_avg_return": float(caution_max_avg_return),
        "caution_max_win_rate": float(caution_max_win_rate),
        "source_analysis_path": outcome_payload.get("analysis_path"),
        "source_analysis_json_path": outcome_payload.get("analysis_json_path"),
        "source_analysis_report_path": outcome_payload.get("analysis_report_path"),
        "source_history_path": outcome_payload.get("source_history_path"),
        "source_history_row_count": outcome_payload.get("source_history_row_count"),
        "source_history_latest_review_date": outcome_payload.get("source_history_latest_review_date"),
        "gate_rows": gate.to_dict(orient="records") if not gate.empty else [],
        "research_only": True,
        "broker_action": "none",
        "note": "Outcome gate is research-only. It does not change model targets, target weights, broker signals, or orders.",
    }
    return gate, snapshot


def _gate_lookup(gate: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if gate.empty or "group_label" not in gate.columns:
        return {}
    return {
        str(row.get("group_label")): row
        for row in gate.to_dict(orient="records")
        if row.get("group_label") not in (None, "")
    }


def _target_group_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for column in ("layer", "review_bucket", "review_stage"):
        value = str(row.get(column) or "").strip()
        if value:
            labels.append(f"{column}={value}")
    return labels


def _target_gate_action(actions: list[str]) -> str:
    action_set = set(actions)
    if "prefer" in action_set and "caution" in action_set:
        return "mixed_review"
    if "prefer" in action_set:
        return "prefer"
    if "caution" in action_set:
        return "caution"
    if "neutral" in action_set:
        return "neutral"
    if action_set.intersection({"insufficient_sample", "insufficient_metrics"}):
        return "insufficient_evidence"
    return "no_gate_match"


def _target_research_priority(action: str) -> str:
    return {
        "prefer": "increase_research_priority",
        "caution": "reduce_research_priority",
        "mixed_review": "manual_compare",
        "neutral": "maintain_research_priority",
        "insufficient_evidence": "wait_for_more_samples",
        "no_gate_match": "unclassified",
    }.get(action, "unclassified")


def build_stock_outcome_target_overlay(targets: pd.DataFrame, gate: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Overlay outcome-gate labels onto current stock target review rows without changing weights."""
    if targets.empty:
        overlay = targets.copy()
        payload = {
            "target_overlay_status": "no_targets",
            "target_overlay_row_count": 0,
            "target_overlay_prefer_count": 0,
            "target_overlay_caution_count": 0,
            "target_overlay_mixed_count": 0,
            "target_overlay_unclassified_count": 0,
            "research_only": True,
            "broker_action": "none",
        }
        return overlay, payload

    lookup = _gate_lookup(gate)
    records: list[dict[str, Any]] = []
    for raw in targets.to_dict(orient="records"):
        record = dict(raw)
        labels = _target_group_labels(record)
        matches = [lookup[label] for label in labels if label in lookup]
        actions = [str(match.get("gate_action") or "") for match in matches if match.get("gate_action") not in (None, "")]
        target_action = _target_gate_action(actions)
        matched_labels = [str(match.get("group_label")) for match in matches if match.get("group_label") not in (None, "")]
        record.update(
            {
                "outcome_gate_target_action": target_action,
                "outcome_gate_research_priority": _target_research_priority(target_action),
                "outcome_gate_matches": "; ".join(matched_labels),
                "outcome_gate_actions": "; ".join(actions),
                "outcome_gate_note": "Research-only overlay; original target weights and broker actions are unchanged.",
                "outcome_gate_broker_action": "none",
                "outcome_gate_research_only": True,
            }
        )
        records.append(record)

    overlay = pd.DataFrame(records)
    action_counts = overlay["outcome_gate_target_action"].value_counts(dropna=False).to_dict()
    payload = {
        "target_overlay_status": "completed",
        "target_overlay_row_count": int(len(overlay)),
        "target_overlay_prefer_count": int(action_counts.get("prefer", 0)),
        "target_overlay_caution_count": int(action_counts.get("caution", 0)),
        "target_overlay_mixed_count": int(action_counts.get("mixed_review", 0)),
        "target_overlay_neutral_count": int(action_counts.get("neutral", 0)),
        "target_overlay_unclassified_count": int(action_counts.get("no_gate_match", 0)),
        "target_overlay_priority_summary": overlay["outcome_gate_research_priority"].value_counts(dropna=False).to_dict(),
        "research_only": True,
        "broker_action": "none",
        "note": "Target overlay is research-only. It does not alter portfolio_target_weight, target_action, or broker_action.",
    }
    return overlay, payload


def _normalised_shadow_weight(raw_weight: pd.Series, baseline_total: float) -> pd.Series:
    raw_total = float(pd.to_numeric(raw_weight, errors="coerce").fillna(0.0).sum())
    if raw_total <= 0 or baseline_total <= 0:
        return pd.Series([0.0] * len(raw_weight), index=raw_weight.index)
    return raw_weight / raw_total * baseline_total


def _shadow_watch_action(priority: str) -> str:
    if priority == "increase_research_priority":
        return "shadow_prefer_watch"
    if priority == "reduce_research_priority":
        return "shadow_reduce_watch"
    if priority == "manual_compare":
        return "shadow_manual_compare"
    if priority == "maintain_research_priority":
        return "shadow_hold_watch"
    return "shadow_unclassified"


def build_stock_outcome_shadow_selection(
    target_overlay: pd.DataFrame,
    relaxed_caution_multiplier: float = 0.5,
    relaxed_prefer_trial_weight: float = 0.20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build research-only shadow weight sets from the target overlay."""
    relaxed_caution_multiplier = max(0.0, min(float(relaxed_caution_multiplier), 1.0))
    relaxed_prefer_trial_weight = max(0.0, float(relaxed_prefer_trial_weight))
    if target_overlay.empty:
        shadow = target_overlay.copy()
        payload = {
            "shadow_selection_status": "no_targets",
            "shadow_selection_row_count": 0,
            "baseline_target_weight_total": 0.0,
            "shadow_reduce_caution_raw_weight_total": 0.0,
            "shadow_prefer_only_raw_weight_total": 0.0,
            "shadow_relaxed_opportunity_raw_weight_total": 0.0,
            "relaxed_caution_multiplier": relaxed_caution_multiplier,
            "relaxed_prefer_trial_weight": relaxed_prefer_trial_weight,
            "research_only": True,
            "broker_action": "none",
        }
        return shadow, payload

    shadow = target_overlay.copy()
    baseline = pd.to_numeric(shadow.get("portfolio_target_weight"), errors="coerce").fillna(0.0)
    priority = shadow.get("outcome_gate_research_priority", pd.Series(["unclassified"] * len(shadow), index=shadow.index)).astype(str)
    reduce_mask = priority == "reduce_research_priority"
    prefer_mask = priority == "increase_research_priority"

    shadow["baseline_target_weight"] = baseline
    shadow["shadow_reduce_caution_raw_weight"] = baseline.mask(reduce_mask, 0.0)
    shadow["shadow_prefer_only_raw_weight"] = baseline.where(prefer_mask, 0.0)
    baseline_total = float(baseline.sum())
    relaxed_base = baseline.mask(reduce_mask, baseline * relaxed_caution_multiplier)
    relaxed_freed_weight = max(baseline_total - float(relaxed_base.sum()), 0.0)
    relaxed_prefer_budget = min(relaxed_freed_weight, baseline_total * relaxed_prefer_trial_weight)
    relaxed_prefer_count = int(prefer_mask.sum())
    relaxed_prefer_add = relaxed_prefer_budget / relaxed_prefer_count if relaxed_prefer_count > 0 else 0.0
    shadow["shadow_relaxed_opportunity_raw_weight"] = relaxed_base + prefer_mask.astype(float) * relaxed_prefer_add
    shadow["shadow_reduce_caution_normalized_weight"] = _normalised_shadow_weight(
        pd.to_numeric(shadow["shadow_reduce_caution_raw_weight"], errors="coerce").fillna(0.0),
        baseline_total,
    )
    shadow["shadow_prefer_only_normalized_weight"] = _normalised_shadow_weight(
        pd.to_numeric(shadow["shadow_prefer_only_raw_weight"], errors="coerce").fillna(0.0),
        baseline_total,
    )
    shadow["shadow_relaxed_opportunity_normalized_weight"] = _normalised_shadow_weight(
        pd.to_numeric(shadow["shadow_relaxed_opportunity_raw_weight"], errors="coerce").fillna(0.0),
        baseline_total,
    )
    shadow["shadow_watch_action"] = priority.map(_shadow_watch_action)
    shadow["shadow_note"] = "Research-only shadow selection; no actual target or broker instruction is changed."

    payload = {
        "shadow_selection_status": "completed",
        "shadow_selection_row_count": int(len(shadow)),
        "baseline_target_weight_total": baseline_total,
        "shadow_reduce_caution_raw_weight_total": float(shadow["shadow_reduce_caution_raw_weight"].sum()),
        "shadow_prefer_only_raw_weight_total": float(shadow["shadow_prefer_only_raw_weight"].sum()),
        "shadow_relaxed_opportunity_raw_weight_total": float(shadow["shadow_relaxed_opportunity_raw_weight"].sum()),
        "shadow_reduce_caution_normalized_weight_total": float(shadow["shadow_reduce_caution_normalized_weight"].sum()),
        "shadow_prefer_only_normalized_weight_total": float(shadow["shadow_prefer_only_normalized_weight"].sum()),
        "shadow_relaxed_opportunity_normalized_weight_total": float(shadow["shadow_relaxed_opportunity_normalized_weight"].sum()),
        "relaxed_caution_multiplier": relaxed_caution_multiplier,
        "relaxed_prefer_trial_weight": relaxed_prefer_trial_weight,
        "relaxed_prefer_budget_used": float(relaxed_prefer_budget),
        "shadow_watch_action_summary": shadow["shadow_watch_action"].value_counts(dropna=False).to_dict(),
        "research_only": True,
        "broker_action": "none",
        "note": "Shadow selection is for future paper A/B tracking only; it does not alter live or paper targets.",
    }
    return shadow, payload


def _render_target_overlay_rows(target_overlay: pd.DataFrame | None) -> str:
    if target_overlay is None:
        return "| `N/A` | N/A | `N/A` | `N/A` | `N/A` | N/A | Not requested |"
    if target_overlay.empty:
        return "| `N/A` | N/A | `N/A` | `N/A` | `N/A` | N/A | No target rows |"
    display = target_overlay.head(40)
    return "\n".join(
        "| `{code}` | {name} | `{layer}` | `{action}` | `{priority}` | {weight} | {matches} |".format(
            code=str(row.get("code") or "").zfill(6),
            name=row.get("name") or "N/A",
            layer=row.get("layer") or "N/A",
            action=row.get("outcome_gate_target_action") or "N/A",
            priority=row.get("outcome_gate_research_priority") or "N/A",
            weight=_pct(row.get("portfolio_target_weight")),
            matches=row.get("outcome_gate_matches") or "N/A",
        )
        for _, row in display.iterrows()
    )


def _render_shadow_selection_rows(shadow_selection: pd.DataFrame | None) -> str:
    if shadow_selection is None:
        return "| `N/A` | N/A | `N/A` | N/A | N/A | N/A | Not requested |"
    if shadow_selection.empty:
        return "| `N/A` | N/A | `N/A` | N/A | N/A | N/A | No target rows |"
    display = shadow_selection.head(40)
    return "\n".join(
        "| `{code}` | {name} | `{action}` | {base} | {reduce_weight} | {prefer_weight} | {relaxed_weight} |".format(
            code=str(row.get("code") or "").zfill(6),
            name=row.get("name") or "N/A",
            action=row.get("shadow_watch_action") or "N/A",
            base=_pct(row.get("baseline_target_weight")),
            reduce_weight=_pct(row.get("shadow_reduce_caution_raw_weight")),
            prefer_weight=_pct(row.get("shadow_prefer_only_raw_weight")),
            relaxed_weight=_pct(row.get("shadow_relaxed_opportunity_raw_weight")),
        )
        for _, row in display.iterrows()
    )


def _render_gate_report(
    snapshot: dict[str, Any],
    gate: pd.DataFrame,
    target_overlay: pd.DataFrame | None = None,
    shadow_selection: pd.DataFrame | None = None,
) -> str:
    if gate.empty:
        rows = "| `N/A` | `N/A` | `N/A` | 0 | N/A | N/A | No ready group evidence |"
    else:
        rows = "\n".join(
            "| `{dimension}` | `{group}` | `{action}` | {count} | {avg} | {win} | {reason} |".format(
                dimension=row.get("dimension") or "N/A",
                group=row.get("group_label") or "N/A",
                action=row.get("gate_action") or "N/A",
                count=row.get("evaluable_count") if row.get("evaluable_count") is not None else 0,
                avg=_pct(row.get("avg_return")),
                win=_pct(row.get("win_rate")),
                reason=row.get("gate_reason") or "N/A",
            )
            for _, row in gate.iterrows()
        )

    return f"""# Stock Outcome Gate

Generated at: `{snapshot.get("generated_at")}`

This is a research-only gate from mature stock-target outcome samples. It does not connect to brokers, place orders, or change model targets.

## Summary

| Item | Value |
| --- | ---: |
| Decision | `{snapshot.get("decision")}` |
| Analysis status | `{snapshot.get("analysis_status")}` |
| Selected horizon | `{snapshot.get("selected_horizon") or "N/A"}` |
| Ready horizons | {", ".join(snapshot.get("ready_horizons") or []) or "None"} |
| Prefer / caution / neutral / insufficient | {snapshot.get("prefer_group_count", 0)} / {snapshot.get("caution_group_count", 0)} / {snapshot.get("neutral_group_count", 0)} / {snapshot.get("insufficient_group_count", 0)} |
| Target overlay status | `{snapshot.get("target_overlay_status", "not_requested")}` |
| Target overlay rows | {snapshot.get("target_overlay_row_count", 0)} |
| Shadow selection status | `{snapshot.get("shadow_selection_status", "not_requested")}` |
| Baseline / reduce-caution / prefer-only raw weight | {_pct(snapshot.get("baseline_target_weight_total"))} / {_pct(snapshot.get("shadow_reduce_caution_raw_weight_total"))} / {_pct(snapshot.get("shadow_prefer_only_raw_weight_total"))} |
| Relaxed-opportunity raw weight | {_pct(snapshot.get("shadow_relaxed_opportunity_raw_weight_total"))} |
| Relaxed parameters | caution x {_pct(snapshot.get("relaxed_caution_multiplier"))}, prefer trial cap {_pct(snapshot.get("relaxed_prefer_trial_weight"))} |
| Broker action | `{snapshot.get("broker_action")}` |

## Gate Rows

| Dimension | Group | Action | Evaluable | Avg return | Win rate | Reason |
| --- | --- | --- | ---: | ---: | ---: | --- |
{rows}

## Groups

- Prefer: {", ".join(f"`{item}`" for item in snapshot.get("prefer_groups") or []) or "None"}
- Caution: {", ".join(f"`{item}`" for item in snapshot.get("caution_groups") or []) or "None"}
- Neutral: {", ".join(f"`{item}`" for item in snapshot.get("neutral_groups") or []) or "None"}
- Insufficient: {", ".join(f"`{item}`" for item in snapshot.get("insufficient_groups") or []) or "None"}

## Target Overlay

| Code | Name | Layer | Gate action | Research priority | Target weight | Matched groups |
| --- | --- | --- | --- | --- | ---: | --- |
{_render_target_overlay_rows(target_overlay)}

## Shadow Selection

| Code | Name | Shadow action | Baseline weight | Reduce-caution raw weight | Prefer-only raw weight | Relaxed-opportunity raw weight |
| --- | --- | --- | ---: | ---: | ---: | ---: |
{_render_shadow_selection_rows(shadow_selection)}

## Source Files

- Outcome analysis CSV: `{snapshot.get("source_analysis_path") or "N/A"}`
- Outcome analysis JSON: `{snapshot.get("source_analysis_json_path") or "N/A"}`
- Outcome analysis report: `{snapshot.get("source_analysis_report_path") or "N/A"}`
- Outcome history CSV: `{snapshot.get("source_history_path") or "N/A"}`
- Target review CSV: `{snapshot.get("source_stock_target_review_path") or "N/A"}`
- Gate CSV: `stock_outcome_gate.csv`
- Target overlay CSV: `{snapshot.get("target_overlay_path") or "N/A"}`
- Shadow selection CSV: `{snapshot.get("shadow_selection_path") or "N/A"}`
- Snapshot JSON: `stock_outcome_gate_snapshot.json`
"""


def _column_or_default(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _safe_text(value: Any, default: str = "N/A") -> str:
    if value in (None, ""):
        return default
    if pd.isna(value):
        return default
    return str(value)


def _weak_trial_trade_date(shadow_selection: pd.DataFrame, pipeline_snapshot: dict[str, Any]) -> str:
    for key in ("paper_latest_date", "as_of_date", "trade_date", "latest_trade_date"):
        value = pipeline_snapshot.get(key)
        if value not in (None, ""):
            return str(value)
    if not shadow_selection.empty and "date" in shadow_selection.columns:
        value = shadow_selection["date"].dropna()
        if not value.empty:
            return str(value.iloc[0])
    return datetime.now().strftime("%Y-%m-%d")


def build_weak_sentiment_opportunity_trial(
    shadow_selection: pd.DataFrame,
    gate_snapshot: dict[str, Any],
    pipeline_snapshot: dict[str, Any] | None = None,
    ledger_record: dict[str, Any] | None = None,
    single_weight_cap: float = 0.05,
    total_weight_cap: float = 0.20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a Chinese research-only weak-sentiment opportunity trial table."""
    pipeline = pipeline_snapshot or {}
    ledger = ledger_record or {}
    single_cap = max(0.0, float(single_weight_cap))
    total_cap = max(0.0, float(total_weight_cap))
    trade_date = _weak_trial_trade_date(shadow_selection, pipeline)
    sentiment_state = str(pipeline.get("sentiment_state") or "")
    dashboard_posture = str(pipeline.get("dashboard_posture") or "")
    defensive_context = (
        sentiment_state in {"weak", "cold"} or dashboard_posture == "defensive_review_only"
    )

    records: list[dict[str, Any]] = []
    status = "completed"
    if shadow_selection.empty:
        status = "no_shadow_selection"
    elif not defensive_context:
        status = "skipped_not_defensive_context"
    else:
        priority = _column_or_default(shadow_selection, "outcome_gate_research_priority", "").astype(str)
        action = _column_or_default(shadow_selection, "outcome_gate_target_action", "").astype(str)
        watch_action = _column_or_default(shadow_selection, "shadow_watch_action", "").astype(str)
        raw_weight = pd.to_numeric(
            _column_or_default(shadow_selection, "shadow_relaxed_opportunity_raw_weight", 0.0),
            errors="coerce",
        ).fillna(0.0)
        baseline_weight = pd.to_numeric(
            _column_or_default(shadow_selection, "baseline_target_weight", 0.0),
            errors="coerce",
        ).fillna(0.0)
        candidate_mask = (
            (priority == "increase_research_priority")
            & (action == "prefer")
            & (watch_action == "shadow_prefer_watch")
            & (raw_weight > baseline_weight)
        )
        candidates = shadow_selection.loc[candidate_mask].copy()
        if candidates.empty:
            status = "no_prefer_opportunity_candidates"
        else:
            candidates["_raw_trial_weight"] = raw_weight.loc[candidate_mask]
            candidates["_baseline_weight"] = baseline_weight.loc[candidate_mask]
            candidates["_priority_score"] = pd.to_numeric(
                _column_or_default(candidates, "review_priority_score", 0.0),
                errors="coerce",
            ).fillna(0.0)
            candidates["_code_sort"] = _column_or_default(candidates, "code", "").astype(str).str.zfill(6)
            candidates = candidates.sort_values(
                ["_priority_score", "_raw_trial_weight", "_code_sort"],
                ascending=[False, False, True],
            )
            remaining = total_cap
            for _, row in candidates.iterrows():
                if remaining <= 0:
                    break
                raw = _as_float(row.get("_raw_trial_weight")) or 0.0
                baseline = _as_float(row.get("_baseline_weight")) or 0.0
                incremental = max(raw - baseline, 0.0)
                trial_weight = min(single_cap, incremental, remaining)
                if trial_weight <= 0:
                    continue
                remaining -= trial_weight
                records.append(
                    {
                        "日期": trade_date,
                        "代码": str(row.get("code") or "").zfill(6),
                        "名称": _safe_text(row.get("name")),
                        "层级": _safe_text(row.get("layer")),
                        "正式目标权重": _pct(baseline),
                        "影子试运行权重": _pct(trial_weight),
                        "影子动作": _safe_text(row.get("shadow_watch_action")),
                        "研究优先级": _safe_text(row.get("outcome_gate_research_priority")),
                        "匹配依据": _safe_text(row.get("outcome_gate_matches")),
                        "备注": "研究-only影子跟踪；不改变正式目标，不连接券商，不自动下单。",
                    }
                )
            if not records:
                status = "no_positive_trial_weight_after_caps"

    trial = pd.DataFrame(
        records,
        columns=[
            "日期",
            "代码",
            "名称",
            "层级",
            "正式目标权重",
            "影子试运行权重",
            "影子动作",
            "研究优先级",
            "匹配依据",
            "备注",
        ],
    )
    assigned_total = sum((_as_float(str(value).replace("%", "")) or 0.0) / 100.0 for value in trial.get("影子试运行权重", []))
    payload = {
        "weak_sentiment_trial_status": status,
        "weak_sentiment_trial_row_count": int(len(trial)),
        "weak_sentiment_trial_trade_date": trade_date,
        "weak_sentiment_trial_assigned_weight_total": float(assigned_total),
        "weak_sentiment_trial_single_weight_cap": single_cap,
        "weak_sentiment_trial_total_weight_cap": total_cap,
        "sentiment_state": pipeline.get("sentiment_state"),
        "dashboard_posture": pipeline.get("dashboard_posture"),
        "live_preflight_decision": pipeline.get("live_preflight_decision"),
        "paper_latest_regime": pipeline.get("paper_latest_regime") or ledger.get("effective_regime"),
        "paper_latest_core_weight": pipeline.get("paper_latest_core_weight") or ledger.get("core_target_weight"),
        "paper_latest_satellite_weight": pipeline.get("paper_latest_satellite_weight") or ledger.get("satellite_target_weight"),
        "paper_latest_cash_weight": pipeline.get("paper_latest_cash_weight") or ledger.get("cash_target_weight"),
        "paper_latest_effective_reason": ledger.get("effective_reason") or pipeline.get("paper_latest_reason"),
        "benchmark_drop": ledger.get("benchmark_drop"),
        "trigger_freshness_status": pipeline.get("trigger_freshness_status"),
        "stock_target_review_action_count": pipeline.get("paper_stock_target_review_action_count"),
        "selected_horizon": gate_snapshot.get("selected_horizon"),
        "prefer_groups": gate_snapshot.get("prefer_groups") or [],
        "caution_groups": gate_snapshot.get("caution_groups") or [],
        "baseline_target_weight_total": gate_snapshot.get("baseline_target_weight_total"),
        "shadow_relaxed_opportunity_raw_weight_total": gate_snapshot.get("shadow_relaxed_opportunity_raw_weight_total"),
        "relaxed_prefer_trial_weight": gate_snapshot.get("relaxed_prefer_trial_weight"),
        "research_only": True,
        "broker_action": "none",
        "note": "Weak-sentiment opportunity trial is research-only and does not change portfolio targets or broker instructions.",
    }
    return trial, payload


def _gate_evidence_rows(snapshot: dict[str, Any]) -> str:
    rows = snapshot.get("gate_rows") or []
    selected = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("gate_action") in {"prefer", "caution"}
    ][:10]
    if not selected:
        return "| N/A | 0 | N/A | N/A | N/A |"
    return "\n".join(
        "| {group} | {count} | {avg} | {win} | {action} |".format(
            group=row.get("group_label") or "N/A",
            count=row.get("evaluable_count") if row.get("evaluable_count") is not None else 0,
            avg=_pct(row.get("avg_return")),
            win=_pct(row.get("win_rate")),
            action=row.get("gate_action") or "N/A",
        )
        for row in selected
    )


def _weak_trial_candidate_rows(trial: pd.DataFrame) -> str:
    if trial.empty:
        return "| N/A | N/A | N/A | N/A | N/A | N/A |"
    return "\n".join(
        "| {code} | {name} | {layer} | {base} | {trial_weight} | {match} |".format(
            code=row.get("代码") or "N/A",
            name=row.get("名称") or "N/A",
            layer=row.get("层级") or "N/A",
            base=row.get("正式目标权重") or "N/A",
            trial_weight=row.get("影子试运行权重") or "N/A",
            match=row.get("匹配依据") or "N/A",
        )
        for _, row in trial.iterrows()
    )


def _render_weak_sentiment_opportunity_report(
    trial: pd.DataFrame,
    payload: dict[str, Any],
    gate_snapshot: dict[str, Any],
) -> str:
    status = payload.get("weak_sentiment_trial_status")
    if status == "completed" and not trial.empty:
        conclusion = (
            "当前不能直接解除防御态；但成熟复盘样本显示 prefer 组有受限观察价值，"
            "因此只生成卫星影子试运行名单。"
        )
    elif status == "skipped_not_defensive_context":
        conclusion = "当前不处于弱情绪防御上下文，本报告只保留审计，不生成新增影子名单。"
    else:
        conclusion = "当前没有满足 prefer、增量权重和风控上限的弱情绪机会候选。"

    return f"""# 弱情绪受限机会试运行

生成日期：{payload.get("weak_sentiment_trial_trade_date")}

本报告只用于研究和影子跟踪，不连接券商、不自动下单、不改变正式纸面账户目标权重。

## 当前状态

| 项目 | 数值 |
| --- | ---: |
| 试运行状态 | `{status}` |
| 市场情绪 | `{payload.get("sentiment_state") or "N/A"}` |
| Dashboard 姿态 | `{payload.get("dashboard_posture") or "N/A"}` |
| 实盘前置结论 | `{payload.get("live_preflight_decision") or "N/A"}` |
| 当前纸面账户状态 | `{payload.get("paper_latest_regime") or "N/A"}` |
| 风险原因 | `{payload.get("paper_latest_effective_reason") or "N/A"}` |
| 当前核心仓位 | {_pct(payload.get("paper_latest_core_weight"))} |
| 当前卫星仓位 | {_pct(payload.get("paper_latest_satellite_weight"))} |
| 当前现金仓位 | {_pct(payload.get("paper_latest_cash_weight"))} |
| 基准相对均线跌幅 | {_pct(payload.get("benchmark_drop"))} |
| 触发器状态 | `{payload.get("trigger_freshness_status") or "N/A"}` |
| 人工复核动作队列 | {payload.get("stock_target_review_action_count", "N/A")} |

## 结论

{conclusion}

## 证据

| 组别 | {payload.get("selected_horizon") or "N/A"}样本数 | 平均收益 | 胜率 | 结论 |
| --- | ---: | ---: | ---: | --- |
{_gate_evidence_rows(gate_snapshot)}

## 影子方案

| 项目 | 权重 |
| --- | ---: |
| 原始正式目标合计 | {_pct(payload.get("baseline_target_weight_total"))} |
| 弱组降权后机会仓位 | {_pct(payload.get("shadow_relaxed_opportunity_raw_weight_total"))} |
| prefer 试运行总上限 | {_pct(payload.get("weak_sentiment_trial_total_weight_cap"))} |
| 单票 prefer 试运行上限 | {_pct(payload.get("weak_sentiment_trial_single_weight_cap"))} |
| 本次已分配影子权重 | {_pct(payload.get("weak_sentiment_trial_assigned_weight_total"))} |

## 受限卫星候选

| 代码 | 名称 | 层级 | 正式目标权重 | 影子试运行权重 | 依据 |
| --- | --- | --- | ---: | ---: | --- |
{_weak_trial_candidate_rows(trial)}

## 风控边界

- 只做影子跟踪，不进入券商下单。
- 不放宽 trigger 组；trigger 若仍为 caution，继续只看不买。
- 不整体提高 core 权重；core 若仍为 caution，不提升组合风险敞口。
- 若情绪从 weak 降为 cold，影子新增候选暂停。
- 若 1日和5日实际跟踪继续跑赢，再考虑把规则纳入正式纸面 A/B。

## 下一步

1. 每日刷新后自动更新本报告和结果跟踪表。
2. 重点观察本次候选的 1日、5日、10日结果。
3. 只有在结果继续验证且风控状态改善后，再讨论是否放宽正式仓位。
"""


def run_stock_outcome_gate(
    outcome_analysis_path: str | Path,
    stock_target_review_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    primary_horizon: str = DEFAULT_PRIMARY_HORIZON,
    min_group_evaluable: int = 5,
    prefer_min_avg_return: float = 0.03,
    prefer_min_win_rate: float = 0.50,
    caution_max_avg_return: float = -0.02,
    caution_max_win_rate: float = 0.35,
    relaxed_caution_multiplier: float = 0.5,
    relaxed_prefer_trial_weight: float = 0.20,
    pipeline_snapshot_path: str | Path | None = None,
    paper_ledger_path: str | Path | None = None,
    weak_sentiment_opportunity_output_dir: str | Path | None = None,
    weak_sentiment_single_weight_cap: float = 0.05,
    weak_sentiment_total_weight_cap: float = 0.20,
    outcome_payload_override: dict[str, Any] | None = None,
    stock_target_review_frame: pd.DataFrame | None = None,
    pipeline_snapshot_override: dict[str, Any] | None = None,
    paper_ledger_record_override: dict[str, Any] | None = None,
) -> StockOutcomeGateResult:
    outcome_path = Path(outcome_analysis_path)
    outcome_payload = outcome_payload_override if outcome_payload_override is not None else _load_json(outcome_path)
    gate, snapshot = build_stock_outcome_gate(
        outcome_payload,
        primary_horizon=primary_horizon,
        min_group_evaluable=min_group_evaluable,
        prefer_min_avg_return=prefer_min_avg_return,
        prefer_min_win_rate=prefer_min_win_rate,
        caution_max_avg_return=caution_max_avg_return,
        caution_max_win_rate=caution_max_win_rate,
    )
    snapshot = {
        **snapshot,
        "source_outcome_analysis_path": str(outcome_path),
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate_path = out_dir / "stock_outcome_gate.csv"
    target_overlay_path = out_dir / "stock_outcome_gate_target_overlay.csv"
    shadow_selection_path = out_dir / "stock_outcome_gate_shadow_selection.csv"
    snapshot_path = out_dir / "stock_outcome_gate_snapshot.json"
    report_path = out_dir / "stock_outcome_gate.md"

    gate.to_csv(gate_path, index=False, encoding="utf-8-sig")

    target_overlay: pd.DataFrame | None = None
    shadow_selection: pd.DataFrame | None = None
    written_target_overlay_path: Path | None = None
    written_shadow_selection_path: Path | None = None
    weak_trial: pd.DataFrame | None = None
    weak_trial_csv_path: Path | None = None
    weak_trial_report_path: Path | None = None
    if stock_target_review_path not in (None, "") or stock_target_review_frame is not None:
        resolved_targets = Path(stock_target_review_path) if stock_target_review_path not in (None, "") else None
        if stock_target_review_frame is not None:
            targets = stock_target_review_frame.copy()
        else:
            if resolved_targets is None or not resolved_targets.exists():
                raise FileNotFoundError(f"Missing stock target review CSV: {resolved_targets}")
            targets = pd.read_csv(resolved_targets)
        target_overlay, overlay_payload = build_stock_outcome_target_overlay(targets, gate)
        shadow_selection, shadow_payload = build_stock_outcome_shadow_selection(
            target_overlay,
            relaxed_caution_multiplier=relaxed_caution_multiplier,
            relaxed_prefer_trial_weight=relaxed_prefer_trial_weight,
        )
        target_overlay.to_csv(target_overlay_path, index=False, encoding="utf-8-sig")
        shadow_selection.to_csv(shadow_selection_path, index=False, encoding="utf-8-sig")
        written_target_overlay_path = target_overlay_path
        written_shadow_selection_path = shadow_selection_path
        snapshot.update(
            {
                **overlay_payload,
                **shadow_payload,
                "source_stock_target_review_path": str(resolved_targets) if resolved_targets is not None else None,
                "target_overlay_path": str(target_overlay_path),
                "shadow_selection_path": str(shadow_selection_path),
            }
        )

        if weak_sentiment_opportunity_output_dir not in (None, ""):
            weak_out_dir = Path(weak_sentiment_opportunity_output_dir)
            weak_out_dir.mkdir(parents=True, exist_ok=True)
            pipeline_snapshot = (
                pipeline_snapshot_override
                if pipeline_snapshot_override is not None
                else _load_optional_json(pipeline_snapshot_path)
            )
            ledger_record = (
                paper_ledger_record_override
                if paper_ledger_record_override is not None
                else _load_latest_csv_record(paper_ledger_path)
            )
            weak_trial, weak_payload = build_weak_sentiment_opportunity_trial(
                shadow_selection,
                snapshot,
                pipeline_snapshot=pipeline_snapshot,
                ledger_record=ledger_record,
                single_weight_cap=weak_sentiment_single_weight_cap,
                total_weight_cap=weak_sentiment_total_weight_cap,
            )
            weak_trial_csv_path = weak_out_dir / "弱情绪受限机会试运行.csv"
            weak_trial_report_path = weak_out_dir / "弱情绪受限机会试运行.md"
            weak_trial.to_csv(weak_trial_csv_path, index=False, encoding="utf-8-sig")
            weak_trial_report_path.write_text(
                _render_weak_sentiment_opportunity_report(weak_trial, weak_payload, snapshot),
                encoding="utf-8",
            )
            snapshot.update(
                {
                    **weak_payload,
                    "weak_sentiment_trial_output_dir": str(weak_out_dir),
                    "weak_sentiment_trial_csv_path": str(weak_trial_csv_path),
                    "weak_sentiment_trial_report_path": str(weak_trial_report_path),
                    "weak_sentiment_trial_pipeline_snapshot_path": str(pipeline_snapshot_path)
                    if pipeline_snapshot_path
                    else None,
                    "weak_sentiment_trial_paper_ledger_path": str(paper_ledger_path)
                    if paper_ledger_path
                    else None,
                }
            )
    else:
        snapshot.update(
            {
                "target_overlay_status": "not_requested",
                "target_overlay_row_count": 0,
                "shadow_selection_status": "not_requested",
                "shadow_selection_row_count": 0,
                "source_stock_target_review_path": None,
                "target_overlay_path": None,
                "shadow_selection_path": None,
            }
        )
        if weak_sentiment_opportunity_output_dir not in (None, ""):
            snapshot.update(
                {
                    "weak_sentiment_trial_status": "not_requested_without_target_review",
                    "weak_sentiment_trial_row_count": 0,
                    "weak_sentiment_trial_csv_path": None,
                    "weak_sentiment_trial_report_path": None,
                }
            )

    snapshot_path.write_text(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_gate_report(snapshot, gate, target_overlay, shadow_selection), encoding="utf-8")

    return StockOutcomeGateResult(
        output_dir=out_dir,
        gate_path=gate_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        gate=gate,
        target_overlay_path=written_target_overlay_path,
        target_overlay=target_overlay,
        shadow_selection_path=written_shadow_selection_path,
        shadow_selection=shadow_selection,
        weak_sentiment_trial_csv_path=weak_trial_csv_path,
        weak_sentiment_trial_report_path=weak_trial_report_path,
        weak_sentiment_trial=weak_trial,
    )
