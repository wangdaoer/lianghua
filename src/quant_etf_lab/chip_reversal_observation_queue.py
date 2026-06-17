"""Observation-only queue for chip-reversal research candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_EVENTS_PATH = Path("outputs/research/chip_reversal_lab_latest/chip_reversal_events.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/chip_reversal_observation_queue_latest")
FORWARD_LABEL_PREFIXES = ("return_", "trade_return_")
FORWARD_LABEL_COLUMNS = {"next_open", "open_gap_next"}


@dataclass(frozen=True)
class ChipReversalObservationQueueResult:
    output_dir: Path
    queue_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    queue: pd.DataFrame


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = Path(path) if path is not None else default
    return raw if raw.is_absolute() else project_root / raw


def _clean_code(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[-6:].zfill(6) if digits else text


def _forward_label_columns(events: pd.DataFrame) -> list[str]:
    columns = []
    for column in events.columns:
        if column in FORWARD_LABEL_COLUMNS or column.startswith(FORWARD_LABEL_PREFIXES):
            columns.append(column)
    return columns


def _first_existing_column(frame: pd.DataFrame, prefix: str) -> str | None:
    return next((column for column in frame.columns if column.startswith(prefix)), None)


def _as_numeric(frame: pd.DataFrame, column: str | None, default: float = 0.0) -> pd.Series:
    if column is None or column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not pd.notna(number):
        return "N/A"
    return f"{number * 100:.2f}%"


def _render_report(snapshot: dict[str, Any], queue: pd.DataFrame) -> str:
    if queue.empty:
        rows = "| N/A | N/A | N/A | N/A | N/A |"
    else:
        rows = "\n".join(
            "| {rank} | `{code}` | {name} | {score:.2f} | `{posture}` |".format(
                rank=int(row.get("observation_rank") or 0),
                code=row.get("code"),
                name=row.get("name"),
                score=float(row.get("priority_score") or 0.0),
                posture=row.get("watch_posture"),
            )
            for _, row in queue.iterrows()
        )
    return f"""# Chip Reversal Observation Queue

Generated at: `{snapshot.get("generated_at")}`

This is an observation-only queue for research candidates. It does not write target weights, quantities, broker orders, or live instructions.

## Snapshot

| Item | Value |
| --- | ---: |
| Status | `{snapshot.get("status")}` |
| As-of date | `{snapshot.get("as_of_date")}` |
| Candidates | {snapshot.get("candidate_count", 0)} |
| Source events | {snapshot.get("source_event_count", 0)} |
| Forward labels ignored | `{snapshot.get("forward_label_columns_ignored")}` |
| Broker action | `{snapshot.get("broker_action")}` |

## Queue

| Rank | Code | Name | Priority | Posture |
| ---: | --- | --- | ---: | --- |
{rows}

## Timing Contract

- Ranking uses selector fields only: chip reversal score, drawdown depth, cost gap depth, amount, and optional theme diagnostics.
- Forward labels such as next open and future returns are ignored for ranking.
- `watch_only` means observation and review, not a buy/sell instruction.

## Files

- Queue CSV: `{snapshot.get("queue_path")}`
- Snapshot JSON: `{snapshot.get("snapshot_path")}`
"""


def build_chip_reversal_observation_queue(
    events: pd.DataFrame,
    as_of_date: str | None = None,
    max_candidates: int = 20,
    min_priority_score: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if events.empty:
        snapshot = {
            "status": "no_events",
            "as_of_date": as_of_date,
            "source_event_count": 0,
            "candidate_count": 0,
            "forward_label_columns_ignored": [],
            "broker_action": "none",
            "research_only": True,
        }
        return pd.DataFrame(), snapshot

    data = events.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["code"] = data["code"].map(_clean_code) if "code" in data.columns else ""
    data = data.dropna(subset=["date", "code"]).copy()
    selected_date = str(as_of_date) if as_of_date else str(data["date"].max())
    source = data[data["date"] == selected_date].copy()
    forward_columns = _forward_label_columns(data)

    if source.empty:
        snapshot = {
            "status": "no_candidates",
            "as_of_date": selected_date,
            "source_event_count": int(len(data)),
            "candidate_count": 0,
            "forward_label_columns_ignored": forward_columns,
            "broker_action": "none",
            "research_only": True,
        }
        return pd.DataFrame(), snapshot

    drawdown_column = _first_existing_column(source, "drawdown_")
    cost_gap_column = _first_existing_column(source, "cost_gap_")
    score = _as_numeric(source, "chip_reversal_score")
    drawdown_depth = (-_as_numeric(source, drawdown_column)).clip(lower=0.0)
    cost_gap_depth = (-_as_numeric(source, cost_gap_column)).clip(lower=0.0)
    amount_yi = _as_numeric(source, "amount_yi")
    liquidity_component = amount_yi.clip(lower=0.0, upper=20.0) / 20.0
    theme_component = pd.Series(0.0, index=source.index)
    if "theme_state" in source.columns:
        theme_component = source["theme_state"].map({"second_activation": 0.08, "healthy": 0.05, "neutral": 0.02}).fillna(0.0)

    source["priority_score"] = (
        score * 100.0
        + drawdown_depth * 45.0
        + cost_gap_depth * 25.0
        + liquidity_component * 10.0
        + theme_component * 100.0
    )
    source = source[source["priority_score"] >= float(min_priority_score)].copy()
    source = source.sort_values(["priority_score", "chip_reversal_score", "amount_yi"], ascending=[False, False, False])
    if max_candidates and max_candidates > 0:
        source = source.head(int(max_candidates)).copy()
    source["observation_rank"] = range(1, len(source) + 1)
    source["source_sleeve"] = "chip_reversal_daily_proxy"
    source["watch_posture"] = "watch_only"
    source["review_status"] = "pending_observation"
    source["paper_queue_status"] = "observation_only_not_order"
    source["recommended_review"] = (
        "Observe next-session open strength, intraday recovery, and 1D/2D follow-up; no broker order is generated."
    )
    source["invalidation_condition"] = "Next session opens weak and fails to reclaim signal-day close."
    source["broker_action"] = "none"
    source["research_only"] = True
    source["forward_label_columns_ignored"] = ",".join(forward_columns)

    keep = [
        "observation_rank",
        "date",
        "code",
        "name",
        "board",
        "source_sleeve",
        "watch_posture",
        "review_status",
        "paper_queue_status",
        "priority_score",
        "score_bucket",
        "chip_reversal_score",
        drawdown_column,
        cost_gap_column,
        "amount_yi",
        "close",
        "theme_group",
        "theme_state",
        "recommended_review",
        "invalidation_condition",
        "forward_label_columns_ignored",
        "broker_action",
        "research_only",
    ]
    keep = [column for column in keep if column is not None and column in source.columns]
    queue = source.reindex(columns=keep).reset_index(drop=True)
    if not queue.empty:
        queue["research_only"] = queue["research_only"].astype(object)
    snapshot = {
        "status": "ok" if not queue.empty else "no_candidates",
        "as_of_date": selected_date,
        "source_event_count": int(len(data)),
        "candidate_count": int(len(queue)),
        "max_candidates": int(max_candidates),
        "min_priority_score": float(min_priority_score),
        "forward_label_columns_ignored": forward_columns,
        "ranking_uses_forward_labels": False,
        "broker_action": "none",
        "research_only": True,
    }
    return queue, snapshot


def run_chip_reversal_observation_queue(
    project_root: str | Path = Path("."),
    events_path: str | Path | None = DEFAULT_EVENTS_PATH,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    as_of_date: str | None = None,
    max_candidates: int = 20,
    min_priority_score: float = 0.0,
) -> ChipReversalObservationQueueResult:
    root = Path(project_root).resolve()
    resolved_events = _resolve(root, events_path, DEFAULT_EVENTS_PATH)
    resolved_output = _resolve(root, output_dir, DEFAULT_OUTPUT_DIR)
    events = pd.read_csv(resolved_events, dtype={"code": str})
    queue, snapshot = build_chip_reversal_observation_queue(
        events,
        as_of_date=as_of_date,
        max_candidates=max_candidates,
        min_priority_score=min_priority_score,
    )
    resolved_output.mkdir(parents=True, exist_ok=True)
    queue_path = resolved_output / "chip_reversal_observation_queue.csv"
    snapshot_path = resolved_output / "chip_reversal_observation_queue_snapshot.json"
    report_path = resolved_output / "chip_reversal_observation_queue.md"
    queue.to_csv(queue_path, index=False, encoding="utf-8-sig")
    snapshot = {
        **snapshot,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "events_path": str(resolved_events),
        "output_dir": str(resolved_output),
        "queue_path": str(queue_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, queue), encoding="utf-8")
    return ChipReversalObservationQueueResult(
        output_dir=resolved_output,
        queue_path=queue_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        queue=queue,
    )
