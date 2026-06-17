"""Point-in-time audit for the chip-reversal daily proxy.

The audit intentionally separates selector columns from future label columns.
It recomputes each sampled signal using data truncated at the signal date and
checks that the event still exists with the same selector values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .chip_reversal_lab import (
    DEFAULT_DATA_DIR,
    build_chip_reversal_events,
    _clean_code,
    _csv_paths,
    _read_price_csv,
    _resolve,
    _validate_score_bucket,
)


DEFAULT_OUTPUT_DIR = Path("outputs/research/chip_reversal_pit_audit_latest")
FUTURE_LABEL_BASE_COLUMNS = ["next_open", "open_gap_next"]
SELECTOR_BASE_COLUMNS = [
    "chip_reversal_score",
    "close",
    "amount",
    "amount_yi",
    "score_bucket",
]


@dataclass(frozen=True)
class ChipReversalPitAuditResult:
    output_dir: Path
    audit_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    audit: pd.DataFrame


def _parse_date(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _selector_columns(events: pd.DataFrame) -> list[str]:
    dynamic = [column for column in events.columns if column.startswith(("drawdown_", "prior_high_", "ma", "cost_gap_"))]
    return [column for column in [*SELECTOR_BASE_COLUMNS, *dynamic] if column in events.columns]


def _future_label_columns(events: pd.DataFrame, horizons: Iterable[int]) -> list[str]:
    columns = [column for column in FUTURE_LABEL_BASE_COLUMNS if column in events.columns]
    for horizon in horizons:
        for prefix in ["return", "trade_return"]:
            column = f"{prefix}_{int(horizon)}d"
            if column in events.columns:
                columns.append(column)
    return columns


def _event_key(frame: pd.DataFrame) -> pd.Series:
    return frame["code"].astype(str).map(_clean_code) + "|" + pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")


def _values_match(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)
    if not pd.notna(left_number) and not pd.notna(right_number):
        return True
    return abs(left_number - right_number) <= tolerance


def audit_chip_reversal_events(
    prices: pd.DataFrame,
    horizons: Iterable[int] = (1, 2),
    drawdown_window: int = 20,
    cost_window: int = 20,
    min_drawdown_pct: float = 12.0,
    min_score: float = 0.08,
    score_bucket: str = "all",
    board_scope: str = "main_chinext",
    start_date: str | None = None,
    end_date: str | None = None,
    max_events: int | None = None,
) -> dict[str, Any]:
    horizons_list = [int(horizon) for horizon in horizons]
    normalized_score_bucket = _validate_score_bucket(score_bucket)
    full_events = build_chip_reversal_events(
        prices,
        horizons=horizons_list,
        drawdown_window=drawdown_window,
        cost_window=cost_window,
        min_drawdown_pct=min_drawdown_pct,
        min_score=min_score,
        board_scope=board_scope,
    )
    if full_events.empty:
        return {
            "status": "no_events",
            "audited_event_count": 0,
            "pit_mismatch_count": 0,
            "future_label_column_count": 0,
            "future_label_columns": [],
            "selector_column_count": 0,
            "selector_columns": [],
            "audit_rows": [],
        }
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    events = full_events.copy()
    if normalized_score_bucket != "all":
        events = events[events["score_bucket"].astype(str) == normalized_score_bucket].copy()
    event_dates = pd.to_datetime(events["date"], errors="coerce")
    if start is not None:
        events = events[event_dates >= start].copy()
        event_dates = pd.to_datetime(events["date"], errors="coerce")
    if end is not None:
        events = events[event_dates <= end].copy()
    if max_events is not None and max_events > 0:
        events = events.head(int(max_events)).copy()

    selector_columns = _selector_columns(full_events)
    future_columns = _future_label_columns(full_events, horizons_list)
    rows: list[dict[str, Any]] = []
    prices_sorted = prices.copy()
    prices_sorted["date"] = pd.to_datetime(prices_sorted["date"], errors="coerce")
    prices_sorted = prices_sorted.sort_values("date")
    for _, event in events.iterrows():
        signal_date = pd.Timestamp(event["date"])
        truncated = prices_sorted[prices_sorted["date"] <= signal_date].copy()
        recomputed = build_chip_reversal_events(
            truncated,
            horizons=horizons_list,
            drawdown_window=drawdown_window,
            cost_window=cost_window,
            min_drawdown_pct=min_drawdown_pct,
            min_score=min_score,
            board_scope=board_scope,
            require_mature=False,
        )
        matching = recomputed[
            (recomputed.get("date", pd.Series(dtype=str)).astype(str) == str(event["date"]))
            & (recomputed.get("code", pd.Series(dtype=str)).astype(str).map(_clean_code) == _clean_code(event["code"]))
        ]
        mismatched_columns: list[str] = []
        if matching.empty:
            pit_status = "missing_in_truncated_history"
        else:
            pit_status = "pass"
            recomputed_event = matching.iloc[0]
            for column in selector_columns:
                if column in recomputed_event.index and column in event.index:
                    if not _values_match(event[column], recomputed_event[column]):
                        mismatched_columns.append(column)
            if mismatched_columns:
                pit_status = "selector_mismatch"
        rows.append(
            {
                "date": str(event["date"]),
                "code": _clean_code(event["code"]),
                "score_bucket": str(event.get("score_bucket", "")),
                "pit_status": pit_status,
                "mismatched_selector_columns": ",".join(mismatched_columns),
                "selector_columns": ",".join(selector_columns),
                "future_label_columns": ",".join(future_columns),
                "future_label_column_count": len(future_columns),
                "broker_action": "none",
                "research_only": True,
            }
        )
    mismatch_count = sum(1 for row in rows if row["pit_status"] != "pass")
    return {
        "status": "pass" if mismatch_count == 0 else "fail",
        "audited_event_count": len(rows),
        "pit_mismatch_count": int(mismatch_count),
        "future_label_column_count": len(future_columns),
        "future_label_columns": future_columns,
        "selector_column_count": len(selector_columns),
        "selector_columns": selector_columns,
        "audit_rows": rows,
    }


def _render_report(snapshot: dict[str, Any]) -> str:
    return f"""# Chip Reversal PIT Audit

Generated at: `{snapshot.get("generated_at")}`

This is a research-only point-in-time audit. It does not connect to brokers, place orders, or provide investment advice.

## Result

| Item | Value |
| --- | ---: |
| Status | `{snapshot.get("status")}` |
| Audited events | {snapshot.get("audited_event_count", 0)} |
| PIT mismatches | {snapshot.get("pit_mismatch_count", 0)} |
| Future label columns | `{snapshot.get("future_label_columns")}` |
| Selector columns | `{snapshot.get("selector_columns")}` |
| Broker action | `{snapshot.get("broker_action")}` |

## Timing Contract

- Signal generation is audited by recomputing candidates with history truncated at the signal date.
- `next_open`, `open_gap_next`, `return_*d`, and `trade_return_*d` are labels/forward execution assumptions, not selector columns.
- Any mismatch means this sleeve stays research-only until the event selection code is fixed.

## Files

- Audit CSV: `{snapshot.get("audit_path")}`
- Snapshot JSON: `{snapshot.get("snapshot_path")}`
"""


def run_chip_reversal_pit_audit(
    project_root: str | Path = Path("."),
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    horizons: Iterable[int] = (1, 2),
    drawdown_window: int = 20,
    cost_window: int = 20,
    min_drawdown_pct: float = 12.0,
    min_score: float = 0.08,
    score_bucket: str = "all",
    board_scope: str = "main_chinext",
    max_symbols: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_events: int = 2000,
    recursive: bool = False,
) -> ChipReversalPitAuditResult:
    root = Path(project_root).resolve()
    resolved_data = _resolve(root, data_dir, DEFAULT_DATA_DIR)
    resolved_output = _resolve(root, output_dir, DEFAULT_OUTPUT_DIR)
    horizons_list = [int(horizon) for horizon in horizons]
    normalized_score_bucket = _validate_score_bucket(score_bucket)
    paths = _csv_paths(resolved_data, recursive=recursive)
    if max_symbols is not None and max_symbols > 0:
        paths = paths[: int(max_symbols)]

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    remaining = int(max_events) if max_events and max_events > 0 else None
    for path in paths:
        if remaining is not None and remaining <= 0:
            break
        try:
            prices = _read_price_csv(path)
            audit = audit_chip_reversal_events(
                prices,
                horizons=horizons_list,
                drawdown_window=drawdown_window,
                cost_window=cost_window,
                min_drawdown_pct=min_drawdown_pct,
                min_score=min_score,
                score_bucket=normalized_score_bucket,
                board_scope=board_scope,
                start_date=start_date,
                end_date=end_date,
                max_events=remaining,
            )
            symbol_rows = audit.get("audit_rows", [])
            if normalized_score_bucket != "all" and symbol_rows:
                # The PIT audit validates selector timing. Bucket filtering is included in
                # the generated selector columns, but row reduction is handled by the lab.
                pass
            rows.extend(symbol_rows)
            if remaining is not None:
                remaining -= len(symbol_rows)
        except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            failures.append({"path": str(path), "error": str(error)})
    audit_frame = pd.DataFrame(rows)
    mismatch_count = int((audit_frame["pit_status"] != "pass").sum()) if not audit_frame.empty else 0
    future_columns = []
    selector_columns = []
    if not audit_frame.empty:
        future_columns = str(audit_frame["future_label_columns"].iloc[0]).split(",")
        selector_columns = str(audit_frame["selector_columns"].iloc[0]).split(",")

    resolved_output.mkdir(parents=True, exist_ok=True)
    audit_path = resolved_output / "chip_reversal_pit_audit.csv"
    snapshot_path = resolved_output / "chip_reversal_pit_audit_snapshot.json"
    report_path = resolved_output / "chip_reversal_pit_audit.md"
    audit_frame.to_csv(audit_path, index=False, encoding="utf-8-sig")
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "pass" if mismatch_count == 0 else "fail",
        "data_dir": str(resolved_data),
        "output_dir": str(resolved_output),
        "file_count": len(paths),
        "audited_event_count": int(len(audit_frame)),
        "pit_mismatch_count": mismatch_count,
        "failure_count": len(failures),
        "failures": failures[:20],
        "horizons": horizons_list,
        "drawdown_window": int(drawdown_window),
        "cost_window": int(cost_window),
        "min_drawdown_pct": float(min_drawdown_pct),
        "min_score": float(min_score),
        "score_bucket_filter": normalized_score_bucket,
        "board_scope": board_scope,
        "start_date": start_date,
        "end_date": end_date,
        "max_events": int(max_events),
        "future_label_columns": future_columns,
        "selector_columns": selector_columns,
        "audit_path": str(audit_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "research_only": True,
        "broker_action": "none",
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot), encoding="utf-8")
    return ChipReversalPitAuditResult(
        output_dir=resolved_output,
        audit_path=audit_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        audit=audit_frame,
    )
