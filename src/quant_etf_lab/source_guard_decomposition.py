"""Compare source-selection runs and explain guard-driven changes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SourceGuardDecompositionResult:
    output_dir: Path
    report_path: Path
    windows_path: Path
    snapshot_path: Path
    snapshot: dict[str, Any]


REQUIRED_SUMMARY_COLUMNS = {
    "window",
    "selected_candidate",
    "selected_source",
    "selected_score",
    "raw_best_candidate",
    "raw_best_source",
    "raw_best_score",
    "test_total_return",
    "test_max_drawdown",
    "test_sharpe",
}

NUMERIC_COLUMNS = {
    "selected_score",
    "raw_best_score",
    "source_switch_margin",
    "score_mode_min_edge",
    "score_mode_edge_vs_baseline",
    "test_total_return",
    "test_max_drawdown",
    "test_sharpe",
}


def _read_summary(path: Path) -> pd.DataFrame:
    summary_path = path / "portfolio_walk_forward_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing required CSV: {summary_path}")
    frame = pd.read_csv(summary_path)
    missing = REQUIRED_SUMMARY_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{summary_path} missing columns: {sorted(missing)}")
    for column in NUMERIC_COLUMNS & set(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _diff(left: Any, right: Any) -> float | None:
    left_float = _safe_float(left)
    right_float = _safe_float(right)
    if left_float is None or right_float is None:
        return None
    return left_float - right_float


def _classify_change(row: dict[str, Any]) -> str:
    if (
        row["default_selected_candidate"] == row["candidate_selected_candidate"]
        and row["default_selected_source"] == row["candidate_selected_source"]
    ):
        return "unchanged"
    if _truthy(row.get("candidate_score_mode_gate_applied")):
        return "score_mode_gate"

    raw_best_changed = (
        row["default_raw_best_candidate"] != row["candidate_raw_best_candidate"]
        or row["default_raw_best_source"] != row["candidate_raw_best_source"]
    )
    candidate_differs_from_raw_best = (
        row["candidate_selected_candidate"] != row["candidate_raw_best_candidate"]
        or row["candidate_selected_source"] != row["candidate_raw_best_source"]
    )
    if candidate_differs_from_raw_best:
        margin = _safe_float(row.get("candidate_source_switch_margin")) or 0.0
        default_source = _text(row.get("candidate_default_source"))
        selected_source = row["candidate_selected_source"]
        selected_candidate = row["candidate_selected_candidate"]
        if margin > 0 and (selected_source in {default_source, "core_only"} or selected_candidate == "core_only"):
            return "source_switch_guard"
        return "guarded_override"
    if raw_best_changed:
        return "scoring_mode_changed_raw_best"
    return "selection_changed"


def _recommendation(changed_by: str, return_delta: float | None) -> str:
    if changed_by == "source_switch_guard":
        if return_delta is not None and return_delta < 0:
            return "review_risk_budget_not_default_source_switch"
        if return_delta is not None and return_delta > 0:
            return "guard_helped_monitor"
        return "review_source_switch_guard"
    if changed_by == "score_mode_gate":
        return "review_score_gate_edge"
    if changed_by == "scoring_mode_changed_raw_best":
        return "review_scoring_mode_candidate"
    if changed_by == "guarded_override":
        return "review_guard_override"
    if changed_by == "selection_changed":
        return "review_selection_change"
    return "keep_current_observation"


def _format_pct(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.2f}%"


def _format_float(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.3f}"


def _row_by_window(frame: pd.DataFrame, window: str) -> pd.Series | None:
    rows = frame.loc[frame["window"].astype(str) == window]
    if rows.empty:
        return None
    return rows.iloc[0]


def _get(row: pd.Series | None, column: str, default: Any = pd.NA) -> Any:
    if row is None or column not in row.index:
        return default
    return row[column]


def _build_window_rows(default_summary: pd.DataFrame, candidate_summary: pd.DataFrame) -> pd.DataFrame:
    windows = sorted(
        {
            *default_summary["window"].astype(str).tolist(),
            *candidate_summary["window"].astype(str).tolist(),
        }
    )
    rows: list[dict[str, Any]] = []
    for window in windows:
        default_row = _row_by_window(default_summary, window)
        candidate_row = _row_by_window(candidate_summary, window)
        record = {
            "window": window,
            "test_start": _get(candidate_row, "test_start", _get(default_row, "test_start")),
            "test_end": _get(candidate_row, "test_end", _get(default_row, "test_end")),
            "default_selected_candidate": _text(_get(default_row, "selected_candidate")),
            "default_selected_source": _text(_get(default_row, "selected_source")),
            "default_selected_score": _get(default_row, "selected_score"),
            "default_raw_best_candidate": _text(_get(default_row, "raw_best_candidate")),
            "default_raw_best_source": _text(_get(default_row, "raw_best_source")),
            "default_raw_best_score": _get(default_row, "raw_best_score"),
            "default_test_total_return": _get(default_row, "test_total_return"),
            "default_test_max_drawdown": _get(default_row, "test_max_drawdown"),
            "default_test_sharpe": _get(default_row, "test_sharpe"),
            "candidate_selected_candidate": _text(_get(candidate_row, "selected_candidate")),
            "candidate_selected_source": _text(_get(candidate_row, "selected_source")),
            "candidate_selected_score": _get(candidate_row, "selected_score"),
            "candidate_raw_best_candidate": _text(_get(candidate_row, "raw_best_candidate")),
            "candidate_raw_best_source": _text(_get(candidate_row, "raw_best_source")),
            "candidate_raw_best_score": _get(candidate_row, "raw_best_score"),
            "candidate_default_source": _text(_get(candidate_row, "default_source")),
            "candidate_source_switch_margin": _get(candidate_row, "source_switch_margin", 0.0),
            "candidate_score_mode": _text(_get(candidate_row, "score_mode")),
            "candidate_score_mode_min_edge": _get(candidate_row, "score_mode_min_edge"),
            "candidate_score_mode_gate_applied": _get(candidate_row, "score_mode_gate_applied", False),
            "candidate_score_mode_edge_vs_baseline": _get(candidate_row, "score_mode_edge_vs_baseline"),
            "candidate_test_total_return": _get(candidate_row, "test_total_return"),
            "candidate_test_max_drawdown": _get(candidate_row, "test_max_drawdown"),
            "candidate_test_sharpe": _get(candidate_row, "test_sharpe"),
        }
        record["raw_edge_vs_candidate_selected"] = _diff(
            record["candidate_raw_best_score"],
            record["candidate_selected_score"],
        )
        record["return_delta_vs_default"] = _diff(
            record["candidate_test_total_return"],
            record["default_test_total_return"],
        )
        record["drawdown_delta_vs_default"] = _diff(
            record["candidate_test_max_drawdown"],
            record["default_test_max_drawdown"],
        )
        record["sharpe_delta_vs_default"] = _diff(
            record["candidate_test_sharpe"],
            record["default_test_sharpe"],
        )
        record["changed_by"] = _classify_change(record)
        record["recommendation"] = _recommendation(record["changed_by"], record["return_delta_vs_default"])
        rows.append(record)
    return pd.DataFrame(rows)


def run_source_guard_decomposition(
    default_dir: str | Path,
    candidate_dir: str | Path,
    output_dir: str | Path = "outputs/research/source_guard_decomposition_latest",
) -> SourceGuardDecompositionResult:
    default_path = Path(default_dir)
    candidate_path = Path(candidate_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    default_summary = _read_summary(default_path)
    candidate_summary = _read_summary(candidate_path)
    windows = _build_window_rows(default_summary, candidate_summary)

    windows_path = out_dir / "source_guard_decomposition_windows.csv"
    snapshot_path = out_dir / "source_guard_decomposition_snapshot.json"
    report_path = out_dir / "source_guard_decomposition.md"
    windows.to_csv(windows_path, index=False, encoding="utf-8-sig")

    changed = windows[windows["changed_by"] != "unchanged"]
    worsened_changed = changed[pd.to_numeric(changed["return_delta_vs_default"], errors="coerce") < 0]
    source_switch_guard_count = int((windows["changed_by"] == "source_switch_guard").sum())
    score_mode_gate_count = int((windows["changed_by"] == "score_mode_gate").sum())
    decision = "review_source_guard_decomposition"
    if source_switch_guard_count > 0 and not worsened_changed.empty:
        decision = "keep_default_review_guard_design"

    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "default_dir": str(default_path),
        "candidate_dir": str(candidate_path),
        "window_count": int(len(windows)),
        "changed_window_count": int(len(changed)),
        "source_switch_guard_count": source_switch_guard_count,
        "score_mode_gate_count": score_mode_gate_count,
        "worsened_changed_window_count": int(len(worsened_changed)),
        "decision": decision,
    }
    snapshot_path.write_text(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")

    table = [
        "| Window | Changed by | Default selected | Candidate selected | Raw edge | Return delta | Sharpe delta | Recommendation |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in windows.itertuples(index=False):
        if row.changed_by == "unchanged":
            continue
        table.append(
            "| {window} | {changed_by} | {default_candidate} | {candidate_candidate} | {raw_edge} | {return_delta} | {sharpe_delta} | {recommendation} |".format(
                window=row.window,
                changed_by=row.changed_by,
                default_candidate=row.default_selected_candidate,
                candidate_candidate=row.candidate_selected_candidate,
                raw_edge=_format_float(row.raw_edge_vs_candidate_selected),
                return_delta=_format_pct(row.return_delta_vs_default),
                sharpe_delta=_format_float(row.sharpe_delta_vs_default),
                recommendation=row.recommendation,
            )
        )
    if len(table) == 2:
        table.append("| n/a | unchanged | n/a | n/a | n/a | n/a | n/a | keep_current_observation |")

    body = f"""# Source Guard Decomposition

Generated at: `{snapshot["generated_at"]}`

This report compares source-selection research runs only. It does not connect to brokers, place orders, or provide investment advice.

## Summary

| Item | Value |
| --- | ---: |
| Windows compared | {snapshot["window_count"]} |
| Changed windows | {snapshot["changed_window_count"]} |
| Source-switch guard changes | {snapshot["source_switch_guard_count"]} |
| Score-mode gate changes | {snapshot["score_mode_gate_count"]} |
| Worsened changed windows | {snapshot["worsened_changed_window_count"]} |
| Decision | `{snapshot["decision"]}` |

## Changed Windows

{chr(10).join(table)}

## Files

- `source_guard_decomposition_windows.csv`: per-window default-vs-candidate selection, guard attribution, and OOS metric deltas.
- `source_guard_decomposition_snapshot.json`: machine-readable summary.
"""
    report_path.write_text(body, encoding="utf-8")

    return SourceGuardDecompositionResult(
        output_dir=out_dir,
        report_path=report_path,
        windows_path=windows_path,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )
