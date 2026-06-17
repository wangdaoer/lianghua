"""Phase 2 model construction status reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .process_lock import list_process_locks


@dataclass(frozen=True)
class Phase2StatusResult:
    output_dir: Path
    components_path: Path
    report_path: Path
    components: pd.DataFrame


DEFAULT_BASELINE_RUN = Path("outputs/backtests/ashare_main_chinext_hold_prefer_warm120_dd_light_pos3_lossguard5d40")
DEFAULT_CORE_WF = Path("outputs/walk_forward/main_chinext_stable_v2_full_20260613_011322")
DEFAULT_SATELLITE_WF = Path("outputs/walk_forward/main_chinext_satellite_quality_v2_full")
DEFAULT_PORTFOLIO_RUN = Path("outputs/portfolios/main_chinext_core_satellite_quality_v2_guarded")
DEFAULT_PORTFOLIO_WF = Path("outputs/portfolio_source_selection/main_chinext_portfolio_source_selection_validation6_v1")


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = Path(path) if path is not None else default
    return raw if raw.is_absolute() else project_root / raw


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _pct(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "N/A"
    return f"{number * 100:.2f}%"


def _num(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "N/A"
    return f"{number:.3f}"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _curve_metrics(path: Path, column: str = "stitched_equity") -> dict[str, float] | None:
    frame = _read_csv(path)
    if frame.empty or column not in frame or "date" not in frame:
        return None
    data = frame[["date", column]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", column]).sort_values("date")
    data = data[data[column] > 0]
    if data.empty:
        return None
    series = data.set_index("date")[column]
    returns = series.pct_change().dropna()
    drawdown = series / series.cummax() - 1.0
    sharpe = 0.0
    if len(returns) > 1 and float(returns.std()) != 0:
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std())
    return {
        "total_return": float(series.iloc[-1] / series.iloc[0] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "observation_days": float(len(series)),
    }


def _summary_window_stats(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty or "test_total_return" not in summary:
        return {}
    returns = pd.to_numeric(summary["test_total_return"], errors="coerce").dropna()
    stats: dict[str, Any] = {
        "window_count": int(len(summary)),
        "positive_oos_window_ratio": float((returns > 0).mean()) if not returns.empty else None,
        "worst_oos_return": float(returns.min()) if not returns.empty else None,
    }
    if "selected_candidate" in summary:
        selected = summary["selected_candidate"].dropna().astype(str)
        stats["selected_candidate_count"] = int(selected.nunique())
        stats["latest_selected_candidate"] = selected.iloc[-1] if not selected.empty else ""
    return stats


def _row(
    layer: str,
    name: str,
    status: str,
    source_path: Path,
    note: str,
    metrics: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = metrics or {}
    stats = stats or {}
    return {
        "layer": layer,
        "name": name,
        "status": status,
        "total_return": _safe_float(metrics.get("total_return")),
        "max_drawdown": _safe_float(metrics.get("max_drawdown")),
        "sharpe": _safe_float(metrics.get("sharpe")),
        "observation_days": _safe_float(metrics.get("observation_days")),
        "positive_oos_window_ratio": _safe_float(stats.get("positive_oos_window_ratio")),
        "worst_oos_return": _safe_float(stats.get("worst_oos_return")),
        "window_count": stats.get("window_count"),
        "selected_candidate_count": stats.get("selected_candidate_count"),
        "latest_selected_candidate": stats.get("latest_selected_candidate", ""),
        "source_path": str(source_path),
        "note": note,
    }


def _backtest_component(path: Path) -> dict[str, Any]:
    metrics = _read_json(path / "metrics.json")
    if metrics is None:
        return _row("baseline", "Current THS-style full-sample baseline", "missing", path, "Missing metrics.json.")
    return _row(
        "baseline",
        "Current THS-style full-sample baseline",
        "complete",
        path,
        "High-return reference baseline; useful, but drawdown remains too deep for the second-stage core.",
        metrics,
    )


def _portfolio_component(path: Path) -> dict[str, Any]:
    metrics = _read_json(path / "metrics.json")
    if metrics is None:
        return _row("portfolio", "Guarded core-satellite portfolio", "missing", path, "Missing metrics.json.")
    note = "Combines defensive core and satellite with benchmark regime plus satellite health filtering."
    excess = _safe_float(metrics.get("excess_return_vs_core"))
    if excess is not None:
        note += f" Excess return vs core: {_pct(excess)}."
        if excess < 0:
            note += " Fixed satellite allocation underperformed the core; prefer allocator-gated use."
    return _row("portfolio", "Guarded core-satellite portfolio", "complete", path, note, metrics)


def _active_lock_note(project_root: Path, marker: str) -> str | None:
    lock_dir = project_root / "outputs" / "locks"
    for lock in list_process_locks(lock_dir):
        command = lock.payload.command
        key = lock.payload.key
        if marker in command or marker in key:
            state = "active" if lock.active else "stale"
            return f"Matching process lock is {state}: PID {lock.payload.pid}, lock `{lock.path}`."
    return None


def _walk_forward_component(path: Path, layer: str, name: str, project_root: Path, portfolio: bool = False) -> dict[str, Any]:
    summary_name = "portfolio_walk_forward_summary.csv" if portfolio else "walk_forward_summary.csv"
    candidate_name = "portfolio_candidate_results.csv" if portfolio else "candidate_results.csv"
    summary = _read_csv(path / summary_name)
    candidate = _read_csv(path / candidate_name)
    curve_metrics = _curve_metrics(path / "oos_equity_stitched.csv")
    required_missing = [
        filename
        for filename in (summary_name, candidate_name, "oos_equity_stitched.csv")
        if not (path / filename).exists()
    ]
    if required_missing:
        lock_note = _active_lock_note(project_root, path.name)
        status = "running" if lock_note and "active" in lock_note else "incomplete"
        return _row(
            layer,
            name,
            status,
            path,
            "Missing required files: " + ", ".join(required_missing) + "." + (f" {lock_note}" if lock_note else ""),
            curve_metrics,
            _summary_window_stats(summary),
        )
    if curve_metrics is None:
        return _row(layer, name, "incomplete", path, "Could not compute stitched OOS curve metrics.", None, _summary_window_stats(summary))
    note = f"Walk-forward complete with {len(candidate)} candidate-window rows."
    return _row(layer, name, "complete", path, note, curve_metrics, _summary_window_stats(summary))


def build_phase2_components(
    project_root: Path,
    baseline_run_dir: str | Path | None = None,
    core_wf_dir: str | Path | None = None,
    satellite_wf_dir: str | Path | None = None,
    portfolio_run_dir: str | Path | None = None,
    portfolio_wf_dir: str | Path | None = None,
) -> pd.DataFrame:
    root = project_root.resolve()
    rows = [
        _backtest_component(_resolve(root, baseline_run_dir, DEFAULT_BASELINE_RUN)),
        _walk_forward_component(
            _resolve(root, core_wf_dir, DEFAULT_CORE_WF),
            "core",
            "Stable defensive core walk-forward",
            root,
        ),
        _walk_forward_component(
            _resolve(root, satellite_wf_dir, DEFAULT_SATELLITE_WF),
            "satellite",
            "Growth satellite walk-forward",
            root,
        ),
        _portfolio_component(_resolve(root, portfolio_run_dir, DEFAULT_PORTFOLIO_RUN)),
        _walk_forward_component(
            _resolve(root, portfolio_wf_dir, DEFAULT_PORTFOLIO_WF),
            "allocator",
            "Portfolio allocation walk-forward",
            root,
            portfolio=True,
        ),
    ]
    return pd.DataFrame(rows)


def _stage_decision(components: pd.DataFrame) -> str:
    statuses = dict(zip(components["layer"], components["status"]))
    if statuses.get("core") == "complete" and statuses.get("portfolio") == "complete" and statuses.get("allocator") == "complete":
        if statuses.get("satellite") == "complete":
            return "Phase 2 scaffold is complete: core, satellite, portfolio, and allocator can move to monitoring-style review."
        return "Phase 2 scaffold is usable: core, portfolio, and allocator are complete; satellite research still needs a clean full walk-forward finish."
    if statuses.get("core") == "complete":
        return "Phase 2 is partially built: defensive core is available, but portfolio or allocator completion is still required."
    return "Phase 2 is not ready: build or repair the defensive core walk-forward first."


def _component_table(components: pd.DataFrame) -> str:
    rows = [
        "| layer | status | total_return | max_drawdown | sharpe | OOS positive | worst OOS | windows | latest selected | note |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in components.to_dict("records"):
        rows.append(
            "| {layer} | {status} | {total} | {drawdown} | {sharpe} | {positive} | {worst} | {windows} | {selected} | {note} |".format(
                layer=row["layer"],
                status=row["status"],
                total=_pct(row.get("total_return")),
                drawdown=_pct(row.get("max_drawdown")),
                sharpe=_num(row.get("sharpe")),
                positive=_pct(row.get("positive_oos_window_ratio")),
                worst=_pct(row.get("worst_oos_return")),
                windows="" if pd.isna(row.get("window_count")) else int(row.get("window_count")),
                selected=str(row.get("latest_selected_candidate", "")),
                note=str(row.get("note", "")).replace("\n", " "),
            )
        )
    return "\n".join(rows)


def _next_step_text(components: pd.DataFrame) -> str:
    statuses = dict(zip(components["layer"], components["status"]))
    incomplete = components[components["status"] != "complete"]
    if not incomplete.empty:
        return (
            "Finish or rerun incomplete components first, then keep the guarded portfolio and portfolio allocator as "
            "the second-stage acceptance gate. A satellite idea should only be promoted when it improves the "
            "allocator's out-of-sample table, not merely its own full-sample return."
        )

    portfolio_rows = components[components["layer"] == "portfolio"]
    allocator_rows = components[components["layer"] == "allocator"]
    portfolio_return = _safe_float(portfolio_rows.iloc[0]["total_return"]) if not portfolio_rows.empty else None
    core_return = _safe_float(components[components["layer"] == "core"].iloc[0]["total_return"]) if "core" in statuses else None
    allocator_sharpe = _safe_float(allocator_rows.iloc[0]["sharpe"]) if not allocator_rows.empty else None
    allocator_drawdown = _safe_float(allocator_rows.iloc[0]["max_drawdown"]) if not allocator_rows.empty else None

    if portfolio_return is not None and core_return is not None and portfolio_return < core_return:
        return (
            "Do not promote the fixed satellite allocation. Keep the defensive core as the default second-stage "
            "base, and use the portfolio allocator as the gate for satellite exposure. The latest allocator result "
            f"has Sharpe {_num(allocator_sharpe)} and max drawdown {_pct(allocator_drawdown)}, so the next build step "
            "is a monitoring-style phase-2 review report rather than another broad factor search."
        )
    return (
        "The second-stage scaffold is complete. Promote the guarded portfolio only after a monitoring-style review "
        "confirms that its allocator-gated out-of-sample behavior remains better than the defensive core."
    )


def _render_report(components: pd.DataFrame, output_dir: Path) -> str:
    incomplete = components[components["status"] != "complete"]
    if incomplete.empty:
        blocker_text = "No incomplete second-stage components were detected."
    else:
        blocker_text = "\n".join(
            f"- `{row.layer}`: {row.note} Source: `{row.source_path}`" for row in incomplete.itertuples(index=False)
        )
    return f"""# Phase 2 Model Status

Generated at: `{datetime.now().isoformat(timespec="seconds")}`

Output directory: `{output_dir}`

This report is research and review only. It does not connect to brokers, place orders, or change live positions.

## Stage Decision

{_stage_decision(components)}

## Component Status

{_component_table(components)}

## Build Notes

- The full-sample THS-style baseline remains a return reference, but its drawdown is too high to be the only second-stage model.
- The defensive core is the lower-drawdown foundation for phase 2.
- The portfolio layer is the current integration point: it can add satellite participation while keeping portfolio drawdown close to the defensive core.
- The allocator walk-forward is the main anti-overfit check for phase 2 because it chooses allocation parameters using only prior windows.
- Sentiment remains skipped for now by request and is not part of this phase-2 decision table.

## Incomplete Or Blocked Items

{blocker_text}

## Next Construction Step

{_next_step_text(components)}
"""


def run_phase2_status(
    project_root: Path | str = Path("."),
    output_dir: Path | str | None = None,
    baseline_run_dir: str | Path | None = None,
    core_wf_dir: str | Path | None = None,
    satellite_wf_dir: str | Path | None = None,
    portfolio_run_dir: str | Path | None = None,
    portfolio_wf_dir: str | Path | None = None,
) -> Phase2StatusResult:
    root = Path(project_root).resolve()
    resolved_output = _resolve(root, output_dir, Path("outputs/research/phase2_model_status_latest"))
    components = build_phase2_components(
        root,
        baseline_run_dir=baseline_run_dir,
        core_wf_dir=core_wf_dir,
        satellite_wf_dir=satellite_wf_dir,
        portfolio_run_dir=portfolio_run_dir,
        portfolio_wf_dir=portfolio_wf_dir,
    )
    resolved_output.mkdir(parents=True, exist_ok=True)
    components_path = resolved_output / "phase2_components.csv"
    report_path = resolved_output / "phase2_status.md"
    components.to_csv(components_path, index=False, encoding="utf-8-sig")
    report_path.write_text(_render_report(components, resolved_output), encoding="utf-8")
    return Phase2StatusResult(
        output_dir=resolved_output,
        components_path=components_path,
        report_path=report_path,
        components=components,
    )
