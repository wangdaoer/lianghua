"""Monitoring-style review reports for the phase-2 model scaffold."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from ._compat import read_text


DEFAULT_PHASE2_COMPONENTS = Path(
    "outputs/research/phase2_model_status_source_selection_default_20260614/phase2_components.csv"
)
DEFAULT_ALLOCATOR_DIR = Path(
    "outputs/portfolio_source_selection/main_chinext_source_selection_highgain_pos8_dd50_cap30_activation_dd50_20260624"
)
DEFAULT_PROMOTION_REVIEW_DIR = Path("outputs/research/allocator_promotion_with_execution_cost_20260614")


@dataclass(frozen=True)
class Phase2ReviewResult:
    output_dir: Path
    report_path: Path
    snapshot_path: Path
    snapshot: dict[str, Any]


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = Path(path) if path is not None else default
    return raw if raw.is_absolute() else project_root / raw


def _components_file(path: Path) -> Path:
    if path.suffix.lower() == ".csv":
        return path
    return path / "phase2_components.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(read_text(path))
    except (OSError, json.JSONDecodeError):
        return {}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _pct(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "N/A"
    return f"{number * 100:.2f}%"


def _num(value: Any, digits: int = 3) -> str:
    number = _as_float(value)
    if number is None:
        return "N/A"
    return f"{number:.{digits}f}"


def _parse_yyyymmdd(value: Any) -> pd.Timestamp:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) == 8 and text.isdigit():
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(value, errors="coerce")


def _format_yyyymmdd(value: Any) -> str:
    parsed = _parse_yyyymmdd(value)
    if pd.isna(parsed):
        return str(value)
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _latest_row(frame: pd.DataFrame, date_column: str) -> pd.Series | None:
    if frame.empty or date_column not in frame:
        return None
    data = frame.copy()
    data["_sort_date"] = data[date_column].map(_parse_yyyymmdd)
    data = data.dropna(subset=["_sort_date"]).sort_values("_sort_date")
    if data.empty:
        return None
    return data.iloc[-1]


def _curve_snapshot(curve_path: Path) -> dict[str, Any]:
    frame = _read_csv(curve_path)
    if frame.empty or not {"date", "stitched_equity"}.issubset(frame.columns):
        return {"curve_status": "missing"}
    data = frame[["date", "window", "stitched_equity"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["stitched_equity"] = pd.to_numeric(data["stitched_equity"], errors="coerce")
    data = data.dropna(subset=["date", "stitched_equity"]).sort_values("date")
    data = data[data["stitched_equity"] > 0]
    if data.empty:
        return {"curve_status": "empty"}

    series = data.set_index("date")["stitched_equity"]
    drawdown = series / series.cummax() - 1.0
    latest = data.iloc[-1]
    snapshot: dict[str, Any] = {
        "curve_status": "ok",
        "latest_date": pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
        "latest_window": str(latest.get("window", "")),
        "latest_equity": float(latest["stitched_equity"]),
        "total_return": float(series.iloc[-1] / series.iloc[0] - 1.0),
        "current_drawdown": float(drawdown.iloc[-1]),
        "max_drawdown": float(drawdown.min()),
        "observation_days": int(len(series)),
    }
    for lookback in (5, 20, 60):
        if len(series) > lookback:
            snapshot[f"return_{lookback}d"] = float(series.iloc[-1] / series.iloc[-lookback - 1] - 1.0)
    return snapshot


def _latest_promotion_review_dir(project_root: Path) -> Path | None:
    candidate_pattern = re.compile(r"allocator_promotion.*(?:_(\d{8})(?:$|_))")
    research_dir = project_root / "outputs" / "research"
    if not research_dir.exists():
        return None

    candidates: list[tuple[tuple[int, float], Path]] = []
    for item in research_dir.iterdir():
        if not item.is_dir():
            continue
        if not item.name.startswith("allocator_promotion"):
            continue
        snapshot_file = item / "allocator_promotion_snapshot.json"
        if not snapshot_file.exists():
            continue
        match = candidate_pattern.search(item.name)
        date_rank = int(match.group(1)) if match else 0
        last_modified = item.stat().st_mtime
        candidates.append(((date_rank, last_modified), item))

    if not candidates:
        return None

    _, latest = max(candidates, key=lambda item: item[0])
    return latest


def _allocator_snapshot(project_root: Path, allocator_dir: Path) -> dict[str, Any]:
    summary = _read_csv(allocator_dir / "portfolio_walk_forward_summary.csv")
    latest = _latest_row(summary, "test_end")
    if latest is None:
        return {"allocator_status": "missing"}

    params_path = Path(str(latest.get("selected_params_path", "")))
    if not params_path.is_absolute():
        params_path = project_root / params_path if str(params_path).startswith("outputs") else allocator_dir / params_path
    params = _read_json(params_path)
    allocation = _as_dict(params.get("allocation")) or params
    weights = _as_dict(allocation.get("weights"))
    risk_on = _as_dict(weights.get("risk_on"))
    risk_off = _as_dict(weights.get("risk_off"))
    regime = _as_dict(allocation.get("regime_overrides"))
    satellite_filter = _as_dict(allocation.get("satellite_filter"))

    return {
        "allocator_status": "ok",
        "window": str(latest.get("window", "")),
        "train_period": f"{_format_yyyymmdd(latest.get('train_start'))} to {_format_yyyymmdd(latest.get('train_end'))}",
        "test_period": f"{_format_yyyymmdd(latest.get('test_start'))} to {_format_yyyymmdd(latest.get('test_end'))}",
        "selected_candidate": str(latest.get("selected_candidate", "")),
        "selected_score": _as_float(latest.get("selected_score")),
        "test_total_return": _as_float(latest.get("test_total_return")),
        "test_max_drawdown": _as_float(latest.get("test_max_drawdown")),
        "test_sharpe": _as_float(latest.get("test_sharpe")),
        "test_core_total_return": _as_float(latest.get("test_core_total_return")),
        "test_satellite_total_return": _as_float(latest.get("test_satellite_total_return")),
        "test_average_core_weight": _as_float(latest.get("test_average_core_weight")),
        "test_average_satellite_weight": _as_float(latest.get("test_average_satellite_weight")),
        "test_risk_on_day_ratio": _as_float(latest.get("test_risk_on_day_ratio")),
        "test_risk_off_day_ratio": _as_float(latest.get("test_risk_off_day_ratio")),
        "test_crash_day_ratio": _as_float(latest.get("test_crash_day_ratio")),
        "risk_on_core_weight": _as_float(risk_on.get("core")),
        "risk_on_satellite_weight": _as_float(risk_on.get("satellite")),
        "risk_off_core_weight": _as_float(risk_off.get("core")),
        "risk_off_satellite_weight": _as_float(risk_off.get("satellite")),
        "regime_ma_window": regime.get("ma_window"),
        "regime_risk_on_drop_threshold": _as_float(regime.get("risk_on_drop_threshold")),
        "regime_crash_drop_threshold": _as_float(regime.get("crash_drop_threshold")),
        "satellite_filter_enabled": bool(satellite_filter.get("enabled", False)),
        "selected_params_path": str(params_path),
    }


def _component_snapshot(components_path: Path) -> dict[str, Any]:
    components = _read_csv(components_path)
    if components.empty:
        return {"component_status": "missing", "incomplete_layers": []}
    incomplete = components[components["status"] != "complete"]
    data: dict[str, Any] = {
        "component_status": "ok",
        "complete_count": int((components["status"] == "complete").sum()),
        "component_count": int(len(components)),
        "incomplete_layers": list(incomplete["layer"].astype(str)),
    }
    for row in components.to_dict("records"):
        layer = str(row.get("layer", ""))
        data[f"{layer}_status"] = row.get("status")
        data[f"{layer}_total_return"] = _as_float(row.get("total_return"))
        data[f"{layer}_max_drawdown"] = _as_float(row.get("max_drawdown"))
        data[f"{layer}_sharpe"] = _as_float(row.get("sharpe"))
        data[f"{layer}_latest_selected"] = row.get("latest_selected_candidate", "")
    return data


def _promotion_snapshot(promotion_dir: Path | None) -> dict[str, Any]:
    if promotion_dir is None:
        return {"promotion_status": "not_configured"}
    snapshot_path = promotion_dir / "allocator_promotion_snapshot.json"
    report_path = promotion_dir / "allocator_promotion_review.md"
    snapshot = _read_json(snapshot_path)
    if not snapshot:
        return {
            "promotion_status": "missing",
            "promotion_review_dir": str(promotion_dir),
            "promotion_snapshot_path": str(snapshot_path),
            "promotion_report_path": str(report_path),
        }
    legacy_support_count = snapshot.get("sensitivity_support_count")
    group_support_count = snapshot.get("sensitivity_group_support_count", legacy_support_count)
    run_support_count = snapshot.get("sensitivity_run_support_count", legacy_support_count)
    total_support_count = snapshot.get("support_group_count", legacy_support_count)
    evidence_support_count = snapshot.get("evidence_support_count")
    return {
        "promotion_status": "ok",
        "promotion_review_dir": str(promotion_dir),
        "promotion_snapshot_path": str(snapshot_path),
        "promotion_report_path": str(report_path),
        "promotion_decision": snapshot.get("decision"),
        "promotion_candidate_passes_headline_gate": bool(snapshot.get("candidate_passes_headline_gate", False)),
        "promotion_sensitivity_support_count": snapshot.get("sensitivity_support_count"),
        "promotion_sensitivity_group_support_count": group_support_count,
        "promotion_support_group_count": total_support_count,
        "promotion_evidence_support_count": evidence_support_count,
        "promotion_sensitivity_run_support_count": run_support_count,
        "promotion_min_sensitivity_support": snapshot.get("min_sensitivity_support"),
        "promotion_return_edge": _as_float(snapshot.get("return_edge")),
        "promotion_sharpe_edge": _as_float(snapshot.get("sharpe_edge")),
        "promotion_drawdown_change": _as_float(snapshot.get("drawdown_change")),
    }


def _posture(snapshot: dict[str, Any]) -> str:
    if snapshot.get("component_status") != "ok" or snapshot.get("incomplete_layers"):
        return "wait_for_components"
    selected = str(snapshot.get("selected_candidate", ""))
    satellite_weight = _as_float(snapshot.get("risk_on_satellite_weight"), 0.0) or 0.0
    fixed_portfolio_return = _as_float(snapshot.get("portfolio_total_return"), 0.0) or 0.0
    core_return = _as_float(snapshot.get("core_total_return"), 0.0) or 0.0
    if fixed_portfolio_return < core_return and satellite_weight > 0:
        return "core_base_allocator_gated_satellite"
    if selected == "core_only" or satellite_weight <= 0:
        return "core_only"
    return "allocator_risk_on_satellite"


def _posture_text(posture: str) -> str:
    return {
        "wait_for_components": "Wait for complete phase-2 components before promoting any portfolio.",
        "core_only": "Defensive core is the active base; the allocator gives no satellite risk budget.",
        "core_base_allocator_gated_satellite": (
            "Defensive core is the base, and satellite exposure is allowed only through the allocator gate. "
            "The fixed satellite portfolio is not promoted."
        ),
        "allocator_risk_on_satellite": (
            "The allocator currently permits conditional satellite exposure. This remains research review, "
            "not a trading instruction."
        ),
    }.get(posture, "Research review.")


def build_phase2_review_snapshot(
    project_root: Path,
    components_path: str | Path | None = None,
    allocator_dir: str | Path | None = None,
    promotion_review_dir: str | Path | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    resolved_components = _components_file(_resolve(root, components_path, DEFAULT_PHASE2_COMPONENTS))
    resolved_allocator = _resolve(root, allocator_dir, DEFAULT_ALLOCATOR_DIR)
    if promotion_review_dir is None:
        resolved_promotion = _latest_promotion_review_dir(root)
    else:
        resolved_promotion = _resolve(root, promotion_review_dir, DEFAULT_PROMOTION_REVIEW_DIR)
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "components_path": str(resolved_components),
        "allocator_dir": str(resolved_allocator),
    }
    snapshot.update(_component_snapshot(resolved_components))
    snapshot.update(_allocator_snapshot(root, resolved_allocator))
    snapshot.update(_curve_snapshot(resolved_allocator / "oos_equity_stitched.csv"))
    snapshot.update(_promotion_snapshot(resolved_promotion))
    snapshot["posture"] = _posture(snapshot)
    return snapshot


def _render_report(snapshot: dict[str, Any]) -> str:
    posture = _posture_text(str(snapshot.get("posture", "")))
    fixed_delta = None
    if _as_float(snapshot.get("portfolio_total_return")) is not None and _as_float(snapshot.get("core_total_return")) is not None:
        fixed_delta = (_as_float(snapshot.get("portfolio_total_return")) or 0.0) - (_as_float(snapshot.get("core_total_return")) or 0.0)
    return f"""# Phase 2 Monitoring Review

Generated at: `{snapshot.get("generated_at")}`

This report is research and review only. It does not connect to brokers, place orders, or provide investment advice.

## Current Posture

{posture}

| Item | Value |
| --- | ---: |
| Latest local equity date | {snapshot.get("latest_date", "N/A")} |
| Latest allocator candidate | {snapshot.get("selected_candidate", "N/A")} |
| Latest test period | {snapshot.get("test_period", "N/A")} |
| Risk-on core weight | {_pct(snapshot.get("risk_on_core_weight"))} |
| Risk-on satellite weight | {_pct(snapshot.get("risk_on_satellite_weight"))} |
| Risk-off satellite weight | {_pct(snapshot.get("risk_off_satellite_weight"))} |
| Regime MA window | {snapshot.get("regime_ma_window", "N/A")} |
| Risk-on drop threshold | {_pct(snapshot.get("regime_risk_on_drop_threshold"))} |
| Crash drop threshold | {_pct(snapshot.get("regime_crash_drop_threshold"))} |
| Satellite health filter enabled | {snapshot.get("satellite_filter_enabled", "N/A")} |

## Evidence

| Metric | Value |
| --- | ---: |
| Phase-2 components complete | {snapshot.get("complete_count", 0)} / {snapshot.get("component_count", 0)} |
| Defensive core OOS return | {_pct(snapshot.get("core_total_return"))} |
| Defensive core OOS max drawdown | {_pct(snapshot.get("core_max_drawdown"))} |
| Fixed core-satellite OOS return | {_pct(snapshot.get("portfolio_total_return"))} |
| Fixed portfolio return minus core | {_pct(fixed_delta)} |
| Activation allocator OOS return | {_pct(snapshot.get("total_return"))} |
| Activation allocator max drawdown | {_pct(snapshot.get("max_drawdown"))} |
| Activation allocator Sharpe | {_num(snapshot.get("allocator_sharpe"))} |
| Latest window return | {_pct(snapshot.get("test_total_return"))} |
| Latest window max drawdown | {_pct(snapshot.get("test_max_drawdown"))} |
| Latest window Sharpe | {_num(snapshot.get("test_sharpe"))} |
| Latest window average satellite weight | {_pct(snapshot.get("test_average_satellite_weight"))} |
| Latest window risk-on day ratio | {_pct(snapshot.get("test_risk_on_day_ratio"))} |
| Current stitched equity drawdown | {_pct(snapshot.get("current_drawdown"))} |
| 20-trading-day stitched return | {_pct(snapshot.get("return_20d"))} |

## Allocator Promotion Watch

| Item | Value |
| --- | ---: |
| Promotion status | `{snapshot.get("promotion_status", "not_configured")}` |
| Promotion decision | `{snapshot.get("promotion_decision", "N/A")}` |
| Candidate passes headline gate | {snapshot.get("promotion_candidate_passes_headline_gate", "N/A")} |
| Independent support groups | {snapshot.get("promotion_support_group_count", snapshot.get("promotion_sensitivity_group_support_count", snapshot.get("promotion_sensitivity_support_count", "N/A")))} / {snapshot.get("promotion_min_sensitivity_support", "N/A")} |
| Sensitivity group support | {snapshot.get("promotion_sensitivity_group_support_count", snapshot.get("promotion_sensitivity_support_count", "N/A"))} |
| Sensitivity run support | {snapshot.get("promotion_sensitivity_run_support_count", snapshot.get("promotion_sensitivity_support_count", "N/A"))} |
| Evidence group support | {snapshot.get("promotion_evidence_support_count", "N/A")} |
| Return edge | {_pct(snapshot.get("promotion_return_edge"))} |
| Sharpe edge | {_num(snapshot.get("promotion_sharpe_edge"))} |
| Drawdown change | {_pct(snapshot.get("promotion_drawdown_change"))} |

## Review Notes

- Fixed v2 satellite allocation is not promoted because it underperformed the defensive core.
- The accepted second-stage posture is allocator-gated satellite exposure, not a fixed satellite weight.
- The latest selected candidate allows satellite only in risk-on regimes, with the configured risk-on satellite weight shown above.
- Promotion candidates stay on watch unless the promotion review decision is `promote_candidate`.
- If any component becomes incomplete or stale, fall back to the defensive core review.

## Files

- Components: `{snapshot.get("components_path")}`
- Allocator directory: `{snapshot.get("allocator_dir")}`
- Selected params: `{snapshot.get("selected_params_path", "N/A")}`
- Promotion review: `{snapshot.get("promotion_report_path", "N/A")}`
"""


def run_phase2_review(
    project_root: str | Path = Path("."),
    output_dir: str | Path | None = None,
    components_path: str | Path | None = None,
    allocator_dir: str | Path | None = None,
    promotion_review_dir: str | Path | None = None,
) -> Phase2ReviewResult:
    root = Path(project_root).resolve()
    resolved_output = _resolve(root, output_dir, Path("outputs/research/phase2_monitoring_review_latest"))
    snapshot = build_phase2_review_snapshot(
        root,
        components_path=components_path,
        allocator_dir=allocator_dir,
        promotion_review_dir=promotion_review_dir,
    )
    resolved_output.mkdir(parents=True, exist_ok=True)
    snapshot_path = resolved_output / "phase2_review_snapshot.json"
    report_path = resolved_output / "phase2_review.md"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot), encoding="utf-8")
    return Phase2ReviewResult(
        output_dir=resolved_output,
        report_path=report_path,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )
