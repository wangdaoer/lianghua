"""Daily after-close research checks for the phase-2 model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .phase2_review import (
    DEFAULT_ALLOCATOR_DIR,
    DEFAULT_PHASE2_COMPONENTS,
    Phase2ReviewResult,
    run_phase2_review,
)


DEFAULT_DAILY_CHECK_OUTPUT = Path("outputs/research/daily_model_check_latest")
DEFAULT_DAILY_CHECK_BASE = Path("outputs/research/daily_model_check")


@dataclass(frozen=True)
class DailyModelCheckResult:
    output_dir: Path
    report_path: Path
    snapshot_path: Path
    phase2_result: Phase2ReviewResult
    snapshot: dict[str, Any]


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = Path(path) if path is not None else default
    return raw if raw.is_absolute() else project_root / raw


def _resolve_output(
    project_root: Path,
    output_dir: str | Path | None,
    as_of: date,
    date_stamp: bool,
) -> Path:
    default = DEFAULT_DAILY_CHECK_BASE if date_stamp else DEFAULT_DAILY_CHECK_OUTPUT
    raw = Path(output_dir) if output_dir is not None else default
    if date_stamp:
        stamp = as_of.strftime("%Y%m%d")
        if not raw.name.endswith(f"_{stamp}"):
            raw = raw.with_name(f"{raw.name}_{stamp}")
    return raw if raw.is_absolute() else project_root / raw


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).date()


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not pd.notna(number):
        return "N/A"
    return f"{number * 100:.2f}%"


def _num(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not pd.notna(number):
        return "N/A"
    return f"{number:.{digits}f}"


def _freshness_status(latest_date: date | None, as_of: date, max_staleness_days: int) -> tuple[str, int | None]:
    if latest_date is None:
        return "missing_latest_date", None
    days = (as_of - latest_date).days
    if days < 0:
        return "future_dated_local_data", days
    if days <= max_staleness_days:
        return "fresh_enough", days
    return "stale", days


def _action_posture(phase2_snapshot: dict[str, Any], freshness: str) -> str:
    if freshness != "fresh_enough":
        return "wait_for_fresh_local_data"
    posture = str(phase2_snapshot.get("posture", ""))
    if posture == "core_base_allocator_gated_satellite":
        return "review_core_base_allocator_gate"
    if posture == "core_only":
        return "review_defensive_core_only"
    if posture == "allocator_risk_on_satellite":
        return "review_allocator_satellite_budget"
    return "review_only_no_promotion"


def _render_report(snapshot: dict[str, Any]) -> str:
    return f"""# Daily Model Check

Generated at: `{snapshot.get("generated_at")}`

This file is a research-only after-close check. It does not connect to brokers, place orders, or provide investment advice.

## Decision State

| Item | Value |
| --- | ---: |
| As-of date | {snapshot.get("as_of_date", "N/A")} |
| Local equity latest date | {snapshot.get("latest_date", "N/A")} |
| Days since local equity date | {snapshot.get("days_since_latest", "N/A")} |
| Data freshness | {snapshot.get("data_freshness_status", "N/A")} |
| Phase-2 posture | {snapshot.get("phase2_posture", "N/A")} |
| Daily action posture | {snapshot.get("action_posture", "N/A")} |
| Selected allocator candidate | {snapshot.get("selected_candidate", "N/A")} |
| Risk-on satellite weight | {_pct(snapshot.get("risk_on_satellite_weight"))} |
| Current drawdown | {_pct(snapshot.get("current_drawdown"))} |
| 20-trading-day return | {_pct(snapshot.get("return_20d"))} |
| Allocator Sharpe | {_num(snapshot.get("allocator_sharpe"))} |
| Promotion decision | {snapshot.get("promotion_decision", "N/A")} |
| Promotion return edge | {_pct(snapshot.get("promotion_return_edge"))} |
| Promotion Sharpe edge | {_num(snapshot.get("promotion_sharpe_edge"))} |
| Promotion independent support groups | {snapshot.get("promotion_support_group_count", snapshot.get("promotion_sensitivity_group_support_count", snapshot.get("promotion_sensitivity_support_count", "N/A")))} / {snapshot.get("promotion_min_sensitivity_support", "N/A")} |
| Promotion sensitivity group support | {snapshot.get("promotion_sensitivity_group_support_count", snapshot.get("promotion_sensitivity_support_count", "N/A"))} |
| Promotion evidence group support | {snapshot.get("promotion_evidence_support_count", "N/A")} |
| Promotion sensitivity run support | {snapshot.get("promotion_sensitivity_run_support_count", snapshot.get("promotion_sensitivity_support_count", "N/A"))} |

## Review Checklist

- If data freshness is not `fresh_enough`, refresh local market data before trusting model posture.
- If posture is `core_base_allocator_gated_satellite`, keep the defensive core as the base and only review satellite exposure through the allocator gate.
- If fixed portfolio return remains below the core return, do not promote the fixed satellite allocation.
- If promotion decision is `watch_candidate`, keep the candidate visible but do not replace the default allocator.
- If promotion decision is `promote_candidate`, keep promotion evidence visible, monitor outcome samples, and retain the previous allocator as a rollback baseline.
- Keep this as a model-status review; use separate stock-level trigger reports for individual watchlist alerts.

## Source Files

- Phase-2 review report: `{snapshot.get("phase2_report_path")}`
- Phase-2 review snapshot: `{snapshot.get("phase2_snapshot_path")}`
- Components: `{snapshot.get("components_path")}`
- Allocator directory: `{snapshot.get("allocator_dir")}`
- Promotion review: `{snapshot.get("promotion_report_path")}`
"""


def run_daily_model_check(
    project_root: str | Path = Path("."),
    output_dir: str | Path | None = None,
    phase2_review_dir: str | Path | None = None,
    components_path: str | Path | None = None,
    allocator_dir: str | Path | None = None,
    promotion_review_dir: str | Path | None = None,
    max_staleness_days: int = 3,
    as_of_date: str | date | None = None,
    date_stamp: bool = False,
) -> DailyModelCheckResult:
    root = Path(project_root).resolve()
    as_of = _parse_date(as_of_date) if as_of_date is not None else datetime.now().date()
    if as_of is None:
        raise ValueError(f"Invalid as_of_date: {as_of_date}")
    resolved_output = _resolve_output(root, output_dir, as_of, date_stamp)
    resolved_phase2_dir = _resolve(root, phase2_review_dir, resolved_output / "phase2_review")
    resolved_components = _resolve(root, components_path, DEFAULT_PHASE2_COMPONENTS)
    resolved_allocator = _resolve(root, allocator_dir, DEFAULT_ALLOCATOR_DIR)

    phase2_result = run_phase2_review(
        project_root=root,
        output_dir=resolved_phase2_dir,
        components_path=resolved_components,
        allocator_dir=resolved_allocator,
        promotion_review_dir=promotion_review_dir,
    )
    phase2_snapshot = phase2_result.snapshot
    latest = _parse_date(phase2_snapshot.get("latest_date"))
    freshness, days_since_latest = _freshness_status(latest, as_of, max_staleness_days)
    snapshot: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": as_of.isoformat(),
        "latest_date": latest.isoformat() if latest is not None else None,
        "days_since_latest": days_since_latest,
        "max_staleness_days": int(max_staleness_days),
        "data_freshness_status": freshness,
        "phase2_posture": phase2_snapshot.get("posture"),
        "action_posture": _action_posture(phase2_snapshot, freshness),
        "selected_candidate": phase2_snapshot.get("selected_candidate"),
        "risk_on_core_weight": phase2_snapshot.get("risk_on_core_weight"),
        "risk_on_satellite_weight": phase2_snapshot.get("risk_on_satellite_weight"),
        "risk_off_satellite_weight": phase2_snapshot.get("risk_off_satellite_weight"),
        "current_drawdown": phase2_snapshot.get("current_drawdown"),
        "return_20d": phase2_snapshot.get("return_20d"),
        "allocator_sharpe": phase2_snapshot.get("allocator_sharpe"),
        "fixed_portfolio_total_return": phase2_snapshot.get("portfolio_total_return"),
        "core_total_return": phase2_snapshot.get("core_total_return"),
        "promotion_status": phase2_snapshot.get("promotion_status"),
        "promotion_decision": phase2_snapshot.get("promotion_decision"),
        "promotion_candidate_passes_headline_gate": phase2_snapshot.get("promotion_candidate_passes_headline_gate"),
        "promotion_sensitivity_support_count": phase2_snapshot.get("promotion_sensitivity_support_count"),
        "promotion_sensitivity_group_support_count": phase2_snapshot.get("promotion_sensitivity_group_support_count"),
        "promotion_support_group_count": phase2_snapshot.get("promotion_support_group_count"),
        "promotion_evidence_support_count": phase2_snapshot.get("promotion_evidence_support_count"),
        "promotion_sensitivity_run_support_count": phase2_snapshot.get("promotion_sensitivity_run_support_count"),
        "promotion_min_sensitivity_support": phase2_snapshot.get("promotion_min_sensitivity_support"),
        "promotion_return_edge": phase2_snapshot.get("promotion_return_edge"),
        "promotion_sharpe_edge": phase2_snapshot.get("promotion_sharpe_edge"),
        "promotion_drawdown_change": phase2_snapshot.get("promotion_drawdown_change"),
        "promotion_report_path": phase2_snapshot.get("promotion_report_path"),
        "components_path": phase2_snapshot.get("components_path", str(resolved_components)),
        "allocator_dir": phase2_snapshot.get("allocator_dir", str(resolved_allocator)),
        "phase2_report_path": str(phase2_result.report_path),
        "phase2_snapshot_path": str(phase2_result.snapshot_path),
    }

    resolved_output.mkdir(parents=True, exist_ok=True)
    snapshot_path = resolved_output / "daily_model_check_snapshot.json"
    report_path = resolved_output / "daily_model_check.md"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot), encoding="utf-8")
    return DailyModelCheckResult(
        output_dir=resolved_output,
        report_path=report_path,
        snapshot_path=snapshot_path,
        phase2_result=phase2_result,
        snapshot=snapshot,
    )
