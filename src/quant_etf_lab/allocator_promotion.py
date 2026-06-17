"""Promotion-gate review for portfolio allocator candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AllocatorPromotionReviewResult:
    output_dir: Path
    report_path: Path
    comparison_path: Path
    snapshot_path: Path
    decision: str
    snapshot: dict[str, Any]


def _read_allocator_metrics(run_dir: Path, label: str) -> dict[str, Any]:
    equity_path = run_dir / "oos_equity_stitched.csv"
    summary_path = run_dir / "portfolio_walk_forward_summary.csv"
    if not equity_path.exists():
        raise FileNotFoundError(f"Missing stitched equity: {equity_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing portfolio summary: {summary_path}")

    equity = pd.read_csv(equity_path)
    if {"date", "stitched_equity"} - set(equity.columns):
        raise ValueError(f"{equity_path} must contain date and stitched_equity columns.")
    series = pd.to_numeric(equity["stitched_equity"], errors="coerce").dropna()
    if len(series) < 2:
        raise ValueError(f"{equity_path} must contain at least two equity values.")
    returns = series.pct_change().dropna()
    running_max = series.cummax()
    drawdown = series / running_max - 1.0
    sharpe = 0.0
    if len(returns) > 1 and float(returns.std()) != 0.0:
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std())

    summary = pd.read_csv(summary_path)
    test_returns = pd.to_numeric(summary.get("test_total_return", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "label": label,
        "run_dir": str(run_dir),
        "total_return": float(series.iloc[-1] / series.iloc[0] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "window_count": int(len(test_returns)),
        "positive_window_ratio": float((test_returns > 0).mean()) if len(test_returns) else 0.0,
        "worst_window_return": float(test_returns.min()) if len(test_returns) else 0.0,
    }


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "support", "supports_candidate"}
    return False


def _evidence_supports_candidate(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status", "")).strip().lower()
    return (
        status == "supports_candidate"
        or _truthy_flag(payload.get("supports_candidate"))
        or _truthy_flag(payload.get("all_cost_rates_support_candidate"))
    )


def _read_evidence_snapshot(snapshot_path: Path, label: str, group: str) -> dict[str, Any]:
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    supports_candidate = _evidence_supports_candidate(payload)
    return {
        "label": label,
        "run_dir": str(snapshot_path),
        "evidence_snapshot_path": str(snapshot_path),
        "sensitivity_group": group,
        "total_return": None,
        "max_drawdown": None,
        "sharpe": None,
        "window_count": None,
        "positive_window_ratio": None,
        "worst_window_return": None,
        "supports_candidate": supports_candidate,
        "evidence_status": payload.get("status", ""),
        "evidence_return_edge": _optional_float(payload.get("min_return_edge", payload.get("return_edge"))),
        "evidence_sharpe_edge": _optional_float(payload.get("min_sharpe_edge", payload.get("sharpe_edge"))),
        "evidence_drawdown_change": _optional_float(payload.get("min_drawdown_change", payload.get("drawdown_change"))),
    }


def _support_passes(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    min_return_edge: float,
    min_sharpe_edge: float,
    max_drawdown_worsening: float,
) -> bool:
    return (
        float(metrics["total_return"]) >= float(baseline["total_return"]) + min_return_edge
        and float(metrics["sharpe"]) >= float(baseline["sharpe"]) + min_sharpe_edge
        and float(metrics["max_drawdown"]) >= float(baseline["max_drawdown"]) - max_drawdown_worsening
    )


def run_allocator_promotion_review(
    baseline_dir: str | Path,
    candidate_dir: str | Path,
    sensitivity_dirs: list[str | Path] | tuple[str | Path, ...] = (),
    sensitivity_groups: list[str] | tuple[str, ...] | None = None,
    evidence_snapshots: list[str | Path] | tuple[str | Path, ...] = (),
    evidence_groups: list[str] | tuple[str, ...] | None = None,
    output_dir: str | Path = "outputs/research/allocator_promotion_review_latest",
    min_return_edge: float = 0.03,
    min_sharpe_edge: float = 0.05,
    max_drawdown_worsening: float = 0.00,
    min_positive_window_ratio: float = 0.80,
    min_sensitivity_support: int = 1,
) -> AllocatorPromotionReviewResult:
    baseline_path = Path(baseline_dir)
    candidate_path = Path(candidate_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sensitivity_paths = [Path(path) for path in sensitivity_dirs]
    if sensitivity_groups is None:
        group_labels = [f"sensitivity_{index:02d}" for index in range(1, len(sensitivity_paths) + 1)]
    else:
        group_labels = [str(group) for group in sensitivity_groups]
        if len(group_labels) != len(sensitivity_paths):
            raise ValueError("sensitivity_groups must have the same length as sensitivity_dirs.")
    evidence_paths = [Path(path) for path in evidence_snapshots]
    if evidence_groups is None:
        evidence_group_labels = [f"evidence_{index:02d}" for index in range(1, len(evidence_paths) + 1)]
    else:
        evidence_group_labels = [str(group) for group in evidence_groups]
        if len(evidence_group_labels) != len(evidence_paths):
            raise ValueError("evidence_groups must have the same length as evidence_snapshots.")

    baseline = _read_allocator_metrics(baseline_path, "baseline")
    candidate = _read_allocator_metrics(candidate_path, "candidate")
    sensitivities = []
    for index, path in enumerate(sensitivity_paths, start=1):
        row = _read_allocator_metrics(path, f"sensitivity_{index:02d}")
        row["sensitivity_group"] = group_labels[index - 1]
        sensitivities.append(row)
    evidence_rows = [
        _read_evidence_snapshot(path, f"evidence_{index:02d}", evidence_group_labels[index - 1])
        for index, path in enumerate(evidence_paths, start=1)
    ]

    candidate_passes = _support_passes(
        candidate,
        baseline,
        min_return_edge=min_return_edge,
        min_sharpe_edge=min_sharpe_edge,
        max_drawdown_worsening=max_drawdown_worsening,
    ) and float(candidate["positive_window_ratio"]) >= min_positive_window_ratio
    support_rows = []
    for row in sensitivities:
        passes = _support_passes(
            row,
            baseline,
            min_return_edge=min_return_edge,
            min_sharpe_edge=min_sharpe_edge,
            max_drawdown_worsening=max_drawdown_worsening,
        ) and float(row["positive_window_ratio"]) >= min_positive_window_ratio
        support_rows.append({**row, "supports_candidate": passes})
    run_support_count = sum(1 for row in support_rows if row["supports_candidate"])
    sensitivity_support_groups = {str(row["sensitivity_group"]) for row in support_rows if row["supports_candidate"]}
    evidence_support_groups = {str(row["sensitivity_group"]) for row in evidence_rows if row["supports_candidate"]}
    support_groups = sensitivity_support_groups | evidence_support_groups
    sensitivity_group_support_count = len(sensitivity_support_groups)
    evidence_support_count = len(evidence_support_groups)
    group_support_count = len(support_groups)

    if not candidate_passes:
        decision = "reject_candidate"
    elif group_support_count < min_sensitivity_support:
        decision = "watch_candidate"
    else:
        decision = "promote_candidate"

    comparison_rows = [
        {**baseline, "role": "baseline", "sensitivity_group": "", "supports_candidate": ""},
        {**candidate, "role": "candidate", "sensitivity_group": "", "supports_candidate": candidate_passes},
        *[{**row, "role": "sensitivity"} for row in support_rows],
        *[{**row, "role": "evidence"} for row in evidence_rows],
    ]
    comparison = pd.DataFrame(comparison_rows)
    comparison_path = out_dir / "allocator_promotion_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision": decision,
        "baseline_dir": str(baseline_path),
        "candidate_dir": str(candidate_path),
        "sensitivity_dirs": [str(path) for path in sensitivity_paths],
        "sensitivity_groups": group_labels,
        "evidence_snapshots": [str(path) for path in evidence_paths],
        "evidence_groups": evidence_group_labels,
        "candidate_passes_headline_gate": candidate_passes,
        "sensitivity_run_support_count": run_support_count,
        "sensitivity_group_support_count": sensitivity_group_support_count,
        "evidence_support_count": evidence_support_count,
        "support_group_count": group_support_count,
        "support_groups": sorted(support_groups),
        "sensitivity_support_count": group_support_count,
        "min_sensitivity_support": int(min_sensitivity_support),
        "return_edge": float(candidate["total_return"]) - float(baseline["total_return"]),
        "sharpe_edge": float(candidate["sharpe"]) - float(baseline["sharpe"]),
        "drawdown_change": float(candidate["max_drawdown"]) - float(baseline["max_drawdown"]),
        "min_return_edge": float(min_return_edge),
        "min_sharpe_edge": float(min_sharpe_edge),
        "max_drawdown_worsening": float(max_drawdown_worsening),
        "min_positive_window_ratio": float(min_positive_window_ratio),
    }
    snapshot_path = out_dir / "allocator_promotion_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    table = [
        "| Role | Group | Total return | Max drawdown | Sharpe | Positive windows | Worst window | Supports |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in comparison_rows:
        supports = row.get("supports_candidate", "")
        if isinstance(supports, bool):
            supports = "yes" if supports else "no"
        table.append(
            "| {role} | {group} | {return_} | {drawdown} | {sharpe} | {positive} | {worst} | {supports} |".format(
                role=row["role"],
                group=row.get("sensitivity_group", ""),
                return_=_format_pct(row.get("total_return")),
                drawdown=_format_pct(row.get("max_drawdown")),
                sharpe=_format_float(row.get("sharpe")),
                positive=_format_pct(row.get("positive_window_ratio")),
                worst=_format_pct(row.get("worst_window_return")),
                supports=supports,
            )
        )

    evidence_table = ["No external evidence snapshots supplied."]
    if evidence_rows:
        evidence_table = [
            "| Group | Snapshot | Status | Return edge | Sharpe edge | Drawdown change | Supports |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
        for row in evidence_rows:
            evidence_table.append(
                "| {group} | `{snapshot}` | `{status}` | {return_edge} | {sharpe_edge} | {drawdown_change} | {supports} |".format(
                    group=row["sensitivity_group"],
                    snapshot=row["evidence_snapshot_path"],
                    status=row.get("evidence_status", ""),
                    return_edge=_format_pct(row.get("evidence_return_edge")),
                    sharpe_edge=_format_float(row.get("evidence_sharpe_edge")),
                    drawdown_change=_format_pct(row.get("evidence_drawdown_change")),
                    supports="yes" if row["supports_candidate"] else "no",
                )
            )

    report_path = out_dir / "allocator_promotion_review.md"
    body = f"""# Allocator Promotion Review

Generated at: `{snapshot["generated_at"]}`

This review checks research allocator outputs only. It does not connect to brokers, place orders, or provide investment advice.

## Decision

`{decision}`

## Gates

| Gate | Value |
| --- | ---: |
| Candidate return edge | {_format_pct(snapshot["return_edge"])} |
| Required return edge | {_format_pct(min_return_edge)} |
| Candidate Sharpe edge | {_format_float(snapshot["sharpe_edge"])} |
| Required Sharpe edge | {_format_float(min_sharpe_edge)} |
| Candidate drawdown change | {_format_pct(snapshot["drawdown_change"])} |
| Max allowed drawdown worsening | {_format_pct(max_drawdown_worsening)} |
| Independent support groups | {group_support_count} / {int(min_sensitivity_support)} |
| Sensitivity group support | {sensitivity_group_support_count} |
| Sensitivity run support | {run_support_count} |
| Evidence group support | {evidence_support_count} |

## Comparison

{chr(10).join(table)}

## External Evidence

{chr(10).join(evidence_table)}

## Files

- `allocator_promotion_comparison.csv`: metrics for baseline, candidate, sensitivity runs, and external evidence snapshots.
- `allocator_promotion_snapshot.json`: machine-readable decision and gates.
"""
    report_path.write_text(body, encoding="utf-8")

    return AllocatorPromotionReviewResult(
        output_dir=out_dir,
        report_path=report_path,
        comparison_path=comparison_path,
        snapshot_path=snapshot_path,
        decision=decision,
        snapshot=snapshot,
    )
