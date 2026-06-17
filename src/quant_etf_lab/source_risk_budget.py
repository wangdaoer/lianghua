"""Research-only review for fragile source edges and satellite risk budget."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SourceRiskBudgetReviewResult:
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
    "test_core_total_return",
    "test_excess_return_vs_core",
    "test_max_drawdown",
    "test_core_max_drawdown",
    "test_drawdown_change_vs_core",
    "test_sharpe",
    "test_average_satellite_weight",
    "test_satellite_active_day_ratio",
    "test_satellite_filter_off_day_ratio",
}

REQUIRED_CANDIDATE_COLUMNS = {"window", "candidate", "source_name", "allocation_candidate", "score"}

NUMERIC_COLUMNS = {
    "selected_score",
    "raw_best_score",
    "test_total_return",
    "test_core_total_return",
    "test_excess_return_vs_core",
    "test_max_drawdown",
    "test_core_max_drawdown",
    "test_drawdown_change_vs_core",
    "test_sharpe",
    "test_average_satellite_weight",
    "test_satellite_active_day_ratio",
    "test_satellite_filter_off_day_ratio",
    "test_risk_on_day_ratio",
    "test_risk_off_day_ratio",
    "test_crash_day_ratio",
    "score",
    "train_score",
    "validation_score",
}


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    for column in NUMERIC_COLUMNS & set(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _source_score_table(candidates: pd.DataFrame) -> pd.DataFrame:
    data = candidates.copy()
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data = data.dropna(subset=["window", "source_name", "score"])
    if data.empty:
        return pd.DataFrame(columns=["window", "source_name", "source_score", "source_rank", "best_candidate"])
    ordered = data.sort_values(["window", "source_name", "score"], ascending=[True, True, False])
    best = ordered.groupby(["window", "source_name"], as_index=False).first()
    best = best.rename(columns={"candidate": "best_candidate", "score": "source_score"})
    best["source_rank"] = best.groupby("window")["source_score"].rank(method="min", ascending=False).astype(int)
    return best[["window", "source_name", "source_score", "source_rank", "best_candidate"]]


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


def _as_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.2f}%"


def _num(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.3f}"


def _optional_guard_decomposition(guard_decomposition_dir: str | Path | None) -> pd.DataFrame:
    if guard_decomposition_dir in (None, ""):
        return pd.DataFrame(columns=["window", "guard_changed_by", "guard_return_delta_vs_default", "guard_recommendation"])
    path = Path(guard_decomposition_dir) / "source_guard_decomposition_windows.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing source guard decomposition CSV: {path}")
    frame = pd.read_csv(path)
    if "window" not in frame.columns:
        raise ValueError(f"{path} missing columns: ['window']")
    rename = {
        "changed_by": "guard_changed_by",
        "return_delta_vs_default": "guard_return_delta_vs_default",
        "recommendation": "guard_recommendation",
    }
    columns = ["window", *[column for column in rename if column in frame.columns]]
    out = frame[columns].rename(columns=rename)
    if "guard_return_delta_vs_default" in out.columns:
        out["guard_return_delta_vs_default"] = pd.to_numeric(out["guard_return_delta_vs_default"], errors="coerce")
    return out


def _with_source_edges(summary: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    scores = _source_score_table(candidates)
    selected = scores[["window", "source_name", "source_score", "source_rank"]].rename(
        columns={
            "source_name": "selected_source",
            "source_score": "selected_source_score",
            "source_rank": "selected_source_rank",
        }
    )
    result = summary.merge(selected, on=["window", "selected_source"], how="left")
    competitor_rows: list[dict[str, Any]] = []
    for row in result[["window", "selected_source"]].drop_duplicates().itertuples(index=False):
        competitors = scores[
            (scores["window"].astype(str) == str(row.window))
            & (scores["source_name"].astype(str) != str(row.selected_source))
        ].sort_values(["source_score", "source_name"], ascending=[False, True])
        if competitors.empty:
            competitor_rows.append({"window": row.window, "second_source": pd.NA, "second_source_score": pd.NA})
            continue
        competitor = competitors.iloc[0]
        competitor_rows.append(
            {
                "window": row.window,
                "second_source": competitor["source_name"],
                "second_source_score": competitor["source_score"],
            }
        )
    second = pd.DataFrame(competitor_rows)
    result = result.merge(second, on="window", how="left")
    result["source_edge_vs_second"] = result["selected_source_score"] - result["second_source_score"]
    return result


def _recommendation(row: pd.Series, fragile_source_edge: float, max_low_risk_satellite_weight: float, min_filter_off_ratio: float) -> str:
    edge = _as_float(row.get("source_edge_vs_second"))
    average_satellite = _as_float(row.get("test_average_satellite_weight")) or 0.0
    filter_off_ratio = _as_float(row.get("test_satellite_filter_off_day_ratio")) or 0.0
    excess_return = _as_float(row.get("test_excess_return_vs_core")) or 0.0
    drawdown_change = _as_float(row.get("test_drawdown_change_vs_core")) or 0.0
    if str(row.get("selected_source") or "") == "core_only" or average_satellite <= 1e-12:
        return "no_fragility_action"
    if edge is None or edge > fragile_source_edge:
        return "no_fragility_action"
    already_low = average_satellite <= max_low_risk_satellite_weight or filter_off_ratio >= min_filter_off_ratio
    if already_low and excess_return >= -1e-12 and drawdown_change >= -1e-12:
        return "observe_without_source_switch"
    if average_satellite > max_low_risk_satellite_weight:
        return "review_satellite_budget_cap"
    if excess_return < -1e-12 or drawdown_change < -1e-12:
        return "review_market_state_filter"
    return "monitor_fragile_source"


def _build_windows(
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
    guard: pd.DataFrame,
    fragile_source_edge: float,
    max_low_risk_satellite_weight: float,
    min_filter_off_ratio: float,
) -> pd.DataFrame:
    data = _with_source_edges(summary, candidates)
    average_satellite = pd.to_numeric(data["test_average_satellite_weight"], errors="coerce").fillna(0.0)
    uses_satellite = (data["selected_source"].astype(str) != "core_only") & (average_satellite > 1e-12)
    data["fragile_source_edge"] = (
        pd.to_numeric(data["source_edge_vs_second"], errors="coerce") <= float(fragile_source_edge)
    ) & uses_satellite
    data["already_low_satellite_risk"] = (
        average_satellite <= float(max_low_risk_satellite_weight)
    ) | (
        pd.to_numeric(data["test_satellite_filter_off_day_ratio"], errors="coerce").fillna(0.0)
        >= float(min_filter_off_ratio)
    )
    data["risk_budget_recommendation"] = data.apply(
        _recommendation,
        axis=1,
        fragile_source_edge=fragile_source_edge,
        max_low_risk_satellite_weight=max_low_risk_satellite_weight,
        min_filter_off_ratio=min_filter_off_ratio,
    )
    if not guard.empty:
        data = data.merge(guard, on="window", how="left")
    for column in ("guard_changed_by", "guard_return_delta_vs_default", "guard_recommendation"):
        if column not in data.columns:
            data[column] = pd.NA
    output_columns = [
        "window",
        "test_start",
        "test_end",
        "selected_candidate",
        "selected_source",
        "selected_score",
        "second_source",
        "second_source_score",
        "source_edge_vs_second",
        "fragile_source_edge",
        "test_total_return",
        "test_core_total_return",
        "test_excess_return_vs_core",
        "test_max_drawdown",
        "test_core_max_drawdown",
        "test_drawdown_change_vs_core",
        "test_sharpe",
        "test_average_satellite_weight",
        "test_satellite_active_day_ratio",
        "test_satellite_filter_off_day_ratio",
        "test_risk_on_day_ratio",
        "test_risk_off_day_ratio",
        "test_crash_day_ratio",
        "already_low_satellite_risk",
        "guard_changed_by",
        "guard_return_delta_vs_default",
        "guard_recommendation",
        "risk_budget_recommendation",
    ]
    for column in output_columns:
        if column not in data.columns:
            data[column] = pd.NA
    return data[output_columns]


def _decision(windows: pd.DataFrame) -> str:
    cap_count = int((windows["risk_budget_recommendation"] == "review_satellite_budget_cap").sum())
    market_filter_count = int((windows["risk_budget_recommendation"] == "review_market_state_filter").sum())
    observe_count = int((windows["risk_budget_recommendation"] == "observe_without_source_switch").sum())
    if cap_count > 0:
        return "review_satellite_budget_cap"
    if market_filter_count > 0:
        return "review_market_state_filter"
    if observe_count > 0:
        return "keep_default_review_market_state_budget"
    return "no_source_risk_budget_change"


def _render_report(snapshot: dict[str, Any], windows: pd.DataFrame) -> str:
    rows = [
        "| Window | Source | Edge | Avg sat | Filter off | Excess vs core | Drawdown change | Guard delta | Recommendation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    flagged = windows[windows["risk_budget_recommendation"] != "no_fragility_action"]
    if flagged.empty:
        rows.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | no_fragility_action |")
    else:
        for row in flagged.itertuples(index=False):
            rows.append(
                "| {window} | {source} | {edge} | {avg_sat} | {filter_off} | {excess} | {dd_change} | {guard_delta} | {rec} |".format(
                    window=row.window,
                    source=row.selected_source,
                    edge=_num(row.source_edge_vs_second),
                    avg_sat=_pct(row.test_average_satellite_weight),
                    filter_off=_pct(row.test_satellite_filter_off_day_ratio),
                    excess=_pct(row.test_excess_return_vs_core),
                    dd_change=_pct(row.test_drawdown_change_vs_core),
                    guard_delta=_pct(row.guard_return_delta_vs_default),
                    rec=row.risk_budget_recommendation,
                )
            )
    return f"""# Source Risk Budget Review

Generated at: `{snapshot["generated_at"]}`

This report reviews source-selection fragility and satellite risk-budget diagnostics only. It does not connect to brokers, place orders, or change model defaults.

## Summary

| Item | Value |
| --- | ---: |
| Windows reviewed | {snapshot["window_count"]} |
| Fragile source windows | {snapshot["fragile_window_count"]} |
| Low-risk fragile windows | {snapshot["low_risk_fragile_window_count"]} |
| Budget-cap review windows | {snapshot["budget_cap_review_count"]} |
| Market-state filter review windows | {snapshot["market_state_filter_review_count"]} |
| Decision | `{snapshot["decision"]}` |

## Flagged Windows

{chr(10).join(rows)}

## Files

- `source_risk_budget_windows.csv`: per-window source edge, satellite exposure, guard comparison, and recommendation.
- `source_risk_budget_snapshot.json`: machine-readable summary.
"""


def run_source_risk_budget_review(
    source_dir: str | Path,
    guard_decomposition_dir: str | Path | None = None,
    output_dir: str | Path = "outputs/research/source_risk_budget_latest",
    fragile_source_edge: float = 0.03,
    max_low_risk_satellite_weight: float = 0.05,
    min_filter_off_ratio: float = 0.50,
) -> SourceRiskBudgetReviewResult:
    source_path = Path(source_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = _read_csv(source_path / "portfolio_walk_forward_summary.csv", REQUIRED_SUMMARY_COLUMNS)
    candidates = _read_csv(source_path / "portfolio_candidate_results.csv", REQUIRED_CANDIDATE_COLUMNS)
    guard = _optional_guard_decomposition(guard_decomposition_dir)
    windows = _build_windows(
        summary,
        candidates,
        guard,
        fragile_source_edge,
        max_low_risk_satellite_weight,
        min_filter_off_ratio,
    )
    fragile = windows[windows["fragile_source_edge"].fillna(False)]
    low_risk_fragile = fragile[fragile["already_low_satellite_risk"].fillna(False)]
    budget_cap = windows[windows["risk_budget_recommendation"] == "review_satellite_budget_cap"]
    market_filter = windows[windows["risk_budget_recommendation"] == "review_market_state_filter"]
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_path),
        "guard_decomposition_dir": str(Path(guard_decomposition_dir)) if guard_decomposition_dir else "",
        "window_count": int(len(windows)),
        "fragile_window_count": int(len(fragile)),
        "low_risk_fragile_window_count": int(len(low_risk_fragile)),
        "budget_cap_review_count": int(len(budget_cap)),
        "market_state_filter_review_count": int(len(market_filter)),
        "fragile_source_edge": float(fragile_source_edge),
        "max_low_risk_satellite_weight": float(max_low_risk_satellite_weight),
        "min_filter_off_ratio": float(min_filter_off_ratio),
        "decision": _decision(windows),
    }
    windows_path = out_dir / "source_risk_budget_windows.csv"
    snapshot_path = out_dir / "source_risk_budget_snapshot.json"
    report_path = out_dir / "source_risk_budget_review.md"
    windows.to_csv(windows_path, index=False, encoding="utf-8-sig")
    snapshot_path.write_text(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, windows), encoding="utf-8")
    return SourceRiskBudgetReviewResult(
        output_dir=out_dir,
        report_path=report_path,
        windows_path=windows_path,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )
