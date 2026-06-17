"""Explain portfolio source-selection decisions window by window."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SourceDecisionReviewResult:
    output_dir: Path
    report_path: Path
    windows_path: Path
    source_scores_path: Path
    snapshot_path: Path
    snapshot: dict[str, Any]


def _format_pct(value: Any) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_float(value: Any) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _read_required_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame


def _source_score_table(candidates: pd.DataFrame) -> pd.DataFrame:
    data = candidates.copy()
    for column in ("score", "train_score", "validation_score"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["window", "source_name", "score"])
    if data.empty:
        return pd.DataFrame(
            columns=[
                "window",
                "source_name",
                "best_candidate",
                "best_allocation",
                "source_score",
                "source_rank",
                "train_score",
                "validation_score",
            ]
        )
    ordered = data.sort_values(["window", "source_name", "score"], ascending=[True, True, False])
    best = ordered.groupby(["window", "source_name"], as_index=False).first()
    best = best.rename(
        columns={
            "candidate": "best_candidate",
            "allocation_candidate": "best_allocation",
            "score": "source_score",
        }
    )
    best["source_rank"] = best.groupby("window")["source_score"].rank(method="min", ascending=False).astype(int)
    return best[
        [
            "window",
            "source_name",
            "best_candidate",
            "best_allocation",
            "source_score",
            "source_rank",
            "train_score",
            "validation_score",
        ]
    ].sort_values(["window", "source_rank"])


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _selection_reason(row: pd.Series) -> str:
    if _truthy(row.get("source_stability_applied", False)):
        return "stability_penalty"
    if str(row.get("selected_source", "")) == str(row.get("raw_best_source", "")):
        return "raw_best_source"
    default_source = str(row.get("default_source", "") or "")
    margin = float(row.get("source_switch_margin", 0.0) or 0.0)
    if default_source and margin > 0 and row.get("selected_source") in {default_source, "core_only"}:
        return "default_source_margin"
    return "guarded_override"


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


def run_source_decision_review(
    source_dir: str | Path,
    baseline_dir: str | Path | None = None,
    output_dir: str | Path = "outputs/research/source_decision_review_latest",
) -> SourceDecisionReviewResult:
    source_path = Path(source_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_required_csv(
        source_path / "portfolio_walk_forward_summary.csv",
        {
            "window",
            "selected_candidate",
            "selected_source",
            "selected_score",
            "raw_best_source",
            "raw_best_score",
            "test_total_return",
            "test_max_drawdown",
            "test_sharpe",
        },
    )
    candidates = _read_required_csv(
        source_path / "portfolio_candidate_results.csv",
        {"window", "candidate", "source_name", "allocation_candidate", "score"},
    )

    for column in ("selected_score", "raw_best_score", "test_total_return", "test_max_drawdown", "test_sharpe"):
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    source_scores = _source_score_table(candidates)

    baseline = None
    if baseline_dir is not None:
        baseline_path = Path(baseline_dir)
        baseline = _read_required_csv(
            baseline_path / "portfolio_walk_forward_summary.csv",
            {"window", "test_total_return", "test_max_drawdown", "test_sharpe"},
        )
        baseline = baseline[["window", "test_total_return", "test_max_drawdown", "test_sharpe"]].rename(
            columns={
                "test_total_return": "baseline_test_total_return",
                "test_max_drawdown": "baseline_test_max_drawdown",
                "test_sharpe": "baseline_test_sharpe",
            }
        )
        for column in ("baseline_test_total_return", "baseline_test_max_drawdown", "baseline_test_sharpe"):
            baseline[column] = pd.to_numeric(baseline[column], errors="coerce")

    windows = summary.copy()
    windows["selection_reason"] = windows.apply(_selection_reason, axis=1)
    selected_scores = source_scores[["window", "source_name", "source_score", "source_rank"]].rename(
        columns={
            "source_name": "selected_source",
            "source_score": "selected_source_score",
            "source_rank": "selected_source_rank",
        }
    )
    windows = windows.merge(selected_scores, on=["window", "selected_source"], how="left")
    competitor_rows: list[dict[str, Any]] = []
    for row in windows[["window", "selected_source"]].drop_duplicates().itertuples(index=False):
        competitors = source_scores[
            (source_scores["window"] == row.window)
            & (source_scores["source_name"] != row.selected_source)
        ].sort_values(["source_score", "source_name"], ascending=[False, True])
        if competitors.empty:
            continue
        competitor = competitors.iloc[0]
        competitor_rows.append(
            {
                "window": row.window,
                "second_source": competitor["source_name"],
                "second_source_score": competitor["source_score"],
            }
        )
    second_scores = pd.DataFrame(competitor_rows, columns=["window", "second_source", "second_source_score"])
    windows = windows.merge(second_scores, on="window", how="left")
    windows["score_edge_vs_second"] = windows["selected_source_score"] - windows["second_source_score"]
    windows["raw_best_matches_selection"] = windows["raw_best_source"].astype(str) == windows["selected_source"].astype(str)
    if baseline is not None:
        windows = windows.merge(baseline, on="window", how="left")
        windows["return_edge_vs_baseline"] = windows["test_total_return"] - windows["baseline_test_total_return"]
        windows["drawdown_edge_vs_baseline"] = windows["test_max_drawdown"] - windows["baseline_test_max_drawdown"]
        windows["sharpe_edge_vs_baseline"] = windows["test_sharpe"] - windows["baseline_test_sharpe"]
    else:
        windows["return_edge_vs_baseline"] = pd.NA
        windows["drawdown_edge_vs_baseline"] = pd.NA
        windows["sharpe_edge_vs_baseline"] = pd.NA

    output_columns = [
        "window",
        "test_start",
        "test_end",
        "selected_source",
        "selected_candidate",
        "selected_score",
        "selected_source_rank",
        "raw_best_source",
        "raw_best_score",
        "second_source",
        "second_source_score",
        "score_edge_vs_second",
        "selection_reason",
        "test_total_return",
        "test_max_drawdown",
        "test_sharpe",
        "return_edge_vs_baseline",
        "drawdown_edge_vs_baseline",
        "sharpe_edge_vs_baseline",
    ]
    for column in output_columns:
        if column not in windows.columns:
            windows[column] = pd.NA
    windows_out = windows[output_columns]

    windows_path = out_dir / "source_decision_windows.csv"
    source_scores_path = out_dir / "source_decision_source_scores.csv"
    snapshot_path = out_dir / "source_decision_snapshot.json"
    report_path = out_dir / "source_decision_review.md"
    windows_out.to_csv(windows_path, index=False, encoding="utf-8-sig")
    source_scores.to_csv(source_scores_path, index=False, encoding="utf-8-sig")

    source_counts = windows_out["selected_source"].value_counts().to_dict()
    return_wins = int((pd.to_numeric(windows_out["return_edge_vs_baseline"], errors="coerce") > 0).sum())
    raw_best_match_count = int(windows["raw_best_matches_selection"].sum())
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_path),
        "baseline_dir": str(Path(baseline_dir)) if baseline_dir is not None else "",
        "window_count": int(len(windows_out)),
        "selected_source_counts": source_counts,
        "raw_best_match_count": raw_best_match_count,
        "baseline_return_win_count": return_wins if baseline is not None else None,
        "decision": "review_source_decisions",
    }
    snapshot_path.write_text(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")

    table = [
        "| Window | Selected source | Score rank | Score edge | OOS return | Return edge vs baseline | Reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in windows_out.itertuples(index=False):
        table.append(
            "| {window} | {source} | {rank} | {edge} | {ret} | {ret_edge} | {reason} |".format(
                window=row.window,
                source=row.selected_source,
                rank=_format_float(row.selected_source_rank),
                edge=_format_float(row.score_edge_vs_second),
                ret=_format_pct(row.test_total_return),
                ret_edge=_format_pct(row.return_edge_vs_baseline),
                reason=row.selection_reason,
            )
        )

    score_lines = [
        "| Window | Rank | Source | Score | Best candidate |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for row in source_scores.itertuples(index=False):
        if int(row.source_rank) > 3:
            continue
        score_lines.append(
            "| {window} | {rank} | {source} | {score} | {candidate} |".format(
                window=row.window,
                rank=int(row.source_rank),
                source=row.source_name,
                score=_format_float(row.source_score),
                candidate=row.best_candidate,
            )
        )

    body = f"""# Source Decision Review

Generated at: `{snapshot["generated_at"]}`

This review explains portfolio source-selection outputs only. It does not connect to brokers, place orders, or provide investment advice.

## Summary

| Item | Value |
| --- | ---: |
| Windows reviewed | {snapshot["window_count"]} |
| Raw-best selections | {snapshot["raw_best_match_count"]} |
| Return wins vs baseline | {snapshot["baseline_return_win_count"] if snapshot["baseline_return_win_count"] is not None else "n/a"} |

Selected source counts: `{json.dumps(source_counts, ensure_ascii=False, sort_keys=True)}`

## Window Decisions

{chr(10).join(table)}

## Source Score Leaders

{chr(10).join(score_lines)}

## Files

- `source_decision_windows.csv`: selected source, ranking, edges, OOS metrics, and reason by window.
- `source_decision_source_scores.csv`: best candidate per source and source ranking by window.
- `source_decision_snapshot.json`: machine-readable summary.
"""
    report_path.write_text(body, encoding="utf-8")

    return SourceDecisionReviewResult(
        output_dir=out_dir,
        report_path=report_path,
        windows_path=windows_path,
        source_scores_path=source_scores_path,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )
