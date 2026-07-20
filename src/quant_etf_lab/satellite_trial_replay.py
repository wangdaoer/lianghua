"""Research-only replay for satellite trial rules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/research/satellite_trial_replay_latest")
DEFAULT_RULES_PATH = Path("outputs/research/satellite_risk_budget_review_latest/satellite_trial_rules.csv")
DEFAULT_OUTCOMES_HISTORY_PATH = Path("outputs/research/stock_target_review_outcomes_history.csv")
DEFAULT_HORIZONS = ("1d", "5d", "10d")
SUMMARY_COLUMNS = [
    "scope",
    "rule_id",
    "dimension",
    "group_value",
    "horizon",
    "sample_count",
    "avg_return",
    "median_return",
    "win_rate",
    "min_return",
    "max_return",
    "baseline_sample_count",
    "baseline_avg_return",
    "baseline_win_rate",
    "avg_return_edge",
    "win_rate_edge",
    "broker_action",
    "research_only",
]


@dataclass(frozen=True)
class SatelliteTrialReplayResult:
    output_dir: Path
    summary_path: Path
    matches_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    summary: pd.DataFrame
    matches: pd.DataFrame


def _normalise_horizon(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return text if text.endswith("d") else f"{text}d"


def _horizon_sort_key(value: Any) -> tuple[int, str]:
    text = _normalise_horizon(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (int(digits), text) if digits else (9999, text)


def _normalise_text(value: Any) -> str:
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _return_column(horizon: str) -> str:
    key = _normalise_horizon(horizon)
    if not key:
        raise ValueError("horizon cannot be empty")
    return f"return_{key}"


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


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if value is pd.NA:
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_ready(record) for record in frame.to_dict(orient="records")]


def _pct(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "N/A"
    return f"{number * 100:.2f}%"


def _read_csv(path: str | Path, label: str) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return pd.read_csv(resolved, encoding="utf-8-sig", dtype=object)


def _clean_horizons(horizons: Iterable[str] | str) -> list[str]:
    if isinstance(horizons, str):
        raw = horizons.split(",")
    else:
        raw = list(horizons)
    cleaned = sorted({_normalise_horizon(item) for item in raw if _normalise_horizon(item)}, key=_horizon_sort_key)
    if not cleaned:
        raise ValueError("At least one horizon is required.")
    return cleaned


def _valid_rules(rules: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if rules.empty:
        return rules.copy(), ["rules_file_empty"]
    required = {"rule_id", "dimension", "group_value"}
    missing = sorted(required - set(rules.columns))
    if missing:
        raise ValueError(f"Rules file missing required columns: {', '.join(missing)}")
    usable = rules.copy()
    usable["rule_id"] = usable["rule_id"].map(_normalise_text)
    usable["dimension"] = usable["dimension"].map(_normalise_text)
    usable["group_value"] = usable["group_value"].map(_normalise_text)
    before = len(usable)
    usable = usable[(usable["rule_id"] != "") & (usable["dimension"] != "") & (usable["group_value"] != "")]
    dropped = before - len(usable)
    if dropped:
        warnings.append(f"dropped_incomplete_rules={dropped}")
    return usable.reset_index(drop=True), warnings


def _numeric_returns(frame: pd.DataFrame, horizon: str) -> pd.Series:
    column = _return_column(horizon)
    if column not in frame.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _stats(frame: pd.DataFrame, horizon: str) -> dict[str, Any]:
    returns = _numeric_returns(frame, horizon)
    if returns.empty:
        return {
            "sample_count": 0,
            "avg_return": None,
            "median_return": None,
            "win_rate": None,
            "min_return": None,
            "max_return": None,
        }
    return {
        "sample_count": int(len(returns)),
        "avg_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "win_rate": float((returns > 0).mean()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
    }


def _summary_row(
    *,
    scope: str,
    rule_id: str,
    dimension: str,
    group_value: str,
    horizon: str,
    stats: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    avg_return = _as_float(stats.get("avg_return"))
    win_rate = _as_float(stats.get("win_rate"))
    baseline_avg = _as_float(baseline.get("avg_return"))
    baseline_win = _as_float(baseline.get("win_rate"))
    return {
        "scope": scope,
        "rule_id": rule_id,
        "dimension": dimension,
        "group_value": group_value,
        "horizon": horizon,
        **stats,
        "baseline_sample_count": int(baseline.get("sample_count") or 0),
        "baseline_avg_return": baseline_avg,
        "baseline_win_rate": baseline_win,
        "avg_return_edge": (avg_return - baseline_avg) if avg_return is not None and baseline_avg is not None else None,
        "win_rate_edge": (win_rate - baseline_win) if win_rate is not None and baseline_win is not None else None,
        "broker_action": "none",
        "research_only": True,
    }


def _match_rules(
    rules: pd.DataFrame,
    history: pd.DataFrame,
) -> tuple[dict[str, list[int]], pd.DataFrame, list[str]]:
    warnings: list[str] = []
    rule_matches: dict[str, list[int]] = {}
    match_map: dict[int, dict[str, Any]] = {}

    for rule in rules.to_dict(orient="records"):
        rule_id = _normalise_text(rule.get("rule_id"))
        dimension = _normalise_text(rule.get("dimension"))
        group_value = _normalise_text(rule.get("group_value"))
        if dimension not in history.columns:
            warnings.append(f"missing_history_column_for_rule={rule_id}:{dimension}")
            rule_matches[rule_id] = []
            continue
        series = history[dimension].map(_normalise_text)
        matched_indices = list(history.index[series == group_value])
        rule_matches[rule_id] = matched_indices
        for index in matched_indices:
            entry = match_map.setdefault(index, {"rule_ids": [], "filters": []})
            entry["rule_ids"].append(rule_id)
            entry["filters"].append(f"{dimension}={group_value}")

    rows: list[dict[str, Any]] = []
    for index in sorted(match_map):
        row = history.loc[index].to_dict()
        row["_source_index"] = int(index)
        row["matched_rule_ids"] = ";".join(match_map[index]["rule_ids"])
        row["matched_rule_filters"] = ";".join(match_map[index]["filters"])
        row["matched_rule_count"] = len(match_map[index]["rule_ids"])
        rows.append(row)
    matches = pd.DataFrame(rows)
    return rule_matches, matches, warnings


def _render_rule_rows(summary: pd.DataFrame) -> str:
    rows = summary[summary["scope"] == "rule"].copy()
    if rows.empty:
        return "| N/A | N/A | N/A | N/A | N/A | N/A |"
    rows["horizon_sort"] = rows["horizon"].map(_horizon_sort_key)
    rows = rows.sort_values(["rule_id", "horizon_sort"])
    rendered = []
    for row in rows.itertuples(index=False):
        rendered.append(
            "| {rule_id} | `{dimension}={group_value}` | {horizon} | {sample_count} | {avg_return} | {win_rate} |".format(
                rule_id=getattr(row, "rule_id"),
                dimension=getattr(row, "dimension"),
                group_value=getattr(row, "group_value"),
                horizon=getattr(row, "horizon"),
                sample_count=getattr(row, "sample_count"),
                avg_return=_pct(getattr(row, "avg_return")),
                win_rate=_pct(getattr(row, "win_rate")),
            )
        )
    return "\n".join(rendered)


def _render_union_rows(summary: pd.DataFrame) -> str:
    rows = summary[summary["scope"].isin(["baseline_all", "rule_union"])].copy()
    if rows.empty:
        return "| N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
    rows["horizon_sort"] = rows["horizon"].map(_horizon_sort_key)
    rows["scope_sort"] = rows["scope"].map({"baseline_all": 0, "rule_union": 1}).fillna(9)
    rows = rows.sort_values(["horizon_sort", "scope_sort"])
    rendered = []
    for row in rows.itertuples(index=False):
        rendered.append(
            "| {scope} | {horizon} | {sample_count} | {avg_return} | {win_rate} | {avg_edge} | {win_edge} |".format(
                scope=getattr(row, "scope"),
                horizon=getattr(row, "horizon"),
                sample_count=getattr(row, "sample_count"),
                avg_return=_pct(getattr(row, "avg_return")),
                win_rate=_pct(getattr(row, "win_rate")),
                avg_edge=_pct(getattr(row, "avg_return_edge")),
                win_edge=_pct(getattr(row, "win_rate_edge")),
            )
        )
    return "\n".join(rendered)


def _render_report(snapshot: dict[str, Any], summary: pd.DataFrame) -> str:
    warnings = snapshot.get("warnings") or []
    warning_text = "\n".join(f"- `{item}`" for item in warnings) if warnings else "- None"
    return f"""# Satellite Trial Replay

Generated at: `{snapshot.get("generated_at")}`

This is a research-only replay of satellite trial rules. It does not connect to brokers, place orders, change live positions, or provide investment advice.

## Summary

| Item | Value |
| --- | ---: |
| Status | `{snapshot.get("status")}` |
| Rules | {snapshot.get("rule_count")} |
| History rows | {snapshot.get("history_row_count")} |
| Matched events | {snapshot.get("matched_event_count")} |
| Horizons | {", ".join(snapshot.get("horizons") or [])} |
| Best union horizon | `{snapshot.get("union_best_horizon") or "N/A"}` |
| Best union avg-return edge | {_pct(snapshot.get("union_best_avg_return_edge"))} |
| Best union win-rate edge | {_pct(snapshot.get("union_best_win_rate_edge"))} |
| Broker action | `{snapshot.get("broker_action")}` |
| Research only | `{snapshot.get("research_only")}` |

## Union vs Baseline

| Scope | Horizon | Samples | Avg return | Win rate | Avg-return edge | Win-rate edge |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{_render_union_rows(summary)}

## Rule Details

| Rule | Filter | Horizon | Samples | Avg return | Win rate |
| --- | --- | --- | ---: | ---: | ---: |
{_render_rule_rows(summary)}

## Warnings

{warning_text}

## Files

- Rules CSV: `{snapshot.get("rules_path")}`
- Outcomes history CSV: `{snapshot.get("outcomes_history_path")}`
- Summary CSV: `{snapshot.get("summary_path")}`
- Matches CSV: `{snapshot.get("matches_path")}`
- Snapshot JSON: `{snapshot.get("snapshot_path")}`
"""


def run_satellite_trial_replay(
    rules_path: str | Path = DEFAULT_RULES_PATH,
    outcomes_history_path: str | Path = DEFAULT_OUTCOMES_HISTORY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    horizons: Iterable[str] | str = DEFAULT_HORIZONS,
) -> SatelliteTrialReplayResult:
    resolved_rules_path = Path(rules_path)
    resolved_history_path = Path(outcomes_history_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizon_list = _clean_horizons(horizons)
    raw_rules = _read_csv(resolved_rules_path, "satellite trial rules")
    history = _read_csv(resolved_history_path, "stock-target outcome history")
    rules, warnings = _valid_rules(raw_rules)

    for horizon in horizon_list:
        column = _return_column(horizon)
        if column not in history.columns:
            warnings.append(f"missing_return_column={column}")

    rule_matches, matches, match_warnings = _match_rules(rules, history)
    warnings.extend(match_warnings)
    if not matches.empty:
        matched_indices = pd.to_numeric(matches["_source_index"], errors="coerce").dropna().astype(int).tolist()
        union_frame = history.loc[sorted(set(matched_indices))].copy()
    else:
        union_frame = history.iloc[0:0].copy()

    baseline_by_horizon = {horizon: _stats(history, horizon) for horizon in horizon_list}
    summary_rows: list[dict[str, Any]] = []
    for horizon in horizon_list:
        baseline = baseline_by_horizon[horizon]
        summary_rows.append(
            _summary_row(
                scope="baseline_all",
                rule_id="baseline_all",
                dimension="overall",
                group_value="all",
                horizon=horizon,
                stats=baseline,
                baseline=baseline,
            )
        )

    rule_lookup = {str(row["rule_id"]): row for row in rules.to_dict(orient="records")}
    for rule_id, indices in rule_matches.items():
        rule = rule_lookup.get(rule_id, {})
        rule_frame = history.loc[sorted(set(indices))].copy() if indices else history.iloc[0:0].copy()
        for horizon in horizon_list:
            summary_rows.append(
                _summary_row(
                    scope="rule",
                    rule_id=rule_id,
                    dimension=_normalise_text(rule.get("dimension")),
                    group_value=_normalise_text(rule.get("group_value")),
                    horizon=horizon,
                    stats=_stats(rule_frame, horizon),
                    baseline=baseline_by_horizon[horizon],
                )
            )

    for horizon in horizon_list:
        summary_rows.append(
            _summary_row(
                scope="rule_union",
                rule_id="rule_union",
                dimension="union",
                group_value="any_trial_rule",
                horizon=horizon,
                stats=_stats(union_frame, horizon),
                baseline=baseline_by_horizon[horizon],
            )
        )

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    union_rows = summary[summary["scope"] == "rule_union"]
    best_row: dict[str, Any] = {}
    if not union_rows.empty:
        ranked = union_rows.copy()
        ranked["_avg_edge"] = pd.to_numeric(ranked["avg_return_edge"], errors="coerce")
        ranked = ranked.sort_values(["_avg_edge", "sample_count"], ascending=[False, False])
        best_row = ranked.iloc[0].to_dict()

    summary_path = out_dir / "satellite_trial_replay_summary.csv"
    matches_path = out_dir / "satellite_trial_replay_matches.csv"
    snapshot_path = out_dir / "satellite_trial_replay_snapshot.json"
    report_path = out_dir / "satellite_trial_replay.md"

    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "rules_path": str(resolved_rules_path),
        "outcomes_history_path": str(resolved_history_path),
        "output_dir": str(out_dir),
        "summary_path": str(summary_path),
        "matches_path": str(matches_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "rule_count": int(len(rules)),
        "history_row_count": int(len(history)),
        "matched_event_count": int(len(union_frame)),
        "horizons": horizon_list,
        "union_best_horizon": best_row.get("horizon"),
        "union_best_avg_return_edge": _as_float(best_row.get("avg_return_edge")),
        "union_best_win_rate_edge": _as_float(best_row.get("win_rate_edge")),
        "warnings": warnings,
        "broker_action": "none",
        "research_only": True,
    }

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    matches.to_csv(matches_path, index=False, encoding="utf-8-sig")
    snapshot_path.write_text(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, summary), encoding="utf-8")

    return SatelliteTrialReplayResult(
        output_dir=out_dir,
        summary_path=summary_path,
        matches_path=matches_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        summary=summary,
        matches=matches,
    )
