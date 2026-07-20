"""Research-only satellite filter diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


DEFAULT_OUTCOMES_HISTORY_PATH = Path("outputs/research/stock_target_review_outcomes_history.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/satellite_filter_research_latest")
DEFAULT_FILTER_CANDIDATES_PATH = Path("outputs/research/satellite_filter_research_latest/satellite_filter_candidates.csv")
DEFAULT_FILTER_MODEL_OUTPUT_DIR = Path("outputs/research/satellite_filter_model_latest")
DEFAULT_HORIZONS = ("1d", "5d", "10d", "20d")
DEFAULT_DIMENSIONS = ("review_bucket", "action_code", "manual_status_normalized")
DEFAULT_CONFIRMATION_HORIZONS = ("5d", "10d")
FILTER_MODEL_SCORE_COLUMNS = [
    "date",
    "layer",
    "code",
    "name",
    "satellite_filter_decision",
    "satellite_filter_status",
    "satellite_filter_score",
    "matched_dimension",
    "matched_group_value",
    "matched_candidate_status",
    "matched_passed_horizons",
    "matched_best_horizon",
    "matched_best_sample_count",
    "matched_best_win_rate",
    "matched_best_avg_return",
    "matched_best_min_return",
    "industry_chain_filter_status",
    "industry_chain_factor_score",
    "serenity_bottleneck_score",
    "industry_chain_factor_bucket",
    "serenity_bottleneck_bucket",
    "broker_action",
    "research_only",
]


@dataclass(frozen=True)
class SatelliteFilterResearchResult:
    output_dir: Path
    report_path: Path
    group_scores_path: Path
    candidates_path: Path
    snapshot_path: Path
    snapshot: dict[str, Any]


@dataclass(frozen=True)
class SatelliteFilterModelResult:
    output_dir: Path
    scores_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    scores: pd.DataFrame


def _as_tuple(value: str | Sequence[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    cleaned = tuple(str(item).strip() for item in parts if str(item).strip())
    return cleaned or tuple(default)


def _normalise_horizon(value: str) -> str:
    text = str(value).strip().lower()
    return text if text.endswith("d") else f"{text}d"


def _return_column(horizon: str) -> str:
    return f"return_{_normalise_horizon(horizon)}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if value is pd.NA:
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _safe_group_value(value: Any) -> str:
    if pd.isna(value):
        return "(missing)"
    text = str(value).strip()
    return text if text else "(blank)"


def _clean_code(value: Any) -> str:
    if value in (None, ""):
        return ""
    digits = "".join(ch for ch in str(value).strip() if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    return digits.zfill(6) if digits else ""


def _score_returns(
    values: pd.Series,
    *,
    min_sample_count: int,
    min_win_rate: float,
    min_avg_return: float,
    max_worst_return: float,
) -> dict[str, Any]:
    returns = pd.to_numeric(values, errors="coerce").dropna()
    sample_count = int(len(returns))
    if sample_count == 0:
        return {
            "sample_count": 0,
            "win_rate": None,
            "avg_return": None,
            "median_return": None,
            "min_return": None,
            "max_return": None,
            "pass_sample_count": False,
            "pass_win_rate": False,
            "pass_avg_return": False,
            "pass_worst_return": False,
            "pass_all": False,
        }
    win_rate = float((returns > 0).mean())
    avg_return = float(returns.mean())
    median_return = float(returns.median())
    min_return = float(returns.min())
    max_return = float(returns.max())
    pass_sample_count = sample_count >= min_sample_count
    pass_win_rate = win_rate >= min_win_rate
    pass_avg_return = avg_return >= min_avg_return
    pass_worst_return = min_return >= max_worst_return
    return {
        "sample_count": sample_count,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "median_return": median_return,
        "min_return": min_return,
        "max_return": max_return,
        "pass_sample_count": pass_sample_count,
        "pass_win_rate": pass_win_rate,
        "pass_avg_return": pass_avg_return,
        "pass_worst_return": pass_worst_return,
        "pass_all": bool(pass_sample_count and pass_win_rate and pass_avg_return and pass_worst_return),
    }


def _baseline_rows(
    frame: pd.DataFrame,
    horizons: Iterable[str],
    *,
    min_sample_count: int,
    min_win_rate: float,
    min_avg_return: float,
    max_worst_return: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scopes = [("all", frame)]
    if "layer" in frame.columns:
        layer_values = frame["layer"].fillna("").astype(str).str.lower()
        scopes.extend(
            [
                ("core", frame[layer_values == "core"]),
                ("satellite", frame[layer_values == "satellite"]),
            ]
        )
    for scope, scope_frame in scopes:
        for horizon in horizons:
            column = _return_column(horizon)
            values = scope_frame[column] if column in scope_frame.columns else pd.Series(dtype=float)
            metrics = _score_returns(
                values,
                min_sample_count=min_sample_count,
                min_win_rate=min_win_rate,
                min_avg_return=min_avg_return,
                max_worst_return=max_worst_return,
            )
            rows.append({"scope": scope, "horizon": _normalise_horizon(horizon), **metrics})
    return rows


def _group_score_rows(
    satellite_frame: pd.DataFrame,
    dimensions: Iterable[str],
    horizons: Iterable[str],
    *,
    min_sample_count: int,
    min_win_rate: float,
    min_avg_return: float,
    max_worst_return: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing_dimensions: list[str] = []
    for dimension in dimensions:
        if dimension not in satellite_frame.columns:
            missing_dimensions.append(dimension)
            continue
        work = satellite_frame.copy()
        work[dimension] = work[dimension].map(_safe_group_value)
        for group_value, group_frame in work.groupby(dimension, dropna=False, sort=True):
            for horizon in horizons:
                column = _return_column(horizon)
                values = group_frame[column] if column in group_frame.columns else pd.Series(dtype=float)
                metrics = _score_returns(
                    values,
                    min_sample_count=min_sample_count,
                    min_win_rate=min_win_rate,
                    min_avg_return=min_avg_return,
                    max_worst_return=max_worst_return,
                )
                row = {
                    "dimension": dimension,
                    "group_value": str(group_value),
                    "horizon": _normalise_horizon(horizon),
                    **metrics,
                }
                row["score_status"] = "pass" if row["pass_all"] else "fail"
                rows.append(row)
    return rows, missing_dimensions


def _candidate_rows(group_scores: pd.DataFrame, confirmation_horizons: Sequence[str]) -> list[dict[str, Any]]:
    if group_scores.empty:
        return []
    candidates: list[dict[str, Any]] = []
    confirmations = tuple(_normalise_horizon(item) for item in confirmation_horizons)
    for (dimension, group_value), group in group_scores.groupby(["dimension", "group_value"], sort=True):
        by_horizon = {str(row["horizon"]): row for _, row in group.iterrows()}
        passed = [horizon for horizon in confirmations if bool(by_horizon.get(horizon, {}).get("pass_all"))]
        sampled = [
            horizon
            for horizon in confirmations
            if int(by_horizon.get(horizon, {}).get("sample_count", 0) or 0) > 0
        ]
        if len(passed) == len(confirmations):
            status = "confirmed"
        elif passed:
            status = "watch"
        elif not sampled:
            status = "immature"
        else:
            status = "rejected"
        if status not in {"confirmed", "watch"}:
            continue
        best_row = max(
            (by_horizon[horizon] for horizon in passed),
            key=lambda row: (float(row.get("avg_return") or 0.0), float(row.get("win_rate") or 0.0)),
        )
        candidates.append(
            {
                "dimension": str(dimension),
                "group_value": str(group_value),
                "candidate_status": status,
                "passed_horizons": ",".join(passed),
                "sampled_confirmation_horizons": ",".join(sampled),
                "best_horizon": str(best_row["horizon"]),
                "best_sample_count": int(best_row["sample_count"]),
                "best_win_rate": best_row["win_rate"],
                "best_avg_return": best_row["avg_return"],
                "best_min_return": best_row["min_return"],
            }
        )
    return candidates


def _write_report(snapshot: dict[str, Any], candidates: pd.DataFrame) -> str:
    lines = [
        "# Satellite Filter Research",
        "",
        f"- generated_at: {snapshot['generated_at']}",
        f"- outcomes_history_path: {snapshot['outcomes_history_path']}",
        f"- research_status: {snapshot['research_status']}",
        f"- total_rows: {snapshot['total_rows']}",
        f"- satellite_rows: {snapshot['satellite_rows']}",
        f"- confirmed_candidate_count: {snapshot['confirmed_candidate_count']}",
        f"- watch_candidate_count: {snapshot['watch_candidate_count']}",
        "",
        "## Candidate Filters",
        "",
    ]
    if candidates.empty:
        lines.append("No candidate filter passed the current research thresholds.")
    else:
        lines.append("| status | dimension | group | passed horizons | best horizon | win rate | avg return |")
        lines.append("| --- | --- | --- | --- | --- | ---: | ---: |")
        for _, row in candidates.iterrows():
            win_rate = row.get("best_win_rate")
            avg_return = row.get("best_avg_return")
            win_text = "N/A" if pd.isna(win_rate) else f"{float(win_rate) * 100:.2f}%"
            avg_text = "N/A" if pd.isna(avg_return) else f"{float(avg_return) * 100:.2f}%"
            lines.append(
                "| "
                f"{row['candidate_status']} | {row['dimension']} | {row['group_value']} | "
                f"{row['passed_horizons']} | {row['best_horizon']} | {win_text} | {avg_text} |"
            )
    lines.extend(
        [
            "",
            "## Research Guardrails",
            "",
            "- This report is research-only.",
            "- It does not change the allocator, paper account, live shadow, or broker signal path.",
            "- Confirmed groups still require daily-pipeline and allocator review before any default-path change.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return pd.read_csv(path, dtype={"code": str})


def _candidate_status_rank(value: Any) -> int:
    status = str(value or "").strip().lower()
    if status == "confirmed":
        return 2
    if status == "watch":
        return 1
    return 0


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _as_int(value: Any, default: int = 0) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    return number


def _rule_model_score(rule: pd.Series | None) -> float:
    if rule is None:
        return 0.0
    rank = _candidate_status_rank(rule.get("candidate_status"))
    win_rate = max(min(_as_float(rule.get("best_win_rate")), 1.0), 0.0)
    avg_return = max(min(_as_float(rule.get("best_avg_return")) / 0.20, 1.0), -1.0)
    sample_score = min(_as_int(rule.get("best_sample_count")) / 20.0, 1.0)
    status_component = 45.0 if rank == 2 else 30.0 if rank == 1 else 0.0
    return round(max(status_component + 25.0 * win_rate + 20.0 * max(avg_return, 0.0) + 10.0 * sample_score, 0.0), 3)


def _best_matching_rule(candidate: pd.Series, rules: pd.DataFrame) -> pd.Series | None:
    if rules.empty:
        return None
    matches: list[pd.Series] = []
    for _, rule in rules.iterrows():
        dimension = str(rule.get("dimension") or "").strip()
        if not dimension:
            continue
        candidate_value = _safe_group_value(candidate.get(dimension))
        if candidate_value != str(rule.get("group_value") or "").strip():
            continue
        if _candidate_status_rank(rule.get("candidate_status")) <= 0:
            continue
        matches.append(rule)
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda row: (
            _candidate_status_rank(row.get("candidate_status")),
            _as_float(row.get("best_avg_return")),
            _as_float(row.get("best_win_rate")),
            _as_int(row.get("best_sample_count")),
        ),
        reverse=True,
    )[0]


def _decision_from_rule(rule: pd.Series | None) -> tuple[str, str]:
    if rule is None:
        return "hold_satellite", "no_filter_match"
    status = str(rule.get("candidate_status") or "").strip().lower()
    if status == "confirmed":
        return "allow_research_trial", "confirmed_filter_match"
    if status == "watch":
        return "watch_only", "watch_filter_match"
    return "hold_satellite", "no_filter_match"


def _load_industry_chain_scores(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        raise FileNotFoundError(f"Missing industry-chain scores: {path}")
    frame = pd.read_csv(path, dtype={"code": str, "security_code": str})
    code_column = "code" if "code" in frame.columns else "security_code" if "security_code" in frame.columns else None
    if code_column is None:
        raise ValueError("industry-chain scores must contain code or security_code.")
    data = frame.copy()
    data["code"] = data[code_column].map(_clean_code)
    for column in [
        "industry_chain_factor_score",
        "serenity_bottleneck_score",
        "factor_bucket",
        "serenity_bottleneck_bucket",
    ]:
        if column not in data.columns:
            data[column] = 0.0 if column.endswith("_score") else ""
    for column in ["industry_chain_factor_score", "serenity_bottleneck_score"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    return data.loc[data["code"] != ""].drop_duplicates(subset=["code"], keep="last")


def _industry_chain_rule(candidate: pd.Series, scores: pd.DataFrame) -> pd.Series | None:
    if scores.empty:
        return None
    code = _clean_code(candidate.get("code"))
    if not code:
        return None
    matches = scores.loc[scores["code"] == code]
    if matches.empty:
        return None
    return matches.iloc[-1]


def _infer_candidate_layer(row: pd.Series) -> str:
    layer = str(row.get("layer") or "").strip().lower()
    if layer:
        return layer
    source_sleeve = str(row.get("source_sleeve") or "").strip().lower()
    source_profile = str(row.get("source_strategy_profile") or "").strip().lower()
    if "satellite" in source_sleeve or "chip_reversal" in source_sleeve:
        return "satellite"
    if "satellite" in source_profile or "chip_reversal" in source_profile:
        return "satellite"
    return layer


def _industry_chain_gate_status(
    rule: pd.Series | None,
    *,
    enabled: bool,
    min_industry_chain_score: float,
    min_bottleneck_score: float,
) -> tuple[str, bool, dict[str, Any]]:
    if not enabled:
        return "not_configured", False, {}
    if rule is None:
        return "industry_chain_missing", False, {}
    industry_score = _as_float(rule.get("industry_chain_factor_score"))
    bottleneck_score = _as_float(rule.get("serenity_bottleneck_score"))
    passed = industry_score >= min_industry_chain_score and bottleneck_score >= min_bottleneck_score
    status = "industry_chain_pass" if passed else "industry_chain_gate_pending"
    details = {
        "industry_chain_factor_score": industry_score,
        "serenity_bottleneck_score": bottleneck_score,
        "industry_chain_factor_bucket": str(rule.get("factor_bucket") or ""),
        "serenity_bottleneck_bucket": str(rule.get("serenity_bottleneck_bucket") or ""),
    }
    return status, passed, details


def _combined_satellite_filter_score(
    *,
    rule: pd.Series | None,
    status: str,
    industry_chain_gate_enabled: bool,
    chain_details: dict[str, Any],
) -> float:
    base_score = _rule_model_score(rule)
    if industry_chain_gate_enabled:
        if status == "industry_chain_gate_pending":
            base_score = min(base_score, 35.0)
        elif status == "industry_chain_only_watch":
            base_score = max(base_score, 45.0)
        industry_score = max(min(_as_float(chain_details.get("industry_chain_factor_score")), 100.0), 0.0)
        bottleneck_score = max(min(_as_float(chain_details.get("serenity_bottleneck_score")), 100.0), 0.0)
        base_score += 0.10 * industry_score + 0.15 * bottleneck_score
    return round(max(min(base_score, 100.0), 0.0), 3)


def _satellite_model_score_row(
    candidate: pd.Series,
    rule: pd.Series | None,
    *,
    industry_chain_rule: pd.Series | None = None,
    industry_chain_gate_enabled: bool = False,
    min_industry_chain_score: float = 0.0,
    min_bottleneck_score: float = 0.0,
) -> dict[str, Any]:
    decision, status = _decision_from_rule(rule)
    chain_status, chain_passed, chain_details = _industry_chain_gate_status(
        industry_chain_rule,
        enabled=industry_chain_gate_enabled,
        min_industry_chain_score=min_industry_chain_score,
        min_bottleneck_score=min_bottleneck_score,
    )
    if industry_chain_gate_enabled:
        if decision == "allow_research_trial" and not chain_passed:
            decision = "watch_only"
            status = "industry_chain_gate_pending"
        elif decision == "hold_satellite" and chain_passed:
            decision = "watch_only"
            status = "industry_chain_only_watch"
    record = {
        "date": candidate.get("date", ""),
        "layer": candidate.get("layer", ""),
        "code": candidate.get("code", ""),
        "name": candidate.get("name", ""),
        "satellite_filter_decision": decision,
        "satellite_filter_status": status,
        "satellite_filter_score": _combined_satellite_filter_score(
            rule=rule,
            status=status,
            industry_chain_gate_enabled=industry_chain_gate_enabled,
            chain_details=chain_details,
        ),
        "matched_dimension": "",
        "matched_group_value": "",
        "matched_candidate_status": "",
        "matched_passed_horizons": "",
        "matched_best_horizon": "",
        "matched_best_sample_count": 0,
        "matched_best_win_rate": None,
        "matched_best_avg_return": None,
        "matched_best_min_return": None,
        "industry_chain_filter_status": chain_status,
        "industry_chain_factor_score": chain_details.get("industry_chain_factor_score"),
        "serenity_bottleneck_score": chain_details.get("serenity_bottleneck_score"),
        "industry_chain_factor_bucket": chain_details.get("industry_chain_factor_bucket", ""),
        "serenity_bottleneck_bucket": chain_details.get("serenity_bottleneck_bucket", ""),
        "broker_action": "none",
        "research_only": True,
    }
    if rule is not None:
        record.update(
            {
                "matched_dimension": rule.get("dimension", ""),
                "matched_group_value": rule.get("group_value", ""),
                "matched_candidate_status": rule.get("candidate_status", ""),
                "matched_passed_horizons": rule.get("passed_horizons", ""),
                "matched_best_horizon": rule.get("best_horizon", ""),
                "matched_best_sample_count": _as_int(rule.get("best_sample_count")),
                "matched_best_win_rate": _as_float(rule.get("best_win_rate")),
                "matched_best_avg_return": _as_float(rule.get("best_avg_return")),
                "matched_best_min_return": _as_float(rule.get("best_min_return")),
            }
        )
    return record


def _empty_model_scores() -> pd.DataFrame:
    return pd.DataFrame(columns=FILTER_MODEL_SCORE_COLUMNS)


def _render_model_report(snapshot: dict[str, Any], scores: pd.DataFrame) -> str:
    lines = [
        "# Satellite Filter Model",
        "",
        f"- generated_at: {snapshot['generated_at']}",
        f"- status: {snapshot['status']}",
        f"- candidates_path: {snapshot['candidates_path']}",
        f"- filter_candidates_path: {snapshot['filter_candidates_path']}",
        f"- satellite_candidate_count: {snapshot['satellite_candidate_count']}",
        f"- allow_count: {snapshot['allow_count']}",
        f"- watch_count: {snapshot['watch_count']}",
        f"- hold_count: {snapshot['hold_count']}",
        "",
        "## Decisions",
        "",
    ]
    if scores.empty:
        lines.append("No satellite candidates were available for filtering.")
    else:
        lines.append("| code | name | decision | status | score | matched rule |")
        lines.append("| --- | --- | --- | --- | ---: | --- |")
        for _, row in scores.head(30).iterrows():
            matched = f"{row.get('matched_dimension')}={row.get('matched_group_value')}"
            lines.append(
                "| "
                f"{row.get('code')} | {row.get('name')} | {row.get('satellite_filter_decision')} | "
                f"{row.get('satellite_filter_status')} | {float(row.get('satellite_filter_score') or 0.0):.3f} | {matched} |"
            )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Research-only model output.",
            "- Broker action remains none.",
            "- This model does not update allocator defaults or paper-account target weights.",
            "",
        ]
    )
    return "\n".join(lines)


def run_satellite_filter_model(
    *,
    candidates_path: str | Path,
    filter_candidates_path: str | Path = DEFAULT_FILTER_CANDIDATES_PATH,
    industry_chain_scores_path: str | Path | None = None,
    min_industry_chain_score: float = 60.0,
    min_bottleneck_score: float = 55.0,
    output_dir: str | Path = DEFAULT_FILTER_MODEL_OUTPUT_DIR,
) -> SatelliteFilterModelResult:
    candidates_file = Path(candidates_path)
    filters_file = Path(filter_candidates_path)
    industry_chain_scores_file = Path(industry_chain_scores_path) if industry_chain_scores_path not in (None, "") else None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    candidates = _read_csv_required(candidates_file, "satellite filter model candidates")
    filters = _read_csv_required(filters_file, "satellite filter candidates")
    industry_chain_scores = _load_industry_chain_scores(industry_chain_scores_file)
    if "layer" not in candidates.columns:
        candidates["layer"] = ""
    candidates["layer"] = candidates.apply(_infer_candidate_layer, axis=1)
    if "code" not in candidates.columns and "security_code" in candidates.columns:
        candidates["code"] = candidates["security_code"].astype(str)
    if "name" not in candidates.columns and "security_name" in candidates.columns:
        candidates["name"] = candidates["security_name"].astype(str)
    for column in ["code", "name"]:
        if column not in candidates.columns:
            candidates[column] = ""

    required_filter_columns = {"dimension", "group_value", "candidate_status"}
    missing = required_filter_columns - set(filters.columns)
    if missing:
        raise ValueError(f"filter candidates missing columns: {sorted(missing)}")
    layer = candidates["layer"].fillna("").astype(str).str.lower()
    satellite = candidates[layer == "satellite"].copy()

    industry_chain_gate_enabled = industry_chain_scores_file is not None
    rows = [
        _satellite_model_score_row(
            candidate,
            _best_matching_rule(candidate, filters),
            industry_chain_rule=_industry_chain_rule(candidate, industry_chain_scores),
            industry_chain_gate_enabled=industry_chain_gate_enabled,
            min_industry_chain_score=min_industry_chain_score,
            min_bottleneck_score=min_bottleneck_score,
        )
        for _, candidate in satellite.iterrows()
    ]
    scores = pd.DataFrame(rows, columns=FILTER_MODEL_SCORE_COLUMNS) if rows else _empty_model_scores()
    if not scores.empty:
        scores = scores.sort_values(
            ["satellite_filter_decision", "satellite_filter_score", "code"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    allow_count = int((scores["satellite_filter_decision"] == "allow_research_trial").sum()) if not scores.empty else 0
    watch_count = int((scores["satellite_filter_decision"] == "watch_only").sum()) if not scores.empty else 0
    hold_count = int((scores["satellite_filter_decision"] == "hold_satellite").sum()) if not scores.empty else 0
    industry_chain_pass_count = int((scores["industry_chain_filter_status"] == "industry_chain_pass").sum()) if not scores.empty else 0
    industry_chain_pending_count = (
        int((scores["industry_chain_filter_status"] == "industry_chain_gate_pending").sum()) if not scores.empty else 0
    )
    industry_chain_missing_count = (
        int((scores["industry_chain_filter_status"] == "industry_chain_missing").sum()) if not scores.empty else 0
    )
    if len(satellite) == 0:
        status = "no_satellite_candidates"
    elif allow_count > 0:
        status = "allow_candidates"
    elif watch_count > 0:
        status = "watch_candidates"
    else:
        status = "all_candidates_held"

    scores_path = output_path / "satellite_filter_model_scores.csv"
    snapshot_path = output_path / "satellite_filter_model_snapshot.json"
    report_path = output_path / "satellite_filter_model.md"
    scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "research_only": True,
        "broker_action": "none",
        "candidates_path": str(candidates_file),
        "filter_candidates_path": str(filters_file),
        "industry_chain_gate_enabled": industry_chain_gate_enabled,
        "industry_chain_scores_path": None if industry_chain_scores_file is None else str(industry_chain_scores_file),
        "min_industry_chain_score": float(min_industry_chain_score),
        "min_bottleneck_score": float(min_bottleneck_score),
        "output_dir": str(output_path),
        "candidate_count": int(len(candidates)),
        "satellite_candidate_count": int(len(satellite)),
        "filter_rule_count": int(len(filters)),
        "industry_chain_score_count": int(len(industry_chain_scores)),
        "allow_count": allow_count,
        "watch_count": watch_count,
        "hold_count": hold_count,
        "industry_chain_pass_count": industry_chain_pass_count,
        "industry_chain_pending_count": industry_chain_pending_count,
        "industry_chain_missing_count": industry_chain_missing_count,
        "scores_path": str(scores_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "top_scores": scores.head(20).to_dict(orient="records") if not scores.empty else [],
    }
    snapshot_path.write_text(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_model_report(snapshot, scores), encoding="utf-8")
    return SatelliteFilterModelResult(
        output_dir=output_path,
        scores_path=scores_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        scores=scores,
    )


def run_satellite_filter_research(
    *,
    outcomes_history_path: str | Path = DEFAULT_OUTCOMES_HISTORY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    horizons: str | Sequence[str] | None = None,
    dimensions: str | Sequence[str] | None = None,
    confirmation_horizons: str | Sequence[str] | None = None,
    min_sample_count: int = 5,
    min_win_rate: float = 0.55,
    min_avg_return: float = 0.0,
    max_worst_return: float = -0.20,
) -> SatelliteFilterResearchResult:
    history_path = Path(outcomes_history_path)
    if not history_path.exists():
        raise FileNotFoundError(f"Missing outcomes history: {history_path}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_horizons = tuple(_normalise_horizon(item) for item in _as_tuple(horizons, DEFAULT_HORIZONS))
    selected_dimensions = _as_tuple(dimensions, DEFAULT_DIMENSIONS)
    selected_confirmation_horizons = tuple(
        _normalise_horizon(item) for item in _as_tuple(confirmation_horizons, DEFAULT_CONFIRMATION_HORIZONS)
    )

    frame = pd.read_csv(history_path, dtype={"code": str})
    if "layer" in frame.columns:
        layer = frame["layer"].fillna("").astype(str).str.lower()
        satellite_frame = frame[layer == "satellite"].copy()
    else:
        satellite_frame = frame.iloc[0:0].copy()

    baseline = _baseline_rows(
        frame,
        selected_horizons,
        min_sample_count=min_sample_count,
        min_win_rate=min_win_rate,
        min_avg_return=min_avg_return,
        max_worst_return=max_worst_return,
    )
    group_rows, missing_dimensions = _group_score_rows(
        satellite_frame,
        selected_dimensions,
        selected_horizons,
        min_sample_count=min_sample_count,
        min_win_rate=min_win_rate,
        min_avg_return=min_avg_return,
        max_worst_return=max_worst_return,
    )
    group_scores = pd.DataFrame(group_rows)
    candidate_rows = _candidate_rows(group_scores, selected_confirmation_horizons)
    candidates = pd.DataFrame(candidate_rows)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["candidate_status", "best_avg_return", "best_win_rate"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

    confirmed_count = int((candidates["candidate_status"] == "confirmed").sum()) if not candidates.empty else 0
    watch_count = int((candidates["candidate_status"] == "watch").sum()) if not candidates.empty else 0
    mature_return_count = 0
    for horizon in selected_horizons:
        column = _return_column(horizon)
        if column in satellite_frame.columns:
            mature_return_count += int(pd.to_numeric(satellite_frame[column], errors="coerce").notna().sum())
    if confirmed_count or watch_count:
        research_status = "candidate_filters_found"
    elif len(satellite_frame) == 0:
        research_status = "no_satellite_rows"
    elif mature_return_count == 0:
        research_status = "waiting_for_mature_samples"
    else:
        research_status = "no_candidate_filters"

    group_scores_path = output_path / "satellite_filter_group_scores.csv"
    candidates_path = output_path / "satellite_filter_candidates.csv"
    baseline_path = output_path / "satellite_filter_baseline.csv"
    snapshot_path = output_path / "satellite_filter_research_snapshot.json"
    report_path = output_path / "satellite_filter_research.md"

    group_scores.to_csv(group_scores_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(baseline).to_csv(baseline_path, index=False, encoding="utf-8-sig")

    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "outcomes_history_path": str(history_path),
        "output_dir": str(output_path),
        "research_only": True,
        "research_status": research_status,
        "total_rows": int(len(frame)),
        "satellite_rows": int(len(satellite_frame)),
        "horizons": list(selected_horizons),
        "dimensions": list(selected_dimensions),
        "confirmation_horizons": list(selected_confirmation_horizons),
        "thresholds": {
            "min_sample_count": int(min_sample_count),
            "min_win_rate": float(min_win_rate),
            "min_avg_return": float(min_avg_return),
            "max_worst_return": float(max_worst_return),
        },
        "missing_dimensions": missing_dimensions,
        "confirmed_candidate_count": confirmed_count,
        "watch_candidate_count": watch_count,
        "candidate_count": int(len(candidates)),
        "mature_return_count": mature_return_count,
        "group_scores_path": str(group_scores_path),
        "candidates_path": str(candidates_path),
        "baseline_path": str(baseline_path),
        "report_path": str(report_path),
        "baseline": baseline,
        "top_candidates": candidates.head(20).to_dict(orient="records") if not candidates.empty else [],
    }
    snapshot_path.write_text(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_write_report(snapshot, candidates), encoding="utf-8")

    return SatelliteFilterResearchResult(
        output_dir=output_path,
        report_path=report_path,
        group_scores_path=group_scores_path,
        candidates_path=candidates_path,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )
