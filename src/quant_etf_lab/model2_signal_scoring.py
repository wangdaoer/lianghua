"""Research-only Model2 candidate scoring from daily feature-pack inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MODEL2_FEATURE_PACK_JSON = Path("outputs/research/model2_data_input_latest/model2_feature_pack.json")
DEFAULT_MODEL2_INPUT_JSON = Path("D:/codex/daily-market-data/exports/latest_model2_input_with_sector.json")
DEFAULT_MODEL2_SIGNAL_OUTPUT_DIR = Path("outputs/research/model2_signal_scores_latest")

FEATURE_PACK_SCHEMA = "quant-model2-feature-pack-v1"
MODEL2_INPUT_SCHEMA = "a-share-model2-input-v1"
SIGNAL_SCORES_SCHEMA = "quant-model2-signal-scores-v1"
MODEL2_CANDIDATE_SCOPE = "main_chinext"


@dataclass(frozen=True)
class Model2SignalScoringResult:
    output_dir: Path
    scores_path: Path
    report_path: Path
    payload: dict[str, Any]
    feature_pack: dict[str, Any]
    source_payload: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Model2 signal scoring JSON not found: {path}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"Model2 signal scoring JSON must be an object: {path}")
    return payload


def _as_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in {float("inf"), float("-inf")}:
        return default
    return number


def _rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _feature_flags(feature_pack: dict[str, Any]) -> list[str]:
    flags = feature_pack.get("feature_flags")
    if not isinstance(flags, list):
        return []
    return [str(flag) for flag in flags]


def _blocking_reasons(feature_pack: dict[str, Any], source_payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if feature_pack.get("schema_version") != FEATURE_PACK_SCHEMA:
        reasons.append("feature_pack_schema_unexpected")
    if source_payload.get("schema_version") != MODEL2_INPUT_SCHEMA:
        reasons.append("model2_input_schema_unexpected")
    if feature_pack.get("research_only") is not True or source_payload.get("broker_action") not in {None, "none"}:
        reasons.append("research_only_contract_missing")
    if feature_pack.get("broker_action") != "none":
        reasons.append("broker_action_not_none")
    feature_status = str(feature_pack.get("status") or "").strip().lower()
    if feature_status == "blocked":
        reasons.append("feature_pack_not_ok")
    elif feature_status not in {"ok", "watch"}:
        reasons.append("feature_pack_status_unexpected")
    if feature_pack.get("trade_date") != source_payload.get("trade_date"):
        reasons.append("trade_date_mismatch")
    return reasons


def _warning_reasons(feature_pack: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    feature_status = str(feature_pack.get("status") or "").strip().lower()
    if feature_status == "watch":
        warnings.append("feature_pack_watch")
    for item in feature_pack.get("warnings") or []:
        text = str(item).strip()
        if text and text not in warnings:
            warnings.append(text)
    return warnings


def _candidate_key(row: dict[str, Any]) -> str:
    return str(row.get("security_code") or row.get("symbol") or "").strip()


def _code_digits(value: Any) -> str:
    return "".join(ch for ch in str(value or "").strip() if ch.isdigit())


def _is_main_chinext_code(value: Any) -> bool:
    digits = _code_digits(value)
    return digits.startswith(("60", "00", "30"))


def _is_special_treatment_name(value: Any) -> bool:
    text = str(value or "").strip().upper().replace(" ", "")
    return text.startswith(("*ST", "ST", "S*ST", "SST"))


def _merge_candidates(source_payload: dict[str, Any]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for source_name in ("top_gainers", "turnover_leaders"):
        for row in _rows(source_payload.get(source_name)):
            code = _candidate_key(row)
            if not code:
                continue
            candidate = merged.setdefault(
                code,
                {
                    "security_code": code,
                    "security_name": row.get("security_name"),
                    "change_ratio": _as_number(row.get("change_ratio")),
                    "turnover": _as_number(row.get("turnover")),
                    "volume": _as_number(row.get("volume")),
                    "source_lists": [],
                },
            )
            if not candidate.get("security_name") and row.get("security_name"):
                candidate["security_name"] = row.get("security_name")
            candidate["change_ratio"] = max(
                _as_number(candidate.get("change_ratio")),
                _as_number(row.get("change_ratio")),
            )
            candidate["turnover"] = max(_as_number(candidate.get("turnover")), _as_number(row.get("turnover")))
            candidate["volume"] = max(_as_number(candidate.get("volume")), _as_number(row.get("volume")))
            if source_name not in candidate["source_lists"]:
                candidate["source_lists"].append(source_name)
    return list(merged.values())


def _filter_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    unsupported_board = 0
    special_treatment = 0
    for candidate in candidates:
        if not _is_main_chinext_code(candidate.get("security_code")):
            unsupported_board += 1
            continue
        if _is_special_treatment_name(candidate.get("security_name")):
            special_treatment += 1
            continue
        kept.append(candidate)
    return kept, {
        "excluded_candidate_count": unsupported_board + special_treatment,
        "excluded_unsupported_board_count": unsupported_board,
        "excluded_special_treatment_count": special_treatment,
    }


def _market_regime_adjustment(market_regime: str) -> float:
    if market_regime == "broad_strength":
        return 10.0
    if market_regime == "weak_breadth":
        return -10.0
    if market_regime == "unknown_breadth":
        return -5.0
    return 0.0


def _score_bucket(score: float) -> str:
    if score >= 80:
        return "high_research_priority"
    if score >= 60:
        return "medium_research_priority"
    return "low_research_priority"


def _score_candidate(candidate: dict[str, Any], feature_pack: dict[str, Any]) -> dict[str, Any]:
    flags = _feature_flags(feature_pack)
    market_regime = str(feature_pack.get("market_regime") or "unknown_breadth")
    source_lists = list(candidate.get("source_lists") or [])
    source_bonus = 0.0
    if "top_gainers" in source_lists:
        source_bonus += 25.0
    if "turnover_leaders" in source_lists:
        source_bonus += 15.0
    momentum_score = max(min(_as_number(candidate.get("change_ratio")), 20.0), -10.0)
    regime_adjustment = _market_regime_adjustment(market_regime)
    context_bonus = 0.0
    if feature_pack.get("sector_theme_status") == "ok" or "sector_theme_ok" in flags:
        context_bonus += 5.0
    if "fund_flow_available" in flags:
        context_bonus += 5.0
    score = max(min(40.0 + source_bonus + momentum_score + regime_adjustment + context_bonus, 100.0), 0.0)
    reasons = [f"source:{source}" for source in source_lists]
    reasons.extend(
        [
            f"market_regime:{market_regime}",
            "sector_theme_ok" if context_bonus >= 5.0 else "sector_theme_not_ok",
            "fund_flow_available" if "fund_flow_available" in flags else "fund_flow_missing",
            "broker_instruction:none",
        ]
    )
    return {
        "security_code": candidate.get("security_code"),
        "security_name": candidate.get("security_name"),
        "model2_score": round(score, 2),
        "score_bucket": _score_bucket(score),
        "research_signal": "research_watch_only",
        "source_lists": source_lists,
        "change_ratio": _as_number(candidate.get("change_ratio")),
        "turnover": _as_number(candidate.get("turnover")),
        "volume": _as_number(candidate.get("volume")),
        "score_components": {
            "base": 40.0,
            "source_bonus": source_bonus,
            "momentum_score": momentum_score,
            "market_regime_adjustment": regime_adjustment,
            "context_bonus": context_bonus,
        },
        "score_reasons": reasons,
    }


def _rank_candidates(
    candidates: list[dict[str, Any]],
    feature_pack: dict[str, Any],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    scored = [_score_candidate(candidate, feature_pack) for candidate in candidates]
    scored.sort(
        key=lambda row: (
            -_as_number(row.get("model2_score")),
            str(row.get("security_code") or ""),
        )
    )
    limited = scored[: max(int(top_n), 0)]
    for index, row in enumerate(limited, start=1):
        row["rank"] = index
    return limited


def _build_payload(
    *,
    feature_pack: dict[str, Any],
    source_payload: dict[str, Any],
    feature_pack_json: Path,
    input_json: Path,
    top_n: int,
) -> dict[str, Any]:
    blocking_reasons = _blocking_reasons(feature_pack, source_payload)
    warnings = [] if blocking_reasons else _warning_reasons(feature_pack)
    status = "blocked" if blocking_reasons else "watch" if warnings else "ok"
    raw_candidates = _merge_candidates(source_payload)
    filtered_candidates, filter_stats = _filter_candidates(raw_candidates)
    candidates = [] if blocking_reasons else _rank_candidates(filtered_candidates, feature_pack, top_n=top_n)
    return {
        "schema_version": SIGNAL_SCORES_SCHEMA,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trade_date": feature_pack.get("trade_date") or source_payload.get("trade_date"),
        "research_only": True,
        "broker_action": "none",
        "status": status,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "market_regime": feature_pack.get("market_regime"),
        "sector_theme_status": feature_pack.get("sector_theme_status"),
        "top_industry_fund_flow": feature_pack.get("top_industry_fund_flow") if isinstance(feature_pack.get("top_industry_fund_flow"), dict) else {},
        "top_concept_fund_flow": feature_pack.get("top_concept_fund_flow") if isinstance(feature_pack.get("top_concept_fund_flow"), dict) else {},
        "feature_flags": _feature_flags(feature_pack),
        "feature_pack_path": str(feature_pack_json),
        "input_json_path": str(input_json),
        "candidate_filter_scope": MODEL2_CANDIDATE_SCOPE,
        "raw_candidate_count": len(raw_candidates),
        **filter_stats,
        "candidate_count": len(candidates),
        "top_n": int(top_n),
        "candidates": candidates,
        "usage_note": "Research scoring only. This is not a broker instruction, order, or investment recommendation.",
    }


def _write_report(payload: dict[str, Any], path: Path) -> None:
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    lines = [
        "# Model2 Signal Scores",
        "",
        f"- Status: {payload.get('status')}",
        f"- Trade date: {payload.get('trade_date')}",
        f"- Research only: {str(payload.get('research_only')).lower()}",
        f"- Broker action: {payload.get('broker_action')}",
        f"- Market regime: {payload.get('market_regime')}",
        f"- Candidate count: {payload.get('candidate_count')}",
        "",
        "## Top Candidates",
    ]
    if candidates:
        for row in candidates[:10]:
            lines.append(
                "- "
                f"#{row.get('rank')} {row.get('security_code')} {row.get('security_name')} "
                f"score={row.get('model2_score')} bucket={row.get('score_bucket')} "
                f"signal={row.get('research_signal')}"
            )
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Usage")
    lines.append("- Use these scores as a research-only candidate ranking input for Model2.")
    lines.append("- Do not use this file as a broker instruction or investment recommendation.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_model2_signal_scoring(
    *,
    feature_pack_json: Path | str = DEFAULT_MODEL2_FEATURE_PACK_JSON,
    input_json: Path | str = DEFAULT_MODEL2_INPUT_JSON,
    output_dir: Path | str = DEFAULT_MODEL2_SIGNAL_OUTPUT_DIR,
    top_n: int = 20,
) -> Model2SignalScoringResult:
    feature_pack_path = Path(feature_pack_json)
    input_path = Path(input_json)
    output_path = Path(output_dir)
    feature_pack = _read_json(feature_pack_path)
    source_payload = _read_json(input_path)
    payload = _build_payload(
        feature_pack=feature_pack,
        source_payload=source_payload,
        feature_pack_json=feature_pack_path,
        input_json=input_path,
        top_n=top_n,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    scores_path = output_path / "model2_signal_scores.json"
    report_path = output_path / "model2_signal_scores.md"
    scores_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    _write_report(payload, report_path)
    return Model2SignalScoringResult(
        output_dir=output_path,
        scores_path=scores_path,
        report_path=report_path,
        payload=payload,
        feature_pack=feature_pack,
        source_payload=source_payload,
    )
