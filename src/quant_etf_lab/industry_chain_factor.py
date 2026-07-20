"""Research-only industry-chain factor scoring.

The factor converts industry-research notes into a measurable companion score:
long-term demand, chain bottlenecks, supply constraints, attention gaps, value
capture, catalysts, and risk penalties. It does not allocate capital or route
broker actions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CANDIDATES_PATH = Path("outputs/research/chip_reversal_daily_candidates_20260622/chip_reversal_daily_candidates.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/industry_chain_factor_latest")
CHAIN_TEMPLATE_COLUMNS = [
    "code",
    "name",
    "industry_chain",
    "chain_node",
    "node_role",
    "demand_certainty",
    "supply_constraint",
    "attention_gap",
    "value_capture",
    "catalyst_strength",
    "upstream_dependency",
    "customer_validation",
    "qualification_cycle",
    "pricing_power",
    "substitution_risk",
    "crowding_risk",
    "valuation_risk",
    "dilution_risk",
    "financing_risk",
    "research_notes",
]
EVIDENCE_COLUMNS = [
    "demand_certainty",
    "supply_constraint",
    "attention_gap",
    "value_capture",
    "catalyst_strength",
    "upstream_dependency",
    "customer_validation",
    "qualification_cycle",
    "pricing_power",
]
RISK_COLUMNS = ["substitution_risk", "crowding_risk", "valuation_risk", "dilution_risk", "financing_risk"]
ROLE_SCORES = {
    "bottleneck": 1.00,
    "critical": 0.95,
    "core": 0.90,
    "scarce": 0.85,
    "alternative": 0.70,
    "upstream": 0.65,
    "downstream": 0.55,
    "supporting": 0.45,
}


@dataclass(frozen=True)
class IndustryChainFactorResult:
    output_dir: Path
    scores_path: Path
    template_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    scores: pd.DataFrame


def _clean_code(value: Any) -> str:
    if value in (None, ""):
        return ""
    digits = "".join(ch for ch in str(value).strip() if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    return digits.zfill(6) if digits else ""


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "nan", "none", "0"}:
        return ""
    return text


def _as_unit(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    if number > 1.0:
        number = number / 100.0
    return max(min(number, 1.0), 0.0)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _load_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing candidates file: {path}")
    frame = pd.read_csv(path, dtype={"code": str, "security_code": str})
    code_column = "code" if "code" in frame.columns else "security_code" if "security_code" in frame.columns else None
    if code_column is None:
        raise ValueError("candidates file must contain code or security_code.")
    name_column = "name" if "name" in frame.columns else "security_name" if "security_name" in frame.columns else None
    data = frame.copy()
    data["code"] = data[code_column].map(_clean_code)
    data["name"] = data[name_column].map(_clean_text) if name_column else data["code"]
    if "priority_score" not in data.columns:
        data["priority_score"] = 0.0
    data["priority_score"] = pd.to_numeric(data["priority_score"], errors="coerce").fillna(0.0)
    return data.loc[data["code"] != ""].drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)


def _load_chain_map(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=CHAIN_TEMPLATE_COLUMNS)
    frame = pd.read_csv(path, dtype={"code": str, "security_code": str})
    code_column = "code" if "code" in frame.columns else "security_code" if "security_code" in frame.columns else None
    if code_column is None:
        raise ValueError("industry chain map must contain code or security_code.")
    data = frame.copy()
    data["code"] = data[code_column].map(_clean_code)
    for column in CHAIN_TEMPLATE_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    for column in EVIDENCE_COLUMNS + RISK_COLUMNS:
        data[column] = data[column].map(_as_unit)
    return data.loc[data["code"] != "", CHAIN_TEMPLATE_COLUMNS].drop_duplicates(subset=["code"], keep="last")


def _load_market_snapshot(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["code", "market_cap", "turnover_rate", "change_ratio"])
    frame = pd.read_csv(path, dtype={"security_code": str, "code": str})
    code_column = "security_code" if "security_code" in frame.columns else "code" if "code" in frame.columns else None
    if code_column is None:
        return pd.DataFrame(columns=["code", "market_cap", "turnover_rate", "change_ratio"])
    data = pd.DataFrame({"code": frame[code_column].map(_clean_code)})
    for column in ["market_cap", "turnover_rate", "change_ratio"]:
        data[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else 0.0
    return data.loc[data["code"] != ""].drop_duplicates(subset=["code"], keep="last")


def _role_score(value: Any) -> float:
    text = _clean_text(value).lower().replace(" ", "_")
    if text in ROLE_SCORES:
        return ROLE_SCORES[text]
    if "bottleneck" in text or "瓶颈" in text:
        return 1.0
    if "core" in text or "核心" in text:
        return 0.9
    if "alternative" in text or "替代" in text:
        return 0.7
    return 0.35 if text else 0.0


def _market_fit_score(row: pd.Series) -> float:
    market_cap = _as_float(row.get("market_cap"))
    turnover_rate = _as_float(row.get("turnover_rate"))
    change_ratio = _as_float(row.get("change_ratio"))
    market_cap_yi = market_cap / 100_000_000.0 if market_cap > 10_000_000 else market_cap
    if 200.0 <= market_cap_yi <= 1000.0:
        cap_score = 1.0
    elif 100.0 <= market_cap_yi < 200.0 or market_cap_yi > 1000.0:
        cap_score = 0.7
    else:
        cap_score = 0.35 if market_cap_yi > 0 else 0.0
    turnover_score = 1.0 - min(max(turnover_rate - 8.0, 0.0) / 12.0, 1.0)
    jump_score = 1.0 - min(max(change_ratio - 10.0, 0.0) / 10.0, 1.0)
    return max(min(0.50 * cap_score + 0.25 * turnover_score + 0.25 * jump_score, 1.0), 0.0)


def _score_row(row: pd.Series) -> dict[str, Any]:
    role_score = _role_score(row.get("node_role"))
    evidence_score = (
        0.22 * _as_unit(row.get("demand_certainty"))
        + 0.22 * _as_unit(row.get("supply_constraint"))
        + 0.16 * _as_unit(row.get("attention_gap"))
        + 0.20 * _as_unit(row.get("value_capture"))
        + 0.12 * _as_unit(row.get("catalyst_strength"))
        + 0.08 * role_score
    )
    risk_penalty = (
        0.15 * _as_unit(row.get("substitution_risk"))
        + 0.10 * _as_unit(row.get("crowding_risk"))
        + 0.10 * _as_unit(row.get("valuation_risk"))
        + 0.05 * _as_unit(row.get("dilution_risk"))
        + 0.05 * _as_unit(row.get("financing_risk"))
    )
    chain_score = max(min(evidence_score - risk_penalty, 1.0), 0.0)
    bottleneck_evidence = (
        0.18 * _as_unit(row.get("demand_certainty"))
        + 0.18 * _as_unit(row.get("supply_constraint"))
        + 0.16 * _as_unit(row.get("upstream_dependency"))
        + 0.14 * _as_unit(row.get("customer_validation"))
        + 0.14 * _as_unit(row.get("qualification_cycle"))
        + 0.12 * _as_unit(row.get("pricing_power"))
        + 0.08 * role_score
    )
    bottleneck_penalty = (
        0.20 * _as_unit(row.get("substitution_risk"))
        + 0.12 * _as_unit(row.get("crowding_risk"))
        + 0.12 * _as_unit(row.get("valuation_risk"))
        + 0.10 * _as_unit(row.get("dilution_risk"))
        + 0.08 * _as_unit(row.get("financing_risk"))
    )
    bottleneck_score = max(min(bottleneck_evidence - bottleneck_penalty, 1.0), 0.0)
    priority_component = max(min(_as_float(row.get("priority_score")) / 100.0, 1.0), 0.0)
    market_fit = _market_fit_score(row)
    score = 100.0 * (0.64 * chain_score + 0.18 * bottleneck_score + 0.10 * priority_component + 0.08 * market_fit)
    has_chain_map = bool(_clean_text(row.get("industry_chain")))
    if chain_score >= 0.70 and score >= 70.0:
        bucket = "core_chain_priority"
    elif chain_score >= 0.45:
        bucket = "chain_watch"
    elif has_chain_map:
        bucket = "mapped_low_conviction"
    else:
        bucket = "needs_chain_research"
    bottleneck_score_pct = 100.0 * bottleneck_score
    if not has_chain_map:
        bottleneck_bucket = "serenity_needs_chain_research"
    elif (
        bottleneck_score_pct >= 70.0
        and _as_unit(row.get("supply_constraint")) >= 0.65
        and _as_unit(row.get("upstream_dependency")) >= 0.55
    ):
        bottleneck_bucket = "serenity_core_bottleneck"
    elif bottleneck_score_pct >= 45.0:
        bottleneck_bucket = "serenity_watch"
    else:
        bottleneck_bucket = "serenity_low_conviction"
    return {
        "industry_chain_evidence_score": round(evidence_score, 6),
        "industry_chain_risk_penalty": round(risk_penalty, 6),
        "serenity_bottleneck_evidence_score": round(bottleneck_evidence, 6),
        "serenity_bottleneck_risk_penalty": round(bottleneck_penalty, 6),
        "serenity_bottleneck_score": round(bottleneck_score_pct, 3),
        "serenity_bottleneck_bucket": bottleneck_bucket,
        "chain_node_role_score": round(role_score, 6),
        "market_fit_score": round(market_fit, 6),
        "industry_chain_factor_score": round(score, 3),
        "factor_bucket": bucket,
    }


def _build_template(scores: pd.DataFrame, top_n: int) -> pd.DataFrame:
    base = scores.sort_values(["industry_chain_factor_score", "priority_score"], ascending=[False, False]).head(max(int(top_n), 0))
    rows: list[dict[str, Any]] = []
    for _, row in base.iterrows():
        item = {column: row.get(column, "") for column in CHAIN_TEMPLATE_COLUMNS}
        item["code"] = row.get("code", "")
        item["name"] = row.get("name", "")
        item["research_notes"] = _clean_text(row.get("research_notes"))
        if row.get("factor_bucket") == "needs_chain_research":
            item["research_notes"] = "Fill chain node, bottleneck role, demand, supply, attention gap, value capture, catalyst, and risks."
        rows.append(item)
    return pd.DataFrame(rows, columns=CHAIN_TEMPLATE_COLUMNS)


def _write_report(snapshot: dict[str, Any], scores: pd.DataFrame) -> str:
    lines = [
        "# Industry Chain Factor Research",
        "",
        f"- generated_at: {snapshot['generated_at']}",
        f"- status: {snapshot['status']}",
        f"- candidate_count: {snapshot['candidate_count']}",
        f"- mapped_candidate_count: {snapshot['mapped_candidate_count']}",
        f"- broker_action: {snapshot['broker_action']}",
        "",
        "## Method",
        "",
        "- Confirm long-term demand before short-term heat.",
        "- Split the chain into key nodes and prefer bottlenecks or scarce alternatives.",
        "- Reward low-attention evidence only when value capture and catalysts are visible.",
        "- Penalize substitution risk, crowding, and valuation risk.",
        "- Keep the output research-only until outcome labels and allocator gates confirm it.",
        "",
        "## Top Scores",
        "",
    ]
    if scores.empty:
        lines.append("No candidates available.")
    else:
        lines.append("| rank | code | name | chain | node | bucket | score |")
        lines.append("| ---: | --- | --- | --- | --- | --- | ---: |")
        top = scores.sort_values(["industry_chain_factor_score", "priority_score"], ascending=[False, False]).head(20)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            lines.append(
                "| "
                f"{rank} | {row.get('code', '')} | {row.get('name', '')} | "
                f"{row.get('industry_chain', '')} | {row.get('chain_node', '')} | "
                f"{row.get('factor_bucket', '')} | {float(row.get('industry_chain_factor_score', 0.0)):.3f} |"
            )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- scores: {snapshot['scores_path']}",
            f"- template: {snapshot['template_path']}",
            f"- snapshot: {snapshot['snapshot_path']}",
            "",
        ]
    )
    return "\n".join(lines)


def run_industry_chain_factor(
    *,
    candidates_path: str | Path = DEFAULT_CANDIDATES_PATH,
    chain_map_path: str | Path | None = None,
    market_snapshot_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n: int = 30,
) -> IndustryChainFactorResult:
    candidates_file = Path(candidates_path)
    chain_map_file = Path(chain_map_path) if chain_map_path not in (None, "") else None
    market_snapshot_file = Path(market_snapshot_path) if market_snapshot_path not in (None, "") else None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    candidates = _load_candidates(candidates_file)
    chain_map = _load_chain_map(chain_map_file)
    market_snapshot = _load_market_snapshot(market_snapshot_file)

    chain_map_for_merge = chain_map.drop(columns=["name"], errors="ignore")
    data = candidates.merge(chain_map_for_merge, on="code", how="left", suffixes=("", "_chain"))
    if "industry_chain" not in data.columns:
        for column in CHAIN_TEMPLATE_COLUMNS:
            if column not in data.columns:
                data[column] = ""
    for column in CHAIN_TEMPLATE_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    for column in EVIDENCE_COLUMNS + RISK_COLUMNS:
        data[column] = data[column].map(_as_unit)
    data = data.merge(market_snapshot, on="code", how="left", suffixes=("", "_market"))
    for column in ["market_cap", "turnover_rate", "change_ratio"]:
        if column not in data.columns:
            data[column] = 0.0
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)

    scored_rows = []
    for _, row in data.iterrows():
        scored = dict(row)
        scored.update(_score_row(row))
        scored_rows.append(scored)
    scores = pd.DataFrame(scored_rows)
    if not scores.empty:
        scores = scores.sort_values(["industry_chain_factor_score", "priority_score"], ascending=[False, False]).reset_index(drop=True)
        scores.insert(0, "industry_chain_factor_rank", range(1, len(scores) + 1))

    template = _build_template(scores, top_n=top_n)
    scores_path = output_path / "industry_chain_factor_scores.csv"
    template_path = output_path / "industry_chain_research_template.csv"
    snapshot_path = output_path / "industry_chain_factor_snapshot.json"
    report_path = output_path / "industry_chain_factor.md"

    scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
    template.to_csv(template_path, index=False, encoding="utf-8-sig")

    mapped_count = int(scores["industry_chain"].fillna("").astype(str).str.strip().ne("").sum()) if not scores.empty else 0
    status = "ok" if mapped_count > 0 else "needs_chain_map"
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "research_only": True,
        "broker_action": "none",
        "method_source": "industry_chain_core_node_research_factor",
        "candidates_path": str(candidates_file),
        "chain_map_path": None if chain_map_file is None else str(chain_map_file),
        "market_snapshot_path": None if market_snapshot_file is None else str(market_snapshot_file),
        "output_dir": str(output_path),
        "candidate_count": int(len(candidates)),
        "mapped_candidate_count": mapped_count,
        "unmapped_candidate_count": int(len(candidates) - mapped_count),
        "scores_path": str(scores_path),
        "template_path": str(template_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "top_n": int(top_n),
        "factor_components": {
            "demand_certainty": 0.22,
            "supply_constraint": 0.22,
            "attention_gap": 0.16,
            "value_capture": 0.20,
            "catalyst_strength": 0.12,
            "node_role": 0.08,
            "risk_penalty": {
                "substitution_risk": 0.15,
                "crowding_risk": 0.10,
                "valuation_risk": 0.10,
                "dilution_risk": 0.05,
                "financing_risk": 0.05,
            },
            "serenity_bottleneck_components": {
                "demand_certainty": 0.18,
                "supply_constraint": 0.18,
                "upstream_dependency": 0.16,
                "customer_validation": 0.14,
                "qualification_cycle": 0.14,
                "pricing_power": 0.12,
                "node_role": 0.08,
                "risk_penalty": {
                    "substitution_risk": 0.20,
                    "crowding_risk": 0.12,
                    "valuation_risk": 0.12,
                    "dilution_risk": 0.10,
                    "financing_risk": 0.08,
                },
            },
        },
        "top_scores": scores.head(20).to_dict(orient="records") if not scores.empty else [],
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_write_report(snapshot, scores), encoding="utf-8")

    return IndustryChainFactorResult(
        output_dir=output_path,
        scores_path=scores_path,
        template_path=template_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        scores=scores,
    )
