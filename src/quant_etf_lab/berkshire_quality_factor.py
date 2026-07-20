"""Research-only Berkshire-style quality factor scoring.

This module turns value-investing checklist ideas into auditable fields. It is
not a broker instruction, allocator switch, or automatic trading signal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CANDIDATES_PATH = Path("outputs/research/paper_account_latest/stock_targets.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/berkshire_quality_factor_latest")

FUNDAMENTAL_COLUMNS = [
    "code",
    "name",
    "sector_type",
    "listed_years",
    "roe_10y_avg",
    "fcf_5y_cumulative",
    "interest_coverage",
    "gross_margin_5y_avg",
    "ocf_to_net_income_5y_avg",
    "net_margin_10y_avg",
    "share_dilution_5y",
    "recent_ocf_positive",
    "net_margin_recovery",
    "business_model_type",
    "research_notes",
]
QUALITATIVE_COLUMNS = [
    "code",
    "circle_of_competence_score",
    "moat_score",
    "management_score",
    "safety_margin_score",
    "thesis_clarity_score",
    "red_flag_count",
    "data_confidence",
    "qualitative_notes",
]
TEMPLATE_COLUMNS = FUNDAMENTAL_COLUMNS + [column for column in QUALITATIVE_COLUMNS if column not in {"code"}]


@dataclass(frozen=True)
class BerkshireQualityFactorResult:
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
    if text.lower() in {"", "nan", "none"}:
        return ""
    return text


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _as_unit(value: Any, default: float = 0.0) -> float:
    number = _as_float(value, default)
    if number > 1.0:
        number = number / 100.0
    return max(min(number, 1.0), 0.0)


def _as_bool(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "ok", "pass"}:
        return True
    if text in {"0", "false", "no", "n", "", "nan", "none"}:
        return False
    return bool(value)


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
    if "priority_score" in data.columns:
        data["priority_score"] = pd.to_numeric(data["priority_score"], errors="coerce").fillna(0.0)
    elif "target_weight" in data.columns:
        weight = pd.to_numeric(data["target_weight"], errors="coerce").fillna(0.0)
        data["priority_score"] = (weight / weight.max() * 100.0).fillna(0.0) if float(weight.max() or 0.0) > 0 else 0.0
    else:
        data["priority_score"] = 0.0
    return data.loc[data["code"] != ""].drop_duplicates(subset=["code"], keep="first").reset_index(drop=True)


def _load_fundamentals(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)
    frame = pd.read_csv(path, dtype={"code": str, "security_code": str})
    code_column = "code" if "code" in frame.columns else "security_code" if "security_code" in frame.columns else None
    if code_column is None:
        raise ValueError("fundamentals file must contain code or security_code.")
    data = frame.copy()
    data["code"] = data[code_column].map(_clean_code)
    for column in FUNDAMENTAL_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    return data.loc[data["code"] != "", FUNDAMENTAL_COLUMNS].drop_duplicates(subset=["code"], keep="last")


def _load_qualitative(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=QUALITATIVE_COLUMNS)
    frame = pd.read_csv(path, dtype={"code": str, "security_code": str})
    code_column = "code" if "code" in frame.columns else "security_code" if "security_code" in frame.columns else None
    if code_column is None:
        raise ValueError("qualitative file must contain code or security_code.")
    data = frame.copy()
    data["code"] = data[code_column].map(_clean_code)
    for column in QUALITATIVE_COLUMNS:
        if column not in data.columns:
            data[column] = ""
    return data.loc[data["code"] != "", QUALITATIVE_COLUMNS].drop_duplicates(subset=["code"], keep="last")


def _gate_status(ok: bool, exempt: bool = False, missing: bool = False) -> str:
    if missing:
        return "missing"
    if exempt:
        return "exempt_pass"
    return "pass" if ok else "fail"


def _status_score(status: str) -> float:
    if status == "pass":
        return 1.0
    if status == "exempt_pass":
        return 0.8
    if status == "missing":
        return 0.35
    return 0.0


def _missing(value: Any) -> bool:
    return _clean_text(value) == ""


def _data_confidence_score(value: Any) -> float:
    text = _clean_text(value).upper()
    if text == "A":
        return 1.0
    if text == "B":
        return 0.75
    if text == "C":
        return 0.45
    return 0.55 if text else 0.35


def _five_point(value: Any) -> float:
    return max(min(_as_float(value, 0.0) / 5.0, 1.0), 0.0)


def _score_row(row: pd.Series) -> dict[str, Any]:
    has_fundamentals = bool(_clean_text(row.get("roe_10y_avg")) or _clean_text(row.get("fcf_5y_cumulative")))
    sector_type = _clean_text(row.get("sector_type")).lower()
    business_model = _clean_text(row.get("business_model_type")).lower()
    gross_margin = _as_unit(row.get("gross_margin_5y_avg"))
    listed_years = _as_float(row.get("listed_years"), 99.0)
    recent_ocf_positive = _as_bool(row.get("recent_ocf_positive"))
    net_margin_recovery = _as_bool(row.get("net_margin_recovery"))
    ocf_quality = _as_float(row.get("ocf_to_net_income_5y_avg"))
    roe = _as_unit(row.get("roe_10y_avg"))
    fcf = _as_float(row.get("fcf_5y_cumulative"))
    interest_coverage = _as_float(row.get("interest_coverage"))
    net_margin = _as_unit(row.get("net_margin_10y_avg"))
    dilution = _as_unit(row.get("share_dilution_5y"))

    strategic_investment_exemption = listed_years < 10 and gross_margin > 0.30 and recent_ocf_positive
    thin_margin_exemption = (
        roe > 0.20
        and ocf_quality > 1.0
        and any(token in business_model for token in ("membership", "platform", "turnover", "asset_light"))
    )
    active_low_margin_exemption = gross_margin > 0.30 and net_margin_recovery
    financial_sector_exemption = sector_type in {"bank", "banks", "insurance", "brokerage", "financial"}

    gates = {
        "roe_gate": _gate_status(roe >= 0.08, exempt=strategic_investment_exemption, missing=_missing(row.get("roe_10y_avg"))),
        "fcf_gate": _gate_status(fcf > 0.0, missing=_missing(row.get("fcf_5y_cumulative"))),
        "interest_coverage_gate": _gate_status(
            interest_coverage >= 2.0,
            exempt=financial_sector_exemption,
            missing=_missing(row.get("interest_coverage")) and not financial_sector_exemption,
        ),
        "gross_margin_gate": _gate_status(
            gross_margin >= 0.15,
            exempt=thin_margin_exemption,
            missing=_missing(row.get("gross_margin_5y_avg")),
        ),
        "ocf_quality_gate": _gate_status(ocf_quality >= 0.70, missing=_missing(row.get("ocf_to_net_income_5y_avg"))),
        "net_margin_gate": _gate_status(
            net_margin >= 0.05,
            exempt=active_low_margin_exemption or thin_margin_exemption,
            missing=_missing(row.get("net_margin_10y_avg")),
        ),
        "dilution_gate": _gate_status(dilution <= 0.20, missing=_missing(row.get("share_dilution_5y"))),
    }
    pass_count = sum(1 for status in gates.values() if status in {"pass", "exempt_pass"})
    fail_count = sum(1 for status in gates.values() if status == "fail")
    missing_count = sum(1 for status in gates.values() if status == "missing")
    financial_gate_score = sum(_status_score(status) for status in gates.values()) / max(len(gates), 1)

    qualitative_score = (
        0.18 * _five_point(row.get("circle_of_competence_score"))
        + 0.24 * _five_point(row.get("moat_score"))
        + 0.22 * _five_point(row.get("management_score"))
        + 0.20 * _five_point(row.get("safety_margin_score"))
        + 0.10 * _five_point(row.get("thesis_clarity_score"))
        + 0.06 * _data_confidence_score(row.get("data_confidence"))
    )
    priority_component = max(min(_as_float(row.get("priority_score")) / 100.0, 1.0), 0.0)
    red_flags = max(int(_as_float(row.get("red_flag_count"), 0.0)), 0)
    hard_veto = bool(red_flags > 0 or fail_count >= 4)
    raw_score = 100.0 * (0.58 * financial_gate_score + 0.32 * qualitative_score + 0.10 * priority_component)
    score = max(min(raw_score - 20.0 * red_flags - 5.0 * max(fail_count - 2, 0), 100.0), 0.0)

    if not has_fundamentals:
        bucket = "needs_fundamental_research"
    elif hard_veto:
        bucket = "berkshire_veto"
    elif score >= 70.0 and pass_count >= 5:
        bucket = "berkshire_quality_priority"
    elif score >= 50.0:
        bucket = "berkshire_watch"
    else:
        bucket = "berkshire_low_quality"

    return {
        **gates,
        "berkshire_financial_gate_score": round(100.0 * financial_gate_score, 3),
        "berkshire_qualitative_score": round(100.0 * qualitative_score, 3),
        "berkshire_quality_score": round(score, 3),
        "berkshire_pass_count": int(pass_count),
        "berkshire_fail_count": int(fail_count),
        "berkshire_missing_count": int(missing_count),
        "berkshire_hard_veto": hard_veto,
        "berkshire_bucket": bucket,
    }


def _build_template(scores: pd.DataFrame, top_n: int) -> pd.DataFrame:
    base = scores.sort_values(["priority_score", "code"], ascending=[False, True]).head(max(int(top_n), 0))
    rows: list[dict[str, Any]] = []
    for _, row in base.iterrows():
        item = {column: row.get(column, "") for column in TEMPLATE_COLUMNS}
        item["code"] = row.get("code", "")
        item["name"] = row.get("name", "")
        if row.get("berkshire_bucket") == "needs_fundamental_research":
            item["research_notes"] = (
                "Fill Berkshire quality data: ROE, FCF, interest coverage, margins, cash conversion, dilution, "
                "moat, management, safety margin, thesis clarity, red flags, and data confidence."
            )
        rows.append(item)
    return pd.DataFrame(rows, columns=TEMPLATE_COLUMNS)


def _write_report(snapshot: dict[str, Any], scores: pd.DataFrame) -> str:
    lines = [
        "# Berkshire Quality Factor",
        "",
        f"- generated_at: {snapshot['generated_at']}",
        f"- status: {snapshot['status']}",
        f"- candidate_count: {snapshot['candidate_count']}",
        f"- mapped_candidate_count: {snapshot['mapped_candidate_count']}",
        f"- broker_action: {snapshot['broker_action']}",
        "",
        "## Method",
        "",
        "- Apply seven value-investing quality gates before interpreting technical signals.",
        "- Penalize red flags, weak cash conversion, high dilution, and missing fundamental evidence.",
        "- Keep the output research-only; downstream models must backtest any use of these fields.",
        "",
        "## Top Scores",
        "",
    ]
    if scores.empty:
        lines.append("No candidates available.")
    else:
        lines.append("| rank | code | name | bucket | score | pass | fail | missing |")
        lines.append("| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |")
        top = scores.sort_values(["berkshire_quality_score", "priority_score"], ascending=[False, False]).head(20)
        for rank, (_, row) in enumerate(top.iterrows(), start=1):
            lines.append(
                "| "
                f"{rank} | {row.get('code', '')} | {row.get('name', '')} | {row.get('berkshire_bucket', '')} | "
                f"{float(row.get('berkshire_quality_score', 0.0)):.3f} | {int(row.get('berkshire_pass_count', 0))} | "
                f"{int(row.get('berkshire_fail_count', 0))} | {int(row.get('berkshire_missing_count', 0))} |"
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


def run_berkshire_quality_factor(
    *,
    candidates_path: str | Path = DEFAULT_CANDIDATES_PATH,
    fundamentals_path: str | Path | None = None,
    qualitative_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    top_n: int = 30,
) -> BerkshireQualityFactorResult:
    candidates_file = Path(candidates_path)
    fundamentals_file = Path(fundamentals_path) if fundamentals_path not in (None, "") else None
    qualitative_file = Path(qualitative_path) if qualitative_path not in (None, "") else None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    candidates = _load_candidates(candidates_file)
    fundamentals = _load_fundamentals(fundamentals_file)
    qualitative = _load_qualitative(qualitative_file)

    data = candidates.merge(fundamentals.drop(columns=["name"], errors="ignore"), on="code", how="left")
    data = data.merge(qualitative, on="code", how="left")
    for column in TEMPLATE_COLUMNS:
        if column not in data.columns:
            data[column] = ""

    scored_rows = []
    for _, row in data.iterrows():
        scored = dict(row)
        scored.update(_score_row(row))
        scored_rows.append(scored)
    scores = pd.DataFrame(scored_rows)
    if not scores.empty:
        scores = scores.sort_values(["berkshire_quality_score", "priority_score"], ascending=[False, False]).reset_index(drop=True)
        scores.insert(0, "berkshire_quality_rank", range(1, len(scores) + 1))

    template = _build_template(scores, top_n=top_n)
    scores_path = output_path / "berkshire_quality_scores.csv"
    template_path = output_path / "berkshire_quality_research_template.csv"
    snapshot_path = output_path / "berkshire_quality_snapshot.json"
    report_path = output_path / "berkshire_quality_factor.md"

    scores.to_csv(scores_path, index=False, encoding="utf-8-sig")
    template.to_csv(template_path, index=False, encoding="utf-8-sig")

    mapped_count = int((scores["berkshire_bucket"] != "needs_fundamental_research").sum()) if not scores.empty else 0
    status = "ok" if mapped_count > 0 else "needs_fundamental_data"
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "research_only": True,
        "broker_action": "none",
        "method_source": "ai_berkshire_quality_screen_and_checklist",
        "candidates_path": str(candidates_file),
        "fundamentals_path": None if fundamentals_file is None else str(fundamentals_file),
        "qualitative_path": None if qualitative_file is None else str(qualitative_file),
        "output_dir": str(output_path),
        "candidate_count": int(len(candidates)),
        "mapped_candidate_count": mapped_count,
        "unmapped_candidate_count": int(len(candidates) - mapped_count),
        "scores_path": str(scores_path),
        "template_path": str(template_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "top_n": int(top_n),
        "gate_columns": [
            "roe_gate",
            "fcf_gate",
            "interest_coverage_gate",
            "gross_margin_gate",
            "ocf_quality_gate",
            "net_margin_gate",
            "dilution_gate",
        ],
        "score_components": {
            "financial_gate_score": 0.58,
            "qualitative_score": 0.32,
            "priority_score": 0.10,
            "red_flag_penalty": 20.0,
        },
        "top_scores": scores.head(20).to_dict(orient="records") if not scores.empty else [],
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_write_report(snapshot, scores), encoding="utf-8")

    return BerkshireQualityFactorResult(
        output_dir=output_path,
        scores_path=scores_path,
        template_path=template_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        scores=scores,
    )
