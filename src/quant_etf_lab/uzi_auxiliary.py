"""Normalize UZI-Skill sidecar outputs into research-only signals."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_UZI_CACHE_ROOT = Path(r"C:\tmp\UZI-Skill\skills\deep-analysis\scripts\.cache")
DEFAULT_TICKERS = ["301165.SZ", "300870.SZ", "688629.SH"]
DEFAULT_OUTPUT_DIR = Path("outputs/research/uzi_auxiliary_signals_latest")
UZI_AUXILIARY_SOURCE_ID = "wbh604/UZI-Skill"

SIGNAL_FIELDS = [
    "ticker",
    "code",
    "market",
    "cache_status",
    "fundamental_score",
    "panel_consensus",
    "bullish_count",
    "neutral_count",
    "bearish_count",
    "skip_count",
    "bearish_ratio",
    "technical_consensus",
    "quant_consensus",
    "bottleneck_consensus",
    "valuation_overheat",
    "dcf_total_yi",
    "dcf_intrinsic_per_share",
    "current_price",
    "dcf_safety_margin_pct",
    "dcf_verdict",
    "data_coverage_pct",
    "critical_missing",
    "financial_label",
    "kline_label",
    "valuation_label",
    "integration_recommendation",
    "position_effect",
    "broker_action",
]


@dataclass(frozen=True)
class UZIAuxiliaryResult:
    output_dir: Path
    snapshot_path: Path
    signals_path: Path
    report_path: Path
    snapshot: dict[str, Any]


def normalize_uzi_ticker(value: str) -> str:
    """Normalize a six-digit A-share code into the UZI cache ticker form."""
    text = str(value).strip().upper()
    if not text:
        raise ValueError("UZI ticker cannot be empty.")
    if "." in text:
        code, market = text.split(".", 1)
        digits = "".join(ch for ch in code if ch.isdigit())[-6:].zfill(6)
        return f"{digits}.{market[:2].upper()}"
    digits = "".join(ch for ch in text if ch.isdigit())[-6:].zfill(6)
    market = "SH" if digits.startswith("6") else "SZ"
    return f"{digits}.{market}"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def _get(mapping: dict[str, Any], path: Iterable[str], default: Any = None) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "-", "--"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _round_or_none(value: Any, digits: int = 4) -> float | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return round(number, digits)


def _parse_pe(label: Any) -> float | None:
    if not label:
        return None
    match = re.search(r"\bPE\s*([0-9]+(?:\.[0-9]+)?)", str(label), flags=re.IGNORECASE)
    if not match:
        return None
    return _float_or_none(match.group(1))


def _center_dcf_value(raw: dict[str, Any]) -> float | None:
    values = _get(raw, ["dimensions", "10_valuation", "data", "dcf_sensitivity", "values"], [])
    if not isinstance(values, list) or not values:
        return None
    mid_row = values[len(values) // 2]
    if not isinstance(mid_row, list) or not mid_row:
        return None
    return _float_or_none(mid_row[len(mid_row) // 2])


def _recommendation(
    critical_missing: bool,
    valuation_overheat: bool,
    bearish_ratio: float | None,
    panel_consensus: float | None,
    bottleneck_consensus: float | None,
) -> str:
    if critical_missing:
        return "data_gap_review"
    if valuation_overheat and bearish_ratio is not None and bearish_ratio >= 0.50:
        return "risk_dampen_candidate"
    if (
        panel_consensus is not None
        and panel_consensus >= 55.0
        and bottleneck_consensus is not None
        and bottleneck_consensus >= 50.0
    ):
        return "positive_watch"
    return "observe_only"


def extract_uzi_auxiliary_signal(cache_root: str | Path, ticker: str) -> dict[str, Any]:
    """Extract one UZI cache directory into a flat research-only signal row."""
    normalized = normalize_uzi_ticker(ticker)
    code, market = normalized.split(".", 1)
    cache_dir = Path(cache_root) / normalized
    dimensions = _read_json(cache_dir / "dimensions.json")
    panel = _read_json(cache_dir / "panel.json")
    raw = _read_json(cache_dir / "raw_data.json")
    gaps = _read_json(cache_dir / "_data_gaps.json")

    cache_status = "ok" if dimensions and panel and raw else "missing_or_partial"
    dimension_rows = dimensions.get("dimensions") if isinstance(dimensions.get("dimensions"), dict) else {}
    signal_distribution = panel.get("signal_distribution") if isinstance(panel.get("signal_distribution"), dict) else {}
    school_scores = panel.get("school_scores") if isinstance(panel.get("school_scores"), dict) else {}

    bullish_count = int(_float_or_none(signal_distribution.get("bullish")) or 0)
    neutral_count = int(_float_or_none(signal_distribution.get("neutral")) or 0)
    bearish_count = int(_float_or_none(signal_distribution.get("bearish")) or 0)
    skip_count = int(_float_or_none(signal_distribution.get("skip")) or 0)
    active_count = bullish_count + neutral_count + bearish_count
    bearish_ratio = bearish_count / active_count if active_count > 0 else None

    dcf_total = _get(raw, ["dimensions", "10_valuation", "data", "dcf_simple", "intrinsic_value_total"])
    dcf_summary = _get(raw, ["dimensions", "20_valuation_models", "data", "summary"], {})
    dcf_model = _get(raw, ["dimensions", "20_valuation_models", "data", "dcf"], {})
    dcf_intrinsic = _get(dcf_summary, ["dcf_intrinsic"], None)
    if dcf_intrinsic is None:
        dcf_intrinsic = _get(dcf_model, ["intrinsic_per_share"], None)
    if dcf_intrinsic is None:
        dcf_intrinsic = _center_dcf_value(raw)

    current_price = _get(raw, ["dimensions", "10_valuation", "data", "dcf_sensitivity", "current_price"])
    if current_price is None:
        current_price = _get(dcf_model, ["current_price"], None)
    dcf_safety_margin = _get(dcf_summary, ["dcf_safety_margin_pct"], None)
    if dcf_safety_margin is None:
        dcf_safety_margin = _get(dcf_model, ["safety_margin_pct"], None)
    intrinsic_number = _float_or_none(dcf_intrinsic)
    price_number = _float_or_none(current_price)
    if dcf_safety_margin is None and intrinsic_number is not None and price_number and price_number > 0:
        dcf_safety_margin = (intrinsic_number / price_number - 1.0) * 100.0
    dcf_verdict = _get(dcf_summary, ["dcf_verdict"], None) or _get(dcf_model, ["verdict"], None)

    valuation_label = _get(dimension_rows, ["10_valuation", "label"], "")
    pe_value = _parse_pe(valuation_label)
    safety_number = _float_or_none(dcf_safety_margin)
    valuation_overheat = bool(
        (pe_value is not None and pe_value >= 80.0)
        or (safety_number is not None and safety_number <= -50.0)
    )
    critical_missing = bool(gaps.get("critical_missing", False))
    coverage = _float_or_none(gaps.get("coverage_pct"))
    if coverage is None:
        coverage = 100.0 if cache_status == "ok" else 0.0

    panel_consensus = _float_or_none(panel.get("panel_consensus"))
    bottleneck_consensus = _float_or_none(_get(school_scores, ["I", "consensus"], None))

    return {
        "ticker": normalized,
        "code": code,
        "market": market,
        "cache_status": cache_status,
        "fundamental_score": _round_or_none(dimensions.get("fundamental_score"), 2),
        "panel_consensus": _round_or_none(panel_consensus, 2),
        "bullish_count": bullish_count,
        "neutral_count": neutral_count,
        "bearish_count": bearish_count,
        "skip_count": skip_count,
        "bearish_ratio": _round_or_none(bearish_ratio, 4),
        "technical_consensus": _round_or_none(_get(school_scores, ["D", "consensus"], None), 2),
        "quant_consensus": _round_or_none(_get(school_scores, ["G", "consensus"], None), 2),
        "bottleneck_consensus": _round_or_none(bottleneck_consensus, 2),
        "valuation_overheat": valuation_overheat,
        "dcf_total_yi": _round_or_none((_float_or_none(dcf_total) or 0.0) / 100_000_000.0 if dcf_total is not None else None, 2),
        "dcf_intrinsic_per_share": _round_or_none(dcf_intrinsic, 2),
        "current_price": _round_or_none(current_price, 2),
        "dcf_safety_margin_pct": _round_or_none(dcf_safety_margin, 2),
        "dcf_verdict": dcf_verdict or "",
        "data_coverage_pct": _round_or_none(coverage, 2),
        "critical_missing": critical_missing,
        "financial_label": _get(dimension_rows, ["1_financials", "label"], ""),
        "kline_label": _get(dimension_rows, ["2_kline", "label"], ""),
        "valuation_label": valuation_label,
        "integration_recommendation": _recommendation(
            critical_missing=critical_missing,
            valuation_overheat=valuation_overheat,
            bearish_ratio=bearish_ratio,
            panel_consensus=panel_consensus,
            bottleneck_consensus=bottleneck_consensus,
        ),
        "position_effect": "none",
        "broker_action": "none",
    }


def build_uzi_auxiliary_snapshot(
    cache_root: str | Path = DEFAULT_UZI_CACHE_ROOT,
    tickers: Iterable[str] = DEFAULT_TICKERS,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a non-trading sidecar snapshot from UZI-Skill cache outputs."""
    rows = [extract_uzi_auxiliary_signal(cache_root, ticker) for ticker in tickers]
    return {
        "status": "uzi_auxiliary_signals_built",
        "source_id": UZI_AUXILIARY_SOURCE_ID,
        "generated_at": generated_at or datetime.now().isoformat(timespec="seconds"),
        "cache_root": str(Path(cache_root)),
        "integration_status": "research_auxiliary_only",
        "position_effect": "none",
        "broker_action": "none",
        "allowed_use": [
            "candidate_quality_review",
            "valuation_overheat_dampener_research",
            "bottleneck_evidence_review",
            "factor_lab_sidecar_ablation",
        ],
        "prohibited_integrations": [
            "default_allocator",
            "paper_account_target_weights",
            "daily_pipeline_position_sizing",
            "live_preflight_orders",
            "broker_order_generation",
        ],
        "promotion_gate": {
            "requires_point_in_time_rebuild": True,
            "requires_oos_ablation": True,
            "requires_data_gap_resolution": True,
            "requires_user_approval_before_allocator_use": True,
        },
        "signals": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIGNAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_report(snapshot: dict[str, Any]) -> str:
    rows = snapshot["signals"]
    table_rows = "\n".join(
        "| {ticker} | {fundamental_score} | {panel_consensus} | {bearish_ratio} | {technical_consensus} | {bottleneck_consensus} | {valuation_overheat} | {integration_recommendation} |".format(**row)
        for row in rows
    )
    prohibited = "\n".join(f"- `{item}`" for item in snapshot["prohibited_integrations"])
    return f"""# UZI-Skill 杈呭姪淇″彿瀵煎嚭

| 椤圭洰 | 鍊?|
| --- | --- |
| 鏉ユ簮 | `{snapshot["source_id"]}` |
| 鐘舵€?| `{snapshot["integration_status"]}` |
| 鐢熸垚鏃堕棿 | `{snapshot["generated_at"]}` |
| 浠撲綅褰卞搷 | `{snapshot["position_effect"]}` |
| 鍒稿晢鍔ㄤ綔 | `{snapshot["broker_action"]}` |

## 鏍锋湰缁撴灉

| 鏍囩殑 | 鍩虹鍒?| 璇勫鍏辫瘑 | 绌烘柟姣斾緥 | 鎶€鏈淳 | AI鐡堕娲?| 浼板€艰繃鐑?| 鎺ュ叆寤鸿 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
{table_rows}

## 纭竟鐣?
褰撳墠缁撴灉鍙綔涓虹爺绌惰緟鍔╁瓧娈碉紝涓嶅弬涓庝粨浣嶇敓鎴愩€佷笉鍙備笌鍒稿晢鎸囦护銆佷笉鏀瑰彉 daily pipeline 榛樿璺緞銆?
绂佹鐩存帴鎺ュ叆锛?{prohibited}

## 涓嬩竴姝ラ獙璇?
- 鍏堝仛 sidecar ablation锛岃瀵?`valuation_overheat` 鍜?`bearish_ratio` 鏄惁鑳藉噺灏戣拷楂樺洖鎾ゃ€?- 瀵?`critical_missing=True` 鐨勬爣鐨勮ˉ榻愯涓氬拰璐㈠姟缂哄彛鍚庡啀姣旇緝銆?- 閫氳繃婊氬姩 OOS 楠岃瘉鍓嶏紝涓嶅厑璁告妸 UZI 杈撳嚭鍗囩骇涓烘寮忎俊鍙锋潈閲嶃€?"""


def run_uzi_auxiliary_export(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    cache_root: str | Path = DEFAULT_UZI_CACHE_ROOT,
    tickers: Iterable[str] = DEFAULT_TICKERS,
    generated_at: str | None = None,
) -> UZIAuxiliaryResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    snapshot = build_uzi_auxiliary_snapshot(cache_root=cache_root, tickers=tickers, generated_at=generated_at)
    snapshot_path = output_path / "uzi_auxiliary_snapshot.json"
    signals_path = output_path / "uzi_auxiliary_signals.csv"
    report_path = output_path / "uzi_auxiliary_report.md"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(signals_path, list(snapshot["signals"]))
    report_path.write_text(_render_report(snapshot), encoding="utf-8")
    return UZIAuxiliaryResult(
        output_dir=output_path,
        snapshot_path=snapshot_path,
        signals_path=signals_path,
        report_path=report_path,
        snapshot=snapshot,
    )
