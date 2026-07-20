"""Research-only model2 input adapter for the unified A-share data export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .market_data_source import DEFAULT_DAILY_MARKET_DATA_DIR, DEFAULT_EXCHANGE_INGEST_DIR, load_market_snapshot_rows


DEFAULT_MODEL2_INPUT_JSON = Path("D:/codex/daily-market-data/exports/latest_model2_input_with_sector.json")
DEFAULT_MODEL2_EXPORT_DIR = Path("D:/codex/daily-market-data/exports")
DEFAULT_MODEL2_OUTPUT_DIR = Path("outputs/research/model2_data_input_latest")
DEFAULT_MIN_MARKET_ROW_COUNT = 5000
EXPECTED_SOURCE_SCHEMA = "a-share-model2-input-v1"


@dataclass(frozen=True)
class Model2DataInputResult:
    output_dir: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    source_payload: dict[str, Any]


@dataclass(frozen=True)
class Model2InputExportResult:
    output_dir: Path
    payload_path: Path
    latest_payload_path: Path
    latest_with_sector_path: Path
    snapshot_path: Path
    payload: dict[str, Any]
    snapshot: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Model2 input JSON not found: {path}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"Model2 input JSON must be an object: {path}")
    return payload


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _preview_rows(rows: Any, *, limit: int = 5) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    preview: list[dict[str, Any]] = []
    keep_fields = ("security_code", "security_name", "change_ratio", "turnover", "volume", "source")
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        preview.append({key: row.get(key) for key in keep_fields if key in row})
    return preview


def _as_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in {float("inf"), float("-inf")}:
        return default
    return number


def _clean_export_row(row: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "market",
        "security_code",
        "security_name",
        "change_ratio",
        "turnover",
        "volume",
        "source",
    ]
    out = {key: row.get(key) for key in keep if key in row}
    for key in ["change_ratio", "turnover", "volume"]:
        if key in out:
            out[key] = _as_number(out.get(key))
    return out


def _top_rows(rows: list[dict[str, Any]], key: str, *, reverse: bool, limit: int = 20) -> list[dict[str, Any]]:
    return [
        _clean_export_row(row)
        for row in sorted(rows, key=lambda item: _as_number(item.get(key)), reverse=reverse)[:limit]
    ]


def _segment_name(code: Any) -> str:
    text = str(code or "").strip().lower()
    digits = "".join(ch for ch in text if ch.isdigit())
    if text.startswith("bj") or digits.startswith(("83", "87", "88", "92")):
        return "beijing_exchange"
    if text.startswith(("sh688", "sh689")) or digits.startswith(("688", "689")):
        return "star_market"
    if text.startswith("sz30") or digits.startswith("30"):
        return "chinext"
    if text.startswith("sh") or digits.startswith("60"):
        return "sse_main"
    if text.startswith("sz00") or digits.startswith("00"):
        return "szse_main"
    return "other"


def _listing_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        segment = _segment_name(row.get("security_code"))
        counts[segment] = counts.get(segment, 0) + 1
    return [{"segment": key, "row_count": counts[key]} for key in sorted(counts)]


def _sector_payload_for_date(path: Path | None, trade_date: str | None) -> tuple[dict[str, Any] | None, str, str | None]:
    if path is None or not path.exists():
        return None, "missing", None
    payload = _read_json(path)
    sector_date = str(payload.get("trade_date") or "")
    if trade_date and sector_date != trade_date:
        return None, "missing_for_trade_date", sector_date or None
    status = str(payload.get("overall_status") or "").strip().lower() or "unknown"
    return payload, status, sector_date or None


def _build_export_payload(
    *,
    rows: list[dict[str, Any]],
    trade_date: str,
    source_path: Path | None,
    sector_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    changes = [_as_number(row.get("change_ratio")) for row in rows]
    advancing = sum(1 for value in changes if value > 0)
    declining = sum(1 for value in changes if value < 0)
    flat = max(len(changes) - advancing - declining, 0)
    payload: dict[str, Any] = {
        "schema_version": EXPECTED_SOURCE_SCHEMA,
        "trade_date": trade_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "market_snapshot": {
            "path": str(source_path) if source_path is not None else None,
            "row_count": int(len(rows)),
            "required_columns": [
                "market",
                "trade_date",
                "security_code",
                "security_name",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "prev_close",
                "last_price",
                "change_amount",
                "change_ratio",
                "volume",
                "turnover",
                "source",
            ],
        },
        "market_breadth": {
            "advancing": int(advancing),
            "declining": int(declining),
            "flat": int(flat),
            "advance_ratio_pct": round(advancing / len(rows) * 100.0, 2) if rows else 0.0,
        },
        "top_gainers": _top_rows(rows, "change_ratio", reverse=True),
        "top_losers": _top_rows(rows, "change_ratio", reverse=False),
        "turnover_leaders": _top_rows(rows, "turnover", reverse=True),
        "listing_segments": _listing_segments(rows),
        "workflow_package": {},
        "usage_note": "Research input only. No broker connection, order placement, or investment advice.",
    }
    if sector_payload is not None:
        payload["sector_theme_data"] = sector_payload
    return payload


def _sector_status(sector_payload: Any) -> str:
    if not isinstance(sector_payload, dict):
        return "missing"
    status = str(sector_payload.get("overall_status") or "").strip().lower()
    return status or "unknown"


def _dict_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _flow_value(row: dict[str, Any]) -> float:
    for key in ("net_inflow", "net_amount", "main_net_inflow", "amount"):
        if key in row:
            return _as_number(row.get(key))
    return 0.0


def _clean_flow_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("board_type", "name", "code", "source", "company_count"):
        if key in row:
            out[key] = row.get(key)
    for key in ("net_inflow", "net_amount"):
        if key in row:
            out[key] = _as_number(row.get(key))
    out["flow_value"] = _flow_value(row)
    return out


def _top_fund_flow(rows: Any) -> dict[str, Any]:
    cleaned = [_clean_flow_row(row) for row in _dict_rows(rows)]
    if not cleaned:
        return {}
    return max(cleaned, key=lambda row: _as_number(row.get("flow_value")))


def _market_regime(advance_ratio_pct: Any) -> str:
    ratio = _as_number(advance_ratio_pct, default=-1.0)
    if ratio < 0:
        return "unknown_breadth"
    if ratio >= 60:
        return "broad_strength"
    if ratio <= 40:
        return "weak_breadth"
    return "mixed_breadth"


def _build_feature_flags(
    *,
    snapshot: dict[str, Any],
    market_regime: str,
    industry_flow_rows: list[dict[str, Any]],
    concept_flow_rows: list[dict[str, Any]],
) -> list[str]:
    flags = [market_regime]
    flags.append("sector_theme_ok" if snapshot.get("sector_theme_status") == "ok" else "sector_theme_not_ok")
    flags.append("data_ready" if snapshot.get("status") == "ok" else f"data_{snapshot.get('status')}")
    if industry_flow_rows or concept_flow_rows:
        flags.append("fund_flow_available")
    else:
        flags.append("fund_flow_missing")
    return flags


def _build_feature_pack(payload: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    sector_theme = payload.get("sector_theme_data") if isinstance(payload.get("sector_theme_data"), dict) else {}
    industry_flow_rows = _dict_rows(sector_theme.get("industry_fund_flow"))
    concept_flow_rows = _dict_rows(sector_theme.get("concept_fund_flow"))
    market_regime = _market_regime(snapshot.get("advance_ratio_pct"))
    status = str(snapshot.get("status") or "unknown")
    pack_status = "blocked" if status == "blocked" else "watch" if status == "watch" else "ok"

    feature_pack = {
        "schema_version": "quant-model2-feature-pack-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trade_date": snapshot.get("trade_date"),
        "research_only": True,
        "broker_action": "none",
        "status": pack_status,
        "source_status": status,
        "blocking_reasons": snapshot.get("blocking_reasons") or [],
        "warnings": snapshot.get("warnings") or [],
        "market_regime": market_regime,
        "advance_ratio_pct": snapshot.get("advance_ratio_pct"),
        "advancing": snapshot.get("advancing"),
        "declining": snapshot.get("declining"),
        "flat": snapshot.get("flat"),
        "market_row_count": snapshot.get("market_row_count"),
        "sector_theme_status": snapshot.get("sector_theme_status"),
        "industry_board_count": snapshot.get("industry_board_count"),
        "concept_board_count": snapshot.get("concept_board_count"),
        "industry_fund_flow_count": len(industry_flow_rows),
        "concept_fund_flow_count": len(concept_flow_rows),
        "top_industry_fund_flow": _top_fund_flow(industry_flow_rows),
        "top_concept_fund_flow": _top_fund_flow(concept_flow_rows),
        "feature_flags": _build_feature_flags(
            snapshot=snapshot,
            market_regime=market_regime,
            industry_flow_rows=industry_flow_rows,
            concept_flow_rows=concept_flow_rows,
        ),
        "usage_note": "Research feature pack only. Do not use as a broker instruction or investment recommendation.",
    }
    return feature_pack


def _write_feature_report(feature_pack: dict[str, Any], path: Path) -> None:
    industry = feature_pack.get("top_industry_fund_flow") if isinstance(feature_pack.get("top_industry_fund_flow"), dict) else {}
    concept = feature_pack.get("top_concept_fund_flow") if isinstance(feature_pack.get("top_concept_fund_flow"), dict) else {}
    flags = feature_pack.get("feature_flags") if isinstance(feature_pack.get("feature_flags"), list) else []
    lines = [
        "# Model2 Feature Pack",
        "",
        f"- Status: {feature_pack.get('status')}",
        f"- Trade date: {feature_pack.get('trade_date')}",
        f"- Research only: {str(feature_pack.get('research_only')).lower()}",
        f"- Broker action: {feature_pack.get('broker_action')}",
        f"- Market regime: {feature_pack.get('market_regime')}",
        f"- Advance ratio pct: {feature_pack.get('advance_ratio_pct')}",
        f"- Market rows: {feature_pack.get('market_row_count')}",
        f"- Sector/theme status: {feature_pack.get('sector_theme_status')}",
        f"- Top industry flow: {industry.get('name')} ({industry.get('flow_value')})",
        f"- Top concept flow: {concept.get('name')} ({concept.get('flow_value')})",
        "",
        "## Feature Flags",
    ]
    lines.extend([f"- {item}" for item in flags] or ["- None"])
    lines.append("")
    lines.append("## Usage")
    lines.append("- Use this as a machine-readable research feature pack for model2.")
    lines.append("- Do not use it as a broker instruction or investment recommendation.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_snapshot(
    payload: dict[str, Any],
    *,
    input_json: Path,
    min_market_row_count: int,
    require_sector_theme: bool,
) -> dict[str, Any]:
    source_schema_version = str(payload.get("schema_version") or payload.get("schema") or "")
    trade_date = payload.get("trade_date")
    market_snapshot = payload.get("market_snapshot") if isinstance(payload.get("market_snapshot"), dict) else {}
    market_breadth = payload.get("market_breadth") if isinstance(payload.get("market_breadth"), dict) else {}
    workflow_package = payload.get("workflow_package") if isinstance(payload.get("workflow_package"), dict) else {}
    sector_theme = payload.get("sector_theme_data")
    sector_status = _sector_status(sector_theme)
    sector_dict = sector_theme if isinstance(sector_theme, dict) else {}

    market_row_count = int(market_snapshot.get("row_count") or 0)
    blocking_reasons: list[str] = []
    warnings: list[str] = []

    if source_schema_version != EXPECTED_SOURCE_SCHEMA:
        blocking_reasons.append("source_schema_unexpected")
    if not trade_date:
        blocking_reasons.append("trade_date_missing")
    if market_row_count < min_market_row_count:
        blocking_reasons.append("market_row_count_below_minimum")
    if require_sector_theme:
        if not isinstance(sector_theme, dict):
            blocking_reasons.append("sector_theme_data_missing")
        elif sector_status != "ok":
            blocking_reasons.append("sector_theme_data_not_ok")
        elif sector_dict.get("trade_date") and trade_date and sector_dict.get("trade_date") != trade_date:
            blocking_reasons.append("sector_theme_trade_date_mismatch")
        elif _list_len(sector_dict.get("industry_boards")) == 0 or _list_len(sector_dict.get("concept_boards")) == 0:
            blocking_reasons.append("sector_theme_board_catalog_empty")
    elif not isinstance(sector_theme, dict):
        warnings.append("sector_theme_data_missing")
    elif sector_status != "ok":
        warnings.append("sector_theme_data_not_ok")

    if _list_len(payload.get("top_gainers")) == 0:
        warnings.append("top_gainers_empty")
    if _list_len(payload.get("top_losers")) == 0:
        warnings.append("top_losers_empty")
    if _list_len(payload.get("turnover_leaders")) == 0:
        warnings.append("turnover_leaders_empty")

    status = "blocked" if blocking_reasons else "watch" if warnings else "ok"
    snapshot: dict[str, Any] = {
        "schema_version": "quant-model2-data-input-v1",
        "source_schema_version": source_schema_version,
        "trade_date": trade_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "research_only": True,
        "broker_action": "none",
        "status": status,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "input_json_path": str(input_json),
        "source_market_snapshot_path": market_snapshot.get("path"),
        "workflow_excel_path": workflow_package.get("excel_path"),
        "workflow_html_path": workflow_package.get("html_path"),
        "market_row_count": market_row_count,
        "min_market_row_count": min_market_row_count,
        "advancing": market_breadth.get("advancing"),
        "declining": market_breadth.get("declining"),
        "flat": market_breadth.get("flat"),
        "advance_ratio_pct": market_breadth.get("advance_ratio_pct"),
        "top_gainers_count": _list_len(payload.get("top_gainers")),
        "top_losers_count": _list_len(payload.get("top_losers")),
        "turnover_leaders_count": _list_len(payload.get("turnover_leaders")),
        "top_gainers_preview": _preview_rows(payload.get("top_gainers")),
        "top_losers_preview": _preview_rows(payload.get("top_losers")),
        "turnover_leaders_preview": _preview_rows(payload.get("turnover_leaders")),
        "listing_segment_count": _list_len(payload.get("listing_segments")),
        "listing_segments": payload.get("listing_segments") if isinstance(payload.get("listing_segments"), list) else [],
        "sector_theme_status": sector_status,
        "sector_schema_version": sector_dict.get("schema_version"),
        "industry_board_count": _list_len(sector_dict.get("industry_boards")),
        "concept_board_count": _list_len(sector_dict.get("concept_boards")),
        "industry_fund_flow_count": _list_len(sector_dict.get("industry_fund_flow")),
        "concept_fund_flow_count": _list_len(sector_dict.get("concept_fund_flow")),
        "usage_note": "Research input only. No broker connection, order placement, or investment advice.",
    }
    return snapshot


def _write_report(snapshot: dict[str, Any], path: Path) -> None:
    blocking = snapshot.get("blocking_reasons") or []
    warnings = snapshot.get("warnings") or []
    lines = [
        "# Model2 Data Input",
        "",
        f"- Status: {snapshot.get('status')}",
        f"- Trade date: {snapshot.get('trade_date')}",
        f"- Research only: {str(snapshot.get('research_only')).lower()}",
        f"- Broker action: {snapshot.get('broker_action')}",
        f"- Market rows: {snapshot.get('market_row_count')}",
        f"- Advance ratio pct: {snapshot.get('advance_ratio_pct')}",
        f"- Sector/theme status: {snapshot.get('sector_theme_status')}",
        f"- Industry boards: {snapshot.get('industry_board_count')}",
        f"- Concept boards: {snapshot.get('concept_board_count')}",
        f"- Industry fund-flow rows: {snapshot.get('industry_fund_flow_count')}",
        f"- Concept fund-flow rows: {snapshot.get('concept_fund_flow_count')}",
        f"- Feature pack status: {snapshot.get('feature_pack_status')}",
        f"- Feature market regime: {snapshot.get('feature_market_regime')}",
        f"- Feature pack: {snapshot.get('feature_pack_path')}",
        f"- Source input: {snapshot.get('input_json_path')}",
        f"- Source market snapshot: {snapshot.get('source_market_snapshot_path')}",
        f"- Workflow Excel: {snapshot.get('workflow_excel_path')}",
        f"- Workflow HTML: {snapshot.get('workflow_html_path')}",
        "",
        "## Blocking Reasons",
    ]
    lines.extend([f"- {item}" for item in blocking] or ["- None"])
    lines.append("")
    lines.append("## Warnings")
    lines.extend([f"- {item}" for item in warnings] or ["- None"])
    lines.append("")
    lines.append("## Feature Pack")
    lines.append(f"- Status: {snapshot.get('feature_pack_status')}")
    lines.append(f"- Market regime: {snapshot.get('feature_market_regime')}")
    lines.append(f"- Top industry flow: {snapshot.get('feature_top_industry_flow_name')}")
    lines.append(f"- Top concept flow: {snapshot.get('feature_top_concept_flow_name')}")
    lines.append(f"- JSON: {snapshot.get('feature_pack_path')}")
    lines.append(f"- Report: {snapshot.get('feature_report_path')}")
    lines.append("")
    lines.append("## Usage")
    lines.append("- Use this file as a research-only upstream data readiness summary for model2.")
    lines.append("- Do not use it as a broker instruction or investment recommendation.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_model2_data_input(
    *,
    input_json: Path | str = DEFAULT_MODEL2_INPUT_JSON,
    output_dir: Path | str = DEFAULT_MODEL2_OUTPUT_DIR,
    min_market_row_count: int = DEFAULT_MIN_MARKET_ROW_COUNT,
    require_sector_theme: bool = True,
) -> Model2DataInputResult:
    input_path = Path(input_json)
    output_path = Path(output_dir)
    payload = _read_json(input_path)
    snapshot = _build_snapshot(
        payload,
        input_json=input_path,
        min_market_row_count=min_market_row_count,
        require_sector_theme=require_sector_theme,
    )

    output_path.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_path / "model2_data_input_snapshot.json"
    report_path = output_path / "model2_data_input_summary.md"
    feature_pack_path = output_path / "model2_feature_pack.json"
    feature_report_path = output_path / "model2_feature_pack.md"
    feature_pack = _build_feature_pack(payload, snapshot)
    feature_pack["snapshot_path"] = str(snapshot_path)
    feature_pack["summary_report_path"] = str(report_path)
    feature_pack["feature_report_path"] = str(feature_report_path)
    top_industry = feature_pack.get("top_industry_fund_flow") if isinstance(feature_pack.get("top_industry_fund_flow"), dict) else {}
    top_concept = feature_pack.get("top_concept_fund_flow") if isinstance(feature_pack.get("top_concept_fund_flow"), dict) else {}
    snapshot.update(
        {
            "feature_pack_status": feature_pack.get("status"),
            "feature_market_regime": feature_pack.get("market_regime"),
            "feature_top_industry_flow_name": top_industry.get("name"),
            "feature_top_concept_flow_name": top_concept.get("name"),
            "feature_flags": feature_pack.get("feature_flags"),
            "feature_pack_path": str(feature_pack_path),
            "feature_report_path": str(feature_report_path),
        }
    )
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=True, indent=2), encoding="utf-8")
    feature_pack_path.write_text(json.dumps(feature_pack, ensure_ascii=True, indent=2), encoding="utf-8")
    _write_report(snapshot, report_path)
    _write_feature_report(feature_pack, feature_report_path)

    return Model2DataInputResult(
        output_dir=output_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        source_payload=payload,
    )


def run_model2_input_export(
    *,
    daily_data_dir: Path | str = DEFAULT_DAILY_MARKET_DATA_DIR,
    ingest_project_dir: Path | str = DEFAULT_EXCHANGE_INGEST_DIR,
    output_dir: Path | str = DEFAULT_MODEL2_EXPORT_DIR,
    trade_date: str | None = None,
    sector_theme_json: Path | str | None = None,
    market: str = "all",
) -> Model2InputExportResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result = load_market_snapshot_rows(
        trade_date=trade_date,
        market=market,
        daily_data_dir=Path(daily_data_dir),
        ingest_project_dir=Path(ingest_project_dir),
        require_success=True,
    )
    resolved_trade_date = result.trade_date or trade_date
    if not resolved_trade_date:
        raise RuntimeError("Unable to resolve model2 export trade date.")
    default_sector_path = output_path / "latest_sector_theme_data.json"
    sector_path = Path(sector_theme_json) if sector_theme_json is not None else default_sector_path
    sector_payload, sector_status, sector_trade_date = _sector_payload_for_date(sector_path, resolved_trade_date)
    payload = _build_export_payload(
        rows=result.rows,
        trade_date=resolved_trade_date,
        source_path=result.source_path,
        sector_payload=sector_payload,
    )
    dated_payload_path = output_path / f"{resolved_trade_date}_model2_input.json"
    latest_payload_path = output_path / "latest_model2_input.json"
    dated_with_sector_path = output_path / f"{resolved_trade_date}_model2_input_with_sector.json"
    latest_with_sector_path = output_path / "latest_model2_input_with_sector.json"
    snapshot_path = output_path / "latest_model2_input_export_snapshot.json"
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    dated_payload_path.write_text(payload_text, encoding="utf-8")
    latest_payload_path.write_text(payload_text, encoding="utf-8")
    dated_with_sector_path.write_text(payload_text, encoding="utf-8")
    latest_with_sector_path.write_text(payload_text, encoding="utf-8")
    snapshot = {
        "schema_version": "quant-model2-input-export-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trade_date": resolved_trade_date,
        "source_kind": result.source_kind,
        "source_path": str(result.source_path) if result.source_path is not None else None,
        "row_count": int(len(result.rows)),
        "output_dir": str(output_path),
        "payload_path": str(dated_payload_path),
        "latest_payload_path": str(latest_payload_path),
        "latest_with_sector_path": str(latest_with_sector_path),
        "sector_theme_path": str(sector_path),
        "sector_theme_status": sector_status,
        "sector_theme_trade_date": sector_trade_date,
        "research_only": True,
        "broker_action": "none",
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return Model2InputExportResult(
        output_dir=output_path,
        payload_path=dated_payload_path,
        latest_payload_path=latest_payload_path,
        latest_with_sector_path=latest_with_sector_path,
        snapshot_path=snapshot_path,
        payload=payload,
        snapshot=snapshot,
    )
