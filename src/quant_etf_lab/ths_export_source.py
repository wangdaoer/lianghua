"""Tonghuashun HS A-share export reader and normalizer."""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd


THS_EXPORT_CUTOFF_DATE = date(2026, 6, 22)
DEFAULT_THS_EXPORT_ROOT = Path("D:/codex/daily-market-data/ths_exports")
THS_SOURCE_KIND = "ths_hs_a_share_export"


@dataclass(frozen=True)
class THSExportNormalizeResult:
    trade_date: str
    raw_path: Path
    normalized_path: Path
    manifest_path: Path
    row_count: int
    schema_hash: str
    status: str


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "security_code": (
        "\u4ee3\u7801",
        "\u8bc1\u5238\u4ee3\u7801",
        "a\u4ea4\u6613\u4ee3\u7801",
        "code",
        "symbol",
        "security_code",
    ),
    "security_name": (
        "\u540d\u79f0",
        "\u8bc1\u5238\u540d\u79f0",
        "name",
        "security_name",
    ),
    "open_price": ("\u5f00\u76d8", "open", "open_price"),
    "high_price": ("\u6700\u9ad8", "high", "high_price"),
    "low_price": ("\u6700\u4f4e", "low", "low_price"),
    "close_price": ("\u73b0\u4ef7", "close", "price", "last", "close_price", "last_price"),
    "prev_close": ("\u6628\u6536", "prev_close", "yesterday_close"),
    "last_price": ("\u73b0\u4ef7", "\u6700\u65b0", "last", "price", "last_price"),
    "change_amount": ("\u6da8\u5e45", "\u66f4\u591f\u91cf", "change", "change_amount"),
    "change_ratio": ("\u6da8\u5e45\u767e\u5206\u6bd4", "\u6da8\u5e45", "pct_chg", "change_ratio"),
    "volume": (
        "\u603b\u624b",
        "\u6218\u529b\u51cf\u91cf",
        "\u6218\u529b\u51cf\u5ea6",
        "\u6210\u4ea4\u91cf",
        "volume",
    ),
    "turnover": (
        "\u603b\u91d1\u989d",
        "\u6210\u4ea4\u989d",
        "\u6362\u624b\u989d",
        "amount",
        "turnover",
    ),
    "turnover_rate": ("\u6362\u624b", "\u6362\u624b\u7387", "turnover", "turnover_rate"),
    "market_cap": ("\u603b\u5e02\u503c", "\u7ecf\u6d4e\u5b57\u8282", "market_cap", "mkt_cap"),
    "float_market_cap": ("\u6d6e\u52a8\u5e02\u503c", "float_market_cap"),
}


def _resolve_export_root(export_root: str | Path | None = None) -> Path:
    root = Path(export_root) if export_root is not None else DEFAULT_THS_EXPORT_ROOT
    if root.name.lower() == "normalized":
        return root.parent
    return root


def normalize_trade_date(value: Any) -> str:
    if value in (None, ""):
        raise ValueError("trade_date is required")
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        parsed = datetime.strptime(text, "%Y%m%d").date()
    else:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Invalid trade_date: {value}")
        parsed = parsed.date()
    return parsed.strftime("%Y-%m-%d")


def parse_trade_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    try:
        return datetime.strptime(normalize_trade_date(value), "%Y-%m-%d").date()
    except ValueError:
        return None


def ths_export_paths(
    trade_date: str | date,
    export_root: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    resolved_trade_date = trade_date.strftime("%Y-%m-%d") if isinstance(trade_date, date) else normalize_trade_date(trade_date)
    root = _resolve_export_root(export_root)
    raw_path = root / "raw" / f"ths_hs_a_share_{resolved_trade_date}.xlsx"
    normalized_path = root / "normalized" / f"ths_hs_a_share_{resolved_trade_date}.csv"
    manifest_path = root / "manifests" / f"ths_hs_a_share_{resolved_trade_date}.json"
    # Keep target folders available for direct file writes in ingestion + test fixtures.
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    return raw_path, normalized_path, manifest_path


def _clean_column_name(value: Any) -> str:
    return re.sub(r"\s+", "", str(value).strip())


def _clean_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    copied = frame.copy()
    copied.columns = [_clean_column_name(column) for column in copied.columns]
    return copied


def _looks_like_excel_workbook(path: Path) -> bool:
    try:
        head = path.read_bytes()[:8]
    except OSError:
        return False
    return head.startswith(b"PK") or head.startswith(b"\xd0\xcf\x11\xe0")


def _read_delimited_text_export(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding, sep=None, engine="python")
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise last_error or ValueError(f"Cannot read text export: {path}")


def _read_raw_export(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"} and _looks_like_excel_workbook(path):
        try:
            frame = pd.read_excel(path, dtype=str)
        except (ImportError, OSError, ValueError, zipfile.BadZipFile):
            frame = _read_delimited_text_export(path)
    elif suffix in {".xlsx", ".xls", ".csv", ".txt"}:
        frame = _read_delimited_text_export(path)
    else:
        raise ValueError(f"Unsupported THS export file type: {path.suffix}")
    frame = _clean_frame_columns(frame)
    return frame.dropna(how="all")


def _find_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    clean_aliases = {_clean_column_name(alias).lower() for alias in aliases}
    for column in frame.columns:
        if column.lower() in clean_aliases:
            return column
    for column in frame.columns:
        lowered = column.lower()
        if any(alias in lowered for alias in clean_aliases):
            return column
    return None


def _series_for_alias(frame: pd.DataFrame, key: str) -> tuple[pd.Series, str | None]:
    column = _find_column(frame, COLUMN_ALIASES[key])
    if column is None:
        return pd.Series([None] * len(frame), index=frame.index), None
    return frame[column], column


def _normalize_security_code(value: Any) -> tuple[str, str, str, str]:
    raw = "" if value is None else str(value).strip()
    lowered = raw.lower()
    match = re.search(r"(\d{6})", lowered)
    if match is None:
        return raw, "", "", ""
    code = match.group(1)
    prefix = ""
    if lowered.startswith("sh") or lowered.endswith((".sh", ".ss")):
        prefix = "sh"
    elif lowered.startswith("sz") or lowered.endswith(".sz"):
        prefix = "sz"
    elif lowered.startswith("bj") or lowered.endswith(".bj"):
        prefix = "bj"
    elif code.startswith("6"):
        prefix = "sh"
    elif code.startswith(("0", "3")):
        prefix = "sz"
    elif code.startswith(("4", "8", "9")):
        prefix = "bj"
    market = {"sh": "sse", "sz": "szse", "bj": "bse"}.get(prefix, "")
    return raw, code, market, f"{prefix}{code}" if prefix else code


def _numeric_multiplier_from_text(text: str) -> float:
    if "\u4ebf" in text:
        return 100_000_000.0
    if "\u4e07" in text:
        return 10_000.0
    return 1.0


def _parse_number(value: Any, column_name: str | None = None) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if text in {"", "--", "-", "nan", "None", "NaN"}:
        return None
    multiplier = _numeric_multiplier_from_text(text)
    if multiplier == 1.0:
        multiplier = _numeric_multiplier_from_text(column_name or "")
    cleaned = (
        text.replace(",", "")
        .replace("%", "")
        .replace(",", "")
        .replace("%", "")
        .replace("\u5143", "")
        .replace("\u4efb", "")
        .replace("\u5143\u5ea6", "")
        .replace("\u4e07", "")
        .replace("\u4ebf", "")
        .strip()
    )
    if cleaned in {"", "--", "-"}:
        return None
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


def _numeric_series(frame: pd.DataFrame, key: str) -> tuple[pd.Series, str | None]:
    series, column = _series_for_alias(frame, key)
    return series.map(lambda value: _parse_number(value, column)), column


def _text_series(frame: pd.DataFrame, key: str) -> tuple[pd.Series, str | None]:
    series, column = _series_for_alias(frame, key)
    return series.map(lambda value: "" if value is None or pd.isna(value) else str(value).strip()), column


def _schema_hash(columns: list[str]) -> str:
    payload = json.dumps(columns, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _require_manifest_status(manifest: dict[str, Any], trade_date: str, require_status_ok: bool) -> None:
    status = str(manifest.get("status", "")).lower()
    if not status:
        if require_status_ok:
            raise ValueError(f"Missing THS HS A-share manifest status for {trade_date}: {manifest}")
        return
    if status != "ok":
        raise ValueError(f"THS HS A-share manifest status is not ok for {trade_date}: {status}")


def write_ths_export_status(
    trade_date: str | date,
    status: str,
    message: str,
    export_root: str | Path | None = None,
    raw_path: str | Path | None = None,
    normalized_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    resolved_trade_date = trade_date.strftime("%Y-%m-%d") if isinstance(trade_date, date) else normalize_trade_date(trade_date)
    default_raw, default_normalized, manifest_path = ths_export_paths(resolved_trade_date, export_root)
    payload: dict[str, Any] = {
        "status": status,
        "message": message,
        "trade_date": resolved_trade_date,
        "source_kind": THS_SOURCE_KIND,
        "source_file": str(raw_path or default_raw),
        "normalized_file": str(normalized_path or default_normalized),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    _write_manifest(manifest_path, payload)
    return manifest_path


def normalize_ths_export(
    raw_path: str | Path,
    trade_date: str | date,
    export_root: str | Path | None = None,
    min_row_count: int = 1,
) -> THSExportNormalizeResult:
    resolved_trade_date = trade_date.strftime("%Y-%m-%d") if isinstance(trade_date, date) else normalize_trade_date(trade_date)
    raw = Path(raw_path)
    if not raw.exists():
        raise FileNotFoundError(f"THS raw export not found: {raw}")
    default_raw, normalized_path, manifest_path = ths_export_paths(resolved_trade_date, export_root)
    frame = _read_raw_export(raw)
    code_series, code_column = _text_series(frame, "security_code")
    name_series, name_column = _text_series(frame, "security_name")

    normalized_codes = code_series.map(_normalize_security_code)
    out = pd.DataFrame(
        {
            "market": normalized_codes.map(lambda item: item[2]),
            "trade_date": resolved_trade_date,
            "security_code": normalized_codes.map(lambda item: item[1]),
            "security_name": name_series,
            "prefixed_security_code": normalized_codes.map(lambda item: item[3]),
            "raw_security_code": normalized_codes.map(lambda item: item[0]),
        }
    )
    numeric_columns: dict[str, str | None] = {}
    for key in (
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
        "turnover_rate",
        "market_cap",
        "float_market_cap",
    ):
        out[key], numeric_columns[key] = _numeric_series(frame, key)

    out["open"] = out["open_price"]
    out["high"] = out["high_price"]
    out["low"] = out["low_price"]
    out["close"] = out["close_price"]
    out["amount"] = out["turnover"]
    out["source"] = THS_SOURCE_KIND
    out = out[(out["security_code"] != "") & (out["market"] != "")].copy()
    out = out.sort_values(["market", "security_code"], kind="stable").reset_index(drop=True)

    row_count = int(len(out))
    schema_hash = _schema_hash(list(out.columns))
    status = "ok" if row_count >= min_row_count else "failed_row_count_below_minimum"
    field_coverage = {}
    for key in ("volume", "turnover"):
        values = pd.to_numeric(out[key], errors="coerce")
        non_null_count = int(values.notna().sum())
        positive_count = int((values > 0).sum())
        field_coverage[key] = {
            "non_null_count": non_null_count,
            "positive_count": positive_count,
            "non_null_ratio": non_null_count / row_count if row_count else 0.0,
            "positive_ratio": positive_count / row_count if row_count else 0.0,
        }
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(normalized_path, index=False, encoding="utf-8-sig")
    manifest_payload = {
        "status": status,
        "trade_date": resolved_trade_date,
        "source_kind": THS_SOURCE_KIND,
        "source_file": str(raw),
        "preferred_raw_file": str(default_raw),
        "accepted_raw_files": [str(default_raw.with_suffix(suffix)) for suffix in (".xlsx", ".xls", ".csv", ".txt")],
        "normalized_file": str(normalized_path),
        "row_count": row_count,
        "min_row_count": int(min_row_count),
        "schema_hash": schema_hash,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_columns": list(frame.columns),
        "mapped_columns": {
            "security_code": code_column,
            "security_name": name_column,
            **numeric_columns,
        },
        "field_coverage": field_coverage,
        "unit_notes": {
            "percent_fields": ["change_ratio", "turnover_rate"],
            "numeric_unit_conversion": "Chinese \u4ebf/\u4e07 unit suffixes are converted to base units when present.",
        },
    }
    _write_manifest(manifest_path, manifest_payload)
    if status != "ok":
        raise ValueError(f"THS export row_count={row_count} below min_row_count={min_row_count}: {raw}")
    return THSExportNormalizeResult(
        trade_date=resolved_trade_date,
        raw_path=raw,
        normalized_path=normalized_path,
        manifest_path=manifest_path,
        row_count=row_count,
        schema_hash=schema_hash,
        status=status,
    )


def read_ths_export_rows(
    trade_date: str | date,
    export_root: str | Path | None = None,
    market: str = "all",
    min_row_count: int = 1,
    require_status_ok: bool = False,
) -> tuple[list[dict[str, Any]], Path]:
    if market not in {"all", "sse", "szse", "bse"}:
        raise ValueError("market must be one of: all, sse, szse, bse")
    if min_row_count < 1:
        raise ValueError("min_row_count must be at least 1")
    resolved_trade_date = trade_date.strftime("%Y-%m-%d") if isinstance(trade_date, date) else normalize_trade_date(trade_date)
    _, normalized_path, manifest_path = ths_export_paths(resolved_trade_date, export_root)
    if not normalized_path.exists():
        raise FileNotFoundError(
            f"Missing THS HS A-share export for {resolved_trade_date}: {normalized_path}"
        )
    manifest = _read_manifest(manifest_path)
    _require_manifest_status(manifest, resolved_trade_date, require_status_ok=require_status_ok)
    frame = pd.read_csv(normalized_path, dtype={"security_code": str, "trade_date": str, "market": str})
    if frame.empty:
        raise ValueError(f"Empty THS HS A-share export file: {normalized_path}")
    if len(frame) < min_row_count:
        raise ValueError(
            f"THS HS A-share export row_count={len(frame)} below required minimum={min_row_count}: {normalized_path}"
        )
    if {"security_code", "trade_date", "market"}.difference(frame.columns):
        raise ValueError(f"THS HS A-share export missing required columns for {resolved_trade_date}: {normalized_path}")
    if market != "all":
        frame = frame[frame["market"] == market]
    return frame.to_dict(orient="records"), normalized_path


def latest_ths_export_trade_date(export_root: str | Path | None = None) -> str | None:
    root = _resolve_export_root(export_root)
    normalized_dir = root / "normalized"
    if not normalized_dir.exists():
        return None
    dates: list[date] = []
    for path in normalized_dir.glob("ths_hs_a_share_*.csv"):
        text = path.stem.removeprefix("ths_hs_a_share_")
        parsed = parse_trade_date(text)
        if parsed is not None:
            dates.append(parsed)
    if not dates:
        return None
    return max(dates).strftime("%Y-%m-%d")
