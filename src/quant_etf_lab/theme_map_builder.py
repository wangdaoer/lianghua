"""Build research-only stock-to-theme maps from local sources.

The builder prefers explicit local theme files, then fills missing snapshot
symbols with board membership. It never changes allocations or broker state.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .market_data_source import DEFAULT_DAILY_MARKET_DATA_DIR, DEFAULT_EXCHANGE_INGEST_DIR, load_market_snapshot_rows
from .theme_state import THEME_SPLIT_PATTERN


DEFAULT_OUTPUT_DIR = Path("outputs/research/theme_map_latest")
DEFAULT_THEME_MAP_NAME = "theme_map.csv"
CODE_PATTERN = re.compile(r"(?<!\d)(?:sh|sz|bj)?([03689]\d{5}|[48]\d{5}|920\d{3})(?!\d)", re.IGNORECASE)
DISCOVERY_TOKENS = (
    "theme",
    "concept",
    "industry",
    "sector",
    "block",
    "stockblock",
    "题材",
    "概念",
    "行业",
    "板块",
)
CSV_CODE_COLUMNS = ("code", "security_code", "symbol", "股票代码", "证券代码", "代码")
CSV_NAME_COLUMNS = ("name", "security_name", "股票名称", "证券名称", "名称", "简称")
CSV_GROUP_COLUMNS = (
    "theme",
    "theme_group",
    "concept",
    "industry",
    "board",
    "sector",
    "block",
    "题材",
    "概念",
    "行业",
    "板块",
    "分类",
)
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk", "utf-16")


@dataclass(frozen=True)
class ThemeMapBuildResult:
    output_dir: Path
    theme_map_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    theme_map: pd.DataFrame


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = default if path is None else Path(path)
    return raw if raw.is_absolute() else project_root / raw


def _clean_code(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip()
    match = CODE_PATTERN.search(text)
    if match:
        return match.group(1).zfill(6)
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    return digits.zfill(6) if digits else ""


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "nan", "none", "0"}:
        return ""
    return text


def _board_from_code(code: str) -> str:
    clean = _clean_code(code)
    if clean.startswith(("300", "301")):
        return "chinext"
    if clean.startswith(("688", "689")):
        return "star"
    if clean.startswith(("4", "8", "920")):
        return "bse"
    if clean.startswith(("000", "001", "002", "003", "600", "601", "603", "605")):
        return "main"
    return "unknown"


def _column_lookup(columns: Iterable[str]) -> dict[str, str]:
    return {str(column).strip().lower(): str(column) for column in columns}


def _first_matching_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = _column_lookup(frame.columns)
    for candidate in candidates:
        actual = lookup.get(candidate.lower())
        if actual is not None:
            return actual
    return None


def _split_groups(value: Any) -> list[str]:
    raw = _clean_text(value)
    if not raw:
        return []
    groups = []
    for item in THEME_SPLIT_PATTERN.split(raw):
        cleaned = _clean_text(item)
        if cleaned:
            groups.append(cleaned)
    return groups


def _read_csv_with_fallback(path: Path) -> pd.DataFrame:
    last_error: UnicodeDecodeError | pd.errors.ParserError | OSError | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return pd.read_csv(path, dtype=str, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
        except pd.errors.ParserError as error:
            last_error = error
            break
        except OSError as error:
            last_error = error
            break
    if last_error is None:
        raise ValueError(f"Cannot read CSV source: {path}")
    raise ValueError(f"Cannot read CSV source {path}: {last_error}") from last_error


def _read_text_with_fallback(path: Path) -> str:
    last_error: UnicodeDecodeError | OSError | None = None
    for encoding in TEXT_ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
        except OSError as error:
            last_error = error
            break
    if last_error is None:
        raise ValueError(f"Cannot read text source: {path}")
    raise ValueError(f"Cannot read text source {path}: {last_error}") from last_error


def _parse_csv_source(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _read_csv_with_fallback(path)
    code_column = _first_matching_column(frame, CSV_CODE_COLUMNS)
    group_column = _first_matching_column(frame, CSV_GROUP_COLUMNS)
    name_column = _first_matching_column(frame, CSV_NAME_COLUMNS)
    if code_column is None or group_column is None:
        return (
            pd.DataFrame(columns=["code", "name", "theme", "source_kind", "source_path"]),
            {
                "path": str(path),
                "source_kind": "csv",
                "status": "skipped",
                "reason": "missing code or group column",
                "columns": list(frame.columns),
            },
        )
    rows: list[dict[str, str]] = []
    for record in frame.to_dict(orient="records"):
        code = _clean_code(record.get(code_column))
        if not code:
            continue
        name = _clean_text(record.get(name_column)) if name_column else ""
        for group in _split_groups(record.get(group_column)):
            rows.append(
                {
                    "code": code,
                    "name": name,
                    "theme": group,
                    "source_kind": f"csv:{group_column}",
                    "source_path": str(path),
                }
            )
    parsed = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["code", "name", "theme", "source_kind", "source_path"])
    return (
        parsed,
        {
            "path": str(path),
            "source_kind": "csv",
            "status": "ok",
            "row_count": int(len(parsed)),
            "code_count": int(parsed["code"].nunique()) if not parsed.empty else 0,
            "group_count": int(parsed["theme"].nunique()) if not parsed.empty else 0,
            "code_column": code_column,
            "group_column": group_column,
        },
    )


def _parse_ini_source(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    text = _read_text_with_fallback(path)
    rows: list[dict[str, str]] = []
    current_group = path.stem
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and "]" in line:
            current_group = _clean_text(line[1 : line.index("]")]) or path.stem
            continue
        codes = [_clean_code(match.group(0)) for match in CODE_PATTERN.finditer(line)]
        for code in codes:
            if code:
                rows.append(
                    {
                        "code": code,
                        "name": "",
                        "theme": current_group,
                        "source_kind": "ini:block",
                        "source_path": str(path),
                    }
                )
    parsed = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(columns=["code", "name", "theme", "source_kind", "source_path"])
    return (
        parsed,
        {
            "path": str(path),
            "source_kind": "ini",
            "status": "ok",
            "row_count": int(len(parsed)),
            "code_count": int(parsed["code"].nunique()) if not parsed.empty else 0,
            "group_count": int(parsed["theme"].nunique()) if not parsed.empty else 0,
        },
    )


def parse_theme_source(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved = Path(path)
    if not resolved.exists():
        return (
            pd.DataFrame(columns=["code", "name", "theme", "source_kind", "source_path"]),
            {"path": str(resolved), "status": "missing", "source_kind": "unknown", "reason": "file not found"},
        )
    suffix = resolved.suffix.lower()
    try:
        if suffix == ".csv":
            return _parse_csv_source(resolved)
        if suffix in {".ini", ".blk", ".txt"}:
            return _parse_ini_source(resolved)
    except (OSError, UnicodeDecodeError, ValueError, pd.errors.ParserError) as error:
        return (
            pd.DataFrame(columns=["code", "name", "theme", "source_kind", "source_path"]),
            {"path": str(resolved), "status": "error", "source_kind": suffix.lstrip("."), "reason": str(error)},
        )
    return (
        pd.DataFrame(columns=["code", "name", "theme", "source_kind", "source_path"]),
        {"path": str(resolved), "status": "skipped", "source_kind": suffix.lstrip("."), "reason": "unsupported suffix"},
    )


def discover_theme_sources(scan_roots: Iterable[str | Path], max_files: int = 2000) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root_value in scan_roots:
        root = Path(root_value)
        if not root.exists():
            continue
        if root.is_file():
            paths = [root]
        else:
            paths = []
            for suffix in ("*.csv", "*.ini", "*.blk", "*.txt"):
                paths.extend(root.rglob(suffix))
        for path in sorted(paths):
            if len(candidates) >= max_files:
                return candidates
            if not path.is_file():
                continue
            name = path.name.lower()
            if not any(token.lower() in name for token in DISCOVERY_TOKENS):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append(path)
    return candidates


def _snapshot_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for row in rows:
        code = _clean_code(row.get("security_code") or row.get("code") or row.get("symbol"))
        if not code:
            continue
        records.append(
            {
                "code": code,
                "snapshot_name": _clean_text(row.get("security_name") or row.get("name")),
                "market": _clean_text(row.get("market")),
                "trade_date": _clean_text(row.get("trade_date")),
            }
        )
    if not records:
        return pd.DataFrame(columns=["code", "snapshot_name", "market", "trade_date"])
    return pd.DataFrame(records).drop_duplicates(subset=["code"], keep="last")


def _apply_snapshot_names(theme_map: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    if theme_map.empty:
        return theme_map
    merged = theme_map.merge(snapshot[["code", "snapshot_name", "market", "trade_date"]], on="code", how="left")
    merged["name"] = merged["name"].where(merged["name"].astype(str).str.strip() != "", merged["snapshot_name"])
    return merged.drop(columns=["snapshot_name"])


def _board_fallback_rows(snapshot: pd.DataFrame, mapped_codes: set[str]) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame(columns=["code", "name", "theme", "source_kind", "source_path", "market", "trade_date"])
    rows = []
    for record in snapshot.to_dict(orient="records"):
        code = str(record.get("code") or "")
        if not code or code in mapped_codes:
            continue
        rows.append(
            {
                "code": code,
                "name": record.get("snapshot_name", ""),
                "theme": _board_from_code(code),
                "source_kind": "board_fallback",
                "source_path": "market_snapshot",
                "market": record.get("market", ""),
                "trade_date": record.get("trade_date", ""),
            }
        )
    return pd.DataFrame(rows)


def _finalize_theme_map(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["code", "name", "theme", "source_kind", "source_path", "market", "trade_date"])
    output = frame.copy()
    for column in ["code", "name", "theme", "source_kind", "source_path", "market", "trade_date"]:
        if column not in output.columns:
            output[column] = ""
    output["code"] = output["code"].map(_clean_code)
    output["theme"] = output["theme"].map(_clean_text)
    output["name"] = output["name"].map(_clean_text)
    output = output.loc[(output["code"] != "") & (output["theme"] != "")]
    return (
        output[["code", "name", "theme", "source_kind", "source_path", "market", "trade_date"]]
        .drop_duplicates(subset=["code", "theme", "source_kind", "source_path"], keep="last")
        .sort_values(["code", "theme", "source_kind"])
        .reset_index(drop=True)
    )


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "No rows."
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].head(max_rows)
    lines = ["| " + " | ".join(subset.columns) + " |", "| " + " | ".join(["---"] * len(subset.columns)) + " |"]
    for row in subset.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _render_report(snapshot: dict[str, Any], theme_map: pd.DataFrame, source_audit: pd.DataFrame) -> str:
    counts = theme_map["source_kind"].value_counts().rename_axis("source_kind").reset_index(name="rows") if not theme_map.empty else pd.DataFrame()
    examples = theme_map.head(20).copy()
    return f"""# Theme Map Build Report

Generated at: `{snapshot.get("generated_at")}`

This report is research-only. It does not connect to brokers, place orders, or change live allocation.

## Summary

| Item | Value |
| --- | ---: |
| Status | `{snapshot.get("status")}` |
| Trade date | `{snapshot.get("trade_date")}` |
| Market snapshot source | `{snapshot.get("market_source_kind")}` |
| Snapshot rows | {snapshot.get("market_snapshot_rows")} |
| Theme map rows | {snapshot.get("theme_map_rows")} |
| Mapped codes | {snapshot.get("mapped_code_count")} |
| Groups | {snapshot.get("group_count")} |
| Explicit source rows | {snapshot.get("explicit_source_rows")} |
| Board fallback rows | {snapshot.get("board_fallback_rows")} |
| Research only | `{snapshot.get("research_only")}` |

## Source Audit

{_markdown_table(source_audit, ["path", "source_kind", "status", "row_count", "code_count", "group_count", "reason"], 20)}

## Rows By Source

{_markdown_table(counts, ["source_kind", "rows"], 20)}

## Sample Theme Map

{_markdown_table(examples, ["code", "name", "theme", "source_kind", "market", "trade_date"], 20)}

## Files

- Theme map CSV: `{snapshot.get("theme_map_path")}`
- Snapshot JSON: `{snapshot.get("snapshot_path")}`
- Report: `{snapshot.get("report_path")}`
"""


def run_theme_map_build(
    project_root: str | Path = Path("."),
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    output: str | Path | None = None,
    sources: Iterable[str | Path] | None = None,
    scan_roots: Iterable[str | Path] | None = None,
    include_default_scan_roots: bool = True,
    include_board_fallback: bool = True,
    max_scan_files: int = 2000,
    trade_date: str | None = None,
    daily_data_dir: str | Path | None = DEFAULT_DAILY_MARKET_DATA_DIR,
    ingest_project_dir: str | Path | None = DEFAULT_EXCHANGE_INGEST_DIR,
    require_success: bool = True,
) -> ThemeMapBuildResult:
    root = Path(project_root).resolve()
    resolved_output = _resolve(root, output_dir, DEFAULT_OUTPUT_DIR)
    theme_map_path = _resolve(root, output, resolved_output / DEFAULT_THEME_MAP_NAME) if output else resolved_output / DEFAULT_THEME_MAP_NAME

    market_result = load_market_snapshot_rows(
        trade_date=trade_date,
        market="all",
        daily_data_dir=daily_data_dir,
        ingest_project_dir=ingest_project_dir,
        require_success=require_success,
    )
    snapshot_symbols = _snapshot_frame(market_result.rows)

    explicit_sources = [Path(source) for source in sources or []]
    roots: list[Path] = [Path(root_value) for root_value in scan_roots or []]
    if include_default_scan_roots:
        roots.extend([root / "configs", root / "data", Path(daily_data_dir or DEFAULT_DAILY_MARKET_DATA_DIR) / "exports"])
    discovered_sources = discover_theme_sources(roots, max_files=max_scan_files)

    all_sources: list[Path] = []
    seen: set[Path] = set()
    for source in [*explicit_sources, *discovered_sources]:
        resolved = source if source.is_absolute() else root / source
        key = resolved.resolve() if resolved.exists() else resolved
        if key in seen:
            continue
        seen.add(key)
        all_sources.append(resolved)

    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for source in all_sources:
        parsed, audit = parse_theme_source(source)
        if not parsed.empty:
            frames.append(parsed)
        audit_rows.append(audit)

    explicit_map = _finalize_theme_map(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
    explicit_map = _apply_snapshot_names(explicit_map, snapshot_symbols)
    mapped_codes = set(explicit_map["code"].astype(str)) if not explicit_map.empty else set()
    fallback = _board_fallback_rows(snapshot_symbols, mapped_codes) if include_board_fallback else pd.DataFrame()
    theme_map = _finalize_theme_map(pd.concat([explicit_map, fallback], ignore_index=True) if not fallback.empty else explicit_map)
    if theme_map.empty:
        raise ValueError("No theme map rows were built from explicit sources or board fallback.")

    resolved_output.mkdir(parents=True, exist_ok=True)
    theme_map_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path = resolved_output / "theme_map_snapshot.json"
    report_path = resolved_output / "theme_map.md"
    source_audit = pd.DataFrame(audit_rows)

    theme_map.to_csv(theme_map_path, index=False, encoding="utf-8-sig")
    source_audit_path = resolved_output / "theme_source_audit.csv"
    source_audit.to_csv(source_audit_path, index=False, encoding="utf-8-sig")

    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok",
        "research_only": True,
        "broker_action": "none",
        "trade_date": market_result.trade_date,
        "market_source_kind": market_result.source_kind,
        "market_source_path": str(market_result.source_path) if market_result.source_path else None,
        "market_snapshot_rows": int(len(market_result.rows)),
        "market_snapshot_code_count": int(snapshot_symbols["code"].nunique()) if not snapshot_symbols.empty else 0,
        "fetch_status": getattr(market_result.fetch_status, "status", ""),
        "fetch_run_id": getattr(market_result.fetch_status, "run_id", ""),
        "source_count": int(len(all_sources)),
        "source_audit_path": str(source_audit_path),
        "explicit_source_rows": int(len(explicit_map)),
        "board_fallback_rows": int(len(fallback)),
        "theme_map_rows": int(len(theme_map)),
        "mapped_code_count": int(theme_map["code"].nunique()),
        "group_count": int(theme_map["theme"].nunique()),
        "include_board_fallback": bool(include_board_fallback),
        "daily_data_dir": str(daily_data_dir or DEFAULT_DAILY_MARKET_DATA_DIR),
        "ingest_project_dir": str(ingest_project_dir or DEFAULT_EXCHANGE_INGEST_DIR),
        "theme_map_path": str(theme_map_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "promotion_note": "Theme maps are candidate-generation inputs only; evaluate with theme-state before using them in allocation research.",
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, theme_map, source_audit), encoding="utf-8")
    return ThemeMapBuildResult(
        output_dir=resolved_output,
        theme_map_path=theme_map_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        theme_map=theme_map,
    )
