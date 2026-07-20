from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_etf_lab.market_data_source import (
    DEFAULT_DAILY_MARKET_DATA_DIR,
    DEFAULT_EXCHANGE_INGEST_DIR,
    load_market_snapshot_rows,
)


CODE_PATTERN = re.compile(r"(\d{6})$")
SNAPSHOT_SOURCE = "daily_market_snapshot"


class SnapshotQuote(NamedTuple):
    code: str
    name: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float


def _normalize_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _parse_after_close_time(value: str) -> tuple[int, int]:
    parts = str(value or "").split(":")
    if len(parts) != 2:
        raise ValueError("after_close_time must use HH:MM format.")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("after_close_time must use HH:MM format.")
    return hour, minute


def _intraday_write_block_reason(
    target_date: str,
    *,
    current_dt: datetime | None,
    source_updated_at: datetime | None,
    after_close_time: str,
    allow_intraday: bool,
    dry_run: bool,
) -> str | None:
    if allow_intraday or dry_run:
        return None
    now = current_dt or datetime.now()
    target = pd.to_datetime(target_date).date()
    hour, minute = _parse_after_close_time(after_close_time)
    cutoff = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target >= now.date() and now < cutoff:
        return "before_after_close_time"
    if target == now.date() and source_updated_at is not None:
        source_cutoff = datetime(target.year, target.month, target.day, hour, minute)
        if source_updated_at < source_cutoff:
            return "source_before_after_close"
    return None


def _normalize_code(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    match = CODE_PATTERN.search(text)
    if match:
        return match.group(1)
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    return digits.zfill(6)[-6:]


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _first_positive(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _to_float(row.get(key), default=0.0)
        if value > 0:
            return value
    return None


def _same_quote_values(existing: pd.Series, quote: SnapshotQuote) -> bool:
    pairs = {
        "open": quote.open,
        "high": quote.high,
        "low": quote.low,
        "close": quote.close,
        "volume": quote.volume,
        "amount": quote.amount,
    }
    for column, quote_value in pairs.items():
        if column not in existing:
            return False
        existing_value = _to_float(existing.get(column), default=math.nan)
        if math.isnan(existing_value) or abs(existing_value - quote_value) > 1e-9:
            return False
    return True


def snapshot_row_to_quote(row: dict[str, Any]) -> SnapshotQuote | None:
    code = _normalize_code(row.get("security_code") or row.get("code") or row.get("symbol"))
    trade_date = _normalize_date(row.get("trade_date") or row.get("date"))
    if not code or not trade_date:
        return None

    close = _first_positive(row, ("close_price", "close", "last_price"))
    if close is None:
        return None

    open_price = _first_positive(row, ("open_price", "open")) or close
    high = _first_positive(row, ("high_price", "high")) or max(open_price, close)
    low = _first_positive(row, ("low_price", "low")) or min(open_price, close)
    name = str(row.get("security_name") or row.get("name") or "").strip()
    return SnapshotQuote(
        code=code,
        name=name,
        date=trade_date,
        open=float(open_price),
        high=float(high),
        low=float(low),
        close=float(close),
        volume=_to_float(row.get("volume"), default=0.0),
        amount=_to_float(row.get("turnover", row.get("amount")), default=0.0),
    )


def append_snapshot_quote_to_csv(
    path: Path,
    quote: SnapshotQuote,
    target_date: str,
    dry_run: bool = False,
    replace_existing: bool = False,
) -> dict[str, Any]:
    target = _normalize_date(target_date)
    result: dict[str, Any] = {
        "path": str(path),
        "code": quote.code,
        "quote_date": quote.date,
        "target_date": target,
        "source": SNAPSHOT_SOURCE,
    }
    if target is None:
        return result | {"status": "invalid_target_date"}
    if quote.date != target:
        return result | {"status": "date_mismatch"}
    if not path.exists():
        return result | {"status": "missing_file"}

    try:
        frame = pd.read_csv(path, dtype={"code": str})
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        return result | {"status": "read_error", "error": str(exc)}
    if "date" not in frame.columns:
        return result | {"status": "missing_date_column"}

    dates = pd.to_datetime(frame["date"], errors="coerce")
    previous_latest = dates.max()
    previous_latest_date = previous_latest.strftime("%Y-%m-%d") if pd.notna(previous_latest) else None
    result["previous_latest_date"] = previous_latest_date
    target_mask = dates.dt.strftime("%Y-%m-%d") == target
    if target_mask.any():
        existing = frame.loc[target_mask].iloc[-1]
        if _same_quote_values(existing, quote) or not replace_existing:
            return result | {"status": "already_present"}
        if dry_run:
            return result | {"status": "would_replace", "latest_date": target}
        replacements = {
            "code": quote.code,
            "open": quote.open,
            "high": quote.high,
            "low": quote.low,
            "close": quote.close,
            "volume": quote.volume,
            "amount": quote.amount,
        }
        for column, value in replacements.items():
            if column in frame.columns:
                frame.loc[target_mask, column] = value
        try:
            frame.to_csv(path, index=False, encoding="utf-8")
        except OSError as exc:
            return result | {"status": "write_error", "error": str(exc)}
        return result | {"status": "replaced", "latest_date": target}
    is_backfill = previous_latest_date is not None and previous_latest_date > target
    if previous_latest_date is not None and not frame.empty and not is_backfill:
        latest_rows = frame.loc[dates == previous_latest]
        if not latest_rows.empty and _same_quote_values(latest_rows.iloc[-1], quote):
            return result | {"status": "stale_duplicate_quote", "latest_date": previous_latest_date}
    if dry_run:
        return result | {"status": "would_backfill" if is_backfill else "would_append", "latest_date": target}

    row: dict[str, Any] = {column: pd.NA for column in frame.columns}
    row.update(
        {
            "date": target,
            "code": quote.code,
            "name": quote.name,
            "open": quote.open,
            "high": quote.high,
            "low": quote.low,
            "close": quote.close,
            "volume": quote.volume,
            "amount": quote.amount,
        }
    )
    if "name" in frame.columns and not frame.empty:
        names = frame["name"].dropna().astype(str)
        if not names.empty and names.iloc[-1].strip():
            row["name"] = names.iloc[-1]

    updated = frame.copy()
    updated.loc[len(updated)] = [row.get(column, pd.NA) for column in updated.columns]
    updated["_date_sort"] = pd.to_datetime(updated["date"], errors="coerce")
    updated = (
        updated.dropna(subset=["_date_sort"])
        .sort_values("_date_sort")
        .drop_duplicates(subset=["_date_sort"], keep="last")
        .drop(columns=["_date_sort"])
        .reset_index(drop=True)
    )
    try:
        updated.to_csv(path, index=False, encoding="utf-8")
    except OSError as exc:
        return result | {"status": "write_error", "error": str(exc)}
    return result | {"status": "backfilled" if is_backfill else "appended", "latest_date": target}


def append_snapshot_quotes_to_csv(
    path: Path,
    quotes: list[SnapshotQuote],
    dry_run: bool = False,
    replace_existing: bool = False,
) -> list[dict[str, Any]]:
    """Append several dated quotes while reading and writing the cache only once."""
    ordered_quotes = sorted(
        {quote.date: quote for quote in quotes}.values(),
        key=lambda quote: quote.date,
    )
    if not ordered_quotes:
        return []

    base_results = [
        {
            "path": str(path),
            "code": quote.code,
            "quote_date": quote.date,
            "target_date": _normalize_date(quote.date),
            "source": SNAPSHOT_SOURCE,
        }
        for quote in ordered_quotes
    ]
    if not path.exists():
        return [result | {"status": "missing_file"} for result in base_results]

    try:
        frame = pd.read_csv(path, dtype={"code": str})
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        return [result | {"status": "read_error", "error": str(exc)} for result in base_results]
    if "date" not in frame.columns:
        return [result | {"status": "missing_date_column"} for result in base_results]

    updated = frame.copy()
    preserved_name = ""
    if "name" in updated.columns and not updated.empty:
        names = updated["name"].dropna().astype(str)
        if not names.empty and names.iloc[-1].strip():
            preserved_name = names.iloc[-1]

    results: list[dict[str, Any]] = []
    changed = False
    for result, quote in zip(base_results, ordered_quotes):
        target = result["target_date"]
        if target is None:
            results.append(result | {"status": "invalid_target_date"})
            continue

        dates = pd.to_datetime(updated["date"], errors="coerce")
        previous_latest = dates.max()
        previous_latest_date = (
            previous_latest.strftime("%Y-%m-%d") if pd.notna(previous_latest) else None
        )
        result["previous_latest_date"] = previous_latest_date
        target_mask = dates.dt.strftime("%Y-%m-%d") == target
        if target_mask.any():
            existing = updated.loc[target_mask].iloc[-1]
            if _same_quote_values(existing, quote) or not replace_existing:
                results.append(result | {"status": "already_present"})
                continue
            if dry_run:
                results.append(result | {"status": "would_replace", "latest_date": target})
                continue
            replacements = {
                "code": quote.code,
                "open": quote.open,
                "high": quote.high,
                "low": quote.low,
                "close": quote.close,
                "volume": quote.volume,
                "amount": quote.amount,
            }
            for column, value in replacements.items():
                if column in updated.columns:
                    updated.loc[target_mask, column] = value
            changed = True
            results.append(result | {"status": "replaced", "latest_date": target})
            continue

        is_backfill = previous_latest_date is not None and previous_latest_date > target
        if previous_latest_date is not None and not updated.empty and not is_backfill:
            latest_rows = updated.loc[dates == previous_latest]
            if not latest_rows.empty and _same_quote_values(latest_rows.iloc[-1], quote):
                results.append(
                    result
                    | {
                        "status": "stale_duplicate_quote",
                        "latest_date": previous_latest_date,
                    }
                )
                continue

        row: dict[str, Any] = {column: pd.NA for column in updated.columns}
        row.update(
            {
                "date": target,
                "code": quote.code,
                "name": preserved_name or quote.name,
                "open": quote.open,
                "high": quote.high,
                "low": quote.low,
                "close": quote.close,
                "volume": quote.volume,
                "amount": quote.amount,
            }
        )
        updated.loc[len(updated)] = [row.get(column, pd.NA) for column in updated.columns]
        changed = True
        status = "would_backfill" if is_backfill else "would_append"
        if not dry_run:
            status = "backfilled" if is_backfill else "appended"
        results.append(result | {"status": status, "latest_date": target})

    if dry_run or not changed:
        return results

    updated["_date_sort"] = pd.to_datetime(updated["date"], errors="coerce")
    updated = (
        updated.dropna(subset=["_date_sort"])
        .sort_values("_date_sort")
        .drop_duplicates(subset=["_date_sort"], keep="last")
        .drop(columns=["_date_sort"])
        .reset_index(drop=True)
    )
    try:
        updated.to_csv(path, index=False, encoding="utf-8")
    except OSError as exc:
        return [
            row | {"status": "write_error", "error": str(exc)}
            if row.get("status") in {"appended", "backfilled", "replaced"}
            else row
            for row in results
        ]
    return results


def _processed_stock_paths(data_dir: Path) -> list[Path]:
    stock_dir = data_dir / "processed" / "stocks"
    if not stock_dir.exists():
        return []
    return sorted(stock_dir.glob("*.csv"))


def _mirror_paths(processed_path: Path, data_dir: Path) -> list[Path]:
    paths = [processed_path]
    try:
        relative = processed_path.relative_to(data_dir / "processed")
    except ValueError:
        return paths
    raw_path = data_dir / "raw" / relative
    if raw_path.exists():
        paths.append(raw_path)
    return paths


def _merge_source(existing: object, source: str) -> str:
    parts = [part.strip() for part in str(existing or "").split("+") if part.strip()]
    if source not in parts:
        parts.append(source)
    return "+".join(parts) if parts else source


def _update_meta(csv_path: Path, quote: SnapshotQuote, target_date: str, data_dir: Path) -> None:
    try:
        relative = csv_path.relative_to(data_dir)
    except ValueError:
        return
    parts = relative.parts
    if not parts or parts[0] not in {"raw", "processed"}:
        return
    subparts = parts[1:-1]
    meta_path = data_dir / "meta" / Path(*subparts) / f"{quote.code}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.update(
        {
            "code": quote.code,
            "name": payload.get("name") or quote.name,
            "source": _merge_source(payload.get("source"), SNAPSHOT_SOURCE),
            "end_date": target_date,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "latest_snapshot_source": SNAPSHOT_SOURCE,
        }
    )
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_with_meta(
    csv_path: Path,
    quote: SnapshotQuote,
    target_date: str,
    data_dir: Path,
    dry_run: bool = False,
    replace_existing: bool = False,
) -> dict[str, Any]:
    row = append_snapshot_quote_to_csv(
        csv_path,
        quote,
        target_date,
        dry_run=dry_run,
        replace_existing=replace_existing,
    )
    if row["status"] in {"appended", "backfilled", "replaced"} and not dry_run:
        _update_meta(csv_path, quote, target_date, data_dir)
    return row


def _append_many_with_meta(
    csv_path: Path,
    quotes: list[SnapshotQuote],
    data_dir: Path,
    dry_run: bool = False,
    replace_existing: bool = False,
) -> list[dict[str, Any]]:
    rows = append_snapshot_quotes_to_csv(
        csv_path,
        quotes,
        dry_run=dry_run,
        replace_existing=replace_existing,
    )
    if dry_run:
        return rows
    successful_dates = {
        str(row["target_date"])
        for row in rows
        if row.get("status") in {"appended", "backfilled", "replaced"}
    }
    successful_quotes = [quote for quote in quotes if quote.date in successful_dates]
    if successful_quotes:
        latest_quote = max(successful_quotes, key=lambda quote: quote.date)
        _update_meta(csv_path, latest_quote, latest_quote.date, data_dir)
    return rows


def _write_outputs(output_dir: Path, audit_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(
        output_dir / "daily_market_snapshot_sync_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "daily_market_snapshot_sync_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def refresh_from_rows(
    data_dir: Path,
    output_dir: Path,
    target_date: str,
    rows: list[dict[str, Any]],
    source_kind: str,
    source_path: Path | None,
    max_symbols: int | None,
    dry_run: bool = False,
    allow_intraday: bool = False,
    after_close_time: str = "15:30",
    current_dt: datetime | None = None,
    source_updated_at: datetime | None = None,
    replace_existing: bool = False,
) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    target = _normalize_date(target_date)
    if target is None:
        raise ValueError("target_date must be a valid date.")

    quotes = {
        quote.code: quote
        for row in rows
        if (quote := snapshot_row_to_quote(row)) is not None and quote.date == target
    }
    targets = _processed_stock_paths(data_dir)
    if max_symbols is not None:
        targets = targets[:max_symbols]
    intraday_block_reason = _intraday_write_block_reason(
        target,
        current_dt=current_dt,
        source_updated_at=source_updated_at,
        after_close_time=after_close_time,
        allow_intraday=allow_intraday,
        dry_run=dry_run,
    )
    intraday_blocked = intraday_block_reason is not None

    audit_rows: list[dict[str, Any]] = []
    for processed_path in targets:
        code = _normalize_code(processed_path.stem)
        quote = quotes.get(code or "")
        if quote is None:
            audit_rows.append(
                {
                    "code": code,
                    "path": str(processed_path),
                    "status": "missing_quote",
                    "source": SNAPSHOT_SOURCE,
                    "target_date": target,
                }
            )
            continue
        for path in _mirror_paths(processed_path, data_dir):
            if intraday_blocked:
                audit_rows.append(
                    {
                        "path": str(path),
                        "code": quote.code,
                        "quote_date": quote.date,
                        "target_date": target,
                        "source": SNAPSHOT_SOURCE,
                        "status": "intraday_target_blocked",
                        "after_close_time": after_close_time,
                        "block_reason": intraday_block_reason,
                    }
                )
            else:
                audit_rows.append(
                    _append_with_meta(
                        path,
                        quote,
                        target,
                        data_dir,
                        dry_run=dry_run,
                        replace_existing=replace_existing,
                    )
                )

    status_counts = Counter(row.get("status", "unknown") for row in audit_rows)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": SNAPSHOT_SOURCE,
        "source_kind": source_kind,
        "source_path": str(source_path) if source_path else None,
        "target_date": target,
        "data_dir": str(data_dir),
        "total_quotes": len(quotes),
        "total_processed_symbols": len(targets),
        "total_target_files": len(audit_rows),
        "max_symbols": max_symbols,
        "dry_run": bool(dry_run),
        "replace_existing": bool(replace_existing),
        "allow_intraday": bool(allow_intraday),
        "after_close_time": after_close_time,
        "intraday_blocked": bool(intraday_blocked),
        "intraday_block_reason": intraday_block_reason,
        "statuses": dict(sorted(status_counts.items())),
    }
    _write_outputs(output_dir, audit_rows, summary)
    return summary


def refresh_dates(
    data_dir: Path,
    output_dir: Path,
    target_dates: list[str],
    daily_data_dir: Path,
    ingest_project_dir: Path,
    market: str,
    max_symbols: int | None,
    dry_run: bool = False,
    allow_intraday: bool = False,
    after_close_time: str = "15:30",
    current_dt: datetime | None = None,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Synchronize several snapshots while touching each stock cache only once."""
    normalized_dates = sorted(
        {
            normalized
            for value in target_dates
            if (normalized := _normalize_date(value)) is not None
        }
    )
    if not normalized_dates:
        raise ValueError("target_dates must contain at least one valid date.")

    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    quotes_by_date: dict[str, dict[str, SnapshotQuote]] = {}
    source_kinds: dict[str, str] = {}
    source_paths: dict[str, Path | None] = {}
    blocked_dates: dict[str, str] = {}

    for target in normalized_dates:
        result = load_market_snapshot_rows(
            trade_date=target,
            market=market,
            daily_data_dir=daily_data_dir,
            ingest_project_dir=ingest_project_dir,
            require_success=True,
        )
        resolved_date = result.trade_date or target
        if _normalize_date(resolved_date) != target:
            raise RuntimeError(
                f"Snapshot source returned trade_date={resolved_date} for requested date={target}."
            )
        quotes_by_date[target] = {
            quote.code: quote
            for row in result.rows
            if (quote := snapshot_row_to_quote(row)) is not None and quote.date == target
        }
        source_kinds[target] = result.source_kind
        source_paths[target] = result.source_path
        source_updated_at = None
        if result.source_path is not None:
            try:
                source_updated_at = datetime.fromtimestamp(result.source_path.stat().st_mtime)
            except OSError:
                source_updated_at = None
        block_reason = _intraday_write_block_reason(
            target,
            current_dt=current_dt,
            source_updated_at=source_updated_at,
            after_close_time=after_close_time,
            allow_intraday=allow_intraday,
            dry_run=dry_run,
        )
        if block_reason is not None:
            blocked_dates[target] = block_reason

    targets = _processed_stock_paths(data_dir)
    if max_symbols is not None:
        targets = targets[:max_symbols]

    audit_rows: list[dict[str, Any]] = []
    for processed_path in targets:
        code = _normalize_code(processed_path.stem)
        available_quotes: list[SnapshotQuote] = []
        for target in normalized_dates:
            quote = quotes_by_date[target].get(code or "")
            if quote is None:
                audit_rows.append(
                    {
                        "code": code,
                        "path": str(processed_path),
                        "status": "missing_quote",
                        "source": SNAPSHOT_SOURCE,
                        "target_date": target,
                    }
                )
            elif target not in blocked_dates:
                available_quotes.append(quote)

        for path in _mirror_paths(processed_path, data_dir):
            for target, block_reason in blocked_dates.items():
                quote = quotes_by_date[target].get(code or "")
                if quote is not None:
                    audit_rows.append(
                        {
                            "path": str(path),
                            "code": quote.code,
                            "quote_date": quote.date,
                            "target_date": target,
                            "source": SNAPSHOT_SOURCE,
                            "status": "intraday_target_blocked",
                            "after_close_time": after_close_time,
                            "block_reason": block_reason,
                        }
                    )
            if available_quotes:
                audit_rows.extend(
                    _append_many_with_meta(
                        path,
                        available_quotes,
                        data_dir,
                        dry_run=dry_run,
                        replace_existing=replace_existing,
                    )
                )

    status_counts = Counter(row.get("status", "unknown") for row in audit_rows)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": SNAPSHOT_SOURCE,
        "source_kinds": source_kinds,
        "source_paths": {
            target: str(path) if path else None for target, path in source_paths.items()
        },
        "target_dates": normalized_dates,
        "data_dir": str(data_dir),
        "total_snapshot_dates": len(normalized_dates),
        "total_quotes": sum(len(quotes) for quotes in quotes_by_date.values()),
        "quotes_by_date": {
            target: len(quotes) for target, quotes in quotes_by_date.items()
        },
        "total_processed_symbols": len(targets),
        "total_target_files": len(audit_rows),
        "max_symbols": max_symbols,
        "dry_run": bool(dry_run),
        "replace_existing": bool(replace_existing),
        "allow_intraday": bool(allow_intraday),
        "after_close_time": after_close_time,
        "intraday_blocked_dates": blocked_dates,
        "statuses": dict(sorted(status_counts.items())),
    }
    _write_outputs(output_dir, audit_rows, summary)
    return summary


def refresh(
    data_dir: Path,
    output_dir: Path,
    target_date: str | None,
    daily_data_dir: Path,
    ingest_project_dir: Path,
    market: str,
    max_symbols: int | None,
    dry_run: bool = False,
    allow_intraday: bool = False,
    after_close_time: str = "15:30",
    current_dt: datetime | None = None,
    replace_existing: bool = False,
) -> dict[str, Any]:
    result = load_market_snapshot_rows(
        trade_date=target_date,
        market=market,
        daily_data_dir=daily_data_dir,
        ingest_project_dir=ingest_project_dir,
        require_success=True,
    )
    resolved_date = result.trade_date or _normalize_date(target_date)
    if resolved_date is None:
        raise RuntimeError("Unable to resolve target trade date from market snapshot source.")
    source_updated_at = None
    if result.source_path is not None:
        try:
            source_updated_at = datetime.fromtimestamp(result.source_path.stat().st_mtime)
        except OSError:
            source_updated_at = None
    return refresh_from_rows(
        data_dir=data_dir,
        output_dir=output_dir,
        target_date=resolved_date,
        rows=result.rows,
        source_kind=result.source_kind,
        source_path=result.source_path,
        max_symbols=max_symbols,
        dry_run=dry_run,
        allow_intraday=allow_intraday,
        after_close_time=after_close_time,
        current_dt=current_dt,
        source_updated_at=source_updated_at,
        replace_existing=replace_existing,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync unified daily market snapshots into local processed/raw stock CSV caches."
    )
    parser.add_argument("--date", default=None, help="Target trading date. Defaults to latest available snapshot.")
    parser.add_argument(
        "--dates",
        default=None,
        help="Comma-separated trading dates to synchronize in one cache pass.",
    )
    parser.add_argument("--data-dir", default=PROJECT_ROOT / "data", type=Path)
    parser.add_argument("--daily-data-dir", default=DEFAULT_DAILY_MARKET_DATA_DIR, type=Path)
    parser.add_argument("--ingest-project-dir", default=DEFAULT_EXCHANGE_INGEST_DIR, type=Path)
    parser.add_argument("--market", choices=["all", "sse", "szse", "bse"], default="all")
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Audit intended cache writes without modifying CSV or meta files.")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace an existing date when normalized quote values changed.",
    )
    parser.add_argument("--allow-intraday", action="store_true", help="Allow writing today's snapshot before the after-close cutoff.")
    parser.add_argument("--after-close-time", default="15:30", help="HH:MM cutoff before today's snapshot writes are blocked.")
    parser.add_argument("--output-dir", default=None, type=Path)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (PROJECT_ROOT / "outputs" / "cache_runs" / f"daily_market_snapshot_sync_{stamp}")
    if args.dates:
        if args.date:
            parser.error("--date and --dates cannot be used together.")
        summary = refresh_dates(
            data_dir=args.data_dir,
            output_dir=output_dir,
            target_dates=[value.strip() for value in args.dates.split(",") if value.strip()],
            daily_data_dir=args.daily_data_dir,
            ingest_project_dir=args.ingest_project_dir,
            market=args.market,
            max_symbols=args.max_symbols,
            dry_run=args.dry_run,
            allow_intraday=args.allow_intraday,
            after_close_time=args.after_close_time,
            replace_existing=args.replace_existing,
        )
    else:
        summary = refresh(
            data_dir=args.data_dir,
            output_dir=output_dir,
            target_date=args.date,
            daily_data_dir=args.daily_data_dir,
            ingest_project_dir=args.ingest_project_dir,
            market=args.market,
            max_symbols=args.max_symbols,
            dry_run=args.dry_run,
            allow_intraday=args.allow_intraday,
            after_close_time=args.after_close_time,
            replace_existing=args.replace_existing,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output_dir)
    return 0 if summary.get("total_quotes", 0) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
