from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINA_URL = "https://hq.sinajs.cn/list={symbols}"
SINA_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://finance.sina.com.cn/",
}
QUOTE_PATTERN = re.compile(r'var\s+hq_str_(?P<symbol>[a-z]{2}\d{6})="(?P<body>.*?)";', re.S)


class SinaQuote(NamedTuple):
    code: str
    name: str
    date: str
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    symbol: str


def market_symbol(code: str) -> str:
    normalized = str(code).strip().replace(".0", "").zfill(6)[-6:]
    prefix = "sh" if normalized.startswith(("5", "6", "9")) else "sz"
    return f"{prefix}{normalized}"


def _to_float(value: str) -> float:
    return float(str(value).strip() or "0")


def _normalize_date(value: str) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def parse_sina_response(text: str) -> dict[str, SinaQuote]:
    quotes: dict[str, SinaQuote] = {}
    for match in QUOTE_PATTERN.finditer(text):
        symbol = match.group("symbol")
        body = match.group("body")
        parts = body.split(",")
        if len(parts) < 32:
            continue
        try:
            open_price = _to_float(parts[1])
            close = _to_float(parts[3])
            high = _to_float(parts[4])
            low = _to_float(parts[5])
            volume = _to_float(parts[8])
            amount = _to_float(parts[9])
            quote_date = _normalize_date(parts[30])
        except (TypeError, ValueError):
            continue
        if min(open_price, close, high, low) <= 0:
            continue
        code = symbol[-6:]
        quotes[code] = SinaQuote(
            code=code,
            name=parts[0].strip(),
            date=quote_date,
            time=parts[31].strip(),
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
            amount=amount,
            symbol=symbol,
        )
    return quotes


def append_quote_to_csv(path: Path, quote: SinaQuote, target_date: str) -> dict[str, Any]:
    target = _normalize_date(target_date)
    result: dict[str, Any] = {
        "path": str(path),
        "code": quote.code,
        "quote_date": quote.date,
        "target_date": target,
        "source": "sina_hq_snapshot",
    }
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
    if (dates.dt.strftime("%Y-%m-%d") == target).any():
        return result | {"status": "already_present"}
    if previous_latest_date is not None and previous_latest_date > target:
        return result | {"status": "newer_date_present"}

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

    updated = pd.concat([frame, pd.DataFrame([row], columns=frame.columns)], ignore_index=True)
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
    return result | {"status": "appended", "latest_date": target}


def _append_with_meta(csv_path: Path, quote: SinaQuote, target_date: str, data_dir: Path) -> dict[str, Any]:
    row = append_quote_to_csv(csv_path, quote, target_date)
    if row["status"] == "appended":
        _update_meta(csv_path, quote, target_date, data_dir)
    return row


def _update_meta(csv_path: Path, quote: SinaQuote, target_date: str, data_dir: Path) -> None:
    try:
        relative = csv_path.relative_to(data_dir)
    except ValueError:
        return
    parts = relative.parts
    if not parts:
        return
    if parts[0] not in {"raw", "processed"}:
        return
    subparts = parts[1:-1]
    meta_path = data_dir / "meta" / Path(*subparts) / f"{quote.code}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.update(
        {
            "code": quote.code,
            "name": payload.get("name") or quote.name,
            "source": _merge_source(payload.get("source"), "sina_hq_snapshot"),
            "end_date": target_date,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "latest_snapshot_source": "sina_hq_snapshot",
            "latest_snapshot_time": quote.time,
        }
    )
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_source(existing: object, source: str) -> str:
    parts = [part.strip() for part in str(existing or "").split("+") if part.strip()]
    if source not in parts:
        parts.append(source)
    return "+".join(parts) if parts else source


def _target_csvs(data_dir: Path, include_etfs: bool) -> list[Path]:
    targets: list[Path] = []
    stock_dir = data_dir / "processed" / "stocks"
    if stock_dir.exists():
        targets.extend(sorted(stock_dir.glob("*.csv")))
    if include_etfs:
        processed_dir = data_dir / "processed"
        if processed_dir.exists():
            for path in sorted(processed_dir.glob("*.csv")):
                if path.name == "stock_market_cap_yi.csv":
                    continue
                targets.append(path)
    return targets


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


def _fetch_sina_batch(symbols: list[str], timeout: float) -> str:
    response = requests.get(
        SINA_URL.format(symbols=",".join(symbols)),
        headers=SINA_HEADERS,
        timeout=timeout,
    )
    response.raise_for_status()
    if not response.encoding:
        response.encoding = "gbk"
    return response.text


def _write_outputs(output_dir: Path, audit_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "sina_refresh_audit.csv", index=False, encoding="utf-8-sig")
    (output_dir / "sina_refresh_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def refresh(
    data_dir: Path,
    output_dir: Path,
    target_date: str,
    include_etfs: bool,
    batch_size: int,
    max_symbols: int | None,
    sleep_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    target = _normalize_date(target_date)
    targets = _target_csvs(data_dir, include_etfs)
    if max_symbols is not None:
        targets = targets[:max_symbols]

    code_to_path = {path.stem.zfill(6)[-6:]: path for path in targets}
    codes = sorted(code_to_path)
    audit_rows: list[dict[str, Any]] = []
    fetched_quotes: dict[str, SinaQuote] = {}
    fetch_failures: list[dict[str, Any]] = []

    for start in range(0, len(codes), max(batch_size, 1)):
        batch_codes = codes[start : start + max(batch_size, 1)]
        symbols = [market_symbol(code) for code in batch_codes]
        try:
            text = _fetch_sina_batch(symbols, timeout_seconds)
            fetched_quotes.update(parse_sina_response(text))
        except requests.RequestException as exc:
            fetch_failures.append(
                {
                    "batch_start": start,
                    "batch_size": len(batch_codes),
                    "symbols": ",".join(symbols),
                    "error": str(exc),
                }
            )
        if sleep_seconds > 0 and start + batch_size < len(codes):
            time.sleep(sleep_seconds)

    for code in codes:
        quote = fetched_quotes.get(code)
        if quote is None:
            audit_rows.append(
                {
                    "code": code,
                    "path": str(code_to_path[code]),
                    "status": "missing_quote",
                    "source": "sina_hq_snapshot",
                    "target_date": target,
                }
            )
            continue
        for path in _mirror_paths(code_to_path[code], data_dir):
            audit_rows.append(_append_with_meta(path, quote, target, data_dir))

    status_counts = Counter(row.get("status", "unknown") for row in audit_rows)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "sina_hq_snapshot",
        "target_date": target,
        "data_dir": str(data_dir),
        "include_etfs": bool(include_etfs),
        "total_targets": len(codes),
        "fetched_quotes": len(fetched_quotes),
        "fetch_failures": len(fetch_failures),
        "statuses": dict(sorted(status_counts.items())),
        "batch_size": int(max(batch_size, 1)),
        "max_symbols": max_symbols,
        "note": "Sina HQ is an after-close snapshot fallback, not a qfq historical data source.",
    }
    if fetch_failures:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "sina_fetch_failures.json").write_text(
            json.dumps(fetch_failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    _write_outputs(output_dir, audit_rows, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Append after-close Sina HQ snapshots to local market caches.")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Target trading date, YYYY-MM-DD.")
    parser.add_argument("--data-dir", default=PROJECT_ROOT / "data", type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--include-etfs", action="store_true")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (PROJECT_ROOT / "outputs" / "cache_runs" / f"sina_after_close_{stamp}")
    summary = refresh(
        data_dir=args.data_dir,
        output_dir=output_dir,
        target_date=args.date,
        include_etfs=args.include_etfs,
        batch_size=args.batch_size,
        max_symbols=args.max_symbols,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output_dir)
    return 0 if summary.get("fetched_quotes", 0) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
