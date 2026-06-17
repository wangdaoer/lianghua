from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_etf_lab.config import ETFSpec, load_config
from quant_etf_lab.data import (
    cache_paths,
    fetch_akshare_history_with_retries,
    filter_history_by_date,
    history_source,
    load_cached_history,
    resolve_universe,
    save_history,
    today_yyyymmdd,
)


def _is_complete(path: Path, start_date: str, end_date: str | None) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing_file"
    try:
        frame = pd.read_csv(path, parse_dates=["date"])
    except Exception as exc:
        return False, f"read_error:{exc}"
    if frame.empty or "date" not in frame:
        return False, "empty_or_missing_date"
    requested_start = pd.to_datetime(start_date, format="%Y%m%d")
    requested_end = pd.to_datetime(end_date or today_yyyymmdd(), format="%Y%m%d")
    actual_start = pd.to_datetime(frame["date"]).min()
    actual_end = pd.to_datetime(frame["date"]).max()
    if actual_end < requested_start or actual_start > requested_end:
        return False, f"no_overlap:{actual_start.date()}:{actual_end.date()}"

    coverage_notes: list[str] = []
    if actual_start > requested_start + pd.Timedelta(days=10):
        coverage_notes.append(f"starts_late:{actual_start.date()}")
    if actual_end < requested_end - pd.Timedelta(days=10):
        coverage_notes.append(f"ends_early:{actual_end.date()}")
    if coverage_notes:
        return True, "complete_with_coverage_note:" + "|".join(coverage_notes)
    return True, "complete"


def _audit_rows(config_path: Path, start_date: str, end_date: str | None) -> list[dict[str, Any]]:
    config = load_config(config_path)
    rows: list[dict[str, Any]] = []
    for item in resolve_universe(config):
        path = cache_paths(config, item)["processed"]
        complete, reason = _is_complete(path, start_date, end_date)
        rows.append(
            {
                "code": item.code,
                "name": item.name,
                "asset_type": item.asset_type,
                "path": str(path),
                "cached": path.exists(),
                "complete": complete,
                "reason": reason,
            }
        )
    return rows


def _write_status(output_dir: Path, rows: list[dict[str, Any]], prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"{prefix}.csv", index=False, encoding="utf-8-sig")
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": int(len(frame)),
        "cached": int(frame["cached"].sum()) if "cached" in frame else 0,
        "complete": int(frame["complete"].sum()) if "complete" in frame else 0,
        "incomplete": int((~frame["complete"]).sum()) if "complete" in frame else 0,
    }
    (output_dir / f"{prefix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def audit(config_path: Path, start_date: str, end_date: str | None, output_dir: Path) -> None:
    rows = _audit_rows(config_path, start_date, end_date)
    _write_status(output_dir, rows, "cache_audit")
    total = len(rows)
    complete = sum(1 for row in rows if row["complete"])
    print(f"Audit completed: total={total}, complete={complete}, incomplete={total - complete}")
    print(output_dir / "cache_audit.csv")


def cache(
    config_path: Path,
    start_date: str,
    end_date: str | None,
    output_dir: Path,
    retry_count: int,
    pause_seconds: float,
    max_symbols: int | None,
    retry_incomplete: bool,
    workers: int,
) -> None:
    config = load_config(config_path)
    universe = list(resolve_universe(config))
    failures: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    candidates: list[tuple[ETFSpec, str]] = []
    for item in universe:
        path = cache_paths(config, item)["processed"]
        complete, reason = _is_complete(path, start_date, end_date)
        if complete and not retry_incomplete:
            skipped.append({"code": item.code, "name": item.name, "reason": "complete"})
            continue
        candidates.append((item, reason))
    if max_symbols is not None:
        candidates = candidates[:max_symbols]

    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(candidates)
    print(f"Cache start: total_universe={len(universe)}, to_fetch={total}, skipped_complete={len(skipped)}")
    def fetch_one(payload: tuple[int, ETFSpec, str]) -> tuple[bool, dict[str, Any]]:
        index, item, reason = payload
        started_at = datetime.now()
        try:
            frame = fetch_akshare_history_with_retries(
                item,
                start_date,
                end_date,
                config.data.period,
                config.data.adjust,
                retry_count=retry_count,
                pause_seconds=pause_seconds,
            )
            frame = filter_history_by_date(frame, start_date, end_date)
            save_history(config, item, frame, source=str(frame.attrs.get("source", history_source(item))))
            row = {
                "index": index,
                "code": item.code,
                "name": item.name,
                "source": str(frame.attrs.get("source", history_source(item))),
                "rows": int(len(frame)),
                "start_date": frame["date"].min().strftime("%Y-%m-%d") if not frame.empty else "",
                "end_date": frame["date"].max().strftime("%Y-%m-%d") if not frame.empty else "",
                "previous_reason": reason,
                "elapsed_seconds": round((datetime.now() - started_at).total_seconds(), 3),
            }
            return True, row
        except Exception as exc:
            row = {
                "index": index,
                "code": item.code,
                "name": item.name,
                "asset_type": item.asset_type,
                "previous_reason": reason,
                "error": str(exc),
                "elapsed_seconds": round((datetime.now() - started_at).total_seconds(), 3),
            }
            return False, row

    payloads = [(index, item, reason) for index, (item, reason) in enumerate(candidates, start=1)]
    if workers <= 1:
        for payload in payloads:
            ok, row = fetch_one(payload)
            if ok:
                successes.append(row)
                print(
                    f"[{row['index']}/{total}] OK {row['code']} {row['name']} rows={row['rows']} "
                    f"source={row['source']} elapsed={row['elapsed_seconds']}s",
                    flush=True,
                )
            else:
                failures.append(row)
                print(f"[{row['index']}/{total}] FAIL {row['code']} {row['name']}: {str(row['error'])[:180]}", flush=True)
            pd.DataFrame(successes).to_csv(output_dir / "cache_successes.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(failures).to_csv(output_dir / "cache_failures.csv", index=False, encoding="utf-8-sig")
            if pause_seconds > 0:
                time.sleep(pause_seconds)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for payload in payloads:
                futures.append(executor.submit(fetch_one, payload))
                if pause_seconds > 0:
                    time.sleep(pause_seconds)
            for future in as_completed(futures):
                ok, row = future.result()
                if ok:
                    successes.append(row)
                    print(
                        f"[{row['index']}/{total}] OK {row['code']} {row['name']} rows={row['rows']} "
                        f"source={row['source']} elapsed={row['elapsed_seconds']}s",
                        flush=True,
                    )
                else:
                    failures.append(row)
                    print(f"[{row['index']}/{total}] FAIL {row['code']} {row['name']}: {str(row['error'])[:180]}", flush=True)
                pd.DataFrame(successes).to_csv(output_dir / "cache_successes.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(failures).to_csv(output_dir / "cache_failures.csv", index=False, encoding="utf-8-sig")

    audit_rows = _audit_rows(config_path, start_date, end_date)
    _write_status(output_dir, audit_rows, "cache_audit")
    complete = sum(1 for row in audit_rows if row["complete"])
    print(f"Cache finished: successes={len(successes)}, failures={len(failures)}, complete={complete}/{len(audit_rows)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable market data cache helper.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", default=Path("outputs/cache_runs/latest"), type=Path)
    parser.add_argument("--retry-count", type=int, default=4)
    parser.add_argument("--pause-seconds", type=float, default=0.25)
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    start = args.start or config.data.start_date
    end = args.end if args.end is not None else config.data.end_date
    if args.audit_only:
        audit(args.config, start, end, args.output_dir)
    else:
        cache(
            args.config,
            start,
            end,
            args.output_dir,
            retry_count=args.retry_count,
            pause_seconds=args.pause_seconds,
            max_symbols=args.max_symbols,
            retry_incomplete=args.retry_incomplete,
            workers=max(args.workers, 1),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
