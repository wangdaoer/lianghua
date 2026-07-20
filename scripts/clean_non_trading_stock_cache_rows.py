from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _normalize_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def _stock_csv_paths(data_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for relative_dir in (Path("processed") / "stocks", Path("raw") / "stocks"):
        stock_dir = data_dir / relative_dir
        if stock_dir.exists():
            paths.extend(sorted(stock_dir.glob("*.csv")))
    return paths


def _relative_to_data_dir(path: Path, data_dir: Path) -> Path:
    try:
        return path.relative_to(data_dir)
    except ValueError:
        return Path(path.name)


def _backup_file(path: Path, data_dir: Path, output_dir: Path) -> Path:
    backup_path = output_dir / "backups" / _relative_to_data_dir(path, data_dir)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)
    return backup_path


def _latest_date(frame: pd.DataFrame) -> str | None:
    if frame.empty or "date" not in frame.columns:
        return None
    dates = pd.to_datetime(frame["date"], errors="coerce")
    latest = dates.max()
    if pd.isna(latest):
        return None
    return latest.strftime("%Y-%m-%d")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"date": str, "code": str})


def _clean_csv(
    path: Path,
    *,
    data_dir: Path,
    output_dir: Path,
    target_date: str,
    dry_run: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "kind": "csv",
        "path": str(path),
        "code": path.stem,
        "target_date": target_date,
    }
    try:
        frame = _read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        return row | {"status": "read_error", "error": str(exc), "removed_rows": 0}
    if "date" not in frame.columns:
        return row | {"status": "missing_date_column", "removed_rows": 0}

    normalized_dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    mask = normalized_dates == target_date
    removed_rows = int(mask.sum())
    previous_latest_date = _latest_date(frame)
    row["previous_latest_date"] = previous_latest_date
    row["removed_rows"] = removed_rows
    if removed_rows == 0:
        return row | {"status": "no_target_date", "new_latest_date": previous_latest_date}

    cleaned = frame.loc[~mask].copy()
    new_latest_date = _latest_date(cleaned)
    row["new_latest_date"] = new_latest_date
    if dry_run:
        return row | {"status": "would_remove"}

    backup_path = _backup_file(path, data_dir, output_dir)
    cleaned.to_csv(path, index=False, encoding="utf-8")
    return row | {"status": "removed", "backup_path": str(backup_path)}


def _load_meta(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _update_meta(
    *,
    data_dir: Path,
    output_dir: Path,
    code: str,
    target_date: str,
    new_latest_date: str | None,
    dry_run: bool,
) -> dict[str, Any] | None:
    if new_latest_date is None:
        return None
    meta_path = data_dir / "meta" / "stocks" / f"{code}.json"
    if not meta_path.exists():
        return None
    payload = _load_meta(meta_path)
    if _normalize_date(payload.get("end_date")) != target_date:
        return {
            "kind": "meta",
            "path": str(meta_path),
            "code": code,
            "target_date": target_date,
            "status": "meta_end_date_unchanged",
            "previous_end_date": payload.get("end_date"),
            "new_end_date": payload.get("end_date"),
        }

    row = {
        "kind": "meta",
        "path": str(meta_path),
        "code": code,
        "target_date": target_date,
        "previous_end_date": payload.get("end_date"),
        "new_end_date": new_latest_date,
    }
    if dry_run:
        return row | {"status": "would_update_meta"}

    backup_path = _backup_file(meta_path, data_dir, output_dir)
    payload["end_date"] = new_latest_date
    payload["non_trading_cleanup_date"] = target_date
    payload["non_trading_cleanup_updated_at"] = datetime.now().isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return row | {"status": "updated_meta", "backup_path": str(backup_path)}


def _write_outputs(output_dir: Path, audit_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(
        output_dir / "non_trading_row_cleanup_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / "non_trading_row_cleanup_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clean_cache(data_dir: Path, output_dir: Path, target_date: str, dry_run: bool = False) -> dict[str, Any]:
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    target = _normalize_date(target_date)
    if target is None:
        raise ValueError("target_date must be a valid date.")

    csv_rows: list[dict[str, Any]] = []
    latest_by_code: dict[str, str | None] = {}
    for path in _stock_csv_paths(data_dir):
        audit_row = _clean_csv(
            path,
            data_dir=data_dir,
            output_dir=output_dir,
            target_date=target,
            dry_run=dry_run,
        )
        csv_rows.append(audit_row)
        if audit_row.get("status") in {"removed", "would_remove"}:
            code = str(audit_row.get("code") or path.stem)
            existing = latest_by_code.get(code)
            new_latest = audit_row.get("new_latest_date")
            if new_latest is not None and (existing is None or str(new_latest) > existing):
                latest_by_code[code] = str(new_latest)
            elif code not in latest_by_code:
                latest_by_code[code] = None

    meta_rows = [
        row
        for code, new_latest_date in sorted(latest_by_code.items())
        if (row := _update_meta(
            data_dir=data_dir,
            output_dir=output_dir,
            code=code,
            target_date=target,
            new_latest_date=new_latest_date,
            dry_run=dry_run,
        ))
        is not None
    ]
    audit_rows = csv_rows + meta_rows
    status_counts = Counter(str(row.get("status", "unknown")) for row in audit_rows)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_date": target,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "dry_run": bool(dry_run),
        "total_csv_files": len(csv_rows),
        "changed_csv_files": status_counts.get("removed", 0) + status_counts.get("would_remove", 0),
        "unchanged_csv_files": status_counts.get("no_target_date", 0),
        "removed_rows": int(sum(int(row.get("removed_rows", 0) or 0) for row in csv_rows)),
        "updated_meta_files": status_counts.get("updated_meta", 0) + status_counts.get("would_update_meta", 0),
        "statuses": dict(sorted(status_counts.items())),
    }
    _write_outputs(output_dir, audit_rows, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove non-trading date rows from local stock OHLCV cache files."
    )
    parser.add_argument("--target-date", "--date", required=True, help="Non-trading date to remove, e.g. 2026-06-19.")
    parser.add_argument("--data-dir", default=PROJECT_ROOT / "data", type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Audit changes without writing CSV or meta files.")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        PROJECT_ROOT / "outputs" / "cache_runs" / f"non_trading_cleanup_{args.target_date}_{stamp}"
    )
    summary = clean_cache(
        data_dir=args.data_dir,
        output_dir=output_dir,
        target_date=args.target_date,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
