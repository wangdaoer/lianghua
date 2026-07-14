from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from build_research_database import (
    normalize_asof_date,
    prepare_daily_price_frame,
    prepare_observation_frame,
    select_supported_observation_files,
)
from research_database import ResearchDatabase, normalize_a_share_symbols
from workspace_paths import daily_data_root


def _daily_candidates(daily_dir: Path, asof_date: str) -> list[Path]:
    return [
        daily_dir / f"ths_hs_a_share_{asof_date}{suffix}"
        for suffix in (".xls", ".xlsx", ".csv")
    ]


def _load_daily_frame(daily_dir: Path, asof_date: str) -> tuple[Path, pd.DataFrame]:
    errors: list[str] = []
    for path in _daily_candidates(daily_dir, asof_date):
        if not path.exists():
            continue
        try:
            frame = prepare_daily_price_frame(path)
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if frame is not None and not frame.empty:
            return path, frame
        errors.append(f"{path.name}: symbol/date/close could not be resolved")
    detail = "; ".join(errors) if errors else "no matching files"
    raise FileNotFoundError(f"No usable daily market file for {asof_date}: {detail}")


def _parsed_dates(frame: pd.DataFrame, label: str) -> pd.Series:
    parsed = pd.to_datetime(frame["date"].astype(str), errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{label} contains invalid dates")
    return parsed


def write_status_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def sync_daily_research_database(
    db_path: Path,
    daily_dir: Path,
    output_dir: Path,
    asof_date: str,
) -> dict[str, object]:
    token = normalize_asof_date(asof_date)
    iso_date = pd.to_datetime(token, format="%Y%m%d").strftime("%Y-%m-%d")
    daily_path, daily_frame = _load_daily_frame(daily_dir, iso_date)
    daily_dates = _parsed_dates(daily_frame, str(daily_path)).dt.strftime("%Y-%m-%d")
    if set(daily_dates.unique()) != {iso_date}:
        raise ValueError(
            f"Daily market file must contain only {iso_date}: found={sorted(daily_dates.unique())}"
        )
    daily_frame = daily_frame.copy()
    daily_frame["date"] = daily_dates

    observation_paths = select_supported_observation_files(output_dir, token)
    if not observation_paths:
        raise FileNotFoundError(f"No dated observation CSV files found for {iso_date} in {output_dir}")

    observation_frames: list[tuple[Path, pd.DataFrame]] = []
    for path in observation_paths:
        frame = prepare_observation_frame(path, token)
        observation_frames.append((path, frame))

    db = ResearchDatabase(db_path)
    expected_symbols = int(normalize_a_share_symbols(daily_frame["symbol"]).dropna().nunique())
    observation_read_rows = sum(len(frame) for _, frame in observation_frames)
    with db.connect() as conn:
        daily_changed = int(db.import_prices_into(conn, daily_frame, str(daily_path), update_existing=True))
        observation_changed = 0
        for path, frame in observation_frames:
            observation_changed += int(db.import_observations_into(conn, frame, path.stem, str(path)))

        coverage_row = conn.execute(
            """SELECT COUNT(*) AS total_rows, COUNT(DISTINCT symbol) AS symbols,
                      MIN(date) AS first_date, MAX(date) AS latest_date,
                      SUM(CASE WHEN date = ? THEN 1 ELSE 0 END) AS asof_rows
               FROM daily_prices""",
            (iso_date,),
        ).fetchone()
        observation_total = int(conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0])
        asof_rows = int(coverage_row[4] or 0)
        if asof_rows < expected_symbols:
            raise RuntimeError(
                f"Daily database coverage incomplete for {iso_date}: rows={asof_rows}, expected_symbols={expected_symbols}"
            )

    return {
        "schema_version": 1,
        "status": "success",
        "asof_date": iso_date,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "database": str(db_path),
        "daily_source": str(daily_path),
        "daily_read_rows": int(len(daily_frame)),
        "daily_expected_symbols": expected_symbols,
        "daily_changed_rows": daily_changed,
        "observation_files": [str(path) for path in observation_paths],
        "observation_file_count": len(observation_paths),
        "observation_read_rows": int(observation_read_rows),
        "observation_changed_rows": int(observation_changed),
        "database_daily_rows": int(coverage_row[0]),
        "database_symbols": int(coverage_row[1]),
        "database_first_date": coverage_row[2],
        "database_latest_date": coverage_row[3],
        "database_asof_rows": asof_rows,
        "database_observation_rows": observation_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally sync today's prices and model observations to SQLite.")
    parser.add_argument("--db", default="data/research.sqlite3")
    parser.add_argument(
        "--daily-dir", default=str(daily_data_root() / "ths_exports" / "normalized")
    )
    parser.add_argument("--output-dir", default="outputs/high_return_v2")
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--status-output", required=True)
    args = parser.parse_args()

    status_path = Path(args.status_output)
    try:
        status = sync_daily_research_database(
            Path(args.db),
            Path(args.daily_dir),
            Path(args.output_dir),
            args.asof_date,
        )
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "asof_date": args.asof_date,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "database": args.db,
            "message": f"{type(exc).__name__}: {exc}",
        }
        write_status_atomic(status_path, failure)
        raise
    write_status_atomic(status_path, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
