"""Unified daily market snapshot reader.

The reader uses the unified daily hub (`daily-market-data`) first and falls back
to the exchange-ingest project only when the hub data is missing for the target
trade date.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd


DEFAULT_DAILY_MARKET_DATA_DIR = Path("D:/codex/daily-market-data")
DEFAULT_EXCHANGE_INGEST_DIR = Path("D:/codex/2026-06-15-exchange-data-ingest")


@dataclass(frozen=True)
class MarketSnapshotLoadResult:
    rows: list[dict[str, Any]]
    source_kind: str
    source_path: Path | None
    trade_date: str | None
    fetch_status: Any


def _load_market_data_utils(ingest_project_dir: Path) -> ModuleType:
    utils_path = ingest_project_dir / "scripts" / "market_data_utils.py"
    if not utils_path.exists():
        raise FileNotFoundError(f"Missing market data utility module: {utils_path}")
    spec = importlib.util.spec_from_file_location("exchange_ingest_market_data_utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import market data utility module: {utils_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    spec.loader.exec_module(module)
    return module


def _normalize_trade_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text


def _compact_trade_date(value: str | None) -> str | None:
    if not value:
        return None
    return value.replace("-", "")


def _filter_market(rows: list[dict[str, Any]], market: str) -> list[dict[str, Any]]:
    if market == "all":
        return rows
    return [row for row in rows if str(row.get("market", "")).lower() == market]


def _read_snapshot_csv(path: Path, market: str) -> list[dict[str, Any]]:
    try:
        frame = pd.read_csv(path, dtype={"security_code": str, "trade_date": str, "market": str})
    except (OSError, pd.errors.EmptyDataError):
        return []
    if "market" not in frame.columns or "trade_date" not in frame.columns:
        return []
    rows = frame.to_dict(orient="records")
    return _filter_market(rows, market)


def _daily_snapshot_csv_candidates(daily_data_dir: Path, trade_date: str | None) -> list[Path]:
    snapshots_dir = daily_data_dir / "snapshots"
    normalized_dir = daily_data_dir / "data" / "normalized"
    if trade_date:
        compact = _compact_trade_date(trade_date)
        names = [
            f"{trade_date}_market_snapshot.csv",
            f"{compact}_market_snapshot.csv",
            f"snapshot_all_{compact}.csv",
        ]
        return [base / name for base in (snapshots_dir, normalized_dir) for name in names]
    candidates: list[Path] = []
    for base in (snapshots_dir, normalized_dir):
        candidates.extend(sorted(base.glob("*_market_snapshot.csv"), reverse=True))
        candidates.extend(sorted(base.glob("snapshot_all_*.csv"), reverse=True))
    return candidates


def _parse_trade_date(value: Any) -> datetime.date | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 8 and text.isdigit():
        try:
            return datetime.strptime(text, "%Y%m%d").date()
        except ValueError:
            return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def _latest_trade_date_from_daily_dir(daily_data_dir: Path) -> str | None:
    db_path = daily_data_dir / "sqlite" / "market.db"
    latest: datetime.date | None = None

    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                row = conn.execute("SELECT MAX(trade_date) FROM market_snapshot").fetchone()
                latest = _parse_trade_date(row[0]) if row else None
            finally:
                conn.close()
        except sqlite3.Error:
            latest = latest

    snapshot_roots = (daily_data_dir / "snapshots", daily_data_dir / "data" / "normalized")
    for root in snapshot_roots:
        if not root.exists():
            continue
        for path in root.glob("*_market_snapshot.csv"):
            stem = path.stem
            candidates = [stem[:10], stem.replace("snapshot_all_", "")]
            for candidate in candidates:
                parsed = _parse_trade_date(candidate)
                if parsed is None:
                    continue
                latest = parsed if latest is None or parsed > latest else latest
        for path in root.glob("snapshot_all_*.csv"):
            stem = path.stem.removeprefix("snapshot_all_")
            parsed = _parse_trade_date(stem)
            if parsed is None:
                continue
            latest = parsed if latest is None or parsed > latest else latest

    if latest is None:
        return None
    return latest.strftime("%Y-%m-%d")


def _get_fetch_status(
    utils: ModuleType,
    ingest_project_dir: Path,
    require_success: bool,
) -> Any:
    fetch_status = None
    ensure_latest_fetch_ok = getattr(utils, "ensure_latest_fetch_ok", None)
    get_latest_fetch_status = getattr(utils, "get_latest_fetch_status", None)

    if not callable(ensure_latest_fetch_ok) and not callable(get_latest_fetch_status):
        raise RuntimeError(
            "market_data_utils must expose ensure_latest_fetch_ok or get_latest_fetch_status."
        )

    if callable(ensure_latest_fetch_ok):
        if require_success:
            ensure_latest_fetch_ok(project_root=str(ingest_project_dir))
    if callable(get_latest_fetch_status):
        fetch_status = get_latest_fetch_status(project_root=str(ingest_project_dir))

    if require_success and not callable(ensure_latest_fetch_ok) and fetch_status is not None:
        status_value = str(getattr(fetch_status, "status", "")).lower()
        if status_value not in {"", "ok"}:
            raise RuntimeError(
                f"Latest exchange-data fetch status is not ok: {getattr(fetch_status, 'status', '')}"
            )

    return fetch_status


def _read_daily_sqlite(daily_data_dir: Path, trade_date: str | None, market: str) -> tuple[list[dict[str, Any]], Path | None]:
    db_path = daily_data_dir / "sqlite" / "market.db"
    if not db_path.exists():
        return [], None
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            resolved_date = trade_date
            if not resolved_date:
                row = conn.execute("SELECT MAX(trade_date) FROM market_snapshot").fetchone()
                resolved_date = _normalize_trade_date(row[0]) if row else None
            if not resolved_date:
                return [], db_path
            sql = "SELECT * FROM market_snapshot WHERE trade_date = ?"
            params: list[Any] = [resolved_date]
            if market != "all":
                sql += " AND market = ?"
                params.append(market)
            frame = pd.read_sql_query(sql, conn, params=params)
            if frame.empty:
                return [], db_path
            return frame.to_dict(orient="records"), db_path
        finally:
            conn.close()
    except sqlite3.Error:
        return [], db_path


def _read_daily_hub(
    daily_data_dir: Path,
    trade_date: str | None,
    market: str,
) -> tuple[list[dict[str, Any]], str, Path | None]:
    sqlite_rows, sqlite_path = _read_daily_sqlite(daily_data_dir, trade_date, market)
    if sqlite_rows:
        return sqlite_rows, "daily_market_data_sqlite", sqlite_path
    for path in _daily_snapshot_csv_candidates(daily_data_dir, trade_date):
        if not path.exists():
            continue
        rows = _read_snapshot_csv(path, market)
        if rows:
            return rows, "daily_market_data_csv", path
    return [], "", None


def _rows_trade_date(rows: list[dict[str, Any]], fallback: str | None) -> str | None:
    if rows:
        return _normalize_trade_date(rows[0].get("trade_date")) or fallback
    return fallback


def _load_exchange_rows(
    utils: Any,
    ingest_dir: Path,
    trade_date: str | None,
    market: str,
    require_success: bool = True,
    fetch_status: Any = None,
) -> MarketSnapshotLoadResult:
    if fetch_status is None:
        fetch_status = _get_fetch_status(utils, ingest_dir, require_success)
    rows = list(
        utils.load_snapshot_rows(
            trade_date=trade_date,
            market=market,
            project_root=str(ingest_dir),
            fallback_to_csv=True,
        )
    )
    return MarketSnapshotLoadResult(
        rows=list(rows),
        source_kind="exchange_ingest",
        source_path=ingest_dir / "data" / "market.db",
        trade_date=_rows_trade_date(rows, trade_date),
        fetch_status=fetch_status,
    )


def _read_from_hub_then_exchange(
    daily_data_dir: Path,
    ingest_dir: Path,
    trade_date: str | None,
    market: str,
    utils: Any,
    require_success: bool,
) -> MarketSnapshotLoadResult:
    fetch_status = _get_fetch_status(utils, ingest_dir, require_success)
    explicit_trade_date = _normalize_trade_date(trade_date)

    if explicit_trade_date:
        rows, source_kind, source_path = _read_daily_hub(daily_data_dir, explicit_trade_date, market)
        if rows:
            return MarketSnapshotLoadResult(
                rows=rows,
                source_kind=source_kind,
                source_path=source_path,
                trade_date=_rows_trade_date(rows, explicit_trade_date),
                fetch_status=fetch_status,
            )

        return _load_exchange_rows(
            utils=utils,
            ingest_dir=ingest_dir,
            trade_date=explicit_trade_date,
            market=market,
            require_success=False,
            fetch_status=fetch_status,
        )

    daily_latest = _normalize_trade_date(_latest_trade_date_from_daily_dir(daily_data_dir))
    exchange_latest = _normalize_trade_date(utils.get_latest_trade_date(project_root=str(ingest_dir)))
    exchange_date = _parse_trade_date(exchange_latest)
    daily_date = _parse_trade_date(daily_latest)
    exchange_is_newer = exchange_date is not None and (daily_date is None or exchange_date > daily_date)

    if exchange_is_newer and exchange_latest:
        return _load_exchange_rows(
            utils=utils,
            ingest_dir=ingest_dir,
            trade_date=exchange_latest,
            market=market,
            require_success=False,
            fetch_status=fetch_status,
        )

    rows, source_kind, source_path = _read_daily_hub(daily_data_dir, daily_latest, market)
    if rows:
        return MarketSnapshotLoadResult(
            rows=rows,
            source_kind=source_kind,
            source_path=source_path,
            trade_date=_rows_trade_date(rows, daily_latest),
            fetch_status=fetch_status,
        )

    return _load_exchange_rows(
            utils=utils,
            ingest_dir=ingest_dir,
            trade_date=daily_latest,
            market=market,
            require_success=False,
            fetch_status=fetch_status,
        )


def load_market_snapshot_rows(
    trade_date: str | None = None,
    market: str = "all",
    daily_data_dir: str | Path | None = None,
    ingest_project_dir: str | Path | None = None,
    require_success: bool = True,
) -> MarketSnapshotLoadResult:
    """Load exchange snapshot rows using the unified local data-source order."""
    if market not in {"all", "sse", "szse"}:
        raise ValueError("market must be one of: all, sse, szse")
    daily_dir = Path(daily_data_dir) if daily_data_dir is not None else DEFAULT_DAILY_MARKET_DATA_DIR
    ingest_dir = Path(ingest_project_dir) if ingest_project_dir is not None else DEFAULT_EXCHANGE_INGEST_DIR
    utils = _load_market_data_utils(ingest_dir)
    return _read_from_hub_then_exchange(
        daily_data_dir=daily_dir,
        ingest_dir=ingest_dir,
        trade_date=trade_date,
        market=market,
        utils=utils,
        require_success=require_success,
    )
