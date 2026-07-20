from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_etf_lab.market_data_source import (  # type: ignore
    DEFAULT_DAILY_MARKET_DATA_DIR,
    DEFAULT_EXCHANGE_INGEST_DIR,
    load_market_snapshot_rows,
    _normalize_trade_date,
)


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


def _serialize_fetch_status(status: Any) -> dict[str, Any] | None:
    if status is None:
        return None
    if isinstance(status, dict):
        return status.copy()
    if isinstance(status, (str, int, float, bool)):
        return {"value": status}

    keys = ("status", "run_id", "run_time", "message", "source", "market_filter", "row_count")
    payload: dict[str, Any] = {name: getattr(status, name) for name in keys if hasattr(status, name)}
    payload["type"] = type(status).__name__
    return payload


def _load_fetch_status(ingest_project_dir: Path, *, skip_check: bool) -> Any:
    utils = _load_market_data_utils(ingest_project_dir)
    ensure_latest_fetch_ok = getattr(utils, "ensure_latest_fetch_ok", None)
    get_latest_fetch_status = getattr(utils, "get_latest_fetch_status", None)
    fetch_status = None

    if callable(ensure_latest_fetch_ok) and not skip_check:
        ensure_latest_fetch_ok(project_root=str(ingest_project_dir))
    if callable(get_latest_fetch_status):
        fetch_status = get_latest_fetch_status(project_root=str(ingest_project_dir))

    if not callable(ensure_latest_fetch_ok) and callable(get_latest_fetch_status):
        status_value = str(getattr(fetch_status, "status", "")).lower()
        if status_value not in {"", "ok"}:
            raise RuntimeError(
                f"Latest exchange-data fetch status is not ok: {getattr(fetch_status, 'status', '')}"
            )

    return fetch_status


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight daily market data availability and freshness for the quant pipeline."
    )
    parser.add_argument(
        "--daily-data-dir",
        default=str(DEFAULT_DAILY_MARKET_DATA_DIR),
        help="Primary unified daily market data root.",
    )
    parser.add_argument(
        "--ingest-project-dir",
        default=str(DEFAULT_EXCHANGE_INGEST_DIR),
        help="Fallback exchange-ingest project directory.",
    )
    parser.add_argument(
        "--trade-date",
        default=None,
        help="Optional explicit trade date (YYYY-MM-DD or YYYYMMDD).",
    )
    parser.add_argument(
        "--min-row-count",
        type=int,
        default=1,
        help="Minimum row count required for a usable snapshot.",
    )
    parser.add_argument(
        "--skip-exchange-fetch-check",
        action="store_true",
        help="Skip checking exchange-ingest market_data_utils status.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON status payload output path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.min_row_count < 1:
        raise ValueError("min-row-count must be at least 1")

    daily_data_dir = Path(args.daily_data_dir)
    ingest_dir = Path(args.ingest_project_dir)
    trade_date = _normalize_trade_date(args.trade_date)
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trade_date": trade_date,
        "daily_data_dir": str(daily_data_dir),
        "ingest_project_dir": str(ingest_dir),
        "min_row_count": args.min_row_count,
        "skip_exch_fetch_check": bool(args.skip_exchange_fetch_check),
    }

    try:
        result = load_market_snapshot_rows(
            trade_date=trade_date,
            daily_data_dir=daily_data_dir,
            ingest_project_dir=ingest_dir,
            require_success=False,
        )
    except Exception as exc:
        payload.update(
            {
                "ok": False,
                "stage": "load_failed",
                "source_kind": None,
                "source_path": None,
                "trade_date": trade_date,
                "row_count": 0,
                "error": str(exc),
            }
        )
        if args.output:
            Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    payload.update(
        {
            "source_kind": result.source_kind,
            "source_path": str(result.source_path) if result.source_path else None,
            "trade_date": result.trade_date or trade_date,
            "row_count": len(result.rows),
            "fetch_status": _serialize_fetch_status(result.fetch_status),
            "reason": None,
        }
    )

    if not result.rows or len(result.rows) < args.min_row_count:
        payload.update({"ok": False, "stage": "empty_snapshot"})
        message = (
            f"No usable snapshot rows found from {payload['source_kind']} "
            f"for {payload['trade_date']} (rows={payload['row_count']}, required={args.min_row_count})."
        )
        payload["reason"] = message
        if args.output:
            Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 2

    if result.source_kind == "exchange_ingest":
        try:
            fetch_status = _load_fetch_status(ingest_dir, skip_check=args.skip_exchange_fetch_check)
        except Exception as exc:
            payload.update(
                {
                    "ok": False,
                    "stage": "exchange_fetch_status_check_failed",
                    "reason": str(exc),
                    "fetch_status": _serialize_fetch_status(payload.get("fetch_status")),
                }
            )
            if args.output:
                Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(payload, ensure_ascii=False))
            return 3
        else:
            if fetch_status is not None:
                payload["fetch_status"] = _serialize_fetch_status(fetch_status)

    payload["ok"] = True
    payload["stage"] = "ok"
    if args.output:
        Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
