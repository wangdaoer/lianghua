from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_etf_lab.ths_export_source import (  # noqa: E402
    DEFAULT_THS_EXPORT_ROOT,
    normalize_trade_date,
    read_ths_export_rows,
    ths_export_paths,
)

DEFAULT_WORKFLOW_DIR = Path("D:/codex/2026-06-18-a-share-research-daily-workflow")


def _today_trade_date() -> str:
    return date.today().strftime("%Y-%m-%d")


def _is_weekend(trade_date: str) -> bool:
    parsed = datetime.strptime(trade_date, "%Y-%m-%d").date()
    return parsed.weekday() >= 5


def _run_export_script(export_root: Path, trade_date: str, min_row_count: int, force: bool, allow_weekend: bool) -> int:
    script_path = PROJECT_ROOT / "scripts" / "run_ths_hs_a_share_export.ps1"
    args = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        "-TradeDate",
        trade_date,
        "-ExportRoot",
        str(export_root),
        "-MinRowCount",
        str(min_row_count),
    ]
    if force:
        args.append("-Force")
    if allow_weekend:
        args.append("-AllowWeekend")
    return subprocess.run(args, cwd=PROJECT_ROOT).returncode


def _safe_file_stats(path: Path) -> tuple[bool, int | None, str | None]:
    try:
        if not path.exists():
            return False, None, None
        stats = path.stat()
        return True, stats.st_size, datetime.fromtimestamp(stats.st_mtime).isoformat(timespec="seconds")
    except OSError:
        return False, None, None


def _source_health_path_for_date(trade_date: str, explicit_path: str | None = None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    candidates = [
        PROJECT_ROOT.parent / "2026-06-18-a-share-research-daily-workflow" / "outputs" / trade_date / "source_health.json",
        DEFAULT_WORKFLOW_DIR / "outputs" / trade_date / "source_health.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def _read_existing_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _write_source_health(
    trade_date: str,
    source_status: dict[str, Any],
    source_health_path: Path | None,
) -> None:
    if source_health_path is None:
        return
    payload = _read_existing_payload(source_health_path)
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        sources = {}

    sources["ths_hs_a_share_export"] = source_status
    payload["sources"] = sources
    payload["run_date"] = trade_date
    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")

    source_state = str(source_status.get("status", "")).lower()
    current_overall = str(payload.get("overall_status", "")).lower()
    if source_state == "failed":
        payload["overall_status"] = "failed"
    elif not current_overall:
        payload["overall_status"] = "ok" if source_state == "ok" else source_state
    elif current_overall == "failed":
        payload["overall_status"] = "failed"
    elif source_state not in {"", "ok", "skipped"}:
        payload["overall_status"] = source_state

    source_health_path.parent.mkdir(parents=True, exist_ok=True)
    source_health_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _source_status_payload(
    trade_date: str,
    status: str,
    source_path: str | None,
    manifest_path: str | None,
    row_count: int | None,
    message: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    exists = False
    size_bytes = None
    modified_at = None
    if source_path:
        exists, size_bytes, modified_at = _safe_file_stats(Path(source_path))
    return {
        "trade_date": trade_date,
        "status": status,
        "critical": False,
        "exists": exists,
        "path": source_path,
        "size_bytes": size_bytes,
        "modified_at": modified_at,
        "normalized_path": source_path,
        "manifest_path": manifest_path,
        "row_count": row_count,
        "message": message if message else f"ths_hs_a_share_export status={status}",
        "error": error,
    }


def _status_payload(
    trade_date: str,
    export_root: Path,
    min_row_count: int,
    success: bool,
    stage: str,
    message: str | None = None,
    source_path: str | None = None,
) -> dict[str, Any]:
    _, normalized_path, manifest_path = ths_export_paths(trade_date, export_root)
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "trade_date": trade_date,
        "export_root": str(export_root),
        "min_row_count": int(min_row_count),
        "success": bool(success),
        "stage": stage,
        "normalized_path": str(normalized_path),
        "manifest_path": str(manifest_path),
        "source_path": source_path,
    }
    if message:
        payload["message"] = message
    return payload


def _check_export(trade_date: str, export_root: Path, min_row_count: int) -> tuple[bool, str | None, str | None, int | None]:
    _, normalized_path, _ = ths_export_paths(trade_date, export_root)
    try:
        rows, source_path = read_ths_export_rows(
            trade_date=trade_date,
            export_root=export_root,
            min_row_count=min_row_count,
            require_status_ok=True,
        )
        return True, None, str(source_path), len(rows)
    except Exception as exc:
        return False, str(exc), str(normalized_path), None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure THS HS A-share export has usable content before daily workflows.")
    parser.add_argument("--trade-date", default=None, help="Trade date YYYY-MM-DD or YYYYMMDD. Default: today.")
    parser.add_argument("--export-root", default=str(DEFAULT_THS_EXPORT_ROOT), help="THS export root directory.")
    parser.add_argument("--min-row-count", type=int, default=5000, help="Minimum normalized row count required.")
    parser.add_argument("--allow-weekend", action="store_true", help="Allow running rerun on weekend dates.")
    parser.add_argument("--force-export", action="store_true", help="Force export rerun even when check already passes.")
    parser.add_argument("--skip-rerun", action="store_true", help="Do not rerun export on validation failure.")
    parser.add_argument("--status-path", default=None, help="Write status JSON to this path.")
    parser.add_argument("--source-health-path", default=None, help="Write/update source_health JSON to this path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trade_date = normalize_trade_date(args.trade_date or _today_trade_date())
    export_root = Path(args.export_root)
    min_row_count = args.min_row_count
    source_health_path = _source_health_path_for_date(trade_date, args.source_health_path)
    if min_row_count < 1:
        raise ValueError("min-row-count must be at least 1")

    _, normalized_path, manifest_path = ths_export_paths(trade_date, export_root)
    manifest_path_str = str(manifest_path)
    normalized_path_str = str(normalized_path)

    if _is_weekend(trade_date) and not args.allow_weekend:
        payload = _status_payload(
            trade_date=trade_date,
            export_root=export_root,
            min_row_count=min_row_count,
            success=True,
            stage="skipped_non_trading_weekend",
            message="Weekend date: skipped THS health check and export rerun.",
        )
        _write_source_health(
            trade_date=trade_date,
            source_status=_source_status_payload(
                trade_date=trade_date,
                status="skipped",
                source_path=None,
                manifest_path=manifest_path_str,
                row_count=None,
                message="Weekend date: skipped THS health check and export rerun.",
                error=None,
            ),
            source_health_path=source_health_path,
        )
        if args.status_path:
            Path(args.status_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    if args.force_export:
        export_exit_code = _run_export_script(export_root, trade_date, min_row_count, force=True, allow_weekend=args.allow_weekend)
        if export_exit_code != 0:
            payload = _status_payload(
                trade_date=trade_date,
                export_root=export_root,
                min_row_count=min_row_count,
                success=False,
                stage="rerun_failed",
                message=f"ths export rerun failed: exit_code={export_exit_code}",
            )
            _write_source_health(
                trade_date=trade_date,
                source_status=_source_status_payload(
                    trade_date=trade_date,
                    status="failed",
                    source_path=normalized_path_str,
                    manifest_path=manifest_path_str,
                    row_count=None,
                    message=f"ths export rerun failed: exit_code={export_exit_code}",
                    error=f"ths export rerun failed: exit_code={export_exit_code}",
                ),
                source_health_path=source_health_path,
            )
            if args.status_path:
                Path(args.status_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(payload, ensure_ascii=False))
            return 1
        passed, message, source_path, row_count = _check_export(trade_date, export_root, min_row_count)
        payload = _status_payload(
            trade_date=trade_date,
            export_root=export_root,
            min_row_count=min_row_count,
            success=passed,
            stage="rerun_and_recheck",
            message=message,
            source_path=source_path,
        )
        _write_source_health(
            trade_date=trade_date,
            source_status=_source_status_payload(
                trade_date=trade_date,
                status="ok" if passed else "failed",
                source_path=source_path,
                manifest_path=manifest_path_str,
                row_count=row_count,
                message=message,
                error=message if not passed else None,
            ),
            source_health_path=source_health_path,
        )
        if args.status_path:
            Path(args.status_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0 if passed else 1

    passed, message, source_path, row_count = _check_export(trade_date, export_root, min_row_count)
    source_status = _source_status_payload(
        trade_date=trade_date,
        status="ok" if passed else "failed",
        source_path=source_path if passed else normalized_path_str,
        manifest_path=manifest_path_str,
        row_count=row_count,
        message=message,
        error=message if not passed else None,
    )
    _write_source_health(
        trade_date=trade_date,
        source_status=source_status,
        source_health_path=source_health_path,
    )

    if passed:
        payload = _status_payload(
            trade_date=trade_date,
            export_root=export_root,
            min_row_count=min_row_count,
            success=True,
            stage="precheck_ok",
            source_path=source_path,
        )
        if args.status_path:
            Path(args.status_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    if args.skip_rerun:
        payload = _status_payload(
            trade_date=trade_date,
            export_root=export_root,
            min_row_count=min_row_count,
            success=False,
            stage="precheck_failed",
            message=message,
            source_path=source_path,
        )
        if args.status_path:
            Path(args.status_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    export_exit_code = _run_export_script(export_root, trade_date, min_row_count, force=True, allow_weekend=args.allow_weekend)
    if export_exit_code != 0:
        payload = _status_payload(
            trade_date=trade_date,
            export_root=export_root,
            min_row_count=min_row_count,
            success=False,
            stage="rerun_failed",
            message=f"ths export rerun failed: exit_code={export_exit_code}; source={message}",
            source_path=source_path,
        )
        _write_source_health(
            trade_date=trade_date,
            source_status=_source_status_payload(
                trade_date=trade_date,
                status="failed",
                source_path=source_path,
                manifest_path=manifest_path_str,
                row_count=row_count,
                message=f"ths export rerun failed: exit_code={export_exit_code}; source={message}",
                error=f"ths export rerun failed: exit_code={export_exit_code}",
            ),
            source_health_path=source_health_path,
        )
        if args.status_path:
            Path(args.status_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return 1

    passed, recheck_message, recheck_source_path, recheck_row_count = _check_export(trade_date, export_root, min_row_count)
    payload = _status_payload(
        trade_date=trade_date,
        export_root=export_root,
        min_row_count=min_row_count,
        success=passed,
        stage="rerun_and_recheck",
        message=recheck_message if not passed else None,
        source_path=recheck_source_path,
    )
    _write_source_health(
        trade_date=trade_date,
        source_status=_source_status_payload(
            trade_date=trade_date,
            status="ok" if passed else "failed",
            source_path=recheck_source_path,
            manifest_path=manifest_path_str,
            row_count=recheck_row_count,
            message=recheck_message,
            error=recheck_message if not passed else None,
        ),
        source_health_path=source_health_path,
    )
    if args.status_path:
        Path(args.status_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
