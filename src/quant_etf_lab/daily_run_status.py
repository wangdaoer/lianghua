"""Read-only status summary for scheduled daily research refresh runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, time
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from .market_data_source import (
    DEFAULT_DAILY_MARKET_DATA_DIR,
    DEFAULT_EXCHANGE_INGEST_DIR,
    load_market_snapshot_rows,
)


@dataclass(frozen=True)
class DailyRunStatusResult:
    output_dir: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]


PROBLEM_RUN_STATES = {
    "no_refresh_status",
    "refresh_failed",
    "live_preflight_failed",
    "live_preflight_blocked",
    "observation_missing",
    "observation_failed",
    "lock_conflict",
    "wrapper_failed",
    "daily_run_status_failed",
    "observation_skipped",
    "daily_pipeline_preflight_smoke_failed",
}
WAITING_RUN_STATES = {
    "waiting_for_scheduled_run",
    "waiting_for_outcome_samples",
}


def _resolve(project_root: Path, path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else project_root / raw


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, JSONDecodeError):
        return None


def _resolve_path_if_present(project_root: Path, path: Any) -> Path | None:
    if path in (None, ""):
        return None
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else project_root / candidate


def _latest_file(root: Path, pattern: str, exclude_prefixes: tuple[str, ...] = ()) -> Path | None:
    if not root.exists():
        return None
    files = [
        path
        for path in root.glob(pattern)
        if path.is_file() and not any(path.name.startswith(prefix) for prefix in exclude_prefixes)
    ]
    if not files:
        return None
    return max(files, key=lambda path: (path.stat().st_mtime, path.name))


def _latest_observation_snapshot(project_root: Path, research_dir: Path, observation_status: dict[str, Any] | None) -> Path | None:
    if observation_status:
        observation_dir = observation_status.get("observation_dir")
        if observation_dir:
            snapshot_path = _resolve(project_root, observation_dir) / "allocator_observation_snapshot.json"
            if snapshot_path.exists():
                return snapshot_path

    if not research_dir.exists():
        return None
    snapshots: list[Path] = []
    for directory in research_dir.glob("allocator_observation*"):
        snapshot_path = directory / "allocator_observation_snapshot.json"
        if snapshot_path.exists():
            snapshots.append(snapshot_path)
    if not snapshots:
        return None
    return max(snapshots, key=lambda path: (path.stat().st_mtime, path.as_posix()))


def _last_nonempty_line(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            buffer = b""
            while position > 0:
                read_size = min(4096, position)
                position -= read_size
                handle.seek(position)
                buffer = handle.read(read_size) + buffer
                lines = buffer.splitlines()
                if len(lines) > 1 or position == 0:
                    for raw in reversed(lines):
                        if raw.strip():
                            return raw.decode("utf-8-sig", errors="ignore")
            return None
    except OSError:
        return None


def _stock_cache_summary(data_cache_dir: Path) -> dict[str, Any]:
    if _is_daily_market_data_hub(data_cache_dir):
        try:
            result = load_market_snapshot_rows(
                daily_data_dir=data_cache_dir,
                ingest_project_dir=DEFAULT_EXCHANGE_INGEST_DIR,
                require_success=True,
            )
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
            return {
                "latest_stock_cache_date": None,
                "stock_cache_file_count": 0,
                "stock_cache_source_kind": "daily_market_data_unavailable",
                "stock_cache_source_path": str(data_cache_dir),
                "stock_cache_snapshot_row_count": 0,
                "stock_cache_fetch_status": "error",
                "stock_cache_error": str(error),
            }
        fetch_status = result.fetch_status
        if result.rows and result.trade_date:
            return {
                "latest_stock_cache_date": result.trade_date,
                "stock_cache_file_count": 1,
                "stock_cache_source_kind": result.source_kind,
                "stock_cache_source_path": str(result.source_path) if result.source_path else str(data_cache_dir),
                "stock_cache_snapshot_row_count": len(result.rows),
                "stock_cache_fetch_status": getattr(fetch_status, "status", ""),
                "stock_cache_fetch_run_id": getattr(fetch_status, "run_id", ""),
                "stock_cache_fetch_run_time": getattr(fetch_status, "run_time", ""),
            }
        return {
            "latest_stock_cache_date": None,
            "stock_cache_file_count": 0,
            "stock_cache_source_kind": result.source_kind or "daily_market_data_empty",
            "stock_cache_source_path": str(result.source_path) if result.source_path else str(data_cache_dir),
            "stock_cache_snapshot_row_count": 0,
            "stock_cache_fetch_status": getattr(fetch_status, "status", ""),
            "stock_cache_fetch_run_id": getattr(fetch_status, "run_id", ""),
            "stock_cache_fetch_run_time": getattr(fetch_status, "run_time", ""),
        }

    if not data_cache_dir.exists():
        return {
            "latest_stock_cache_date": None,
            "stock_cache_file_count": 0,
            "stock_cache_source_kind": "legacy_stock_csv_missing",
            "stock_cache_source_path": str(data_cache_dir),
            "stock_cache_snapshot_row_count": 0,
        }

    latest: str | None = None
    file_count = 0
    for path in data_cache_dir.glob("*.csv"):
        file_count += 1
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                header = handle.readline().strip()
        except OSError:
            continue
        if not header:
            continue
        try:
            date_index = next(csv.reader([header])).index("date")
        except (ValueError, csv.Error):
            continue
        last_line = _last_nonempty_line(path)
        if not last_line or last_line == header:
            continue
        try:
            row = next(csv.reader([last_line]))
        except csv.Error:
            continue
        if date_index >= len(row):
            continue
        candidate = row[date_index].strip()
        if candidate and (latest is None or candidate > latest):
            latest = candidate
    return {
        "latest_stock_cache_date": latest,
        "stock_cache_file_count": file_count,
        "stock_cache_source_kind": "legacy_stock_csv",
        "stock_cache_source_path": str(data_cache_dir),
        "stock_cache_snapshot_row_count": 0,
    }


def _is_daily_market_data_hub(path: Path) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    default_normalized = str(DEFAULT_DAILY_MARKET_DATA_DIR).replace("\\", "/").lower()
    return (
        path.name == "daily-market-data"
        or normalized == default_normalized
        or (path / "snapshots").exists()
        or (path / "sqlite").exists()
    )


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "y", "yes", "on"}
    return False


def _parse_hhmm(value: str) -> time | None:
    try:
        hour_text, minute_text = str(value).split(":", 1)
        return time(int(hour_text), int(minute_text))
    except (TypeError, ValueError):
        return None


def _run_state(
    wrapper_status: dict[str, Any] | None,
    refresh_status: dict[str, Any] | None,
    observation_status: dict[str, Any] | None,
    observation_snapshot: dict[str, Any] | None,
    live_preflight_snapshot: dict[str, Any] | None,
    skip_live_preflight: bool,
    current_dt: datetime,
    scheduled_run_time: str,
) -> str:
    wrapper_stage = str((wrapper_status or {}).get("final_stage") or "")
    if wrapper_stage in {
        "lock_conflict",
        "wrapper_failed",
        "refresh_failed",
        "observation_failed",
        "live_preflight_failed",
        "live_preflight_blocked",
        "daily_run_status_failed",
        "observation_skipped",
        "daily_pipeline_preflight_smoke_failed",
    }:
        return wrapper_stage
    wrapper_exit_code = _as_int((wrapper_status or {}).get("exit_code"))
    if wrapper_status is not None and wrapper_exit_code not in (None, 0):
        return "wrapper_failed"
    if refresh_status is None:
        return "no_refresh_status"
    refresh_exit_code = _as_int(refresh_status.get("exit_code"))
    if refresh_exit_code != 0:
        return "refresh_failed"
    if observation_status is None:
        next_review_date = str((observation_snapshot or {}).get("paper_stock_target_review_outcome_calendar_next_action_date") or "")
        scheduled_time = _parse_hhmm(scheduled_run_time)
        if next_review_date == current_dt.date().isoformat() and scheduled_time is not None and current_dt.time() < scheduled_time:
            return "waiting_for_scheduled_run"
        return "observation_missing"
    observation_exit_code = _as_int(observation_status.get("exit_code"))
    if observation_exit_code != 0:
        return "observation_failed"
    if not skip_live_preflight and live_preflight_snapshot is not None and str(
        (live_preflight_snapshot.get("decision") or "")
    ).strip().lower() == "blocked":
        return "live_preflight_blocked"
    ready_count = _as_int((observation_snapshot or {}).get("outcome_ready_horizon_count")) or 0
    if ready_count > 0:
        return "outcome_ready"
    return "waiting_for_outcome_samples"


def _run_state_severity(run_state: str) -> str:
    if run_state in PROBLEM_RUN_STATES:
        return "problem"
    if run_state in WAITING_RUN_STATES:
        return "waiting"
    return "ok"


def _render_report(snapshot: dict[str, Any]) -> str:
    return f"""# Daily Research Run Status

- Generated at: {snapshot.get("generated_at")}
- Run state: `{snapshot.get("run_state")}`
- Run state severity: `{snapshot.get("run_state_severity")}`
- Problem state: `{snapshot.get("problem_state")}`
- Wrapper final stage: `{snapshot.get("latest_wrapper_final_stage")}`
- Wrapper exit code: `{snapshot.get("latest_wrapper_exit_code")}`
- Latest refresh exit code: `{snapshot.get("latest_refresh_exit_code")}`
- Pipeline preflight smoke exit code: `{snapshot.get("latest_pipeline_preflight_smoke_exit_code")}`
- Latest live preflight exit code: `{snapshot.get("latest_live_preflight_exit_code")}`
- Latest observation exit code: `{snapshot.get("latest_observation_exit_code")}`
- Live preflight status: `{snapshot.get("latest_live_preflight_status")}`
- Live preflight decision: `{snapshot.get("latest_live_preflight_decision")}`
- Live preflight blocker count: `{snapshot.get("latest_live_preflight_blocking_items_count")}`
- Live preflight monitor count: `{snapshot.get("latest_live_preflight_monitor_items_count")}`
- Live-shadow review decisions: `{snapshot.get("latest_live_preflight_live_shadow_review_decision_count")}`
- Live-shadow review blocking / monitor: `{snapshot.get("latest_live_preflight_live_shadow_review_blocking_decision_count")}` / `{snapshot.get("latest_live_preflight_live_shadow_review_monitor_decision_count")}`
- Live preflight broker connection: `{snapshot.get("latest_live_preflight_broker_connection_status")}`
- Observation as-of date: `{snapshot.get("observation_as_of_date")}`
- Outcome ready horizons: `{snapshot.get("outcome_ready_horizon_count")}`
- Outcome analysis status: `{snapshot.get("outcome_analysis_status")}`
- Next review: `{snapshot.get("next_review_date")}` / `{snapshot.get("next_review_horizon")}`
- Next 1D maturity date: `{snapshot.get("next_1d_maturity_date")}`
- Risk budget decision: `{snapshot.get("risk_budget_decision")}`
- Latest stock cache date: `{snapshot.get("latest_stock_cache_date")}` ({snapshot.get("stock_cache_file_count")} files)
- Stock cache source: `{snapshot.get("stock_cache_source_kind")}`
- Stock cache source path: `{snapshot.get("stock_cache_source_path")}`
- Stock cache fetch status: `{snapshot.get("stock_cache_fetch_status")}`
- Stock cache snapshot rows: `{snapshot.get("stock_cache_snapshot_row_count")}`

## Paths

- Refresh status: `{snapshot.get("latest_refresh_status_path")}`
- Wrapper status: `{snapshot.get("latest_wrapper_status_path")}`
- Observation status: `{snapshot.get("latest_observation_status_path")}`
- Observation snapshot: `{snapshot.get("latest_observation_snapshot_path")}`
- Pipeline snapshot: `{snapshot.get("latest_pipeline_snapshot_path")}`
- Pipeline preflight smoke status: `{snapshot.get("latest_pipeline_preflight_smoke_status_path")}`
- Live preflight status: `{snapshot.get("latest_live_preflight_status_path")}`
- Live preflight snapshot: `{snapshot.get("latest_live_preflight_snapshot_path")}`
- Live preflight report: `{snapshot.get("latest_live_preflight_report_path")}`
- Live preflight output dir: `{snapshot.get("latest_live_preflight_output_dir")}`
"""


def run_daily_run_status(
    project_root: str | Path = ".",
    output_dir: str | Path = "outputs/research/daily_run_status_latest",
    logs_dir: str | Path = "outputs/logs",
    research_dir: str | Path = "outputs/research",
    data_cache_dir: str | Path = DEFAULT_DAILY_MARKET_DATA_DIR,
    scheduled_run_time: str = "16:10",
    current_dt: datetime | None = None,
) -> DailyRunStatusResult:
    root = Path(project_root).resolve()
    resolved_output = _resolve(root, output_dir)
    resolved_logs = _resolve(root, logs_dir)
    resolved_research = _resolve(root, research_dir)
    resolved_cache = _resolve(root, data_cache_dir)

    refresh_status_path = _latest_file(
        resolved_logs,
        "daily_research_refresh_*.status.json",
        exclude_prefixes=("daily_research_refresh_with_observation_",),
    )
    wrapper_status_path = _latest_file(resolved_logs, "daily_research_refresh_with_observation_*.status.json")
    observation_status_path = _latest_file(resolved_logs, "allocator_observation_*.status.json")
    pipeline_preflight_smoke_status_path: Path | None = None
    live_preflight_status_path: Path | None = None
    refresh_status = _read_json(refresh_status_path)
    wrapper_status = _read_json(wrapper_status_path)
    skip_live_preflight = _as_bool((wrapper_status or {}).get("skip_live_preflight"))
    observation_status = _read_json(observation_status_path)
    wrapper_live_preflight_status_path = None
    if wrapper_status:
        wrapper_live_preflight_status_path = _resolve_path_if_present(
            root,
            wrapper_status.get("live_preflight_status_path"),
        )
    if wrapper_live_preflight_status_path is not None:
        live_preflight_status_path = wrapper_live_preflight_status_path
    elif not skip_live_preflight:
        live_preflight_status_path = _latest_file(resolved_logs, "live_preflight_*.status.json")
    if wrapper_status and wrapper_status.get("pipeline_preflight_smoke_status_path"):
        pipeline_preflight_smoke_status_path = _resolve_path_if_present(
            root,
            wrapper_status.get("pipeline_preflight_smoke_status_path"),
        )
    live_preflight_status = _read_json(live_preflight_status_path)
    live_preflight_output_dir: Path | None = None
    live_preflight_snapshot_path: Path | None = None
    live_preflight_report_path: Path | None = None
    if live_preflight_status is not None:
        live_preflight_output_dir = _resolve_path_if_present(root, live_preflight_status.get("output_dir"))
        live_preflight_snapshot_path = _resolve_path_if_present(root, live_preflight_status.get("snapshot_path"))
        live_preflight_report_path = _resolve_path_if_present(root, live_preflight_status.get("report_path"))
    if live_preflight_output_dir is None and live_preflight_status_path is not None:
        live_preflight_output_dir = live_preflight_status_path.parent
    if live_preflight_output_dir is not None:
        if live_preflight_snapshot_path is None:
            candidate_snapshot = live_preflight_output_dir / "live_preflight_snapshot.json"
            if candidate_snapshot.exists():
                live_preflight_snapshot_path = candidate_snapshot
        if live_preflight_report_path is None:
            candidate_report = live_preflight_output_dir / "live_preflight.md"
            if candidate_report.exists():
                live_preflight_report_path = candidate_report
    live_preflight_snapshot = _read_json(live_preflight_snapshot_path)
    observation_snapshot_path = _latest_observation_snapshot(root, resolved_research, observation_status)
    observation_snapshot = _read_json(observation_snapshot_path)
    stock_cache_summary = _stock_cache_summary(resolved_cache)
    generated_at = current_dt or datetime.now()
    run_state = _run_state(
        wrapper_status,
        refresh_status,
        observation_status,
        observation_snapshot,
        live_preflight_snapshot,
        skip_live_preflight=skip_live_preflight,
        current_dt=generated_at,
        scheduled_run_time=scheduled_run_time,
    )
    run_state_severity = _run_state_severity(run_state)

    snapshot: dict[str, Any] = {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "scheduled_run_time": scheduled_run_time,
        "run_state": run_state,
        "run_state_severity": run_state_severity,
        "problem_state": run_state_severity == "problem",
        "latest_wrapper_status_path": str(wrapper_status_path) if wrapper_status_path else None,
        "latest_wrapper_exit_code": _as_int((wrapper_status or {}).get("exit_code")),
        "latest_wrapper_final_stage": (wrapper_status or {}).get("final_stage"),
        "latest_wrapper_refresh_exit_code": _as_int((wrapper_status or {}).get("refresh_exit_code")),
        "latest_wrapper_observation_exit_code": _as_int((wrapper_status or {}).get("observation_exit_code")),
        "latest_wrapper_daily_run_status_exit_code": _as_int((wrapper_status or {}).get("daily_run_status_exit_code")),
        "skip_live_preflight": skip_live_preflight,
        "latest_wrapper_started_at": (wrapper_status or {}).get("started_at"),
        "latest_wrapper_finished_at": (wrapper_status or {}).get("finished_at"),
        "latest_wrapper_error": (wrapper_status or {}).get("wrapper_error"),
        "latest_pipeline_preflight_smoke_exit_code": _as_int(
            (wrapper_status or {}).get("pipeline_preflight_smoke_exit_code")
        ),
        "latest_pipeline_preflight_smoke_status_path": str(pipeline_preflight_smoke_status_path)
        if pipeline_preflight_smoke_status_path
        else None,
        "latest_live_preflight_status_path": str(live_preflight_status_path) if live_preflight_status_path else None,
        "latest_live_preflight_exit_code": _as_int((live_preflight_status or {}).get("exit_code")),
        "latest_live_preflight_started_at": (live_preflight_status or {}).get("started_at"),
        "latest_live_preflight_finished_at": (live_preflight_status or {}).get("finished_at"),
        "latest_live_preflight_stdout_log": (live_preflight_status or {}).get("stdout_log"),
        "latest_live_preflight_stderr_log": (live_preflight_status or {}).get("stderr_log"),
        "latest_live_preflight_output_dir": str(live_preflight_output_dir) if live_preflight_output_dir else None,
        "latest_live_preflight_snapshot_path": str(live_preflight_snapshot_path) if live_preflight_snapshot_path else None,
        "latest_live_preflight_report_path": str(live_preflight_report_path) if live_preflight_report_path else None,
        "latest_live_preflight_command": (live_preflight_status or {}).get("command"),
        "latest_live_preflight_status": (live_preflight_snapshot or {}).get("status"),
        "latest_live_preflight_decision": (live_preflight_snapshot or {}).get("decision"),
        "latest_live_preflight_blocking_items_count": len((live_preflight_snapshot or {}).get("blocking_items") or []),
        "latest_live_preflight_monitor_items_count": len((live_preflight_snapshot or {}).get("monitor_items") or []),
        "latest_live_preflight_broker_connection_status": (live_preflight_snapshot or {}).get("broker_connection_status"),
        "latest_live_preflight_active_target_count": (live_preflight_snapshot or {}).get("active_target_count"),
        "latest_live_preflight_stock_target_review_action_count": (live_preflight_snapshot or {}).get("stock_target_review_action_count"),
        "latest_live_preflight_live_shadow_review_decisions_path": (live_preflight_snapshot or {}).get(
            "live_shadow_review_decisions_path"
        ),
        "latest_live_preflight_live_shadow_review_decision_count": (live_preflight_snapshot or {}).get(
            "live_shadow_review_decision_count"
        ),
        "latest_live_preflight_live_shadow_review_blocking_decision_count": (live_preflight_snapshot or {}).get(
            "live_shadow_review_blocking_decision_count"
        ),
        "latest_live_preflight_live_shadow_review_monitor_decision_count": (live_preflight_snapshot or {}).get(
            "live_shadow_review_monitor_decision_count"
        ),
        "latest_live_preflight_live_shadow_review_unknown_decision_count": (live_preflight_snapshot or {}).get(
            "live_shadow_review_unknown_decision_count"
        ),
        "latest_refresh_status_path": str(refresh_status_path) if refresh_status_path else None,
        "latest_refresh_exit_code": _as_int((refresh_status or {}).get("exit_code")),
        "latest_refresh_started_at": (refresh_status or {}).get("started_at"),
        "latest_refresh_finished_at": (refresh_status or {}).get("finished_at"),
        "latest_refresh_stdout_log": (refresh_status or {}).get("stdout_log"),
        "latest_refresh_stderr_log": (refresh_status or {}).get("stderr_log"),
        "latest_observation_status_path": str(observation_status_path) if observation_status_path else None,
        "latest_observation_exit_code": _as_int((observation_status or {}).get("exit_code")),
        "latest_observation_started_at": (observation_status or {}).get("started_at"),
        "latest_observation_finished_at": (observation_status or {}).get("finished_at"),
        "latest_observation_snapshot_path": str(observation_snapshot_path) if observation_snapshot_path else None,
        "latest_pipeline_snapshot_path": (observation_status or {}).get("pipeline_snapshot")
        or (observation_snapshot or {}).get("pipeline_snapshot"),
        "observation_as_of_date": (observation_snapshot or {}).get("as_of_date"),
        "observation_status": (observation_snapshot or {}).get("status"),
        "next_action_stage": (observation_snapshot or {}).get("next_action_stage"),
        "risk_budget_decision": (observation_snapshot or {}).get("risk_budget_decision"),
        "outcome_ready_horizon_count": _as_int((observation_snapshot or {}).get("outcome_ready_horizon_count")) or 0,
        "outcome_analysis_status": (observation_snapshot or {}).get("outcome_analysis_status"),
        "next_review_date": (observation_snapshot or {}).get("paper_stock_target_review_outcome_calendar_next_action_date"),
        "next_review_horizon": (observation_snapshot or {}).get("paper_stock_target_review_outcome_calendar_next_action_horizon"),
        "next_1d_maturity_date": (observation_snapshot or {}).get("paper_stock_target_review_outcome_maturity_next_1d_date"),
    }
    snapshot.update(stock_cache_summary)

    resolved_output.mkdir(parents=True, exist_ok=True)
    snapshot_path = resolved_output / "daily_run_status_snapshot.json"
    report_path = resolved_output / "daily_run_status.md"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot), encoding="utf-8")
    return DailyRunStatusResult(
        output_dir=resolved_output,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
    )
