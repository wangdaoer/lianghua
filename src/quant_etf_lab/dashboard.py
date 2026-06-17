"""Unified local dashboard for daily A-share research review."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .market_data_source import DEFAULT_DAILY_MARKET_DATA_DIR, load_market_snapshot_rows


DEFAULT_DASHBOARD_OUTPUT = Path("outputs/research/latest_dashboard")
DEFAULT_DASHBOARD_BASE = Path("outputs/research/daily_dashboard")
DEFAULT_DAILY_CHECK_DIR = Path("outputs/research/daily_model_check_latest")
DEFAULT_PAPER_ACCOUNT_DIR = Path("outputs/research/paper_account_latest")
DEFAULT_SENTIMENT_DIR = Path("outputs/research/market_sentiment_state_latest")
DEFAULT_DATA_CACHE_DIR = DEFAULT_DAILY_MARKET_DATA_DIR
DEFAULT_ALLOCATOR_DIR = Path("outputs/portfolio_source_selection/main_chinext_portfolio_source_selection_validation6_v1")
DEFAULT_TRIGGER_REPORT = Path("D:/codex/outputs/trigger_reports/latest_trigger.md")
DEFAULT_MODEL_AUDIT_DIR = Path("outputs/research/model_build_audit_latest")
DEFAULT_PIPELINE_HISTORY_DIR = Path("outputs/research/pipeline_history_review_latest")
DEFAULT_DAILY_RUN_STATUS_DIR = Path("outputs/research/daily_run_status_latest")
DEFAULT_ALLOCATOR_OBSERVATION_DIR = Path("outputs/research/allocator_observation_latest")


@dataclass(frozen=True)
class DailyDashboardResult:
    output_dir: Path
    report_path: Path
    snapshot_path: Path
    snapshot: dict[str, Any]


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = Path(path) if path is not None else default
    return raw if raw.is_absolute() else project_root / raw


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) == 8 and text.isdigit():
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        if pd.isna(parsed):
            return None
        return pd.Timestamp(parsed).date()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).date()


def _resolve_output(project_root: Path, output_dir: str | Path | None, as_of: date, date_stamp: bool) -> Path:
    default = DEFAULT_DASHBOARD_BASE if date_stamp else DEFAULT_DASHBOARD_OUTPUT
    raw = Path(output_dir) if output_dir is not None else default
    if date_stamp:
        stamp = as_of.strftime("%Y%m%d")
        if not raw.name.endswith(f"_{stamp}"):
            raw = raw.with_name(f"{raw.name}_{stamp}")
    return raw if raw.is_absolute() else project_root / raw


def _latest_dir(base: Path, prefix: str, as_of: date | None = None) -> Path:
    if base.exists():
        return base
    parent = base.parent
    if not parent.exists():
        return base
    if as_of is not None:
        canonical = parent / f"{prefix}_{as_of.strftime('%Y%m%d')}"
        if canonical.exists():
            return canonical
    matches = [path for path in parent.glob(f"{prefix}_*") if path.is_dir()]
    if not matches:
        return base
    canonical_matches = [path for path in matches if re.fullmatch(rf"{re.escape(prefix)}_\d{{8}}", path.name)]
    if canonical_matches:
        return max(canonical_matches, key=lambda path: path.stat().st_mtime)
    return max(matches, key=lambda path: path.stat().st_mtime)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _read_text_preview(path: Path, max_lines: int = 8) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    preview: list[str] = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        preview.append(cleaned)
        if len(preview) >= max_lines:
            break
    return preview


def _last_csv_date(path: Path) -> date | None:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 4096))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(text.splitlines()):
        cleaned = line.strip()
        if not cleaned or cleaned.lower().startswith("date,"):
            continue
        return _parse_date(cleaned.split(",", 1)[0])
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not pd.notna(number):
        return "N/A"
    return f"{number * 100:.2f}%"


def _pct_points(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not pd.notna(number):
        return "N/A"
    return f"{number:.2f}%"


def _num(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not pd.notna(number):
        return "N/A"
    return f"{number:.{digits}f}"


def _freshness(latest: date | None, as_of: date, max_staleness_days: int) -> tuple[str, int | None]:
    if latest is None:
        return "missing", None
    days = (as_of - latest).days
    if days < 0:
        return "future_dated", days
    if days <= max_staleness_days:
        return "fresh_enough", days
    return "stale", days


def _is_daily_market_data_hub(path: Path) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    default_normalized = str(DEFAULT_DAILY_MARKET_DATA_DIR).replace("\\", "/").lower()
    return (
        path.name == "daily-market-data"
        or normalized == default_normalized
        or (path / "snapshots").exists()
        or (path / "sqlite").exists()
    )


def _summarize_daily_market_data_hub(
    cache_dir: Path,
    as_of: date,
    max_staleness_days: int,
) -> dict[str, Any]:
    try:
        result = load_market_snapshot_rows(daily_data_dir=cache_dir, require_success=True)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        return {
            "market_cache_status": "missing",
            "market_cache_dir": str(cache_dir),
            "market_cache_source_kind": "daily_market_data_unavailable",
            "market_cache_error": str(error),
            "market_cache_file_count": 0,
            "market_cache_dated_file_count": 0,
        }
    latest_date = _parse_date(result.trade_date)
    if latest_date is None or not result.rows:
        return {
            "market_cache_status": "missing",
            "market_cache_dir": str(cache_dir),
            "market_cache_source_kind": result.source_kind,
            "market_cache_source_path": str(result.source_path) if result.source_path else "",
            "market_cache_file_count": 0,
            "market_cache_dated_file_count": 0,
        }
    freshness, days = _freshness(latest_date, as_of, max_staleness_days)
    fetch_status = result.fetch_status
    return {
        "market_cache_status": freshness,
        "market_cache_dir": str(cache_dir),
        "market_cache_source_kind": result.source_kind,
        "market_cache_source_path": str(result.source_path) if result.source_path else "",
        "market_cache_latest_date": latest_date.isoformat(),
        "market_cache_oldest_date": latest_date.isoformat(),
        "market_cache_days_since_latest": days,
        "market_cache_file_count": 1,
        "market_cache_dated_file_count": 1,
        "market_cache_fresh_file_count": 1,
        "market_cache_fresh_ratio": 1.0,
        "market_cache_stale_examples": [],
        "market_cache_snapshot_row_count": len(result.rows),
        "market_cache_fetch_status": getattr(fetch_status, "status", ""),
        "market_cache_fetch_run_id": getattr(fetch_status, "run_id", ""),
    }


def _relative_or_absolute(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _summarize_market_cache(
    project_root: Path,
    data_cache_dir: str | Path | None,
    as_of: date,
    max_staleness_days: int,
    min_cache_fresh_ratio: float,
) -> dict[str, Any]:
    cache_dir = _resolve(project_root, data_cache_dir, DEFAULT_DATA_CACHE_DIR)
    if _is_daily_market_data_hub(cache_dir):
        return _summarize_daily_market_data_hub(cache_dir, as_of, max_staleness_days)
    csv_paths = sorted(cache_dir.rglob("*.csv")) if cache_dir.exists() else []
    dated_files: list[tuple[Path, date]] = []
    for path in csv_paths:
        latest = _last_csv_date(path)
        if latest is not None:
            dated_files.append((path, latest))
    if not dated_files:
        return {
            "market_cache_status": "missing",
            "market_cache_dir": str(cache_dir),
            "market_cache_file_count": len(csv_paths),
            "market_cache_dated_file_count": 0,
        }

    latest_date = max(item_date for _, item_date in dated_files)
    oldest_date = min(item_date for _, item_date in dated_files)
    freshness, days = _freshness(latest_date, as_of, max_staleness_days)
    fresh_count = sum(item_date == latest_date for _, item_date in dated_files)
    fresh_ratio = fresh_count / len(dated_files)
    if freshness != "fresh_enough":
        status = freshness
    elif fresh_ratio >= min_cache_fresh_ratio:
        status = "fresh_enough"
    else:
        status = "partial_stale"
    stale_examples = [
        _relative_or_absolute(project_root, path)
        for path, item_date in dated_files
        if item_date != latest_date
    ][:5]
    return {
        "market_cache_status": status,
        "market_cache_dir": str(cache_dir),
        "market_cache_latest_date": latest_date.isoformat(),
        "market_cache_oldest_date": oldest_date.isoformat(),
        "market_cache_days_since_latest": days,
        "market_cache_file_count": len(csv_paths),
        "market_cache_dated_file_count": len(dated_files),
        "market_cache_fresh_file_count": fresh_count,
        "market_cache_fresh_ratio": fresh_ratio,
        "market_cache_min_fresh_ratio": float(min_cache_fresh_ratio),
        "market_cache_stale_examples": stale_examples,
    }


def _summarize_allocator(
    project_root: Path,
    allocator_dir: str | Path | None,
    as_of: date,
    max_staleness_days: int,
) -> dict[str, Any]:
    resolved_dir = _resolve(project_root, allocator_dir, DEFAULT_ALLOCATOR_DIR)
    curve = _read_csv(resolved_dir / "oos_equity_stitched.csv")
    summary = _read_csv(resolved_dir / "portfolio_walk_forward_summary.csv")
    curve_latest = None
    if "date" in curve:
        curve_dates = [_parse_date(value) for value in curve["date"]]
        curve_dates = [value for value in curve_dates if value is not None]
        curve_latest = max(curve_dates) if curve_dates else None

    summary_latest = None
    selected_params_path: Path | None = None
    if "test_end" in summary:
        data = summary.copy()
        data["_sort_date"] = data["test_end"].map(_parse_date)
        data = data.dropna(subset=["_sort_date"]).sort_values("_sort_date")
        if not data.empty:
            latest_row = data.iloc[-1]
            summary_latest = pd.Timestamp(latest_row["_sort_date"]).date()
            raw_path = str(latest_row.get("selected_params_path", "")).strip()
            if raw_path:
                selected_params_path = Path(raw_path)
                if not selected_params_path.is_absolute():
                    selected_params_path = project_root / selected_params_path if raw_path.startswith("outputs") else resolved_dir / selected_params_path

    curve_freshness, curve_days = _freshness(curve_latest, as_of, max_staleness_days)
    summary_freshness, summary_days = _freshness(summary_latest, as_of, max_staleness_days)
    selected_params_exists = bool(selected_params_path and selected_params_path.exists())
    if curve.empty and summary.empty:
        status = "missing"
    elif curve_freshness == "fresh_enough" and summary_freshness == "fresh_enough" and selected_params_exists:
        status = "fresh_enough"
    elif not selected_params_exists:
        status = "missing_selected_params"
    else:
        status = "stale"
    return {
        "allocator_input_status": status,
        "allocator_dir": str(resolved_dir),
        "allocator_curve_latest_date": curve_latest.isoformat() if curve_latest else None,
        "allocator_curve_days_since_latest": curve_days,
        "allocator_curve_freshness_status": curve_freshness,
        "allocator_summary_latest_test_end": summary_latest.isoformat() if summary_latest else None,
        "allocator_summary_days_since_latest": summary_days,
        "allocator_summary_freshness_status": summary_freshness,
        "allocator_selected_params_path": str(selected_params_path) if selected_params_path else None,
        "allocator_selected_params_exists": selected_params_exists,
    }


def _trigger_date_from_text(path: Path, preview: list[str]) -> date | None:
    candidates = [path.name, *preview]
    pattern = re.compile(r"(20\d{2}-\d{2}-\d{2})[_ ]?(\d{6})?")
    for text in candidates:
        match = pattern.search(text)
        if match:
            return _parse_date(match.group(1))
    return None


def _summarize_daily_check(project_root: Path, daily_check_dir: str | Path | None, as_of: date) -> dict[str, Any]:
    raw_dir = _resolve(project_root, daily_check_dir, DEFAULT_DAILY_CHECK_DIR)
    resolved_dir = _latest_dir(raw_dir, "daily_model_check", as_of)
    snapshot_path = resolved_dir / "daily_model_check_snapshot.json"
    report_path = resolved_dir / "daily_model_check.md"
    snapshot = _read_json(snapshot_path)
    return {
        "daily_check_status": "ok" if snapshot else "missing",
        "daily_check_dir": str(resolved_dir),
        "daily_check_snapshot_path": str(snapshot_path),
        "daily_check_report_path": str(report_path),
        "daily_check_generated_at": snapshot.get("generated_at"),
        "daily_as_of_date": snapshot.get("as_of_date"),
        "local_equity_latest_date": snapshot.get("latest_date"),
        "data_freshness_status": snapshot.get("data_freshness_status", "missing"),
        "action_posture": snapshot.get("action_posture", "missing"),
        "phase2_posture": snapshot.get("phase2_posture", "missing"),
        "selected_candidate": snapshot.get("selected_candidate"),
        "risk_on_satellite_weight": snapshot.get("risk_on_satellite_weight"),
        "current_drawdown": snapshot.get("current_drawdown"),
        "return_20d": snapshot.get("return_20d"),
        "allocator_sharpe": snapshot.get("allocator_sharpe"),
        "promotion_status": snapshot.get("promotion_status", "not_configured"),
        "promotion_decision": snapshot.get("promotion_decision"),
        "promotion_candidate_passes_headline_gate": snapshot.get("promotion_candidate_passes_headline_gate"),
        "promotion_sensitivity_support_count": snapshot.get("promotion_sensitivity_support_count"),
        "promotion_sensitivity_group_support_count": snapshot.get("promotion_sensitivity_group_support_count"),
        "promotion_support_group_count": snapshot.get("promotion_support_group_count"),
        "promotion_evidence_support_count": snapshot.get("promotion_evidence_support_count"),
        "promotion_sensitivity_run_support_count": snapshot.get("promotion_sensitivity_run_support_count"),
        "promotion_min_sensitivity_support": snapshot.get("promotion_min_sensitivity_support"),
        "promotion_return_edge": snapshot.get("promotion_return_edge"),
        "promotion_sharpe_edge": snapshot.get("promotion_sharpe_edge"),
        "promotion_drawdown_change": snapshot.get("promotion_drawdown_change"),
        "promotion_report_path": snapshot.get("promotion_report_path"),
        "phase2_report_path": snapshot.get("phase2_report_path"),
        "phase2_snapshot_path": snapshot.get("phase2_snapshot_path"),
    }


def _summarize_paper_account(
    project_root: Path,
    paper_account_dir: str | Path | None,
    as_of: date,
    max_staleness_days: int,
) -> dict[str, Any]:
    raw_dir = _resolve(project_root, paper_account_dir, DEFAULT_PAPER_ACCOUNT_DIR)
    resolved_dir = _latest_dir(raw_dir, "paper_account", as_of)
    metrics_path = resolved_dir / "metrics.json"
    report_path = resolved_dir / "paper_account.md"
    ledger_path = resolved_dir / "ledger.csv"
    audit_path = resolved_dir / "rebalance_audit.csv"
    metrics = _read_json(metrics_path)
    latest_date = _parse_date(metrics.get("latest_date"))
    freshness, days = _freshness(latest_date, as_of, max_staleness_days)
    return {
        "paper_account_status": "ok" if metrics else "missing",
        "paper_account_dir": str(resolved_dir),
        "paper_account_metrics_path": str(metrics_path),
        "paper_account_report_path": str(report_path),
        "paper_account_ledger_path": str(ledger_path),
        "paper_account_audit_path": str(audit_path),
        "paper_account_generated_at": metrics.get("generated_at"),
        "paper_latest_date": latest_date.isoformat() if latest_date else None,
        "paper_days_since_latest": days,
        "paper_freshness_status": freshness,
        "paper_latest_window": metrics.get("latest_window"),
        "paper_latest_candidate": metrics.get("latest_candidate"),
        "paper_latest_regime": metrics.get("latest_regime"),
        "paper_latest_core_weight": metrics.get("latest_core_weight"),
        "paper_latest_satellite_weight": metrics.get("latest_satellite_weight"),
        "paper_latest_cash_weight": metrics.get("latest_cash_weight"),
        "paper_final_equity": metrics.get("final_equity"),
        "paper_total_return": metrics.get("total_return"),
        "paper_cagr": metrics.get("cagr"),
        "paper_max_drawdown": metrics.get("max_drawdown"),
        "paper_current_drawdown": metrics.get("current_drawdown"),
        "paper_sharpe": metrics.get("sharpe"),
        "paper_average_satellite_weight": metrics.get("average_satellite_weight"),
        "paper_satellite_active_day_ratio": metrics.get("satellite_active_day_ratio"),
        "paper_total_estimated_fee": metrics.get("total_estimated_fee"),
        "paper_audit_event_count": metrics.get("audit_event_count"),
    }


def _summarize_sentiment(project_root: Path, sentiment_dir: str | Path | None, as_of: date, max_staleness_days: int) -> dict[str, Any]:
    raw_dir = _resolve(project_root, sentiment_dir, DEFAULT_SENTIMENT_DIR)
    resolved_dir = _latest_dir(raw_dir, "market_sentiment_state", as_of)
    snapshot_path = resolved_dir / "latest_market_sentiment.json"
    report_path = resolved_dir / "latest_market_sentiment.md"
    snapshot = _read_json(snapshot_path)
    latest = _parse_date(snapshot.get("date"))
    freshness, days = _freshness(latest, as_of, max_staleness_days)
    return {
        "sentiment_status": "ok" if snapshot else "missing",
        "sentiment_dir": str(resolved_dir),
        "sentiment_snapshot_path": str(snapshot_path),
        "sentiment_report_path": str(report_path),
        "sentiment_date": latest.isoformat() if latest else None,
        "sentiment_days_since_latest": days,
        "sentiment_freshness_status": freshness,
        "sentiment_state": snapshot.get("sentiment_state", "missing"),
        "sentiment_score": snapshot.get("sentiment_score"),
        "reference_exposure": snapshot.get("reference_exposure"),
        "coverage_count": snapshot.get("coverage_count"),
        "advance_ratio": snapshot.get("advance_ratio"),
        "limit_up_count": snapshot.get("limit_up_count"),
        "limit_down_count": snapshot.get("limit_down_count"),
        "market_return": snapshot.get("market_return"),
    }


def _summarize_trigger(trigger_report: str | Path | None, as_of: date, max_staleness_days: int) -> dict[str, Any]:
    path = Path(trigger_report) if trigger_report is not None else DEFAULT_TRIGGER_REPORT
    preview = _read_text_preview(path)
    report_date = _trigger_date_from_text(path, preview)
    freshness, days = _freshness(report_date, as_of, max_staleness_days)
    return {
        "trigger_status": "ok" if path.exists() else "missing",
        "trigger_report_path": str(path),
        "trigger_report_date": report_date.isoformat() if report_date else None,
        "trigger_days_since_latest": days,
        "trigger_freshness_status": freshness,
        "trigger_last_write_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.exists() else None,
        "trigger_report_size": path.stat().st_size if path.exists() else None,
        "trigger_preview": preview,
    }


def _summarize_model_audit(
    project_root: Path,
    model_audit_dir: str | Path | None,
    as_of: date,
    max_staleness_days: int,
) -> dict[str, Any]:
    raw_dir = _resolve(project_root, model_audit_dir, DEFAULT_MODEL_AUDIT_DIR)
    resolved_dir = _latest_dir(raw_dir, "model_build_audit", as_of)
    snapshot_path = resolved_dir / "model_build_audit_snapshot.json"
    report_path = resolved_dir / "model_build_audit.md"
    actions_path = resolved_dir / "walk_forward_run_actions.csv"
    config_map_path = resolved_dir / "config_inheritance_map.csv"
    snapshot = _read_json(snapshot_path)
    generated_date = _parse_date(snapshot.get("generated_at"))
    freshness, days = _freshness(generated_date, as_of, max_staleness_days)
    action_items = int(snapshot.get("walk_forward_action_items") or 0) if snapshot else 0
    duplicate_groups = int(snapshot.get("duplicate_config_groups") or 0) if snapshot else 0
    root_without_extends = int(snapshot.get("root_configs_without_extends") or 0) if snapshot else 0
    if not snapshot:
        status = "missing"
    elif freshness not in {"fresh_enough", "future_dated"}:
        status = freshness
    elif duplicate_groups or root_without_extends or action_items:
        status = "needs_attention"
    else:
        status = "ok"
    actions = _read_csv(actions_path)
    top_action = None
    if not actions.empty and "recommended_action" in actions:
        top_action = str(actions.iloc[0]["recommended_action"])
    return {
        "model_audit_status": status,
        "model_audit_dir": str(resolved_dir),
        "model_audit_snapshot_path": str(snapshot_path),
        "model_audit_report_path": str(report_path),
        "model_audit_actions_path": str(actions_path),
        "model_audit_config_map_path": str(config_map_path),
        "model_audit_generated_at": snapshot.get("generated_at"),
        "model_audit_days_since_generated": days,
        "model_audit_duplicate_config_groups": duplicate_groups if snapshot else None,
        "model_audit_root_configs_without_extends": root_without_extends if snapshot else None,
        "model_audit_walk_forward_action_items": action_items if snapshot else None,
        "model_audit_walk_forward_resume_candidates": snapshot.get("walk_forward_resume_candidates") if snapshot else None,
        "model_audit_walk_forward_archive_review_candidates": snapshot.get("walk_forward_archive_review_candidates") if snapshot else None,
        "model_audit_top_action": top_action,
    }


def _summarize_pipeline_history(
    project_root: Path,
    pipeline_history_dir: str | Path | None,
    as_of: date,
) -> dict[str, Any]:
    raw_dir = _resolve(project_root, pipeline_history_dir, DEFAULT_PIPELINE_HISTORY_DIR)
    resolved_dir = _latest_dir(raw_dir, "pipeline_history_review", as_of)
    snapshot_path = resolved_dir / "pipeline_history_review_snapshot.json"
    report_path = resolved_dir / "pipeline_history_review.md"
    snapshot = _read_json(snapshot_path)
    if not snapshot:
        return {
            "pipeline_history_status": "missing",
            "pipeline_history_dir": str(resolved_dir),
            "pipeline_history_snapshot_path": str(snapshot_path),
            "pipeline_history_report_path": str(report_path),
            "pipeline_history_health_state": "missing",
            "pipeline_history_alert_count": None,
        }
    return {
        "pipeline_history_status": "ok",
        "pipeline_history_dir": str(resolved_dir),
        "pipeline_history_snapshot_path": str(snapshot_path),
        "pipeline_history_report_path": str(report_path),
        "pipeline_history_generated_at": snapshot.get("generated_at"),
        "pipeline_history_health_state": snapshot.get("health_state", "missing"),
        "pipeline_history_alert_count": snapshot.get("alert_count"),
        "pipeline_history_latest_as_of_date": snapshot.get("latest_as_of_date"),
        "pipeline_history_history_freshness_status": snapshot.get("history_freshness_status"),
        "pipeline_history_latest_satellite_risk_budget_status": snapshot.get("latest_satellite_risk_budget_status"),
        "pipeline_history_latest_satellite_risk_budget_decision": snapshot.get("latest_satellite_risk_budget_decision"),
        "pipeline_history_previous_satellite_risk_budget_decision": snapshot.get("previous_satellite_risk_budget_decision"),
        "pipeline_history_satellite_risk_budget_decision_changed": snapshot.get(
            "satellite_risk_budget_decision_changed"
        ),
        "pipeline_history_latest_satellite_risk_budget_recommended_satellite_weight": snapshot.get(
            "latest_satellite_risk_budget_recommended_satellite_weight"
        ),
        "pipeline_history_satellite_risk_budget_recommended_weight_change": snapshot.get(
            "satellite_risk_budget_recommended_weight_change"
        ),
        "pipeline_history_latest_satellite_risk_budget_selected_horizon": snapshot.get(
            "latest_satellite_risk_budget_selected_horizon"
        ),
        "pipeline_history_latest_satellite_risk_budget_report_path": snapshot.get(
            "latest_satellite_risk_budget_report_path"
        ),
    }


def _summarize_allocator_observation(
    project_root: Path,
    daily_run_status_dir: str | Path | None,
    allocator_observation_dir: str | Path | None,
    as_of: date,
    max_staleness_days: int,
) -> dict[str, Any]:
    raw_status_dir = _resolve(project_root, daily_run_status_dir, DEFAULT_DAILY_RUN_STATUS_DIR)
    status_snapshot_path = raw_status_dir / "daily_run_status_snapshot.json"
    status_snapshot = _read_json(status_snapshot_path)

    observation_snapshot_path: Path | None = None
    raw_observation_snapshot_path = status_snapshot.get("latest_observation_snapshot_path")
    if raw_observation_snapshot_path:
        observation_snapshot_path = Path(str(raw_observation_snapshot_path))
        if not observation_snapshot_path.is_absolute():
            observation_snapshot_path = project_root / observation_snapshot_path

    if observation_snapshot_path is None or not observation_snapshot_path.exists():
        raw_observation_dir = _resolve(project_root, allocator_observation_dir, DEFAULT_ALLOCATOR_OBSERVATION_DIR)
        observation_dir = _latest_dir(raw_observation_dir, "allocator_observation", as_of)
        observation_snapshot_path = observation_dir / "allocator_observation_snapshot.json"

    observation_snapshot = _read_json(observation_snapshot_path)
    observation_as_of = _parse_date(observation_snapshot.get("as_of_date"))
    freshness, days = _freshness(observation_as_of, as_of, max_staleness_days)
    report_path = observation_snapshot_path.parent / "allocator_observation.md"
    blocking_items = observation_snapshot.get("blocking_items") or []
    monitor_items = observation_snapshot.get("monitor_items") or []

    return {
        "daily_run_status_dir": str(raw_status_dir),
        "daily_run_status_snapshot_path": str(status_snapshot_path),
        "daily_run_status_status": "ok" if status_snapshot else "missing",
        "daily_run_state": status_snapshot.get("run_state"),
        "daily_run_state_severity": status_snapshot.get("run_state_severity"),
        "daily_run_problem_state": status_snapshot.get("problem_state"),
        "live_preflight_status": status_snapshot.get("latest_live_preflight_status"),
        "live_preflight_decision": status_snapshot.get("latest_live_preflight_decision"),
        "live_preflight_status_path": status_snapshot.get("latest_live_preflight_status_path"),
        "live_preflight_report_path": status_snapshot.get("latest_live_preflight_report_path"),
        "live_preflight_snapshot_path": status_snapshot.get("latest_live_preflight_snapshot_path"),
        "daily_preflight_skipped": status_snapshot.get("skip_live_preflight", False),
        "live_preflight_blocking_items_count": status_snapshot.get("latest_live_preflight_blocking_items_count"),
        "live_preflight_monitor_items_count": status_snapshot.get("latest_live_preflight_monitor_items_count"),
        "live_preflight_live_shadow_review_decisions_path": status_snapshot.get(
            "latest_live_preflight_live_shadow_review_decisions_path"
        ),
        "live_preflight_live_shadow_review_decision_count": status_snapshot.get(
            "latest_live_preflight_live_shadow_review_decision_count"
        ),
        "live_preflight_live_shadow_review_blocking_decision_count": status_snapshot.get(
            "latest_live_preflight_live_shadow_review_blocking_decision_count"
        ),
        "live_preflight_live_shadow_review_monitor_decision_count": status_snapshot.get(
            "latest_live_preflight_live_shadow_review_monitor_decision_count"
        ),
        "live_preflight_live_shadow_review_unknown_decision_count": status_snapshot.get(
            "latest_live_preflight_live_shadow_review_unknown_decision_count"
        ),
        "allocator_observation_status": "ok" if observation_snapshot else "missing",
        "allocator_observation_snapshot_path": str(observation_snapshot_path),
        "allocator_observation_report_path": str(report_path),
        "allocator_observation_freshness_status": freshness,
        "allocator_observation_days_since_as_of": days,
        "allocator_observation_generated_at": observation_snapshot.get("generated_at"),
        "allocator_observation_as_of_date": observation_as_of.isoformat() if observation_as_of else None,
        "allocator_observation_decision_status": observation_snapshot.get("observation_status"),
        "allocator_observation_next_action_stage": observation_snapshot.get("next_action_stage"),
        "allocator_observation_risk_budget_decision": observation_snapshot.get("risk_budget_decision"),
        "allocator_observation_next_review_date": observation_snapshot.get("next_review_date"),
        "allocator_observation_next_review_horizon": observation_snapshot.get("next_review_horizon"),
        "allocator_observation_outcome_analysis_status": observation_snapshot.get("outcome_analysis_status"),
        "allocator_observation_outcome_ready_horizon_count": observation_snapshot.get("outcome_ready_horizon_count"),
        "allocator_observation_outcome_due_count": observation_snapshot.get("outcome_due_count"),
        "allocator_observation_blocking_item_count": len(blocking_items) if observation_snapshot else None,
        "allocator_observation_monitor_item_count": len(monitor_items) if observation_snapshot else None,
        "allocator_observation_latest_status_path": status_snapshot.get("latest_observation_status_path"),
    }


def _dashboard_posture(snapshot: dict[str, Any]) -> str:
    if snapshot.get("data_freshness_status") != "fresh_enough":
        return "wait_for_fresh_model_data"
    if snapshot.get("market_cache_status") != "fresh_enough":
        return "wait_for_fresh_market_cache"
    if snapshot.get("allocator_input_status") != "fresh_enough":
        return "wait_for_fresh_allocator_inputs"
    if snapshot.get("paper_account_status") == "ok" and snapshot.get("paper_freshness_status") != "fresh_enough":
        return "wait_for_fresh_paper_account"
    if snapshot.get("sentiment_freshness_status") != "fresh_enough":
        return "model_ok_wait_for_fresh_sentiment"
    if snapshot.get("trigger_freshness_status") != "fresh_enough":
        return "model_ok_wait_for_current_trigger"
    sentiment_state = str(snapshot.get("sentiment_state", ""))
    action_posture = str(snapshot.get("action_posture", ""))
    if sentiment_state in {"cold", "weak"}:
        return "defensive_review_only"
    if action_posture == "review_core_base_allocator_gate":
        return "core_base_watch_allocator_gate"
    return "review_only"


def _render_report(snapshot: dict[str, Any]) -> str:
    trigger_preview = snapshot.get("trigger_preview") or []
    preview_text = "\n".join(f"- {line}" for line in trigger_preview[:6]) if trigger_preview else "- N/A"
    return f"""# Latest Dashboard

Generated at: `{snapshot.get("generated_at")}`

This dashboard is a local research overview only. It does not connect to brokers, place orders, or provide investment advice.

## One-Page State

| Area | Status |
| --- | --- |
| Dashboard posture | `{snapshot.get("dashboard_posture")}` |
| Daily model check | `{snapshot.get("daily_check_status")}` / `{snapshot.get("data_freshness_status")}` |
| Phase-2 posture | `{snapshot.get("phase2_posture")}` |
| Daily action posture | `{snapshot.get("action_posture")}` |
| Paper account | `{snapshot.get("paper_account_status")}` / `{snapshot.get("paper_freshness_status", "missing")}` |
| Market sentiment | `{snapshot.get("sentiment_state")}` / `{snapshot.get("sentiment_freshness_status")}` |
| Trigger monitor | `{snapshot.get("trigger_status")}` / `{snapshot.get("trigger_freshness_status")}` |
| Market cache | `{snapshot.get("market_cache_status")}` |
| Allocator inputs | `{snapshot.get("allocator_input_status")}` |
| Model build audit | `{snapshot.get("model_audit_status")}` |
| Pipeline history | `{snapshot.get("pipeline_history_status")}` / `{snapshot.get("pipeline_history_health_state")}` |
| Allocator observation | `{snapshot.get("allocator_observation_status")}` / `{snapshot.get("allocator_observation_decision_status")}` |
| Live preflight | `{snapshot.get("live_preflight_status")}` / `{snapshot.get("live_preflight_decision")}` |
| Live preflight skipped | `{snapshot.get("daily_preflight_skipped")}` |

## Model

| Metric | Value |
| --- | ---: |
| As-of date | {snapshot.get("as_of_date", "N/A")} |
| Local equity latest date | {snapshot.get("local_equity_latest_date", "N/A")} |
| Selected allocator candidate | {snapshot.get("selected_candidate", "N/A")} |
| Risk-on satellite weight | {_pct(snapshot.get("risk_on_satellite_weight"))} |
| Current drawdown | {_pct(snapshot.get("current_drawdown"))} |
| 20-trading-day return | {_pct(snapshot.get("return_20d"))} |
| Allocator Sharpe | {_num(snapshot.get("allocator_sharpe"))} |

## Allocator Promotion Watch

| Metric | Value |
| --- | ---: |
| Promotion status | `{snapshot.get("promotion_status", "N/A")}` |
| Promotion decision | `{snapshot.get("promotion_decision", "N/A")}` |
| Candidate passes headline gate | {snapshot.get("promotion_candidate_passes_headline_gate", "N/A")} |
| Independent support groups | {snapshot.get("promotion_support_group_count", snapshot.get("promotion_sensitivity_group_support_count", snapshot.get("promotion_sensitivity_support_count", "N/A")))} / {snapshot.get("promotion_min_sensitivity_support", "N/A")} |
| Sensitivity group support | {snapshot.get("promotion_sensitivity_group_support_count", snapshot.get("promotion_sensitivity_support_count", "N/A"))} |
| Sensitivity run support | {snapshot.get("promotion_sensitivity_run_support_count", snapshot.get("promotion_sensitivity_support_count", "N/A"))} |
| Evidence group support | {snapshot.get("promotion_evidence_support_count", "N/A")} |
| Return edge | {_pct(snapshot.get("promotion_return_edge"))} |
| Sharpe edge | {_num(snapshot.get("promotion_sharpe_edge"))} |
| Drawdown change | {_pct(snapshot.get("promotion_drawdown_change"))} |

## Freshness Gates

| Gate | Latest | Status |
| --- | ---: | --- |
| Daily model check equity | {snapshot.get("local_equity_latest_date", "N/A")} | `{snapshot.get("data_freshness_status")}` |
| Market cache | {snapshot.get("market_cache_latest_date", "N/A")} | `{snapshot.get("market_cache_status")}` |
| Allocator curve | {snapshot.get("allocator_curve_latest_date", "N/A")} | `{snapshot.get("allocator_curve_freshness_status")}` |
| Allocator summary | {snapshot.get("allocator_summary_latest_test_end", "N/A")} | `{snapshot.get("allocator_summary_freshness_status")}` |
| Paper account | {snapshot.get("paper_latest_date", "N/A")} | `{snapshot.get("paper_freshness_status")}` |
| Market sentiment | {snapshot.get("sentiment_date", "N/A")} | `{snapshot.get("sentiment_freshness_status")}` |
| Trigger monitor | {snapshot.get("trigger_report_date", "N/A")} | `{snapshot.get("trigger_freshness_status")}` |

Market cache coverage: {snapshot.get("market_cache_fresh_file_count", "N/A")} / {snapshot.get("market_cache_dated_file_count", "N/A")} files at latest date, ratio {_pct(snapshot.get("market_cache_fresh_ratio"))}.

## Pipeline History

| Metric | Value |
| --- | ---: |
| History status | `{snapshot.get("pipeline_history_status", "N/A")}` |
| Health state | `{snapshot.get("pipeline_history_health_state", "N/A")}` |
| Alert count | {snapshot.get("pipeline_history_alert_count", "N/A")} |
| Latest history as-of | {snapshot.get("pipeline_history_latest_as_of_date", "N/A")} |
| History freshness | `{snapshot.get("pipeline_history_history_freshness_status", "N/A")}` |
| Satellite risk-budget status | `{snapshot.get("pipeline_history_latest_satellite_risk_budget_status", "N/A")}` |
| Satellite risk-budget decision | `{snapshot.get("pipeline_history_latest_satellite_risk_budget_decision", "N/A")}` |
| Previous risk-budget decision | `{snapshot.get("pipeline_history_previous_satellite_risk_budget_decision", "N/A")}` |
| Decision changed | {snapshot.get("pipeline_history_satellite_risk_budget_decision_changed", "N/A")} |
| Recommended satellite budget | {_pct(snapshot.get("pipeline_history_latest_satellite_risk_budget_recommended_satellite_weight"))} |
| Recommended budget change | {_pct(snapshot.get("pipeline_history_satellite_risk_budget_recommended_weight_change"))} |
| Selected outcome horizon | `{snapshot.get("pipeline_history_latest_satellite_risk_budget_selected_horizon", "N/A")}` |
| Risk-budget review | `{snapshot.get("pipeline_history_latest_satellite_risk_budget_report_path", "N/A")}` |

## Allocator Observation

| Metric | Value |
| --- | ---: |
| Daily run state | `{snapshot.get("daily_run_state", "N/A")}` / `{snapshot.get("daily_run_state_severity", "N/A")}` |
| Problem state | {snapshot.get("daily_run_problem_state", "N/A")} |
| Observation status | `{snapshot.get("allocator_observation_decision_status", "N/A")}` |
| Observation freshness | `{snapshot.get("allocator_observation_freshness_status", "N/A")}` |
| Observation as-of date | {snapshot.get("allocator_observation_as_of_date", "N/A")} |
| Next action stage | `{snapshot.get("allocator_observation_next_action_stage", "N/A")}` |
| Risk-budget decision | `{snapshot.get("allocator_observation_risk_budget_decision", "N/A")}` |
| Outcome analysis status | `{snapshot.get("allocator_observation_outcome_analysis_status", "N/A")}` |
| Ready outcome horizons | {snapshot.get("allocator_observation_outcome_ready_horizon_count", "N/A")} |
| Due outcome rows | {snapshot.get("allocator_observation_outcome_due_count", "N/A")} |
| Next review | {snapshot.get("allocator_observation_next_review_date", "N/A")} / `{snapshot.get("allocator_observation_next_review_horizon", "N/A")}` |
| Blocking / monitor items | {snapshot.get("allocator_observation_blocking_item_count", "N/A")} / {snapshot.get("allocator_observation_monitor_item_count", "N/A")} |
| Live preflight blocking / monitor | {snapshot.get("live_preflight_blocking_items_count", "N/A")} / {snapshot.get("live_preflight_monitor_items_count", "N/A")} |
| Live-shadow review decisions | {snapshot.get("live_preflight_live_shadow_review_decision_count", "N/A")} |
| Live-shadow review blocking / monitor / unknown | {snapshot.get("live_preflight_live_shadow_review_blocking_decision_count", "N/A")} / {snapshot.get("live_preflight_live_shadow_review_monitor_decision_count", "N/A")} / {snapshot.get("live_preflight_live_shadow_review_unknown_decision_count", "N/A")} |
| Live-shadow review decisions CSV | `{snapshot.get("live_preflight_live_shadow_review_decisions_path", "N/A")}` |

## Model Build Hygiene

| Metric | Value |
| --- | ---: |
| Audit status | `{snapshot.get("model_audit_status", "N/A")}` |
| Duplicate config groups | {snapshot.get("model_audit_duplicate_config_groups", "N/A")} |
| Root configs without extends | {snapshot.get("model_audit_root_configs_without_extends", "N/A")} |
| Walk-forward action items | {snapshot.get("model_audit_walk_forward_action_items", "N/A")} |
| Resume/finalize candidates | {snapshot.get("model_audit_walk_forward_resume_candidates", "N/A")} |
| Archive-review candidates | {snapshot.get("model_audit_walk_forward_archive_review_candidates", "N/A")} |
| Top action | `{snapshot.get("model_audit_top_action", "N/A")}` |

## Paper Account

| Metric | Value |
| --- | ---: |
| Latest paper date | {snapshot.get("paper_latest_date", "N/A")} |
| Latest paper candidate | {snapshot.get("paper_latest_candidate", "N/A")} |
| Latest paper regime | {snapshot.get("paper_latest_regime", "N/A")} |
| Core target weight | {_pct(snapshot.get("paper_latest_core_weight"))} |
| Satellite target weight | {_pct(snapshot.get("paper_latest_satellite_weight"))} |
| Cash target weight | {_pct(snapshot.get("paper_latest_cash_weight"))} |
| Final equity | {_num(snapshot.get("paper_final_equity"), 2)} |
| Total return | {_pct(snapshot.get("paper_total_return"))} |
| Max drawdown | {_pct(snapshot.get("paper_max_drawdown"))} |
| Current drawdown | {_pct(snapshot.get("paper_current_drawdown"))} |
| Sharpe | {_num(snapshot.get("paper_sharpe"))} |
| Audit events | {snapshot.get("paper_audit_event_count", "N/A")} |

## Market Sentiment

| Metric | Value |
| --- | ---: |
| Sentiment date | {snapshot.get("sentiment_date", "N/A")} |
| State | {snapshot.get("sentiment_state", "N/A")} |
| Score | {_num(snapshot.get("sentiment_score"))} |
| Reference exposure | {_pct(snapshot.get("reference_exposure"))} |
| Coverage stocks | {snapshot.get("coverage_count", "N/A")} |
| Market average return | {_pct_points(snapshot.get("market_return"))} |
| Advancing ratio | {_pct(snapshot.get("advance_ratio"))} |
| Limit up / down | {snapshot.get("limit_up_count", "N/A")} / {snapshot.get("limit_down_count", "N/A")} |

## Trigger Monitor

| Metric | Value |
| --- | --- |
| Trigger report date | {snapshot.get("trigger_report_date", "N/A")} |
| Days since trigger report | {snapshot.get("trigger_days_since_latest", "N/A")} |
| Last write time | {snapshot.get("trigger_last_write_time", "N/A")} |
| Report path | `{snapshot.get("trigger_report_path", "N/A")}` |

Preview:

{preview_text}

## Source Files

- Daily model check: `{snapshot.get("daily_check_report_path")}`
- Phase-2 review: `{snapshot.get("phase2_report_path")}`
- Allocator promotion review: `{snapshot.get("promotion_report_path")}`
- Paper account: `{snapshot.get("paper_account_report_path")}`
- Market sentiment: `{snapshot.get("sentiment_report_path")}`
- Trigger monitor: `{snapshot.get("trigger_report_path")}`
- Market cache: `{snapshot.get("market_cache_dir")}`
- Allocator inputs: `{snapshot.get("allocator_dir")}`
- Model build audit: `{snapshot.get("model_audit_report_path")}`
- Pipeline history: `{snapshot.get("pipeline_history_report_path")}`
- Allocator observation: `{snapshot.get("allocator_observation_report_path")}`
- Daily run status: `{snapshot.get("daily_run_status_snapshot_path")}`
- Live preflight status: `{snapshot.get("live_preflight_status_path")}`
- Live preflight snapshot: `{snapshot.get("live_preflight_snapshot_path")}`
- Live preflight report: `{snapshot.get("live_preflight_report_path")}`
- Daily preflight skipped: `{snapshot.get("daily_preflight_skipped")}`
""" 


def run_daily_dashboard(
    project_root: str | Path = Path("."),
    output_dir: str | Path | None = None,
    daily_check_dir: str | Path | None = None,
    paper_account_dir: str | Path | None = None,
    sentiment_dir: str | Path | None = None,
    data_cache_dir: str | Path | None = None,
    allocator_dir: str | Path | None = None,
    trigger_report: str | Path | None = None,
    model_audit_dir: str | Path | None = None,
    pipeline_history_dir: str | Path | None = None,
    daily_run_status_dir: str | Path | None = None,
    allocator_observation_dir: str | Path | None = None,
    as_of_date: str | date | None = None,
    max_staleness_days: int = 3,
    min_cache_fresh_ratio: float = 0.90,
    date_stamp: bool = False,
) -> DailyDashboardResult:
    root = Path(project_root).resolve()
    as_of = _parse_date(as_of_date) if as_of_date is not None else datetime.now().date()
    if as_of is None:
        raise ValueError(f"Invalid as_of_date: {as_of_date}")
    resolved_output = _resolve_output(root, output_dir, as_of, date_stamp)
    snapshot: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": as_of.isoformat(),
        "max_staleness_days": int(max_staleness_days),
    }
    snapshot.update(_summarize_daily_check(root, daily_check_dir, as_of))
    snapshot.update(_summarize_market_cache(root, data_cache_dir, as_of, max_staleness_days, min_cache_fresh_ratio))
    snapshot.update(_summarize_allocator(root, allocator_dir, as_of, max_staleness_days))
    snapshot.update(_summarize_paper_account(root, paper_account_dir, as_of, max_staleness_days))
    snapshot.update(_summarize_sentiment(root, sentiment_dir, as_of, max_staleness_days))
    snapshot.update(_summarize_trigger(trigger_report, as_of, max_staleness_days))
    snapshot.update(_summarize_model_audit(root, model_audit_dir, as_of, max_staleness_days))
    snapshot.update(_summarize_pipeline_history(root, pipeline_history_dir, as_of))
    snapshot.update(
        _summarize_allocator_observation(
            root,
            daily_run_status_dir,
            allocator_observation_dir,
            as_of,
            max_staleness_days,
        )
    )
    snapshot["dashboard_posture"] = _dashboard_posture(snapshot)

    resolved_output.mkdir(parents=True, exist_ok=True)
    snapshot_path = resolved_output / "latest_dashboard_snapshot.json"
    report_path = resolved_output / "latest_dashboard.md"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot), encoding="utf-8")
    return DailyDashboardResult(
        output_dir=resolved_output,
        report_path=report_path,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )
