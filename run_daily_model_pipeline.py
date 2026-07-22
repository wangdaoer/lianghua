"""Run the default post-refresh daily model workflow."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter, sleep
from uuid import uuid4

from daily_run_card import write_daily_run_card
from pipeline_cache import CACHEABLE_STEPS, PipelineStepCache, build_global_fingerprint
from workspace_paths import (
    daily_data_root,
    dashboard_root,
    fallback_data_root,
    resolve_workspace_path,
)


DEFAULT_DAILY_DATA_DIR = daily_data_root()
DEFAULT_FALLBACK_DATA_DIR = fallback_data_root()
DEFAULT_DAILY_NORMALIZED_DIR = DEFAULT_DAILY_DATA_DIR / "ths_exports" / "normalized"
DEFAULT_FETCH_STATUS_UTILS = DEFAULT_FALLBACK_DATA_DIR / "scripts" / "market_data_utils.py"
DEFAULT_BENCHMARK = DEFAULT_DAILY_DATA_DIR / "benchmarks" / "510300.csv"
DEFAULT_ALLOWED_TREND_STATES = "趋势确认,生命线健康,回调可观察"
DEFAULT_MARKETLENS_DASHBOARD_OUTPUT = dashboard_root() / "data" / "quant-model3-latest.json"


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]
    max_attempts: int = 1
    retry_delay_seconds: float = 0.0


class PipelineStepExecutionError(RuntimeError):
    def __init__(
        self,
        step: PipelineStep,
        returncode: int,
        execution: dict[str, object] | None = None,
        completed_executions: list[dict[str, object]] | None = None,
    ):
        super().__init__(f"Pipeline step {step.name} failed with exit code {returncode}")
        self.step = step
        self.returncode = returncode
        self.execution = execution
        self.completed_executions = list(completed_executions or [])


@dataclass(frozen=True)
class PipelineConfig:
    asof_date: str
    python_exe: str = sys.executable
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    output_root: Path = Path("outputs/high_return_v2")
    base_panel: Path = Path("data/base_panel.csv")
    daily_dir: Path = DEFAULT_DAILY_NORMALIZED_DIR
    daily_data_dir: Path = DEFAULT_DAILY_DATA_DIR
    fallback_data_dir: Path = DEFAULT_FALLBACK_DATA_DIR
    fetch_status_utils: Path = DEFAULT_FETCH_STATUS_UTILS
    daily_start: str = "2026-06-22"
    benchmark: Path = DEFAULT_BENCHMARK
    research_db: Path = Path("data/research.sqlite3")
    rules: Path = Path("configs/personal_trade_habit_overlay.yaml")
    institutional_accumulation_config: Path = Path(
        "configs/institutional_accumulation_shadow.yaml"
    )
    world_event_config: Path = Path("configs/world_event_shadow.yaml")
    world_event_snapshot: Path | None = None
    symbol_history: Path | None = None
    shadow_account_review: Path | None = None
    train_model: bool = True
    top_n: int = 40
    candidate_pool_n: int = 120
    rebalance_frequency: int = 5
    retrain_frequency: int = 20
    train_days: int = 252
    max_position_weight: float = 0.029
    leverage: float = 0.93
    max_abs_daily_return: float = 0.22
    market_ma_window: int = 120
    market_below_ma_exposure: float = 0.60
    market_risk_off_drawdown_20d: float = -0.08
    market_crash_exposure: float = 0.0
    allowed_trend_states: str = DEFAULT_ALLOWED_TREND_STATES
    include_factor_decay_monitor: bool = True
    include_strategy_stability_report: bool = True
    include_strategy_family_forward_report: bool = True
    include_strategy_arena: bool = True
    include_benchmark_refresh: bool = True
    include_regime_shadow_compare: bool = True
    include_regime_shadow_tracking: bool = True
    include_dynamic_breadth_tracking: bool = True
    include_world_event_shadow: bool = True
    include_research_db_sync: bool = True
    include_marketlens_export: bool = False
    include_trend_ignition_shadow: bool = False
    marketlens_dashboard_output: Path = DEFAULT_MARKETLENS_DASHBOARD_OUTPUT
    regime_shadow_config: Path = Path("configs/evolution_strong_pullback.yaml")
    regime_shadow_candidate_id: str = "regime_090_balanced"
    dynamic_breadth_preregistration: Path = Path(
        "configs/dynamic_breadth_overlay_preregistration.json"
    )
    arena_breadth_ma_window: int = 60
    arena_breadth_threshold: float = 0.45
    arena_breadth_below_exposure: float = 0.55
    arena_breadth_crash_threshold: float = 0.32
    arena_breadth_crash_exposure: float = 0.20
    trend_ignition_scorer: Path = Path(
        "outputs/trend_ignition_training/scorer_v3_shortlist_exploratory/binned_scorer.json"
    )
    trend_ignition_scorer_summary: Path = Path(
        "outputs/trend_ignition_training/scorer_v3_shortlist_exploratory/scorer_summary.json"
    )
    trend_ignition_selection_status: str = "exploratory_posthoc"
    max_parallel_steps: int = 1
    enable_step_cache: bool = False


PIPELINE_STEP_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "benchmark_refresh": (),
    "update_panel": (),
    "trend_state": ("update_panel",),
    "regime_shadow_compare": ("update_panel", "benchmark_refresh"),
    "regime_shadow_tracking": ("regime_shadow_compare",),
    "train_rank_model": ("update_panel", "benchmark_refresh"),
    "train_breadth_guard_challenger": ("update_panel", "benchmark_refresh"),
    "dynamic_breadth_overlay_tracking": (
        "train_rank_model",
        "train_breadth_guard_challenger",
    ),
    "world_event_shadow": (),
    "rank_candidates": ("trend_state", "train_rank_model"),
    "personal_overlay": ("rank_candidates",),
    "early_pattern_watchlist": ("update_panel",),
    "main_net_volume_shadow": ("early_pattern_watchlist",),
    "institutional_accumulation_shadow": ("update_panel",),
    "institutional_accumulation_tracking": ("institutional_accumulation_shadow",),
    "czsc_structure_shadow": ("personal_overlay", "early_pattern_watchlist"),
    "hidden_accumulation_tracking": ("early_pattern_watchlist",),
    "factor_decay_monitor": ("update_panel",),
    "daily_chinese_report": (
        "personal_overlay",
        "early_pattern_watchlist",
        "train_rank_model",
    ),
    "merged_daily_outputs": ("daily_chinese_report", "trend_state"),
    "trend_ignition_shadow": ("merged_daily_outputs",),
    "strategy_family_forward_report": (
        "merged_daily_outputs",
        "main_net_volume_shadow",
        "institutional_accumulation_tracking",
        "czsc_structure_shadow",
        "hidden_accumulation_tracking",
        "trend_ignition_shadow",
    ),
    "strategy_arena": (
        "strategy_family_forward_report",
        "regime_shadow_tracking",
        "train_rank_model",
        "train_breadth_guard_challenger",
        "czsc_structure_shadow",
        "main_net_volume_shadow",
    ),
    "strategy_stability_report": ("train_rank_model",),
}


def _date_token(asof_date: str) -> str:
    return asof_date.replace("-", "")


def _resolve(root: Path, path: Path) -> Path:
    return resolve_workspace_path(root, path)


def panel_end_date(path: Path) -> str | None:
    match = re.fullmatch(
        r"data_panel_history_main_chinext_\d{8}_(\d{8})\.(?:csv|parquet)",
        path.name,
    )
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").strftime("%Y-%m-%d")


def discover_base_panel(project_root: Path, asof_date: str) -> Path:
    target_token = _date_token(asof_date)
    candidates: list[tuple[str, int, Path]] = []
    for suffix in ("csv", "parquet"):
        for path in project_root.glob(f"data_panel_history_main_chinext_*.{suffix}"):
            end_date = panel_end_date(path)
            if end_date is None:
                continue
            token = _date_token(end_date)
            if token < target_token:
                candidates.append((token, int(path.suffix.lower() == ".parquet"), path))
    if not candidates:
        raise FileNotFoundError(
            f"No base panel ending before {asof_date} was found under {project_root}. "
            "Provide --base-panel explicitly."
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def effective_daily_start(config: PipelineConfig) -> str:
    configured = datetime.strptime(config.daily_start, "%Y-%m-%d")
    base_end = panel_end_date(config.base_panel)
    if base_end is None:
        return config.daily_start
    next_session_floor = datetime.strptime(base_end, "%Y-%m-%d") + timedelta(days=1)
    return max(configured, next_session_floor).strftime("%Y-%m-%d")


def default_stability_inputs_available(project_root: Path) -> bool:
    from summarize_core_risk_filter_stability import DEFAULT_SCHEMES

    for _label, path_text in DEFAULT_SCHEMES:
        scheme = _resolve(project_root, Path(path_text))
        if not (scheme / "metrics.json").exists() or not (scheme / "equity_curve.csv").exists():
            return False
    return True


def _latest_artifact(path: Path) -> Path:
    candidates = [path] if path.exists() else []
    candidates.extend(path.parent.glob(f"{path.stem}_pending*{path.suffix}"))
    if not candidates:
        return path
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _artifact_reference(path: Path, *, hidden: bool = False) -> str | None:
    return None if hidden else str(_latest_artifact(path))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def factor_decay_monitor_status(
    path: Path,
    expected_asof_date: str,
    *,
    enabled: bool = True,
) -> dict[str, object]:
    if not enabled:
        return {"status": "skipped", "path": None}
    if not path.exists():
        return {"status": "missing", "path": None}
    try:
        payload = _read_json_object(path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {"status": "invalid", "path": str(path), "message": str(exc)}
    asof_date = str(payload.get("asof_date") or "")
    if _date_token(asof_date) != _date_token(expected_asof_date):
        return {
            "status": "stale",
            "path": str(path),
            "asof_date": asof_date or None,
        }
    replacement = payload.get("replacement_competition")
    replacement = replacement if isinstance(replacement, dict) else {}
    factor_metadata = payload.get("factor_metadata")
    factor_metadata = factor_metadata if isinstance(factor_metadata, dict) else {}
    incremental_checkpoint = payload.get("incremental_checkpoint")
    incremental_checkpoint = (
        incremental_checkpoint if isinstance(incremental_checkpoint, dict) else {}
    )
    checkpoint_parent = incremental_checkpoint.get("parent")
    checkpoint_parent = checkpoint_parent if isinstance(checkpoint_parent, dict) else {}
    return {
        "status": payload.get("overall_status", "unknown"),
        "path": str(path),
        "asof_date": asof_date,
        "factor": payload.get("factor"),
        "automatic_model_change": payload.get("automatic_model_change", False),
        "research_only": payload.get("research_only", True),
        "factor_metadata": factor_metadata,
        "calculation_mode": payload.get("calculation_mode", "full"),
        "recompute_start": incremental_checkpoint.get("recompute_start"),
        "checkpoint_parent_asof_date": checkpoint_parent.get("source_asof_date"),
        "replacement_status": replacement.get("status"),
        "replacement_shortlist": replacement.get("shortlist", []),
        "replacement_preregistered_candidates": replacement.get(
            "preregistered_candidates", []
        ),
        "replacement_validation_start_date": replacement.get(
            "validation_start_date"
        ),
        "replacement_tracking_status": replacement.get("tracking_status"),
    }


def _value_counts(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = (row.get(column) or "").strip() or "missing"
        counts[value] = counts.get(value, 0) + 1
    return counts


def _missing_count(rows: list[dict[str, str]], column: str) -> int | None:
    if not rows or column not in rows[0]:
        return None
    return sum(1 for row in rows if not (row.get(column) or "").strip())


def _top_rows(rows: list[dict[str, str]], columns: list[str], limit: int = 10) -> list[dict[str, str]]:
    return [{column: row.get(column, "") for column in columns} for row in rows[:limit]]


def _tracking_state_gate(
    steps: list[PipelineStep],
    run_status: str,
    failure: dict[str, object] | None,
) -> tuple[str | None, bool]:
    step_names = [step.name for step in steps]
    if "regime_shadow_tracking" not in step_names:
        return ("missing" if run_status == "failed" else None), run_status != "failed"
    if run_status != "failed":
        return None, True
    if not isinstance(failure, dict):
        return "missing", False
    failure_step = failure.get("step")
    if not isinstance(failure_step, str) or failure_step not in step_names:
        return "missing", False
    failure_index = step_names.index(failure_step)
    tracking_index = step_names.index("regime_shadow_tracking")
    if failure_index < tracking_index:
        return "not_run", False
    if failure_index == tracking_index:
        return "failed", False
    return None, True


def _pipeline_artifact_status_override(
    steps: list[PipelineStep],
    run_status: str,
    failure: dict[str, object] | None,
    step_executions: list[dict[str, object]] | None,
    step_name: str,
) -> str | None:
    """Hide same-day artifacts left by an earlier attempt after a failed run."""

    if run_status != "failed":
        return None
    step_names = [step.name for step in steps]
    if step_name not in step_names:
        return None

    executions = {
        str(item.get("name")): str(item.get("status"))
        for item in (step_executions or [])
        if item.get("name")
    }
    execution_status = executions.get(step_name)
    if execution_status in {"success", "cached"}:
        return None
    if execution_status == "failed":
        return "failed"

    failure_step = (failure or {}).get("step")
    if failure_step == step_name:
        return "failed"
    if isinstance(failure_step, str) and failure_step in step_names:
        if step_names.index(failure_step) > step_names.index(step_name):
            return None
    return "not_run"


def _research_database_sync_state(
    config: PipelineConfig,
    steps: list[PipelineStep],
    paths: dict[str, Path],
    run_status: str,
    failure: dict[str, object] | None,
) -> tuple[dict[str, object], str | None]:
    if not config.include_research_db_sync:
        return {"status": "skipped"}, None
    step_names = [step.name for step in steps]
    sync_step = "research_database_sync"
    if run_status == "failed" and sync_step in step_names and isinstance(failure, dict):
        failed_step = failure.get("step")
        if isinstance(failed_step, str) and failed_step in step_names:
            if step_names.index(failed_step) < step_names.index(sync_step):
                return {"status": "not_run"}, None
            if failed_step == sync_step:
                status_path = paths["research_database_sync_status"]
                if status_path.exists():
                    status = _read_json_object(status_path)
                    if _date_token(str(status.get("asof_date", ""))) == _date_token(config.asof_date):
                        return status, str(_latest_artifact(status_path))
                return {"status": "failed"}, None
    status_path = paths["research_database_sync_status"]
    if not status_path.exists():
        return {"status": "missing"}, None
    status = _read_json_object(status_path)
    if _date_token(str(status.get("asof_date", ""))) != _date_token(config.asof_date):
        return {"status": "stale", "recorded_asof_date": status.get("asof_date")}, None
    return status, str(_latest_artifact(status_path))


def _world_event_shadow_state(
    config: PipelineConfig,
    steps: list[PipelineStep],
    paths: dict[str, Path],
    run_status: str,
    failure: dict[str, object] | None,
    step_executions: list[dict[str, object]] | None,
) -> tuple[dict[str, object], dict[str, str | None]]:
    artifacts = {"json": None, "csv": None, "report": None}
    if not config.include_world_event_shadow:
        return {"status": "skipped"}, artifacts

    override = _pipeline_artifact_status_override(
        steps,
        run_status,
        failure,
        step_executions,
        "world_event_shadow",
    )
    if override:
        return {"status": override}, artifacts

    metadata = _read_json_object(paths["world_event_shadow_json"])
    if not metadata:
        return {"status": "missing"}, artifacts
    if metadata.get("asof_date") != config.asof_date:
        return {
            "status": "stale",
            "source_asof_date": metadata.get("asof_date"),
        }, artifacts

    required_outputs = (
        paths["world_event_shadow_json"],
        paths["world_event_shadow_csv"],
        paths["world_event_shadow_report"],
    )
    if not all(path.exists() for path in required_outputs):
        return {"status": "incomplete_outputs"}, artifacts

    artifacts.update(
        {
            "json": str(_latest_artifact(paths["world_event_shadow_json"])),
            "csv": str(_latest_artifact(paths["world_event_shadow_csv"])),
            "report": str(_latest_artifact(paths["world_event_shadow_report"])),
        }
    )
    return metadata, artifacts


def _dynamic_breadth_tracking_state(
    config: PipelineConfig,
    steps: list[PipelineStep],
    paths: dict[str, Path],
    run_status: str,
    failure: dict[str, object] | None,
    step_executions: list[dict[str, object]] | None,
) -> tuple[dict[str, object], dict[str, str | None]]:
    enabled = (
        config.include_dynamic_breadth_tracking
        and config.train_model
        and config.include_strategy_arena
        and config.include_regime_shadow_compare
    )
    artifacts = {
        "preregistration": (
            str(paths["dynamic_breadth_preregistration"])
            if paths["dynamic_breadth_preregistration"].exists()
            else None
        ),
        "ledger": None,
        "summary": None,
        "report": None,
    }
    if not enabled:
        return {"status": "skipped"}, artifacts

    override = _pipeline_artifact_status_override(
        steps,
        run_status,
        failure,
        step_executions,
        "dynamic_breadth_overlay_tracking",
    )
    if override:
        return {"status": override}, artifacts

    summary_path = paths["dynamic_breadth_tracking_summary"]
    if not summary_path.exists():
        return {"status": "missing"}, artifacts
    summary = _read_json_object(summary_path)
    if summary.get("source_latest_date") != config.asof_date:
        return {
            "status": "stale",
            "source_latest_date": summary.get("source_latest_date"),
        }, artifacts

    required_outputs = (
        paths["dynamic_breadth_tracking_ledger"],
        summary_path,
        paths["dynamic_breadth_tracking_report"],
    )
    if not all(path.exists() for path in required_outputs):
        return {"status": "incomplete_outputs"}, artifacts

    artifacts.update(
        {
            "ledger": str(_latest_artifact(paths["dynamic_breadth_tracking_ledger"])),
            "summary": str(_latest_artifact(summary_path)),
            "report": str(_latest_artifact(paths["dynamic_breadth_tracking_report"])),
        }
    )
    return summary, artifacts


def _trend_ignition_shadow_state(
    config: PipelineConfig,
    steps: list[PipelineStep],
    paths: dict[str, Path],
    run_status: str,
    failure: dict[str, object] | None,
) -> tuple[dict[str, object], str | None]:
    if not config.include_trend_ignition_shadow:
        return {"status": "skipped"}, None
    step_names = [step.name for step in steps]
    shadow_step = "trend_ignition_shadow"
    if run_status == "failed" and shadow_step in step_names and isinstance(failure, dict):
        failed_step = failure.get("step")
        if isinstance(failed_step, str) and failed_step in step_names:
            if step_names.index(failed_step) < step_names.index(shadow_step):
                return {"status": "not_run"}, None
            if failed_step == shadow_step:
                return {"status": "failed"}, None

    manifest_path = paths["trend_ignition_shadow_manifest"]
    if not manifest_path.exists():
        return {"status": "missing"}, None
    manifest = _read_json_object(manifest_path)
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        return {"status": "invalid_manifest"}, None
    if inputs.get("start") != config.asof_date or inputs.get("end") != config.asof_date:
        return {
            "status": "stale",
            "recorded_start": inputs.get("start"),
            "recorded_end": inputs.get("end"),
        }, None
    if manifest.get("research_only") is not True or manifest.get("trade_instruction") is not False:
        return {"status": "invalid_research_boundary"}, None
    manifest["status"] = "complete"
    return manifest, str(_latest_artifact(manifest_path))


def _jsonable(value: object) -> object:
    if dataclass_isinstance(value):
        return {key: _jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def dataclass_isinstance(value: object) -> bool:
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)


def fetch_status_summary(fetch_status_utils: Path) -> dict[str, object]:
    if not fetch_status_utils.exists():
        return {"status": "missing_utils", "path": str(fetch_status_utils)}
    try:
        spec = importlib.util.spec_from_file_location("market_data_utils", fetch_status_utils)
        if spec is None or spec.loader is None:
            return {"status": "load_error", "path": str(fetch_status_utils)}
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if hasattr(module, "get_latest_fetch_status"):
            status = module.get_latest_fetch_status()
        elif hasattr(module, "ensure_latest_fetch_ok"):
            status = module.ensure_latest_fetch_ok()
        else:
            return {"status": "no_status_function", "path": str(fetch_status_utils)}
        result = _jsonable(status)
        return result if isinstance(result, dict) else {"status": "ok", "value": result}
    except Exception as exc:
        return {"status": "status_error", "path": str(fetch_status_utils), "message": str(exc)}


def ensure_fetch_status_ok(status: dict[str, object]) -> None:
    if status.get("status") != "ok":
        raise RuntimeError(
            "Daily market data fetch status is not ready: "
            + json.dumps(status, ensure_ascii=False, default=str)
        )


def latest_daily_date(files: list[Path]) -> str:
    dates: list[str] = []
    for path in files:
        match = re.search(r"ths_hs_a_share_(20\d{2}-\d{2}-\d{2})\.(?:csv|xls|xlsx)$", path.name)
        if match:
            dates.append(match.group(1))
    if not dates:
        raise ValueError("未找到可用的每日行情文件。")
    return max(dates)


def discover_latest_daily_date(daily_dir: Path) -> str:
    files = list(daily_dir.glob("ths_hs_a_share_20*.csv")) + list(daily_dir.glob("ths_hs_a_share_20*.xls"))
    return latest_daily_date(files)


def latest_symbol_history(project_root: Path) -> Path | None:
    files = sorted(
        (project_root / "outputs").glob("personal_trade_review_*/by_symbol.csv"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def latest_shadow_account_review(
    project_root: Path,
    asof_date: str | None = None,
) -> Path | None:
    asof_token = _date_token(asof_date) if asof_date else None
    eligible: list[tuple[str, float, Path]] = []
    for path in (project_root / "outputs").glob(
        "personal_trade_review_*/shadow_account/shadow_account_review.json"
    ):
        match = re.fullmatch(
            r"personal_trade_review_(\d{8})", path.parent.parent.name
        )
        if not match:
            continue
        review_token = match.group(1)
        if asof_token and review_token > asof_token:
            continue
        eligible.append((review_token, path.stat().st_mtime, path))
    return max(eligible, key=lambda item: (item[0], item[1]))[2] if eligible else None


def names_source_for_date(asof_date: str, daily_data_dir: Path) -> Path:
    normalized = daily_data_dir / "ths_exports" / "normalized"
    for suffix in (".csv", ".xls", ".xlsx"):
        candidate = normalized / f"ths_hs_a_share_{asof_date}{suffix}"
        if candidate.exists():
            return candidate
    return normalized / f"ths_hs_a_share_{asof_date}.csv"


def metrics_path_for_date(asof_date: str, output_root: Path) -> Path:
    token = _date_token(asof_date)
    exact = output_root / f"personal_behavior_overlay_best_20220101_{token}" / "metrics.json"
    if exact.exists():
        return exact
    candidates = sorted(
        output_root.glob("personal_behavior_overlay_best_*/metrics.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else exact


def _paths(config: PipelineConfig) -> dict[str, Path]:
    root = config.project_root
    output_root = _resolve(root, config.output_root)
    daily_dir = _resolve(root, config.daily_dir)
    daily_data_dir = _resolve(root, config.daily_data_dir)
    token = _date_token(config.asof_date)
    model_dir = output_root / f"next_open_rank_model_stock_focus_lev093_marketfilter_bench{token}_20220101_{token}"
    breadth_model_dir = output_root / f"next_open_rank_model_breadth_guard_bench{token}_20220101_{token}"
    return {
        "base_panel": _resolve(root, config.base_panel),
        "daily_dir": daily_dir,
        "daily_data_dir": daily_data_dir,
        "fallback_data_dir": _resolve(root, config.fallback_data_dir),
        "fetch_status_utils": _resolve(root, config.fetch_status_utils),
        "benchmark": _resolve(root, config.benchmark),
        "output_root": output_root,
        "panel": root / f"data_panel_history_main_chinext_20220101_{token}.parquet",
        "trend_dir": output_root / f"trend_state_{token}",
        "model_dir": model_dir,
        "breadth_model_dir": breadth_model_dir,
        "weights": model_dir / "rolling_feature_weights.csv",
        "base_candidates": output_root / f"rank_model_candidates_trend_gated_bench{token}_{token}.csv",
        "overlay_candidates": output_root / f"rank_model_candidates_trend_gated_personal_overlay_{token}.csv",
        "early_watchlist": output_root / f"early_pattern_watchlist_{token}.csv",
        "main_net_volume_shadow": output_root / f"main_net_volume_shadow_{token}.csv",
        "main_net_volume_shadow_metadata": output_root / f"main_net_volume_shadow_{token}.json",
        "main_net_volume_shadow_report": output_root / f"main_net_volume_shadow_{token}.md",
        "main_net_volume_shadow_chart": output_root / f"main_net_volume_shadow_{token}.png",
        "institutional_accumulation_shadow": output_root
        / f"institutional_accumulation_shadow_{token}.csv",
        "institutional_accumulation_metadata": output_root
        / f"institutional_accumulation_shadow_{token}.json",
        "institutional_accumulation_report": output_root
        / f"institutional_accumulation_shadow_{token}.md",
        "institutional_accumulation_tracking": output_root
        / f"institutional_accumulation_tracking_{token}.csv",
        "institutional_accumulation_tracking_summary": output_root
        / f"institutional_accumulation_tracking_{token}.json",
        "institutional_accumulation_tracking_report": output_root
        / f"institutional_accumulation_tracking_{token}.md",
        "czsc_structure_shadow": output_root / f"czsc_structure_shadow_{token}.csv",
        "czsc_structure_shadow_metadata": output_root / f"czsc_structure_shadow_{token}.json",
        "czsc_structure_shadow_report": output_root / f"czsc_structure_shadow_{token}.md",
        "hidden_accumulation_tracking": output_root / f"hidden_accumulation_trade_watch_tracking_{token}.csv",
        "factor_decay_monitor_csv": output_root / f"factor_decay_monitor_{token}.csv",
        "factor_decay_monitor_json": output_root / f"factor_decay_monitor_{token}.json",
        "factor_decay_monitor_report": output_root / f"factor_decay_monitor_{token}.md",
        "factor_decay_observation": output_root / f"liquidity_stability_observation_{token}.csv",
        "factor_decay_history": output_root / "factor_decay_monitor_history.csv",
        "factor_replacement_daily_ic": output_root / f"factor_replacement_daily_ic_{token}.csv",
        "factor_replacement_competition": output_root
        / f"factor_replacement_competition_{token}.csv",
        "factor_replacement_report": output_root
        / f"factor_replacement_competition_{token}.md",
        "factor_replacement_preregistration": root
        / "configs"
        / "factor_replacement_preregistration.json",
        "factor_replacement_tracking": output_root
        / f"factor_replacement_tracking_{token}.csv",
        "factor_replacement_tracking_json": output_root
        / f"factor_replacement_tracking_{token}.json",
        "merged_state_pattern_scan": output_root / f"merged_state_pattern_scan_{token}.csv",
        "merged_model_decision_table": output_root / f"merged_model_decision_table_{token}.csv",
        "merged_priority_watchlist": output_root / f"merged_priority_watchlist_{token}.csv",
        "strategy_family_forward_summary": output_root / f"strategy_family_forward_summary_{token}.csv",
        "strategy_family_health": output_root / f"strategy_family_health_{token}.csv",
        "strategy_family_forward_report": output_root / f"strategy_family_forward_report_{token}.md",
        "strategy_arena_portfolio": output_root / f"strategy_arena_portfolio_{token}.csv",
        "strategy_arena_pairwise": output_root / f"strategy_arena_pairwise_{token}.csv",
        "strategy_arena_signal_division": output_root / f"strategy_arena_signal_division_{token}.csv",
        "strategy_arena_metadata": output_root / f"strategy_arena_{token}.json",
        "strategy_arena_report": output_root / f"strategy_arena_{token}.md",
        "strategy_arena_history": output_root / "strategy_arena_history.csv",
        "daily_run_card_json": output_root / f"daily_run_card_{token}.json",
        "daily_run_card_markdown": output_root / f"daily_run_card_{token}.md",
        "daily_report": output_root / f"daily_personal_overlay_report_{token}.md",
        "selected": output_root / f"daily_personal_overlay_selected_{token}.csv",
        "changes": output_root / f"daily_personal_overlay_changes_{token}.csv",
        "metrics": metrics_path_for_date(config.asof_date, output_root),
        "names_source": names_source_for_date(config.asof_date, daily_data_dir),
        "rules": _resolve(root, config.rules),
        "strategy_stability_prefix": output_root / f"core_risk_filter_finalist_stability_{token}",
        "strategy_stability_report": output_root / f"core_risk_filter_finalist_stability_{token}.md",
        "regime_shadow_dir": output_root / f"regime_shadow_compare_{token}",
        "regime_shadow_report": output_root / f"regime_shadow_compare_{token}" / "report.md",
        "regime_shadow_comparison": output_root / f"regime_shadow_compare_{token}" / "comparison.json",
        "regime_shadow_tracking_ledger": output_root / "regime_shadow_tracking.csv",
        "regime_shadow_tracking_summary": output_root / "regime_shadow_tracking_summary.json",
        "regime_shadow_tracking_report": output_root / "regime_shadow_tracking_report.md",
        "dynamic_breadth_preregistration": _resolve(
            root, config.dynamic_breadth_preregistration
        ),
        "dynamic_breadth_tracking_ledger": output_root
        / "dynamic_breadth_overlay_tracking.csv",
        "dynamic_breadth_tracking_summary": output_root
        / "dynamic_breadth_overlay_tracking_summary.json",
        "dynamic_breadth_tracking_report": output_root
        / "dynamic_breadth_overlay_tracking_report.md",
        "world_event_config": _resolve(root, config.world_event_config),
        "world_event_shadow_json": output_root / f"world_event_shadow_{token}.json",
        "world_event_shadow_csv": output_root / f"world_event_shadow_{token}.csv",
        "world_event_shadow_report": output_root / f"world_event_shadow_{token}.md",
        "world_event_shadow_cache": output_root / "world_event_shadow_payload_cache.json",
        "benchmark_refresh_status": output_root / f"benchmark_refresh_status_{token}.json",
        "pipeline_step_cache": output_root / f"pipeline_step_cache_{token}.json",
        "research_db": _resolve(root, config.research_db),
        "research_database_sync_status": output_root / f"research_database_sync_{token}.json",
        "marketlens_feed": output_root / "marketlens_model3_latest.json",
        "marketlens_state_input": output_root / f".marketlens_state_{token}.json",
        "marketlens_dashboard_feed": _resolve(root, config.marketlens_dashboard_output),
        "trend_ignition_scorer": _resolve(root, config.trend_ignition_scorer),
        "trend_ignition_scorer_summary": _resolve(root, config.trend_ignition_scorer_summary),
        "trend_ignition_shadow_dir": output_root / f"trend_ignition_shadow_{token}",
        "trend_ignition_shadow_manifest": output_root / f"trend_ignition_shadow_{token}" / "manifest.json",
        "trend_ignition_shadow_scores": output_root
        / f"trend_ignition_shadow_{token}"
        / "trend_ignition_shadow_scores.csv",
        "trend_ignition_shadow_report": output_root
        / f"trend_ignition_shadow_{token}"
        / "trend_ignition_shadow_report.md",
    }


def build_daily_pipeline_steps(config: PipelineConfig) -> list[PipelineStep]:
    tracking_enabled = getattr(config, "include_regime_shadow_tracking", True)
    if tracking_enabled and not config.include_regime_shadow_compare:
        raise ValueError("regime shadow tracking requires regime shadow comparison output")
    paths = _paths(config)
    token = _date_token(config.asof_date)
    base_target_weight = min(config.max_position_weight, config.leverage / max(config.top_n, 1))
    steps: list[PipelineStep] = []
    if config.include_benchmark_refresh:
        steps.append(
            PipelineStep(
                "benchmark_refresh",
                (
                    config.python_exe,
                    str(config.project_root / "update_benchmark_510300.py"),
                    "--benchmark",
                    str(paths["benchmark"]),
                    "--asof-date",
                    config.asof_date,
                    "--status-output",
                    str(paths["benchmark_refresh_status"]),
                ),
                max_attempts=2,
                retry_delay_seconds=2.0,
            )
        )
    steps.extend([
        PipelineStep(
            "update_panel",
            (
                config.python_exe,
                str(config.project_root / "update_model_panel_from_daily_data.py"),
                "--base-panel",
                str(paths["base_panel"]),
                "--daily-dir",
                str(paths["daily_dir"]),
                "--daily-start",
                effective_daily_start(config),
                "--daily-end",
                config.asof_date,
                "--output",
                str(paths["panel"]),
            ),
            max_attempts=2,
            retry_delay_seconds=2.0,
        ),
        PipelineStep(
            "trend_state",
            (
                config.python_exe,
                str(config.project_root / "trend_state.py"),
                "--data",
                str(paths["panel"]),
                "--output-dir",
                str(paths["trend_dir"]),
                "--asof-date",
                config.asof_date,
                "--max-abs-daily-return",
                str(config.max_abs_daily_return),
            ),
        ),
    ])
    if config.include_world_event_shadow:
        world_event_command = [
            config.python_exe,
            str(config.project_root / "world_event_shadow.py"),
            "--asof-date",
            config.asof_date,
            "--config",
            str(paths["world_event_config"]),
            "--output",
            str(paths["world_event_shadow_json"]),
            "--cache",
            str(paths["world_event_shadow_cache"]),
        ]
        if config.world_event_snapshot is not None:
            world_event_command.extend(
                ["--snapshot", str(_resolve(config.project_root, config.world_event_snapshot))]
            )
        steps.append(PipelineStep("world_event_shadow", tuple(world_event_command)))
    if config.include_regime_shadow_compare:
        steps.append(
            PipelineStep(
                "regime_shadow_compare",
                (
                    config.python_exe,
                    str(config.project_root / "run_daily_regime_shadow_compare.py"),
                    "--config",
                    str(_resolve(config.project_root, config.regime_shadow_config)),
                    "--data",
                    str(paths["panel"]),
                    "--benchmark",
                    str(paths["benchmark"]),
                    "--output-dir",
                    str(paths["regime_shadow_dir"]),
                    "--asof-date",
                    config.asof_date,
                    "--candidate-id",
                    config.regime_shadow_candidate_id,
                    "--python",
                    config.python_exe,
                ),
            )
        )
    if tracking_enabled:
        steps.append(
            PipelineStep(
                "regime_shadow_tracking",
                (
                    config.python_exe,
                    str(config.project_root / "update_regime_shadow_tracking.py"),
                    "--comparison-dir",
                    str(paths["regime_shadow_dir"]),
                    "--ledger",
                    str(paths["regime_shadow_tracking_ledger"]),
                    "--summary",
                    str(paths["regime_shadow_tracking_summary"]),
                    "--report",
                    str(paths["regime_shadow_tracking_report"]),
                    "--target-days",
                    "20",
                ),
            )
        )
    if config.train_model:
        steps.append(
            PipelineStep(
                "train_rank_model",
                (
                    config.python_exe,
                    str(config.project_root / "train_next_open_rank_model.py"),
                    "--data",
                    str(paths["panel"]),
                    "--output-dir",
                    str(paths["model_dir"]),
                    "--train-days",
                    str(config.train_days),
                    "--retrain-frequency",
                    str(config.retrain_frequency),
                    "--rebalance-frequency",
                    str(config.rebalance_frequency),
                    "--top-n",
                    str(config.top_n),
                    "--max-position-weight",
                    str(config.max_position_weight),
                    "--leverage",
                    str(config.leverage),
                    "--benchmark",
                    str(paths["benchmark"]),
                    "--market-ma-window",
                    str(config.market_ma_window),
                    "--market-below-ma-exposure",
                    str(config.market_below_ma_exposure),
                    "--market-risk-off-drawdown-20d",
                    str(config.market_risk_off_drawdown_20d),
                    "--market-crash-exposure",
                    str(config.market_crash_exposure),
                ),
            )
        )
        if config.include_strategy_arena and config.include_regime_shadow_compare:
            steps.append(
                PipelineStep(
                    "train_breadth_guard_challenger",
                    (
                        config.python_exe,
                        str(config.project_root / "train_next_open_rank_model.py"),
                        "--data",
                        str(paths["panel"]),
                        "--output-dir",
                        str(paths["breadth_model_dir"]),
                        "--train-days",
                        str(config.train_days),
                        "--retrain-frequency",
                        str(config.retrain_frequency),
                        "--rebalance-frequency",
                        str(config.rebalance_frequency),
                        "--top-n",
                        str(config.top_n),
                        "--max-position-weight",
                        str(config.max_position_weight),
                        "--leverage",
                        str(config.leverage),
                        "--benchmark",
                        str(paths["benchmark"]),
                        "--market-ma-window",
                        str(config.market_ma_window),
                        "--market-below-ma-exposure",
                        str(config.market_below_ma_exposure),
                        "--market-risk-off-drawdown-20d",
                        str(config.market_risk_off_drawdown_20d),
                        "--market-crash-exposure",
                        str(config.market_crash_exposure),
                        "--breadth-filter",
                        "--breadth-ma-window",
                        str(config.arena_breadth_ma_window),
                        "--breadth-threshold",
                        str(config.arena_breadth_threshold),
                        "--breadth-below-exposure",
                        str(config.arena_breadth_below_exposure),
                        "--breadth-crash-threshold",
                        str(config.arena_breadth_crash_threshold),
                        "--breadth-crash-exposure",
                        str(config.arena_breadth_crash_exposure),
                    ),
                )
            )
        if (
            config.include_dynamic_breadth_tracking
            and config.include_strategy_arena
            and config.include_regime_shadow_compare
        ):
            steps.append(
                PipelineStep(
                    "dynamic_breadth_overlay_tracking",
                    (
                        config.python_exe,
                        str(config.project_root / "track_dynamic_breadth_overlay.py"),
                        "--incumbent-equity",
                        str(paths["model_dir"] / "equity_curve.csv"),
                        "--challenger-equity",
                        str(paths["breadth_model_dir"] / "equity_curve.csv"),
                        "--ledger",
                        str(paths["dynamic_breadth_tracking_ledger"]),
                        "--summary",
                        str(paths["dynamic_breadth_tracking_summary"]),
                        "--report",
                        str(paths["dynamic_breadth_tracking_report"]),
                        "--config",
                        str(paths["dynamic_breadth_preregistration"]),
                    ),
                )
            )
    steps.append(
        PipelineStep(
            "rank_candidates",
            (
                config.python_exe,
                str(config.project_root / "generate_rank_model_candidates.py"),
                "--data",
                str(paths["panel"]),
                "--weights",
                str(paths["weights"]),
                "--output",
                str(paths["base_candidates"]),
                "--trend-state",
                str(paths["trend_dir"] / "trend_state.csv"),
                "--asof-date",
                config.asof_date,
                "--top-n",
                str(config.candidate_pool_n),
                "--selected-n",
                str(config.top_n),
                "--max-position-weight",
                str(config.max_position_weight),
                "--leverage",
                str(config.leverage),
                "--max-abs-daily-return",
                str(config.max_abs_daily_return),
                "--allowed-trend-states",
                config.allowed_trend_states,
            ),
        )
    )

    overlay_command = [
        config.python_exe,
        str(config.project_root / "apply_personal_trade_overlay.py"),
        "--candidates",
        str(paths["base_candidates"]),
        "--rules",
        str(paths["rules"]),
        "--names-source",
        str(paths["names_source"]),
        "--output",
        str(paths["overlay_candidates"]),
        "--reselect-top-n",
        str(config.top_n),
        "--base-target-weight",
        str(base_target_weight),
        "--selection-mode",
        "conservative_fill",
    ]
    if config.symbol_history is not None:
        overlay_command.extend(["--symbol-history", str(config.symbol_history)])
    steps.append(PipelineStep("personal_overlay", tuple(overlay_command)))
    shadow_review = config.shadow_account_review or latest_shadow_account_review(
        config.project_root, config.asof_date
    )
    shadow_args: tuple[str, ...] = ()
    if shadow_review is not None and shadow_review.exists():
        shadow_args = ("--shadow-account-review", str(shadow_review))

    steps.append(
        PipelineStep(
            "early_pattern_watchlist",
            (
                config.python_exe,
                str(config.project_root / "early_pattern_watchlist.py"),
                "--data",
                str(paths["panel"]),
                "--output",
                str(paths["early_watchlist"]),
                "--asof-date",
                config.asof_date,
                "--top-n",
                "80",
                "--max-abs-daily-return",
                str(config.max_abs_daily_return),
                "--names-source",
                str(paths["names_source"]),
            ),
        )
    )

    steps.append(
        PipelineStep(
            "main_net_volume_shadow",
            (
                config.python_exe,
                str(config.project_root / "main_net_volume_shadow.py"),
                "--data",
                str(paths["panel"]),
                "--output",
                str(paths["main_net_volume_shadow"]),
                "--asof-date",
                config.asof_date,
                "--names-source",
                str(paths["names_source"]),
                "--early-watchlist",
                str(paths["early_watchlist"]),
                "--minimum-history",
                "5",
                "--top-n",
                "30",
            ),
        )
    )

    steps.append(
        PipelineStep(
            "institutional_accumulation_shadow",
            (
                config.python_exe,
                str(config.project_root / "institutional_accumulation_shadow.py"),
                "--data",
                str(paths["panel"]),
                "--output",
                str(paths["institutional_accumulation_shadow"]),
                "--asof-date",
                config.asof_date,
                "--config",
                str(_resolve(config.project_root, config.institutional_accumulation_config)),
                "--names-source",
                str(paths["names_source"]),
            ),
        )
    )

    steps.append(
        PipelineStep(
            "institutional_accumulation_tracking",
            (
                config.python_exe,
                str(config.project_root / "track_institutional_accumulation_shadow.py"),
                "--data",
                str(paths["panel"]),
                "--watchlist-dir",
                str(paths["output_root"]),
                "--output",
                str(paths["institutional_accumulation_tracking"]),
                "--config",
                str(_resolve(config.project_root, config.institutional_accumulation_config)),
                "--horizons",
                "1,3,5,10",
                "--max-abs-daily-return",
                str(config.max_abs_daily_return),
            ),
        )
    )

    steps.append(
        PipelineStep(
            "czsc_structure_shadow",
            (
                config.python_exe,
                str(config.project_root / "czsc_structure_shadow.py"),
                "--data",
                str(paths["panel"]),
                "--candidates",
                str(paths["overlay_candidates"]),
                "--candidates",
                str(paths["early_watchlist"]),
                "--output",
                str(paths["czsc_structure_shadow"]),
                "--asof-date",
                config.asof_date,
                "--names-source",
                str(paths["names_source"]),
                "--min-bars",
                "100",
                "--max-abs-daily-return",
                str(config.max_abs_daily_return),
                "--top-n",
                "30",
            ),
        )
    )

    steps.append(
        PipelineStep(
            "hidden_accumulation_tracking",
            (
                config.python_exe,
                str(config.project_root / "track_hidden_accumulation_watch.py"),
                "--data",
                str(paths["panel"]),
                "--watchlist-dir",
                str(paths["output_root"]),
                "--output",
                str(paths["hidden_accumulation_tracking"]),
                "--horizons",
                "1,3,5",
                "--max-abs-daily-return",
                str(config.max_abs_daily_return),
            ),
        )
    )

    if config.include_factor_decay_monitor:
        steps.append(
            PipelineStep(
                "factor_decay_monitor",
                (
                    config.python_exe,
                    str(config.project_root / "monitor_factor_decay.py"),
                    "--data",
                    str(paths["panel"]),
                    "--base-panel",
                    str(paths["base_panel"]),
                    "--asof-date",
                    config.asof_date,
                    "--output-dir",
                    str(paths["output_root"]),
                    "--replacement-preregistration",
                    str(paths["factor_replacement_preregistration"]),
                ),
            )
        )

    steps.append(
        PipelineStep(
            "daily_chinese_report",
            (
                config.python_exe,
                str(config.project_root / "build_daily_personal_overlay_report.py"),
                "--base-candidates",
                str(paths["base_candidates"]),
                "--overlay-candidates",
                str(paths["overlay_candidates"]),
                "--metrics",
                str(paths["metrics"]),
                "--rules",
                str(paths["rules"]),
                "--output-dir",
                str(paths["output_root"]),
                "--asof-date",
                config.asof_date,
                "--early-watchlist",
                str(paths["early_watchlist"]),
                "--names-source",
                str(paths["names_source"]),
                "--daily-data-dir",
                str(paths["daily_data_dir"]),
                "--fallback-data-dir",
                str(paths["fallback_data_dir"]),
                "--fetch-status-utils",
                str(paths["fetch_status_utils"]),
                *shadow_args,
            ),
        )
    )
    steps.append(
        PipelineStep(
            "merged_daily_outputs",
            (
                config.python_exe,
                str(config.project_root / "merged_daily_outputs.py"),
                "--trend-state",
                str(paths["trend_dir"] / "trend_state.csv"),
                "--early-watchlist",
                str(paths["early_watchlist"]),
                "--base-candidates",
                str(paths["base_candidates"]),
                "--overlay-candidates",
                str(paths["overlay_candidates"]),
                "--output-dir",
                str(paths["output_root"]),
                "--asof-date",
                config.asof_date,
                "--names-source",
                str(paths["names_source"]),
                *shadow_args,
            ),
        )
    )
    if config.include_trend_ignition_shadow:
        steps.append(
            PipelineStep(
                "trend_ignition_shadow",
                (
                    config.python_exe,
                    str(config.project_root / "score_trend_ignition_shadow.py"),
                    "--data",
                    str(paths["panel"]),
                    "--watchlist-dir",
                    str(paths["output_root"]),
                    "--scorer",
                    str(paths["trend_ignition_scorer"]),
                    "--scorer-summary",
                    str(paths["trend_ignition_scorer_summary"]),
                    "--output-dir",
                    str(paths["trend_ignition_shadow_dir"]),
                    "--start",
                    config.asof_date,
                    "--end",
                    config.asof_date,
                    "--selection-status",
                    config.trend_ignition_selection_status,
                    "--max-abs-daily-return",
                    str(config.max_abs_daily_return),
                ),
            )
        )
    if config.include_strategy_family_forward_report:
        steps.append(
            PipelineStep(
                "strategy_family_forward_report",
                (
                    config.python_exe,
                    str(config.project_root / "evaluate_strategy_family_forward_returns.py"),
                    "--data",
                    str(paths["panel"]),
                    "--watchlist-dir",
                    str(paths["output_root"]),
                    "--output-dir",
                    str(paths["output_root"]),
                    "--end",
                    config.asof_date,
                    "--horizons",
                    "1,3,5,10",
                    "--token",
                    token,
                    "--max-abs-daily-return",
                    str(config.max_abs_daily_return),
                ),
            )
        )
    if config.include_strategy_arena and config.include_regime_shadow_compare:
        arena_command = [
            config.python_exe,
            str(config.project_root / "build_strategy_arena_report.py"),
            "--asof-date",
            config.asof_date,
            "--champion-equity",
            str(paths["model_dir"] / "equity_curve.csv"),
            "--champion-metrics",
            str(paths["model_dir"] / "metrics.json"),
            "--breadth-equity",
            str(paths["breadth_model_dir"] / "equity_curve.csv"),
            "--breadth-metrics",
            str(paths["breadth_model_dir"] / "metrics.json"),
            "--baseline-equity",
            str(paths["regime_shadow_dir"] / "baseline" / "equity_curve.csv"),
            "--baseline-metrics",
            str(paths["regime_shadow_dir"] / "baseline" / "metrics.json"),
            "--dynamic-equity",
            str(paths["regime_shadow_dir"] / "dynamic" / "equity_curve.csv"),
            "--dynamic-metrics",
            str(paths["regime_shadow_dir"] / "dynamic" / "metrics.json"),
            "--tracking-summary",
            str(paths["regime_shadow_tracking_summary"]),
            "--czsc-metadata",
            str(paths["czsc_structure_shadow_metadata"]),
            "--flow-metadata",
            str(paths["main_net_volume_shadow_metadata"]),
            "--output-dir",
            str(paths["output_root"]),
            "--history",
            str(paths["strategy_arena_history"]),
            "--min-common-days",
            "252",
        ]
        if config.include_strategy_family_forward_report:
            arena_command.extend(["--family-health", str(paths["strategy_family_health"])])
        steps.append(PipelineStep("strategy_arena", tuple(arena_command)))
    if config.include_strategy_stability_report:
        steps.append(
            PipelineStep(
                "strategy_stability_report",
                (
                    config.python_exe,
                    str(config.project_root / "summarize_core_risk_filter_stability.py"),
                    "--output-prefix",
                    str(paths["strategy_stability_prefix"]),
                ),
            )
        )
    if config.include_research_db_sync:
        steps.append(
            PipelineStep(
                "research_database_sync",
                (
                    config.python_exe,
                    str(config.project_root / "sync_daily_research_database.py"),
                    "--db",
                    str(paths["research_db"]),
                    "--daily-dir",
                    str(paths["daily_dir"]),
                    "--output-dir",
                    str(paths["output_root"]),
                    "--asof-date",
                    config.asof_date,
                    "--status-output",
                    str(paths["research_database_sync_status"]),
                ),
                max_attempts=2,
                retry_delay_seconds=2.0,
            )
        )
    return steps


def build_marketlens_export_step(config: PipelineConfig) -> PipelineStep | None:
    if not config.include_marketlens_export:
        return None
    paths = _paths(config)
    return PipelineStep(
        "marketlens_export",
        (
            config.python_exe,
            str(config.project_root / "export_marketlens_model3_feed.py"),
            "--asof-date",
            config.asof_date,
            "--output-root",
            str(paths["output_root"]),
            "--state-record",
            str(paths["marketlens_state_input"]),
            "--output",
            str(paths["marketlens_feed"]),
            "--dashboard-output",
            str(paths["marketlens_dashboard_feed"]),
            "--require-complete-inputs",
        ),
        max_attempts=2,
        retry_delay_seconds=2.0,
    )


def pipeline_step_dependencies(steps: list[PipelineStep]) -> dict[str, tuple[str, ...]]:
    names = [step.name for step in steps]
    if len(names) != len(set(names)):
        raise ValueError("Pipeline step names must be unique")
    available = set(names)
    dependencies: dict[str, tuple[str, ...]] = {}
    previous_name: str | None = None
    for step in steps:
        configured = PIPELINE_STEP_DEPENDENCIES.get(step.name)
        if configured is None:
            dependencies[step.name] = (previous_name,) if previous_name else ()
        else:
            dependencies[step.name] = tuple(
                dependency for dependency in configured if dependency in available
            )
        previous_name = step.name

    terminal_steps = {"research_database_sync"}
    for step in steps:
        if step.name in terminal_steps:
            dependencies[step.name] = tuple(
                prior_name for prior_name in names if prior_name != step.name
            )
    return dependencies


def _run_pipeline_step(
    step: PipelineStep,
    project_root: Path,
    index: int,
    total: int,
) -> dict[str, object]:
    if step.max_attempts < 1:
        raise ValueError(f"Pipeline step {step.name} max_attempts must be at least 1")
    if step.retry_delay_seconds < 0:
        raise ValueError(
            f"Pipeline step {step.name} retry_delay_seconds cannot be negative"
        )
    started_at = datetime.now().isoformat(timespec="milliseconds")
    started = perf_counter()
    print(f"[{index}/{total}] {step.name}")
    print(" ".join(step.command))
    attempt = 0
    while attempt < step.max_attempts:
        attempt += 1
        try:
            subprocess.run(step.command, cwd=project_root, check=True)
            break
        except subprocess.CalledProcessError as exc:
            if attempt < step.max_attempts:
                print(
                    f"[{index}/{total}] {step.name} failed with exit code "
                    f"{exc.returncode}; retrying ({attempt + 1}/{step.max_attempts})"
                )
                if step.retry_delay_seconds:
                    sleep(step.retry_delay_seconds)
                continue
            execution = {
                "name": step.name,
                "status": "failed",
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="milliseconds"),
                "duration_seconds": round(perf_counter() - started, 6),
                "returncode": int(exc.returncode),
                "cache_hit": False,
                "attempts": attempt,
                "retried": attempt > 1,
            }
            raise PipelineStepExecutionError(
                step,
                int(exc.returncode),
                execution=execution,
            ) from exc
    return {
        "name": step.name,
        "status": "success",
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="milliseconds"),
        "duration_seconds": round(perf_counter() - started, 6),
        "returncode": 0,
        "cache_hit": False,
        "attempts": attempt,
        "retried": attempt > 1,
    }


def run_steps(
    steps: list[PipelineStep],
    project_root: Path,
    dry_run: bool = False,
    max_parallel_steps: int = 1,
    step_cache: PipelineStepCache | None = None,
) -> list[dict[str, object]]:
    if max_parallel_steps < 1:
        raise ValueError("max_parallel_steps must be at least 1")
    if dry_run:
        for index, step in enumerate(steps, start=1):
            print(f"[{index}/{len(steps)}] {step.name}")
            print(" ".join(step.command))
        return []
    if not steps:
        return []

    dependencies = pipeline_step_dependencies(steps)
    order = {step.name: index for index, step in enumerate(steps)}
    pending = {step.name: step for step in steps}
    completed: set[str] = set()
    executions: list[dict[str, object]] = []
    execution_by_name: dict[str, dict[str, object]] = {}
    running: dict[Future[dict[str, object]], PipelineStep] = {}
    executor = ThreadPoolExecutor(max_workers=min(max_parallel_steps, len(steps)))
    try:
        while pending or running:
            ready = [
                step
                for step in steps
                if step.name in pending
                and set(dependencies[step.name]).issubset(completed)
            ]
            restored_any = False
            if step_cache is not None:
                for step in ready:
                    dependency_cache_hits = (
                        bool(execution_by_name[dependency].get("cache_hit"))
                        for dependency in dependencies[step.name]
                        if dependency in CACHEABLE_STEPS
                    )
                    started_at = datetime.now().isoformat(timespec="milliseconds")
                    started = perf_counter()
                    if not step_cache.restore(
                        step.name,
                        step.command,
                        dependency_cache_hits,
                    ):
                        continue
                    pending.pop(step.name)
                    execution = {
                        "name": step.name,
                        "status": "cached",
                        "started_at": started_at,
                        "finished_at": datetime.now().isoformat(timespec="milliseconds"),
                        "duration_seconds": round(perf_counter() - started, 6),
                        "returncode": 0,
                        "cache_hit": True,
                        "attempts": 0,
                        "retried": False,
                    }
                    print(f"[{order[step.name] + 1}/{len(steps)}] {step.name} [cache hit]")
                    executions.append(execution)
                    execution_by_name[step.name] = execution
                    completed.add(step.name)
                    restored_any = True
                if restored_any:
                    continue
            while ready and len(running) < max_parallel_steps:
                step = ready.pop(0)
                pending.pop(step.name)
                future = executor.submit(
                    _run_pipeline_step,
                    step,
                    project_root,
                    order[step.name] + 1,
                    len(steps),
                )
                running[future] = step

            if not running:
                unresolved = {
                    name: dependencies[name]
                    for name in pending
                }
                raise RuntimeError(f"Pipeline dependency cycle or missing dependency: {unresolved}")

            finished, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
            for future in finished:
                step = running.pop(future)
                try:
                    execution = future.result()
                except PipelineStepExecutionError as exc:
                    if exc.execution is not None:
                        executions.append(exc.execution)
                    exc.completed_executions = sorted(
                        executions,
                        key=lambda item: order.get(str(item.get("name")), len(order)),
                    )
                    for outstanding in running:
                        outstanding.cancel()
                    raise
                executions.append(execution)
                execution_by_name[step.name] = execution
                completed.add(step.name)
                if step_cache is not None:
                    step_cache.record_success(step.name, step.command)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    return sorted(
        executions,
        key=lambda item: order.get(str(item.get("name")), len(order)),
    )


def execution_summary(
    step_executions: list[dict[str, object]] | None,
    max_parallel_steps: int,
) -> dict[str, object]:
    executions = list(step_executions or [])
    durations = [float(item.get("duration_seconds") or 0.0) for item in executions]
    status_counts: dict[str, int] = {}
    for item in executions:
        status = str(item.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    started = [str(item["started_at"]) for item in executions if item.get("started_at")]
    finished = [str(item["finished_at"]) for item in executions if item.get("finished_at")]
    wall_duration = 0.0
    if started and finished:
        wall_duration = (
            datetime.fromisoformat(max(finished)) - datetime.fromisoformat(min(started))
        ).total_seconds()
    return {
        "max_parallel_steps": max_parallel_steps,
        "wall_duration_seconds": round(max(wall_duration, 0.0), 6),
        "summed_step_duration_seconds": round(sum(durations), 6),
        "status_counts": status_counts,
        "cache_hits": sum(bool(item.get("cache_hit")) for item in executions),
        "retried_steps": sum(bool(item.get("retried")) for item in executions),
        "total_attempts": sum(int(item.get("attempts") or 0) for item in executions),
        "steps": executions,
    }


def build_daily_run_state(
    config: PipelineConfig,
    steps: list[PipelineStep],
    argv: list[str] | None = None,
    test_status: str = "not_run_by_pipeline",
    run_status: str = "success",
    failure: dict[str, object] | None = None,
    step_executions: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    paths = _paths(config)
    tracking_enabled = getattr(config, "include_regime_shadow_tracking", True)
    failure_step = str((failure or {}).get("step") or "")
    if not config.include_marketlens_export:
        marketlens_export_status = "skipped"
    elif failure_step == "marketlens_export":
        marketlens_export_status = "failed"
    elif run_status == "failed":
        marketlens_export_status = "not_run"
    else:
        marketlens_export_status = "pending"
    artifact_overrides = {
        step_name: _pipeline_artifact_status_override(
            steps,
            run_status,
            failure,
            step_executions,
            step_name,
        )
        for step_name in (
            "main_net_volume_shadow",
            "institutional_accumulation_shadow",
            "institutional_accumulation_tracking",
            "czsc_structure_shadow",
            "strategy_arena",
            "world_event_shadow",
        )
    }
    priority_path = _latest_artifact(paths["merged_priority_watchlist"])
    early_path = _latest_artifact(paths["early_watchlist"])
    main_net_volume_shadow_path = _latest_artifact(paths["main_net_volume_shadow"])
    institutional_accumulation_path = _latest_artifact(
        paths["institutional_accumulation_shadow"]
    )
    institutional_accumulation_tracking_path = _latest_artifact(
        paths["institutional_accumulation_tracking"]
    )
    czsc_structure_shadow_path = _latest_artifact(paths["czsc_structure_shadow"])
    hidden_tracking_path = _latest_artifact(paths["hidden_accumulation_tracking"])
    model_path = _latest_artifact(paths["merged_model_decision_table"])
    strategy_family_summary_path = _latest_artifact(paths["strategy_family_forward_summary"])
    strategy_family_health_path = _latest_artifact(paths["strategy_family_health"])
    selected_path = _latest_artifact(paths["selected"])
    changes_path = _latest_artifact(paths["changes"])
    priority_cn_base = paths["merged_priority_watchlist"].with_name(
        f"{paths['merged_priority_watchlist'].stem}_cn{paths['merged_priority_watchlist'].suffix}"
    )
    priority_rows = _read_csv_rows(priority_path)
    early_rows = _read_csv_rows(early_path)
    main_override = artifact_overrides["main_net_volume_shadow"]
    main_net_volume_shadow_rows = (
        [] if main_override else _read_csv_rows(main_net_volume_shadow_path)
    )
    main_net_volume_shadow_metadata = (
        {"status": main_override}
        if main_override
        else _read_json_object(paths["main_net_volume_shadow_metadata"])
    )
    institutional_override = artifact_overrides["institutional_accumulation_shadow"]
    institutional_accumulation_rows = (
        [] if institutional_override else _read_csv_rows(institutional_accumulation_path)
    )
    institutional_accumulation_metadata = (
        {"status": institutional_override}
        if institutional_override
        else _read_json_object(paths["institutional_accumulation_metadata"])
    )
    institutional_tracking_override = artifact_overrides[
        "institutional_accumulation_tracking"
    ]
    institutional_accumulation_tracking_rows = (
        []
        if institutional_tracking_override
        else _read_csv_rows(institutional_accumulation_tracking_path)
    )
    institutional_accumulation_tracking_summary = (
        {"status": institutional_tracking_override}
        if institutional_tracking_override
        else _read_json_object(paths["institutional_accumulation_tracking_summary"])
    )
    czsc_override = artifact_overrides["czsc_structure_shadow"]
    czsc_structure_shadow_rows = (
        [] if czsc_override else _read_csv_rows(czsc_structure_shadow_path)
    )
    czsc_structure_shadow_metadata = (
        {"status": czsc_override}
        if czsc_override
        else _read_json_object(paths["czsc_structure_shadow_metadata"])
    )
    arena_enabled = config.include_strategy_arena and config.include_regime_shadow_compare
    arena_override = artifact_overrides["strategy_arena"] if arena_enabled else None
    strategy_arena_metadata = {"status": arena_override} if arena_override else {}
    if arena_enabled and not arena_override:
        strategy_arena_metadata = _read_json_object(paths["strategy_arena_metadata"])
    if not arena_override and strategy_arena_metadata.get("asof_date") != config.asof_date:
        strategy_arena_metadata = {}
    strategy_arena_portfolio_path = _latest_artifact(paths["strategy_arena_portfolio"])
    strategy_arena_portfolio_rows = (
        _read_csv_rows(strategy_arena_portfolio_path)
        if strategy_arena_metadata and not arena_override
        else []
    )
    hidden_tracking_rows = _read_csv_rows(hidden_tracking_path)
    model_rows = _read_csv_rows(model_path)
    strategy_family_summary_rows = _read_csv_rows(strategy_family_summary_path)
    strategy_family_health_rows = _read_csv_rows(strategy_family_health_path)
    selected_rows = _read_csv_rows(selected_path)
    change_rows = _read_csv_rows(changes_path)
    regime_shadow = _read_json_object(paths["regime_shadow_comparison"])
    if tracking_enabled:
        tracking_status_override, tracking_may_be_current = _tracking_state_gate(
            steps,
            run_status,
            failure,
        )
        if tracking_may_be_current:
            tracking_summary = _read_json_object(paths["regime_shadow_tracking_summary"])
            if tracking_summary.get("latest_asof_date") != config.asof_date:
                tracking_status = "missing"
                tracking_valid = None
                tracking_target = None
                tracking_remaining = None
                tracking_invalid = None
                tracking_return_delta = None
                tracking_benchmark_fresh = None
                tracking_risk_regime = None
                tracking_target_leverage = None
                tracking_ledger_artifact = None
                tracking_summary_artifact = None
                tracking_report_artifact = None
            else:
                tracking_risk_state = tracking_summary.get("latest_risk_state")
                if not isinstance(tracking_risk_state, dict):
                    tracking_risk_state = {}
                tracking_status = tracking_summary.get("status", "missing")
                tracking_valid = tracking_summary.get("valid_observation_count")
                tracking_target = tracking_summary.get("target_days")
                tracking_remaining = tracking_summary.get("remaining_days")
                tracking_invalid = tracking_summary.get("invalid_observation_count")
                tracking_return_delta = tracking_summary.get("cumulative_return_delta")
                tracking_benchmark_fresh = tracking_summary.get("latest_benchmark_fresh")
                tracking_risk_regime = tracking_risk_state.get("risk_regime")
                tracking_target_leverage = tracking_risk_state.get("target_leverage")
                tracking_ledger_artifact = str(_latest_artifact(paths["regime_shadow_tracking_ledger"]))
                tracking_summary_artifact = str(_latest_artifact(paths["regime_shadow_tracking_summary"]))
                tracking_report_artifact = str(_latest_artifact(paths["regime_shadow_tracking_report"]))
        else:
            tracking_status = tracking_status_override or "missing"
            tracking_valid = None
            tracking_target = None
            tracking_remaining = None
            tracking_invalid = None
            tracking_return_delta = None
            tracking_benchmark_fresh = None
            tracking_risk_regime = None
            tracking_target_leverage = None
            tracking_ledger_artifact = None
            tracking_summary_artifact = None
            tracking_report_artifact = None
    else:
        tracking_status = "skipped"
        tracking_valid = None
        tracking_target = None
        tracking_remaining = None
        tracking_invalid = None
        tracking_return_delta = None
        tracking_benchmark_fresh = None
        tracking_risk_regime = None
        tracking_target_leverage = None
        tracking_ledger_artifact = None
        tracking_summary_artifact = None
        tracking_report_artifact = None
    dynamic_breadth_tracking, dynamic_breadth_artifacts = _dynamic_breadth_tracking_state(
        config,
        steps,
        paths,
        run_status,
        failure,
        step_executions,
    )
    world_event_shadow, world_event_artifacts = _world_event_shadow_state(
        config,
        steps,
        paths,
        run_status,
        failure,
        step_executions,
    )
    if not config.include_benchmark_refresh:
        benchmark_refresh = {"status": "skipped"}
    elif paths["benchmark_refresh_status"].exists():
        benchmark_refresh = _read_json_object(paths["benchmark_refresh_status"])
    else:
        benchmark_refresh = {"status": "missing"}
    research_database_sync, research_database_sync_artifact = _research_database_sync_state(
        config,
        steps,
        paths,
        run_status,
        failure,
    )
    trend_ignition_shadow, trend_ignition_shadow_manifest_artifact = (
        _trend_ignition_shadow_state(config, steps, paths, run_status, failure)
    )
    trend_ignition_coverage = trend_ignition_shadow.get("coverage")
    if not isinstance(trend_ignition_coverage, dict):
        trend_ignition_coverage = {}
    trend_ignition_scorer = trend_ignition_shadow.get("scorer")
    if not isinstance(trend_ignition_scorer, dict):
        trend_ignition_scorer = {}
    factor_decay = factor_decay_monitor_status(
        paths["factor_decay_monitor_json"],
        config.asof_date,
        enabled=config.include_factor_decay_monitor,
    )
    regime_delta = regime_shadow.get("delta")
    if not isinstance(regime_delta, dict):
        regime_delta = {}
    regime_latest = regime_shadow.get("latest_dynamic_state")
    if not isinstance(regime_latest, dict):
        regime_latest = {}
    if regime_shadow.get("benchmark_fresh") is not True:
        regime_latest = {}
    return {
        "schema_version": 2,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "run_status": run_status,
        "failure": failure,
        "asof_date": config.asof_date,
        "project_root": str(config.project_root),
        "run_type": "full_train" if config.train_model else "skip_train",
        "data_source": str(paths["names_source"]),
        "data_source_exists": paths["names_source"].exists(),
        "fallback_fetch_status": fetch_status_summary(paths["fetch_status_utils"]),
        "argv": argv or [],
        "steps": [step.name for step in steps],
        "commands": [" ".join(step.command) for step in steps],
        "execution": execution_summary(step_executions, config.max_parallel_steps),
        "verification": {
            "tests": test_status,
            "priority_rows": len(priority_rows),
            "missing_stock_names": _missing_count(priority_rows, "stock_name"),
            "priority_bucket_counts": _value_counts(priority_rows, "priority_bucket"),
            "priority_strategy_family_counts": _value_counts(priority_rows, "strategy_family"),
            "early_pattern_rows": len(early_rows),
            "early_pattern_counts": _value_counts(early_rows, "pattern_type"),
            "main_net_volume_shadow_status": main_net_volume_shadow_metadata.get("status"),
            "main_net_volume_shadow_rows": len(main_net_volume_shadow_rows),
            "main_net_volume_source_latest_date": main_net_volume_shadow_metadata.get(
                "source_latest_date"
            ),
            "main_net_volume_source_sessions": main_net_volume_shadow_metadata.get(
                "available_source_sessions"
            ),
            "main_net_volume_latest_coverage": main_net_volume_shadow_metadata.get(
                "latest_source_coverage"
            ),
            "main_net_volume_eligible_5d_rows": main_net_volume_shadow_metadata.get(
                "eligible_5d_rows"
            ),
            "main_net_volume_early_pattern_overlap": main_net_volume_shadow_metadata.get(
                "early_pattern_overlap_count"
            ),
            "main_net_volume_selection_effect": main_net_volume_shadow_metadata.get(
                "selection_effect"
            ),
            "institutional_accumulation_status": institutional_accumulation_metadata.get(
                "status"
            ),
            "institutional_accumulation_rows": len(institutional_accumulation_rows),
            "institutional_accumulation_active_signals": institutional_accumulation_metadata.get(
                "active_signal_rows"
            ),
            "institutional_accumulation_flow_sessions": institutional_accumulation_metadata.get(
                "available_flow_sessions"
            ),
            "institutional_accumulation_source_columns_available": institutional_accumulation_metadata.get(
                "source_columns_available"
            ),
            "institutional_accumulation_source_column_latest_dates": institutional_accumulation_metadata.get(
                "source_column_latest_dates"
            ),
            "institutional_accumulation_registration_id": institutional_accumulation_metadata.get(
                "registration_id"
            ),
            "institutional_accumulation_validation_start_date": institutional_accumulation_metadata.get(
                "validation_start_date"
            ),
            "institutional_accumulation_config_sha256": institutional_accumulation_metadata.get(
                "config_sha256"
            ),
            "institutional_accumulation_implementation_sha256": institutional_accumulation_metadata.get(
                "implementation_sha256"
            ),
            "institutional_accumulation_selection_effect": institutional_accumulation_metadata.get(
                "selection_effect"
            ),
            "institutional_accumulation_tracking_rows": len(
                institutional_accumulation_tracking_rows
            ),
            "institutional_accumulation_tracking_status": institutional_accumulation_tracking_summary.get(
                "status"
            ),
            "institutional_accumulation_completed_5d": institutional_accumulation_tracking_summary.get(
                "primary_completed_samples"
            ),
            "institutional_accumulation_gate_allowed": institutional_accumulation_tracking_summary.get(
                "gate_evaluation_allowed"
            ),
            "institutional_accumulation_promotion_allowed": institutional_accumulation_tracking_summary.get(
                "promotion_allowed"
            ),
            "czsc_structure_shadow_status": czsc_structure_shadow_metadata.get("status"),
            "czsc_structure_shadow_rows": len(czsc_structure_shadow_rows),
            "czsc_structure_analyzed_rows": czsc_structure_shadow_metadata.get(
                "analyzed_count"
            ),
            "czsc_structure_pattern_confluence": czsc_structure_shadow_metadata.get(
                "pattern_confluence_count"
            ),
            "czsc_structure_second_buy_count": czsc_structure_shadow_metadata.get(
                "second_buy_count"
            ),
            "czsc_structure_third_buy_consistent_count": czsc_structure_shadow_metadata.get(
                "third_buy_zone_consistent_count"
            ),
            "czsc_structure_selection_effect": czsc_structure_shadow_metadata.get(
                "selection_effect"
            ),
            "strategy_arena_status": (
                strategy_arena_metadata.get("status")
                if arena_enabled
                else "skipped"
            ),
            "strategy_arena_portfolio_entrants": len(strategy_arena_portfolio_rows),
            "strategy_arena_production_champion": strategy_arena_metadata.get(
                "production_champion"
            ),
            "strategy_arena_pareto_by_league": strategy_arena_metadata.get(
                "historical_pareto_by_league"
            ),
            "strategy_arena_observation_status": strategy_arena_metadata.get(
                "independent_observation_status"
            ),
            "strategy_arena_observation_count": strategy_arena_metadata.get(
                "independent_observation_count"
            ),
            "strategy_arena_observation_target": strategy_arena_metadata.get(
                "independent_observation_target"
            ),
            "strategy_arena_automatic_promotion": strategy_arena_metadata.get(
                "automatic_promotion"
            ),
            "hidden_trade_watch_tracking_rows": len(hidden_tracking_rows),
            "hidden_trade_watch_tracking_status_counts": _value_counts(hidden_tracking_rows, "tracking_status"),
            "model_decision_rows": len(model_rows),
            "strategy_family_summary_rows": len(strategy_family_summary_rows),
            "strategy_family_health_rows": len(strategy_family_health_rows),
            "strategy_family_health_counts": _value_counts(strategy_family_health_rows, "family_health_status"),
            "selected_rows": len(selected_rows),
            "change_rows": len(change_rows),
            "regime_shadow_decision": regime_shadow.get("decision"),
            "regime_shadow_risk_regime": regime_latest.get("risk_regime"),
            "regime_shadow_target_leverage": regime_latest.get("target_leverage"),
            "regime_shadow_total_return_delta": regime_delta.get("total_return"),
            "regime_shadow_max_drawdown_delta": regime_delta.get("max_drawdown"),
            "regime_shadow_benchmark_last_date": regime_shadow.get("benchmark_last_date"),
            "regime_shadow_benchmark_fresh": regime_shadow.get("benchmark_fresh"),
            "regime_shadow_tracking_status": tracking_status,
            "regime_shadow_tracking_valid_observations": tracking_valid,
            "regime_shadow_tracking_target_days": tracking_target,
            "regime_shadow_tracking_remaining_days": tracking_remaining,
            "regime_shadow_tracking_invalid_observations": tracking_invalid,
            "regime_shadow_tracking_cumulative_return_delta": tracking_return_delta,
            "regime_shadow_tracking_benchmark_fresh": tracking_benchmark_fresh,
            "regime_shadow_tracking_risk_regime": tracking_risk_regime,
            "regime_shadow_tracking_target_leverage": tracking_target_leverage,
            "dynamic_breadth_tracking_status": dynamic_breadth_tracking.get("status"),
            "dynamic_breadth_tracking_source_latest_date": dynamic_breadth_tracking.get(
                "source_latest_date"
            ),
            "dynamic_breadth_tracking_valid_observations": dynamic_breadth_tracking.get(
                "valid_observation_count"
            ),
            "dynamic_breadth_tracking_target_days": dynamic_breadth_tracking.get(
                "target_valid_trade_days"
            ),
            "dynamic_breadth_tracking_remaining_days": dynamic_breadth_tracking.get(
                "remaining_observation_days"
            ),
            "dynamic_breadth_tracking_cumulative_excess_return": dynamic_breadth_tracking.get(
                "cumulative_excess_return"
            ),
            "dynamic_breadth_tracking_positive_excess_day_ratio": dynamic_breadth_tracking.get(
                "positive_excess_day_ratio"
            ),
            "dynamic_breadth_tracking_gate_allowed": dynamic_breadth_tracking.get(
                "gate_evaluation_allowed"
            ),
            "dynamic_breadth_tracking_provisional_only": dynamic_breadth_tracking.get(
                "provisional_only"
            ),
            "dynamic_breadth_tracking_promotion_allowed": dynamic_breadth_tracking.get(
                "promotion_allowed"
            ),
            "world_event_shadow_status": world_event_shadow.get("status"),
            "world_event_global_risk_score": world_event_shadow.get(
                "global_risk_score"
            ),
            "world_event_risk_level": world_event_shadow.get("risk_level"),
            "world_event_source_count": world_event_shadow.get("event_source_count"),
            "world_event_confidence": world_event_shadow.get("event_confidence"),
            "world_event_data_freshness": world_event_shadow.get("data_freshness"),
            "world_event_selection_effect": world_event_shadow.get(
                "selection_effect"
            ),
            "benchmark_refresh_status": benchmark_refresh.get("status"),
            "benchmark_latest_date": benchmark_refresh.get("latest_date"),
            "benchmark_source_agreement": benchmark_refresh.get("source_agreement"),
            "benchmark_rows_added": benchmark_refresh.get("rows_added"),
            "research_database_sync_status": research_database_sync.get("status"),
            "research_database_latest_date": research_database_sync.get("database_latest_date"),
            "research_database_source_latest_date": research_database_sync.get(
                "source_latest_date"
            ),
            "research_database_price_usable_latest_date": research_database_sync.get(
                "price_usable_latest_date"
            ),
            "research_database_factor_usable_latest_date": research_database_sync.get(
                "factor_usable_latest_date"
            ),
            "research_database_price_usable_rows": research_database_sync.get(
                "price_usable_rows"
            ),
            "research_database_price_usable_ratio": research_database_sync.get(
                "price_usable_ratio"
            ),
            "research_database_factor_usable_rows": research_database_sync.get(
                "factor_usable_rows"
            ),
            "research_database_factor_usable_ratio": research_database_sync.get(
                "factor_usable_ratio"
            ),
            "research_database_factor_column_coverage": research_database_sync.get(
                "factor_column_coverage"
            ),
            "research_database_asof_rows": research_database_sync.get("database_asof_rows"),
            "research_database_daily_rows": research_database_sync.get("database_daily_rows"),
            "research_database_observation_rows": research_database_sync.get("database_observation_rows"),
            "research_database_removed_source_alias_rows": research_database_sync.get(
                "observation_removed_source_alias_rows"
            ),
            "marketlens_export_status": marketlens_export_status,
            "trend_ignition_shadow_status": trend_ignition_shadow.get("status"),
            "trend_ignition_shadow_source_rows": trend_ignition_coverage.get("source_rows"),
            "trend_ignition_shadow_eligible_rows": trend_ignition_coverage.get("eligible_rows"),
            "trend_ignition_shadow_eligibility_ratio": trend_ignition_coverage.get(
                "eligibility_ratio"
            ),
            "trend_ignition_shadow_bucket_counts": trend_ignition_coverage.get(
                "score_bucket_counts"
            ),
            "trend_ignition_shadow_training_end_date": trend_ignition_scorer.get(
                "training_end_date"
            ),
            "trend_ignition_shadow_selection_status": trend_ignition_scorer.get(
                "selection_status"
            ),
            "trend_ignition_shadow_research_gate_passed": trend_ignition_scorer.get(
                "passes_research_gate"
            ),
            "factor_decay_monitor": factor_decay,
            "factor_decay_calculation_mode": factor_decay.get("calculation_mode"),
            "factor_decay_recompute_start": factor_decay.get("recompute_start"),
            "factor_decay_checkpoint_parent_asof_date": factor_decay.get(
                "checkpoint_parent_asof_date"
            ),
            "factor_current_availability": (
                factor_decay.get("factor_metadata", {}).get("factor_current_availability")
                if isinstance(factor_decay.get("factor_metadata"), dict)
                else None
            ),
            "factor_current_coverage": (
                factor_decay.get("factor_metadata", {}).get("factor_current_coverage")
                if isinstance(factor_decay.get("factor_metadata"), dict)
                else None
            ),
            "factor_input_latest_dates": (
                factor_decay.get("factor_metadata", {}).get("factor_input_latest_dates")
                if isinstance(factor_decay.get("factor_metadata"), dict)
                else None
            ),
            "factor_replacement_tracking_status": factor_decay.get(
                "replacement_tracking_status"
            ),
            "factor_replacement_shortlist": factor_decay.get(
                "replacement_shortlist"
            ),
            "factor_replacement_preregistered_candidates": factor_decay.get(
                "replacement_preregistered_candidates"
            ),
        },
        "top10": _top_rows(priority_rows, ["symbol", "stock_name", "strategy_family", "priority_bucket"]),
        "artifacts": {
            "pipeline_step_cache": (
                str(paths["pipeline_step_cache"])
                if config.enable_step_cache
                else None
            ),
            "priority": str(priority_path),
            "priority_cn": str(_latest_artifact(priority_cn_base)),
            "report": str(_latest_artifact(paths["daily_report"])),
            "selected": str(selected_path),
            "changes": str(changes_path),
            "early_watchlist": str(early_path),
            "main_net_volume_shadow": _artifact_reference(
                main_net_volume_shadow_path, hidden=bool(main_override)
            ),
            "main_net_volume_shadow_cn": _artifact_reference(
                paths["main_net_volume_shadow"].with_name(
                    f"{paths['main_net_volume_shadow'].stem}_cn.csv"
                ),
                hidden=bool(main_override),
            ),
            "main_net_volume_shadow_metadata": _artifact_reference(
                paths["main_net_volume_shadow_metadata"], hidden=bool(main_override)
            ),
            "main_net_volume_shadow_report": _artifact_reference(
                paths["main_net_volume_shadow_report"], hidden=bool(main_override)
            ),
            "main_net_volume_shadow_chart": _artifact_reference(
                paths["main_net_volume_shadow_chart"], hidden=bool(main_override)
            ),
            "institutional_accumulation_shadow": _artifact_reference(
                institutional_accumulation_path, hidden=bool(institutional_override)
            ),
            "institutional_accumulation_shadow_cn": _artifact_reference(
                paths["institutional_accumulation_shadow"].with_name(
                    f"{paths['institutional_accumulation_shadow'].stem}_cn.csv"
                ),
                hidden=bool(institutional_override),
            ),
            "institutional_accumulation_metadata": _artifact_reference(
                paths["institutional_accumulation_metadata"],
                hidden=bool(institutional_override),
            ),
            "institutional_accumulation_report": _artifact_reference(
                paths["institutional_accumulation_report"],
                hidden=bool(institutional_override),
            ),
            "institutional_accumulation_tracking": _artifact_reference(
                institutional_accumulation_tracking_path,
                hidden=bool(institutional_tracking_override),
            ),
            "institutional_accumulation_tracking_summary": _artifact_reference(
                paths["institutional_accumulation_tracking_summary"],
                hidden=bool(institutional_tracking_override),
            ),
            "institutional_accumulation_tracking_report": _artifact_reference(
                paths["institutional_accumulation_tracking_report"],
                hidden=bool(institutional_tracking_override),
            ),
            "czsc_structure_shadow": _artifact_reference(
                czsc_structure_shadow_path, hidden=bool(czsc_override)
            ),
            "czsc_structure_shadow_cn": _artifact_reference(
                paths["czsc_structure_shadow"].with_name(
                    f"{paths['czsc_structure_shadow'].stem}_cn.csv"
                ),
                hidden=bool(czsc_override),
            ),
            "czsc_structure_shadow_metadata": _artifact_reference(
                paths["czsc_structure_shadow_metadata"], hidden=bool(czsc_override)
            ),
            "czsc_structure_shadow_report": _artifact_reference(
                paths["czsc_structure_shadow_report"], hidden=bool(czsc_override)
            ),
            "strategy_arena_portfolio": (
                str(strategy_arena_portfolio_path)
                if strategy_arena_metadata and not arena_override
                else None
            ),
            "strategy_arena_portfolio_cn": (
                str(
                    _latest_artifact(
                        paths["strategy_arena_portfolio"].with_name(
                            f"{paths['strategy_arena_portfolio'].stem}_cn.csv"
                        )
                    )
                )
                if strategy_arena_metadata and not arena_override
                else None
            ),
            "strategy_arena_pairwise": (
                str(_latest_artifact(paths["strategy_arena_pairwise"]))
                if strategy_arena_metadata and not arena_override
                else None
            ),
            "strategy_arena_signal_division": (
                str(_latest_artifact(paths["strategy_arena_signal_division"]))
                if strategy_arena_metadata and not arena_override
                else None
            ),
            "strategy_arena_metadata": (
                str(_latest_artifact(paths["strategy_arena_metadata"]))
                if strategy_arena_metadata and not arena_override
                else None
            ),
            "strategy_arena_report": (
                str(_latest_artifact(paths["strategy_arena_report"]))
                if strategy_arena_metadata and not arena_override
                else None
            ),
            "strategy_arena_history": (
                str(paths["strategy_arena_history"])
                if strategy_arena_metadata
                and not arena_override
                and paths["strategy_arena_history"].exists()
                else None
            ),
            "hidden_accumulation_tracking": str(hidden_tracking_path),
            "factor_decay_monitor_csv": (
                str(_latest_artifact(paths["factor_decay_monitor_csv"]))
                if paths["factor_decay_monitor_csv"].exists()
                else None
            ),
            "factor_decay_monitor_json": factor_decay.get("path"),
            "factor_decay_monitor_report": (
                str(_latest_artifact(paths["factor_decay_monitor_report"]))
                if paths["factor_decay_monitor_report"].exists()
                else None
            ),
            "factor_decay_observation": (
                str(_latest_artifact(paths["factor_decay_observation"]))
                if paths["factor_decay_observation"].exists()
                else None
            ),
            "factor_decay_history": (
                str(paths["factor_decay_history"])
                if paths["factor_decay_history"].exists()
                else None
            ),
            "factor_replacement_daily_ic": (
                str(_latest_artifact(paths["factor_replacement_daily_ic"]))
                if paths["factor_replacement_daily_ic"].exists()
                else None
            ),
            "factor_replacement_competition": (
                str(_latest_artifact(paths["factor_replacement_competition"]))
                if paths["factor_replacement_competition"].exists()
                else None
            ),
            "factor_replacement_report": (
                str(_latest_artifact(paths["factor_replacement_report"]))
                if paths["factor_replacement_report"].exists()
                else None
            ),
            "factor_replacement_preregistration": (
                str(paths["factor_replacement_preregistration"])
                if paths["factor_replacement_preregistration"].exists()
                else None
            ),
            "factor_replacement_tracking": (
                str(_latest_artifact(paths["factor_replacement_tracking"]))
                if paths["factor_replacement_tracking"].exists()
                else None
            ),
            "factor_replacement_tracking_json": (
                str(_latest_artifact(paths["factor_replacement_tracking_json"]))
                if paths["factor_replacement_tracking_json"].exists()
                else None
            ),
            "model_decision": str(model_path),
            "strategy_family_forward_report": str(_latest_artifact(paths["strategy_family_forward_report"])),
            "strategy_family_forward_summary": str(strategy_family_summary_path),
            "strategy_family_health": str(strategy_family_health_path),
            "stability_report": (
                str(_latest_artifact(paths["strategy_stability_report"]))
                if config.include_strategy_stability_report
                else None
            ),
            "regime_shadow_report": str(_latest_artifact(paths["regime_shadow_report"])),
            "regime_shadow_comparison": str(_latest_artifact(paths["regime_shadow_comparison"])),
            "regime_shadow_tracking_ledger": tracking_ledger_artifact,
            "regime_shadow_tracking_summary": tracking_summary_artifact,
            "regime_shadow_tracking_report": tracking_report_artifact,
            "dynamic_breadth_preregistration": dynamic_breadth_artifacts[
                "preregistration"
            ],
            "dynamic_breadth_tracking_ledger": dynamic_breadth_artifacts["ledger"],
            "dynamic_breadth_tracking_summary": dynamic_breadth_artifacts["summary"],
            "dynamic_breadth_tracking_report": dynamic_breadth_artifacts["report"],
            "world_event_shadow_json": world_event_artifacts["json"],
            "world_event_shadow_csv": world_event_artifacts["csv"],
            "world_event_shadow_report": world_event_artifacts["report"],
            "benchmark_refresh_status": (
                str(_latest_artifact(paths["benchmark_refresh_status"]))
                if config.include_benchmark_refresh
                and paths["benchmark_refresh_status"].exists()
                else None
            ),
            "research_database": str(paths["research_db"]) if config.include_research_db_sync else None,
            "research_database_sync_status": research_database_sync_artifact,
            "marketlens_feed": None,
            "marketlens_dashboard_feed": None,
            "trend_ignition_shadow_manifest": trend_ignition_shadow_manifest_artifact,
            "trend_ignition_shadow_scores": (
                str(_latest_artifact(paths["trend_ignition_shadow_scores"]))
                if trend_ignition_shadow.get("status") == "complete"
                and paths["trend_ignition_shadow_scores"].exists()
                else None
            ),
            "trend_ignition_shadow_report": (
                str(_latest_artifact(paths["trend_ignition_shadow_report"]))
                if trend_ignition_shadow.get("status") == "complete"
                and paths["trend_ignition_shadow_report"].exists()
                else None
            ),
        },
    }


def append_daily_run_state(state_log: Path, record: dict[str, object]) -> None:
    state_log.parent.mkdir(parents=True, exist_ok=True)
    with state_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def attach_daily_run_card(
    config: PipelineConfig,
    record: dict[str, object],
    generated_at: str | None = None,
) -> dict[str, Path]:
    paths = _paths(config)
    card_paths = write_daily_run_card(
        paths["output_root"], record, generated_at=generated_at
    )
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        record["artifacts"] = artifacts
    artifacts["daily_run_card_json"] = str(card_paths["json"])
    artifacts["daily_run_card_markdown"] = str(card_paths["markdown"])
    return card_paths


def attach_daily_run_card_safely(
    config: PipelineConfig,
    record: dict[str, object],
) -> dict[str, Path] | None:
    try:
        return attach_daily_run_card(config, record)
    except (OSError, TypeError, ValueError) as exc:
        record["daily_run_card_error"] = str(exc)
        return None


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _mark_marketlens_export_success(
    record: dict[str, object],
    config: PipelineConfig,
) -> None:
    paths = _paths(config)
    verification = record.get("verification")
    if not isinstance(verification, dict):
        verification = {}
        record["verification"] = verification
    verification["marketlens_export_status"] = "success"
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        record["artifacts"] = artifacts
    artifacts["marketlens_feed"] = str(paths["marketlens_feed"])
    artifacts["marketlens_dashboard_feed"] = str(paths["marketlens_dashboard_feed"])


def _run_marketlens_export(
    config: PipelineConfig,
    step: PipelineStep,
    record: dict[str, object],
) -> list[dict[str, object]]:
    state_input = _paths(config)["marketlens_state_input"]
    try:
        _write_json_atomic(state_input, record)
        return run_steps([step], config.project_root)
    except PipelineStepExecutionError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise PipelineStepExecutionError(step, 1) from exc
    finally:
        state_input.unlink(missing_ok=True)


def build_pipeline_step_cache(config: PipelineConfig) -> PipelineStepCache:
    paths = _paths(config)
    fingerprint, inputs = build_global_fingerprint(
        project_root=config.project_root,
        base_panel=paths["base_panel"],
        daily_dir=paths["daily_dir"],
        daily_start=effective_daily_start(config),
        asof_date=config.asof_date,
        benchmark=paths["benchmark"],
    )
    return PipelineStepCache(paths["pipeline_step_cache"], fingerprint, inputs)


def execute_pipeline(
    config: PipelineConfig,
    steps: list[PipelineStep],
    *,
    state_log: Path,
    argv: list[str] | None = None,
    test_status: str = "not_run_by_pipeline",
    skip_state_log: bool = False,
    dry_run: bool = False,
) -> None:
    marketlens_step = build_marketlens_export_step(config)
    audit_steps = [*steps, marketlens_step] if marketlens_step is not None else list(steps)
    step_executions: list[dict[str, object]] = []
    step_cache: PipelineStepCache | None = None
    try:
        if dry_run or not config.enable_step_cache:
            step_executions = run_steps(
                steps,
                config.project_root,
                dry_run=dry_run,
                max_parallel_steps=config.max_parallel_steps,
            )
        else:
            benchmark_steps = [step for step in steps if step.name == "benchmark_refresh"]
            remaining_steps = [step for step in steps if step.name != "benchmark_refresh"]
            if benchmark_steps:
                step_executions.extend(
                    run_steps(
                        benchmark_steps,
                        config.project_root,
                        max_parallel_steps=1,
                    )
                )
            step_cache = build_pipeline_step_cache(config)
            step_executions.extend(
                run_steps(
                    remaining_steps,
                    config.project_root,
                    max_parallel_steps=config.max_parallel_steps,
                    step_cache=step_cache,
                )
            )
    except PipelineStepExecutionError as exc:
        step_executions.extend(exc.completed_executions)
        if step_cache is not None:
            step_cache.save()
        if not dry_run and not skip_state_log:
            record = build_daily_run_state(
                config,
                audit_steps,
                argv=argv,
                test_status=test_status,
                run_status="failed",
                failure={
                    "step": exc.step.name,
                    "returncode": exc.returncode,
                    "message": str(exc),
                },
                step_executions=step_executions,
            )
            attach_daily_run_card_safely(config, record)
            append_daily_run_state(state_log, record)
        raise
    if step_cache is not None:
        step_cache.save()
    if dry_run:
        if marketlens_step is not None:
            run_steps([marketlens_step], config.project_root, dry_run=True)
        return

    record = build_daily_run_state(
        config,
        audit_steps,
        argv=argv,
        test_status=test_status,
        step_executions=step_executions,
    )
    if marketlens_step is not None:
        try:
            marketlens_execution = _run_marketlens_export(config, marketlens_step, record)
        except PipelineStepExecutionError as exc:
            if not skip_state_log:
                failed_record = build_daily_run_state(
                    config,
                    audit_steps,
                    argv=argv,
                    test_status=test_status,
                    run_status="failed",
                    failure={
                        "step": exc.step.name,
                        "returncode": exc.returncode,
                        "message": str(exc),
                    },
                    step_executions=[*step_executions, *exc.completed_executions],
                )
                attach_daily_run_card_safely(config, failed_record)
                append_daily_run_state(state_log, failed_record)
            raise
        step_executions.extend(marketlens_execution)
        record["execution"] = execution_summary(
            step_executions,
            config.max_parallel_steps,
        )
        _mark_marketlens_export_success(record, config)
    if not skip_state_log:
        attach_daily_run_card_safely(config, record)
        append_daily_run_state(state_log, record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily model workflow after market data refresh.")
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--python", dest="python_exe", default=sys.executable)
    parser.add_argument(
        "--base-panel",
        default=None,
        help="Base panel path. Defaults to the newest panel ending before --asof-date.",
    )
    parser.add_argument("--daily-dir", default=str(DEFAULT_DAILY_NORMALIZED_DIR))
    parser.add_argument("--daily-data-dir", default=str(DEFAULT_DAILY_DATA_DIR))
    parser.add_argument("--fallback-data-dir", default=str(DEFAULT_FALLBACK_DATA_DIR))
    parser.add_argument("--fetch-status-utils", default=str(DEFAULT_FETCH_STATUS_UTILS))
    parser.add_argument("--daily-start", default="2000-01-01")
    parser.add_argument("--output-root", default="outputs/high_return_v2")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--research-db", default="data/research.sqlite3")
    parser.add_argument("--rules", default="configs/personal_trade_habit_overlay.yaml")
    parser.add_argument("--symbol-history", default=None)
    parser.add_argument("--shadow-account-review", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-factor-decay-monitor", action="store_true")
    parser.add_argument("--skip-strategy-family-forward", action="store_true")
    parser.add_argument("--skip-strategy-arena", action="store_true")
    parser.add_argument("--skip-strategy-stability", action="store_true")
    parser.add_argument("--enable-strategy-stability", action="store_true")
    parser.add_argument("--skip-benchmark-refresh", action="store_true")
    parser.add_argument("--skip-regime-shadow-compare", action="store_true")
    parser.add_argument("--skip-regime-shadow-tracking", action="store_true")
    parser.add_argument("--skip-dynamic-breadth-tracking", action="store_true")
    parser.add_argument("--skip-world-event-shadow", action="store_true")
    parser.add_argument(
        "--world-event-config",
        default="configs/world_event_shadow.yaml",
    )
    parser.add_argument("--world-event-snapshot", default=None)
    parser.add_argument("--skip-research-db-sync", action="store_true")
    parser.add_argument("--skip-marketlens-export", action="store_true")
    parser.add_argument(
        "--marketlens-dashboard-output",
        default=str(DEFAULT_MARKETLENS_DASHBOARD_OUTPUT),
    )
    parser.add_argument("--enable-trend-ignition-shadow", action="store_true")
    parser.add_argument("--regime-shadow-config", default="configs/evolution_strong_pullback.yaml")
    parser.add_argument("--regime-shadow-candidate-id", default="regime_090_balanced")
    parser.add_argument(
        "--trend-ignition-scorer",
        default="outputs/trend_ignition_training/scorer_v3_shortlist_exploratory/binned_scorer.json",
    )
    parser.add_argument(
        "--trend-ignition-scorer-summary",
        default="outputs/trend_ignition_training/scorer_v3_shortlist_exploratory/scorer_summary.json",
    )
    parser.add_argument(
        "--trend-ignition-selection-status",
        choices=("preregistered", "exploratory_posthoc"),
        default="exploratory_posthoc",
    )
    parser.add_argument("--state-log", default="daily_run_state.jsonl")
    parser.add_argument("--state-test-status", default="not_run_by_pipeline")
    parser.add_argument(
        "--max-parallel-steps",
        type=int,
        default=2,
        help="Maximum number of dependency-safe pipeline subprocesses to run concurrently.",
    )
    parser.add_argument(
        "--disable-step-cache",
        action="store_true",
        help="Recompute every pipeline step even when strict content hashes match.",
    )
    parser.add_argument("--skip-state-log", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_parallel_steps < 1:
        raise ValueError("--max-parallel-steps must be at least 1")
    project_root = Path(args.project_root).resolve()
    daily_dir = _resolve(project_root, Path(args.daily_dir))
    asof_date = args.asof_date or discover_latest_daily_date(daily_dir)
    base_panel = (
        _resolve(project_root, Path(args.base_panel))
        if args.base_panel
        else discover_base_panel(project_root, asof_date)
    )
    symbol_history = (
        _resolve(project_root, Path(args.symbol_history))
        if args.symbol_history
        else latest_symbol_history(project_root)
    )
    shadow_account_review = (
        _resolve(project_root, Path(args.shadow_account_review))
        if args.shadow_account_review
        else latest_shadow_account_review(project_root, asof_date)
    )
    stability_available = default_stability_inputs_available(project_root)
    include_strategy_stability = (
        args.enable_strategy_stability
        if not args.skip_strategy_stability
        else False
    )
    if not args.enable_strategy_stability and not args.skip_strategy_stability:
        include_strategy_stability = stability_available
    if not include_strategy_stability and not args.skip_strategy_stability:
        print(
            "Strategy stability report skipped: default legacy metrics/equity inputs are incomplete. "
            "Use --enable-strategy-stability only after restoring all inputs."
        )
    config = PipelineConfig(
        asof_date=asof_date,
        python_exe=args.python_exe,
        project_root=project_root,
        output_root=Path(args.output_root),
        base_panel=base_panel,
        daily_dir=daily_dir,
        daily_data_dir=Path(args.daily_data_dir),
        fallback_data_dir=Path(args.fallback_data_dir),
        fetch_status_utils=Path(args.fetch_status_utils),
        daily_start=args.daily_start,
        benchmark=Path(args.benchmark),
        research_db=Path(args.research_db),
        rules=Path(args.rules),
        symbol_history=symbol_history,
        shadow_account_review=shadow_account_review,
        train_model=not args.skip_train,
        include_factor_decay_monitor=not args.skip_factor_decay_monitor,
        include_strategy_family_forward_report=not args.skip_strategy_family_forward,
        include_strategy_arena=not args.skip_strategy_arena,
        include_strategy_stability_report=include_strategy_stability,
        include_benchmark_refresh=not args.skip_benchmark_refresh,
        include_regime_shadow_compare=not args.skip_regime_shadow_compare,
        include_regime_shadow_tracking=not args.skip_regime_shadow_tracking,
        include_dynamic_breadth_tracking=not args.skip_dynamic_breadth_tracking,
        include_world_event_shadow=not args.skip_world_event_shadow,
        world_event_config=Path(args.world_event_config),
        world_event_snapshot=(
            Path(args.world_event_snapshot) if args.world_event_snapshot else None
        ),
        include_research_db_sync=not args.skip_research_db_sync,
        include_marketlens_export=not args.skip_marketlens_export,
        include_trend_ignition_shadow=args.enable_trend_ignition_shadow,
        marketlens_dashboard_output=Path(args.marketlens_dashboard_output),
        regime_shadow_config=Path(args.regime_shadow_config),
        regime_shadow_candidate_id=args.regime_shadow_candidate_id,
        trend_ignition_scorer=Path(args.trend_ignition_scorer),
        trend_ignition_scorer_summary=Path(args.trend_ignition_scorer_summary),
        trend_ignition_selection_status=args.trend_ignition_selection_status,
        max_parallel_steps=args.max_parallel_steps,
        enable_step_cache=not args.disable_step_cache,
    )
    fetch_status = fetch_status_summary(_resolve(project_root, config.fetch_status_utils))
    print("Daily market data fetch status:")
    print(json.dumps(fetch_status, ensure_ascii=False, indent=2, default=str))
    if not args.dry_run:
        ensure_fetch_status_ok(fetch_status)
    state_log = _resolve(project_root, Path(args.state_log))
    steps = build_daily_pipeline_steps(config)
    execute_pipeline(
        config,
        steps,
        state_log=state_log,
        argv=sys.argv[1:],
        test_status=args.state_test_status,
        skip_state_log=args.skip_state_log,
        dry_run=args.dry_run,
    )
    token = _date_token(asof_date)
    output_root = _resolve(project_root, Path(args.output_root))
    print("Daily workflow outputs:")
    print(output_root / f"daily_personal_overlay_report_{token}.md")
    print(output_root / f"daily_personal_overlay_selected_{token}.csv")
    print(output_root / f"daily_personal_overlay_changes_{token}.csv")
    print(output_root / f"hidden_accumulation_trade_watch_tracking_{token}.csv")
    print(output_root / f"institutional_accumulation_shadow_{token}.md")
    print(output_root / f"institutional_accumulation_tracking_{token}.md")
    if config.include_factor_decay_monitor:
        print(output_root / f"factor_decay_monitor_{token}.md")
        print(output_root / f"factor_replacement_competition_{token}.md")
        print(output_root / f"factor_replacement_tracking_{token}.json")
    print(output_root / f"merged_state_pattern_scan_{token}.csv")
    print(output_root / f"merged_model_decision_table_{token}.csv")
    print(output_root / f"merged_priority_watchlist_{token}.csv")
    if config.include_trend_ignition_shadow:
        print(output_root / f"trend_ignition_shadow_{token}" / "trend_ignition_shadow_report.md")
        print(output_root / f"trend_ignition_shadow_{token}" / "manifest.json")
    if config.include_strategy_family_forward_report:
        print(output_root / f"strategy_family_forward_report_{token}.md")
    if config.include_strategy_arena and config.include_regime_shadow_compare:
        print(output_root / f"strategy_arena_{token}.md")
        print(output_root / f"strategy_arena_portfolio_{token}_cn.csv")
    if config.include_strategy_stability_report:
        print(output_root / f"core_risk_filter_finalist_stability_{token}.md")
    if config.include_marketlens_export:
        print(output_root / "marketlens_model3_latest.json")
        print(_resolve(project_root, config.marketlens_dashboard_output))
    if config.include_regime_shadow_compare:
        print(output_root / f"regime_shadow_compare_{token}" / "report.md")
    if config.include_regime_shadow_tracking:
        print(output_root / "regime_shadow_tracking.csv")
        print(output_root / "regime_shadow_tracking_summary.json")
        print(output_root / "regime_shadow_tracking_report.md")
    if (
        config.include_dynamic_breadth_tracking
        and config.train_model
        and config.include_strategy_arena
        and config.include_regime_shadow_compare
    ):
        print(output_root / "dynamic_breadth_overlay_tracking.csv")
        print(output_root / "dynamic_breadth_overlay_tracking_summary.json")
        print(output_root / "dynamic_breadth_overlay_tracking_report.md")
    if config.include_benchmark_refresh:
        print(output_root / f"benchmark_refresh_status_{token}.json")
    if config.include_world_event_shadow:
        print(output_root / f"world_event_shadow_{token}.json")
        print(output_root / f"world_event_shadow_{token}.md")
    if config.include_research_db_sync:
        print(output_root / f"research_database_sync_{token}.json")
    if not args.dry_run and not args.skip_state_log:
        print("Daily run state:")
        print(state_log)
        print(output_root / f"daily_run_card_{token}.json")
        print(output_root / f"daily_run_card_{token}.md")


if __name__ == "__main__":
    main()
