"""Run the default post-refresh daily model workflow."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


DEFAULT_DAILY_DATA_DIR = Path(r"D:\codex\daily-market-data")
DEFAULT_FALLBACK_DATA_DIR = Path(r"D:\codex\2026-06-15-exchange-data-ingest")
DEFAULT_DAILY_NORMALIZED_DIR = DEFAULT_DAILY_DATA_DIR / "ths_exports" / "normalized"
DEFAULT_FETCH_STATUS_UTILS = DEFAULT_FALLBACK_DATA_DIR / "scripts" / "market_data_utils.py"
DEFAULT_BENCHMARK = DEFAULT_DAILY_DATA_DIR / "benchmarks" / "510300.csv"
DEFAULT_ALLOWED_TREND_STATES = "趋势确认,生命线健康,回调可观察"


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: tuple[str, ...]


class PipelineStepExecutionError(RuntimeError):
    def __init__(self, step: PipelineStep, returncode: int):
        super().__init__(f"Pipeline step {step.name} failed with exit code {returncode}")
        self.step = step
        self.returncode = returncode


@dataclass(frozen=True)
class PipelineConfig:
    asof_date: str
    python_exe: str = sys.executable
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    output_root: Path = Path("outputs/high_return_v2")
    base_panel: Path = Path("data_panel_history_main_chinext_20220101_20260626.csv")
    daily_dir: Path = DEFAULT_DAILY_NORMALIZED_DIR
    daily_data_dir: Path = DEFAULT_DAILY_DATA_DIR
    fallback_data_dir: Path = DEFAULT_FALLBACK_DATA_DIR
    fetch_status_utils: Path = DEFAULT_FETCH_STATUS_UTILS
    daily_start: str = "2026-06-22"
    benchmark: Path = DEFAULT_BENCHMARK
    rules: Path = Path("configs/personal_trade_habit_overlay.yaml")
    symbol_history: Path | None = None
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
    include_strategy_stability_report: bool = True
    include_strategy_family_forward_report: bool = True
    include_benchmark_refresh: bool = True
    include_regime_shadow_compare: bool = True
    include_regime_shadow_tracking: bool = True
    regime_shadow_config: Path = Path("configs/evolution_strong_pullback.yaml")
    regime_shadow_candidate_id: str = "regime_090_balanced"


def _date_token(asof_date: str) -> str:
    return asof_date.replace("-", "")


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _latest_artifact(path: Path) -> Path:
    candidates = [path] if path.exists() else []
    candidates.extend(path.parent.glob(f"{path.stem}_pending*{path.suffix}"))
    if not candidates:
        return path
    return max(candidates, key=lambda item: item.stat().st_mtime)


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
    token = _date_token(config.asof_date)
    model_dir = output_root / f"next_open_rank_model_stock_focus_lev093_marketfilter_bench{token}_20220101_{token}"
    return {
        "base_panel": _resolve(root, config.base_panel),
        "daily_dir": config.daily_dir,
        "output_root": output_root,
        "panel": root / f"data_panel_history_main_chinext_20220101_{token}.csv",
        "trend_dir": output_root / f"trend_state_{token}",
        "model_dir": model_dir,
        "weights": model_dir / "rolling_feature_weights.csv",
        "base_candidates": output_root / f"rank_model_candidates_trend_gated_bench{token}_{token}.csv",
        "overlay_candidates": output_root / f"rank_model_candidates_trend_gated_personal_overlay_{token}.csv",
        "early_watchlist": output_root / f"early_pattern_watchlist_{token}.csv",
        "hidden_accumulation_tracking": output_root / f"hidden_accumulation_trade_watch_tracking_{token}.csv",
        "merged_state_pattern_scan": output_root / f"merged_state_pattern_scan_{token}.csv",
        "merged_model_decision_table": output_root / f"merged_model_decision_table_{token}.csv",
        "merged_priority_watchlist": output_root / f"merged_priority_watchlist_{token}.csv",
        "strategy_family_forward_summary": output_root / f"strategy_family_forward_summary_{token}.csv",
        "strategy_family_health": output_root / f"strategy_family_health_{token}.csv",
        "strategy_family_forward_report": output_root / f"strategy_family_forward_report_{token}.md",
        "daily_report": output_root / f"daily_personal_overlay_report_{token}.md",
        "selected": output_root / f"daily_personal_overlay_selected_{token}.csv",
        "changes": output_root / f"daily_personal_overlay_changes_{token}.csv",
        "metrics": metrics_path_for_date(config.asof_date, output_root),
        "names_source": names_source_for_date(config.asof_date, config.daily_data_dir),
        "rules": _resolve(root, config.rules),
        "strategy_stability_prefix": output_root / f"core_risk_filter_finalist_stability_{token}",
        "strategy_stability_report": output_root / f"core_risk_filter_finalist_stability_{token}.md",
        "regime_shadow_dir": output_root / f"regime_shadow_compare_{token}",
        "regime_shadow_report": output_root / f"regime_shadow_compare_{token}" / "report.md",
        "regime_shadow_comparison": output_root / f"regime_shadow_compare_{token}" / "comparison.json",
        "regime_shadow_tracking_ledger": output_root / "regime_shadow_tracking.csv",
        "regime_shadow_tracking_summary": output_root / "regime_shadow_tracking_summary.json",
        "regime_shadow_tracking_report": output_root / "regime_shadow_tracking_report.md",
        "benchmark_refresh_status": output_root / f"benchmark_refresh_status_{token}.json",
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
                    str(config.benchmark),
                    "--asof-date",
                    config.asof_date,
                    "--status-output",
                    str(paths["benchmark_refresh_status"]),
                ),
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
                config.daily_start,
                "--daily-end",
                config.asof_date,
                "--output",
                str(paths["panel"]),
            ),
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
                    str(config.benchmark),
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
                    str(config.benchmark),
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
                str(config.daily_data_dir),
                "--fallback-data-dir",
                str(config.fallback_data_dir),
                "--fetch-status-utils",
                str(config.fetch_status_utils),
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
    return steps


def run_steps(steps: list[PipelineStep], project_root: Path, dry_run: bool = False) -> None:
    for index, step in enumerate(steps, start=1):
        print(f"[{index}/{len(steps)}] {step.name}")
        print(" ".join(step.command))
        if not dry_run:
            try:
                subprocess.run(step.command, cwd=project_root, check=True)
            except subprocess.CalledProcessError as exc:
                raise PipelineStepExecutionError(step, int(exc.returncode)) from exc


def build_daily_run_state(
    config: PipelineConfig,
    steps: list[PipelineStep],
    argv: list[str] | None = None,
    test_status: str = "not_run_by_pipeline",
    run_status: str = "success",
    failure: dict[str, object] | None = None,
) -> dict[str, object]:
    paths = _paths(config)
    tracking_enabled = getattr(config, "include_regime_shadow_tracking", True)
    priority_path = _latest_artifact(paths["merged_priority_watchlist"])
    early_path = _latest_artifact(paths["early_watchlist"])
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
    hidden_tracking_rows = _read_csv_rows(hidden_tracking_path)
    model_rows = _read_csv_rows(model_path)
    strategy_family_summary_rows = _read_csv_rows(strategy_family_summary_path)
    strategy_family_health_rows = _read_csv_rows(strategy_family_health_path)
    selected_rows = _read_csv_rows(selected_path)
    change_rows = _read_csv_rows(changes_path)
    regime_shadow = _read_json_object(paths["regime_shadow_comparison"])
    if tracking_enabled:
        tracking_summary = _read_json_object(paths["regime_shadow_tracking_summary"])
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
    if not config.include_benchmark_refresh:
        benchmark_refresh = {"status": "skipped"}
    elif paths["benchmark_refresh_status"].exists():
        benchmark_refresh = _read_json_object(paths["benchmark_refresh_status"])
    else:
        benchmark_refresh = {"status": "missing"}
    regime_delta = regime_shadow.get("delta")
    if not isinstance(regime_delta, dict):
        regime_delta = {}
    regime_latest = regime_shadow.get("latest_dynamic_state")
    if not isinstance(regime_latest, dict):
        regime_latest = {}
    if regime_shadow.get("benchmark_fresh") is not True:
        regime_latest = {}
    return {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "run_status": run_status,
        "failure": failure,
        "asof_date": config.asof_date,
        "project_root": str(config.project_root),
        "run_type": "full_train" if config.train_model else "skip_train",
        "data_source": str(paths["names_source"]),
        "data_source_exists": paths["names_source"].exists(),
        "fallback_fetch_status": fetch_status_summary(config.fetch_status_utils),
        "argv": argv or [],
        "steps": [step.name for step in steps],
        "commands": [" ".join(step.command) for step in steps],
        "verification": {
            "tests": test_status,
            "priority_rows": len(priority_rows),
            "missing_stock_names": _missing_count(priority_rows, "stock_name"),
            "priority_bucket_counts": _value_counts(priority_rows, "priority_bucket"),
            "priority_strategy_family_counts": _value_counts(priority_rows, "strategy_family"),
            "early_pattern_rows": len(early_rows),
            "early_pattern_counts": _value_counts(early_rows, "pattern_type"),
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
            "benchmark_refresh_status": benchmark_refresh.get("status"),
            "benchmark_latest_date": benchmark_refresh.get("latest_date"),
            "benchmark_source_agreement": benchmark_refresh.get("source_agreement"),
            "benchmark_rows_added": benchmark_refresh.get("rows_added"),
        },
        "top10": _top_rows(priority_rows, ["symbol", "stock_name", "strategy_family", "priority_bucket"]),
        "artifacts": {
            "priority": str(priority_path),
            "priority_cn": str(_latest_artifact(priority_cn_base)),
            "report": str(_latest_artifact(paths["daily_report"])),
            "selected": str(selected_path),
            "changes": str(changes_path),
            "early_watchlist": str(early_path),
            "hidden_accumulation_tracking": str(hidden_tracking_path),
            "model_decision": str(model_path),
            "strategy_family_forward_report": str(_latest_artifact(paths["strategy_family_forward_report"])),
            "strategy_family_forward_summary": str(strategy_family_summary_path),
            "strategy_family_health": str(strategy_family_health_path),
            "stability_report": str(_latest_artifact(paths["strategy_stability_report"])),
            "regime_shadow_report": str(_latest_artifact(paths["regime_shadow_report"])),
            "regime_shadow_comparison": str(_latest_artifact(paths["regime_shadow_comparison"])),
            "regime_shadow_tracking_ledger": tracking_ledger_artifact,
            "regime_shadow_tracking_summary": tracking_summary_artifact,
            "regime_shadow_tracking_report": tracking_report_artifact,
            "benchmark_refresh_status": (
                str(_latest_artifact(paths["benchmark_refresh_status"]))
                if config.include_benchmark_refresh
                and paths["benchmark_refresh_status"].exists()
                else None
            ),
        },
    }


def append_daily_run_state(state_log: Path, record: dict[str, object]) -> None:
    state_log.parent.mkdir(parents=True, exist_ok=True)
    with state_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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
    try:
        run_steps(steps, config.project_root, dry_run=dry_run)
    except PipelineStepExecutionError as exc:
        if not dry_run and not skip_state_log:
            record = build_daily_run_state(
                config,
                steps,
                argv=argv,
                test_status=test_status,
                run_status="failed",
                failure={
                    "step": exc.step.name,
                    "returncode": exc.returncode,
                    "message": str(exc),
                },
            )
            append_daily_run_state(state_log, record)
        raise
    if not dry_run and not skip_state_log:
        record = build_daily_run_state(
            config,
            steps,
            argv=argv,
            test_status=test_status,
        )
        append_daily_run_state(state_log, record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily model workflow after market data refresh.")
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--python", dest="python_exe", default=sys.executable)
    parser.add_argument("--base-panel", default="data_panel_history_main_chinext_20220101_20260626.csv")
    parser.add_argument("--daily-dir", default=str(DEFAULT_DAILY_NORMALIZED_DIR))
    parser.add_argument("--daily-data-dir", default=str(DEFAULT_DAILY_DATA_DIR))
    parser.add_argument("--fallback-data-dir", default=str(DEFAULT_FALLBACK_DATA_DIR))
    parser.add_argument("--fetch-status-utils", default=str(DEFAULT_FETCH_STATUS_UTILS))
    parser.add_argument("--daily-start", default="2026-06-22")
    parser.add_argument("--output-root", default="outputs/high_return_v2")
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--rules", default="configs/personal_trade_habit_overlay.yaml")
    parser.add_argument("--symbol-history", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-strategy-family-forward", action="store_true")
    parser.add_argument("--skip-strategy-stability", action="store_true")
    parser.add_argument("--skip-benchmark-refresh", action="store_true")
    parser.add_argument("--skip-regime-shadow-compare", action="store_true")
    parser.add_argument("--skip-regime-shadow-tracking", action="store_true")
    parser.add_argument("--regime-shadow-config", default="configs/evolution_strong_pullback.yaml")
    parser.add_argument("--regime-shadow-candidate-id", default="regime_090_balanced")
    parser.add_argument("--state-log", default="daily_run_state.jsonl")
    parser.add_argument("--state-test-status", default="not_run_by_pipeline")
    parser.add_argument("--skip-state-log", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    daily_dir = Path(args.daily_dir)
    asof_date = args.asof_date or discover_latest_daily_date(daily_dir)
    symbol_history = Path(args.symbol_history) if args.symbol_history else latest_symbol_history(project_root)
    config = PipelineConfig(
        asof_date=asof_date,
        python_exe=args.python_exe,
        project_root=project_root,
        output_root=Path(args.output_root),
        base_panel=Path(args.base_panel),
        daily_dir=daily_dir,
        daily_data_dir=Path(args.daily_data_dir),
        fallback_data_dir=Path(args.fallback_data_dir),
        fetch_status_utils=Path(args.fetch_status_utils),
        daily_start=args.daily_start,
        benchmark=Path(args.benchmark),
        rules=Path(args.rules),
        symbol_history=symbol_history,
        train_model=not args.skip_train,
        include_strategy_family_forward_report=not args.skip_strategy_family_forward,
        include_strategy_stability_report=not args.skip_strategy_stability,
        include_benchmark_refresh=not args.skip_benchmark_refresh,
        include_regime_shadow_compare=not args.skip_regime_shadow_compare,
        include_regime_shadow_tracking=not args.skip_regime_shadow_tracking,
        regime_shadow_config=Path(args.regime_shadow_config),
        regime_shadow_candidate_id=args.regime_shadow_candidate_id,
    )
    fetch_status = fetch_status_summary(config.fetch_status_utils)
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
    print(output_root / f"merged_state_pattern_scan_{token}.csv")
    print(output_root / f"merged_model_decision_table_{token}.csv")
    print(output_root / f"merged_priority_watchlist_{token}.csv")
    if config.include_strategy_family_forward_report:
        print(output_root / f"strategy_family_forward_report_{token}.md")
    if config.include_strategy_stability_report:
        print(output_root / f"core_risk_filter_finalist_stability_{token}.md")
    if config.include_regime_shadow_compare:
        print(output_root / f"regime_shadow_compare_{token}" / "report.md")
    if config.include_regime_shadow_tracking:
        print(output_root / "regime_shadow_tracking.csv")
        print(output_root / "regime_shadow_tracking_summary.json")
        print(output_root / "regime_shadow_tracking_report.md")
    if config.include_benchmark_refresh:
        print(output_root / f"benchmark_refresh_status_{token}.json")
    if not args.dry_run and not args.skip_state_log:
        print("Daily run state:")
        print(state_log)


if __name__ == "__main__":
    main()
