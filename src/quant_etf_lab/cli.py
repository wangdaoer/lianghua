"""Command line interface for the ETF lab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .allocator_observation import run_allocator_observation
from .allocator_promotion import run_allocator_promotion_review
from .allocator_switch import run_allocator_switch_readiness
from .attribution import run_drawdown_attribution
from .backtest import run_backtest
from .chip_reversal_candidate_outcomes import run_chip_reversal_candidate_outcomes
from .chip_reversal_daily_candidates import run_chip_reversal_daily_candidates
from .chip_reversal_lab import run_chip_reversal_lab
from .chip_reversal_observation_queue import run_chip_reversal_observation_queue
from .chip_reversal_pit_audit import run_chip_reversal_pit_audit
from .config import load_config
from .data import update_data
from .dashboard import run_daily_dashboard
from .daily_pipeline import run_daily_pipeline
from .daily_check import run_daily_model_check
from .daily_run_status import run_daily_run_status
from .diagnostics import run_batch_diagnostics, run_diagnostics
from .factor_lab import run_factor_lab
from .live_shadow import (
    run_live_shadow,
    run_live_shadow_import_template,
    run_live_shadow_preflight,
    run_live_shadow_review_decisions,
    run_live_shadow_review_queue,
)
from .market_cap import DEFAULT_STOCK_MARKET_CAP_PATH, load_stock_market_cap_symbols, update_stock_market_cap_cache
from .market_sentiment import run_market_sentiment_reference
from .market_data_source import DEFAULT_DAILY_MARKET_DATA_DIR, DEFAULT_EXCHANGE_INGEST_DIR
from .market_timing_state import run_market_timing_state
from .model_audit import run_model_build_audit
from .momentum_focus import DEFAULT_NAME_MAP_PATH, run_momentum_focus
from .momentum_outcomes import run_momentum_outcome_analysis
from .network_lab import run_network_lab
from .paper_account import apply_stock_target_review_decision_template, run_paper_account
from .live_preflight import run_live_preflight
from .phase2 import run_phase2_status
from .phase2_review import run_phase2_review
from .pipeline_history import run_pipeline_history_review
from .portfolio import (
    load_portfolio_config,
    load_portfolio_source_selection_config,
    run_portfolio_combine,
    run_portfolio_source_selection_walk_forward,
    run_portfolio_walk_forward,
)
from .portfolio_window_attribution import run_portfolio_window_attribution
from .process_lock import ProcessLockError, process_lock
from .report import resolve_run_dir, write_report
from .satellite_risk_budget import run_satellite_risk_budget_review
from .source_decision import run_source_decision_review
from .source_guard_decomposition import run_source_guard_decomposition
from .source_risk_budget import run_source_risk_budget_review
from .theme_map_builder import DEFAULT_OUTPUT_DIR as DEFAULT_THEME_MAP_OUTPUT_DIR
from .theme_map_builder import run_theme_map_build
from .theme_state import run_theme_state
from .validation import run_validation
from .walk_forward import run_walk_forward


DEFAULT_CONFIG = Path("configs/etf_trend.yaml")
DEFAULT_PORTFOLIO_CONFIG = Path("configs/portfolio_core_satellite_v1.yaml")
DEFAULT_RESEARCH_ALLOCATOR_DIR = Path("outputs/portfolio_source_selection/main_chinext_portfolio_source_selection_validation6_v1")
DEFAULT_PHASE2_COMPONENTS_PATH = Path(
    "outputs/research/phase2_model_status_source_selection_default_20260614/phase2_components.csv"
)
DEFAULT_RESEARCH_ALLOCATOR_DIR_ARG = DEFAULT_RESEARCH_ALLOCATOR_DIR.as_posix()
DEFAULT_PHASE2_COMPONENTS_PATH_ARG = DEFAULT_PHASE2_COMPONENTS_PATH.as_posix()
DEFAULT_DAILY_MARKET_DATA_DIR_ARG = DEFAULT_DAILY_MARKET_DATA_DIR.as_posix()
DEFAULT_EXCHANGE_INGEST_DIR_ARG = DEFAULT_EXCHANGE_INGEST_DIR.as_posix()
WALK_FORWARD_DEFAULTS = {
    "train_years": 3,
    "test_months": 12,
    "step_months": 12,
    "grid": "compact",
    "objective": "robust",
    "max_train_drawdown": 0.35,
    "selection_validation_months": 0,
}
ALERT_LEVEL_RANK = {"none": -1, "normal": 0, "info": 1, "warning": 2, "critical": 3}
HISTORY_HEALTH_RANK = {"none": -1, "ok": 0, "watch": 1, "blocked": 2, "unknown": 2}
ALERT_GATE_EXIT_CODE = 4
HISTORY_GATE_EXIT_CODE = 5
DAILY_RUN_STATUS_PROBLEM_EXIT_CODE = 6
LIVE_PREFLIGHT_BLOCKED_EXIT_CODE = 7
WALK_FORWARD_PRESETS = {
    "bear-v2": {
        "train_years": 4,
        "test_months": 12,
        "step_months": 12,
        "grid": "bear",
        "objective": "robust",
        "max_train_drawdown": 0.30,
    },
    "opportunity-v2": {
        "train_years": 4,
        "test_months": 12,
        "step_months": 12,
        "grid": "opportunity",
        "objective": "robust",
        "max_train_drawdown": 0.25,
    },
    "main-chinext-stable": {
        "train_years": 4,
        "test_months": 6,
        "step_months": 6,
        "grid": "stable",
        "objective": "stable",
        "max_train_drawdown": 0.25,
    },
    "main-chinext-stable-v2": {
        "train_years": 4,
        "test_months": 6,
        "step_months": 6,
        "grid": "stable_v2",
        "objective": "stable",
        "max_train_drawdown": 0.20,
    },
    "main-chinext-reversal-v1": {
        "train_years": 4,
        "test_months": 6,
        "step_months": 6,
        "grid": "reversal",
        "objective": "stable",
        "max_train_drawdown": 0.25,
    },
    "main-chinext-satellite-v1": {
        "train_years": 4,
        "test_months": 6,
        "step_months": 6,
        "grid": "satellite",
        "objective": "satellite",
        "max_train_drawdown": 0.35,
    },
    "main-chinext-satellite-v2": {
        "train_years": 4,
        "test_months": 6,
        "step_months": 6,
        "grid": "satellite_v2",
        "objective": "satellite",
        "max_train_drawdown": 0.35,
    }
}


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML config.")


def _command_line() -> str:
    return " ".join([sys.executable, "-m", "quant_etf_lab", *sys.argv[1:]])


def _lock_key(*parts: object) -> str:
    return "|".join(str(part) for part in parts)


def _walk_forward_lock_key(args: argparse.Namespace, walk_options: dict[str, object]) -> str:
    return _lock_key(
        "walk-forward",
        Path(args.config).resolve(),
        Path(args.output_dir).resolve(),
        args.run_id_prefix or "auto",
        walk_options["train_years"],
        walk_options["test_months"],
        walk_options["step_months"],
        walk_options["grid"],
        walk_options["objective"],
        walk_options["max_train_drawdown"],
        walk_options["selection_validation_months"],
    )


def _lock_dir() -> Path:
    return Path("outputs") / "locks"


def _duplicate_process_exit(error: ProcessLockError) -> int:
    print(f"Duplicate process blocked: {error}")
    return 3


def _parse_source_float_map(raw_items: list[str] | None, option_name: str, value_label: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw in raw_items or []:
        if "=" not in raw:
            raise ValueError(f"{option_name} must use SOURCE={value_label}.")
        source, value = raw.split("=", 1)
        source = source.strip()
        if not source:
            raise ValueError(f"{option_name} source cannot be empty.")
        try:
            parsed = float(value)
        except ValueError as error:
            raise ValueError(f"Invalid {value_label.lower()} for {source}: {value}") from error
        if parsed < 0.0:
            raise ValueError(f"{value_label} cannot be negative: {source}={parsed}")
        values[source] = parsed
    return values


def _parse_source_max_satellite_weight(raw_items: list[str] | None) -> dict[str, float]:
    return _parse_source_float_map(raw_items, "--source-max-satellite-weight", "WEIGHT")


def _parse_source_switch_margin_by_source(raw_items: list[str] | None) -> dict[str, float]:
    return _parse_source_float_map(raw_items, "--source-switch-margin-by-source", "MARGIN")


def _alert_gate_failed(level: object, threshold: str) -> bool:
    if threshold == "none":
        return False
    return ALERT_LEVEL_RANK.get(str(level), 0) >= ALERT_LEVEL_RANK[threshold]


def _history_gate_failed(health: object, threshold: str) -> bool:
    if threshold == "none":
        return False
    return HISTORY_HEALTH_RANK.get(str(health), 2) >= HISTORY_HEALTH_RANK[threshold]


def _parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_csv(value: str | None) -> list[str] | None:
    if value in (None, ""):
        return None
    parsed = [item.strip() for item in str(value).split(",") if item.strip()]
    return parsed or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant_etf_lab", description="A-share ETF research backtesting lab.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data", help="Data cache commands.")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    update_parser = data_subparsers.add_parser("update", help="Fetch and cache ETF history.")
    _add_config_arg(update_parser)
    update_parser.add_argument("--start", default=None, help="Start date in YYYYMMDD.")
    update_parser.add_argument("--end", default=None, help="End date in YYYYMMDD.")
    update_parser.add_argument("--skip-existing", action="store_true", help="Reuse cached files and fetch missing symbols only.")
    update_parser.add_argument("--retry-count", type=int, default=3, help="Fetch retry count per symbol.")
    update_parser.add_argument("--pause-seconds", type=float, default=0.5, help="Pause between external data requests.")
    update_parser.add_argument("--continue-on-error", action="store_true", help="Continue updating other symbols when one symbol fails.")
    market_cap_parser = data_subparsers.add_parser("update-market-cap", help="Fetch and cache A-share total market-cap snapshot.")
    market_cap_parser.add_argument(
        "--output",
        default=str(DEFAULT_STOCK_MARKET_CAP_PATH),
        help="Output CSV for normalized market cap in Yi Yuan.",
    )
    market_cap_parser.add_argument("--retry-count", type=int, default=3, help="Fetch retry count.")
    market_cap_parser.add_argument("--pause-seconds", type=float, default=1.0, help="Pause between external data retries.")
    market_cap_parser.add_argument(
        "--symbols-file",
        default=None,
        help="Optional CSV with code/name columns; fetch market cap only for these symbols using the daily outstanding-share fallback.",
    )
    market_cap_parser.add_argument("--as-of-date", default=None, help="As-of date for --symbols-file fallback, YYYY-MM-DD or YYYYMMDD.")
    market_cap_parser.add_argument("--lookback-days", type=int, default=10, help="Calendar-day lookback for the symbol fallback.")

    backtest_parser = subparsers.add_parser("backtest", help="Run a configured backtest.")
    _add_config_arg(backtest_parser)
    backtest_parser.add_argument("--run-id", default=None, help="Optional run id for output directory.")
    backtest_parser.add_argument("--skip-missing", action="store_true", help="Skip symbols whose history is missing or cannot be fetched.")

    report_parser = subparsers.add_parser("report", help="Generate a Markdown report for a run.")
    report_parser.add_argument("--run-id", default="latest", help="Run id or 'latest'.")
    report_parser.add_argument("--output-dir", default="outputs/backtests", help="Backtest output directory.")

    diagnostics_parser = subparsers.add_parser(
        "diagnostics",
        help="Run linear algebra, statistics, and econometrics diagnostics for a backtest run.",
    )
    diagnostics_parser.add_argument("--run-id", default="latest", help="Run id or 'latest'.")
    diagnostics_parser.add_argument("--output-dir", default="outputs/backtests", help="Backtest output directory.")
    diagnostics_parser.add_argument("--equity-column", default="equity", help="Equity column in equity_curve.csv.")
    diagnostics_parser.add_argument(
        "--benchmark-column",
        default="benchmark_equity",
        help="Benchmark equity column in benchmark.csv.",
    )
    diagnostics_parser.add_argument(
        "--hac-lags",
        default="auto",
        help="Newey-West/HAC lag count, or 'auto'.",
    )

    attribution_parser = subparsers.add_parser(
        "attribution",
        help="Attribute drawdowns using equity, trades, benchmark, and risk records.",
    )
    attribution_parser.add_argument("--run-id", default="latest", help="Run id or 'latest'.")
    attribution_parser.add_argument("--output-dir", default="outputs/backtests", help="Backtest output directory.")
    attribution_parser.add_argument("--min-depth", type=float, default=0.05, help="Minimum drawdown depth to report.")
    attribution_parser.add_argument("--top-n", type=int, default=5, help="Number of deepest episodes to attribute.")

    diagnostics_batch_parser = subparsers.add_parser(
        "diagnostics-batch",
        help="Run diagnostics for multiple backtest runs and write a comparison table.",
    )
    diagnostics_batch_parser.add_argument("run_ids", nargs="*", help="Run ids under --output-dir.")
    diagnostics_batch_parser.add_argument("--run-glob", action="append", default=[], help="Glob pattern under --output-dir.")
    diagnostics_batch_parser.add_argument("--output-dir", default="outputs/backtests", help="Backtest output directory.")
    diagnostics_batch_parser.add_argument(
        "--batch-output-dir",
        default="outputs/diagnostics_batch",
        help="Directory for combined diagnostics outputs.",
    )
    diagnostics_batch_parser.add_argument("--equity-column", default="equity", help="Equity column in equity_curve.csv.")
    diagnostics_batch_parser.add_argument(
        "--benchmark-column",
        default="benchmark_equity",
        help="Benchmark equity column in benchmark.csv.",
    )
    diagnostics_batch_parser.add_argument(
        "--hac-lags",
        default="auto",
        help="Newey-West/HAC lag count, or 'auto'.",
    )

    factor_lab_parser = subparsers.add_parser(
        "factor-lab",
        help="Run local cross-sectional factor IC and quantile-return research.",
    )
    factor_lab_parser.add_argument("--data-dir", default="data/processed/stocks", help="Directory containing cached OHLCV CSV files.")
    factor_lab_parser.add_argument("--output-dir", default="outputs/research/factor_lab_latest", help="Directory for factor-lab outputs.")
    factor_lab_parser.add_argument("--factors", default=None, help="Comma-separated factor list. Defaults to built-in baseline factors.")
    factor_lab_parser.add_argument("--horizons", default="1,5,20", help="Comma-separated forward return horizons in trading days.")
    factor_lab_parser.add_argument("--quantiles", type=int, default=5, help="Number of factor quantiles.")
    factor_lab_parser.add_argument("--min-obs", type=int, default=30, help="Minimum cross-section observations per date.")
    factor_lab_parser.add_argument("--max-symbols", type=int, default=None, help="Optional symbol cap for faster exploratory runs.")
    factor_lab_parser.add_argument("--start-date", default=None, help="Optional factor evaluation start date, YYYY-MM-DD or YYYYMMDD.")
    factor_lab_parser.add_argument("--end-date", default=None, help="Optional factor evaluation end date, YYYY-MM-DD or YYYYMMDD.")
    factor_lab_parser.add_argument("--warmup-days", type=int, default=180, help="Calendar-day warmup before --start-date for rolling factors.")
    factor_lab_parser.add_argument("--no-save-panel", dest="save_panel", action="store_false", default=True, help="Skip writing the large factor_panel.csv file.")
    factor_lab_parser.add_argument("--recursive", action="store_true", help="Read CSV files recursively under --data-dir.")

    theme_state_parser = subparsers.add_parser(
        "theme-state",
        help="Build research-only theme breadth, second-activation, and beta-gap diagnostics.",
    )
    theme_state_parser.add_argument("--data-dir", default="data/processed/stocks", help="Directory containing cached OHLCV CSV files.")
    theme_state_parser.add_argument("--output-dir", default="outputs/research/theme_state_latest", help="Directory for theme-state outputs.")
    theme_state_parser.add_argument("--theme-map", default=None, help="Optional CSV with code plus theme/concept/industry/board column.")
    theme_state_parser.add_argument("--group-column", default=None, help="Optional group column already present in stock history CSV files.")
    theme_state_parser.add_argument("--horizons", default="1,5,20", help="Comma-separated forward return horizons in trading days.")
    theme_state_parser.add_argument("--max-symbols", type=int, default=None, help="Optional symbol cap for faster exploratory runs.")
    theme_state_parser.add_argument("--start-date", default=None, help="Optional evaluation start date, YYYY-MM-DD or YYYYMMDD.")
    theme_state_parser.add_argument("--end-date", default=None, help="Optional evaluation end date, YYYY-MM-DD or YYYYMMDD.")
    theme_state_parser.add_argument("--warmup-days", type=int, default=180, help="Calendar-day warmup before --start-date for rolling state.")
    theme_state_parser.add_argument("--recursive", action="store_true", help="Read CSV files recursively under --data-dir.")
    theme_state_parser.add_argument("--breadth-window", type=int, default=20, help="Rolling window for smoothed theme breadth.")
    theme_state_parser.add_argument("--slope-window", type=int, default=5, help="Lookback used for breadth slope.")
    theme_state_parser.add_argument("--quiet-lookback", type=int, default=20, help="Lookback used to detect second activation from a quiet state.")
    theme_state_parser.add_argument("--activation-breadth", type=float, default=0.55, help="Smoothed breadth threshold for theme activation.")
    theme_state_parser.add_argument("--prior-breadth-ceiling", type=float, default=0.55, help="Prior smoothed breadth ceiling for second activation.")
    theme_state_parser.add_argument("--min-breadth-slope", type=float, default=0.03, help="Minimum breadth slope for second activation.")
    theme_state_parser.add_argument("--min-rs-20d", type=float, default=0.0, help="Minimum 20-day relative strength for activation.")
    theme_state_parser.add_argument("--beta-window", type=int, default=60, help="Rolling window for beta-gap calculation.")
    theme_state_parser.add_argument("--include-inactive-beta-gap", dest="theme_active_only", action="store_false", default=True, help="Keep beta-gap rows even when the theme is not healthy or second-activation.")

    theme_map_parser = subparsers.add_parser(
        "theme-map",
        help="Build stock-to-theme maps from local CSV/THS block files plus market-snapshot board fallback.",
    )
    theme_map_subparsers = theme_map_parser.add_subparsers(dest="theme_map_command", required=True)
    theme_map_build_parser = theme_map_subparsers.add_parser("build", help="Build a research-only theme map CSV.")
    theme_map_build_parser.add_argument(
        "--output-dir",
        default=DEFAULT_THEME_MAP_OUTPUT_DIR.as_posix(),
        help="Directory for theme-map build outputs.",
    )
    theme_map_build_parser.add_argument("--output", default=None, help="Optional output CSV path. Defaults to output-dir/theme_map.csv.")
    theme_map_build_parser.add_argument("--source", action="append", default=None, help="Explicit CSV/INI source path. May be repeated.")
    theme_map_build_parser.add_argument("--scan-root", action="append", default=None, help="Directory or file to scan for theme sources. May be repeated.")
    theme_map_build_parser.add_argument(
        "--no-default-scan-roots",
        dest="include_default_scan_roots",
        action="store_false",
        default=True,
        help="Do not scan project configs/data/outputs and daily-market-data exports.",
    )
    theme_map_build_parser.add_argument(
        "--no-board-fallback",
        dest="include_board_fallback",
        action="store_false",
        default=True,
        help="Do not fill unmapped snapshot symbols with main/chinext/star/bse board groups.",
    )
    theme_map_build_parser.add_argument("--max-scan-files", type=int, default=2000, help="Maximum discovered source files to inspect.")
    theme_map_build_parser.add_argument("--trade-date", default=None, help="Market snapshot date, YYYY-MM-DD or YYYYMMDD. Defaults to latest.")
    theme_map_build_parser.add_argument(
        "--data-cache-dir",
        default=DEFAULT_DAILY_MARKET_DATA_DIR_ARG,
        help="Unified daily market data hub.",
    )
    theme_map_build_parser.add_argument(
        "--ingest-project-dir",
        default=DEFAULT_EXCHANGE_INGEST_DIR_ARG,
        help="Fallback exchange-ingest project with market_data_utils.py.",
    )
    theme_map_build_parser.add_argument(
        "--allow-stale-fetch",
        dest="require_success",
        action="store_false",
        default=True,
        help="Allow reading even if latest exchange-ingest status is not ok.",
    )

    market_timing_parser = subparsers.add_parser(
        "market-timing-state",
        help="Build research-only five-dimensional market timing observation state.",
    )
    market_timing_parser.add_argument("--data-dir", default="data/processed/stocks", help="Directory containing cached OHLCV CSV files.")
    market_timing_parser.add_argument(
        "--output-dir",
        default="outputs/research/market_timing_state_latest",
        help="Directory for market-timing-state outputs.",
    )
    market_timing_parser.add_argument(
        "--data-cache-dir",
        default=DEFAULT_DAILY_MARKET_DATA_DIR_ARG,
        help="Unified daily market data hub.",
    )
    market_timing_parser.add_argument(
        "--ingest-project-dir",
        default=DEFAULT_EXCHANGE_INGEST_DIR_ARG,
        help="Fallback exchange-ingest project with market_data_utils.py.",
    )
    market_timing_parser.add_argument("--trade-date", default=None, help="Market snapshot trade date, YYYY-MM-DD or YYYYMMDD. Defaults to latest.")
    market_timing_parser.add_argument("--lookback-days", type=int, default=90, help="Recent history window used for local trend metrics.")
    market_timing_parser.add_argument("--max-symbols", type=int, default=None, help="Optional symbol cap for faster exploratory runs.")
    market_timing_parser.add_argument("--recursive", action="store_true", help="Read CSV files recursively under --data-dir.")
    market_timing_parser.add_argument(
        "--allow-stale-fetch",
        dest="require_success",
        action="store_false",
        default=True,
        help="Allow reading even if latest exchange-ingest status is not ok.",
    )

    network_lab_parser = subparsers.add_parser(
        "network-lab",
        help="Build a research-only stock dependency network from local log returns.",
    )
    network_lab_parser.add_argument("--data-dir", default="data/processed/stocks", help="Directory containing cached OHLCV CSV files.")
    network_lab_parser.add_argument("--output-dir", default="outputs/research/network_lab_latest", help="Directory for network-lab outputs.")
    network_lab_parser.add_argument("--symbols", default=None, help="Comma-separated symbol list. Defaults to CSV files under --data-dir.")
    network_lab_parser.add_argument("--symbols-file", default=None, help="Optional CSV file with a code column or first-column symbols.")
    network_lab_parser.add_argument("--max-symbols", type=int, default=None, help="Optional symbol cap for faster exploratory runs.")
    network_lab_parser.add_argument("--start-date", default=None, help="Optional return panel start date, YYYY-MM-DD or YYYYMMDD.")
    network_lab_parser.add_argument("--end-date", default=None, help="Optional return panel end date, YYYY-MM-DD or YYYYMMDD.")
    network_lab_parser.add_argument("--lookback-days", type=int, default=120, help="Number of recent observations to keep per symbol before return calculation.")
    network_lab_parser.add_argument("--top-edges", type=int, default=50, help="Number of strongest hidden-linkage edges to write.")
    network_lab_parser.add_argument("--bins", type=int, default=8, help="Histogram bins for mutual-information estimation.")
    network_lab_parser.add_argument("--min-obs", type=int, default=60, help="Minimum overlapping return observations per pair.")
    network_lab_parser.add_argument("--recursive", action="store_true", help="Read CSV files recursively under --data-dir.")

    momentum_focus_parser = subparsers.add_parser(
        "momentum-focus",
        help="Build a research-only focus pool from limit-up and >=7%% A-share gainers.",
    )
    momentum_focus_parser.add_argument(
        "--output-dir",
        default="outputs/research/momentum_focus_latest",
        help="Directory for momentum-focus outputs.",
    )
    momentum_focus_parser.add_argument(
        "--data-cache-dir",
        default=DEFAULT_DAILY_MARKET_DATA_DIR_ARG,
        help="Unified daily market data hub.",
    )
    momentum_focus_parser.add_argument(
        "--ingest-project-dir",
        default=DEFAULT_EXCHANGE_INGEST_DIR_ARG,
        help="Exchange ingest project used for fetch-status checks and fallback data.",
    )
    momentum_focus_parser.add_argument("--as-of-date", default=None, help="Override the focus date, YYYY-MM-DD or YYYYMMDD.")
    momentum_focus_parser.add_argument(
        "--trade-date",
        default=None,
        help="Override the market snapshot trade date; defaults to --as-of-date.",
    )
    momentum_focus_parser.add_argument(
        "--strong-gain-threshold-pct",
        type=float,
        default=7.0,
        help="Minimum daily percentage gain for the strong-gain research bucket.",
    )
    momentum_focus_parser.add_argument(
        "--board-scope",
        choices=["main_chinext", "all"],
        default="main_chinext",
        help="Stock board universe for the focus pool.",
    )
    momentum_focus_parser.add_argument(
        "--outcome-summary-path",
        default="outputs/research/momentum_outcomes_latest/momentum_outcome_summary.csv",
        help="Optional path to momentum-outcome summary CSV.",
    )
    momentum_focus_parser.add_argument(
        "--name-map-path",
        default=DEFAULT_NAME_MAP_PATH.as_posix(),
        help="Optional code/name CSV used to fill unknown names in the focus pool.",
    )
    momentum_focus_parser.add_argument(
        "--target-horizon",
        type=int,
        default=5,
        help="Target outcome horizon used for prior lookup.",
    )

    momentum_outcomes_parser = subparsers.add_parser(
        "momentum-outcomes",
        help="Evaluate historical outcomes for limit-up and >=7%% A-share events.",
    )
    momentum_outcomes_parser.add_argument(
        "--data-dir",
        default="data/processed/stocks",
        help="Directory containing locally cached stock daily CSV files.",
    )
    momentum_outcomes_parser.add_argument(
        "--output-dir",
        default="outputs/research/momentum_outcomes_latest",
        help="Directory for momentum outcome outputs.",
    )
    momentum_outcomes_parser.add_argument("--horizons", default="1,3,5,10", help="Comma-separated holding horizons.")
    momentum_outcomes_parser.add_argument(
        "--strong-gain-threshold-pct",
        type=float,
        default=7.0,
        help="Minimum signal-day gain for the strong-gain research bucket.",
    )
    momentum_outcomes_parser.add_argument(
        "--board-scope",
        choices=["main_chinext", "all"],
        default="main_chinext",
        help="Stock board universe for historical event evaluation.",
    )
    momentum_outcomes_parser.add_argument("--min-events", type=int, default=20, help="Minimum mature events per summary row.")
    momentum_outcomes_parser.add_argument("--max-symbols", type=int, default=None, help="Optional symbol cap for faster smoke runs.")
    momentum_outcomes_parser.add_argument("--start-date", default=None, help="Optional evaluation start date, YYYY-MM-DD or YYYYMMDD.")
    momentum_outcomes_parser.add_argument("--end-date", default=None, help="Optional evaluation end date, YYYY-MM-DD or YYYYMMDD.")
    momentum_outcomes_parser.add_argument("--recursive", action="store_true", help="Read CSV files recursively under --data-dir.")

    chip_reversal_parser = subparsers.add_parser(
        "chip-reversal-lab",
        help="Evaluate research-only daily proxy outcomes for enhanced chip reversal setups.",
    )
    chip_reversal_parser.add_argument(
        "--data-dir",
        default="data/processed/stocks",
        help="Directory containing locally cached stock daily CSV files.",
    )
    chip_reversal_parser.add_argument(
        "--output-dir",
        default="outputs/research/chip_reversal_lab_latest",
        help="Directory for chip reversal lab outputs.",
    )
    chip_reversal_parser.add_argument("--horizons", default="1,2", help="Comma-separated holding horizons.")
    chip_reversal_parser.add_argument(
        "--drawdown-window",
        type=int,
        default=20,
        help="Prior high window for the daily drawdown proxy.",
    )
    chip_reversal_parser.add_argument(
        "--cost-window",
        type=int,
        default=20,
        help="Moving-average window used as the daily cost proxy.",
    )
    chip_reversal_parser.add_argument(
        "--min-drawdown-pct",
        type=float,
        default=12.0,
        help="Minimum drawdown from prior high for candidate events.",
    )
    chip_reversal_parser.add_argument(
        "--min-score",
        type=float,
        default=0.08,
        help="Minimum daily proxy chip reversal score.",
    )
    chip_reversal_parser.add_argument(
        "--score-bucket",
        choices=["all", "base", "strong", "deep"],
        default="all",
        help="Optional score bucket filter applied after daily-proxy candidate generation.",
    )
    chip_reversal_parser.add_argument(
        "--min-amount-yi",
        type=float,
        default=0.0,
        help="Optional minimum signal-day amount in 100M CNY units.",
    )
    chip_reversal_parser.add_argument(
        "--max-next-open-gap-pct",
        type=float,
        default=None,
        help="Optional next-open high-gap filter, applied as a research execution filter.",
    )
    chip_reversal_parser.add_argument(
        "--theme-state-path",
        default=None,
        help="Optional theme_state.csv used as a research-only theme confirmation gate.",
    )
    chip_reversal_parser.add_argument(
        "--theme-symbol-panel-path",
        default=None,
        help="Optional theme_symbol_panel.csv used to map stock/date events to theme groups.",
    )
    chip_reversal_parser.add_argument(
        "--allowed-theme-states",
        default="healthy,second_activation",
        help="Comma-separated theme states allowed by the optional theme confirmation gate.",
    )
    chip_reversal_parser.add_argument(
        "--board-scope",
        choices=["main_chinext", "all"],
        default="main_chinext",
        help="Stock board universe for chip reversal evaluation.",
    )
    chip_reversal_parser.add_argument("--min-events", type=int, default=20, help="Minimum mature events per summary row.")
    chip_reversal_parser.add_argument("--max-symbols", type=int, default=None, help="Optional symbol cap for faster smoke runs.")
    chip_reversal_parser.add_argument("--start-date", default=None, help="Optional evaluation start date, YYYY-MM-DD or YYYYMMDD.")
    chip_reversal_parser.add_argument("--end-date", default=None, help="Optional evaluation end date, YYYY-MM-DD or YYYYMMDD.")
    chip_reversal_parser.add_argument("--recursive", action="store_true", help="Read CSV files recursively under --data-dir.")

    chip_reversal_pit_parser = subparsers.add_parser(
        "chip-reversal-pit-audit",
        help="Audit chip-reversal daily proxy events for point-in-time selector consistency.",
    )
    chip_reversal_pit_parser.add_argument(
        "--data-dir",
        default="data/processed/stocks",
        help="Directory containing locally cached stock daily CSV files.",
    )
    chip_reversal_pit_parser.add_argument(
        "--output-dir",
        default="outputs/research/chip_reversal_pit_audit_latest",
        help="Directory for PIT audit outputs.",
    )
    chip_reversal_pit_parser.add_argument("--horizons", default="1,2", help="Comma-separated holding horizons.")
    chip_reversal_pit_parser.add_argument("--drawdown-window", type=int, default=20, help="Prior high window.")
    chip_reversal_pit_parser.add_argument("--cost-window", type=int, default=20, help="Moving-average cost window.")
    chip_reversal_pit_parser.add_argument("--min-drawdown-pct", type=float, default=12.0, help="Minimum drawdown.")
    chip_reversal_pit_parser.add_argument("--min-score", type=float, default=0.08, help="Minimum daily proxy score.")
    chip_reversal_pit_parser.add_argument(
        "--score-bucket",
        choices=["all", "base", "strong", "deep"],
        default="all",
        help="Score bucket label recorded in the audit snapshot.",
    )
    chip_reversal_pit_parser.add_argument(
        "--board-scope",
        choices=["main_chinext", "all"],
        default="main_chinext",
        help="Stock board universe for the PIT audit.",
    )
    chip_reversal_pit_parser.add_argument("--max-symbols", type=int, default=None, help="Optional symbol cap for smoke runs.")
    chip_reversal_pit_parser.add_argument("--start-date", default=None, help="Optional signal start date.")
    chip_reversal_pit_parser.add_argument("--end-date", default=None, help="Optional signal end date.")
    chip_reversal_pit_parser.add_argument("--max-events", type=int, default=2000, help="Maximum events to audit.")
    chip_reversal_pit_parser.add_argument("--recursive", action="store_true", help="Read CSV files recursively under --data-dir.")

    chip_reversal_queue_parser = subparsers.add_parser(
        "chip-reversal-observation-queue",
        help="Build an observation-only queue from chip-reversal research events.",
    )
    chip_reversal_queue_parser.add_argument(
        "--events-path",
        default="outputs/research/chip_reversal_lab_latest/chip_reversal_events.csv",
        help="Input chip_reversal_events.csv path.",
    )
    chip_reversal_queue_parser.add_argument(
        "--output-dir",
        default="outputs/research/chip_reversal_observation_queue_latest",
        help="Directory for observation queue outputs.",
    )
    chip_reversal_queue_parser.add_argument("--as-of-date", default=None, help="Signal date to queue; defaults to latest date.")
    chip_reversal_queue_parser.add_argument("--max-candidates", type=int, default=20, help="Maximum queue rows.")
    chip_reversal_queue_parser.add_argument("--min-priority-score", type=float, default=0.0, help="Minimum priority score.")

    chip_reversal_daily_parser = subparsers.add_parser(
        "chip-reversal-daily-candidates",
        help="Build close-based, non-mature, observation-only chip-reversal candidates.",
    )
    chip_reversal_daily_parser.add_argument(
        "--data-dir",
        default="data/processed/stocks",
        help="Directory containing locally cached stock daily CSV files.",
    )
    chip_reversal_daily_parser.add_argument(
        "--output-dir",
        default="outputs/research/chip_reversal_daily_candidates_latest",
        help="Directory for latest daily candidate outputs.",
    )
    chip_reversal_daily_parser.add_argument("--as-of-date", default=None, help="Candidate date; defaults to latest cached date.")
    chip_reversal_daily_parser.add_argument("--horizons", default="1,2", help="Comma-separated horizons to redact from non-mature output.")
    chip_reversal_daily_parser.add_argument("--drawdown-window", type=int, default=20, help="Prior high window.")
    chip_reversal_daily_parser.add_argument("--cost-window", type=int, default=20, help="Moving-average cost window.")
    chip_reversal_daily_parser.add_argument("--min-drawdown-pct", type=float, default=12.0, help="Minimum drawdown.")
    chip_reversal_daily_parser.add_argument("--min-score", type=float, default=0.08, help="Minimum daily proxy score.")
    chip_reversal_daily_parser.add_argument(
        "--score-bucket",
        choices=["all", "base", "strong", "deep"],
        default="deep",
        help="Score bucket filter for latest candidates.",
    )
    chip_reversal_daily_parser.add_argument(
        "--min-amount-yi",
        type=float,
        default=1.5,
        help="Minimum signal-day amount in 100M CNY units.",
    )
    chip_reversal_daily_parser.add_argument(
        "--board-scope",
        choices=["main_chinext", "all"],
        default="main_chinext",
        help="Stock board universe for latest candidates.",
    )
    chip_reversal_daily_parser.add_argument("--max-candidates", type=int, default=50, help="Maximum candidate rows.")
    chip_reversal_daily_parser.add_argument("--max-symbols", type=int, default=None, help="Optional symbol cap for smoke runs.")
    chip_reversal_daily_parser.add_argument("--recursive", action="store_true", help="Read CSV files recursively under --data-dir.")
    chip_reversal_daily_parser.add_argument(
        "--daily-data-dir",
        default=DEFAULT_DAILY_MARKET_DATA_DIR_ARG,
        help="Unified daily market data directory used before exchange-ingest fallback.",
    )
    chip_reversal_daily_parser.add_argument(
        "--ingest-project-dir",
        default=DEFAULT_EXCHANGE_INGEST_DIR_ARG,
        help="Exchange ingest project used for fetch-status checks and fallback rows.",
    )
    chip_reversal_daily_parser.set_defaults(market_snapshot_overlay=True)
    chip_reversal_daily_parser.add_argument(
        "--no-market-snapshot-overlay",
        dest="market_snapshot_overlay",
        action="store_false",
        help="Disable overlaying the latest unified market snapshot onto historical stock CSVs.",
    )

    chip_reversal_outcomes_parser = subparsers.add_parser(
        "chip-reversal-candidate-outcomes",
        help="Backfill maturity and forward outcomes for observation-only chip-reversal candidates.",
    )
    chip_reversal_outcomes_parser.add_argument(
        "--candidates-path",
        default="outputs/research/chip_reversal_daily_candidates_latest/chip_reversal_daily_candidates.csv",
        help="Input chip_reversal_daily_candidates.csv path.",
    )
    chip_reversal_outcomes_parser.add_argument(
        "--data-dir",
        default="data/processed/stocks",
        help="Directory containing locally cached stock daily CSV files.",
    )
    chip_reversal_outcomes_parser.add_argument(
        "--output-dir",
        default="outputs/research/chip_reversal_candidate_outcomes_latest",
        help="Directory for candidate outcome outputs.",
    )
    chip_reversal_outcomes_parser.add_argument("--horizons", default="1,2", help="Comma-separated forward horizons.")
    chip_reversal_outcomes_parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.03,
        help="Close-to-close return threshold considered successful.",
    )
    chip_reversal_outcomes_parser.add_argument(
        "--min-ready-per-horizon",
        type=int,
        default=30,
        help="Minimum ready outcomes required for each horizon before chip-reversal can pass the research gate.",
    )
    chip_reversal_outcomes_parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.55,
        help="Minimum success rate required for each horizon before chip-reversal can pass the research gate.",
    )
    chip_reversal_outcomes_parser.add_argument(
        "--daily-data-dir",
        default=DEFAULT_DAILY_MARKET_DATA_DIR_ARG,
        help="Unified daily market data directory used before exchange-ingest fallback.",
    )
    chip_reversal_outcomes_parser.add_argument(
        "--ingest-project-dir",
        default=DEFAULT_EXCHANGE_INGEST_DIR_ARG,
        help="Exchange ingest project used for fetch-status checks and fallback rows.",
    )
    chip_reversal_outcomes_parser.set_defaults(market_snapshot_overlay=True)
    chip_reversal_outcomes_parser.add_argument(
        "--no-market-snapshot-overlay",
        dest="market_snapshot_overlay",
        action="store_false",
        help="Disable overlaying the latest unified market snapshot onto candidate outcome histories.",
    )

    validate_parser = subparsers.add_parser("validate", help="Run train/validation/out-of-sample checks.")
    _add_config_arg(validate_parser)
    validate_parser.add_argument("--output-dir", default="outputs/validation", help="Validation output directory.")
    validate_parser.add_argument("--run-id-prefix", default=None, help="Optional validation directory name.")
    validate_parser.add_argument("--allow-fetch", action="store_true", help="Fetch missing data instead of requiring cache.")

    sentiment_parser = subparsers.add_parser("sentiment", help="Write a local A-share market sentiment state report.")
    _add_config_arg(sentiment_parser)
    sentiment_parser.add_argument(
        "--output-dir",
        default="outputs/research/market_sentiment_state_latest",
        help="Directory for sentiment state outputs.",
    )
    sentiment_parser.add_argument("--max-symbols", type=int, default=None, help="Optional universe sample size.")
    sentiment_parser.add_argument("--window", type=int, default=120, help="Rolling z-score window.")
    sentiment_parser.add_argument("--strict", action="store_true", help="Fail when any cached history is missing.")

    phase2_parser = subparsers.add_parser("phase2", help="Write the phase-2 model construction status report.")
    phase2_parser.add_argument(
        "--output-dir",
        default="outputs/research/phase2_model_status_latest",
        help="Directory for phase-2 status outputs.",
    )
    phase2_parser.add_argument("--baseline-run-dir", default=None, help="Backtest run directory for the current baseline.")
    phase2_parser.add_argument("--core-wf-dir", default=None, help="Walk-forward directory for the defensive core.")
    phase2_parser.add_argument("--satellite-wf-dir", default=None, help="Walk-forward directory for the satellite model.")
    phase2_parser.add_argument("--portfolio-run-dir", default=None, help="Portfolio combine run directory.")
    phase2_parser.add_argument("--portfolio-wf-dir", default=None, help="Portfolio allocation walk-forward directory.")

    phase2_review_parser = subparsers.add_parser(
        "phase2-review",
        help="Write a monitoring-style review for a phase-2 allocator walk-forward run.",
    )
    phase2_review_parser.add_argument(
        "--allocator-dir",
        "--allocator-wf-dir",
        dest="allocator_dir",
        default=DEFAULT_RESEARCH_ALLOCATOR_DIR_ARG,
        help="Portfolio allocator walk-forward directory.",
    )
    phase2_review_parser.add_argument(
        "--components-path",
        "--phase2-status-dir",
        dest="components_path",
        default=DEFAULT_PHASE2_COMPONENTS_PATH_ARG,
        help="Path to phase2_components.csv or its containing phase-2 status directory.",
    )
    phase2_review_parser.add_argument(
        "--output-dir",
        default="outputs/research/phase2_monitoring_review_latest",
        help="Directory for monitoring review outputs.",
    )
    phase2_review_parser.add_argument(
        "--promotion-review-dir",
        default=None,
        help="Optional allocator-promotion-review output directory.",
    )

    daily_check_parser = subparsers.add_parser(
        "daily-check",
        help="Run the after-close daily model check using the current phase-2 review inputs.",
    )
    daily_check_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for daily check outputs. Defaults to latest, or a dated directory with --date-stamp.",
    )
    daily_check_parser.add_argument(
        "--phase2-review-dir",
        default=None,
        help="Optional nested directory for the phase-2 review generated by this daily check.",
    )
    daily_check_parser.add_argument(
        "--allocator-dir",
        "--allocator-wf-dir",
        dest="allocator_dir",
        default=DEFAULT_RESEARCH_ALLOCATOR_DIR_ARG,
        help="Portfolio allocator walk-forward directory.",
    )
    daily_check_parser.add_argument(
        "--components-path",
        "--phase2-status-dir",
        dest="components_path",
        default=DEFAULT_PHASE2_COMPONENTS_PATH_ARG,
        help="Path to phase2_components.csv or its containing phase-2 status directory.",
    )
    daily_check_parser.add_argument(
        "--promotion-review-dir",
        default=None,
        help="Optional allocator-promotion-review output directory.",
    )
    daily_check_parser.add_argument(
        "--max-staleness-days",
        type=int,
        default=3,
        help="Maximum allowed days between local equity date and as-of date.",
    )
    daily_check_parser.add_argument("--as-of-date", default=None, help="Override the check date, YYYY-MM-DD or YYYYMMDD.")
    daily_check_parser.add_argument(
        "--date-stamp",
        action="store_true",
        help="Write to an output directory suffixed with the as-of date.",
    )

    paper_account_parser = subparsers.add_parser(
        "paper-account",
        help="Build a paper-account ledger and rebalance audit from portfolio allocator outputs.",
    )
    paper_account_parser.add_argument(
        "--config",
        default="configs/portfolio_core_satellite_quality_v2_guarded.yaml",
        help="Path to the core-satellite portfolio config.",
    )
    paper_account_parser.add_argument(
        "--allocator-dir",
        "--allocator-wf-dir",
        dest="allocator_dir",
        default=DEFAULT_RESEARCH_ALLOCATOR_DIR_ARG,
        help="Portfolio allocator walk-forward directory.",
    )
    paper_account_parser.add_argument(
        "--output-dir",
        default="outputs/research/paper_account_latest",
        help="Directory for paper-account outputs.",
    )
    paper_account_parser.add_argument(
        "--rebalance-cost-rate",
        type=float,
        default=0.0,
        help="Optional allocation-level target-change cost rate. Defaults to 0 to avoid double-counting strategy costs.",
    )
    paper_account_parser.add_argument(
        "--trigger-signal-path",
        default=None,
        help="Optional latest trigger-monitor signals CSV. Defaults to D:/codex/outputs/signal_history/signals_latest.csv.",
    )
    paper_account_parser.add_argument(
        "--stock-market-cap-path",
        default=None,
        help="Normalized A-share market-cap cache CSV. Defaults to data/processed/stock_market_cap_yi.csv.",
    )
    paper_account_parser.add_argument(
        "--stock-tracking-max-market-cap-yi",
        type=float,
        default=1500.0,
        help="Skip stock-target tracking when total market cap is above this Yi Yuan threshold.",
    )
    paper_account_parser.add_argument(
        "--stock-review-notes-path",
        default=None,
        help="Persistent manual notes CSV for stock-target review rows.",
    )
    paper_account_parser.add_argument(
        "--stock-review-outcomes-history-path",
        default=None,
        help="Persistent cumulative outcome-history CSV for stock-target review rows.",
    )
    paper_account_parser.add_argument("--stock-review-drawdown-threshold", type=float, default=-0.10, help="Unrealized-return threshold for drawdown_review rows.")
    paper_account_parser.add_argument("--stock-review-watch-drawdown-threshold", type=float, default=-0.07, help="Unrealized-return threshold for watch-level loss rows.")
    paper_account_parser.add_argument("--stock-review-loss-attention-threshold", type=float, default=-0.05, help="Unrealized-return threshold for lower-priority loss attention.")
    paper_account_parser.add_argument("--stock-review-gain-attention-threshold", type=float, default=0.10, help="Unrealized-return threshold for trailing-risk gain attention.")
    paper_account_parser.add_argument("--stock-review-watch-score-threshold", type=float, default=30.0, help="Priority score threshold for watch_review rows.")
    paper_account_parser.add_argument("--stock-review-outcome-min-evaluable", type=int, default=20, help="Minimum evaluable rows before outcome analysis is treated as review-ready.")
    paper_account_parser.add_argument("--stock-review-outcome-min-group-evaluable", type=int, default=5, help="Minimum evaluable rows per group before grouped outcome analysis is treated as review-ready.")

    stock_review_apply_parser = subparsers.add_parser(
        "stock-review-apply-template",
        help="Apply filled stock-target review decision-template rows to the persistent local notes CSV.",
    )
    stock_review_apply_parser.add_argument(
        "--template-path",
        default="outputs/research/paper_account_latest/stock_target_review_decision_template.csv",
        help="Filled stock_target_review_decision_template CSV/XLSX to apply.",
    )
    stock_review_apply_parser.add_argument(
        "--notes-path",
        default="outputs/research/stock_target_review_notes.csv",
        help="Persistent stock-target review notes CSV to update.",
    )
    stock_review_apply_parser.add_argument(
        "--output-dir",
        default="outputs/research/stock_review_template_apply_latest",
        help="Directory for apply audit, snapshot, and report outputs.",
    )
    stock_review_apply_parser.add_argument(
        "--reviewed-at",
        default=None,
        help="Optional timestamp to write into reviewed_at for applied rows.",
    )

    live_shadow_parser = subparsers.add_parser(
        "live-shadow",
        help="Build a manual-review shadow account plan from live holdings and local stock targets.",
    )
    live_shadow_parser.add_argument("--holdings-file", required=True, help="CSV with code, quantity, and optional current_price/market_value.")
    live_shadow_parser.add_argument(
        "--targets-file",
        default="outputs/research/paper_account_latest/stock_targets.csv",
        help="CSV with stock-level target weights. Defaults to latest paper-account stock_targets.csv.",
    )
    live_shadow_parser.add_argument("--prices-file", default=None, help="Optional CSV with code and price/current_price/close.")
    live_shadow_parser.add_argument("--cash", type=float, required=True, help="Current available cash for the shadow account.")
    live_shadow_parser.add_argument(
        "--output-dir",
        default="outputs/research/live_shadow_latest",
        help="Directory for live shadow outputs.",
    )
    live_shadow_parser.add_argument("--as-of-date", default=None, help="Override the shadow plan date, YYYY-MM-DD or YYYYMMDD.")
    live_shadow_parser.add_argument("--lot-size", type=int, default=100, help="A-share buy lot size used for shadow rounding.")
    live_shadow_parser.add_argument("--min-trade-value", type=float, default=1000.0, help="Suppress shadow trades below this amount.")
    live_shadow_parser.add_argument("--max-position-weight", type=float, default=0.20, help="Maximum target weight per stock.")
    live_shadow_parser.add_argument("--max-gross-exposure", type=float, default=1.0, help="Maximum total target stock exposure.")

    live_shadow_preflight_parser = subparsers.add_parser(
        "live-shadow-preflight",
        help="Dry-run the live shadow account plan and write a blocking/warning snapshot.",
    )
    live_shadow_preflight_parser.add_argument(
        "--holdings-file",
        required=True,
        help="CSV with code, quantity, and optional current_price/market_value.",
    )
    live_shadow_preflight_parser.add_argument(
        "--targets-file",
        default="outputs/research/paper_account_latest/stock_targets.csv",
        help="CSV with stock-level target weights. Defaults to latest paper-account stock_targets.csv.",
    )
    live_shadow_preflight_parser.add_argument("--prices-file", default=None, help="Optional CSV with code and price/current_price/close.")
    live_shadow_preflight_parser.add_argument("--cash", type=float, default=0.0, help="Current available cash for the shadow preflight.")
    live_shadow_preflight_parser.add_argument(
        "--output-dir",
        default="outputs/research/live_shadow_preflight_latest",
        help="Directory for preflight output.",
    )
    live_shadow_preflight_parser.add_argument("--as-of-date", default=None, help="Override the shadow as-of date, YYYY-MM-DD or YYYYMMDD.")
    live_shadow_preflight_parser.add_argument("--lot-size", type=int, default=100, help="A-share buy lot size used for shadow rounding.")
    live_shadow_preflight_parser.add_argument(
        "--min-trade-value",
        type=float,
        default=1000.0,
        help="Suppress shadow trades below this amount.",
    )
    live_shadow_preflight_parser.add_argument("--max-position-weight", type=float, default=0.20, help="Maximum target weight per stock.")
    live_shadow_preflight_parser.add_argument(
        "--max-gross-exposure",
        type=float,
        default=1.0,
        help="Maximum total target stock exposure.",
    )
    live_shadow_preflight_parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return a non-zero exit code when the preflight status is blocked.",
    )

    live_shadow_template_parser = subparsers.add_parser(
        "live-shadow-template",
        help="Write a highlighted broker holdings/cash import template for manual live-shadow reconciliation.",
    )
    live_shadow_template_parser.add_argument(
        "--targets-file",
        default="outputs/research/paper_account_latest/stock_targets.csv",
        help="CSV with stock-level target weights. Defaults to latest paper-account stock_targets.csv.",
    )
    live_shadow_template_parser.add_argument(
        "--output-dir",
        default="outputs/research/live_shadow_import_latest",
        help="Directory for live-shadow import template outputs.",
    )
    live_shadow_template_parser.add_argument("--as-of-date", default=None, help="Override the template date, YYYY-MM-DD or YYYYMMDD.")
    live_shadow_template_parser.add_argument(
        "--blank-rows",
        type=int,
        default=10,
        help="Extra blank rows for live holdings that are not in current paper targets.",
    )

    live_shadow_review_parser = subparsers.add_parser(
        "live-shadow-review",
        help="Build a highlighted manual review queue from live-shadow differences and local tracking rules.",
    )
    live_shadow_review_parser.add_argument(
        "--orders-file",
        required=True,
        help="CSV from live-shadow containing manual-review shadow differences.",
    )
    live_shadow_review_parser.add_argument(
        "--targets-file",
        default="outputs/research/paper_account_latest/stock_targets.csv",
        help="CSV with stock-level targets and tracking-rule status.",
    )
    live_shadow_review_parser.add_argument(
        "--output-dir",
        default="outputs/research/live_shadow_review_latest",
        help="Directory for live-shadow review queue outputs.",
    )
    live_shadow_review_parser.add_argument("--as-of-date", default=None, help="Override the review date, YYYY-MM-DD or YYYYMMDD.")
    live_shadow_review_parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return a non-zero exit code when tracking-rule blockers are present.",
    )

    live_shadow_review_apply_parser = subparsers.add_parser(
        "live-shadow-review-apply",
        help="Read a filled live-shadow review workbook and write research-only decision audit outputs.",
    )
    live_shadow_review_apply_parser.add_argument(
        "--review-file",
        required=True,
        help="Filled live-shadow review queue CSV/XLSX.",
    )
    live_shadow_review_apply_parser.add_argument(
        "--output-dir",
        default="outputs/research/live_shadow_review_decisions_latest",
        help="Directory for live-shadow review decision outputs.",
    )
    live_shadow_review_apply_parser.add_argument("--as-of-date", default=None, help="Override the review date, YYYY-MM-DD or YYYYMMDD.")

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Write a unified local research dashboard from daily-check, sentiment, and trigger outputs.",
    )
    dashboard_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for dashboard outputs. Defaults to latest, or a dated directory with --date-stamp.",
    )
    dashboard_parser.add_argument("--daily-check-dir", default=None, help="Directory containing daily_model_check_snapshot.json.")
    dashboard_parser.add_argument("--paper-account-dir", default=None, help="Directory containing paper-account metrics.json.")
    dashboard_parser.add_argument("--sentiment-dir", default=None, help="Directory containing latest_market_sentiment.json.")
    dashboard_parser.add_argument(
        "--data-cache-dir",
        default=DEFAULT_DAILY_MARKET_DATA_DIR_ARG,
        help="Unified daily market data hub. Falls back to the exchange ingest project when latest data is missing.",
    )
    dashboard_parser.add_argument(
        "--allocator-dir",
        "--allocator-wf-dir",
        dest="allocator_dir",
        default=DEFAULT_RESEARCH_ALLOCATOR_DIR_ARG,
        help="Portfolio allocator walk-forward directory.",
    )
    dashboard_parser.add_argument(
        "--trigger-report",
        default="D:/codex/outputs/trigger_reports/latest_trigger.md",
        help="Path to the trigger monitor latest report.",
    )
    dashboard_parser.add_argument(
        "--model-audit-dir",
        default=None,
        help="Directory containing model_build_audit_snapshot.json and walk_forward_run_actions.csv.",
    )
    dashboard_parser.add_argument(
        "--pipeline-history-dir",
        default=None,
        help="Directory containing pipeline_history_review_snapshot.json.",
    )
    dashboard_parser.add_argument(
        "--daily-run-status-dir",
        default=None,
        help="Directory containing daily_run_status_snapshot.json for scheduled refresh and observation status.",
    )
    dashboard_parser.add_argument(
        "--allocator-observation-dir",
        default=None,
        help="Directory containing allocator_observation_snapshot.json when daily-run-status does not point to one.",
    )
    dashboard_parser.add_argument("--max-staleness-days", type=int, default=3, help="Maximum allowed staleness in days.")
    dashboard_parser.add_argument("--min-cache-fresh-ratio", type=float, default=0.90, help="Minimum share of cached CSVs at the latest date.")
    dashboard_parser.add_argument("--as-of-date", default=None, help="Override the check date, YYYY-MM-DD or YYYYMMDD.")
    dashboard_parser.add_argument(
        "--date-stamp",
        action="store_true",
        help="Write to an output directory suffixed with the as-of date.",
    )

    daily_pipeline_parser = subparsers.add_parser(
        "daily-pipeline",
        help="Run daily-check, paper-account, and dashboard as one after-close research workflow.",
    )
    daily_pipeline_parser.add_argument("--output-dir", default=None, help="Directory for pipeline summary outputs.")
    daily_pipeline_parser.add_argument("--daily-check-output-dir", default=None, help="Override daily-check output directory.")
    daily_pipeline_parser.add_argument("--paper-account-output-dir", default=None, help="Override paper-account output directory.")
    daily_pipeline_parser.add_argument("--dashboard-output-dir", default=None, help="Override dashboard output directory.")
    daily_pipeline_parser.add_argument("--momentum-focus-output-dir", default=None, help="Override momentum-focus output directory.")
    daily_pipeline_parser.add_argument(
        "--momentum-focus-board-scope",
        choices=["main_chinext", "all"],
        default="main_chinext",
        help="Board universe for momentum-focus research candidates.",
    )
    daily_pipeline_parser.add_argument(
        "--momentum-focus-threshold",
        type=float,
        default=7.0,
        help="Minimum daily percentage gain for the momentum-focus strong-gain bucket.",
    )
    daily_pipeline_parser.add_argument(
        "--momentum-focus-outcome-summary-path",
        default="outputs/research/momentum_outcomes_latest/momentum_outcome_summary.csv",
        help="Optional path to momentum-outcome summary CSV.",
    )
    daily_pipeline_parser.add_argument(
        "--momentum-focus-target-horizon",
        type=int,
        default=5,
        help="Target horizon used for momentum-focus outcome prior lookup.",
    )
    daily_pipeline_parser.add_argument(
        "--components-path",
        "--phase2-status-dir",
        dest="components_path",
        default=DEFAULT_PHASE2_COMPONENTS_PATH_ARG,
        help="Path to phase2_components.csv or its containing phase-2 status directory.",
    )
    daily_pipeline_parser.add_argument(
        "--allocator-dir",
        "--allocator-wf-dir",
        dest="allocator_dir",
        default=DEFAULT_RESEARCH_ALLOCATOR_DIR_ARG,
        help="Portfolio allocator walk-forward directory.",
    )
    daily_pipeline_parser.add_argument(
        "--config",
        default="configs/portfolio_core_satellite_quality_v2_guarded.yaml",
        help="Path to the core-satellite portfolio config for the paper account.",
    )
    daily_pipeline_parser.add_argument("--phase2-review-dir", default=None, help="Optional nested directory for phase-2 review outputs.")
    daily_pipeline_parser.add_argument("--promotion-review-dir", default=None, help="Optional allocator-promotion-review output directory.")
    daily_pipeline_parser.add_argument("--sentiment-dir", default=None, help="Directory containing latest_market_sentiment.json.")
    daily_pipeline_parser.add_argument(
        "--data-cache-dir",
        default=DEFAULT_DAILY_MARKET_DATA_DIR_ARG,
        help="Unified daily market data hub. Falls back to the exchange ingest project when latest data is missing.",
    )
    daily_pipeline_parser.add_argument(
        "--history-file",
        default="outputs/research/daily_pipeline_history.csv",
        help="CSV file that receives one daily-pipeline history row per run.",
    )
    daily_pipeline_parser.add_argument(
        "--history-review-output-dir",
        default=None,
        help="Directory for the automatic pipeline-history review produced after the daily run.",
    )
    daily_pipeline_parser.add_argument(
        "--satellite-risk-budget-output-dir",
        default=None,
        help="Directory for the automatic satellite risk-budget review produced after the daily run.",
    )
    daily_pipeline_parser.add_argument(
        "--network-lab-snapshot",
        default=None,
        help="Optional network_lab_snapshot.json used as additional hidden-linkage monitoring in satellite risk-budget review.",
    )
    daily_pipeline_parser.add_argument(
        "--network-max-cluster-count-warning",
        type=int,
        default=1,
        help="Hold trial when network cluster_count is below or equal to this threshold.",
    )
    daily_pipeline_parser.add_argument(
        "--network-residual-mi-warning",
        type=float,
        default=0.20,
        help="Hold trial when network top residual MI is above this threshold.",
    )
    daily_pipeline_parser.add_argument(
        "--live-shadow-output-dir",
        default=None,
        help="Directory for optional live-shadow manual review outputs.",
    )
    daily_pipeline_parser.add_argument(
        "--live-shadow-preflight",
        action="store_true",
        default=False,
        help="Run live-shadow preflight before building the live-shadow plan.",
    )
    daily_pipeline_parser.add_argument(
        "--live-shadow-preflight-fail-on-blocked",
        action="store_true",
        default=True,
        help="When enabled, skip live-shadow plan execution if preflight decision is blocked.",
    )
    daily_pipeline_parser.add_argument(
        "--no-live-shadow-preflight-fail-on-blocked",
        action="store_false",
        dest="live_shadow_preflight_fail_on_blocked",
        help="Do not skip live-shadow plan execution when preflight is blocked.",
    )
    daily_pipeline_parser.add_argument(
        "--live-preflight-output-dir",
        default=None,
        help="Directory for optional live-preflight readiness outputs.",
    )
    daily_pipeline_parser.add_argument(
        "--live-shadow-review-decisions-file",
        default=None,
        help="Optional live_shadow_review_decisions.csv generated after manual live-shadow review.",
    )
    daily_pipeline_parser.add_argument(
        "--live-shadow-holdings-file",
        default=None,
        help="Optional live holdings CSV. When set with --live-shadow-cash, daily-pipeline writes a manual-review shadow plan.",
    )
    daily_pipeline_parser.add_argument(
        "--live-shadow-prices-file",
        default=None,
        help="Optional live prices CSV for the daily live-shadow step.",
    )
    daily_pipeline_parser.add_argument(
        "--live-shadow-cash",
        type=float,
        default=None,
        help="Optional current cash for the daily live-shadow step.",
    )
    daily_pipeline_parser.add_argument("--live-shadow-lot-size", type=int, default=100, help="A-share buy lot size for live-shadow.")
    daily_pipeline_parser.add_argument(
        "--live-shadow-min-trade-value",
        type=float,
        default=1000.0,
        help="Suppress live-shadow trades below this amount.",
    )
    daily_pipeline_parser.add_argument(
        "--live-shadow-max-position-weight",
        type=float,
        default=0.20,
        help="Maximum per-stock target weight for live-shadow.",
    )
    daily_pipeline_parser.add_argument(
        "--live-shadow-max-gross-exposure",
        type=float,
        default=1.0,
        help="Maximum total target stock exposure for live-shadow.",
    )
    daily_pipeline_parser.add_argument(
        "--trigger-report",
        default="D:/codex/outputs/trigger_reports/latest_trigger.md",
        help="Path to the latest trigger monitor Markdown report.",
    )
    daily_pipeline_parser.add_argument(
        "--model-audit-dir",
        default=None,
        help="Directory containing model_build_audit_snapshot.json and walk_forward_run_actions.csv.",
    )
    daily_pipeline_parser.add_argument("--max-staleness-days", type=int, default=3, help="Maximum allowed staleness in days.")
    daily_pipeline_parser.add_argument("--min-cache-fresh-ratio", type=float, default=0.90, help="Minimum share of cached CSVs at the latest date.")
    daily_pipeline_parser.add_argument("--rebalance-cost-rate", type=float, default=0.0, help="Optional paper-account allocation-level cost rate.")
    daily_pipeline_parser.add_argument("--stock-market-cap-path", default=None, help="Normalized A-share market-cap cache CSV for stock tracking rules.")
    daily_pipeline_parser.add_argument(
        "--stock-tracking-max-market-cap-yi",
        type=float,
        default=1500.0,
        help="Skip stock-target tracking when total market cap is above this Yi Yuan threshold.",
    )
    daily_pipeline_parser.add_argument("--stock-review-notes-path", default=None, help="Persistent manual notes CSV for stock-target review rows.")
    daily_pipeline_parser.add_argument("--stock-review-outcomes-history-path", default=None, help="Persistent cumulative outcome-history CSV for stock-target review rows.")
    daily_pipeline_parser.add_argument("--stock-review-drawdown-threshold", type=float, default=-0.10, help="Unrealized-return threshold for stock target drawdown_review rows.")
    daily_pipeline_parser.add_argument("--stock-review-watch-drawdown-threshold", type=float, default=-0.07, help="Unrealized-return threshold for stock target watch-level loss rows.")
    daily_pipeline_parser.add_argument("--stock-review-loss-attention-threshold", type=float, default=-0.05, help="Unrealized-return threshold for lower-priority stock target loss attention.")
    daily_pipeline_parser.add_argument("--stock-review-gain-attention-threshold", type=float, default=0.10, help="Unrealized-return threshold for stock target trailing-risk gain attention.")
    daily_pipeline_parser.add_argument("--stock-review-watch-score-threshold", type=float, default=30.0, help="Priority score threshold for stock target watch_review rows.")
    daily_pipeline_parser.add_argument("--stock-review-outcome-min-evaluable", type=int, default=20, help="Minimum evaluable rows before stock-target outcome analysis is review-ready.")
    daily_pipeline_parser.add_argument("--stock-review-outcome-min-group-evaluable", type=int, default=5, help="Minimum evaluable rows per group before stock-target outcome analysis is review-ready.")
    daily_pipeline_parser.add_argument(
        "--stock-review-warning-only-after-close",
        action="store_true",
        help="Downgrade stock-target review warning alerts to info unless the A-share after-close gate is ready.",
    )
    daily_pipeline_parser.add_argument(
        "--chip-reversal-candidate-outcomes",
        action="store_true",
        help="Run chip-reversal candidate outcome backfill as part of the daily research pipeline.",
    )
    daily_pipeline_parser.add_argument(
        "--chip-reversal-candidates-path",
        default="outputs/research/chip_reversal_daily_candidates_latest/chip_reversal_daily_candidates.csv",
        help="Input chip_reversal_daily_candidates.csv path for daily pipeline outcome backfill.",
    )
    daily_pipeline_parser.add_argument(
        "--chip-reversal-outcomes-data-dir",
        default="data/processed/stocks",
        help="Directory containing locally cached stock daily CSV files for daily pipeline outcome backfill.",
    )
    daily_pipeline_parser.add_argument(
        "--chip-reversal-outcomes-output-dir",
        default=None,
        help="Output directory for daily pipeline chip-reversal candidate outcomes.",
    )
    daily_pipeline_parser.add_argument(
        "--chip-reversal-outcome-horizons",
        default="1,2",
        help="Comma-separated chip-reversal outcome horizons for daily pipeline.",
    )
    daily_pipeline_parser.add_argument(
        "--chip-reversal-outcome-success-threshold",
        type=float,
        default=0.03,
        help="Close-to-close return threshold considered successful for chip-reversal outcomes.",
    )
    daily_pipeline_parser.add_argument(
        "--chip-reversal-outcome-min-ready-per-horizon",
        type=int,
        default=30,
        help="Minimum ready chip-reversal outcomes required per horizon.",
    )
    daily_pipeline_parser.add_argument(
        "--chip-reversal-outcome-min-success-rate",
        type=float,
        default=0.55,
        help="Minimum chip-reversal success rate required per horizon.",
    )
    daily_pipeline_parser.add_argument(
        "--skip-history-review",
        dest="run_history_review",
        action="store_false",
        default=True,
        help="Do not run the automatic pipeline-history review after appending history.",
    )
    daily_pipeline_parser.add_argument("--history-lookback-runs", type=int, default=20, help="Recent history rows to review automatically.")
    daily_pipeline_parser.add_argument(
        "--history-drawdown-watch-threshold",
        type=float,
        default=-0.08,
        help="Paper current drawdown threshold for the automatic history review.",
    )
    daily_pipeline_parser.add_argument(
        "--history-min-sharpe-watch",
        type=float,
        default=0.5,
        help="Paper Sharpe threshold for the automatic history review.",
    )
    daily_pipeline_parser.add_argument(
        "--fail-on-alert-level",
        choices=["none", "info", "warning", "critical"],
        default="none",
        help="Return exit code 4 when daily alert level reaches this threshold.",
    )
    daily_pipeline_parser.add_argument(
        "--fail-on-history-health",
        choices=["none", "watch", "blocked"],
        default="none",
        help="Return exit code 5 when automatic history health reaches this threshold.",
    )
    daily_pipeline_parser.add_argument("--as-of-date", default=None, help="Override the check date, YYYY-MM-DD or YYYYMMDD.")
    daily_pipeline_parser.add_argument(
        "--date-stamp",
        dest="date_stamp",
        action="store_true",
        default=True,
        help="Write dated output directories. Enabled by default.",
    )
    daily_pipeline_parser.add_argument(
        "--no-date-stamp",
        dest="date_stamp",
        action="store_false",
        help="Write to latest-style output directories instead of dated directories.",
    )

    daily_run_status_parser = subparsers.add_parser(
        "daily-run-status",
        help="Summarize the latest scheduled refresh, allocator observation, and cache status without refreshing data.",
    )
    daily_run_status_parser.add_argument(
        "--output-dir",
        default="outputs/research/daily_run_status_latest",
        help="Directory for the daily run status summary.",
    )
    daily_run_status_parser.add_argument("--logs-dir", default="outputs/logs", help="Directory containing run status JSON logs.")
    daily_run_status_parser.add_argument(
        "--research-dir",
        default="outputs/research",
        help="Directory containing daily pipeline and allocator observation outputs.",
    )
    daily_run_status_parser.add_argument(
        "--data-cache-dir",
        default=DEFAULT_DAILY_MARKET_DATA_DIR_ARG,
        help="Unified daily market data hub. Falls back to the exchange ingest project when latest data is missing.",
    )
    daily_run_status_parser.add_argument(
        "--scheduled-run-time",
        default="16:10",
        help="Expected daily scheduled refresh time, HH:MM, used to label pre-run waiting status.",
    )
    daily_run_status_parser.add_argument(
        "--fail-on-problem-state",
        action="store_true",
        help="Return a non-zero exit code when the read-only status summary is in a problem state.",
    )
    daily_run_status_parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full status snapshot payload as JSON after the human-readable summary.",
    )
    daily_run_status_parser.add_argument(
        "--json-path",
        default=None,
        help="Optional path to write the status snapshot JSON payload for monitoring systems.",
    )

    pipeline_history_parser = subparsers.add_parser(
        "pipeline-history",
        help="Review daily-pipeline history and write local health/drift alerts.",
    )
    pipeline_history_parser.add_argument(
        "--history-file",
        default="outputs/research/daily_pipeline_history.csv",
        help="CSV history file written by daily-pipeline.",
    )
    pipeline_history_parser.add_argument(
        "--output-dir",
        default="outputs/research/pipeline_history_review_latest",
        help="Directory for the history review report.",
    )
    pipeline_history_parser.add_argument("--as-of-date", default=None, help="Override the review date, YYYY-MM-DD or YYYYMMDD.")
    pipeline_history_parser.add_argument("--lookback-runs", type=int, default=20, help="Number of recent history rows to review.")
    pipeline_history_parser.add_argument("--max-staleness-days", type=int, default=3, help="Maximum allowed days since latest history row.")
    pipeline_history_parser.add_argument(
        "--drawdown-watch-threshold",
        type=float,
        default=-0.08,
        help="Paper current drawdown threshold that raises a watch alert.",
    )
    pipeline_history_parser.add_argument(
        "--min-sharpe-watch",
        type=float,
        default=0.5,
        help="Paper Sharpe threshold below which a watch alert is raised.",
    )
    pipeline_history_parser.add_argument(
        "--fail-on-health",
        choices=["none", "watch", "blocked"],
        default="none",
        help="Return exit code 5 when pipeline-history health reaches this threshold.",
    )

    model_audit_parser = subparsers.add_parser(
        "model-audit",
        help="Audit model-build speed controls, redundancy, and walk-forward completeness.",
    )
    model_audit_parser.add_argument("--config-dir", default="configs", help="Config directory to scan.")
    model_audit_parser.add_argument(
        "--walk-forward-dir",
        default="outputs/walk_forward",
        help="Walk-forward output directory to scan.",
    )
    model_audit_parser.add_argument(
        "--walk-forward-resolution-path",
        default="outputs/research/walk_forward_run_resolutions.csv",
        help="CSV registry for resolved or superseded walk-forward directories.",
    )
    model_audit_parser.add_argument("--lock-dir", default="outputs/locks", help="Process-lock directory to inspect.")
    model_audit_parser.add_argument(
        "--output-dir",
        default="outputs/research/model_build_audit_latest",
        help="Audit output directory.",
    )

    promotion_parser = subparsers.add_parser(
        "allocator-promotion-review",
        help="Review whether an allocator candidate is ready to replace the baseline.",
    )
    promotion_parser.add_argument("--baseline-dir", required=True, help="Baseline portfolio walk-forward output directory.")
    promotion_parser.add_argument("--candidate-dir", required=True, help="Candidate allocator output directory.")
    promotion_parser.add_argument(
        "--sensitivity-dir",
        action="append",
        default=[],
        help="Additional sensitivity allocator output directory. Repeat for multiple runs.",
    )
    promotion_parser.add_argument(
        "--sensitivity-group",
        action="append",
        default=[],
        help="Optional group label for each sensitivity directory. Repeat in the same order as --sensitivity-dir.",
    )
    promotion_parser.add_argument(
        "--evidence-snapshot",
        action="append",
        default=[],
        help="External evidence snapshot JSON. Repeat for independent evidence checks such as execution-cost stress.",
    )
    promotion_parser.add_argument(
        "--evidence-group",
        action="append",
        default=[],
        help="Optional group label for each evidence snapshot. Repeat in the same order as --evidence-snapshot.",
    )
    promotion_parser.add_argument(
        "--output-dir",
        default="outputs/research/allocator_promotion_review_latest",
        help="Promotion review output directory.",
    )
    promotion_parser.add_argument("--min-return-edge", type=float, default=0.03, help="Minimum candidate total-return edge over baseline.")
    promotion_parser.add_argument("--min-sharpe-edge", type=float, default=0.05, help="Minimum candidate Sharpe edge over baseline.")
    promotion_parser.add_argument("--max-drawdown-worsening", type=float, default=0.0, help="Maximum allowed max-drawdown worsening versus baseline.")
    promotion_parser.add_argument("--min-positive-window-ratio", type=float, default=0.80, help="Minimum candidate positive OOS window ratio.")
    promotion_parser.add_argument("--min-sensitivity-support", type=int, default=1, help="Minimum sensitivity runs that must also beat the baseline.")

    source_decision_parser = subparsers.add_parser(
        "source-decision-review",
        help="Explain source-selection decisions window by window.",
    )
    source_decision_parser.add_argument("--source-dir", required=True, help="Portfolio source-selection output directory.")
    source_decision_parser.add_argument("--baseline-dir", default=None, help="Optional baseline portfolio walk-forward output directory.")
    source_decision_parser.add_argument(
        "--output-dir",
        default="outputs/research/source_decision_review_latest",
        help="Source decision review output directory.",
    )

    source_guard_parser = subparsers.add_parser(
        "source-guard-decomposition",
        help="Compare default and guarded source-selection runs window by window.",
    )
    source_guard_parser.add_argument("--default-dir", required=True, help="Default source-selection output directory.")
    source_guard_parser.add_argument("--candidate-dir", required=True, help="Guarded or score-mode source-selection output directory.")
    source_guard_parser.add_argument(
        "--output-dir",
        default="outputs/research/source_guard_decomposition_latest",
        help="Source guard decomposition output directory.",
    )

    source_risk_budget_parser = subparsers.add_parser(
        "source-risk-budget-review",
        help="Review fragile source edges against satellite risk-budget diagnostics.",
    )
    source_risk_budget_parser.add_argument("--source-dir", required=True, help="Portfolio source-selection output directory.")
    source_risk_budget_parser.add_argument(
        "--guard-decomposition-dir",
        default=None,
        help="Optional source-guard-decomposition output directory for guard-vs-default deltas.",
    )
    source_risk_budget_parser.add_argument(
        "--output-dir",
        default="outputs/research/source_risk_budget_latest",
        help="Source risk-budget review output directory.",
    )
    source_risk_budget_parser.add_argument("--fragile-source-edge", type=float, default=0.03, help="Source score edge at or below this value is flagged as fragile.")
    source_risk_budget_parser.add_argument(
        "--max-low-risk-satellite-weight",
        type=float,
        default=0.05,
        help="Average satellite weight at or below this value counts as already low risk.",
    )
    source_risk_budget_parser.add_argument(
        "--min-filter-off-ratio",
        type=float,
        default=0.50,
        help="Satellite filter-off ratio at or above this value counts as already low risk.",
    )

    window_attr_parser = subparsers.add_parser(
        "portfolio-window-attribution",
        help="Replay selected portfolio params for one window and write day-level attribution.",
    )
    window_attr_parser.add_argument("--config", required=True, help="Path to portfolio source-selection YAML config.")
    window_attr_parser.add_argument("--start", required=True, help="Window start date, e.g. 20240704.")
    window_attr_parser.add_argument("--end", required=True, help="Window end date, e.g. 20250103.")
    window_attr_parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Selected params variant in LABEL=PATH format. Can be passed multiple times.",
    )
    window_attr_parser.add_argument(
        "--include-core-only",
        action="store_true",
        help="Include a core-only replay baseline for the same window.",
    )
    window_attr_parser.add_argument(
        "--output-dir",
        default="outputs/research/portfolio_window_attribution_latest",
        help="Portfolio window-attribution output directory.",
    )

    switch_parser = subparsers.add_parser(
        "allocator-switch-readiness",
        help="Compare default and candidate allocator daily-pipeline snapshots before a controlled default switch.",
    )
    switch_parser.add_argument("--default-snapshot", required=True, help="Default allocator daily_pipeline_snapshot.json.")
    switch_parser.add_argument("--candidate-snapshot", required=True, help="Candidate allocator daily_pipeline_snapshot.json.")
    switch_parser.add_argument(
        "--output-dir",
        default="outputs/research/allocator_switch_readiness_latest",
        help="Allocator switch-readiness output directory.",
    )
    switch_parser.add_argument("--default-label", default="Default quality-v2", help="Display label for the default snapshot.")
    switch_parser.add_argument("--candidate-label", default="Candidate allocator", help="Display label for the candidate snapshot.")
    switch_parser.add_argument(
        "--outcome-analysis-path",
        default=None,
        help="Optional stock_target_review_outcome_analysis.json to include in the risk-budget readiness section.",
    )

    observation_parser = subparsers.add_parser(
        "allocator-observation",
        help="Summarize post-switch allocator observation gates from a daily-pipeline snapshot.",
    )
    observation_parser.add_argument("--pipeline-snapshot", required=True, help="daily_pipeline_snapshot.json to observe.")
    observation_parser.add_argument(
        "--output-dir",
        default="outputs/research/allocator_observation_latest",
        help="Allocator observation output directory.",
    )
    observation_parser.add_argument(
        "--baseline-label",
        default="Rollback quality-v2",
        help="Label for the retained rollback/baseline allocator.",
    )

    risk_budget_parser = subparsers.add_parser(
        "satellite-risk-budget-review",
        help="Review whether mature stock-target outcomes justify a research-only satellite risk-budget trial.",
    )
    risk_budget_parser.add_argument("--pipeline-snapshot", required=True, help="daily_pipeline_snapshot.json to review.")
    risk_budget_parser.add_argument(
        "--outcome-analysis-path",
        default=None,
        help="Optional stock_target_review_outcome_analysis.json. Defaults to the path recorded in the pipeline snapshot.",
    )
    risk_budget_parser.add_argument(
        "--network-lab-snapshot",
        default=None,
        help="Optional network_lab_snapshot.json. Use to gate/annotate network-linkage risk.",
    )
    risk_budget_parser.add_argument(
        "--network-max-cluster-count-warning",
        type=int,
        default=1,
        help="Hold trial when network cluster_count is below or equal to this threshold.",
    )
    risk_budget_parser.add_argument(
        "--network-residual-mi-warning",
        type=float,
        default=0.20,
        help="Hold trial when network top residual MI is above this threshold.",
    )
    risk_budget_parser.add_argument(
        "--output-dir",
        default="outputs/research/satellite_risk_budget_review_latest",
        help="Satellite risk-budget review output directory.",
    )
    risk_budget_parser.add_argument("--trial-satellite-budget", type=float, default=0.05, help="Initial research-only satellite budget cap.")
    risk_budget_parser.add_argument("--max-satellite-budget", type=float, default=0.20, help="Maximum satellite budget allowed by this review.")
    risk_budget_parser.add_argument("--min-overall-win-rate", type=float, default=0.55, help="Minimum overall win rate required for a trial.")
    risk_budget_parser.add_argument("--min-overall-avg-return", type=float, default=0.0, help="Minimum overall average return required for a trial.")
    risk_budget_parser.add_argument("--max-worst-group-loss", type=float, default=-0.05, help="Worst-group average-return loss guard.")

    live_preflight_parser = subparsers.add_parser(
        "live-preflight",
        help="Run a read-only pre-live readiness checklist from dashboard and paper-account outputs.",
    )
    live_preflight_parser.add_argument(
        "--dashboard-snapshot",
        default="outputs/research/latest_dashboard/latest_dashboard_snapshot.json",
        help="latest_dashboard_snapshot.json to use as the readiness source.",
    )
    live_preflight_parser.add_argument(
        "--paper-account-dir",
        default=None,
        help="Optional paper-account output directory. Defaults to the dashboard-recorded directory.",
    )
    live_preflight_parser.add_argument(
        "--pipeline-snapshot",
        default=None,
        help="Optional daily_pipeline_snapshot.json used to fill pipeline-only readiness gates.",
    )
    live_preflight_parser.add_argument(
        "--daily-run-status-snapshot",
        default=None,
        help="Optional daily_run_status_snapshot.json problem-state gate.",
    )
    live_preflight_parser.add_argument(
        "--live-shadow-review-decisions-file",
        default=None,
        help="Optional live_shadow_review_decisions.csv generated after manual live-shadow review.",
    )
    live_preflight_parser.add_argument(
        "--output-dir",
        default="outputs/research/live_preflight_latest",
        help="Live preflight output directory.",
    )
    live_preflight_parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Return a non-zero exit code when the preflight decision is blocked.",
    )

    walk_parser = subparsers.add_parser(
        "walk-forward",
        aliases=["rolling-train"],
        help="Run rolling training and out-of-sample walk-forward tests.",
    )
    _add_config_arg(walk_parser)
    walk_parser.add_argument("--preset", choices=sorted(WALK_FORWARD_PRESETS), default=None, help="Named walk-forward preset.")
    walk_parser.add_argument("--train-years", type=int, default=None, help="Training window length in years.")
    walk_parser.add_argument("--test-months", type=int, default=None, help="Out-of-sample window length in months.")
    walk_parser.add_argument("--step-months", type=int, default=None, help="Rolling step length in months.")
    walk_parser.add_argument(
        "--grid",
        choices=[
            "compact",
            "bear",
            "opportunity",
            "stable",
            "stable_v2",
            "reversal",
            "satellite",
            "satellite_v2",
            "risk",
        ],
        default=None,
        help="Candidate parameter grid.",
    )
    walk_parser.add_argument(
        "--objective",
        choices=["balanced", "robust", "stable", "satellite", "capital", "opportunity_quality"],
        default=None,
        help="Training objective used to select parameters.",
    )
    walk_parser.add_argument(
        "--max-train-drawdown",
        type=float,
        default=None,
        help="Drawdown limit used by the training score penalty.",
    )
    walk_parser.add_argument(
        "--selection-validation-months",
        type=int,
        default=None,
        help="Hold out the latest N training months and select candidates on that internal validation segment.",
    )
    walk_parser.add_argument("--output-dir", default="outputs/walk_forward", help="Walk-forward output directory.")
    walk_parser.add_argument("--run-id-prefix", default=None, help="Optional walk-forward directory name.")
    walk_parser.add_argument("--allow-fetch", action="store_true", help="Fetch missing data instead of requiring cache.")
    walk_parser.add_argument("--skip-missing", action="store_true", help="Skip symbols whose history is missing from the cache.")
    walk_parser.add_argument("--resume", action="store_true", help="Resume from completed walk-forward windows in the output directory.")
    walk_parser.add_argument("--window-limit", type=int, default=None, help="Run only the first N walk-forward windows for smoke tests.")

    portfolio_parser = subparsers.add_parser("portfolio", help="Portfolio allocation commands.")
    portfolio_subparsers = portfolio_parser.add_subparsers(dest="portfolio_command", required=True)
    combine_parser = portfolio_subparsers.add_parser("combine", help="Combine existing equity curves into a portfolio.")
    combine_parser.add_argument("--config", default=str(DEFAULT_PORTFOLIO_CONFIG), help="Path to portfolio YAML config.")
    combine_parser.add_argument("--run-id", default=None, help="Optional portfolio run id.")
    portfolio_walk_parser = portfolio_subparsers.add_parser(
        "walk-forward",
        help="Run rolling training for portfolio allocation parameters.",
    )
    portfolio_walk_parser.add_argument("--config", default=str(DEFAULT_PORTFOLIO_CONFIG), help="Path to portfolio YAML config.")
    portfolio_walk_parser.add_argument("--train-months", type=int, default=24, help="Training window length in months.")
    portfolio_walk_parser.add_argument("--test-months", type=int, default=6, help="Out-of-sample window length in months.")
    portfolio_walk_parser.add_argument("--step-months", type=int, default=6, help="Rolling step length in months.")
    portfolio_walk_parser.add_argument(
        "--grid",
        choices=["compact", "guarded", "activation"],
        default="guarded",
        help="Allocation candidate grid.",
    )
    portfolio_walk_parser.add_argument(
        "--max-train-drawdown",
        type=float,
        default=0.16,
        help="Training drawdown limit used by the allocation score.",
    )
    portfolio_walk_parser.add_argument(
        "--score-mode",
        choices=["balanced", "capital_efficiency"],
        default="balanced",
        help="Training score mode for allocator selection.",
    )
    portfolio_walk_parser.add_argument(
        "--score-mode-min-edge",
        type=float,
        default=0.0,
        help="For non-balanced scoring, require this active-score edge over the balanced-score selection before switching.",
    )
    portfolio_walk_parser.add_argument("--output-dir", default="outputs/portfolio_walk_forward", help="Portfolio walk-forward output directory.")
    portfolio_walk_parser.add_argument("--run-id-prefix", default=None, help="Optional portfolio walk-forward directory name.")
    portfolio_source_parser = portfolio_subparsers.add_parser(
        "source-selection",
        help="Run rolling training across multiple satellite sources and allocation parameters.",
    )
    portfolio_source_parser.add_argument("--config", default=str(DEFAULT_PORTFOLIO_CONFIG), help="Path to portfolio source-selection YAML config.")
    portfolio_source_parser.add_argument("--train-months", type=int, default=24, help="Training window length in months.")
    portfolio_source_parser.add_argument("--test-months", type=int, default=6, help="Out-of-sample window length in months.")
    portfolio_source_parser.add_argument("--step-months", type=int, default=6, help="Rolling step length in months.")
    portfolio_source_parser.add_argument(
        "--grid",
        choices=["compact", "guarded", "activation"],
        default="activation",
        help="Allocation candidate grid crossed with every satellite source.",
    )
    portfolio_source_parser.add_argument(
        "--max-train-drawdown",
        type=float,
        default=0.16,
        help="Training drawdown limit used by the allocation score.",
    )
    portfolio_source_parser.add_argument(
        "--score-mode",
        choices=["balanced", "capital_efficiency"],
        default="balanced",
        help="Training score mode for source and allocation selection.",
    )
    portfolio_source_parser.add_argument(
        "--score-mode-min-edge",
        type=float,
        default=0.0,
        help="For non-balanced scoring, require this active-score edge over the balanced-score selection before switching.",
    )
    portfolio_source_parser.add_argument(
        "--default-source",
        default=None,
        help="Optional incumbent satellite source. Other sources must beat it by --source-switch-margin.",
    )
    portfolio_source_parser.add_argument(
        "--source-switch-margin",
        type=float,
        default=0.0,
        help="Minimum training-score edge required before switching away from --default-source.",
    )
    portfolio_source_parser.add_argument(
        "--source-switch-margin-by-source",
        action="append",
        default=[],
        help="Optional source-specific score edge required before selecting a fragile source, e.g. pos10=0.02. Can be repeated.",
    )
    portfolio_source_parser.add_argument(
        "--source-validation-months",
        type=int,
        default=0,
        help="Use the last N months inside each training window as the source-selection validation period.",
    )
    portfolio_source_parser.add_argument(
        "--source-stability-penalty",
        type=float,
        default=0.0,
        help="Score edge required before switching away from the previous window's selected source.",
    )
    portfolio_source_parser.add_argument(
        "--max-satellite-weight",
        type=float,
        default=None,
        help="Optional cap for risk-on satellite weight in the source-selection allocation grid.",
    )
    portfolio_source_parser.add_argument(
        "--source-max-satellite-weight",
        action="append",
        default=[],
        help="Optional source-specific risk-on satellite cap, e.g. reversal=0.20. Can be repeated.",
    )
    portfolio_source_parser.add_argument(
        "--output-dir",
        default="outputs/portfolio_source_selection",
        help="Portfolio source-selection output directory.",
    )
    portfolio_source_parser.add_argument("--run-id-prefix", default=None, help="Optional portfolio source-selection directory name.")

    return parser


def resolve_walk_forward_options(args: argparse.Namespace) -> dict[str, object]:
    options = dict(WALK_FORWARD_DEFAULTS)
    if getattr(args, "preset", None) is not None:
        options.update(WALK_FORWARD_PRESETS[args.preset])
    for key in WALK_FORWARD_DEFAULTS:
        value = getattr(args, key)
        if value is not None:
            options[key] = value
    return options


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "data":
        if args.data_command == "update":
            key = _lock_key("data-update", Path(args.config).resolve(), args.start, args.end)
            try:
                with process_lock(_lock_dir(), key, _command_line()):
                    config = load_config(args.config)
                    written = update_data(
                        config,
                        start_date=args.start,
                        end_date=args.end,
                        skip_existing=args.skip_existing,
                        retry_count=args.retry_count,
                        pause_seconds=args.pause_seconds,
                        continue_on_error=args.continue_on_error,
                    )
            except ProcessLockError as error:
                return _duplicate_process_exit(error)
            print(f"Data files ready: {len(written)}")
            for path in written:
                print(path)
            return 0
        if args.data_command == "update-market-cap":
            key = _lock_key(
                "data-update-market-cap",
                Path(args.output).resolve(),
                Path(args.symbols_file).resolve() if args.symbols_file else "all-symbols",
                args.as_of_date or "latest",
                args.retry_count,
            )
            try:
                with process_lock(_lock_dir(), key, _command_line()):
                    symbols = load_stock_market_cap_symbols(args.symbols_file) if args.symbols_file else None
                    frame = update_stock_market_cap_cache(
                        output_path=Path(args.output),
                        retry_count=args.retry_count,
                        pause_seconds=args.pause_seconds,
                        symbols=symbols,
                        as_of_date=args.as_of_date,
                        lookback_days=args.lookback_days,
                    )
            except ProcessLockError as error:
                return _duplicate_process_exit(error)
            print(f"Stock market-cap cache ready: {Path(args.output)}")
            print(f"Rows: {len(frame)}")
            return 0
        raise ValueError(f"Unsupported data command: {args.data_command}")

    if args.command == "backtest":
        key = _lock_key("backtest", Path(args.config).resolve(), args.run_id or "auto")
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                config = load_config(args.config)
                result = run_backtest(config, run_id=args.run_id, skip_missing=args.skip_missing)
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Backtest completed: {result.run_dir}")
        print(f"Final equity: {result.metrics['final_equity']:.2f}")
        print(f"Total return: {result.metrics['total_return'] * 100:.2f}%")
        print(f"Max drawdown: {result.metrics['max_drawdown'] * 100:.2f}%")
        return 0

    if args.command == "report":
        run_dir = resolve_run_dir(Path(args.output_dir), args.run_id)
        report_path = write_report(run_dir)
        print(f"Report written: {report_path}")
        return 0

    if args.command == "diagnostics":
        run_dir = resolve_run_dir(Path(args.output_dir), args.run_id)
        hac_lags = None if str(args.hac_lags).lower() == "auto" else int(args.hac_lags)
        result = run_diagnostics(
            run_dir,
            equity_column=args.equity_column,
            benchmark_column=args.benchmark_column,
            hac_lags=hac_lags,
        )
        print(f"Diagnostics completed: {result.output_dir}")
        print(f"Report written: {result.report_path}")
        print(f"Beta: {result.metrics['ols_beta']:.3f}")
        print(f"Annualized alpha: {result.metrics['ols_alpha_annual'] * 100:.2f}%")
        print(f"R-squared: {result.metrics['ols_r_squared'] * 100:.2f}%")
        print(f"Information ratio: {result.metrics['information_ratio']:.3f}")
        return 0

    if args.command == "attribution":
        run_dir = resolve_run_dir(Path(args.output_dir), args.run_id)
        result = run_drawdown_attribution(run_dir, min_depth=args.min_depth, top_n=args.top_n)
        print(f"Attribution completed: {result.output_dir}")
        print(f"Report written: {result.report_path}")
        print(f"Drawdown episodes: {result.metrics['drawdown_episode_count']}")
        print(f"Max drawdown: {result.metrics['max_drawdown'] * 100:.2f}%")
        return 0

    if args.command == "diagnostics-batch":
        base = Path(args.output_dir).resolve()
        run_dirs: list[Path] = []
        for run_id in args.run_ids:
            run_dirs.append(resolve_run_dir(base, run_id))
        for pattern in args.run_glob:
            run_dirs.extend(path for path in base.glob(pattern) if path.is_dir())
        unique_run_dirs = list(dict.fromkeys(path.resolve() for path in run_dirs))
        if not unique_run_dirs:
            parser.error("diagnostics-batch requires at least one run id or --run-glob pattern.")
        hac_lags = None if str(args.hac_lags).lower() == "auto" else int(args.hac_lags)
        result = run_batch_diagnostics(
            unique_run_dirs,
            output_dir=Path(args.batch_output_dir),
            equity_column=args.equity_column,
            benchmark_column=args.benchmark_column,
            hac_lags=hac_lags,
        )
        print(f"Batch diagnostics completed: {result.output_dir}")
        print(f"Comparison CSV: {result.output_dir / 'diagnostic_compare.csv'}")
        print(f"Report written: {result.report_path}")
        print(f"Successful runs: {len(result.comparison)}")
        print(f"Failed runs: {len(result.failures)}")
        return 0

    if args.command == "momentum-focus":
        key = _lock_key(
            "momentum-focus",
            Path(args.output_dir).resolve(),
            Path(args.data_cache_dir).resolve(),
            Path(args.ingest_project_dir).resolve(),
            args.as_of_date or "today",
            args.trade_date or "as-of-date",
            args.strong_gain_threshold_pct,
            args.board_scope,
            args.outcome_summary_path,
            args.name_map_path,
            args.target_horizon,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_momentum_focus(
                    project_root=Path("."),
                    output_dir=Path(args.output_dir),
                    daily_data_dir=Path(args.data_cache_dir),
                    ingest_project_dir=Path(args.ingest_project_dir),
                    as_of_date=args.as_of_date,
                    trade_date=args.trade_date,
                    strong_gain_threshold_pct=args.strong_gain_threshold_pct,
                    board_scope=args.board_scope,
                    outcome_summary_path=args.outcome_summary_path,
                    name_map_path=args.name_map_path,
                    target_horizon=args.target_horizon,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Momentum focus completed: {result.output_dir}")
        print(f"Status: {result.snapshot.get('status')}")
        print(f"Trade date: {result.snapshot.get('trade_date')}")
        print(f"Source kind: {result.snapshot.get('source_kind')}")
        print(f"Candidate rows: {result.snapshot.get('candidate_count')}")
        print(f"Limit-up rows: {result.snapshot.get('limit_up_count')}")
        print(f">=7% rows: {result.snapshot.get('strong_gain_count')}")
        print(f"Candidates CSV: {result.candidates_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Broker action: {result.snapshot.get('broker_action')}")
        return 0

    if args.command == "momentum-outcomes":
        horizons = _parse_int_csv(args.horizons)
        key = _lock_key(
            "momentum-outcomes",
            Path(args.output_dir).resolve(),
            Path(args.data_dir).resolve(),
            args.horizons,
            args.strong_gain_threshold_pct,
            args.board_scope,
            args.min_events,
            args.max_symbols or "all-symbols",
            args.start_date or "no-start",
            args.end_date or "no-end",
            args.recursive,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_momentum_outcome_analysis(
                    project_root=Path("."),
                    data_dir=Path(args.data_dir),
                    output_dir=Path(args.output_dir),
                    horizons=horizons,
                    strong_gain_threshold_pct=args.strong_gain_threshold_pct,
                    board_scope=args.board_scope,
                    min_events=args.min_events,
                    max_symbols=args.max_symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    recursive=args.recursive,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Momentum outcomes completed: {result.output_dir}")
        print(f"Status: {result.snapshot.get('status')}")
        print(f"Event rows: {result.snapshot.get('event_count')}")
        print(f"Summary rows: {result.snapshot.get('summary_row_count')}")
        print(f"Loaded symbols: {result.snapshot.get('loaded_symbol_count')} / {result.snapshot.get('file_count')}")
        print(f"Events CSV: {result.events_path}")
        print(f"Summary CSV: {result.summary_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Broker action: {result.snapshot.get('broker_action')}")
        return 0

    if args.command == "chip-reversal-lab":
        horizons = _parse_int_csv(args.horizons)
        allowed_theme_states = _parse_str_csv(args.allowed_theme_states) or ["healthy", "second_activation"]
        key = _lock_key(
            "chip-reversal-lab",
            Path(args.output_dir).resolve(),
            Path(args.data_dir).resolve(),
            args.horizons,
            args.drawdown_window,
            args.cost_window,
            args.min_drawdown_pct,
            args.min_score,
            args.score_bucket,
            args.min_amount_yi,
            args.max_next_open_gap_pct if args.max_next_open_gap_pct is not None else "no-gap-filter",
            Path(args.theme_state_path).resolve() if args.theme_state_path else "no-theme-state-gate",
            Path(args.theme_symbol_panel_path).resolve() if args.theme_symbol_panel_path else "no-theme-symbol-panel",
            tuple(allowed_theme_states),
            args.board_scope,
            args.min_events,
            args.max_symbols or "all-symbols",
            args.start_date or "no-start",
            args.end_date or "no-end",
            args.recursive,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_chip_reversal_lab(
                    project_root=Path("."),
                    data_dir=Path(args.data_dir),
                    output_dir=Path(args.output_dir),
                    horizons=horizons,
                    drawdown_window=args.drawdown_window,
                    cost_window=args.cost_window,
                    min_drawdown_pct=args.min_drawdown_pct,
                    min_score=args.min_score,
                    score_bucket=args.score_bucket,
                    min_amount_yi=args.min_amount_yi,
                    max_next_open_gap_pct=args.max_next_open_gap_pct,
                    theme_state_path=Path(args.theme_state_path) if args.theme_state_path else None,
                    theme_symbol_panel_path=Path(args.theme_symbol_panel_path) if args.theme_symbol_panel_path else None,
                    allowed_theme_states=allowed_theme_states,
                    board_scope=args.board_scope,
                    min_events=args.min_events,
                    max_symbols=args.max_symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    recursive=args.recursive,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Chip reversal lab completed: {result.output_dir}")
        print(f"Status: {result.snapshot.get('status')}")
        print(f"Event rows: {result.snapshot.get('event_count')}")
        print(f"Summary rows: {result.snapshot.get('summary_row_count')}")
        print(f"Loaded symbols: {result.snapshot.get('loaded_symbol_count')} / {result.snapshot.get('file_count')}")
        print(f"Events CSV: {result.events_path}")
        print(f"Summary CSV: {result.summary_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Auction confirmation: {result.snapshot.get('auction_confirmation_status')}")
        print(f"Broker action: {result.snapshot.get('broker_action')}")
        return 0

    if args.command == "chip-reversal-pit-audit":
        horizons = _parse_int_csv(args.horizons)
        key = _lock_key(
            "chip-reversal-pit-audit",
            Path(args.output_dir).resolve(),
            Path(args.data_dir).resolve(),
            args.horizons,
            args.drawdown_window,
            args.cost_window,
            args.min_drawdown_pct,
            args.min_score,
            args.score_bucket,
            args.board_scope,
            args.max_symbols or "all-symbols",
            args.start_date or "no-start",
            args.end_date or "no-end",
            args.max_events,
            args.recursive,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_chip_reversal_pit_audit(
                    project_root=Path("."),
                    data_dir=Path(args.data_dir),
                    output_dir=Path(args.output_dir),
                    horizons=horizons,
                    drawdown_window=args.drawdown_window,
                    cost_window=args.cost_window,
                    min_drawdown_pct=args.min_drawdown_pct,
                    min_score=args.min_score,
                    score_bucket=args.score_bucket,
                    board_scope=args.board_scope,
                    max_symbols=args.max_symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    max_events=args.max_events,
                    recursive=args.recursive,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Chip reversal PIT audit completed: {result.output_dir}")
        print(f"Status: {result.snapshot.get('status')}")
        print(f"Audited events: {result.snapshot.get('audited_event_count')}")
        print(f"PIT mismatches: {result.snapshot.get('pit_mismatch_count')}")
        print(f"Audit CSV: {result.audit_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Broker action: {result.snapshot.get('broker_action')}")
        return 0

    if args.command == "chip-reversal-observation-queue":
        key = _lock_key(
            "chip-reversal-observation-queue",
            Path(args.output_dir).resolve(),
            Path(args.events_path).resolve(),
            args.as_of_date or "latest",
            args.max_candidates,
            args.min_priority_score,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_chip_reversal_observation_queue(
                    project_root=Path("."),
                    events_path=Path(args.events_path),
                    output_dir=Path(args.output_dir),
                    as_of_date=args.as_of_date,
                    max_candidates=args.max_candidates,
                    min_priority_score=args.min_priority_score,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Chip reversal observation queue completed: {result.output_dir}")
        print(f"Status: {result.snapshot.get('status')}")
        print(f"As-of date: {result.snapshot.get('as_of_date')}")
        print(f"Candidates: {result.snapshot.get('candidate_count')}")
        print(f"Queue CSV: {result.queue_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Broker action: {result.snapshot.get('broker_action')}")
        return 0

    if args.command == "chip-reversal-daily-candidates":
        horizons = _parse_int_csv(args.horizons)
        key = _lock_key(
            "chip-reversal-daily-candidates",
            Path(args.output_dir).resolve(),
            Path(args.data_dir).resolve(),
            args.as_of_date or "latest",
            args.horizons,
            args.drawdown_window,
            args.cost_window,
            args.min_drawdown_pct,
            args.min_score,
            args.score_bucket,
            args.min_amount_yi,
            args.board_scope,
            args.max_candidates,
            args.max_symbols or "all-symbols",
            args.recursive,
            args.market_snapshot_overlay,
            Path(args.daily_data_dir).resolve(),
            Path(args.ingest_project_dir).resolve(),
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_chip_reversal_daily_candidates(
                    project_root=Path("."),
                    data_dir=Path(args.data_dir),
                    output_dir=Path(args.output_dir),
                    as_of_date=args.as_of_date,
                    horizons=horizons,
                    drawdown_window=args.drawdown_window,
                    cost_window=args.cost_window,
                    min_drawdown_pct=args.min_drawdown_pct,
                    min_score=args.min_score,
                    score_bucket=args.score_bucket,
                    min_amount_yi=args.min_amount_yi,
                    board_scope=args.board_scope,
                    max_candidates=args.max_candidates,
                    max_symbols=args.max_symbols,
                    recursive=args.recursive,
                    market_snapshot_overlay=args.market_snapshot_overlay,
                    daily_data_dir=Path(args.daily_data_dir),
                    ingest_project_dir=Path(args.ingest_project_dir),
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Chip reversal daily candidates completed: {result.output_dir}")
        print(f"Status: {result.snapshot.get('status')}")
        print(f"As-of date: {result.snapshot.get('as_of_date')}")
        print(f"Candidates: {result.snapshot.get('candidate_count')}")
        print(f"Market source: {result.snapshot.get('market_source_kind')} {result.snapshot.get('market_trade_date')}")
        print(f"Candidates CSV: {result.candidates_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Broker action: {result.snapshot.get('broker_action')}")
        return 0

    if args.command == "chip-reversal-candidate-outcomes":
        horizons = _parse_int_csv(args.horizons)
        key = _lock_key(
            "chip-reversal-candidate-outcomes",
            Path(args.candidates_path).resolve(),
            Path(args.data_dir).resolve(),
            Path(args.output_dir).resolve(),
            args.horizons,
            args.success_threshold,
            args.min_ready_per_horizon,
            args.min_success_rate,
            args.market_snapshot_overlay,
            Path(args.daily_data_dir).resolve(),
            Path(args.ingest_project_dir).resolve(),
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_chip_reversal_candidate_outcomes(
                    project_root=Path("."),
                    candidates_path=Path(args.candidates_path),
                    data_dir=Path(args.data_dir),
                    output_dir=Path(args.output_dir),
                    horizons=horizons,
                    success_threshold=args.success_threshold,
                    min_ready_per_horizon=args.min_ready_per_horizon,
                    min_success_rate=args.min_success_rate,
                    market_snapshot_overlay=args.market_snapshot_overlay,
                    daily_data_dir=Path(args.daily_data_dir),
                    ingest_project_dir=Path(args.ingest_project_dir),
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Chip reversal candidate outcomes completed: {result.output_dir}")
        print(f"Status: {result.snapshot.get('status')}")
        print(f"Candidates: {result.snapshot.get('candidate_count')}")
        print(f"Ready outcomes: {result.snapshot.get('ready_outcome_count')}")
        print(f"Pending outcomes: {result.snapshot.get('pending_outcome_count')}")
        print(f"Market source: {result.snapshot.get('market_source_kind')} {result.snapshot.get('market_trade_date')}")
        print(f"Promotion gate: {result.snapshot.get('promotion_gate_status')}")
        print(f"Promotion gate reasons: {result.snapshot.get('promotion_gate_reasons')}")
        print(f"Outcomes CSV: {result.outcomes_path}")
        print(f"Summary CSV: {result.summary_path}")
        print(f"Group summary CSV: {result.group_summary_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Broker action: {result.snapshot.get('broker_action')}")
        return 0

    if args.command == "factor-lab":
        horizons = _parse_int_csv(args.horizons)
        factors = _parse_str_csv(args.factors)
        key = _lock_key(
            "factor-lab",
            Path(args.data_dir).resolve(),
            Path(args.output_dir).resolve(),
            args.factors or "default",
            args.horizons,
            args.quantiles,
            args.min_obs,
            args.max_symbols or "all",
            args.start_date or "start",
            args.end_date or "end",
            args.warmup_days,
            args.save_panel,
            args.recursive,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_factor_lab(
                    project_root=Path("."),
                    data_dir=Path(args.data_dir),
                    output_dir=Path(args.output_dir),
                    factors=factors,
                    horizons=horizons,
                    quantiles=args.quantiles,
                    min_obs=args.min_obs,
                    max_symbols=args.max_symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    warmup_days=args.warmup_days,
                    save_panel=args.save_panel,
                    recursive=args.recursive,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Factor lab completed: {result.output_dir}")
        print(f"Report written: {result.report_path}")
        print(f"Loaded symbols: {result.snapshot.get('loaded_symbol_count')}")
        print(f"Panel rows: {result.snapshot.get('panel_row_count')}")
        print(f"Best IC factor: {result.snapshot.get('best_ic_factor')} h={result.snapshot.get('best_ic_horizon')}")
        print(f"Best spread factor: {result.snapshot.get('best_spread_factor')} h={result.snapshot.get('best_spread_horizon')}")
        return 0

    if args.command == "theme-map":
        if args.theme_map_command == "build":
            key = _lock_key(
                "theme-map-build",
                Path(args.output_dir).resolve(),
                Path(args.output).resolve() if args.output else "default-output",
                tuple(str(Path(source).resolve()) for source in args.source or []),
                tuple(str(Path(root).resolve()) for root in args.scan_root or []),
                args.include_default_scan_roots,
                args.include_board_fallback,
                args.max_scan_files,
                args.trade_date or "latest",
                Path(args.data_cache_dir).resolve(),
                Path(args.ingest_project_dir).resolve(),
                args.require_success,
            )
            try:
                with process_lock(_lock_dir(), key, _command_line()):
                    result = run_theme_map_build(
                        project_root=Path("."),
                        output_dir=Path(args.output_dir),
                        output=Path(args.output) if args.output else None,
                        sources=[Path(source) for source in args.source or []],
                        scan_roots=[Path(root) for root in args.scan_root or []],
                        include_default_scan_roots=args.include_default_scan_roots,
                        include_board_fallback=args.include_board_fallback,
                        max_scan_files=args.max_scan_files,
                        trade_date=args.trade_date,
                        daily_data_dir=Path(args.data_cache_dir),
                        ingest_project_dir=Path(args.ingest_project_dir),
                        require_success=args.require_success,
                    )
            except ProcessLockError as error:
                return _duplicate_process_exit(error)
            print(f"Theme map build completed: {result.output_dir}")
            print(f"Theme map written: {result.theme_map_path}")
            print(f"Report written: {result.report_path}")
            print(f"Trade date: {result.snapshot.get('trade_date')}")
            print(f"Market source: {result.snapshot.get('market_source_kind')}")
            print(f"Theme map rows: {result.snapshot.get('theme_map_rows')}")
            print(f"Mapped codes: {result.snapshot.get('mapped_code_count')}")
            print(f"Groups: {result.snapshot.get('group_count')}")
            print(f"Board fallback rows: {result.snapshot.get('board_fallback_rows')}")
            return 0

    if args.command == "market-timing-state":
        key = _lock_key(
            "market-timing-state",
            Path(args.data_dir).resolve(),
            Path(args.output_dir).resolve(),
            Path(args.data_cache_dir).resolve(),
            Path(args.ingest_project_dir).resolve(),
            args.trade_date or "latest",
            args.lookback_days,
            args.max_symbols or "all",
            args.recursive,
            args.require_success,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_market_timing_state(
                    project_root=Path("."),
                    data_dir=Path(args.data_dir),
                    output_dir=Path(args.output_dir),
                    daily_data_dir=Path(args.data_cache_dir),
                    ingest_project_dir=Path(args.ingest_project_dir),
                    trade_date=args.trade_date,
                    lookback_days=args.lookback_days,
                    max_symbols=args.max_symbols,
                    recursive=args.recursive,
                    require_success=args.require_success,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Market timing state completed: {result.output_dir}")
        print(f"Report written: {result.report_path}")
        print(f"Trade date: {result.snapshot.get('trade_date')}")
        print(f"Market regime: {result.snapshot.get('market_regime')}")
        print(f"Position reference: {result.snapshot.get('position_reference')}")
        print(f"Loaded symbols: {result.snapshot.get('loaded_symbol_count')} / {result.snapshot.get('file_count')}")
        print(f"Market source: {result.snapshot.get('market_source_kind')}")
        print(f"Broker action: {result.snapshot.get('broker_action')}")
        print(f"Allocator action: {result.snapshot.get('allocator_action')}")
        return 0

    if args.command == "theme-state":
        horizons = _parse_int_csv(args.horizons)
        key = _lock_key(
            "theme-state",
            Path(args.data_dir).resolve(),
            Path(args.output_dir).resolve(),
            Path(args.theme_map).resolve() if args.theme_map else "no-theme-map",
            args.group_column or "auto-group",
            args.horizons,
            args.max_symbols or "all",
            args.start_date or "start",
            args.end_date or "end",
            args.warmup_days,
            args.recursive,
            args.breadth_window,
            args.slope_window,
            args.quiet_lookback,
            args.activation_breadth,
            args.prior_breadth_ceiling,
            args.min_breadth_slope,
            args.min_rs_20d,
            args.beta_window,
            args.theme_active_only,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_theme_state(
                    project_root=Path("."),
                    data_dir=Path(args.data_dir),
                    output_dir=Path(args.output_dir),
                    theme_map_path=Path(args.theme_map) if args.theme_map else None,
                    group_column=args.group_column,
                    horizons=horizons,
                    max_symbols=args.max_symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    warmup_days=args.warmup_days,
                    recursive=args.recursive,
                    breadth_window=args.breadth_window,
                    slope_window=args.slope_window,
                    quiet_lookback=args.quiet_lookback,
                    activation_breadth=args.activation_breadth,
                    prior_breadth_ceiling=args.prior_breadth_ceiling,
                    min_breadth_slope=args.min_breadth_slope,
                    min_rs_20d=args.min_rs_20d,
                    beta_window=args.beta_window,
                    theme_active_only=args.theme_active_only,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Theme state completed: {result.output_dir}")
        print(f"Report written: {result.report_path}")
        print(f"Loaded symbols: {result.snapshot.get('loaded_symbol_count')}")
        print(f"Groups: {result.snapshot.get('group_count')}")
        print(f"Latest date: {result.snapshot.get('latest_date')}")
        print(f"Latest second-activation groups: {result.snapshot.get('latest_second_activation_count')}")
        print(f"Beta-gap rows: {result.snapshot.get('beta_gap_row_count')}")
        return 0

    if args.command == "network-lab":
        symbols = _parse_str_csv(args.symbols)
        key = _lock_key(
            "network-lab",
            Path(args.data_dir).resolve(),
            Path(args.output_dir).resolve(),
            args.symbols or "all-symbols",
            Path(args.symbols_file).resolve() if args.symbols_file else "no-symbols-file",
            args.max_symbols or "all",
            args.start_date or "start",
            args.end_date or "end",
            args.lookback_days,
            args.top_edges,
            args.bins,
            args.min_obs,
            args.recursive,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_network_lab(
                    project_root=Path("."),
                    data_dir=Path(args.data_dir),
                    output_dir=Path(args.output_dir),
                    symbols=symbols,
                    symbols_file=Path(args.symbols_file) if args.symbols_file else None,
                    max_symbols=args.max_symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    lookback_days=args.lookback_days,
                    top_edges=args.top_edges,
                    bins=args.bins,
                    min_obs=args.min_obs,
                    recursive=args.recursive,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Network lab completed: {result.output_dir}")
        print(f"Report written: {result.report_path}")
        print(f"Loaded symbols: {result.snapshot.get('loaded_symbol_count')}")
        print(f"Edges: {result.snapshot.get('edge_count')}")
        print(f"MST edges: {result.snapshot.get('mst_edge_count')}")
        print(f"Top residual MI: {result.snapshot.get('top_residual_mutual_information')}")
        return 0

    if args.command == "validate":
        key = _lock_key("validate", Path(args.config).resolve(), Path(args.output_dir).resolve(), args.run_id_prefix or "auto")
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                config = load_config(args.config)
                validation_dir, summary = run_validation(
                    config,
                    output_dir=Path(args.output_dir),
                    run_id_prefix=args.run_id_prefix,
                    allow_fetch=args.allow_fetch,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Validation completed: {validation_dir}")
        print(f"Summary CSV: {summary.attrs.get('summary_csv')}")
        print(f"Summary report: {summary.attrs.get('summary_md')}")
        for row in summary.itertuples(index=False):
            print(
                f"{row.split}: total_return={row.total_return * 100:.2f}%, "
                f"max_drawdown={row.max_drawdown * 100:.2f}%, sharpe={row.sharpe:.3f}"
            )
        return 0

    if args.command == "sentiment":
        key = _lock_key("sentiment", Path(args.config).resolve(), Path(args.output_dir).resolve(), args.max_symbols, args.window)
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_market_sentiment_reference(
                    Path(args.config),
                    Path(args.output_dir),
                    max_symbols=args.max_symbols,
                    skip_missing=not args.strict,
                    window=args.window,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Market sentiment completed: {result.output_dir}")
        print(f"Latest date: {result.latest['date']}")
        print(f"State: {result.latest['sentiment_state']}")
        print(f"Score: {float(result.latest['sentiment_score']):.3f}")
        print(f"Report written: {result.report_path}")
        return 0

    if args.command == "phase2":
        result = run_phase2_status(
            project_root=Path("."),
            output_dir=Path(args.output_dir),
            baseline_run_dir=args.baseline_run_dir,
            core_wf_dir=args.core_wf_dir,
            satellite_wf_dir=args.satellite_wf_dir,
            portfolio_run_dir=args.portfolio_run_dir,
            portfolio_wf_dir=args.portfolio_wf_dir,
        )
        print(f"Phase 2 status completed: {result.output_dir}")
        print(f"Components CSV: {result.components_path}")
        print(f"Report written: {result.report_path}")
        complete_count = int((result.components["status"] == "complete").sum())
        print(f"Complete components: {complete_count}/{len(result.components)}")
        return 0

    if args.command == "phase2-review":
        result = run_phase2_review(
            project_root=Path("."),
            output_dir=Path(args.output_dir),
            components_path=Path(args.components_path),
            allocator_dir=Path(args.allocator_dir),
            promotion_review_dir=Path(args.promotion_review_dir) if args.promotion_review_dir else None,
        )
        print(f"Phase 2 review completed: {result.output_dir}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Posture: {result.snapshot.get('posture')}")
        print(f"Latest date: {result.snapshot.get('latest_date')}")
        print(f"Selected candidate: {result.snapshot.get('selected_candidate')}")
        return 0

    if args.command == "daily-check":
        key = _lock_key(
            "daily-check",
            Path(args.output_dir or "outputs/research/daily_model_check_latest").resolve(),
            Path(args.components_path).resolve(),
            Path(args.allocator_dir).resolve(),
            Path(args.promotion_review_dir).resolve() if args.promotion_review_dir else "no-promotion-review",
            args.phase2_review_dir or "nested",
            args.max_staleness_days,
            args.as_of_date or "today",
            args.date_stamp,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_daily_model_check(
                    project_root=Path("."),
                    output_dir=Path(args.output_dir) if args.output_dir else None,
                    phase2_review_dir=Path(args.phase2_review_dir) if args.phase2_review_dir else None,
                    components_path=Path(args.components_path),
                    allocator_dir=Path(args.allocator_dir),
                    promotion_review_dir=Path(args.promotion_review_dir) if args.promotion_review_dir else None,
                    max_staleness_days=args.max_staleness_days,
                    as_of_date=args.as_of_date,
                    date_stamp=args.date_stamp,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Daily model check completed: {result.output_dir}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Data freshness: {result.snapshot.get('data_freshness_status')}")
        print(f"Action posture: {result.snapshot.get('action_posture')}")
        print(f"Phase-2 posture: {result.snapshot.get('phase2_posture')}")
        return 0

    if args.command == "paper-account":
        key = _lock_key(
            "paper-account",
            Path(args.config).resolve(),
            Path(args.allocator_dir).resolve(),
            Path(args.output_dir).resolve(),
            args.rebalance_cost_rate,
            Path(args.trigger_signal_path).resolve() if args.trigger_signal_path else "default-trigger-signal-path",
            Path(args.stock_market_cap_path).resolve() if args.stock_market_cap_path else "default-stock-market-cap-path",
            args.stock_tracking_max_market_cap_yi,
            Path(args.stock_review_notes_path).resolve() if args.stock_review_notes_path else "default-stock-review-notes-path",
            Path(args.stock_review_outcomes_history_path).resolve() if args.stock_review_outcomes_history_path else "default-stock-review-outcomes-history-path",
            args.stock_review_drawdown_threshold,
            args.stock_review_watch_drawdown_threshold,
            args.stock_review_loss_attention_threshold,
            args.stock_review_gain_attention_threshold,
            args.stock_review_watch_score_threshold,
            args.stock_review_outcome_min_evaluable,
            args.stock_review_outcome_min_group_evaluable,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_paper_account(
                    project_root=Path("."),
                    config_path=Path(args.config),
                    allocator_dir=Path(args.allocator_dir),
                    output_dir=Path(args.output_dir),
                    rebalance_cost_rate=args.rebalance_cost_rate,
                    trigger_signal_path=Path(args.trigger_signal_path) if args.trigger_signal_path else None,
                    stock_market_cap_path=Path(args.stock_market_cap_path) if args.stock_market_cap_path else None,
                    stock_tracking_max_market_cap_yi=args.stock_tracking_max_market_cap_yi,
                    stock_review_notes_path=Path(args.stock_review_notes_path) if args.stock_review_notes_path else None,
                    stock_review_outcomes_history_path=Path(args.stock_review_outcomes_history_path) if args.stock_review_outcomes_history_path else None,
                    stock_review_drawdown_threshold=args.stock_review_drawdown_threshold,
                    stock_review_watch_drawdown_threshold=args.stock_review_watch_drawdown_threshold,
                    stock_review_loss_attention_threshold=args.stock_review_loss_attention_threshold,
                    stock_review_gain_attention_threshold=args.stock_review_gain_attention_threshold,
                    stock_review_watch_score_threshold=args.stock_review_watch_score_threshold,
                    stock_review_outcome_min_evaluable=args.stock_review_outcome_min_evaluable,
                    stock_review_outcome_min_group_evaluable=args.stock_review_outcome_min_group_evaluable,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Paper account completed: {result.output_dir}")
        print(f"Ledger CSV: {result.ledger_path}")
        print(f"Rebalance audit CSV: {result.audit_path}")
        print(f"Target holdings CSV: {result.target_holdings_path}")
        print(f"Target holdings report: {result.target_holdings_report_path}")
        print(f"Stock targets CSV: {result.stock_targets_path}")
        print(f"Stock targets report: {result.stock_targets_report_path}")
        print(f"Stock target review CSV: {result.stock_target_review_path}")
        print(f"Stock target review report: {result.stock_target_review_report_path}")
        print(f"Stock target review notes: {result.stock_target_review_notes_path}")
        print(f"Stock target review actions: {result.stock_target_review_actions_report_path}")
        print(f"Stock target review assistant: {result.stock_target_review_assistant_report_path}")
        print(f"Stock target review decision template: {result.stock_target_review_decision_template_report_path}")
        print(f"Stock target review decision template XLSX: {result.stock_target_review_decision_template_xlsx_path}")
        print(f"Stock target review outcomes: {result.stock_target_review_outcomes_report_path}")
        print(f"Stock target review outcome history: {result.stock_target_review_outcomes_history_report_path}")
        print(f"Stock target review outcome analysis: {result.stock_target_review_outcome_analysis_report_path}")
        print(f"Stock target review outcome calendar: {result.stock_target_review_outcome_calendar_report_path}")
        print(f"Stock target review outcome due queue: {result.stock_target_review_outcome_due_report_path}")
        print(f"Report written: {result.report_path}")
        print(f"Latest date: {result.metrics.get('latest_date')}")
        print(f"Latest candidate: {result.metrics.get('latest_candidate')}")
        print(f"Final equity: {float(result.metrics.get('final_equity', 0.0)):.2f}")
        print(f"Max drawdown: {float(result.metrics.get('max_drawdown', 0.0)) * 100:.2f}%")
        return 0

    if args.command == "stock-review-apply-template":
        key = _lock_key(
            "stock-review-apply-template",
            Path(args.template_path).resolve(),
            Path(args.notes_path).resolve(),
            Path(args.output_dir).resolve(),
            args.reviewed_at or "auto-reviewed-at",
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = apply_stock_target_review_decision_template(
                    Path(args.template_path),
                    Path(args.notes_path),
                    Path(args.output_dir),
                    reviewed_at=args.reviewed_at,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Stock review template apply completed: {result.output_dir}")
        print(f"Template CSV: {result.template_path}")
        print(f"Persistent notes CSV: {result.notes_path}")
        print(f"Notes snapshot CSV: {result.notes_snapshot_path}")
        print(f"Apply audit CSV: {result.audit_path}")
        print(f"Apply report: {result.report_path}")
        print(f"Applied rows: {result.payload.get('applied_count')}")
        print(f"Blank ignored rows: {result.payload.get('blank_ignored_count')}")
        print(f"Invalid status rows: {result.payload.get('invalid_status_count')}")
        print(f"Broker action: {result.payload.get('broker_action')}")
        return 0

    if args.command == "live-shadow":
        key = _lock_key(
            "live-shadow",
            Path(args.holdings_file).resolve(),
            Path(args.targets_file).resolve(),
            Path(args.prices_file).resolve() if args.prices_file else "no-prices-file",
            Path(args.output_dir).resolve(),
            args.cash,
            args.as_of_date or "today",
            args.lot_size,
            args.min_trade_value,
            args.max_position_weight,
            args.max_gross_exposure,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_live_shadow(
                    project_root=Path("."),
                    holdings_file=Path(args.holdings_file),
                    targets_file=Path(args.targets_file),
                    prices_file=Path(args.prices_file) if args.prices_file else None,
                    output_dir=Path(args.output_dir),
                    cash=args.cash,
                    as_of_date=args.as_of_date,
                    lot_size=args.lot_size,
                    min_trade_value=args.min_trade_value,
                    max_position_weight=args.max_position_weight,
                    max_gross_exposure=args.max_gross_exposure,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Live shadow plan completed: {result.output_dir}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Orders CSV: {result.orders_path}")
        print(f"Reconcile CSV: {result.reconcile_path}")
        print(f"Trade plan status: {result.snapshot.get('trade_plan_status')}")
        print(f"Broker action: {result.snapshot.get('broker_action')}")
        print(f"Order count: {result.snapshot.get('order_count')}")
        print(f"Estimated cash after shadow orders: {float(result.snapshot.get('estimated_cash_after_orders', 0.0)):.2f}")
        return 0

    if args.command == "live-shadow-preflight":
        key = _lock_key(
            "live-shadow-preflight",
            Path(args.holdings_file).resolve(),
            Path(args.targets_file).resolve(),
            Path(args.prices_file).resolve() if args.prices_file else "no-prices-file",
            Path(args.output_dir).resolve(),
            float(args.cash) if args.cash is not None else 0.0,
            args.as_of_date or "today",
            args.lot_size,
            args.min_trade_value,
            args.max_position_weight,
            args.max_gross_exposure,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_live_shadow_preflight(
                    project_root=Path("."),
                    holdings_file=Path(args.holdings_file),
                    targets_file=Path(args.targets_file),
                    prices_file=Path(args.prices_file) if args.prices_file else None,
                    cash=args.cash,
                    output_dir=Path(args.output_dir),
                    as_of_date=args.as_of_date,
                    lot_size=args.lot_size,
                    min_trade_value=args.min_trade_value,
                    max_position_weight=args.max_position_weight,
                    max_gross_exposure=args.max_gross_exposure,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Live shadow preflight completed: {result.output_dir}")
        print(f"Decision: {result.snapshot.get('status')}")
        print(f"Blocking items: {len(result.snapshot.get('blockers') or [])}")
        print(f"Order count: {result.snapshot.get('order_count', 0)}")
        print(f"Estimated cash after shadow orders: {float(result.snapshot.get('estimated_cash_after_orders', 0.0)):.2f}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        if args.fail_on_blocked and result.snapshot.get("status") == "blocked":
            return LIVE_PREFLIGHT_BLOCKED_EXIT_CODE
        return 0

    if args.command == "live-shadow-template":
        key = _lock_key(
            "live-shadow-template",
            Path(args.targets_file).resolve(),
            Path(args.output_dir).resolve(),
            args.as_of_date or "today",
            args.blank_rows,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_live_shadow_import_template(
                    project_root=Path("."),
                    targets_file=Path(args.targets_file),
                    output_dir=Path(args.output_dir),
                    as_of_date=args.as_of_date,
                    blank_rows=args.blank_rows,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Live shadow import template completed: {result.output_dir}")
        print(f"Highlighted workbook: {result.workbook_path}")
        print(f"Holdings CSV template: {result.holdings_template_path}")
        print(f"Cash CSV template: {result.cash_template_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Broker action: {result.snapshot.get('broker_action')}")
        return 0

    if args.command == "live-shadow-review":
        key = _lock_key(
            "live-shadow-review",
            Path(args.orders_file).resolve(),
            Path(args.targets_file).resolve(),
            Path(args.output_dir).resolve(),
            args.as_of_date or "today",
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_live_shadow_review_queue(
                    project_root=Path("."),
                    orders_file=Path(args.orders_file),
                    targets_file=Path(args.targets_file),
                    output_dir=Path(args.output_dir),
                    as_of_date=args.as_of_date,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Live shadow review queue completed: {result.output_dir}")
        print(f"Status: {result.snapshot.get('status')}")
        print(f"Review rows: {result.snapshot.get('review_row_count')}")
        print(f"Tracking-rule blocked rows: {result.snapshot.get('tracking_blocked_count')}")
        print(f"Review queue CSV: {result.review_queue_path}")
        print(f"Highlighted workbook: {result.workbook_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Broker action: {result.snapshot.get('broker_action')}")
        if args.fail_on_blocked and result.snapshot.get("tracking_blocked_count", 0):
            return LIVE_PREFLIGHT_BLOCKED_EXIT_CODE
        return 0

    if args.command == "live-shadow-review-apply":
        key = _lock_key(
            "live-shadow-review-apply",
            Path(args.review_file).resolve(),
            Path(args.output_dir).resolve(),
            args.as_of_date or "today",
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_live_shadow_review_decisions(
                    project_root=Path("."),
                    review_file=Path(args.review_file),
                    output_dir=Path(args.output_dir),
                    as_of_date=args.as_of_date,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Live shadow review decisions completed: {result.output_dir}")
        print(f"Status: {result.snapshot.get('status')}")
        print(f"Review rows: {result.snapshot.get('review_row_count')}")
        print(f"Filled statuses: {result.snapshot.get('filled_review_status_count')}")
        print(f"Invalid statuses: {result.snapshot.get('invalid_status_count')}")
        print(f"Blank statuses: {result.snapshot.get('blank_status_count')}")
        print(f"Tracking-rule blocked rows: {result.snapshot.get('tracking_blocked_count')}")
        print(f"Decisions CSV: {result.decisions_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Broker action: {result.snapshot.get('broker_action')}")
        return 0

    if args.command == "dashboard":
        key = _lock_key(
            "dashboard",
            Path(args.output_dir or "outputs/research/latest_dashboard").resolve(),
            args.daily_check_dir or "latest-daily-check",
            args.paper_account_dir or "latest-paper-account",
            args.sentiment_dir or "latest-sentiment",
            Path(args.data_cache_dir).resolve(),
            Path(args.allocator_dir).resolve(),
            args.trigger_report,
            args.model_audit_dir or "latest-model-audit",
            args.pipeline_history_dir or "latest-pipeline-history",
            args.daily_run_status_dir or "latest-daily-run-status",
            args.allocator_observation_dir or "latest-allocator-observation",
            args.max_staleness_days,
            args.min_cache_fresh_ratio,
            args.as_of_date or "today",
            args.date_stamp,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_daily_dashboard(
                    project_root=Path("."),
                    output_dir=Path(args.output_dir) if args.output_dir else None,
                    daily_check_dir=Path(args.daily_check_dir) if args.daily_check_dir else None,
                    paper_account_dir=Path(args.paper_account_dir) if args.paper_account_dir else None,
                    sentiment_dir=Path(args.sentiment_dir) if args.sentiment_dir else None,
                    data_cache_dir=Path(args.data_cache_dir),
                    allocator_dir=Path(args.allocator_dir),
                    trigger_report=Path(args.trigger_report) if args.trigger_report else None,
                    model_audit_dir=Path(args.model_audit_dir) if args.model_audit_dir else None,
                    pipeline_history_dir=Path(args.pipeline_history_dir) if args.pipeline_history_dir else None,
                    daily_run_status_dir=Path(args.daily_run_status_dir) if args.daily_run_status_dir else None,
                    allocator_observation_dir=Path(args.allocator_observation_dir)
                    if args.allocator_observation_dir
                    else None,
                    as_of_date=args.as_of_date,
                    max_staleness_days=args.max_staleness_days,
                    min_cache_fresh_ratio=args.min_cache_fresh_ratio,
                    date_stamp=args.date_stamp,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Dashboard completed: {result.output_dir}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Dashboard posture: {result.snapshot.get('dashboard_posture')}")
        print(f"Daily action posture: {result.snapshot.get('action_posture')}")
        print(f"Market sentiment: {result.snapshot.get('sentiment_state')}")
        print(f"Market cache: {result.snapshot.get('market_cache_status')}")
        print(f"Allocator inputs: {result.snapshot.get('allocator_input_status')}")
        print(f"Trigger freshness: {result.snapshot.get('trigger_freshness_status')}")
        print(f"Model audit: {result.snapshot.get('model_audit_status')}")
        print(f"Pipeline history: {result.snapshot.get('pipeline_history_health_state')}")
        print(f"Allocator observation: {result.snapshot.get('allocator_observation_decision_status')}")
        print(f"Live preflight status: {result.snapshot.get('live_preflight_status')}")
        print(f"Live preflight decision: {result.snapshot.get('live_preflight_decision')}")
        print(
            "Live preflight blockers/monitors: "
            f"{result.snapshot.get('live_preflight_blocking_items_count')} / "
            f"{result.snapshot.get('live_preflight_monitor_items_count')}"
        )
        print(f"Daily preflight skipped: `{result.snapshot.get('daily_preflight_skipped')}`")
        print(f"Live preflight status path: {result.snapshot.get('live_preflight_status_path')}")
        print(f"Live preflight report path: {result.snapshot.get('live_preflight_report_path')}")
        print(f"Live preflight snapshot path: {result.snapshot.get('live_preflight_snapshot_path')}")
        return 0

    if args.command == "daily-pipeline":
        key = _lock_key(
            "daily-pipeline",
            Path(args.output_dir or "outputs/research/daily_pipeline").resolve(),
            Path(args.components_path).resolve(),
            Path(args.allocator_dir).resolve(),
            Path(args.config).resolve(),
            Path(args.promotion_review_dir).resolve() if args.promotion_review_dir else "no-promotion-review",
            args.sentiment_dir or "latest-sentiment",
            args.data_cache_dir,
            Path(args.history_file).resolve(),
            args.history_review_output_dir or "auto-history-review",
            args.satellite_risk_budget_output_dir or "auto-satellite-risk-budget-review",
            args.network_lab_snapshot or "auto-network-lab-snapshot",
            args.network_max_cluster_count_warning,
            args.network_residual_mi_warning,
            args.momentum_focus_output_dir or "auto-momentum-focus",
            args.momentum_focus_board_scope,
            args.momentum_focus_threshold,
            args.momentum_focus_outcome_summary_path,
            args.momentum_focus_target_horizon,
            args.live_shadow_output_dir or "auto-live-shadow",
            args.live_shadow_preflight,
            args.live_shadow_preflight_fail_on_blocked,
            args.live_preflight_output_dir or "auto-live-preflight",
            Path(args.live_shadow_review_decisions_file).resolve()
            if args.live_shadow_review_decisions_file
            else "no-live-shadow-review-decisions",
            Path(args.live_shadow_holdings_file).resolve() if args.live_shadow_holdings_file else "no-live-shadow-holdings",
            Path(args.live_shadow_prices_file).resolve() if args.live_shadow_prices_file else "no-live-shadow-prices",
            args.live_shadow_cash if args.live_shadow_cash is not None else "no-live-shadow-cash",
            args.live_shadow_lot_size,
            args.live_shadow_min_trade_value,
            args.live_shadow_max_position_weight,
            args.live_shadow_max_gross_exposure,
            args.trigger_report,
            Path(args.model_audit_dir).resolve() if args.model_audit_dir else "latest-model-audit",
            args.max_staleness_days,
            args.min_cache_fresh_ratio,
            args.rebalance_cost_rate,
            Path(args.stock_market_cap_path).resolve() if args.stock_market_cap_path else "default-stock-market-cap-path",
            args.stock_tracking_max_market_cap_yi,
            Path(args.stock_review_notes_path).resolve() if args.stock_review_notes_path else "default-stock-review-notes-path",
            Path(args.stock_review_outcomes_history_path).resolve() if args.stock_review_outcomes_history_path else "default-stock-review-outcomes-history-path",
            args.stock_review_drawdown_threshold,
            args.stock_review_watch_drawdown_threshold,
            args.stock_review_loss_attention_threshold,
            args.stock_review_gain_attention_threshold,
            args.stock_review_watch_score_threshold,
            args.stock_review_outcome_min_evaluable,
            args.stock_review_outcome_min_group_evaluable,
            args.stock_review_warning_only_after_close,
            args.chip_reversal_candidate_outcomes,
            Path(args.chip_reversal_candidates_path).resolve(),
            Path(args.chip_reversal_outcomes_data_dir).resolve(),
            args.chip_reversal_outcomes_output_dir or "auto-chip-reversal-outcomes",
            args.chip_reversal_outcome_horizons,
            args.chip_reversal_outcome_success_threshold,
            args.chip_reversal_outcome_min_ready_per_horizon,
            args.chip_reversal_outcome_min_success_rate,
            args.run_history_review,
            args.history_lookback_runs,
            args.history_drawdown_watch_threshold,
            args.history_min_sharpe_watch,
            args.fail_on_alert_level,
            args.fail_on_history_health,
            args.as_of_date or "today",
            args.date_stamp,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_daily_pipeline(
                    project_root=Path("."),
                    output_dir=Path(args.output_dir) if args.output_dir else None,
                    daily_check_output_dir=Path(args.daily_check_output_dir) if args.daily_check_output_dir else None,
                    paper_account_output_dir=Path(args.paper_account_output_dir) if args.paper_account_output_dir else None,
                    dashboard_output_dir=Path(args.dashboard_output_dir) if args.dashboard_output_dir else None,
                    momentum_focus_output_dir=(
                        Path(args.momentum_focus_output_dir) if args.momentum_focus_output_dir else None
                    ),
                    momentum_focus_board_scope=args.momentum_focus_board_scope,
                    momentum_focus_strong_gain_threshold_pct=args.momentum_focus_threshold,
                    momentum_focus_outcome_summary_path=args.momentum_focus_outcome_summary_path,
                    momentum_focus_target_horizon=args.momentum_focus_target_horizon,
                    components_path=Path(args.components_path),
                    allocator_dir=Path(args.allocator_dir),
                    portfolio_config_path=Path(args.config),
                    sentiment_dir=Path(args.sentiment_dir) if args.sentiment_dir else None,
                    data_cache_dir=Path(args.data_cache_dir) if args.data_cache_dir else None,
                    trigger_report=Path(args.trigger_report) if args.trigger_report else None,
                    model_audit_dir=Path(args.model_audit_dir) if args.model_audit_dir else None,
                    history_path=Path(args.history_file),
                    history_review_output_dir=Path(args.history_review_output_dir) if args.history_review_output_dir else None,
                    satellite_risk_budget_output_dir=(
                        Path(args.satellite_risk_budget_output_dir) if args.satellite_risk_budget_output_dir else None
                    ),
                    network_lab_snapshot=Path(args.network_lab_snapshot) if args.network_lab_snapshot else None,
                    network_max_cluster_count_warning=args.network_max_cluster_count_warning,
                    network_residual_mi_warning=args.network_residual_mi_warning,
                    live_shadow_output_dir=Path(args.live_shadow_output_dir) if args.live_shadow_output_dir else None,
                    live_shadow_preflight=args.live_shadow_preflight,
                    live_shadow_preflight_fail_on_blocked=args.live_shadow_preflight_fail_on_blocked,
                    live_preflight_output_dir=Path(args.live_preflight_output_dir) if args.live_preflight_output_dir else None,
                    live_shadow_review_decisions_file=(
                        Path(args.live_shadow_review_decisions_file) if args.live_shadow_review_decisions_file else None
                    ),
                    live_shadow_holdings_file=Path(args.live_shadow_holdings_file) if args.live_shadow_holdings_file else None,
                    live_shadow_prices_file=Path(args.live_shadow_prices_file) if args.live_shadow_prices_file else None,
                    live_shadow_cash=args.live_shadow_cash,
                    live_shadow_lot_size=args.live_shadow_lot_size,
                    live_shadow_min_trade_value=args.live_shadow_min_trade_value,
                    live_shadow_max_position_weight=args.live_shadow_max_position_weight,
                    live_shadow_max_gross_exposure=args.live_shadow_max_gross_exposure,
                    phase2_review_dir=Path(args.phase2_review_dir) if args.phase2_review_dir else None,
                    promotion_review_dir=Path(args.promotion_review_dir) if args.promotion_review_dir else None,
                    max_staleness_days=args.max_staleness_days,
                    min_cache_fresh_ratio=args.min_cache_fresh_ratio,
                    rebalance_cost_rate=args.rebalance_cost_rate,
                    stock_market_cap_path=Path(args.stock_market_cap_path) if args.stock_market_cap_path else None,
                    stock_tracking_max_market_cap_yi=args.stock_tracking_max_market_cap_yi,
                    stock_review_notes_path=Path(args.stock_review_notes_path) if args.stock_review_notes_path else None,
                    stock_review_outcomes_history_path=Path(args.stock_review_outcomes_history_path) if args.stock_review_outcomes_history_path else None,
                    stock_review_drawdown_threshold=args.stock_review_drawdown_threshold,
                    stock_review_watch_drawdown_threshold=args.stock_review_watch_drawdown_threshold,
                    stock_review_loss_attention_threshold=args.stock_review_loss_attention_threshold,
                    stock_review_gain_attention_threshold=args.stock_review_gain_attention_threshold,
                    stock_review_watch_score_threshold=args.stock_review_watch_score_threshold,
                    stock_review_outcome_min_evaluable=args.stock_review_outcome_min_evaluable,
                    stock_review_outcome_min_group_evaluable=args.stock_review_outcome_min_group_evaluable,
                    stock_review_warning_only_after_close=args.stock_review_warning_only_after_close,
                    chip_reversal_candidate_outcomes=args.chip_reversal_candidate_outcomes,
                    chip_reversal_candidates_path=Path(args.chip_reversal_candidates_path),
                    chip_reversal_outcomes_data_dir=Path(args.chip_reversal_outcomes_data_dir),
                    chip_reversal_outcomes_output_dir=(
                        Path(args.chip_reversal_outcomes_output_dir) if args.chip_reversal_outcomes_output_dir else None
                    ),
                    chip_reversal_outcome_horizons=_parse_int_csv(args.chip_reversal_outcome_horizons),
                    chip_reversal_outcome_success_threshold=args.chip_reversal_outcome_success_threshold,
                    chip_reversal_outcome_min_ready_per_horizon=args.chip_reversal_outcome_min_ready_per_horizon,
                    chip_reversal_outcome_min_success_rate=args.chip_reversal_outcome_min_success_rate,
                    run_history_review=args.run_history_review,
                    history_review_lookback_runs=args.history_lookback_runs,
                    history_review_drawdown_watch_threshold=args.history_drawdown_watch_threshold,
                    history_review_min_sharpe_watch=args.history_min_sharpe_watch,
                    as_of_date=args.as_of_date,
                    date_stamp=args.date_stamp,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Daily pipeline completed: {result.output_dir}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Daily check: {result.daily_check_result.report_path}")
        print(f"Paper account: {result.paper_account_result.report_path}")
        print(f"Paper target holdings: {result.paper_account_result.target_holdings_report_path}")
        print(f"Paper stock targets: {result.paper_account_result.stock_targets_report_path}")
        print(f"Paper stock target review: {result.paper_account_result.stock_target_review_report_path}")
        print(f"Paper stock target review notes: {result.paper_account_result.stock_target_review_notes_path}")
        print(f"Paper stock target review actions: {result.paper_account_result.stock_target_review_actions_report_path}")
        print(f"Paper stock target review assistant: {result.paper_account_result.stock_target_review_assistant_report_path}")
        print(f"Paper stock target review decision template: {result.paper_account_result.stock_target_review_decision_template_report_path}")
        print(f"Paper stock target review decision template XLSX: {result.paper_account_result.stock_target_review_decision_template_xlsx_path}")
        print(f"Paper stock target review outcomes: {result.paper_account_result.stock_target_review_outcomes_report_path}")
        print(f"Paper stock target review outcome history: {result.paper_account_result.stock_target_review_outcomes_history_report_path}")
        print(f"Paper stock target review outcome analysis: {result.paper_account_result.stock_target_review_outcome_analysis_report_path}")
        print(f"Paper stock target review outcome calendar: {result.paper_account_result.stock_target_review_outcome_calendar_report_path}")
        print(f"Paper stock target review outcome due queue: {result.paper_account_result.stock_target_review_outcome_due_report_path}")
        print(f"Dashboard: {result.dashboard_result.report_path}")
        print(f"Dashboard posture: {result.snapshot.get('dashboard_posture')}")
        print(f"Trading-day gate: {result.snapshot.get('trading_day_gate_status')}")
        print(f"After-close data: {result.snapshot.get('after_close_data_status')}")
        print(f"Paper latest regime: {result.snapshot.get('paper_latest_regime')}")
        print(f"History: {result.history_path}")
        print(f"History review: {result.snapshot.get('history_review_report_path')}")
        print(f"History health: {result.snapshot.get('history_review_health_state')}")
        print(f"History alerts: {result.snapshot.get('history_review_alert_count')}")
        print(f"Satellite risk-budget review: {result.snapshot.get('satellite_risk_budget_report_path')}")
        print(f"Satellite risk-budget decision: {result.snapshot.get('satellite_risk_budget_decision')}")
        print(f"Momentum focus: {result.snapshot.get('momentum_focus_report_path')}")
        print(f"Momentum focus status: {result.snapshot.get('momentum_focus_status')}")
        print(f"Momentum focus candidates: {result.snapshot.get('momentum_focus_candidate_count')}")
        print(f"Momentum focus limit-up rows: {result.snapshot.get('momentum_focus_limit_up_count')}")
        print(f"Chip reversal outcomes: {result.snapshot.get('chip_reversal_candidate_outcomes_report_path')}")
        print(f"Chip reversal outcome readiness: {result.snapshot.get('chip_reversal_candidate_outcomes_readiness_status')}")
        print(f"Live shadow preflight: {result.snapshot.get('live_shadow_preflight_report_path')}")
        print(f"Live shadow preflight decision: {result.snapshot.get('live_shadow_preflight_decision')}")
        print(f"Live shadow: {result.snapshot.get('live_shadow_report_path')}")
        print(f"Live preflight: {result.snapshot.get('live_preflight_report_path')}")
        print(f"Live preflight decision: {result.snapshot.get('live_preflight_decision')}")
        print(f"Daily preflight skipped: {result.snapshot.get('daily_preflight_skipped')}")
        print(f"Live shadow status: {result.snapshot.get('live_shadow_status')}")
        print(f"Live shadow orders: {result.snapshot.get('live_shadow_order_count')}")
        print(f"Model audit: {result.snapshot.get('model_audit_status')}")
        print(f"Model audit actions: {result.snapshot.get('model_audit_walk_forward_action_items')}")
        print(f"Promotion decision: {result.snapshot.get('promotion_decision')}")
        print(f"Promotion review: {result.snapshot.get('promotion_report_path')}")
        print(f"Alert level: {result.snapshot.get('alert_level')}")
        print(f"Action stage: {result.snapshot.get('alert_action_stage')}")
        print(f"Alert count: {result.snapshot.get('alert_count')}")
        print(f"Alert report: {result.snapshot.get('alerts_report_path')}")
        if _alert_gate_failed(result.snapshot.get("alert_level"), args.fail_on_alert_level):
            print(
                "Alert gate failed: "
                f"alert_level={result.snapshot.get('alert_level')} threshold={args.fail_on_alert_level}"
            )
            return ALERT_GATE_EXIT_CODE
        if _history_gate_failed(result.snapshot.get("history_review_health_state"), args.fail_on_history_health):
            print(
                "History health gate failed: "
                f"health={result.snapshot.get('history_review_health_state')} threshold={args.fail_on_history_health}"
            )
            return HISTORY_GATE_EXIT_CODE
        return 0

    if args.command == "daily-run-status":
        key = _lock_key(
            "daily-run-status",
            Path(args.output_dir).resolve(),
            Path(args.logs_dir).resolve(),
            Path(args.research_dir).resolve(),
            Path(args.data_cache_dir).resolve(),
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_daily_run_status(
                    project_root=Path("."),
                    output_dir=Path(args.output_dir),
                    logs_dir=Path(args.logs_dir),
                    research_dir=Path(args.research_dir),
                    data_cache_dir=Path(args.data_cache_dir),
                    scheduled_run_time=args.scheduled_run_time,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Daily run status completed: {result.output_dir}")
        print(f"Run state: {result.snapshot.get('run_state')}")
        print(f"Run state severity: {result.snapshot.get('run_state_severity')}")
        print(f"Latest refresh exit code: {result.snapshot.get('latest_refresh_exit_code')}")
        print(f"Latest observation exit code: {result.snapshot.get('latest_observation_exit_code')}")
        print(f"Wrapper final stage: {result.snapshot.get('latest_wrapper_final_stage')}")
        print(f"Live preflight status: {result.snapshot.get('latest_live_preflight_status')}")
        print(f"Live preflight decision: {result.snapshot.get('latest_live_preflight_decision')}")
        print(
            "Live preflight blockers/monitors: "
            f"{result.snapshot.get('latest_live_preflight_blocking_items_count')} / "
            f"{result.snapshot.get('latest_live_preflight_monitor_items_count')}"
        )
        print(f"Live preflight status path: {result.snapshot.get('latest_live_preflight_status_path')}")
        print(f"Live preflight report path: {result.snapshot.get('latest_live_preflight_report_path')}")
        print(f"Live preflight snapshot path: {result.snapshot.get('latest_live_preflight_snapshot_path')}")
        print(f"Outcome ready horizons: {result.snapshot.get('outcome_ready_horizon_count')}")
        print(f"Next review: {result.snapshot.get('next_review_date')} / {result.snapshot.get('next_review_horizon')}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        if args.json_path:
            Path(args.json_path).write_text(json.dumps(result.snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.print_json:
            print(json.dumps(result.snapshot, ensure_ascii=False, sort_keys=True))
        if args.fail_on_problem_state and result.snapshot.get("problem_state"):
            print(f"Daily run status gate failed: run_state={result.snapshot.get('run_state')}")
            return DAILY_RUN_STATUS_PROBLEM_EXIT_CODE
        return 0

    if args.command == "pipeline-history":
        key = _lock_key(
            "pipeline-history",
            Path(args.history_file).resolve(),
            Path(args.output_dir).resolve(),
            args.as_of_date or "today",
            args.lookback_runs,
            args.max_staleness_days,
            args.drawdown_watch_threshold,
            args.min_sharpe_watch,
            args.fail_on_health,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_pipeline_history_review(
                    project_root=Path("."),
                    history_file=Path(args.history_file),
                    output_dir=Path(args.output_dir),
                    as_of_date=args.as_of_date,
                    lookback_runs=args.lookback_runs,
                    max_staleness_days=args.max_staleness_days,
                    drawdown_watch_threshold=args.drawdown_watch_threshold,
                    min_sharpe_watch=args.min_sharpe_watch,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Pipeline history review completed: {result.output_dir}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Health state: {result.snapshot.get('health_state')}")
        print(f"Alerts: {result.snapshot.get('alert_count')}")
        print(f"Latest history as-of: {result.snapshot.get('latest_as_of_date')}")
        if _history_gate_failed(result.snapshot.get("health_state"), args.fail_on_health):
            print(
                "History health gate failed: "
                f"health={result.snapshot.get('health_state')} threshold={args.fail_on_health}"
            )
            return HISTORY_GATE_EXIT_CODE
        return 0

    if args.command == "model-audit":
        key = _lock_key(
            "model-audit",
            Path(args.config_dir).resolve(),
            Path(args.walk_forward_dir).resolve(),
            Path(args.walk_forward_resolution_path).resolve(),
            Path(args.lock_dir).resolve(),
            Path(args.output_dir).resolve(),
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_model_build_audit(
                    project_root=Path("."),
                    config_dir=Path(args.config_dir),
                    walk_forward_dir=Path(args.walk_forward_dir),
                    walk_forward_resolution_path=Path(args.walk_forward_resolution_path),
                    lock_dir=Path(args.lock_dir),
                    output_dir=Path(args.output_dir),
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Model audit completed: {result.output_dir}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        print(f"Config redundancy CSV: {result.config_redundancy_path}")
        print(f"Config inheritance map CSV: {result.config_map_path}")
        print(f"Walk-forward runs CSV: {result.walk_forward_runs_path}")
        print(f"Walk-forward run actions CSV: {result.walk_forward_actions_path}")
        print(f"Duplicate config groups: {result.snapshot.get('duplicate_config_groups')}")
        print(f"Incomplete walk-forward runs: {result.snapshot.get('incomplete_walk_forward_runs')}")
        print(f"Unresolved incomplete walk-forward runs: {result.snapshot.get('unresolved_incomplete_walk_forward_runs')}")
        print(f"Active lock files: {result.snapshot.get('active_lock_files')}")
        return 0

    if args.command == "allocator-promotion-review":
        key = _lock_key(
            "allocator-promotion-review",
            Path(args.baseline_dir).resolve(),
            Path(args.candidate_dir).resolve(),
            tuple(str(Path(path).resolve()) for path in args.sensitivity_dir),
            tuple(args.sensitivity_group),
            tuple(str(Path(path).resolve()) for path in args.evidence_snapshot),
            tuple(args.evidence_group),
            Path(args.output_dir).resolve(),
            args.min_return_edge,
            args.min_sharpe_edge,
            args.max_drawdown_worsening,
            args.min_positive_window_ratio,
            args.min_sensitivity_support,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_allocator_promotion_review(
                    baseline_dir=Path(args.baseline_dir),
                    candidate_dir=Path(args.candidate_dir),
                    sensitivity_dirs=[Path(path) for path in args.sensitivity_dir],
                    sensitivity_groups=args.sensitivity_group or None,
                    evidence_snapshots=[Path(path) for path in args.evidence_snapshot],
                    evidence_groups=args.evidence_group or None,
                    output_dir=Path(args.output_dir),
                    min_return_edge=args.min_return_edge,
                    min_sharpe_edge=args.min_sharpe_edge,
                    max_drawdown_worsening=args.max_drawdown_worsening,
                    min_positive_window_ratio=args.min_positive_window_ratio,
                    min_sensitivity_support=args.min_sensitivity_support,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Allocator promotion review completed: {result.output_dir}")
        print(f"Decision: {result.decision}")
        print(f"Comparison CSV: {result.comparison_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        return 0

    if args.command == "source-decision-review":
        key = _lock_key(
            "source-decision-review",
            Path(args.source_dir).resolve(),
            Path(args.baseline_dir).resolve() if args.baseline_dir else "no-baseline",
            Path(args.output_dir).resolve(),
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_source_decision_review(
                    source_dir=Path(args.source_dir),
                    baseline_dir=Path(args.baseline_dir) if args.baseline_dir else None,
                    output_dir=Path(args.output_dir),
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Source decision review completed: {result.output_dir}")
        print(f"Windows CSV: {result.windows_path}")
        print(f"Source scores CSV: {result.source_scores_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        return 0

    if args.command == "source-guard-decomposition":
        key = _lock_key(
            "source-guard-decomposition",
            Path(args.default_dir).resolve(),
            Path(args.candidate_dir).resolve(),
            Path(args.output_dir).resolve(),
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_source_guard_decomposition(
                    default_dir=Path(args.default_dir),
                    candidate_dir=Path(args.candidate_dir),
                    output_dir=Path(args.output_dir),
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Source guard decomposition completed: {result.output_dir}")
        print(f"Windows CSV: {result.windows_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        return 0

    if args.command == "source-risk-budget-review":
        key = _lock_key(
            "source-risk-budget-review",
            Path(args.source_dir).resolve(),
            Path(args.guard_decomposition_dir).resolve() if args.guard_decomposition_dir else "no-guard-decomposition",
            Path(args.output_dir).resolve(),
            args.fragile_source_edge,
            args.max_low_risk_satellite_weight,
            args.min_filter_off_ratio,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_source_risk_budget_review(
                    source_dir=Path(args.source_dir),
                    guard_decomposition_dir=Path(args.guard_decomposition_dir) if args.guard_decomposition_dir else None,
                    output_dir=Path(args.output_dir),
                    fragile_source_edge=args.fragile_source_edge,
                    max_low_risk_satellite_weight=args.max_low_risk_satellite_weight,
                    min_filter_off_ratio=args.min_filter_off_ratio,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Source risk-budget review completed: {result.output_dir}")
        print(f"Windows CSV: {result.windows_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        return 0

    if args.command == "portfolio-window-attribution":
        key = _lock_key(
            "portfolio-window-attribution",
            Path(args.config).resolve(),
            args.start,
            args.end,
            tuple(args.variant),
            args.include_core_only,
            Path(args.output_dir).resolve(),
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_portfolio_window_attribution(
                    config_path=Path(args.config),
                    start=args.start,
                    end=args.end,
                    variants=args.variant,
                    include_core_only=args.include_core_only,
                    output_dir=Path(args.output_dir),
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Portfolio window attribution completed: {result.output_dir}")
        print(f"Metrics CSV: {result.metrics_path}")
        print(f"Daily CSV: {result.daily_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        return 0

    if args.command == "allocator-switch-readiness":
        key = _lock_key(
            "allocator-switch-readiness",
            Path(args.default_snapshot).resolve(),
            Path(args.candidate_snapshot).resolve(),
            Path(args.output_dir).resolve(),
            Path(args.outcome_analysis_path).resolve() if args.outcome_analysis_path else "auto-outcome-analysis",
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_allocator_switch_readiness(
                    default_snapshot=Path(args.default_snapshot),
                    candidate_snapshot=Path(args.candidate_snapshot),
                    output_dir=Path(args.output_dir),
                    default_label=args.default_label,
                    candidate_label=args.candidate_label,
                    outcome_analysis_path=Path(args.outcome_analysis_path) if args.outcome_analysis_path else None,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Allocator switch-readiness completed: {result.output_dir}")
        print(f"Decision: {result.snapshot.get('decision')}")
        print(f"Risk budget decision: {result.snapshot.get('risk_budget_decision')}")
        print(f"Comparison CSV: {result.comparison_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        return 0

    if args.command == "allocator-observation":
        key = _lock_key(
            "allocator-observation",
            Path(args.pipeline_snapshot).resolve(),
            Path(args.output_dir).resolve(),
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_allocator_observation(
                    pipeline_snapshot=Path(args.pipeline_snapshot),
                    output_dir=Path(args.output_dir),
                    baseline_label=args.baseline_label,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Allocator observation completed: {result.output_dir}")
        print(f"Observation status: {result.snapshot.get('observation_status')}")
        print(f"Next action stage: {result.snapshot.get('next_action_stage')}")
        print(f"Checklist CSV: {result.checklist_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        return 0

    if args.command == "satellite-risk-budget-review":
        key = _lock_key(
            "satellite-risk-budget-review",
            Path(args.pipeline_snapshot).resolve(),
            Path(args.output_dir).resolve(),
            Path(args.network_lab_snapshot).resolve() if args.network_lab_snapshot else "no-network-lab-snapshot",
            args.network_max_cluster_count_warning,
            args.network_residual_mi_warning,
            Path(args.outcome_analysis_path).resolve() if args.outcome_analysis_path else "auto-outcome-analysis",
            args.trial_satellite_budget,
            args.max_satellite_budget,
            args.min_overall_win_rate,
            args.min_overall_avg_return,
            args.max_worst_group_loss,
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_satellite_risk_budget_review(
                    pipeline_snapshot=Path(args.pipeline_snapshot),
                    outcome_analysis_path=Path(args.outcome_analysis_path) if args.outcome_analysis_path else None,
                    output_dir=Path(args.output_dir),
                    network_lab_snapshot=Path(args.network_lab_snapshot) if args.network_lab_snapshot else None,
                    network_max_cluster_count_warning=args.network_max_cluster_count_warning,
                    network_residual_mi_warning=args.network_residual_mi_warning,
                    trial_satellite_budget=args.trial_satellite_budget,
                    max_satellite_budget=args.max_satellite_budget,
                    min_overall_win_rate=args.min_overall_win_rate,
                    min_overall_avg_return=args.min_overall_avg_return,
                    max_worst_group_loss=args.max_worst_group_loss,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Satellite risk-budget review completed: {result.output_dir}")
        print(f"Risk budget decision: {result.snapshot.get('risk_budget_decision')}")
        print(f"Recommended satellite budget: {result.snapshot.get('recommended_satellite_budget')}")
        print(f"Checklist CSV: {result.checklist_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        return 0

    if args.command == "live-preflight":
        key = _lock_key(
            "live-preflight",
            Path(args.dashboard_snapshot).resolve(),
            Path(args.paper_account_dir).resolve() if args.paper_account_dir else "dashboard-paper-account-dir",
            Path(args.pipeline_snapshot).resolve() if args.pipeline_snapshot else "optional-pipeline-snapshot",
            Path(args.daily_run_status_snapshot).resolve() if args.daily_run_status_snapshot else "optional-daily-run-status",
            Path(args.live_shadow_review_decisions_file).resolve()
            if args.live_shadow_review_decisions_file
            else "optional-live-shadow-review-decisions",
            Path(args.output_dir).resolve(),
        )
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                result = run_live_preflight(
                    dashboard_snapshot=Path(args.dashboard_snapshot),
                    paper_account_dir=Path(args.paper_account_dir) if args.paper_account_dir else None,
                    pipeline_snapshot=Path(args.pipeline_snapshot) if args.pipeline_snapshot else None,
                    daily_run_status_snapshot=Path(args.daily_run_status_snapshot) if args.daily_run_status_snapshot else None,
                    live_shadow_review_decisions_file=(
                        Path(args.live_shadow_review_decisions_file) if args.live_shadow_review_decisions_file else None
                    ),
                    output_dir=Path(args.output_dir),
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Live preflight completed: {result.output_dir}")
        print(f"Decision: {result.snapshot.get('decision')}")
        print(f"Broker connection status: {result.snapshot.get('broker_connection_status')}")
        print(f"Blocking items: {len(result.snapshot.get('blocking_items') or [])}")
        print(f"Checklist CSV: {result.checklist_path}")
        print(f"Snapshot JSON: {result.snapshot_path}")
        print(f"Report written: {result.report_path}")
        if args.fail_on_blocked and result.snapshot.get("decision") == "blocked":
            return LIVE_PREFLIGHT_BLOCKED_EXIT_CODE
        return 0

    if args.command in {"walk-forward", "rolling-train"}:
        walk_options = resolve_walk_forward_options(args)
        key = _walk_forward_lock_key(args, walk_options)
        try:
            with process_lock(_lock_dir(), key, _command_line()):
                config = load_config(args.config)
                wf_dir, summary = run_walk_forward(
                    config,
                    train_years=int(walk_options["train_years"]),
                    test_months=int(walk_options["test_months"]),
                    step_months=int(walk_options["step_months"]),
                    grid=str(walk_options["grid"]),
                    objective=str(walk_options["objective"]),
                    max_train_drawdown=float(walk_options["max_train_drawdown"]),
                    selection_validation_months=int(walk_options["selection_validation_months"]),
                    output_dir=Path(args.output_dir),
                    run_id_prefix=args.run_id_prefix,
                    allow_fetch=args.allow_fetch,
                    skip_missing=args.skip_missing,
                    resume=args.resume,
                    window_limit=args.window_limit,
                )
        except ProcessLockError as error:
            return _duplicate_process_exit(error)
        print(f"Walk-forward completed: {wf_dir}")
        print(f"Summary CSV: {summary.attrs.get('summary_csv')}")
        print(f"Candidate CSV: {summary.attrs.get('candidate_csv')}")
        print(f"Overfit audit CSV: {summary.attrs.get('overfit_audit_csv')}")
        print(f"Summary report: {summary.attrs.get('summary_md')}")
        print(f"Stitched OOS equity: {summary.attrs.get('stitched_csv')}")
        print(f"Stitched OOS total return: {summary.attrs.get('stitched_total_return') * 100:.2f}%")
        print(f"Stitched OOS max drawdown: {summary.attrs.get('stitched_max_drawdown') * 100:.2f}%")
        print(f"Stitched OOS Sharpe: {summary.attrs.get('stitched_sharpe'):.3f}")
        for row in summary.itertuples(index=False):
            print(
                f"{row.window}: selected={row.selected_candidate}, "
                f"test_return={row.test_total_return * 100:.2f}%, "
                f"test_drawdown={row.test_max_drawdown * 100:.2f}%, "
                f"test_sharpe={row.test_sharpe:.3f}"
            )
        return 0

    if args.command == "portfolio":
        if args.portfolio_command == "combine":
            key = _lock_key("portfolio-combine", Path(args.config).resolve(), args.run_id or "auto")
            try:
                with process_lock(_lock_dir(), key, _command_line()):
                    config = load_portfolio_config(args.config)
                    result = run_portfolio_combine(config, run_id=args.run_id)
            except ProcessLockError as error:
                return _duplicate_process_exit(error)
            print(f"Portfolio completed: {result.run_dir}")
            print(f"Final equity: {result.metrics['final_equity']:.2f}")
            print(f"Total return: {result.metrics['total_return'] * 100:.2f}%")
            print(f"Max drawdown: {result.metrics['max_drawdown'] * 100:.2f}%")
            print(f"Excess return vs core: {result.metrics['excess_return_vs_core'] * 100:.2f}%")
            print(f"Average satellite weight: {result.metrics['average_satellite_weight'] * 100:.2f}%")
            return 0
        if args.portfolio_command == "walk-forward":
            key = _lock_key(
                "portfolio-walk-forward",
                Path(args.config).resolve(),
                Path(args.output_dir).resolve(),
                args.run_id_prefix or "auto",
                args.train_months,
                args.test_months,
                args.step_months,
                args.grid,
                args.max_train_drawdown,
                args.score_mode,
                args.score_mode_min_edge,
            )
            try:
                with process_lock(_lock_dir(), key, _command_line()):
                    config = load_portfolio_config(args.config)
                    wf_dir, summary = run_portfolio_walk_forward(
                        config,
                        train_months=args.train_months,
                        test_months=args.test_months,
                        step_months=args.step_months,
                        grid=args.grid,
                        max_train_drawdown=args.max_train_drawdown,
                        score_mode=args.score_mode,
                        score_mode_min_edge=args.score_mode_min_edge,
                        output_dir=Path(args.output_dir),
                        run_id_prefix=args.run_id_prefix,
                    )
            except ProcessLockError as error:
                return _duplicate_process_exit(error)
            print(f"Portfolio walk-forward completed: {wf_dir}")
            print(f"Summary CSV: {summary.attrs.get('summary_csv')}")
            print(f"Candidate CSV: {summary.attrs.get('candidate_csv')}")
            print(f"Summary report: {summary.attrs.get('summary_md')}")
            print(f"Stitched OOS equity: {summary.attrs.get('stitched_csv')}")
            print(f"Stitched OOS total return: {summary.attrs.get('stitched_total_return') * 100:.2f}%")
            print(f"Stitched OOS max drawdown: {summary.attrs.get('stitched_max_drawdown') * 100:.2f}%")
            print(f"Stitched OOS Sharpe: {summary.attrs.get('stitched_sharpe'):.3f}")
            for row in summary.itertuples(index=False):
                print(
                    f"{row.window}: selected={row.selected_candidate}, "
                    f"test_return={row.test_total_return * 100:.2f}%, "
                    f"test_drawdown={row.test_max_drawdown * 100:.2f}%, "
                    f"test_sharpe={row.test_sharpe:.3f}"
                )
            return 0
        if args.portfolio_command == "source-selection":
            try:
                source_max_satellite_weight = _parse_source_max_satellite_weight(args.source_max_satellite_weight)
                source_switch_margin_by_source = _parse_source_switch_margin_by_source(
                    args.source_switch_margin_by_source
                )
            except ValueError as error:
                print(f"Invalid source-specific option: {error}")
                return 2
            key = _lock_key(
                "portfolio-source-selection",
                Path(args.config).resolve(),
                Path(args.output_dir).resolve(),
                args.run_id_prefix or "auto",
                args.train_months,
                args.test_months,
                args.step_months,
                args.grid,
                args.max_train_drawdown,
                args.score_mode,
                args.score_mode_min_edge,
                args.default_source or "",
                args.source_switch_margin,
                json.dumps(source_switch_margin_by_source, sort_keys=True),
                args.source_validation_months,
                args.source_stability_penalty,
                args.max_satellite_weight if args.max_satellite_weight is not None else "no-max-satellite-weight",
                json.dumps(source_max_satellite_weight, sort_keys=True),
            )
            try:
                with process_lock(_lock_dir(), key, _command_line()):
                    config, satellite_sources = load_portfolio_source_selection_config(args.config)
                    wf_dir, summary = run_portfolio_source_selection_walk_forward(
                        config,
                        satellite_sources,
                        train_months=args.train_months,
                        test_months=args.test_months,
                        step_months=args.step_months,
                        grid=args.grid,
                        max_train_drawdown=args.max_train_drawdown,
                        score_mode=args.score_mode,
                        score_mode_min_edge=args.score_mode_min_edge,
                        default_source_name=args.default_source,
                        source_switch_margin=args.source_switch_margin,
                        source_switch_margin_by_source=source_switch_margin_by_source,
                        source_validation_months=args.source_validation_months,
                        source_stability_penalty=args.source_stability_penalty,
                        max_satellite_weight=args.max_satellite_weight,
                        source_max_satellite_weight=source_max_satellite_weight,
                        output_dir=Path(args.output_dir),
                        run_id_prefix=args.run_id_prefix,
                    )
            except ProcessLockError as error:
                return _duplicate_process_exit(error)
            print(f"Portfolio source-selection completed: {wf_dir}")
            print(f"Summary CSV: {summary.attrs.get('summary_csv')}")
            print(f"Candidate CSV: {summary.attrs.get('candidate_csv')}")
            print(f"Summary report: {summary.attrs.get('summary_md')}")
            print(f"Stitched OOS equity: {summary.attrs.get('stitched_csv')}")
            print(f"Stitched OOS total return: {summary.attrs.get('stitched_total_return') * 100:.2f}%")
            print(f"Stitched OOS max drawdown: {summary.attrs.get('stitched_max_drawdown') * 100:.2f}%")
            print(f"Stitched OOS Sharpe: {summary.attrs.get('stitched_sharpe'):.3f}")
            for row in summary.itertuples(index=False):
                print(
                    f"{row.window}: selected={row.selected_candidate}, "
                    f"test_return={row.test_total_return * 100:.2f}%, "
                    f"test_drawdown={row.test_max_drawdown * 100:.2f}%, "
                    f"test_sharpe={row.test_sharpe:.3f}"
                )
            return 0

    parser.error("Unknown command")
    return 2
