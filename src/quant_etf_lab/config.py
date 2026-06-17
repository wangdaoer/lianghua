"""Configuration loading for the ETF lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .market_cap import DEFAULT_STOCK_MARKET_CAP_PATH


@dataclass(frozen=True)
class ETFSpec:
    code: str
    name: str
    asset_type: str = "etf"


@dataclass(frozen=True)
class UniverseSourceConfig:
    type: str
    symbol: str
    asset_type: str
    limit: int | None = None


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    initial_cash: float
    data_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class DataConfig:
    start_date: str
    end_date: str | None
    period: str
    adjust: str


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    fast_window: int
    slow_window: int
    trend_window: int
    rsi_window: int
    rsi_oversold: float
    rsi_recover: float
    rsi_exit: float
    max_positions: int
    allocation_buffer: float
    lot_size: int = 100
    min_score: float = 60.0
    min_hold_days: int = 0
    score_weighted_allocation: bool = False
    score_allocation_method: str = "proportional"
    score_allocation_tilt_strength: float = 0.5
    score_allocation_min_multiplier: float = 0.5
    score_allocation_max_multiplier: float = 1.5
    volatility_stop_loss_enabled: bool = False
    volatility_stop_window: int = 20
    volatility_stop_multiplier: float = 3.0
    volatility_stop_min_pct: float = 0.05
    volatility_stop_max_pct: float = 0.18
    time_stop_enabled: bool = False
    time_stop_min_hold_days: int = 20
    time_stop_return_threshold_pct: float = 0.0
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    prefer_existing_positions: bool = False
    signal_min_history: int = 0
    rebalance_loss_guard_pct: float | None = None
    rebalance_loss_guard_max_days: int = 0
    market_breadth_enabled: bool = False
    market_breadth_window: int = 120
    market_breadth_min_ratio: float = 0.0
    market_breadth_min_count: int = 1
    market_breadth_exposure_enabled: bool = False
    market_breadth_weak_ratio: float = 0.30
    market_breadth_weak_exposure: float = 0.50
    cross_sectional_score_enabled: bool = False
    cross_sectional_score_weights: dict[str, float] = field(default_factory=dict)
    loss_cooldown_days: int = 0
    loss_cooldown_min_losses: int = 1
    factor_momentum_window: int = 60
    factor_reversal_window: int = 5
    factor_volatility_window: int = 20
    factor_liquidity_window: int = 20
    factor_trend_window: int = 120
    factor_convergence_windows: tuple[int, ...] = (5, 10, 20, 60, 120)
    factor_volume_price_window: int = 20
    factor_shadow_window: int = 20
    factor_chip_reversal_drawdown_window: int = 20
    factor_chip_reversal_cost_window: int = 20
    factor_chip_reversal_min_drawdown_pct: float = 12.0
    factor_chip_reversal_min_score: float = 0.08
    factor_min_history: int = 120
    factor_rebalance_interval: int = 5
    factor_weights: dict[str, float] = field(default_factory=dict)
    factor_entry_filter_enabled: bool = False
    factor_entry_trend_window: int = 120
    factor_entry_min_trend: float = 0.0
    factor_entry_momentum_window: int = 20
    factor_entry_min_momentum: float = -0.05
    momentum_focus_enabled: bool = False
    momentum_focus_board_scope: str = "main_chinext"
    momentum_focus_threshold_pct: float = 7.0
    momentum_focus_limit_up_boost: float = 0.0
    momentum_focus_strong_gain_boost: float = 0.0
    momentum_focus_only: bool = False


@dataclass(frozen=True)
class CostsConfig:
    commission_rate: float
    slippage_rate: float
    stamp_tax_rate: float = 0.0


@dataclass(frozen=True)
class RiskConfig:
    enabled: bool = False
    benchmark_code: str = "510300"
    benchmark_name: str = "CSI300 ETF"
    benchmark_asset_type: str = "etf"
    benchmark_ma_window: int = 120
    benchmark_drop_window: int = 20
    benchmark_drop_threshold: float = -0.08
    benchmark_off_exposure: float = 0.5
    benchmark_crash_exposure: float = 0.0
    benchmark_rsrs_enabled: bool = False
    benchmark_rsrs_window: int = 18
    benchmark_rsrs_zscore_window: int = 600
    benchmark_rsrs_threshold: float = -0.7
    benchmark_rsrs_off_exposure: float = 0.35
    drawdown_levels: tuple[tuple[float, float], ...] = ((0.10, 0.70), (0.20, 0.40), (0.30, 0.0))
    protection_drawdown: float = 0.30
    recovery_drawdown: float = 0.15


@dataclass(frozen=True)
class LabConfig:
    project_root: Path
    project: ProjectConfig
    data: DataConfig
    universe: tuple[ETFSpec, ...]
    strategy: StrategyConfig
    costs: CostsConfig
    universe_file: Path | None = None
    universe_source: UniverseSourceConfig | None = None
    stock_market_cap_path: Path | None = None
    stock_tracking_max_market_cap_yi: float = 1500.0
    risk: RiskConfig = RiskConfig()


def _as_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _project_root_from_config(config_path: Path) -> Path:
    parent = config_path.resolve().parent
    for candidate in (parent, *parent.parents):
        if candidate.name.lower() == "configs":
            return candidate.parent
    return Path.cwd().resolve()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return raw


def _extends_values(raw: dict[str, Any], config_path: Path) -> list[Path]:
    has_extends = raw.get("extends") is not None
    has_base_config = raw.get("base_config") is not None
    if has_extends and has_base_config:
        raise ValueError(f"Config file cannot define both extends and base_config: {config_path}")
    value = raw.get("extends") if has_extends else raw.get("base_config")
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(f"Config extends must be a string or list: {config_path}")
    paths: list[Path] = []
    for item in values:
        item_path = Path(str(item))
        if not item_path.is_absolute():
            item_path = config_path.parent / item_path
        paths.append(item_path.resolve())
    return paths


def _deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in {"extends", "base_config"}:
            continue
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_config(current, value)
        else:
            merged[key] = value
    return merged


def load_config_mapping(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if config_path in _seen:
        chain = " -> ".join(str(item) for item in (*_seen, config_path))
        raise ValueError(f"Circular config extends detected: {chain}")
    raw = _load_yaml_mapping(config_path)
    merged: dict[str, Any] = {}
    for parent_path in _extends_values(raw, config_path):
        parent_raw = load_config_mapping(parent_path, (*_seen, config_path))
        merged = _deep_merge_config(merged, parent_raw)
    return _deep_merge_config(merged, raw)


def _int_tuple(raw: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    if isinstance(raw, str):
        values = raw.split(",")
    elif isinstance(raw, (int, float)):
        values = [raw]
    else:
        values = raw
    return tuple(max(int(value), 1) for value in values)


def load_config(path: str | Path) -> LabConfig:
    config_path = Path(path).resolve()
    project_root = _project_root_from_config(config_path)
    raw = load_config_mapping(config_path)
    return parse_config(raw, project_root)


def parse_config(raw: dict[str, Any], project_root: Path) -> LabConfig:
    project_raw = raw.get("project", {})
    data_raw = raw.get("data", {})
    strategy_raw = raw.get("strategy", {})
    costs_raw = raw.get("costs", {})
    risk_raw = raw.get("risk", {})
    stock_cap_path_raw = raw.get("stock_market_cap_path")
    stock_tracking_max_market_cap_yi = float(raw.get("stock_tracking_max_market_cap_yi", 1500.0))

    project = ProjectConfig(
        name=str(project_raw.get("name", "etf_backtest")),
        initial_cash=float(project_raw.get("initial_cash", 1_000_000)),
        data_dir=_as_path(project_root, str(project_raw.get("data_dir", "data"))),
        output_dir=_as_path(project_root, str(project_raw.get("output_dir", "outputs/backtests"))),
    )
    data = DataConfig(
        start_date=str(data_raw.get("start_date", "20180101")),
        end_date=data_raw.get("end_date"),
        period=str(data_raw.get("period", "daily")),
        adjust=str(data_raw.get("adjust", "qfq")),
    )
    universe = tuple(
        ETFSpec(
            code=str(item["code"]).zfill(6),
            name=str(item.get("name", item["code"])),
            asset_type=str(item.get("asset_type", "etf")).lower(),
        )
        for item in raw.get("universe", [])
    )
    universe_source_raw = raw.get("universe_source")
    universe_source = None
    if universe_source_raw:
        universe_source = UniverseSourceConfig(
            type=str(universe_source_raw.get("type", "csindex")).lower(),
            symbol=str(universe_source_raw.get("symbol", "000300")),
            asset_type=str(universe_source_raw.get("asset_type", "stock")).lower(),
            limit=(
                int(universe_source_raw["limit"])
                if universe_source_raw.get("limit") is not None
                else None
            ),
        )
    universe_file = raw.get("universe_file")
    universe_file_path = _as_path(project_root, str(universe_file)) if universe_file else None
    if not universe and universe_source is None and universe_file_path is None:
        raise ValueError("Config must define universe, universe_file, or universe_source.")
    if stock_cap_path_raw is None:
        stock_market_cap_path = project_root / DEFAULT_STOCK_MARKET_CAP_PATH
    else:
        stock_market_cap_path = _as_path(project_root, str(stock_cap_path_raw))

    strategy = StrategyConfig(
        name=str(strategy_raw.get("name", "trend")),
        fast_window=int(strategy_raw.get("fast_window", 20)),
        slow_window=int(strategy_raw.get("slow_window", 60)),
        trend_window=int(strategy_raw.get("trend_window", 120)),
        rsi_window=int(strategy_raw.get("rsi_window", 14)),
        rsi_oversold=float(strategy_raw.get("rsi_oversold", 35)),
        rsi_recover=float(strategy_raw.get("rsi_recover", 40)),
        rsi_exit=float(strategy_raw.get("rsi_exit", 65)),
        max_positions=int(strategy_raw.get("max_positions", 3)),
        allocation_buffer=float(strategy_raw.get("allocation_buffer", 0.995)),
        lot_size=int(strategy_raw.get("lot_size", 100)),
        min_score=float(strategy_raw.get("min_score", 60)),
        min_hold_days=max(int(strategy_raw.get("min_hold_days", 0)), 0),
        score_weighted_allocation=bool(strategy_raw.get("score_weighted_allocation", False)),
        score_allocation_method=str(strategy_raw.get("score_allocation_method", "proportional")).lower(),
        score_allocation_tilt_strength=max(float(strategy_raw.get("score_allocation_tilt_strength", 0.5)), 0.0),
        score_allocation_min_multiplier=max(float(strategy_raw.get("score_allocation_min_multiplier", 0.5)), 0.0),
        score_allocation_max_multiplier=max(float(strategy_raw.get("score_allocation_max_multiplier", 1.5)), 0.0),
        volatility_stop_loss_enabled=bool(strategy_raw.get("volatility_stop_loss_enabled", False)),
        volatility_stop_window=max(int(strategy_raw.get("volatility_stop_window", 20)), 2),
        volatility_stop_multiplier=max(float(strategy_raw.get("volatility_stop_multiplier", 3.0)), 0.0),
        volatility_stop_min_pct=max(float(strategy_raw.get("volatility_stop_min_pct", 0.05)), 0.0),
        volatility_stop_max_pct=max(float(strategy_raw.get("volatility_stop_max_pct", 0.18)), 0.0),
        time_stop_enabled=bool(strategy_raw.get("time_stop_enabled", False)),
        time_stop_min_hold_days=max(int(strategy_raw.get("time_stop_min_hold_days", 20)), 1),
        time_stop_return_threshold_pct=float(strategy_raw.get("time_stop_return_threshold_pct", 0.0)),
        stop_loss_pct=(
            float(strategy_raw["stop_loss_pct"])
            if strategy_raw.get("stop_loss_pct") is not None
            else None
        ),
        take_profit_pct=(
            float(strategy_raw["take_profit_pct"])
            if strategy_raw.get("take_profit_pct") is not None
            else None
        ),
        trailing_stop_pct=(
            float(strategy_raw["trailing_stop_pct"])
            if strategy_raw.get("trailing_stop_pct") is not None
            else None
        ),
        prefer_existing_positions=bool(strategy_raw.get("prefer_existing_positions", False)),
        signal_min_history=max(int(strategy_raw.get("signal_min_history", 0)), 0),
        rebalance_loss_guard_pct=(
            float(strategy_raw["rebalance_loss_guard_pct"])
            if strategy_raw.get("rebalance_loss_guard_pct") is not None
            else None
        ),
        rebalance_loss_guard_max_days=max(int(strategy_raw.get("rebalance_loss_guard_max_days", 0)), 0),
        market_breadth_enabled=bool(strategy_raw.get("market_breadth_enabled", False)),
        market_breadth_window=max(int(strategy_raw.get("market_breadth_window", 120)), 1),
        market_breadth_min_ratio=float(strategy_raw.get("market_breadth_min_ratio", 0.0)),
        market_breadth_min_count=max(int(strategy_raw.get("market_breadth_min_count", 1)), 1),
        market_breadth_exposure_enabled=bool(strategy_raw.get("market_breadth_exposure_enabled", False)),
        market_breadth_weak_ratio=float(strategy_raw.get("market_breadth_weak_ratio", 0.30)),
        market_breadth_weak_exposure=float(strategy_raw.get("market_breadth_weak_exposure", 0.50)),
        cross_sectional_score_enabled=bool(strategy_raw.get("cross_sectional_score_enabled", False)),
        cross_sectional_score_weights={
            str(name): float(weight)
            for name, weight in strategy_raw.get(
                "cross_sectional_score_weights",
                {
                    "signal": 0.60,
                    "momentum": 0.20,
                    "trend": 0.15,
                    "liquidity": 0.05,
                    "volatility": 0.00,
                },
            ).items()
        },
        loss_cooldown_days=max(int(strategy_raw.get("loss_cooldown_days", 0)), 0),
        loss_cooldown_min_losses=max(int(strategy_raw.get("loss_cooldown_min_losses", 1)), 1),
        factor_momentum_window=int(strategy_raw.get("factor_momentum_window", 60)),
        factor_reversal_window=int(strategy_raw.get("factor_reversal_window", 5)),
        factor_volatility_window=int(strategy_raw.get("factor_volatility_window", 20)),
        factor_liquidity_window=int(strategy_raw.get("factor_liquidity_window", 20)),
        factor_trend_window=int(strategy_raw.get("factor_trend_window", 120)),
        factor_convergence_windows=_int_tuple(
            strategy_raw.get("factor_convergence_windows"),
            (5, 10, 20, 60, 120),
        ),
        factor_volume_price_window=max(int(strategy_raw.get("factor_volume_price_window", 20)), 1),
        factor_shadow_window=max(int(strategy_raw.get("factor_shadow_window", 20)), 1),
        factor_chip_reversal_drawdown_window=max(
            int(strategy_raw.get("factor_chip_reversal_drawdown_window", 20)),
            2,
        ),
        factor_chip_reversal_cost_window=max(int(strategy_raw.get("factor_chip_reversal_cost_window", 20)), 2),
        factor_chip_reversal_min_drawdown_pct=float(strategy_raw.get("factor_chip_reversal_min_drawdown_pct", 12.0)),
        factor_chip_reversal_min_score=float(strategy_raw.get("factor_chip_reversal_min_score", 0.08)),
        factor_min_history=int(strategy_raw.get("factor_min_history", 120)),
        factor_rebalance_interval=max(int(strategy_raw.get("factor_rebalance_interval", 5)), 1),
        factor_weights={
            str(name): float(weight)
            for name, weight in strategy_raw.get(
                "factor_weights",
                {
                    "momentum": 0.35,
                    "trend": 0.25,
                    "reversal": 0.15,
                    "volatility": 0.15,
                    "liquidity": 0.10,
                },
            ).items()
        },
        factor_entry_filter_enabled=bool(strategy_raw.get("factor_entry_filter_enabled", False)),
        factor_entry_trend_window=max(int(strategy_raw.get("factor_entry_trend_window", 120)), 1),
        factor_entry_min_trend=float(strategy_raw.get("factor_entry_min_trend", 0.0)),
        factor_entry_momentum_window=max(int(strategy_raw.get("factor_entry_momentum_window", 20)), 1),
        factor_entry_min_momentum=float(strategy_raw.get("factor_entry_min_momentum", -0.05)),
        momentum_focus_enabled=bool(strategy_raw.get("momentum_focus_enabled", False)),
        momentum_focus_board_scope=str(strategy_raw.get("momentum_focus_board_scope", "main_chinext")).lower(),
        momentum_focus_threshold_pct=float(strategy_raw.get("momentum_focus_threshold_pct", 7.0)),
        momentum_focus_limit_up_boost=float(strategy_raw.get("momentum_focus_limit_up_boost", 0.0)),
        momentum_focus_strong_gain_boost=float(strategy_raw.get("momentum_focus_strong_gain_boost", 0.0)),
        momentum_focus_only=bool(strategy_raw.get("momentum_focus_only", False)),
    )
    costs = CostsConfig(
        commission_rate=float(costs_raw.get("commission_rate", 0.0003)),
        slippage_rate=float(costs_raw.get("slippage_rate", 0.0005)),
        stamp_tax_rate=float(costs_raw.get("stamp_tax_rate", 0.0)),
    )
    drawdown_levels_raw = risk_raw.get(
        "drawdown_levels",
        [
            {"drawdown": 0.10, "exposure": 0.70},
            {"drawdown": 0.20, "exposure": 0.40},
            {"drawdown": 0.30, "exposure": 0.0},
        ],
    )
    drawdown_levels = tuple(
        sorted(
            (
                (float(item["drawdown"]), float(item["exposure"]))
                for item in drawdown_levels_raw
            ),
            key=lambda item: item[0],
        )
    )
    risk = RiskConfig(
        enabled=bool(risk_raw.get("enabled", False)),
        benchmark_code=str(risk_raw.get("benchmark_code", "510300")).zfill(6),
        benchmark_name=str(risk_raw.get("benchmark_name", "CSI300 ETF")),
        benchmark_asset_type=str(risk_raw.get("benchmark_asset_type", "etf")).lower(),
        benchmark_ma_window=int(risk_raw.get("benchmark_ma_window", 120)),
        benchmark_drop_window=int(risk_raw.get("benchmark_drop_window", 20)),
        benchmark_drop_threshold=float(risk_raw.get("benchmark_drop_threshold", -0.08)),
        benchmark_off_exposure=float(risk_raw.get("benchmark_off_exposure", 0.5)),
        benchmark_crash_exposure=float(risk_raw.get("benchmark_crash_exposure", 0.0)),
        benchmark_rsrs_enabled=bool(risk_raw.get("benchmark_rsrs_enabled", False)),
        benchmark_rsrs_window=max(int(risk_raw.get("benchmark_rsrs_window", 18)), 2),
        benchmark_rsrs_zscore_window=max(int(risk_raw.get("benchmark_rsrs_zscore_window", 600)), 2),
        benchmark_rsrs_threshold=float(risk_raw.get("benchmark_rsrs_threshold", -0.7)),
        benchmark_rsrs_off_exposure=float(risk_raw.get("benchmark_rsrs_off_exposure", 0.35)),
        drawdown_levels=drawdown_levels,
        protection_drawdown=float(risk_raw.get("protection_drawdown", 0.30)),
        recovery_drawdown=float(risk_raw.get("recovery_drawdown", 0.15)),
    )
    return LabConfig(
        project_root=project_root,
        project=project,
        data=data,
        universe=universe,
        strategy=strategy,
        costs=costs,
        universe_file=universe_file_path,
        universe_source=universe_source,
        stock_market_cap_path=stock_market_cap_path,
        stock_tracking_max_market_cap_yi=stock_tracking_max_market_cap_yi,
        risk=risk,
    )
