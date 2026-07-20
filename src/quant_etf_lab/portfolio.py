"""Portfolio-level curve allocation tools."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .config import load_config_mapping


@dataclass(frozen=True)
class CurveConfig:
    name: str
    path: Path
    equity_column: str


@dataclass(frozen=True)
class SatelliteFilterConfig:
    enabled: bool = False
    ma_window: int = 60
    momentum_window: int = 20
    min_momentum: float = 0.0
    max_drawdown: float = 0.15
    reduced_drawdown: float = 0.08
    reduced_scale: float = 0.5
    default_scale: float = 0.0
    require_above_ma: bool = True
    reallocate_to: str = "core"


@dataclass(frozen=True)
class PortfolioConfig:
    project_root: Path
    name: str
    initial_cash: float
    output_dir: Path
    core: CurveConfig
    satellite: CurveConfig
    benchmark_path: Path
    benchmark_close_column: str
    ma_window: int
    drop_window: int
    risk_on_drop_threshold: float
    crash_drop_threshold: float
    default_regime: str
    weights: dict[str, dict[str, float]]
    satellite_filter: SatelliteFilterConfig = SatelliteFilterConfig()
    start_date: str | None = None
    end_date: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class PortfolioResult:
    run_id: str
    run_dir: Path
    equity: pd.DataFrame
    allocation_events: pd.DataFrame
    monthly_returns: pd.DataFrame
    metrics: dict[str, Any]


@dataclass(frozen=True)
class PortfolioCandidate:
    name: str
    weights: dict[str, dict[str, float]]
    satellite_filter: SatelliteFilterConfig
    regime_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioSourceCandidate:
    name: str
    source_name: str
    satellite: CurveConfig | None
    allocation: PortfolioCandidate


@dataclass(frozen=True)
class PortfolioWindow:
    name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def _project_root_from_config(config_path: Path) -> Path:
    parent = config_path.resolve().parent
    if parent.name.lower() == "configs":
        return parent.parent
    return parent


def _as_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _as_date(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    if len(text) == 8 and text.isdigit():
        return pd.to_datetime(text, format="%Y%m%d")
    return pd.to_datetime(text)


def _yyyymmdd(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _read_curve(spec: CurveConfig) -> pd.DataFrame:
    if not spec.path.exists():
        raise FileNotFoundError(f"Curve file not found: {spec.path}")
    frame = pd.read_csv(spec.path)
    missing = {"date", spec.equity_column} - set(frame.columns)
    if missing:
        raise ValueError(f"{spec.name} curve missing columns: {sorted(missing)}")
    data = frame[["date", spec.equity_column]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data[spec.name] = pd.to_numeric(data[spec.equity_column], errors="coerce")
    data = data.dropna(subset=["date", spec.name])
    data = data[data[spec.name] > 0]
    data = data.drop_duplicates(subset=["date"]).sort_values("date")
    return data[["date", spec.name]].reset_index(drop=True)


def _read_benchmark(path: Path, close_column: str, dates: pd.Series) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    frame = pd.read_csv(path)
    missing = {"date", close_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Benchmark file missing columns: {sorted(missing)}")
    data = frame[["date", close_column]].copy()
    data["date"] = pd.to_datetime(data["date"])
    data["benchmark_close"] = pd.to_numeric(data[close_column], errors="coerce")
    data = data.dropna(subset=["date", "benchmark_close"])
    data = data.drop_duplicates(subset=["date"]).sort_values("date").set_index("date")
    target_index = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    aligned = data.reindex(target_index).ffill()
    aligned.index.name = "date"
    return aligned.reset_index()[["date", "benchmark_close"]]


def _validate_weights(weights: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    normalized: dict[str, dict[str, float]] = {}
    for regime in ("risk_on", "risk_off", "crash"):
        raw = weights.get(regime, {})
        core = float(raw.get("core", 0.0))
        satellite = float(raw.get("satellite", 0.0))
        configured_cash = raw.get("cash")
        if core < 0 or satellite < 0:
            raise ValueError(f"{regime} weights cannot be negative.")
        if core + satellite > 1.0000001:
            raise ValueError(f"{regime} core + satellite weights cannot exceed 1.")
        cash = 1.0 - core - satellite
        if configured_cash is not None and abs(float(configured_cash) - cash) > 1e-6:
            raise ValueError(f"{regime} cash must equal 1 - core - satellite.")
        normalized[regime] = {"core": core, "satellite": satellite, "cash": cash}
    return normalized


def _parse_satellite_filter(raw: dict[str, Any]) -> SatelliteFilterConfig:
    reallocate_to = str(raw.get("reallocate_to", "core")).lower()
    if reallocate_to not in {"core", "cash"}:
        raise ValueError("satellite_filter.reallocate_to must be 'core' or 'cash'.")
    reduced_scale = float(raw.get("reduced_scale", 0.5))
    default_scale = float(raw.get("default_scale", 0.0))
    if not 0 <= reduced_scale <= 1:
        raise ValueError("satellite_filter.reduced_scale must be between 0 and 1.")
    if not 0 <= default_scale <= 1:
        raise ValueError("satellite_filter.default_scale must be between 0 and 1.")
    return SatelliteFilterConfig(
        enabled=bool(raw.get("enabled", False)),
        ma_window=max(int(raw.get("ma_window", 60)), 1),
        momentum_window=max(int(raw.get("momentum_window", 20)), 1),
        min_momentum=float(raw.get("min_momentum", 0.0)),
        max_drawdown=max(float(raw.get("max_drawdown", 0.15)), 0.0),
        reduced_drawdown=max(float(raw.get("reduced_drawdown", 0.08)), 0.0),
        reduced_scale=reduced_scale,
        default_scale=default_scale,
        require_above_ma=bool(raw.get("require_above_ma", True)),
        reallocate_to=reallocate_to,
    )


def load_portfolio_config(path: str | Path) -> PortfolioConfig:
    config_path = Path(path).resolve()
    project_root = _project_root_from_config(config_path)
    raw = load_config_mapping(config_path)

    project_raw = raw.get("project", {})
    curves_raw = raw.get("curves", {})
    regime_raw = raw.get("regime", {})
    weights_raw = raw.get("weights", {})
    data_raw = raw.get("data", {})
    satellite_filter_raw = raw.get("satellite_filter", {})

    core_raw = curves_raw.get("core", {})
    satellite_raw = curves_raw.get("satellite", {})
    if not core_raw or not satellite_raw:
        raise ValueError("Portfolio config must define curves.core and curves.satellite.")

    output_dir = _as_path(project_root, str(project_raw.get("output_dir", "outputs/portfolios")))
    benchmark_path = _as_path(project_root, str(regime_raw.get("benchmark_path", "data/processed/510300.csv")))
    return PortfolioConfig(
        project_root=project_root,
        name=str(project_raw.get("name", "portfolio_core_satellite")),
        initial_cash=float(project_raw.get("initial_cash", 1_000_000)),
        output_dir=output_dir,
        core=CurveConfig(
            name="core",
            path=_as_path(project_root, str(core_raw["path"])),
            equity_column=str(core_raw.get("equity_column", "equity")),
        ),
        satellite=CurveConfig(
            name="satellite",
            path=_as_path(project_root, str(satellite_raw["path"])),
            equity_column=str(satellite_raw.get("equity_column", "equity")),
        ),
        benchmark_path=benchmark_path,
        benchmark_close_column=str(regime_raw.get("benchmark_close_column", "close")),
        ma_window=max(int(regime_raw.get("ma_window", 120)), 1),
        drop_window=max(int(regime_raw.get("drop_window", 20)), 1),
        risk_on_drop_threshold=float(regime_raw.get("risk_on_drop_threshold", -0.05)),
        crash_drop_threshold=float(regime_raw.get("crash_drop_threshold", -0.08)),
        default_regime=str(regime_raw.get("default_regime", "risk_off")),
        weights=_validate_weights(weights_raw),
        satellite_filter=_parse_satellite_filter(satellite_filter_raw),
        start_date=data_raw.get("start_date"),
        end_date=data_raw.get("end_date"),
        notes=str(raw.get("notes", "")),
    )


def load_portfolio_source_selection_config(path: str | Path) -> tuple[PortfolioConfig, dict[str, CurveConfig]]:
    config_path = Path(path).resolve()
    project_root = _project_root_from_config(config_path)
    raw = load_config_mapping(config_path)
    config = load_portfolio_config(config_path)
    sources_raw = raw.get("satellite_sources") or raw.get("curves", {}).get("satellite_sources", {})
    if not sources_raw:
        raise ValueError("Portfolio source-selection config must define satellite_sources.")
    sources: dict[str, CurveConfig] = {}
    for name, source_raw in sources_raw.items():
        if not isinstance(source_raw, dict):
            raise ValueError(f"satellite_sources.{name} must be a mapping.")
        if "path" not in source_raw:
            raise ValueError(f"satellite_sources.{name} must define path.")
        sources[str(name)] = CurveConfig(
            name="satellite",
            path=_as_path(project_root, str(source_raw["path"])),
            equity_column=str(source_raw.get("equity_column", "equity")),
        )
    return config, sources


def _filter_dates(frame: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    data = frame.copy()
    start = _as_date(start_date)
    end = _as_date(end_date)
    if start is not None:
        data = data[data["date"] >= start]
    if end is not None:
        data = data[data["date"] <= end]
    return data.sort_values("date").reset_index(drop=True)


def _build_regime_frame(config: PortfolioConfig, dates: pd.Series) -> pd.DataFrame:
    benchmark = _read_benchmark(config.benchmark_path, config.benchmark_close_column, dates)
    close = benchmark["benchmark_close"]
    ma = close.rolling(config.ma_window, min_periods=config.ma_window).mean()
    drop = close.pct_change(config.drop_window)

    raw_regime: list[str] = []
    raw_reason: list[str] = []
    for close_value, ma_value, drop_value in zip(close, ma, drop):
        if not np.isfinite(close_value) or not np.isfinite(ma_value) or not np.isfinite(drop_value):
            raw_regime.append(config.default_regime)
            raw_reason.append("not_enough_benchmark_history")
        elif drop_value <= config.crash_drop_threshold:
            raw_regime.append("crash")
            raw_reason.append("benchmark_crash_drop")
        elif close_value >= ma_value and drop_value > config.risk_on_drop_threshold:
            raw_regime.append("risk_on")
            raw_reason.append("benchmark_above_ma")
        elif drop_value <= config.risk_on_drop_threshold:
            raw_regime.append("risk_off")
            raw_reason.append("benchmark_weak_drop")
        else:
            raw_regime.append("risk_off")
            raw_reason.append("benchmark_below_ma")

    regime = pd.DataFrame(
        {
            "date": benchmark["date"],
            "benchmark_close": close,
            "benchmark_ma": ma,
            "benchmark_drop": drop,
            "signal_regime": raw_regime,
            "signal_reason": raw_reason,
        }
    )
    regime["effective_regime"] = regime["signal_regime"].shift(1).fillna(config.default_regime)
    regime["effective_reason"] = regime["signal_reason"].shift(1).fillna("initial_default")
    return regime


def _build_satellite_filter_frame(config: PortfolioConfig, satellite_norm: pd.Series, dates: pd.Series) -> pd.DataFrame:
    if not config.satellite_filter.enabled:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(dates),
                "satellite_signal_scale": 1.0,
                "satellite_signal_reason": "filter_disabled",
                "satellite_effective_scale": 1.0,
                "satellite_effective_reason": "filter_disabled",
                "satellite_ma": np.nan,
                "satellite_momentum": np.nan,
                "satellite_drawdown": 0.0,
            }
        )

    settings = config.satellite_filter
    series = pd.to_numeric(satellite_norm, errors="coerce")
    ma = series.rolling(settings.ma_window, min_periods=settings.ma_window).mean()
    momentum = series.pct_change(settings.momentum_window)
    drawdown = series / series.cummax() - 1.0
    signal_scale: list[float] = []
    signal_reason: list[str] = []
    for value, ma_value, momentum_value, drawdown_value in zip(series, ma, momentum, drawdown):
        has_required_history = np.isfinite(value) and np.isfinite(momentum_value)
        if settings.require_above_ma:
            has_required_history = has_required_history and np.isfinite(ma_value)
        if not has_required_history:
            signal_scale.append(settings.default_scale)
            signal_reason.append("not_enough_satellite_history")
        elif abs(float(drawdown_value)) >= settings.max_drawdown:
            signal_scale.append(0.0)
            signal_reason.append("satellite_drawdown_stop")
        elif settings.require_above_ma and float(value) < float(ma_value):
            signal_scale.append(0.0)
            signal_reason.append("satellite_below_ma")
        elif float(momentum_value) < settings.min_momentum:
            signal_scale.append(0.0)
            signal_reason.append("satellite_weak_momentum")
        elif abs(float(drawdown_value)) >= settings.reduced_drawdown:
            signal_scale.append(settings.reduced_scale)
            signal_reason.append("satellite_drawdown_reduce")
        else:
            signal_scale.append(1.0)
            signal_reason.append("satellite_healthy")

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "satellite_signal_scale": signal_scale,
            "satellite_signal_reason": signal_reason,
            "satellite_ma": ma,
            "satellite_momentum": momentum,
            "satellite_drawdown": drawdown,
        }
    )
    frame["satellite_effective_scale"] = frame["satellite_signal_scale"].shift(1).fillna(settings.default_scale)
    frame["satellite_effective_reason"] = frame["satellite_signal_reason"].shift(1).fillna("initial_default")
    return frame


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    running_max = series.cummax()
    return float((series / running_max - 1).min())


def _curve_stats(series: pd.Series, initial_cash: float) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"final_equity": initial_cash, "total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
    returns = clean.pct_change().dropna()
    final_equity = float(clean.iloc[-1])
    total_return = final_equity / initial_cash - 1.0
    days = max((clean.index[-1] - clean.index[0]).days, 1) if len(clean.index) > 1 else 1
    years = days / 365.25
    cagr = (final_equity / initial_cash) ** (1 / years) - 1 if years > 0 and final_equity > 0 else 0.0
    sharpe = 0.0
    if len(returns) > 1 and float(returns.std()) != 0:
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std())
    return {
        "final_equity": final_equity,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": _max_drawdown(clean),
        "sharpe": float(sharpe),
    }


def _monthly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    series = equity.set_index("date")["portfolio_equity"]
    try:
        monthly_equity = series.resample("ME").last()
    except ValueError:
        monthly_equity = series.resample("M").last()
    monthly = monthly_equity.pct_change().dropna().reset_index()
    monthly.columns = ["date", "monthly_return"]
    return monthly


def _allocation_events(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame(columns=["date", "from_regime", "to_regime", "core_weight", "satellite_weight", "cash_weight", "reason"])
    events: list[dict[str, Any]] = []
    previous = None
    for row in equity.itertuples(index=False):
        regime = str(row.effective_regime)
        current = (
            regime,
            round(float(row.core_weight), 8),
            round(float(row.satellite_weight), 8),
            round(float(row.cash_weight), 8),
        )
        if previous != current:
            filter_reason = getattr(row, "satellite_effective_reason", "")
            reason = str(row.effective_reason)
            if filter_reason and filter_reason not in {"filter_disabled", "initial_default"}:
                reason = f"{reason};{filter_reason}"
            events.append(
                {
                    "date": row.date,
                    "from_regime": previous[0] if previous else "",
                    "to_regime": regime,
                    "core_weight": float(row.core_weight),
                    "satellite_weight": float(row.satellite_weight),
                    "cash_weight": float(row.cash_weight),
                    "reason": reason,
                }
            )
        previous = current
    return pd.DataFrame(events)


def _compute_metrics(equity: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    indexed = equity.set_index("date")
    portfolio_stats = _curve_stats(indexed["portfolio_equity"], initial_cash)
    core_stats = _curve_stats(indexed["core_equity"], initial_cash)
    satellite_stats = _curve_stats(indexed["satellite_equity"], initial_cash)
    metrics: dict[str, Any] = {
        "initial_cash": float(initial_cash),
        **portfolio_stats,
        "core_total_return": core_stats["total_return"],
        "core_max_drawdown": core_stats["max_drawdown"],
        "core_sharpe": core_stats["sharpe"],
        "satellite_total_return": satellite_stats["total_return"],
        "satellite_max_drawdown": satellite_stats["max_drawdown"],
        "satellite_sharpe": satellite_stats["sharpe"],
        "excess_return_vs_core": portfolio_stats["total_return"] - core_stats["total_return"],
        "drawdown_change_vs_core": portfolio_stats["max_drawdown"] - core_stats["max_drawdown"],
        "observation_days": int(len(equity)),
        "average_core_weight": float(equity["core_weight"].mean()) if not equity.empty else 0.0,
        "average_satellite_weight": float(equity["satellite_weight"].mean()) if not equity.empty else 0.0,
        "average_cash_weight": float(equity["cash_weight"].mean()) if not equity.empty else 0.0,
        "satellite_active_day_ratio": float((equity["satellite_weight"] > 0).mean()) if not equity.empty else 0.0,
    }
    if "satellite_effective_scale" in equity:
        scale = pd.to_numeric(equity["satellite_effective_scale"], errors="coerce").fillna(1.0)
        metrics["average_satellite_scale"] = float(scale.mean())
        metrics["satellite_filter_off_day_ratio"] = float((scale <= 0.0).mean())
        metrics["satellite_filter_reduced_day_ratio"] = float(((scale > 0.0) & (scale < 1.0)).mean())
    for regime in ("risk_on", "risk_off", "crash"):
        metrics[f"{regime}_day_ratio"] = float((equity["effective_regime"] == regime).mean()) if not equity.empty else 0.0
    return metrics


def _portfolio_candidate_weights(satellite_weight: float) -> dict[str, dict[str, float]]:
    satellite_weight = max(min(float(satellite_weight), 1.0), 0.0)
    return {
        "risk_on": {"core": 1.0 - satellite_weight, "satellite": satellite_weight, "cash": 0.0},
        "risk_off": {"core": 1.0, "satellite": 0.0, "cash": 0.0},
        "crash": {"core": 1.0, "satellite": 0.0, "cash": 0.0},
    }


def build_portfolio_candidate_grid(grid: str = "guarded") -> tuple[PortfolioCandidate, ...]:
    disabled = SatelliteFilterConfig(enabled=False)
    if grid not in {"compact", "guarded", "activation", "activation_dd50"}:
        raise ValueError("grid must be 'compact', 'guarded', 'activation', or 'activation_dd50'.")

    candidates = [
        PortfolioCandidate("core_only", _portfolio_candidate_weights(0.0), disabled),
        PortfolioCandidate("sat10_unfiltered", _portfolio_candidate_weights(0.10), disabled),
        PortfolioCandidate("sat20_unfiltered", _portfolio_candidate_weights(0.20), disabled),
        PortfolioCandidate("sat30_unfiltered", _portfolio_candidate_weights(0.30), disabled),
    ]
    if grid == "compact":
        return tuple(candidates)

    filters = {
        "health_mom10_dd20": SatelliteFilterConfig(
            enabled=True,
            ma_window=60,
            momentum_window=10,
            min_momentum=-0.08,
            max_drawdown=0.20,
            reduced_drawdown=0.12,
            reduced_scale=0.75,
            default_scale=0.0,
            require_above_ma=False,
            reallocate_to="core",
        ),
        "health_mom20_dd20": SatelliteFilterConfig(
            enabled=True,
            ma_window=60,
            momentum_window=20,
            min_momentum=-0.05,
            max_drawdown=0.20,
            reduced_drawdown=0.12,
            reduced_scale=0.75,
            default_scale=0.0,
            require_above_ma=False,
            reallocate_to="core",
        ),
        "health_mom40_dd20": SatelliteFilterConfig(
            enabled=True,
            ma_window=60,
            momentum_window=40,
            min_momentum=-0.08,
            max_drawdown=0.20,
            reduced_drawdown=0.12,
            reduced_scale=0.75,
            default_scale=0.0,
            require_above_ma=False,
            reallocate_to="core",
        ),
        "health_ma60_mom20_dd15": SatelliteFilterConfig(
            enabled=True,
            ma_window=60,
            momentum_window=20,
            min_momentum=0.03,
            max_drawdown=0.15,
            reduced_drawdown=0.09,
            reduced_scale=0.50,
            default_scale=0.0,
            require_above_ma=True,
            reallocate_to="core",
        ),
    }
    for satellite_weight in (0.15, 0.20, 0.25, 0.30, 0.35):
        for filter_name, satellite_filter in filters.items():
            candidates.append(
                PortfolioCandidate(
                    f"sat{int(satellite_weight * 100):02d}_{filter_name}",
                    _portfolio_candidate_weights(satellite_weight),
                    satellite_filter,
                )
            )
    if grid == "activation":
        activation_candidates = [PortfolioCandidate("core_only", _portfolio_candidate_weights(0.0), disabled)]
        regime_profiles = {
            "ma60_drop03": {"ma_window": 60, "risk_on_drop_threshold": -0.03, "crash_drop_threshold": -0.07},
            "ma120_drop03": {"ma_window": 120, "risk_on_drop_threshold": -0.03, "crash_drop_threshold": -0.07},
            "ma120_drop05": {"ma_window": 120, "risk_on_drop_threshold": -0.05, "crash_drop_threshold": -0.08},
            "ma200_drop05": {"ma_window": 200, "risk_on_drop_threshold": -0.05, "crash_drop_threshold": -0.08},
        }
        activation_filters = {
            "unfiltered": disabled,
            "health_mom10_dd20": filters["health_mom10_dd20"],
            "health_ma60_mom20_dd15": filters["health_ma60_mom20_dd15"],
        }
        for regime_name, regime_overrides in regime_profiles.items():
            for satellite_weight in (0.10, 0.15, 0.20, 0.25):
                for filter_name, satellite_filter in activation_filters.items():
                    activation_candidates.append(
                        PortfolioCandidate(
                            f"sat{int(satellite_weight * 100):02d}_{regime_name}_{filter_name}",
                            _portfolio_candidate_weights(satellite_weight),
                            satellite_filter,
                            dict(regime_overrides),
                        )
                    )
        return tuple(activation_candidates)
    if grid == "activation_dd50":
        dd50_candidates = [PortfolioCandidate("core_only", _portfolio_candidate_weights(0.0), disabled)]
        regime_profiles = {
            "ma60_drop03": {"ma_window": 60, "risk_on_drop_threshold": -0.03, "crash_drop_threshold": -0.07},
            "ma120_drop03": {"ma_window": 120, "risk_on_drop_threshold": -0.03, "crash_drop_threshold": -0.07},
            "ma120_drop05": {"ma_window": 120, "risk_on_drop_threshold": -0.05, "crash_drop_threshold": -0.08},
            "ma200_drop05": {"ma_window": 200, "risk_on_drop_threshold": -0.05, "crash_drop_threshold": -0.08},
        }
        activation_filters = {
            "unfiltered": disabled,
            "health_mom10_dd20": filters["health_mom10_dd20"],
            "health_ma60_mom20_dd15": filters["health_ma60_mom20_dd15"],
        }
        for regime_name, regime_overrides in regime_profiles.items():
            for satellite_weight in (0.30, 0.35, 0.50):
                for filter_name, satellite_filter in activation_filters.items():
                    dd50_candidates.append(
                        PortfolioCandidate(
                            f"sat{int(satellite_weight * 100):02d}_{regime_name}_{filter_name}",
                            _portfolio_candidate_weights(satellite_weight),
                            satellite_filter,
                            dict(regime_overrides),
                        )
                    )
        return tuple(dd50_candidates)
    return tuple(candidates)


def _candidate_satellite_weight(candidate: PortfolioCandidate) -> float:
    risk_on = candidate.weights.get("risk_on", {})
    return float(risk_on.get("satellite", 0.0) or 0.0)


def build_portfolio_source_candidate_grid(
    satellite_sources: dict[str, CurveConfig],
    grid: str = "activation",
    max_satellite_weight: float | None = None,
    source_max_satellite_weight: dict[str, float] | None = None,
) -> tuple[PortfolioSourceCandidate, ...]:
    if not satellite_sources:
        raise ValueError("At least one satellite source is required.")
    if max_satellite_weight is not None and float(max_satellite_weight) < 0.0:
        raise ValueError("max_satellite_weight cannot be negative.")
    source_caps = {str(key): float(value) for key, value in (source_max_satellite_weight or {}).items()}
    negative_sources = [source for source, cap in source_caps.items() if cap < 0.0]
    if negative_sources:
        raise ValueError(f"source_max_satellite_weight cannot be negative: {sorted(negative_sources)}")
    allocation_candidates = build_portfolio_candidate_grid(grid)
    candidates: list[PortfolioSourceCandidate] = []
    for allocation in allocation_candidates:
        if _candidate_satellite_weight(allocation) <= 0.0:
            candidates.append(
                PortfolioSourceCandidate(
                    name="core_only",
                    source_name="core_only",
                    satellite=None,
                    allocation=allocation,
                )
            )
            break
    if not candidates:
        disabled = SatelliteFilterConfig(enabled=False)
        candidates.append(
            PortfolioSourceCandidate(
                name="core_only",
                source_name="core_only",
                satellite=None,
                allocation=PortfolioCandidate("core_only", _portfolio_candidate_weights(0.0), disabled),
            )
        )
    for source_name, source in satellite_sources.items():
        for allocation in allocation_candidates:
            satellite_weight = _candidate_satellite_weight(allocation)
            if satellite_weight <= 0.0:
                continue
            source_cap = source_caps.get(str(source_name))
            active_cap = max_satellite_weight
            if source_cap is not None:
                active_cap = source_cap if active_cap is None else min(float(active_cap), float(source_cap))
            if active_cap is not None and satellite_weight > float(active_cap) + 1e-12:
                continue
            candidates.append(
                PortfolioSourceCandidate(
                    name=f"{source_name}__{allocation.name}",
                    source_name=source_name,
                    satellite=source,
                    allocation=allocation,
                )
            )
    return tuple(candidates)


def score_portfolio_metrics(
    metrics: dict[str, Any],
    max_drawdown_limit: float = 0.16,
    score_mode: str = "balanced",
) -> float:
    if score_mode not in {"balanced", "capital_efficiency"}:
        raise ValueError("score_mode must be 'balanced' or 'capital_efficiency'.")
    total_return = float(metrics.get("total_return", 0.0) or 0.0)
    cagr = float(metrics.get("cagr", 0.0) or 0.0)
    sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
    max_drawdown = abs(float(metrics.get("max_drawdown", 0.0) or 0.0))
    average_satellite_weight = float(metrics.get("average_satellite_weight", 0.0) or 0.0)
    loss_penalty = max(0.0, -total_return)
    drawdown_penalty = max(0.0, max_drawdown - max_drawdown_limit)
    overuse_penalty = max(0.0, average_satellite_weight - 0.12) * 0.40
    if score_mode == "balanced":
        return (
            0.55 * sharpe
            + 0.45 * cagr
            + 0.25 * total_return
            - 6.0 * drawdown_penalty
            - 1.0 * loss_penalty
            - overuse_penalty
        )

    excess_return = float(metrics.get("excess_return_vs_core", total_return) or 0.0)
    capital_efficiency = excess_return / max(average_satellite_weight, 0.05)
    capital_efficiency = max(min(capital_efficiency, 1.0), -1.0)
    return (
        0.45 * sharpe
        + 0.30 * cagr
        + 0.15 * total_return
        + 0.20 * capital_efficiency
        - 6.0 * drawdown_penalty
        - 1.0 * loss_penalty
        - overuse_penalty
    )


def _candidate_name(candidate: Any) -> str:
    return str(getattr(candidate, "name", ""))


def _apply_score_mode_min_edge_gate(
    selected: tuple[float, Any, PortfolioResult],
    balanced_selected: tuple[float, Any, PortfolioResult],
    active_scores_by_candidate: dict[str, float],
    score_mode: str,
    score_mode_min_edge: float = 0.0,
) -> tuple[float, Any, PortfolioResult, dict[str, Any]]:
    if score_mode_min_edge < 0:
        raise ValueError("score_mode_min_edge cannot be negative.")
    selected_score, selected_candidate, selected_result = selected
    baseline_score, baseline_candidate, baseline_result = balanced_selected
    selected_name = _candidate_name(selected_candidate)
    baseline_name = _candidate_name(baseline_candidate)
    baseline_active_score = float(active_scores_by_candidate.get(baseline_name, baseline_score))
    edge = float(selected_score) - baseline_active_score
    gate_applied = bool(
        score_mode != "balanced"
        and float(score_mode_min_edge) > 0.0
        and selected_name != baseline_name
        and edge < float(score_mode_min_edge)
    )
    payload = {
        "score_mode_min_edge": float(score_mode_min_edge),
        "score_mode_gate_applied": gate_applied,
        "score_mode_baseline_candidate": baseline_name,
        "score_mode_baseline_score": baseline_active_score,
        "score_mode_edge_vs_baseline": edge,
    }
    if gate_applied:
        return baseline_active_score, baseline_candidate, baseline_result, payload
    return float(selected_score), selected_candidate, selected_result, payload


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _write_summary(config: PortfolioConfig, result: PortfolioResult) -> Path:
    path = result.run_dir / "summary.md"
    metrics = result.metrics
    notes = f"\n## Notes\n\n{config.notes}\n" if config.notes else ""
    body = f"""# Core-Satellite Portfolio Summary

Output directory: `{result.run_dir}`

This is a research-only allocation test. It combines existing backtest equity curves and does not connect to brokers or place orders.

## Aggregate Result

| Metric | Portfolio | Core | Satellite |
| --- | ---: | ---: | ---: |
| Total return | {_format_pct(metrics.get("total_return"))} | {_format_pct(metrics.get("core_total_return"))} | {_format_pct(metrics.get("satellite_total_return"))} |
| Max drawdown | {_format_pct(metrics.get("max_drawdown"))} | {_format_pct(metrics.get("core_max_drawdown"))} | {_format_pct(metrics.get("satellite_max_drawdown"))} |
| Sharpe | {float(metrics.get("sharpe", 0.0)):.3f} | {float(metrics.get("core_sharpe", 0.0)):.3f} | {float(metrics.get("satellite_sharpe", 0.0)):.3f} |

## Allocation

| Metric | Value |
| --- | ---: |
| Excess return vs core | {_format_pct(metrics.get("excess_return_vs_core"))} |
| Drawdown change vs core | {_format_pct(metrics.get("drawdown_change_vs_core"))} |
| Average core weight | {_format_pct(metrics.get("average_core_weight"))} |
| Average satellite weight | {_format_pct(metrics.get("average_satellite_weight"))} |
| Average cash weight | {_format_pct(metrics.get("average_cash_weight"))} |
| Satellite active day ratio | {_format_pct(metrics.get("satellite_active_day_ratio"))} |
| Average satellite filter scale | {_format_pct(metrics.get("average_satellite_scale"))} |
| Satellite filter off day ratio | {_format_pct(metrics.get("satellite_filter_off_day_ratio"))} |
| Satellite filter reduced day ratio | {_format_pct(metrics.get("satellite_filter_reduced_day_ratio"))} |
| Risk-on day ratio | {_format_pct(metrics.get("risk_on_day_ratio"))} |
| Risk-off day ratio | {_format_pct(metrics.get("risk_off_day_ratio"))} |
| Crash day ratio | {_format_pct(metrics.get("crash_day_ratio"))} |

## Files

- `portfolio_equity.csv`: daily portfolio, source curve, benchmark regime, and weights.
- `allocation_events.csv`: regime and allocation changes.
- `monthly_returns.csv`: monthly portfolio returns.
- `metrics.json`: aggregate metrics.
- `config_used.yaml`: resolved run configuration.
{notes}
"""
    path.write_text(body, encoding="utf-8")
    return path


def _write_outputs(config: PortfolioConfig, result: PortfolioResult) -> None:
    result.run_dir.mkdir(parents=True, exist_ok=True)
    result.equity.to_csv(result.run_dir / "portfolio_equity.csv", index=False, encoding="utf-8")
    result.allocation_events.to_csv(result.run_dir / "allocation_events.csv", index=False, encoding="utf-8")
    result.monthly_returns.to_csv(result.run_dir / "monthly_returns.csv", index=False, encoding="utf-8")
    (result.run_dir / "metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config_dump = {
        "project": {
            "name": config.name,
            "initial_cash": config.initial_cash,
            "output_dir": str(config.output_dir),
        },
        "curves": {
            "core": {"path": str(config.core.path), "equity_column": config.core.equity_column},
            "satellite": {"path": str(config.satellite.path), "equity_column": config.satellite.equity_column},
        },
        "regime": {
            "benchmark_path": str(config.benchmark_path),
            "benchmark_close_column": config.benchmark_close_column,
            "ma_window": config.ma_window,
            "drop_window": config.drop_window,
            "risk_on_drop_threshold": config.risk_on_drop_threshold,
            "crash_drop_threshold": config.crash_drop_threshold,
            "default_regime": config.default_regime,
        },
        "weights": config.weights,
        "satellite_filter": config.satellite_filter.__dict__,
        "data": {"start_date": config.start_date, "end_date": config.end_date},
        "notes": config.notes,
    }
    (result.run_dir / "config_used.yaml").write_text(yaml.safe_dump(config_dump, sort_keys=False), encoding="utf-8")
    _write_summary(config, result)


def run_portfolio_combine(config: PortfolioConfig, run_id: str | None = None, write_outputs: bool = True) -> PortfolioResult:
    core = _read_curve(config.core)
    satellite = _read_curve(config.satellite)
    merged = core.merge(satellite, on="date", how="inner")
    merged = _filter_dates(merged, config.start_date, config.end_date)
    if merged.empty:
        raise ValueError("No overlapping dates between core and satellite curves.")

    merged["core_norm"] = merged["core"] / float(merged["core"].iloc[0])
    merged["satellite_norm"] = merged["satellite"] / float(merged["satellite"].iloc[0])
    merged["core_return"] = merged["core_norm"].pct_change().fillna(0.0)
    merged["satellite_return"] = merged["satellite_norm"].pct_change().fillna(0.0)
    merged["core_equity"] = merged["core_norm"] * config.initial_cash
    merged["satellite_equity"] = merged["satellite_norm"] * config.initial_cash

    regime = _build_regime_frame(config, merged["date"])
    satellite_filter = _build_satellite_filter_frame(config, merged["satellite_norm"], merged["date"])
    equity = merged.merge(regime, on="date", how="left").merge(satellite_filter, on="date", how="left")
    for weight_name in ("core", "satellite", "cash"):
        equity[f"{weight_name}_weight"] = equity["effective_regime"].map(
            {regime_name: values[weight_name] for regime_name, values in config.weights.items()}
        )
    equity[["core_weight", "satellite_weight", "cash_weight"]] = equity[
        ["core_weight", "satellite_weight", "cash_weight"]
    ].fillna(0.0)
    base_satellite_weight = equity["satellite_weight"].copy()
    equity["satellite_effective_scale"] = pd.to_numeric(equity["satellite_effective_scale"], errors="coerce").fillna(1.0)
    equity["satellite_weight"] = base_satellite_weight * equity["satellite_effective_scale"].clip(lower=0.0, upper=1.0)
    removed_satellite_weight = base_satellite_weight - equity["satellite_weight"]
    if config.satellite_filter.reallocate_to == "cash":
        equity["cash_weight"] = equity["cash_weight"] + removed_satellite_weight
    else:
        equity["core_weight"] = equity["core_weight"] + removed_satellite_weight
    for weight_column in ("core_weight", "satellite_weight", "cash_weight"):
        equity.loc[equity[weight_column].abs() < 1e-12, weight_column] = 0.0
    equity["portfolio_return"] = (
        equity["core_weight"] * equity["core_return"]
        + equity["satellite_weight"] * equity["satellite_return"]
    )
    equity["portfolio_equity"] = (1.0 + equity["portfolio_return"]).cumprod() * config.initial_cash
    if not equity.empty:
        equity.loc[equity.index[0], "portfolio_equity"] = config.initial_cash

    output_columns = [
        "date",
        "portfolio_equity",
        "portfolio_return",
        "core_equity",
        "core_return",
        "satellite_equity",
        "satellite_return",
        "satellite_ma",
        "satellite_momentum",
        "satellite_drawdown",
        "satellite_signal_scale",
        "satellite_signal_reason",
        "satellite_effective_scale",
        "satellite_effective_reason",
        "benchmark_close",
        "benchmark_ma",
        "benchmark_drop",
        "signal_regime",
        "signal_reason",
        "effective_regime",
        "effective_reason",
        "core_weight",
        "satellite_weight",
        "cash_weight",
    ]
    equity = equity[output_columns].copy()
    allocation_events = _allocation_events(equity)
    monthly_returns = _monthly_returns(equity)
    metrics = _compute_metrics(equity, config.initial_cash)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_run_id = run_id or f"{stamp}_{config.name}"
    result = PortfolioResult(
        run_id=resolved_run_id,
        run_dir=config.output_dir / resolved_run_id,
        equity=equity,
        allocation_events=allocation_events,
        monthly_returns=monthly_returns,
        metrics=metrics,
    )
    if write_outputs:
        _write_outputs(config, result)
    return result


def _portfolio_window_dates(config: PortfolioConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    core = _read_curve(config.core)
    satellite = _read_curve(config.satellite)
    merged = _filter_dates(core.merge(satellite, on="date", how="inner"), config.start_date, config.end_date)
    if merged.empty:
        raise ValueError("No overlapping dates between core and satellite curves.")
    return pd.Timestamp(merged["date"].min()).normalize(), pd.Timestamp(merged["date"].max()).normalize()


def generate_portfolio_windows(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
) -> tuple[PortfolioWindow, ...]:
    if train_months <= 0 or test_months <= 0 or step_months <= 0:
        raise ValueError("train_months, test_months, and step_months must be positive.")
    windows: list[PortfolioWindow] = []
    cursor = pd.Timestamp(start_date).normalize()
    index = 1
    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=train_months) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        if test_start > end_date:
            break
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), end_date)
        windows.append(
            PortfolioWindow(
                name=f"pf_{index:02d}_{_yyyymmdd(test_start)}_{_yyyymmdd(test_end)}",
                train_start=_yyyymmdd(train_start),
                train_end=_yyyymmdd(train_end),
                test_start=_yyyymmdd(test_start),
                test_end=_yyyymmdd(test_end),
            )
        )
        cursor = cursor + pd.DateOffset(months=step_months)
        index += 1
    return tuple(windows)


def _candidate_config(config: PortfolioConfig, candidate: PortfolioCandidate, start_date: str, end_date: str) -> PortfolioConfig:
    regime = candidate.regime_overrides
    return PortfolioConfig(
        project_root=config.project_root,
        name=config.name,
        initial_cash=config.initial_cash,
        output_dir=config.output_dir,
        core=config.core,
        satellite=config.satellite,
        benchmark_path=config.benchmark_path,
        benchmark_close_column=config.benchmark_close_column,
        ma_window=int(regime.get("ma_window", config.ma_window)),
        drop_window=int(regime.get("drop_window", config.drop_window)),
        risk_on_drop_threshold=float(regime.get("risk_on_drop_threshold", config.risk_on_drop_threshold)),
        crash_drop_threshold=float(regime.get("crash_drop_threshold", config.crash_drop_threshold)),
        default_regime=str(regime.get("default_regime", config.default_regime)),
        weights=candidate.weights,
        satellite_filter=candidate.satellite_filter,
        start_date=start_date,
        end_date=end_date,
        notes=config.notes,
    )


def _candidate_payload(candidate: PortfolioCandidate) -> str:
    return json.dumps(
        {
            "weights": candidate.weights,
            "satellite_filter": candidate.satellite_filter.__dict__,
            "regime_overrides": candidate.regime_overrides,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _source_candidate_config(
    config: PortfolioConfig,
    candidate: PortfolioSourceCandidate,
    start_date: str,
    end_date: str,
) -> PortfolioConfig:
    source_config = replace(config, satellite=candidate.satellite or config.satellite)
    return _candidate_config(source_config, candidate.allocation, start_date, end_date)


def _source_candidate_payload(candidate: PortfolioSourceCandidate) -> str:
    return json.dumps(
        {
            "source_name": candidate.source_name,
            "source_path": str(candidate.satellite.path) if candidate.satellite is not None else None,
            "allocation": json.loads(_candidate_payload(candidate.allocation)),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _source_validation_dates(window: PortfolioWindow, validation_months: int) -> tuple[str, str] | None:
    if validation_months <= 0:
        return None
    train_start = _as_date(window.train_start)
    train_end = _as_date(window.train_end)
    if train_start is None or train_end is None:
        raise ValueError("Portfolio source-selection windows must have train dates.")
    validation_start = train_end - pd.DateOffset(months=validation_months) + pd.Timedelta(days=1)
    validation_start = pd.Timestamp(validation_start).normalize()
    train_start = pd.Timestamp(train_start).normalize()
    train_end = pd.Timestamp(train_end).normalize()
    if validation_start <= train_start:
        raise ValueError("source_validation_months must be shorter than the training window.")
    return _yyyymmdd(validation_start), _yyyymmdd(train_end)


def _select_source_candidate(
    scored: list[tuple[float, PortfolioSourceCandidate, PortfolioResult]],
    default_source_name: str | None = None,
    source_switch_margin: float = 0.0,
    source_switch_margin_by_source: dict[str, float] | None = None,
    previous_source_name: str | None = None,
    source_stability_penalty: float = 0.0,
) -> tuple[float, PortfolioSourceCandidate, PortfolioResult, tuple[float, PortfolioSourceCandidate, PortfolioResult]]:
    scored.sort(key=lambda item: item[0], reverse=True)
    raw_best = scored[0]
    selected = raw_best
    source_margins = {str(key): float(value) for key, value in (source_switch_margin_by_source or {}).items()}
    negative_sources = [source for source, margin in source_margins.items() if margin < 0.0]
    if negative_sources:
        raise ValueError(f"source_switch_margin_by_source cannot be negative: {sorted(negative_sources)}")
    source_margin = source_margins.get(raw_best[1].source_name, 0.0)
    if source_margin > 0.0:
        alternatives = [item for item in scored if item[1].source_name != raw_best[1].source_name]
        if alternatives and raw_best[0] < alternatives[0][0] + source_margin:
            selected = alternatives[0]
    if default_source_name:
        default_candidates = [
            item for item in scored if item[1].source_name in {default_source_name, "core_only"}
        ]
        if not default_candidates:
            raise ValueError(f"default source not found in candidate grid: {default_source_name}")
        default_best = default_candidates[0]
        if selected[1].source_name not in {default_source_name, "core_only"}:
            if selected[0] < default_best[0] + float(source_switch_margin):
                return default_best[0], default_best[1], default_best[2], raw_best
    if previous_source_name and source_stability_penalty > 0:
        previous_candidates = [
            item for item in scored if item[1].source_name == previous_source_name
        ]
        if previous_candidates and selected[1].source_name not in {previous_source_name, "core_only"}:
            previous_best = previous_candidates[0]
            if selected[0] < previous_best[0] + float(source_stability_penalty):
                return previous_best[0], previous_best[1], previous_best[2], raw_best
    return selected[0], selected[1], selected[2], raw_best


def _normalize_source_groups(source_groups: dict[str, list[str]] | None) -> dict[str, tuple[str, ...]]:
    normalized: dict[str, tuple[str, ...]] = {}
    seen: dict[str, str] = {}
    for group_name, raw_sources in (source_groups or {}).items():
        group = str(group_name).strip()
        if not group:
            raise ValueError("source group name cannot be empty.")
        sources = tuple(str(source).strip() for source in raw_sources if str(source).strip())
        if len(sources) < 2:
            raise ValueError(f"source group {group} must contain at least two sources.")
        for source in sources:
            if source in seen:
                raise ValueError(f"source {source} appears in multiple source groups: {seen[source]} and {group}.")
            seen[source] = group
        normalized[group] = sources
    return normalized


def _apply_source_group_preselection(
    scored: list[tuple[float, PortfolioSourceCandidate, PortfolioResult]],
    source_groups: dict[str, list[str]] | None,
    source_switch_margin_by_source: dict[str, float] | None = None,
) -> tuple[list[tuple[float, PortfolioSourceCandidate, PortfolioResult]], dict[str, dict[str, Any]]]:
    groups = _normalize_source_groups(source_groups)
    if not groups:
        return scored, {}
    source_margins = {str(key): float(value) for key, value in (source_switch_margin_by_source or {}).items()}
    negative_sources = [source for source, margin in source_margins.items() if margin < 0.0]
    if negative_sources:
        raise ValueError(f"source_switch_margin_by_source cannot be negative: {sorted(negative_sources)}")

    source_to_group = {
        source: group_name
        for group_name, sources in groups.items()
        for source in sources
    }
    scored_by_group: dict[str, list[tuple[float, PortfolioSourceCandidate, PortfolioResult]]] = {}
    for score, candidate, result in scored:
        group = source_to_group.get(candidate.source_name)
        if group is None:
            continue
        scored_by_group.setdefault(group, []).append((score, candidate, result))

    best_by_group: dict[str, tuple[float, str]] = {}
    for group, group_scored in scored_by_group.items():
        group_scored.sort(key=lambda item: item[0], reverse=True)
        raw_best = group_scored[0]
        winner = raw_best
        source_margin = source_margins.get(raw_best[1].source_name, 0.0)
        if source_margin > 0.0:
            alternatives = [
                item
                for item in group_scored
                if item[1].source_name != raw_best[1].source_name
            ]
            if alternatives and float(raw_best[0]) < float(alternatives[0][0]) + float(source_margin):
                winner = alternatives[0]
        best_by_group[group] = (float(winner[0]), winner[1].source_name)

    status: dict[str, dict[str, Any]] = {}
    for group, sources in groups.items():
        winner = best_by_group.get(group, (None, ""))[1]
        for source in sources:
            status[source] = {
                "source_group": group,
                "source_group_winner": winner,
                "source_group_selected": bool(source == winner),
            }

    filtered = [
        item
        for item in scored
        if item[1].source_name not in source_to_group
        or status[item[1].source_name]["source_group_selected"]
    ]
    return filtered, status


def _stitch_portfolio_equity(results: list[PortfolioResult], initial_cash: float) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    base_equity = float(initial_cash)
    for result in results:
        equity = result.equity.copy()
        if equity.empty:
            continue
        first = float(equity["portfolio_equity"].iloc[0])
        if first <= 0:
            continue
        equity["stitched_equity"] = equity["portfolio_equity"] / first * base_equity
        equity["window"] = result.run_id
        base_equity = float(equity["stitched_equity"].iloc[-1])
        records.append(equity[["date", "window", "stitched_equity"]])
    if not records:
        return pd.DataFrame(columns=["date", "window", "stitched_equity"])
    return pd.concat(records, ignore_index=True)


def _stitched_metrics(stitched: pd.DataFrame, initial_cash: float) -> dict[str, float]:
    if stitched.empty:
        return {"stitched_total_return": 0.0, "stitched_max_drawdown": 0.0, "stitched_sharpe": 0.0}
    series = stitched.set_index("date")["stitched_equity"].sort_index()
    returns = series.pct_change().dropna()
    running_max = series.cummax()
    drawdown = series / running_max - 1.0
    sharpe = 0.0
    if len(returns) > 1 and float(returns.std()) != 0:
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std())
    return {
        "stitched_total_return": float(series.iloc[-1] / initial_cash - 1.0),
        "stitched_max_drawdown": float(drawdown.min()),
        "stitched_sharpe": float(sharpe),
    }


def _write_portfolio_walk_forward_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    aggregate: dict[str, float],
    grid: str,
    score_mode: str,
    score_mode_min_edge: float,
    train_months: int,
    test_months: int,
    step_months: int,
    max_train_drawdown: float,
) -> tuple[Path, Path, Path]:
    summary = pd.DataFrame(rows)
    candidates = pd.DataFrame(candidate_rows)
    summary_path = output_dir / "portfolio_walk_forward_summary.csv"
    candidates_path = output_dir / "portfolio_candidate_results.csv"
    report_path = output_dir / "summary.md"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    candidates.to_csv(candidates_path, index=False, encoding="utf-8")

    positive_ratio = float((pd.to_numeric(summary["test_total_return"], errors="coerce") > 0).mean()) if not summary.empty else 0.0
    worst_return = float(pd.to_numeric(summary["test_total_return"], errors="coerce").min()) if not summary.empty else 0.0
    selected_count = int(summary["selected_candidate"].nunique()) if not summary.empty else 0

    table = [
        "| window | train | test | selected | train_return | train_dd | train_sharpe | test_return | test_dd | test_sharpe |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            "| {window} | {train_start} to {train_end} | {test_start} to {test_end} | {selected} | "
            "{train_return} | {train_dd} | {train_sharpe:.3f} | {test_return} | {test_dd} | {test_sharpe:.3f} |".format(
                window=row["window"],
                train_start=row["train_start"],
                train_end=row["train_end"],
                test_start=row["test_start"],
                test_end=row["test_end"],
                selected=row["selected_candidate"],
                train_return=_format_pct(row.get("train_total_return")),
                train_dd=_format_pct(row.get("train_max_drawdown")),
                train_sharpe=float(row.get("train_sharpe", 0.0) or 0.0),
                test_return=_format_pct(row.get("test_total_return")),
                test_dd=_format_pct(row.get("test_max_drawdown")),
                test_sharpe=float(row.get("test_sharpe", 0.0) or 0.0),
            )
        )
    body = f"""# Portfolio Walk-Forward Summary

Output directory: `{output_dir}`

Each window selects core-satellite allocation parameters using only the training period, then tests the selected allocation on the next out-of-sample period. This is research analysis only; it does not connect to brokers or place orders.

## Aggregate Out-Of-Sample

| Metric | Value |
| --- | ---: |
| Stitched OOS total return | {_format_pct(aggregate.get("stitched_total_return"))} |
| Stitched OOS max drawdown | {_format_pct(aggregate.get("stitched_max_drawdown"))} |
| Stitched OOS Sharpe | {float(aggregate.get("stitched_sharpe", 0.0)):.3f} |
| Candidate grid | {grid} |
| Score mode | {score_mode} |
| Score mode min edge | {float(score_mode_min_edge):.4f} |
| Candidate runs | {len(candidate_rows)} |
| Train months | {train_months} |
| Test months | {test_months} |
| Step months | {step_months} |
| Training drawdown limit | {_format_pct(max_train_drawdown)} |

## Overfit Audit

| Check | Value |
| --- | ---: |
| Positive OOS window ratio | {_format_pct(positive_ratio)} |
| Worst OOS window return | {_format_pct(worst_return)} |
| Distinct selected candidates | {selected_count} |

## Window Results

{chr(10).join(table)}

## Files

- `portfolio_walk_forward_summary.csv`: selected allocation and train/test metrics by window.
- `portfolio_candidate_results.csv`: all allocation candidates by window.
- `oos_equity_stitched.csv`: out-of-sample windows compounded in chronological order.
"""
    report_path.write_text(body, encoding="utf-8")
    return summary_path, candidates_path, report_path


def run_portfolio_walk_forward(
    config: PortfolioConfig,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
    grid: str = "guarded",
    max_train_drawdown: float = 0.16,
    score_mode: str = "balanced",
    score_mode_min_edge: float = 0.0,
    output_dir: Path | None = None,
    run_id_prefix: str | None = None,
) -> tuple[Path, pd.DataFrame]:
    start, end = _portfolio_window_dates(config)
    windows = generate_portfolio_windows(start, end, train_months, test_months, step_months)
    if not windows:
        raise ValueError("No portfolio walk-forward windows generated.")
    candidates = build_portfolio_candidate_grid(grid)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = output_dir or (config.project_root / "outputs" / "portfolio_walk_forward")
    wf_dir = root / (run_id_prefix or f"{stamp}_{config.name}")
    wf_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    test_results: list[PortfolioResult] = []
    for window in windows:
        scored: list[tuple[float, PortfolioCandidate, PortfolioResult]] = []
        balanced_scored: list[tuple[float, PortfolioCandidate, PortfolioResult]] = []
        for candidate in candidates:
            train_config = _candidate_config(config, candidate, window.train_start, window.train_end)
            train_result = run_portfolio_combine(train_config, run_id=f"{window.name}_{candidate.name}_train", write_outputs=False)
            score = score_portfolio_metrics(train_result.metrics, max_drawdown_limit=max_train_drawdown, score_mode=score_mode)
            balanced_score = score_portfolio_metrics(train_result.metrics, max_drawdown_limit=max_train_drawdown, score_mode="balanced")
            scored.append((score, candidate, train_result))
            balanced_scored.append((balanced_score, candidate, train_result))
            candidate_rows.append(
                {
                    "window": window.name,
                    "candidate": candidate.name,
                    "score": score,
                    "score_mode": score_mode,
                    "score_mode_min_edge": float(score_mode_min_edge),
                    "balanced_score": balanced_score,
                    "parameters": _candidate_payload(candidate),
                    **{f"train_{key}": value for key, value in train_result.metrics.items()},
                }
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        balanced_scored.sort(key=lambda item: item[0], reverse=True)
        active_scores_by_candidate = {_candidate_name(candidate): float(score) for score, candidate, _ in scored}
        best_score, best_candidate, best_train_result, score_gate_payload = _apply_score_mode_min_edge_gate(
            scored[0],
            balanced_scored[0],
            active_scores_by_candidate,
            score_mode=score_mode,
            score_mode_min_edge=score_mode_min_edge,
        )
        test_config = _candidate_config(config, best_candidate, window.test_start, window.test_end)
        test_result = run_portfolio_combine(test_config, run_id=f"{window.name}_{best_candidate.name}_test", write_outputs=False)
        test_results.append(test_result)
        selected_path = wf_dir / f"{window.name}_selected_params.json"
        selected_path.write_text(_candidate_payload(best_candidate), encoding="utf-8")
        rows.append(
            {
                "window": window.name,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "selected_candidate": best_candidate.name,
                "selected_score": best_score,
                "score_mode": score_mode,
                **score_gate_payload,
                "selected_params_path": str(selected_path),
                **{f"train_{key}": value for key, value in best_train_result.metrics.items()},
                **{f"test_{key}": value for key, value in test_result.metrics.items()},
            }
        )

    stitched = _stitch_portfolio_equity(test_results, config.initial_cash)
    stitched_path = wf_dir / "oos_equity_stitched.csv"
    stitched.to_csv(stitched_path, index=False, encoding="utf-8")
    aggregate = _stitched_metrics(stitched, config.initial_cash)
    summary_path, candidates_path, report_path = _write_portfolio_walk_forward_report(
        wf_dir,
        rows,
        candidate_rows,
        aggregate,
        grid,
        score_mode,
        score_mode_min_edge,
        train_months,
        test_months,
        step_months,
        max_train_drawdown,
    )
    summary = pd.read_csv(summary_path)
    summary.attrs["summary_csv"] = str(summary_path)
    summary.attrs["candidate_csv"] = str(candidates_path)
    summary.attrs["summary_md"] = str(report_path)
    summary.attrs["stitched_csv"] = str(stitched_path)
    summary.attrs.update(aggregate)
    return wf_dir, summary


def run_portfolio_source_selection_walk_forward(
    config: PortfolioConfig,
    satellite_sources: dict[str, CurveConfig],
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
    grid: str = "activation",
    max_train_drawdown: float = 0.16,
    score_mode: str = "balanced",
    score_mode_min_edge: float = 0.0,
    default_source_name: str | None = None,
    source_switch_margin: float = 0.0,
    source_switch_margin_by_source: dict[str, float] | None = None,
    source_validation_months: int = 0,
    source_stability_penalty: float = 0.0,
    max_satellite_weight: float | None = None,
    source_max_satellite_weight: dict[str, float] | None = None,
    source_groups: dict[str, list[str]] | None = None,
    output_dir: Path | None = None,
    run_id_prefix: str | None = None,
) -> tuple[Path, pd.DataFrame]:
    start, end = _portfolio_window_dates(config)
    windows = generate_portfolio_windows(start, end, train_months, test_months, step_months)
    if not windows:
        raise ValueError("No portfolio source-selection windows generated.")
    normalized_source_groups = _normalize_source_groups(source_groups)
    grouped_sources = {source for sources in normalized_source_groups.values() for source in sources}
    missing_group_sources = sorted(grouped_sources - set(satellite_sources))
    if missing_group_sources:
        raise ValueError(f"source_groups reference unknown satellite sources: {missing_group_sources}")
    candidates = build_portfolio_source_candidate_grid(
        satellite_sources,
        grid,
        max_satellite_weight=max_satellite_weight,
        source_max_satellite_weight=source_max_satellite_weight,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = output_dir or (config.project_root / "outputs" / "portfolio_source_selection")
    wf_dir = root / (run_id_prefix or f"{stamp}_{config.name}_source_selection")
    wf_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    test_results: list[PortfolioResult] = []
    previous_selected_source: str | None = None
    for window in windows:
        window_candidate_start = len(candidate_rows)
        validation_dates = _source_validation_dates(window, source_validation_months)
        scored: list[tuple[float, PortfolioSourceCandidate, PortfolioResult]] = []
        balanced_scored: list[tuple[float, PortfolioSourceCandidate, PortfolioResult]] = []
        for candidate in candidates:
            train_config = _source_candidate_config(config, candidate, window.train_start, window.train_end)
            train_result = run_portfolio_combine(train_config, run_id=f"{window.name}_{candidate.name}_train", write_outputs=False)
            train_score = score_portfolio_metrics(train_result.metrics, max_drawdown_limit=max_train_drawdown, score_mode=score_mode)
            balanced_train_score = score_portfolio_metrics(
                train_result.metrics,
                max_drawdown_limit=max_train_drawdown,
                score_mode="balanced",
            )
            validation_score: float | None = None
            balanced_validation_score: float | None = None
            validation_result: PortfolioResult | None = None
            score = train_score
            balanced_score = balanced_train_score
            if validation_dates is not None:
                validation_start, validation_end = validation_dates
                validation_config = _source_candidate_config(config, candidate, validation_start, validation_end)
                validation_result = run_portfolio_combine(
                    validation_config,
                    run_id=f"{window.name}_{candidate.name}_validation",
                    write_outputs=False,
                )
                validation_score = score_portfolio_metrics(
                    validation_result.metrics,
                    max_drawdown_limit=max_train_drawdown,
                    score_mode=score_mode,
                )
                balanced_validation_score = score_portfolio_metrics(
                    validation_result.metrics,
                    max_drawdown_limit=max_train_drawdown,
                    score_mode="balanced",
                )
                score = validation_score
                balanced_score = balanced_validation_score
            scored.append((score, candidate, train_result))
            balanced_scored.append((balanced_score, candidate, train_result))
            validation_metrics = validation_result.metrics if validation_result is not None else {}
            candidate_rows.append(
                {
                    "window": window.name,
                    "candidate": candidate.name,
                    "source_name": candidate.source_name,
                    "allocation_candidate": candidate.allocation.name,
                    "score": score,
                    "score_mode": score_mode,
                    "score_mode_min_edge": float(score_mode_min_edge),
                    "max_satellite_weight": max_satellite_weight if max_satellite_weight is not None else pd.NA,
                    "source_max_satellite_weight": json.dumps(source_max_satellite_weight or {}, sort_keys=True),
                    "balanced_score": balanced_score,
                    "train_score": train_score,
                    "balanced_train_score": balanced_train_score,
                    "validation_score": validation_score,
                    "balanced_validation_score": balanced_validation_score,
                    "source_validation_months": int(source_validation_months),
                    "previous_selected_source": previous_selected_source or "",
                    "source_stability_penalty": float(source_stability_penalty),
                    "source_switch_margin_by_source": json.dumps(source_switch_margin_by_source or {}, sort_keys=True),
                    "stability_adjusted_score": (
                        score
                        if not previous_selected_source
                        or candidate.source_name in {previous_selected_source, "core_only"}
                        else score - float(source_stability_penalty)
                    ),
                    "parameters": _source_candidate_payload(candidate),
                    **{f"train_{key}": value for key, value in train_result.metrics.items()},
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                }
            )
        scored, source_group_status = _apply_source_group_preselection(
            scored,
            source_groups,
            source_switch_margin_by_source=source_switch_margin_by_source,
        )
        if source_group_status:
            kept_sources = {candidate.source_name for _, candidate, _ in scored}
            balanced_scored = [
                item
                for item in balanced_scored
                if item[1].source_name in kept_sources or item[1].source_name == "core_only"
            ]
            for row in candidate_rows[window_candidate_start:]:
                status = source_group_status.get(str(row["source_name"]), {})
                row["source_group"] = status.get("source_group", "")
                row["source_group_winner"] = status.get("source_group_winner", "")
                row["source_group_selected"] = status.get("source_group_selected", True)
        else:
            for row in candidate_rows[window_candidate_start:]:
                row["source_group"] = ""
                row["source_group_winner"] = ""
                row["source_group_selected"] = True
        best_score, best_candidate, best_train_result, raw_best = _select_source_candidate(
            scored,
            default_source_name=default_source_name,
            source_switch_margin=source_switch_margin,
            source_switch_margin_by_source=source_switch_margin_by_source,
            previous_source_name=previous_selected_source,
            source_stability_penalty=source_stability_penalty,
        )
        balanced_score, balanced_candidate, balanced_train_result, _ = _select_source_candidate(
            balanced_scored,
            default_source_name=default_source_name,
            source_switch_margin=source_switch_margin,
            source_switch_margin_by_source=source_switch_margin_by_source,
            previous_source_name=previous_selected_source,
            source_stability_penalty=source_stability_penalty,
        )
        active_scores_by_candidate = {_candidate_name(candidate): float(score) for score, candidate, _ in scored}
        best_score, best_candidate, best_train_result, score_gate_payload = _apply_score_mode_min_edge_gate(
            (best_score, best_candidate, best_train_result),
            (balanced_score, balanced_candidate, balanced_train_result),
            active_scores_by_candidate,
            score_mode=score_mode,
            score_mode_min_edge=score_mode_min_edge,
        )
        test_config = _source_candidate_config(config, best_candidate, window.test_start, window.test_end)
        test_result = run_portfolio_combine(test_config, run_id=f"{window.name}_{best_candidate.name}_test", write_outputs=False)
        test_results.append(test_result)
        selected_path = wf_dir / f"{window.name}_selected_params.json"
        selected_path.write_text(_source_candidate_payload(best_candidate), encoding="utf-8")
        rows.append(
            {
                "window": window.name,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "selected_candidate": best_candidate.name,
                "selected_source": best_candidate.source_name,
                "selected_allocation": best_candidate.allocation.name,
                "selected_score": best_score,
                "score_mode": score_mode,
                "max_satellite_weight": max_satellite_weight if max_satellite_weight is not None else pd.NA,
                "source_max_satellite_weight": json.dumps(source_max_satellite_weight or {}, sort_keys=True),
                **score_gate_payload,
                "selection_score_type": "validation" if validation_dates is not None else "train",
                "validation_score": best_score if validation_dates is not None else None,
                "raw_best_candidate": raw_best[1].name,
                "raw_best_source": raw_best[1].source_name,
                "raw_best_score": raw_best[0],
                "default_source": default_source_name or "",
                "source_switch_margin": float(source_switch_margin),
                "source_switch_margin_by_source": json.dumps(source_switch_margin_by_source or {}, sort_keys=True),
                "source_groups": json.dumps(normalized_source_groups, sort_keys=True),
                "selected_source_group": source_group_status.get(best_candidate.source_name, {}).get("source_group", ""),
                "selected_source_group_winner": source_group_status.get(best_candidate.source_name, {}).get("source_group_winner", ""),
                "score_mode_min_edge": float(score_mode_min_edge),
                "source_validation_months": int(source_validation_months),
                "previous_selected_source": previous_selected_source or "",
                "source_stability_penalty": float(source_stability_penalty),
                "source_stability_applied": bool(
                    previous_selected_source
                    and source_stability_penalty > 0
                    and raw_best[1].source_name != best_candidate.source_name
                    and best_candidate.source_name == previous_selected_source
                ),
                "raw_best_edge_vs_selected": float(raw_best[0]) - float(best_score),
                "selected_params_path": str(selected_path),
                **{f"train_{key}": value for key, value in best_train_result.metrics.items()},
                **{f"test_{key}": value for key, value in test_result.metrics.items()},
            }
        )
        previous_selected_source = best_candidate.source_name

    stitched = _stitch_portfolio_equity(test_results, config.initial_cash)
    stitched_path = wf_dir / "oos_equity_stitched.csv"
    stitched.to_csv(stitched_path, index=False, encoding="utf-8")
    aggregate = _stitched_metrics(stitched, config.initial_cash)
    summary_path, candidates_path, report_path = _write_portfolio_walk_forward_report(
        wf_dir,
        rows,
        candidate_rows,
        aggregate,
        f"source-selection/{grid}",
        score_mode,
        score_mode_min_edge,
        train_months,
        test_months,
        step_months,
        max_train_drawdown,
    )
    summary = pd.read_csv(summary_path)
    summary.attrs["summary_csv"] = str(summary_path)
    summary.attrs["candidate_csv"] = str(candidates_path)
    summary.attrs["summary_md"] = str(report_path)
    summary.attrs["stitched_csv"] = str(stitched_path)
    summary.attrs.update(aggregate)
    return wf_dir, summary
