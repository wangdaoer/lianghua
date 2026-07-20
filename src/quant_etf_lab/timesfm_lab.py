"""TimesFM optional research-control forecast feature lab."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .timesfm_control import DEFAULT_CONTROL_GROUP_ID


DEFAULT_OUTPUT_DIR = Path("outputs/research/timesfm_lab_latest")
DEFAULT_DATA_DIR = Path("data/processed/stocks")
DEFAULT_HORIZONS = [1, 5, 20]
DEFAULT_MODEL_REPO = "google/timesfm-2.5-200m-pytorch"

ForecastFn = Callable[[list[float], int], dict[str, float]]

_EMPTY_FEATURE_COLUMNS = [
    "date",
    "code",
    "name",
    "horizon",
    "context_rows",
    "latest_close",
    "timesfm_point_close",
    "timesfm_q10_close",
    "timesfm_q50_close",
    "timesfm_q90_close",
    "timesfm_point_return",
    "timesfm_q10_return",
    "timesfm_q90_return",
    "timesfm_uncertainty_width",
    "timesfm_downside_risk",
    "signal_role",
    "position_effect",
    "broker_action",
]
_EMPTY_ANOMALY_COLUMNS = [
    "date",
    "code",
    "name",
    "horizon",
    "anomaly_status",
    "reason",
    "position_effect",
    "broker_action",
]
_READ_EXCEPTIONS = (
    FileNotFoundError,
    OSError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
    ValueError,
)
_MODEL_INIT_EXCEPTIONS = (
    ImportError,
    ModuleNotFoundError,
    AttributeError,
    RuntimeError,
    OSError,
    ValueError,
)


@dataclass(frozen=True)
class TimesFMLabResult:
    output_dir: Path
    features_path: Path
    uncertainty_path: Path
    anomaly_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    features: pd.DataFrame


def _parse_positive_ints(values: Iterable[int]) -> list[int]:
    parsed = sorted({int(value) for value in values})
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError("TimesFM lab horizons must be positive integers.")
    return parsed


def _normalize_symbols(symbols: Iterable[str] | None) -> list[str] | None:
    if symbols is None:
        return None
    parsed = [str(symbol).strip().zfill(6) for symbol in symbols if str(symbol).strip()]
    return parsed or None


def _csv_paths(data_dir: Path, symbols: Iterable[str] | None, max_symbols: int | None, recursive: bool) -> list[Path]:
    symbol_set = set(_normalize_symbols(symbols) or [])
    pattern = "**/*.csv" if recursive else "*.csv"
    paths = sorted(path for path in data_dir.glob(pattern) if path.is_file())
    if symbol_set:
        paths = [
            path
            for path in paths
            if path.stem.zfill(6) in symbol_set or path.stem[:6].zfill(6) in symbol_set
        ]
    if max_symbols is not None and max_symbols > 0:
        paths = paths[: int(max_symbols)]
    return paths


def _read_price_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "code", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str).str.zfill(6)
    if "name" not in data.columns:
        data["name"] = data["code"]
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["date", "code", "close"])
    data = data[data["close"] > 0].sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return data[["date", "code", "name", "close"]].reset_index(drop=True)


def _load_histories(
    data_dir: Path,
    symbols: Iterable[str] | None,
    max_symbols: int | None,
    recursive: bool,
) -> tuple[list[pd.DataFrame], list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for path in _csv_paths(data_dir, symbols=symbols, max_symbols=max_symbols, recursive=recursive):
        try:
            frame = _read_price_csv(path)
            if not frame.empty:
                frames.append(frame)
        except _READ_EXCEPTIONS as error:
            failures.append({"path": str(path), "error": str(error)})
    return frames, failures


def _is_timesfm_available() -> bool:
    return importlib.util.find_spec("timesfm") is not None


def _build_timesfm_forecaster(
    model_repo: str,
    context_length: int,
    max_horizon: int,
) -> ForecastFn:
    import timesfm  # type: ignore[import-not-found]

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_repo)
    config = timesfm.ForecastConfig(
        max_context=int(context_length),
        max_horizon=int(max_horizon),
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        fix_quantile_crossing=True,
        infer_is_positive=True,
    )
    model.compile(forecast_config=config)

    def _forecast(close_values: list[float], horizon: int) -> dict[str, float]:
        inputs = [np.asarray(close_values[-int(context_length) :], dtype=float)]
        point_forecast, quantile_forecast = model.forecast(horizon=int(horizon), inputs=inputs)
        quantiles = np.asarray(quantile_forecast)[0, int(horizon) - 1]
        return {
            "point": float(np.asarray(point_forecast)[0, int(horizon) - 1]),
            "q10": float(quantiles[1]),
            "q50": float(quantiles[5]),
            "q90": float(quantiles[9]),
        }

    return _forecast


def _empty_features() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_FEATURE_COLUMNS)


def _empty_anomalies() -> pd.DataFrame:
    return pd.DataFrame(columns=_EMPTY_ANOMALY_COLUMNS)


def _build_features(
    histories: list[pd.DataFrame],
    horizons: list[int],
    context_length: int,
    min_context_rows: int,
    forecaster: ForecastFn,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for history in histories:
        code = str(history["code"].iloc[-1]).zfill(6)
        name = str(history["name"].iloc[-1])
        context = history.tail(int(context_length))
        required_context_rows = min(int(min_context_rows), int(context_length))
        if len(context) < required_context_rows:
            failures.append({"code": code, "error": "insufficient_context_rows"})
            continue
        close_values = [float(value) for value in context["close"].tolist()]
        latest_close = close_values[-1]
        latest_date = pd.Timestamp(context["date"].iloc[-1]).strftime("%Y-%m-%d")
        for horizon in horizons:
            forecast = forecaster(close_values, int(horizon))
            point = float(forecast["point"])
            q10 = float(forecast["q10"])
            q50 = float(forecast["q50"])
            q90 = float(forecast["q90"])
            rows.append(
                {
                    "date": latest_date,
                    "code": code,
                    "name": name,
                    "horizon": int(horizon),
                    "context_rows": int(len(context)),
                    "latest_close": latest_close,
                    "timesfm_point_close": point,
                    "timesfm_q10_close": q10,
                    "timesfm_q50_close": q50,
                    "timesfm_q90_close": q90,
                    "timesfm_point_return": point / latest_close - 1.0,
                    "timesfm_q10_return": q10 / latest_close - 1.0,
                    "timesfm_q90_return": q90 / latest_close - 1.0,
                    "timesfm_uncertainty_width": q90 / latest_close - q10 / latest_close,
                    "timesfm_downside_risk": q50 / latest_close - q10 / latest_close,
                    "signal_role": "timesfm_control_feature",
                    "position_effect": "none",
                    "broker_action": "none",
                }
            )
    features = pd.DataFrame(rows, columns=_EMPTY_FEATURE_COLUMNS)
    return features, failures


def _write_outputs(
    output_dir: Path,
    features: pd.DataFrame,
    snapshot: dict[str, Any],
) -> TimesFMLabResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "timesfm_forecast_features.csv"
    uncertainty_path = output_dir / "timesfm_uncertainty_reference.csv"
    anomaly_path = output_dir / "timesfm_anomaly_reference.csv"
    snapshot_path = output_dir / "timesfm_lab_snapshot.json"
    report_path = output_dir / "timesfm_lab_report.md"

    features.to_csv(features_path, index=False, encoding="utf-8-sig")
    uncertainty_columns = [
        "date",
        "code",
        "name",
        "horizon",
        "timesfm_uncertainty_width",
        "timesfm_downside_risk",
        "position_effect",
        "broker_action",
    ]
    uncertainty = features[uncertainty_columns] if not features.empty else pd.DataFrame(columns=uncertainty_columns)
    uncertainty.to_csv(uncertainty_path, index=False, encoding="utf-8-sig")
    anomalies = _empty_anomalies()
    anomalies.to_csv(anomaly_path, index=False, encoding="utf-8-sig")
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot), encoding="utf-8")
    return TimesFMLabResult(
        output_dir=output_dir,
        features_path=features_path,
        uncertainty_path=uncertainty_path,
        anomaly_path=anomaly_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        features=features,
    )


def _base_snapshot(
    status: str,
    generated_at: str | None,
    data_dir: Path,
    horizons: list[int],
    context_length: int,
    min_context_rows: int,
    model_repo: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "control_group_id": DEFAULT_CONTROL_GROUP_ID,
        "generated_at": generated_at or datetime.now().isoformat(timespec="seconds"),
        "allowed_role": "control_feature_generation_only",
        "position_effect": "none",
        "broker_action": "none",
        "data_dir": str(data_dir),
        "horizons": horizons,
        "context_length": int(context_length),
        "min_context_rows": int(min_context_rows),
        "model_repo": model_repo,
        "forecast_feature_rows": 0,
        "uncertainty_reference_rows": 0,
        "anomaly_reference_rows": 0,
        "loaded_symbol_count": 0,
        "failure_count": 0,
        "failures": [],
        "prohibited_integrations": [
            "default_allocator",
            "paper_account_target_weights",
            "daily_pipeline_position_sizing",
            "live_preflight_orders",
            "broker_order_generation",
        ],
    }


def _render_report(snapshot: dict[str, Any]) -> str:
    blocker = snapshot.get("blocker") or "none"
    return f"""# TimesFM 对照实验室

| 项目 | 值 |
| --- | --- |
| 状态 | `{snapshot.get("status")}` |
| 对照组 | `{snapshot.get("control_group_id")}` |
| 允许角色 | `{snapshot.get("allowed_role")}` |
| 仓位影响 | `{snapshot.get("position_effect")}` |
| 券商动作 | `{snapshot.get("broker_action")}` |
| 特征行数 | `{snapshot.get("forecast_feature_rows")}` |
| 已加载标的 | `{snapshot.get("loaded_symbol_count")}` |
| 阻断项 | `{blocker}` |

本输出只用于 TimesFM 对照组研究，不生成目标仓位、券商订单或实盘前置动作。
"""


def run_timesfm_lab(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    symbols: Iterable[str] | None = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    context_length: int = 512,
    min_context_rows: int = 32,
    max_symbols: int | None = None,
    recursive: bool = False,
    model_enabled: bool = False,
    model_repo: str = DEFAULT_MODEL_REPO,
    timesfm_available: bool | None = None,
    forecaster: ForecastFn | None = None,
    generated_at: str | None = None,
) -> TimesFMLabResult:
    resolved_data_dir = Path(data_dir)
    resolved_output_dir = Path(output_dir)
    horizon_values = _parse_positive_ints(horizons)
    snapshot = _base_snapshot(
        status="model_disabled",
        generated_at=generated_at,
        data_dir=resolved_data_dir,
        horizons=horizon_values,
        context_length=context_length,
        min_context_rows=min_context_rows,
        model_repo=model_repo,
    )

    histories, load_failures = _load_histories(
        resolved_data_dir,
        symbols=symbols,
        max_symbols=max_symbols,
        recursive=recursive,
    )
    snapshot["loaded_symbol_count"] = len(histories)
    snapshot["failure_count"] = len(load_failures)
    snapshot["failures"] = load_failures[:20]

    if forecaster is None:
        if not model_enabled:
            snapshot["status"] = "model_disabled"
            snapshot["blocker"] = "enable_model_not_set"
            return _write_outputs(resolved_output_dir, _empty_features(), snapshot)
        available = _is_timesfm_available() if timesfm_available is None else bool(timesfm_available)
        if not available:
            snapshot["status"] = "blocked_missing_dependency"
            snapshot["blocker"] = "timesfm_dependency_missing"
            snapshot["install_hint"] = 'pip install "timesfm[torch]"'
            return _write_outputs(resolved_output_dir, _empty_features(), snapshot)
        try:
            forecaster = _build_timesfm_forecaster(
                model_repo=model_repo,
                context_length=context_length,
                max_horizon=max(horizon_values),
            )
        except _MODEL_INIT_EXCEPTIONS as error:
            snapshot["status"] = "blocked_model_initialization_failed"
            snapshot["blocker"] = "timesfm_model_initialization_failed"
            snapshot["error"] = str(error)
            return _write_outputs(resolved_output_dir, _empty_features(), snapshot)

    if not histories:
        snapshot["status"] = "blocked_no_price_data"
        snapshot["blocker"] = "no_local_price_data"
        return _write_outputs(resolved_output_dir, _empty_features(), snapshot)

    features, forecast_failures = _build_features(
        histories=histories,
        horizons=horizon_values,
        context_length=context_length,
        min_context_rows=min_context_rows,
        forecaster=forecaster,
    )
    all_failures = [*load_failures, *forecast_failures]
    snapshot["status"] = "timesfm_lab_completed" if not features.empty else "completed_with_no_feature_rows"
    snapshot["forecast_feature_rows"] = int(len(features))
    snapshot["uncertainty_reference_rows"] = int(len(features))
    snapshot["anomaly_reference_rows"] = 0
    snapshot["failure_count"] = len(all_failures)
    snapshot["failures"] = all_failures[:20]
    return _write_outputs(resolved_output_dir, features, snapshot)
