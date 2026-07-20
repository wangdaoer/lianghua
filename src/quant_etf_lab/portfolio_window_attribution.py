"""Day-level attribution for one portfolio source-selection window."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ._compat import read_text
from .portfolio import (
    CurveConfig,
    PortfolioConfig,
    SatelliteFilterConfig,
    load_portfolio_config,
    load_portfolio_source_selection_config,
    run_portfolio_combine,
)


@dataclass(frozen=True)
class PortfolioWindowAttributionResult:
    output_dir: Path
    report_path: Path
    metrics_path: Path
    daily_path: Path
    snapshot_path: Path
    snapshot: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Variant must use LABEL=PATH format: {value}")
    label, path_text = value.split("=", 1)
    label = label.strip()
    path_text = path_text.strip()
    if not label:
        raise ValueError(f"Variant label cannot be empty: {value}")
    if not path_text:
        raise ValueError(f"Variant path cannot be empty: {value}")
    return label, Path(path_text)


def _read_selected_params(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Selected params file not found: {path}")
    payload = json.loads(read_text(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Selected params must be a JSON object: {path}")
    return payload


def _satellite_filter_from_payload(base: SatelliteFilterConfig, raw: Any) -> SatelliteFilterConfig:
    if raw is None:
        return base
    if not isinstance(raw, dict):
        raise ValueError("allocation.satellite_filter must be a mapping.")
    allowed = {field.name for field in fields(SatelliteFilterConfig)}
    merged = {field.name: getattr(base, field.name) for field in fields(SatelliteFilterConfig)}
    merged.update({str(key): value for key, value in raw.items() if str(key) in allowed})
    return SatelliteFilterConfig(
        enabled=bool(merged["enabled"]),
        ma_window=max(int(merged["ma_window"]), 1),
        momentum_window=max(int(merged["momentum_window"]), 1),
        min_momentum=float(merged["min_momentum"]),
        max_drawdown=max(float(merged["max_drawdown"]), 0.0),
        reduced_drawdown=max(float(merged["reduced_drawdown"]), 0.0),
        reduced_scale=min(max(float(merged["reduced_scale"]), 0.0), 1.0),
        default_scale=min(max(float(merged["default_scale"]), 0.0), 1.0),
        require_above_ma=bool(merged["require_above_ma"]),
        reallocate_to=str(merged["reallocate_to"]),
    )


def _weights_from_payload(base: dict[str, dict[str, float]], raw: Any) -> dict[str, dict[str, float]]:
    if raw is None:
        return base
    if not isinstance(raw, dict):
        raise ValueError("allocation.weights must be a mapping.")
    weights: dict[str, dict[str, float]] = {}
    for regime in ("risk_on", "risk_off", "crash"):
        regime_raw = raw.get(regime, base.get(regime, {}))
        if not isinstance(regime_raw, dict):
            raise ValueError(f"allocation.weights.{regime} must be a mapping.")
        core = float(regime_raw.get("core", 0.0))
        satellite = float(regime_raw.get("satellite", 0.0))
        if core < 0.0 or satellite < 0.0:
            raise ValueError(f"allocation.weights.{regime} cannot be negative.")
        if core + satellite > 1.0000001:
            raise ValueError(f"allocation.weights.{regime} core + satellite cannot exceed 1.")
        cash = float(regime_raw.get("cash", 1.0 - core - satellite))
        if abs(cash - (1.0 - core - satellite)) > 1e-6:
            raise ValueError(f"allocation.weights.{regime}.cash must equal 1 - core - satellite.")
        weights[regime] = {"core": core, "satellite": satellite, "cash": cash}
    return weights


def _load_config_and_sources(config_path: Path) -> tuple[PortfolioConfig, dict[str, CurveConfig]]:
    try:
        return load_portfolio_source_selection_config(config_path)
    except ValueError as error:
        if "satellite_sources" not in str(error):
            raise
        return load_portfolio_config(config_path), {}


def _source_from_payload(
    base_config: PortfolioConfig,
    sources: dict[str, CurveConfig],
    payload: dict[str, Any],
) -> tuple[CurveConfig, list[str]]:
    notes: list[str] = []
    source_name = str(payload.get("source_name") or "").strip()
    source_path = Path(str(payload.get("source_path"))) if payload.get("source_path") else None
    if source_name == "core_only":
        return base_config.satellite, notes
    if source_name and source_name in sources:
        if source_path is not None and source_path != sources[source_name].path:
            notes.append("stale_source_path_ignored")
        return sources[source_name], notes
    if source_path is not None and source_path.exists():
        notes.append("source_path_used")
        return CurveConfig(name="satellite", path=source_path, equity_column=base_config.satellite.equity_column), notes
    if source_name:
        notes.append("source_name_not_in_config_used_default_satellite")
    return base_config.satellite, notes


def _config_from_selected_params(
    base_config: PortfolioConfig,
    sources: dict[str, CurveConfig],
    params_path: Path,
    start: str,
    end: str,
) -> tuple[PortfolioConfig, list[str]]:
    payload = _read_selected_params(params_path)
    allocation = payload.get("allocation", payload)
    if not isinstance(allocation, dict):
        raise ValueError(f"Selected params allocation must be a mapping: {params_path}")
    satellite, notes = _source_from_payload(base_config, sources, payload)
    regime = allocation.get("regime_overrides") or {}
    if not isinstance(regime, dict):
        raise ValueError("allocation.regime_overrides must be a mapping.")
    return (
        PortfolioConfig(
            project_root=base_config.project_root,
            name=base_config.name,
            initial_cash=base_config.initial_cash,
            output_dir=base_config.output_dir,
            core=base_config.core,
            satellite=satellite,
            benchmark_path=base_config.benchmark_path,
            benchmark_close_column=base_config.benchmark_close_column,
            ma_window=int(regime.get("ma_window", base_config.ma_window)),
            drop_window=int(regime.get("drop_window", base_config.drop_window)),
            risk_on_drop_threshold=float(regime.get("risk_on_drop_threshold", base_config.risk_on_drop_threshold)),
            crash_drop_threshold=float(regime.get("crash_drop_threshold", base_config.crash_drop_threshold)),
            default_regime=str(regime.get("default_regime", base_config.default_regime)),
            weights=_weights_from_payload(base_config.weights, allocation.get("weights")),
            satellite_filter=_satellite_filter_from_payload(base_config.satellite_filter, allocation.get("satellite_filter")),
            start_date=start,
            end_date=end,
            notes=base_config.notes,
        ),
        notes,
    )


def _core_only_config(base_config: PortfolioConfig, start: str, end: str) -> PortfolioConfig:
    zero_satellite_weights = {
        regime: {"core": 1.0, "satellite": 0.0, "cash": 0.0} for regime in ("risk_on", "risk_off", "crash")
    }
    return replace(
        base_config,
        weights=zero_satellite_weights,
        start_date=start,
        end_date=end,
        satellite_filter=SatelliteFilterConfig(enabled=False),
    )


def _drawdown(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    return clean / clean.cummax() - 1.0


def _variant_metrics(label: str, result: Any, notes: list[str]) -> dict[str, Any]:
    metrics = dict(result.metrics)
    equity = result.equity
    return {
        "variant": label,
        "total_return": metrics.get("total_return"),
        "cagr": metrics.get("cagr"),
        "max_drawdown": metrics.get("max_drawdown"),
        "sharpe": metrics.get("sharpe"),
        "average_satellite_weight": metrics.get("average_satellite_weight"),
        "satellite_active_day_ratio": metrics.get("satellite_active_day_ratio"),
        "risk_on_day_ratio": metrics.get("risk_on_day_ratio"),
        "final_equity": metrics.get("final_equity"),
        "start_date": equity["date"].min() if not equity.empty else pd.NaT,
        "end_date": equity["date"].max() if not equity.empty else pd.NaT,
        "notes": ",".join(notes),
    }


def _build_daily_comparison(results: dict[str, Any]) -> pd.DataFrame:
    daily: pd.DataFrame | None = None
    for label, result in results.items():
        equity = result.equity[["date", "portfolio_equity", "portfolio_return", "satellite_weight", "effective_regime"]].copy()
        equity[f"{label}_indexed_equity"] = equity["portfolio_equity"] / float(equity["portfolio_equity"].iloc[0])
        equity[f"{label}_daily_return"] = equity["portfolio_return"]
        equity[f"{label}_drawdown"] = _drawdown(equity["portfolio_equity"])
        equity[f"{label}_satellite_weight"] = equity["satellite_weight"]
        equity[f"{label}_effective_regime"] = equity["effective_regime"]
        keep = [
            "date",
            f"{label}_indexed_equity",
            f"{label}_daily_return",
            f"{label}_drawdown",
            f"{label}_satellite_weight",
            f"{label}_effective_regime",
        ]
        daily = equity[keep] if daily is None else daily.merge(equity[keep], on="date", how="inner")
    if daily is None:
        return pd.DataFrame()
    labels = list(results)
    for left in labels:
        for right in labels:
            if left == right:
                continue
            daily[f"{left}_minus_{right}_indexed_equity"] = daily[f"{left}_indexed_equity"] - daily[f"{right}_indexed_equity"]
            daily[f"{left}_minus_{right}_daily_return"] = daily[f"{left}_daily_return"] - daily[f"{right}_daily_return"]
    return daily


def _format_pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number * 100:.2f}%"


def _format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.3f}"


def _render_report(snapshot: dict[str, Any], metrics: pd.DataFrame) -> str:
    rows = [
        "| Variant | Total return | Max drawdown | Sharpe | Avg satellite | Notes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in metrics.itertuples(index=False):
        rows.append(
            "| {variant} | {ret} | {dd} | {sharpe} | {sat} | {notes} |".format(
                variant=row.variant,
                ret=_format_pct(row.total_return),
                dd=_format_pct(row.max_drawdown),
                sharpe=_format_float(row.sharpe),
                sat=_format_pct(row.average_satellite_weight),
                notes=row.notes or "",
            )
        )
    return f"""# Portfolio Window Daily Attribution

Generated at: `{snapshot["generated_at"]}`

This report replays one historical source-selection window for research only. It does not connect to brokers, place orders, or change model defaults.

## Summary

| Item | Value |
| --- | --- |
| Config | `{snapshot["config_path"]}` |
| Start | `{snapshot["start"]}` |
| End | `{snapshot["end"]}` |
| Variants | `{", ".join(snapshot["variants"])}` |
| Best by total return | `{snapshot["best_variant_by_total_return"]}` |
| Worst day delta pair | `{snapshot["worst_daily_delta_pair"]}` |
| Worst day delta date | `{snapshot["worst_daily_delta_date"]}` |
| Worst day delta | `{_format_pct(snapshot["worst_daily_delta"])}` |

## Variant Metrics

{chr(10).join(rows)}

## Files

- `portfolio_window_attribution_metrics.csv`: per-variant return, drawdown, Sharpe, and exposure summary.
- `portfolio_window_attribution_daily.csv`: day-level indexed equity, return, drawdown, exposure, and pairwise deltas.
- `portfolio_window_attribution_snapshot.json`: machine-readable summary.
"""


def run_portfolio_window_attribution(
    config_path: str | Path,
    start: str,
    end: str,
    variants: list[str] | tuple[str, ...],
    output_dir: str | Path = "outputs/research/portfolio_window_attribution_latest",
    include_core_only: bool = False,
) -> PortfolioWindowAttributionResult:
    if not variants:
        raise ValueError("At least one --variant LABEL=PATH is required.")
    config_path = Path(config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_config, sources = _load_config_and_sources(config_path)
    parsed_variants = [_parse_variant(value) for value in variants]
    results: dict[str, Any] = {}
    notes_by_variant: dict[str, list[str]] = {}
    for label, params_path in parsed_variants:
        variant_config, notes = _config_from_selected_params(base_config, sources, params_path, start, end)
        results[label] = run_portfolio_combine(
            variant_config,
            run_id=f"window_attribution_{label}",
            write_outputs=False,
        )
        notes_by_variant[label] = notes
    if include_core_only:
        results["core_only"] = run_portfolio_combine(
            _core_only_config(base_config, start, end),
            run_id="window_attribution_core_only",
            write_outputs=False,
        )
        notes_by_variant["core_only"] = []

    metrics = pd.DataFrame([_variant_metrics(label, result, notes_by_variant[label]) for label, result in results.items()])
    daily = _build_daily_comparison(results)
    best_variant = ""
    if not metrics.empty:
        best_row = metrics.sort_values(["total_return", "sharpe"], ascending=[False, False]).iloc[0]
        best_variant = str(best_row["variant"])

    worst_pair = ""
    worst_date = ""
    worst_value: float | None = None
    for column in [name for name in daily.columns if name.endswith("_daily_return") and "_minus_" in name]:
        series = pd.to_numeric(daily[column], errors="coerce")
        if series.empty or series.isna().all():
            continue
        idx = series.idxmin()
        value = float(series.loc[idx])
        if worst_value is None or value < worst_value:
            worst_value = value
            worst_pair = column.removesuffix("_daily_return")
            worst_date = str(pd.to_datetime(daily.loc[idx, "date"]).date())

    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "start": start,
        "end": end,
        "variants": list(results),
        "variant_notes": notes_by_variant,
        "best_variant_by_total_return": best_variant,
        "worst_daily_delta_pair": worst_pair,
        "worst_daily_delta_date": worst_date,
        "worst_daily_delta": worst_value,
    }

    metrics_path = output_path / "portfolio_window_attribution_metrics.csv"
    daily_path = output_path / "portfolio_window_attribution_daily.csv"
    snapshot_path = output_path / "portfolio_window_attribution_snapshot.json"
    report_path = output_path / "portfolio_window_attribution.md"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    snapshot_path.write_text(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, metrics), encoding="utf-8")

    return PortfolioWindowAttributionResult(
        output_dir=output_path,
        report_path=report_path,
        metrics_path=metrics_path,
        daily_path=daily_path,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )

