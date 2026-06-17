"""Cross-sectional factor research on locally cached A-share histories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .indicators import rsi


DEFAULT_DATA_DIR = Path("data/processed/stocks")
DEFAULT_OUTPUT_DIR = Path("outputs/research/factor_lab_latest")
DEFAULT_FACTORS = [
    "momentum_20d",
    "momentum_60d",
    "reversal_5d",
    "volatility_20d",
    "volume_ratio_20d",
    "amount_ratio_20d",
    "ma20_gap",
    "ma60_gap",
    "rsi14",
    "breakout_20d",
    "liquidity_log_amount",
]
DEFAULT_HORIZONS = [1, 5, 20]


@dataclass(frozen=True)
class FactorLabResult:
    output_dir: Path
    factor_panel_path: Path
    ic_path: Path
    ic_summary_path: Path
    quantile_path: Path
    quantile_summary_path: Path
    correlation_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    panel: pd.DataFrame
    ic_summary: pd.DataFrame
    quantile_summary: pd.DataFrame


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = Path(path) if path is not None else default
    return raw if raw.is_absolute() else project_root / raw


def _parse_date(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) == 8 and text.isdigit():
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _csv_paths(data_dir: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(path for path in data_dir.glob(pattern) if path.is_file())


def _limit_paths(paths: list[Path], max_symbols: int | None) -> list[Path]:
    if max_symbols is None or max_symbols <= 0:
        return paths
    return paths[:max_symbols]


def _read_price_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "code", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "code", "close"]).sort_values("date")
    data = data[data["close"] > 0].drop_duplicates(subset=["date"], keep="last")
    return data.reset_index(drop=True)


def build_symbol_factor_frame(
    prices: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Build per-symbol factors using only data available at each row's close."""
    if prices.empty:
        return pd.DataFrame()
    data = prices.sort_values("date").copy()
    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce")
    amount = pd.to_numeric(data["amount"], errors="coerce")
    returns = close.pct_change()

    out = pd.DataFrame(
        {
            "date": data["date"],
            "code": data["code"].astype(str).str.zfill(6),
            "name": data.get("name", data["code"]).astype(str),
            "close": close,
            "amount": amount,
            "volume": volume,
        }
    )
    out["momentum_20d"] = close / close.shift(20) - 1.0
    out["momentum_60d"] = close / close.shift(60) - 1.0
    out["reversal_5d"] = -(close / close.shift(5) - 1.0)
    out["volatility_20d"] = returns.rolling(20, min_periods=20).std()
    out["volume_ratio_20d"] = volume / volume.rolling(20, min_periods=20).mean() - 1.0
    out["amount_ratio_20d"] = amount / amount.rolling(20, min_periods=20).mean() - 1.0
    out["ma20_gap"] = close / close.rolling(20, min_periods=20).mean() - 1.0
    out["ma60_gap"] = close / close.rolling(60, min_periods=60).mean() - 1.0
    out["rsi14"] = rsi(close, 14)
    prior_high_20 = high.shift(1).rolling(20, min_periods=20).max()
    out["breakout_20d"] = close / prior_high_20 - 1.0
    out["liquidity_log_amount"] = np.log(amount.where(amount > 0))
    for horizon in horizons:
        horizon_int = int(horizon)
        if horizon_int <= 0:
            raise ValueError("horizons must be positive integers.")
        out[f"fwd_return_{horizon_int}d"] = close.shift(-horizon_int) / close - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def build_factor_panel(
    data_dir: str | Path,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    max_symbols: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    recursive: bool = False,
    warmup_days: int = 180,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved_dir = Path(data_dir)
    paths = _limit_paths(_csv_paths(resolved_dir, recursive=recursive), max_symbols)
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            prices = _read_price_csv(path)
            if start is not None and warmup_days > 0:
                prices = prices[prices["date"] >= start - timedelta(days=int(warmup_days))]
            if end is not None:
                prices = prices[prices["date"] <= end]
            factor_frame = build_symbol_factor_frame(prices, horizons=horizons)
            if start is not None:
                factor_frame = factor_frame[factor_frame["date"] >= start]
            if end is not None:
                factor_frame = factor_frame[factor_frame["date"] <= end]
            if not factor_frame.empty:
                frames.append(factor_frame)
        except Exception as error:
            failures.append({"path": str(path), "error": str(error)})
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not panel.empty:
        panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    meta = {
        "data_dir": str(resolved_dir),
        "file_count": len(paths),
        "loaded_symbol_count": int(panel["code"].nunique()) if not panel.empty else 0,
        "row_count": int(len(panel)),
        "failure_count": len(failures),
        "failures": failures[:20],
    }
    return panel, meta


def _valid_factor_names(panel: pd.DataFrame, factors: Iterable[str] | None) -> list[str]:
    requested = list(factors or DEFAULT_FACTORS)
    return [factor for factor in requested if factor in panel.columns]


def compute_ic_table(
    panel: pd.DataFrame,
    factors: Iterable[str] | None = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    min_obs: int = 30,
) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    factor_names = _valid_factor_names(panel, factors)
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        target = f"fwd_return_{int(horizon)}d"
        if target not in panel.columns:
            continue
        for factor in factor_names:
            subset = panel[["date", factor, target]].dropna()
            if subset.empty:
                continue
            for date_value, group in subset.groupby("date", sort=True):
                if len(group) < min_obs or group[factor].nunique() < 3 or group[target].nunique() < 3:
                    continue
                ic = group[factor].corr(group[target], method="spearman")
                if pd.notna(ic):
                    rows.append(
                        {
                            "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                            "factor": factor,
                            "horizon": int(horizon),
                            "ic": float(ic),
                            "obs": int(len(group)),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_ic(ic_table: pd.DataFrame) -> pd.DataFrame:
    if ic_table.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (factor, horizon), group in ic_table.groupby(["factor", "horizon"], sort=True):
        ic = pd.to_numeric(group["ic"], errors="coerce").dropna()
        if ic.empty:
            continue
        std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
        mean = float(ic.mean())
        rows.append(
            {
                "factor": factor,
                "horizon": int(horizon),
                "ic_mean": mean,
                "ic_std": std,
                "ic_ir": mean / std if std else 0.0,
                "ic_t_stat": mean / (std / np.sqrt(len(ic))) if std else 0.0,
                "ic_positive_ratio": float((ic > 0).mean()),
                "ic_abs_mean": float(ic.abs().mean()),
                "obs_dates": int(len(ic)),
                "avg_cross_section_obs": float(pd.to_numeric(group["obs"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "ic_abs_mean"], ascending=[True, False]).reset_index(drop=True)


def compute_quantile_returns(
    panel: pd.DataFrame,
    factors: Iterable[str] | None = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    quantiles: int = 5,
    min_obs: int = 30,
) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2.")
    factor_names = _valid_factor_names(panel, factors)
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        target = f"fwd_return_{int(horizon)}d"
        if target not in panel.columns:
            continue
        for factor in factor_names:
            subset = panel[["date", factor, target]].dropna()
            for date_value, group in subset.groupby("date", sort=True):
                if len(group) < min_obs or group[factor].nunique() < quantiles:
                    continue
                ranked = group.copy()
                try:
                    ranked["quantile"] = pd.qcut(ranked[factor].rank(method="first"), quantiles, labels=False) + 1
                except ValueError:
                    continue
                means = ranked.groupby("quantile", observed=True)[target].mean()
                if 1 not in means.index or quantiles not in means.index:
                    continue
                rows.append(
                    {
                        "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                        "factor": factor,
                        "horizon": int(horizon),
                        "bottom_return": float(means.loc[1]),
                        "top_return": float(means.loc[quantiles]),
                        "long_short_return": float(means.loc[quantiles] - means.loc[1]),
                        "obs": int(len(ranked)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_quantiles(quantile_returns: pd.DataFrame) -> pd.DataFrame:
    if quantile_returns.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (factor, horizon), group in quantile_returns.groupby(["factor", "horizon"], sort=True):
        spread = pd.to_numeric(group["long_short_return"], errors="coerce").dropna()
        if spread.empty:
            continue
        rows.append(
            {
                "factor": factor,
                "horizon": int(horizon),
                "top_return_mean": float(pd.to_numeric(group["top_return"], errors="coerce").mean()),
                "bottom_return_mean": float(pd.to_numeric(group["bottom_return"], errors="coerce").mean()),
                "long_short_mean": float(spread.mean()),
                "long_short_std": float(spread.std(ddof=1)) if len(spread) > 1 else 0.0,
                "long_short_win_ratio": float((spread > 0).mean()),
                "obs_dates": int(len(spread)),
                "avg_cross_section_obs": float(pd.to_numeric(group["obs"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "long_short_mean"], ascending=[True, False]).reset_index(drop=True)


def compute_factor_correlation(panel: pd.DataFrame, factors: Iterable[str] | None = None, sample_rows: int = 500000) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame()
    factor_names = _valid_factor_names(panel, factors)
    data = panel[factor_names].dropna(how="all")
    if sample_rows > 0 and len(data) > sample_rows:
        data = data.tail(sample_rows)
    return data.corr(method="spearman")


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not pd.notna(number):
        return "N/A"
    return f"{number * 100:.2f}%"


def _num(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not pd.notna(number):
        return "N/A"
    return f"{number:.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    if frame.empty:
        return "No rows."
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].head(max_rows)
    lines = ["| " + " | ".join(subset.columns) + " |", "| " + " | ".join(["---"] * len(subset.columns)) + " |"]
    for row in subset.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(_num(value, 4))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _render_report(snapshot: dict[str, Any], ic_summary: pd.DataFrame, quantile_summary: pd.DataFrame) -> str:
    best_ic = ic_summary.sort_values(["ic_abs_mean"], ascending=False) if not ic_summary.empty else pd.DataFrame()
    best_spread = quantile_summary.sort_values(["long_short_mean"], ascending=False) if not quantile_summary.empty else pd.DataFrame()
    return f"""# Factor Lab Report

Generated at: `{snapshot.get("generated_at")}`

This report is a local research artifact only. It does not connect to brokers, place orders, or provide investment advice.

## Data

| Item | Value |
| --- | ---: |
| Data dir | `{snapshot.get("data_dir")}` |
| Files loaded | {snapshot.get("loaded_symbol_count")} / {snapshot.get("file_count")} |
| Panel rows | {snapshot.get("panel_row_count")} |
| Date range | {snapshot.get("start_date", "N/A")} to {snapshot.get("end_date", "N/A")} |
| Factors | {snapshot.get("factor_count")} |
| Horizons | `{snapshot.get("horizons")}` |
| Min cross-section obs | {snapshot.get("min_obs")} |
| Quantiles | {snapshot.get("quantiles")} |
| Warmup days | {snapshot.get("warmup_days")} |
| Save panel | `{snapshot.get("save_panel")}` |

## IC Leaders

{_markdown_table(best_ic, ["factor", "horizon", "ic_mean", "ic_abs_mean", "ic_t_stat", "ic_positive_ratio", "obs_dates"], 12)}

## Quantile Spread Leaders

{_markdown_table(best_spread, ["factor", "horizon", "long_short_mean", "long_short_win_ratio", "obs_dates"], 12)}

## Files

- Factor panel: `{snapshot.get("factor_panel_path")}`
- IC table: `{snapshot.get("ic_path")}`
- IC summary: `{snapshot.get("ic_summary_path")}`
- Quantile returns: `{snapshot.get("quantile_path")}`
- Quantile summary: `{snapshot.get("quantile_summary_path")}`
- Factor correlation: `{snapshot.get("correlation_path")}`
- Snapshot: `{snapshot.get("snapshot_path")}`
"""


def run_factor_lab(
    project_root: str | Path = Path("."),
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    factors: Iterable[str] | None = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    quantiles: int = 5,
    min_obs: int = 30,
    max_symbols: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    recursive: bool = False,
    warmup_days: int = 180,
    save_panel: bool = True,
) -> FactorLabResult:
    root = Path(project_root).resolve()
    resolved_data = _resolve(root, data_dir, DEFAULT_DATA_DIR)
    resolved_output = _resolve(root, output_dir, DEFAULT_OUTPUT_DIR)
    horizons_list = [int(horizon) for horizon in horizons]
    panel, meta = build_factor_panel(
        resolved_data,
        horizons=horizons_list,
        max_symbols=max_symbols,
        start_date=start_date,
        end_date=end_date,
        recursive=recursive,
        warmup_days=warmup_days,
    )
    if panel.empty:
        raise ValueError(f"No factor rows built from {resolved_data}")
    factor_names = _valid_factor_names(panel, factors)
    if not factor_names:
        raise ValueError("No requested factors are available in the factor panel.")
    ic_table = compute_ic_table(panel, factors=factor_names, horizons=horizons_list, min_obs=min_obs)
    ic_summary = summarize_ic(ic_table)
    quantile_returns = compute_quantile_returns(panel, factors=factor_names, horizons=horizons_list, quantiles=quantiles, min_obs=min_obs)
    quantile_summary = summarize_quantiles(quantile_returns)
    correlation = compute_factor_correlation(panel, factors=factor_names)

    resolved_output.mkdir(parents=True, exist_ok=True)
    factor_panel_path = resolved_output / "factor_panel.csv"
    ic_path = resolved_output / "ic_by_date.csv"
    ic_summary_path = resolved_output / "ic_summary.csv"
    quantile_path = resolved_output / "quantile_returns.csv"
    quantile_summary_path = resolved_output / "quantile_summary.csv"
    correlation_path = resolved_output / "factor_correlation.csv"
    snapshot_path = resolved_output / "factor_lab_snapshot.json"
    report_path = resolved_output / "factor_lab.md"

    if save_panel:
        panel.to_csv(factor_panel_path, index=False)
    ic_table.to_csv(ic_path, index=False)
    ic_summary.to_csv(ic_summary_path, index=False)
    quantile_returns.to_csv(quantile_path, index=False)
    quantile_summary.to_csv(quantile_summary_path, index=False)
    correlation.to_csv(correlation_path)

    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(resolved_data),
        "output_dir": str(resolved_output),
        "file_count": meta["file_count"],
        "loaded_symbol_count": meta["loaded_symbol_count"],
        "panel_row_count": meta["row_count"],
        "failure_count": meta["failure_count"],
        "failures": meta["failures"],
        "start_date": str(panel["date"].min().date()) if not panel.empty else None,
        "end_date": str(panel["date"].max().date()) if not panel.empty else None,
        "factor_count": len(factor_names),
        "factors": factor_names,
        "horizons": horizons_list,
        "quantiles": int(quantiles),
        "min_obs": int(min_obs),
        "max_symbols": max_symbols,
        "warmup_days": int(warmup_days),
        "save_panel": bool(save_panel),
        "factor_panel_path": str(factor_panel_path) if save_panel else None,
        "ic_path": str(ic_path),
        "ic_summary_path": str(ic_summary_path),
        "quantile_path": str(quantile_path),
        "quantile_summary_path": str(quantile_summary_path),
        "correlation_path": str(correlation_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
    }
    if not ic_summary.empty:
        best_ic = ic_summary.sort_values("ic_abs_mean", ascending=False).iloc[0]
        snapshot["best_ic_factor"] = str(best_ic["factor"])
        snapshot["best_ic_horizon"] = int(best_ic["horizon"])
        snapshot["best_ic_abs_mean"] = float(best_ic["ic_abs_mean"])
    if not quantile_summary.empty:
        best_spread = quantile_summary.sort_values("long_short_mean", ascending=False).iloc[0]
        snapshot["best_spread_factor"] = str(best_spread["factor"])
        snapshot["best_spread_horizon"] = int(best_spread["horizon"])
        snapshot["best_spread_mean"] = float(best_spread["long_short_mean"])

    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, ic_summary, quantile_summary), encoding="utf-8")
    return FactorLabResult(
        output_dir=resolved_output,
        factor_panel_path=factor_panel_path,
        ic_path=ic_path,
        ic_summary_path=ic_summary_path,
        quantile_path=quantile_path,
        quantile_summary_path=quantile_summary_path,
        correlation_path=correlation_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        panel=panel,
        ic_summary=ic_summary,
        quantile_summary=quantile_summary,
    )
