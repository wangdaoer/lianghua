"""Research-only theme state and beta-gap diagnostics.

This module turns the video-derived ideas into measurable research artifacts:
theme breadth, second activation, and beta-gap laggard candidates. It does not
change allocations or connect to brokers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("data/processed/stocks")
DEFAULT_OUTPUT_DIR = Path("outputs/research/theme_state_latest")
DEFAULT_HORIZONS = [1, 5, 20]
DEFAULT_GROUP_COLUMNS = ["theme", "theme_group", "concept", "industry", "board", "sector", "block"]
THEME_SPLIT_PATTERN = re.compile(r"[;,，；|、]+")


@dataclass(frozen=True)
class ThemeStateResult:
    output_dir: Path
    symbol_panel_path: Path
    theme_state_path: Path
    beta_gap_path: Path
    beta_gap_ic_path: Path
    beta_gap_summary_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    symbol_panel: pd.DataFrame
    theme_state: pd.DataFrame
    beta_gap: pd.DataFrame
    beta_gap_summary: pd.DataFrame


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = default if path is None else Path(path)
    return raw if raw.is_absolute() else project_root / raw


def _parse_date(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(text, errors="coerce")
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


def _clean_code(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6) if digits else text


def _board_from_code(code: str) -> str:
    clean = _clean_code(code)
    if clean.startswith(("300", "301")):
        return "chinext"
    if clean.startswith(("688", "689")):
        return "star"
    if clean.startswith(("4", "8", "920")):
        return "bse"
    if clean.startswith(("000", "001", "002", "003", "600", "601", "603", "605")):
        return "main"
    return "unknown"


def _as_float(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not pd.notna(number):
        return default
    return number


def _read_theme_map(theme_map_path: Path | None) -> pd.DataFrame:
    if theme_map_path is None:
        return pd.DataFrame()
    if not theme_map_path.exists():
        raise FileNotFoundError(f"Theme map does not exist: {theme_map_path}")
    frame = pd.read_csv(theme_map_path)
    if "code" not in frame.columns:
        raise ValueError("theme map must contain a code column.")
    theme_column = next((column for column in DEFAULT_GROUP_COLUMNS if column in frame.columns), None)
    if theme_column is None:
        raise ValueError("theme map must contain one of: theme, concept, industry, board.")
    rows: list[dict[str, str]] = []
    for raw in frame[["code", theme_column]].to_dict(orient="records"):
        code = _clean_code(raw.get("code"))
        if not code:
            continue
        raw_group = str(raw.get(theme_column, "")).strip()
        if not raw_group or raw_group.lower() in {"nan", "none", "0"}:
            continue
        for group in THEME_SPLIT_PATTERN.split(raw_group):
            cleaned = group.strip()
            if cleaned and cleaned.lower() not in {"nan", "none", "0"}:
                rows.append({"code": code, "theme_group": cleaned})
    if not rows:
        return pd.DataFrame(columns=["code", "theme_group"])
    return pd.DataFrame(rows).drop_duplicates(subset=["code", "theme_group"], keep="last")


def _choose_group(data: pd.DataFrame, group_column: str | None) -> tuple[pd.Series, str]:
    if group_column and group_column in data.columns:
        values = data[group_column].astype(str).str.strip()
        return values.where(values != "", "unknown"), f"history_column:{group_column}"
    for column in DEFAULT_GROUP_COLUMNS:
        if column in data.columns:
            values = data[column].astype(str).str.strip()
            return values.where(values != "", "unknown"), f"history_column:{column}"
    return data["code"].map(_board_from_code), "board_fallback"


def _read_price_csv(path: Path, group_column: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "code", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].map(_clean_code)
    data["name"] = data.get("name", data["code"]).astype(str)
    for column in ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "code", "close"]).sort_values("date")
    data = data.loc[data["close"] > 0].drop_duplicates(subset=["date"], keep="last")
    data["theme_group"], source = _choose_group(data, group_column)
    data.attrs["group_source"] = source
    return data.reset_index(drop=True)


def _prepare_symbol_frame(
    prices: pd.DataFrame,
    horizons: Iterable[int],
    beta_window: int,
) -> pd.DataFrame:
    data = prices.sort_values("date").copy()
    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    amount = pd.to_numeric(data["amount"], errors="coerce")
    returns = close.pct_change()
    out = pd.DataFrame(
        {
            "date": data["date"],
            "code": data["code"].astype(str).str.zfill(6),
            "name": data["name"].astype(str),
            "theme_group": data["theme_group"].astype(str),
            "close": close,
            "amount": amount,
            "daily_return": returns,
            "above_ma20": close > close.rolling(20, min_periods=20).mean(),
            "above_ma60": close > close.rolling(60, min_periods=60).mean(),
            "momentum_20d": close / close.shift(20) - 1.0,
            "breakout_20d": close / high.shift(1).rolling(20, min_periods=20).max() - 1.0,
        }
    )
    for horizon in horizons:
        horizon_int = int(horizon)
        out[f"fwd_return_{horizon_int}d"] = close.shift(-horizon_int) / close - 1.0
    out["beta_window"] = int(beta_window)
    return out.replace([np.inf, -np.inf], np.nan)


def build_theme_symbol_panel(
    data_dir: str | Path,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    max_symbols: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    recursive: bool = False,
    warmup_days: int = 180,
    group_column: str | None = None,
    theme_map_path: str | Path | None = None,
    beta_window: int = 60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved_dir = Path(data_dir)
    paths = _limit_paths(_csv_paths(resolved_dir, recursive=recursive), max_symbols)
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    warmup_start = start - timedelta(days=int(warmup_days)) if start is not None and warmup_days > 0 else None
    theme_map = _read_theme_map(Path(theme_map_path) if theme_map_path is not None else None)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    group_sources: set[str] = set()
    for path in paths:
        try:
            prices = _read_price_csv(path, group_column=group_column)
            group_sources.add(str(prices.attrs.get("group_source", "unknown")))
            source_frames = [prices]
            if not theme_map.empty:
                code = str(prices["code"].iloc[0])
                matches = sorted(set(theme_map.loc[theme_map["code"] == code, "theme_group"].astype(str)))
                if matches:
                    source_frames = []
                    for group_name in matches:
                        mapped = prices.copy()
                        mapped["theme_group"] = group_name
                        source_frames.append(mapped)
                    group_sources.add("theme_map")
            for source_frame in source_frames:
                if warmup_start is not None:
                    source_frame = source_frame.loc[source_frame["date"] >= warmup_start]
                if end is not None:
                    source_frame = source_frame.loc[source_frame["date"] <= end]
                symbol_frame = _prepare_symbol_frame(source_frame, horizons=horizons, beta_window=beta_window)
                if start is not None:
                    symbol_frame = symbol_frame.loc[symbol_frame["date"] >= start]
                if end is not None:
                    symbol_frame = symbol_frame.loc[symbol_frame["date"] <= end]
                if not symbol_frame.empty:
                    frames.append(symbol_frame)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            failures.append({"path": str(path), "error": str(error)})
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not panel.empty:
        panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    meta = {
        "data_dir": str(resolved_dir),
        "file_count": len(paths),
        "loaded_symbol_count": int(panel["code"].nunique()) if not panel.empty else 0,
        "row_count": int(len(panel)),
        "group_count": int(panel["theme_group"].nunique()) if not panel.empty else 0,
        "group_sources": sorted(group_sources),
        "failure_count": len(failures),
        "failures": failures[:20],
    }
    return panel, meta


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not bool(valid.any()):
        return float(values.mean()) if not values.dropna().empty else np.nan
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def compute_theme_state(
    symbol_panel: pd.DataFrame,
    breadth_window: int = 20,
    slope_window: int = 5,
    quiet_lookback: int = 20,
    activation_breadth: float = 0.55,
    prior_breadth_ceiling: float = 0.55,
    min_breadth_slope: float = 0.03,
    min_rs_20d: float = 0.0,
) -> pd.DataFrame:
    if symbol_panel.empty:
        return pd.DataFrame()
    market_rows = []
    for date_value, group in symbol_panel.groupby("date", sort=True):
        market_rows.append(
            {
                "date": pd.Timestamp(date_value),
                "market_return": _weighted_average(group["daily_return"], group["amount"]),
            }
        )
    market = pd.DataFrame(market_rows)
    rows: list[dict[str, Any]] = []
    for (date_value, theme), group in symbol_panel.groupby(["date", "theme_group"], sort=True):
        amount = pd.to_numeric(group["amount"], errors="coerce").fillna(0.0)
        above_ma20 = group["above_ma20"].astype(float)
        above_ma60 = group["above_ma60"].astype(float)
        rows.append(
            {
                "date": pd.Timestamp(date_value),
                "theme_group": str(theme),
                "symbol_count": int(group["code"].nunique()),
                "theme_return": _weighted_average(group["daily_return"], amount),
                "equal_breadth_ma20": float(above_ma20.mean()),
                "weighted_breadth_ma20": _weighted_average(above_ma20, amount),
                "weighted_breadth_ma60": _weighted_average(above_ma60, amount),
                "avg_momentum_20d": float(pd.to_numeric(group["momentum_20d"], errors="coerce").mean()),
                "total_amount": float(amount.sum()),
            }
        )
    theme_state = pd.DataFrame(rows)
    if theme_state.empty:
        return theme_state
    theme_state = theme_state.merge(market, on="date", how="left")
    theme_state["relative_return"] = theme_state["theme_return"] - theme_state["market_return"]
    parts: list[pd.DataFrame] = []
    for _, group in theme_state.groupby("theme_group", sort=True):
        data = group.sort_values("date").copy()
        data["breadth_ma20_smooth"] = data["weighted_breadth_ma20"].rolling(
            breadth_window, min_periods=max(3, min(breadth_window, 10))
        ).mean()
        data["breadth_slope"] = data["breadth_ma20_smooth"] - data["breadth_ma20_smooth"].shift(slope_window)
        data["theme_rs_20d"] = data["relative_return"].rolling(20, min_periods=10).sum()
        data["prior_breadth_ma20_max"] = data["breadth_ma20_smooth"].shift(1).rolling(
            quiet_lookback, min_periods=max(3, min(quiet_lookback, 10))
        ).max()
        data["theme_second_activation"] = (
            (data["breadth_ma20_smooth"] >= float(activation_breadth))
            & (data["breadth_slope"] >= float(min_breadth_slope))
            & (data["theme_rs_20d"] >= float(min_rs_20d))
            & (data["prior_breadth_ma20_max"] <= float(prior_breadth_ceiling))
        )
        data["theme_state"] = np.select(
            [
                data["theme_second_activation"],
                (data["breadth_ma20_smooth"] >= activation_breadth) & (data["theme_rs_20d"] >= min_rs_20d),
                data["breadth_ma20_smooth"] < 0.35,
            ],
            ["second_activation", "healthy", "weak"],
            default="neutral",
        )
        parts.append(data)
    return pd.concat(parts, ignore_index=True).sort_values(["date", "theme_group"]).reset_index(drop=True)


def compute_beta_gap(
    symbol_panel: pd.DataFrame,
    theme_state: pd.DataFrame,
    beta_window: int = 60,
    theme_active_only: bool = True,
) -> pd.DataFrame:
    if symbol_panel.empty or theme_state.empty:
        return pd.DataFrame()
    theme_returns = theme_state[["date", "theme_group", "theme_return", "theme_state", "theme_second_activation"]].copy()
    data = symbol_panel.merge(theme_returns, on=["date", "theme_group"], how="left")
    parts: list[pd.DataFrame] = []
    for _, group in data.groupby(["code", "theme_group"], sort=True):
        symbol = group.sort_values("date").copy()
        stock_ret = pd.to_numeric(symbol["daily_return"], errors="coerce")
        theme_ret = pd.to_numeric(symbol["theme_return"], errors="coerce")
        min_periods = min(int(beta_window), max(3, min(30, int(beta_window))))
        prior_stock_ret = stock_ret.shift(1)
        prior_theme_ret = theme_ret.shift(1)
        cov = prior_stock_ret.rolling(beta_window, min_periods=min_periods).cov(prior_theme_ret)
        var = prior_theme_ret.rolling(beta_window, min_periods=min_periods).var()
        symbol["beta_rolling"] = cov / var.replace(0.0, np.nan)
        symbol["beta_gap"] = symbol["beta_rolling"] * theme_ret - stock_ret
        symbol["beta_gap_positive"] = symbol["beta_gap"] > 0
        parts.append(symbol)
    beta_gap = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    if theme_active_only:
        beta_gap = beta_gap.loc[beta_gap["theme_state"].isin(["second_activation", "healthy"])]
    keep_columns = [
        "date",
        "code",
        "name",
        "theme_group",
        "theme_state",
        "theme_second_activation",
        "daily_return",
        "theme_return",
        "beta_rolling",
        "beta_gap",
        "beta_gap_positive",
        "momentum_20d",
        "breakout_20d",
        "amount",
    ]
    for horizon in DEFAULT_HORIZONS:
        column = f"fwd_return_{horizon}d"
        if column in beta_gap.columns and column not in keep_columns:
            keep_columns.append(column)
    beta_gap = beta_gap[[column for column in keep_columns if column in beta_gap.columns]]
    return beta_gap.sort_values(["date", "beta_gap"], ascending=[True, False]).reset_index(drop=True)


def compute_beta_gap_evaluation(
    beta_gap: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    quantiles: int = 5,
    min_obs: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate beta_gap as candidate-generation evidence by date and horizon."""
    if beta_gap.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        horizon_int = int(horizon)
        target = f"fwd_return_{horizon_int}d"
        if target not in beta_gap.columns:
            continue
        subset = beta_gap[["date", "theme_group", "theme_state", "beta_gap", target]].dropna()
        for date_value, group in subset.groupby("date", sort=True):
            if len(group) < min_obs or group["beta_gap"].nunique() < max(3, min(quantiles, len(group))):
                continue
            ic = group["beta_gap"].corr(group[target], method="spearman")
            try:
                ranked = group.copy()
                ranked["quantile"] = pd.qcut(
                    ranked["beta_gap"].rank(method="first"),
                    min(quantiles, len(group)),
                    labels=False,
                ) + 1
            except ValueError:
                continue
            top_bucket = int(ranked["quantile"].max())
            bottom = pd.to_numeric(ranked.loc[ranked["quantile"] == 1, target], errors="coerce").mean()
            top = pd.to_numeric(ranked.loc[ranked["quantile"] == top_bucket, target], errors="coerce").mean()
            rows.append(
                {
                    "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                    "horizon": horizon_int,
                    "ic": float(ic) if pd.notna(ic) else np.nan,
                    "top_return": float(top) if pd.notna(top) else np.nan,
                    "bottom_return": float(bottom) if pd.notna(bottom) else np.nan,
                    "long_short_return": float(top - bottom) if pd.notna(top) and pd.notna(bottom) else np.nan,
                    "obs": int(len(group)),
                    "active_theme_count": int(group["theme_group"].nunique()),
                }
            )
    by_date = pd.DataFrame(rows)
    if by_date.empty:
        return by_date, pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    for horizon, group in by_date.groupby("horizon", sort=True):
        ic = pd.to_numeric(group["ic"], errors="coerce").dropna()
        spread = pd.to_numeric(group["long_short_return"], errors="coerce").dropna()
        summary_rows.append(
            {
                "factor": "beta_gap",
                "horizon": int(horizon),
                "ic_mean": float(ic.mean()) if not ic.empty else np.nan,
                "ic_abs_mean": float(ic.abs().mean()) if not ic.empty else np.nan,
                "ic_positive_ratio": float((ic > 0).mean()) if not ic.empty else np.nan,
                "long_short_mean": float(spread.mean()) if not spread.empty else np.nan,
                "long_short_win_ratio": float((spread > 0).mean()) if not spread.empty else np.nan,
                "obs_dates": int(len(group)),
                "avg_cross_section_obs": float(pd.to_numeric(group["obs"], errors="coerce").mean()),
                "avg_active_theme_count": float(pd.to_numeric(group["active_theme_count"], errors="coerce").mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("horizon").reset_index(drop=True)
    return by_date, summary


def _latest_rows(frame: pd.DataFrame, date_column: str = "date", max_rows: int = 12) -> pd.DataFrame:
    if frame.empty or date_column not in frame.columns:
        return pd.DataFrame()
    latest = frame[date_column].max()
    return frame.loc[frame[date_column] == latest].head(max_rows).copy()


def _num(value: Any, digits: int = 4) -> str:
    number = _as_float(value)
    if not pd.notna(number):
        return "N/A"
    return f"{number:.{digits}f}"


def _pct(value: Any) -> str:
    number = _as_float(value)
    if not pd.notna(number):
        return "N/A"
    return f"{number * 100:.2f}%"


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    if frame.empty:
        return "No rows."
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].head(max_rows)
    lines = ["| " + " | ".join(subset.columns) + " |", "| " + " | ".join(["---"] * len(subset.columns)) + " |"]
    for row in subset.itertuples(index=False):
        values = []
        for column, value in zip(subset.columns, row):
            if column in {
                "theme_return",
                "market_return",
                "relative_return",
                "theme_rs_20d",
                "daily_return",
                "beta_gap",
                "long_short_mean",
            }:
                values.append(_pct(value))
            elif column in {"ic_positive_ratio", "long_short_win_ratio"}:
                values.append(_pct(value))
            elif column in {
                "weighted_breadth_ma20",
                "breadth_ma20_smooth",
                "breadth_slope",
                "beta_rolling",
                "ic_mean",
                "ic_abs_mean",
            }:
                values.append(_num(value, 4))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _render_report(
    snapshot: dict[str, Any],
    theme_state: pd.DataFrame,
    beta_gap: pd.DataFrame,
    beta_gap_summary: pd.DataFrame,
) -> str:
    latest_theme = _latest_rows(
        theme_state.sort_values(["date", "theme_second_activation", "breadth_ma20_smooth"], ascending=[True, False, False]),
        max_rows=12,
    )
    latest_beta = _latest_rows(beta_gap.sort_values(["date", "beta_gap"], ascending=[True, False]), max_rows=20)
    activation_count = int(snapshot.get("latest_second_activation_count") or 0)
    return f"""# Theme State Research

Generated at: `{snapshot.get("generated_at")}`

This report is research-only. It does not connect to brokers, place orders, or change live allocation.

## Summary

| Item | Value |
| --- | ---: |
| Status | `{snapshot.get("status")}` |
| Data dir | `{snapshot.get("data_dir")}` |
| Group source | `{snapshot.get("group_source")}` |
| Symbols loaded | {snapshot.get("loaded_symbol_count")} / {snapshot.get("file_count")} |
| Groups | {snapshot.get("group_count")} |
| Date range | {snapshot.get("start_date")} to {snapshot.get("end_date")} |
| Latest date | {snapshot.get("latest_date")} |
| Latest second-activation groups | {activation_count} |
| Beta-gap rows | {snapshot.get("beta_gap_row_count")} |
| Theme-active only | `{snapshot.get("theme_active_only")}` |
| Research only | `{snapshot.get("research_only")}` |

## Latest Theme State

{_markdown_table(latest_theme, ["theme_group", "theme_state", "symbol_count", "theme_return", "weighted_breadth_ma20", "breadth_ma20_smooth", "breadth_slope", "theme_rs_20d", "theme_second_activation"], 12)}

## Latest Beta-Gap Candidates

{_markdown_table(latest_beta, ["code", "name", "theme_group", "theme_state", "daily_return", "theme_return", "beta_rolling", "beta_gap", "momentum_20d"], 20)}

## Beta-Gap Evaluation

{_markdown_table(beta_gap_summary, ["factor", "horizon", "ic_mean", "ic_abs_mean", "ic_positive_ratio", "long_short_mean", "long_short_win_ratio", "obs_dates", "avg_cross_section_obs"], 12)}

## Interpretation

- `theme_second_activation` is candidate-generation evidence only.
- `beta_gap` is a tactical laggard-elasticity signal and must stay gated by theme health, liquidity, stop loss, and holding-time rules.
- These outputs should be promoted only after smoke, full walk-forward, and portfolio-level comparison.

## Files

- Symbol panel: `{snapshot.get("symbol_panel_path")}`
- Theme state CSV: `{snapshot.get("theme_state_path")}`
- Beta-gap CSV: `{snapshot.get("beta_gap_path")}`
- Beta-gap IC by date: `{snapshot.get("beta_gap_ic_path")}`
- Beta-gap summary: `{snapshot.get("beta_gap_summary_path")}`
- Snapshot JSON: `{snapshot.get("snapshot_path")}`
"""


def run_theme_state(
    project_root: str | Path = Path("."),
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    theme_map_path: str | Path | None = None,
    group_column: str | None = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    max_symbols: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    recursive: bool = False,
    warmup_days: int = 180,
    breadth_window: int = 20,
    slope_window: int = 5,
    quiet_lookback: int = 20,
    activation_breadth: float = 0.55,
    prior_breadth_ceiling: float = 0.55,
    min_breadth_slope: float = 0.03,
    min_rs_20d: float = 0.0,
    beta_window: int = 60,
    theme_active_only: bool = True,
) -> ThemeStateResult:
    root = Path(project_root).resolve()
    resolved_data = _resolve(root, data_dir, DEFAULT_DATA_DIR)
    resolved_output = _resolve(root, output_dir, DEFAULT_OUTPUT_DIR)
    resolved_theme_map = _resolve(root, theme_map_path, Path("")) if theme_map_path is not None else None
    horizons_list = [int(horizon) for horizon in horizons]
    symbol_panel, meta = build_theme_symbol_panel(
        resolved_data,
        horizons=horizons_list,
        max_symbols=max_symbols,
        start_date=start_date,
        end_date=end_date,
        recursive=recursive,
        warmup_days=warmup_days,
        group_column=group_column,
        theme_map_path=resolved_theme_map,
        beta_window=beta_window,
    )
    if symbol_panel.empty:
        raise ValueError(f"No theme rows built from {resolved_data}")
    theme_state = compute_theme_state(
        symbol_panel,
        breadth_window=breadth_window,
        slope_window=slope_window,
        quiet_lookback=quiet_lookback,
        activation_breadth=activation_breadth,
        prior_breadth_ceiling=prior_breadth_ceiling,
        min_breadth_slope=min_breadth_slope,
        min_rs_20d=min_rs_20d,
    )
    beta_gap = compute_beta_gap(
        symbol_panel=symbol_panel,
        theme_state=theme_state,
        beta_window=beta_window,
        theme_active_only=theme_active_only,
    )

    resolved_output.mkdir(parents=True, exist_ok=True)
    symbol_panel_path = resolved_output / "theme_symbol_panel.csv"
    theme_state_path = resolved_output / "theme_state.csv"
    beta_gap_path = resolved_output / "beta_gap_candidates.csv"
    beta_gap_ic_path = resolved_output / "beta_gap_ic_by_date.csv"
    beta_gap_summary_path = resolved_output / "beta_gap_summary.csv"
    snapshot_path = resolved_output / "theme_state_snapshot.json"
    report_path = resolved_output / "theme_state.md"

    symbol_panel.to_csv(symbol_panel_path, index=False)
    theme_state.to_csv(theme_state_path, index=False)
    beta_gap.to_csv(beta_gap_path, index=False)
    beta_gap_ic, beta_gap_summary = compute_beta_gap_evaluation(
        beta_gap=beta_gap,
        horizons=horizons_list,
        quantiles=5,
        min_obs=30,
    )
    beta_gap_ic.to_csv(beta_gap_ic_path, index=False)
    beta_gap_summary.to_csv(beta_gap_summary_path, index=False)

    latest_date = str(pd.Timestamp(theme_state["date"].max()).date()) if not theme_state.empty else None
    latest_theme = theme_state.loc[theme_state["date"] == theme_state["date"].max()] if not theme_state.empty else pd.DataFrame()
    group_source = "mixed:" + ",".join(meta["group_sources"]) if len(meta["group_sources"]) > 1 else (meta["group_sources"][0] if meta["group_sources"] else "unknown")
    best_beta_gap_horizon = None
    best_beta_gap_spread = None
    if not beta_gap_summary.empty:
        best_row = beta_gap_summary.sort_values("long_short_mean", ascending=False).iloc[0]
        best_beta_gap_horizon = int(best_row["horizon"])
        best_beta_gap_spread = _as_float(best_row["long_short_mean"])
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok",
        "research_only": True,
        "broker_action": "none",
        "data_dir": str(resolved_data),
        "output_dir": str(resolved_output),
        "theme_map_path": str(resolved_theme_map) if resolved_theme_map is not None else None,
        "group_source": group_source,
        "file_count": meta["file_count"],
        "loaded_symbol_count": meta["loaded_symbol_count"],
        "panel_row_count": meta["row_count"],
        "group_count": meta["group_count"],
        "failure_count": meta["failure_count"],
        "failures": meta["failures"],
        "start_date": str(pd.Timestamp(symbol_panel["date"].min()).date()),
        "end_date": str(pd.Timestamp(symbol_panel["date"].max()).date()),
        "latest_date": latest_date,
        "latest_second_activation_count": int(latest_theme["theme_second_activation"].sum()) if not latest_theme.empty else 0,
        "latest_healthy_group_count": int((latest_theme["theme_state"] == "healthy").sum()) if not latest_theme.empty else 0,
        "theme_state_row_count": int(len(theme_state)),
        "beta_gap_row_count": int(len(beta_gap)),
        "beta_gap_ic_row_count": int(len(beta_gap_ic)),
        "beta_gap_summary_row_count": int(len(beta_gap_summary)),
        "best_beta_gap_horizon": best_beta_gap_horizon,
        "best_beta_gap_long_short_mean": best_beta_gap_spread,
        "horizons": horizons_list,
        "breadth_window": int(breadth_window),
        "slope_window": int(slope_window),
        "quiet_lookback": int(quiet_lookback),
        "activation_breadth": float(activation_breadth),
        "prior_breadth_ceiling": float(prior_breadth_ceiling),
        "min_breadth_slope": float(min_breadth_slope),
        "min_rs_20d": float(min_rs_20d),
        "beta_window": int(beta_window),
        "theme_active_only": bool(theme_active_only),
        "symbol_panel_path": str(symbol_panel_path),
        "theme_state_path": str(theme_state_path),
        "beta_gap_path": str(beta_gap_path),
        "beta_gap_ic_path": str(beta_gap_ic_path),
        "beta_gap_summary_path": str(beta_gap_summary_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "promotion_note": "Candidate-generation evidence only; requires smoke, full walk-forward, and portfolio-level comparison.",
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, theme_state, beta_gap, beta_gap_summary), encoding="utf-8")
    return ThemeStateResult(
        output_dir=resolved_output,
        symbol_panel_path=symbol_panel_path,
        theme_state_path=theme_state_path,
        beta_gap_path=beta_gap_path,
        beta_gap_ic_path=beta_gap_ic_path,
        beta_gap_summary_path=beta_gap_summary_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        symbol_panel=symbol_panel,
        theme_state=theme_state,
        beta_gap=beta_gap,
        beta_gap_summary=beta_gap_summary,
    )
