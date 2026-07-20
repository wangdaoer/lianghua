"""Failure-filter candidates for chip-reversal research.

The filters in this module are signal-day filters only. They must not use
forward return labels or post-entry trade outcomes when deciding whether an
event can enter a candidate backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


SIGNAL_FILTER_COLUMNS = {
    "chip_reversal_score",
    "daily_return_pct",
    "drawdown_20d",
    "cost_gap_ma20",
    "board",
    "score_bucket",
}


@dataclass(frozen=True)
class FailureFilterSpec:
    name: str
    description: str
    min_chip_score: float | None = None
    min_daily_gain_pct: float | None = None
    max_daily_gain_pct: float | None = None
    excluded_daily_gain_ranges: tuple[tuple[float, float], ...] = ()
    boards: tuple[str, ...] = ()


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def apply_failure_filter(events: pd.DataFrame, spec: FailureFilterSpec) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply one pre-trade filter candidate and return an audit payload."""

    data = events.copy()
    mask = pd.Series(True, index=data.index)
    used_columns: set[str] = set()

    if spec.min_chip_score is not None:
        used_columns.add("chip_reversal_score")
        mask &= _numeric_series(data, "chip_reversal_score") >= float(spec.min_chip_score)

    if spec.min_daily_gain_pct is not None:
        used_columns.add("daily_return_pct")
        mask &= _numeric_series(data, "daily_return_pct") >= float(spec.min_daily_gain_pct)

    if spec.max_daily_gain_pct is not None:
        used_columns.add("daily_return_pct")
        mask &= _numeric_series(data, "daily_return_pct") < float(spec.max_daily_gain_pct)

    if spec.excluded_daily_gain_ranges:
        used_columns.add("daily_return_pct")
        gain = _numeric_series(data, "daily_return_pct")
        for lower, upper in spec.excluded_daily_gain_ranges:
            mask &= ~((gain >= float(lower)) & (gain < float(upper)))

    if spec.boards:
        used_columns.add("board")
        allowed = {str(board) for board in spec.boards}
        mask &= data.get("board", pd.Series("", index=data.index)).astype(str).isin(allowed)

    filtered = data.loc[mask].copy()
    input_count = int(len(data))
    kept_count = int(len(filtered))
    audit = {
        "filter_name": spec.name,
        "filter_description": spec.description,
        "input_event_count": input_count,
        "kept_event_count": kept_count,
        "removed_event_count": int(input_count - kept_count),
        "kept_event_ratio": float(kept_count / input_count) if input_count else 0.0,
        "used_filter_columns": sorted(used_columns & SIGNAL_FILTER_COLUMNS),
        "research_only": True,
        "broker_action": "none",
    }
    return filtered, audit


def rank_filter_results(results: pd.DataFrame, *, max_drawdown_limit: float) -> pd.DataFrame:
    """Rank filter backtests, selecting highest return inside the drawdown gate."""

    if max_drawdown_limit < 0:
        raise ValueError("max_drawdown_limit must be non-negative.")
    if results.empty:
        return results.copy()

    ranked = results.copy()
    ranked["max_drawdown"] = pd.to_numeric(ranked["max_drawdown"], errors="coerce")
    ranked["total_return"] = pd.to_numeric(ranked["total_return"], errors="coerce")
    ranked["sharpe"] = pd.to_numeric(ranked.get("sharpe"), errors="coerce")
    ranked["drawdown_abs"] = ranked["max_drawdown"].abs()
    ranked["passes_drawdown_gate"] = ranked["max_drawdown"] >= -float(max_drawdown_limit)
    ranked["selection_status"] = np.where(
        ranked["passes_drawdown_gate"],
        "eligible_under_drawdown_gate",
        "rejected_drawdown_over_limit",
    )

    ranked = ranked.sort_values(
        ["passes_drawdown_gate", "total_return", "sharpe", "drawdown_abs"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    if not ranked.empty and bool(ranked.loc[0, "passes_drawdown_gate"]):
        ranked.loc[0, "selection_status"] = "selected_under_drawdown_gate"
    return ranked


def generate_year_walk_forward_windows(
    start_year: int,
    end_year: int,
    *,
    train_years: int = 3,
    test_years: int = 1,
) -> list[dict[str, str]]:
    """Generate non-leaky train-then-test calendar-year windows."""

    if train_years <= 0:
        raise ValueError("train_years must be positive.")
    if test_years <= 0:
        raise ValueError("test_years must be positive.")
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year.")

    windows: list[dict[str, str]] = []
    latest_train_start = end_year - train_years - test_years + 1
    for train_start_year in range(start_year, latest_train_start + 1):
        train_end_year = train_start_year + train_years - 1
        test_start_year = train_end_year + 1
        test_end_year = test_start_year + test_years - 1
        windows.append(
            {
                "window": (
                    f"train_{train_start_year}_{train_end_year}_"
                    f"test_{test_start_year}_{test_end_year}"
                ),
                "train_start": f"{train_start_year}-01-01",
                "train_end": f"{train_end_year}-12-31",
                "test_start": f"{test_start_year}-01-01",
                "test_end": f"{test_end_year}-12-31",
            }
        )
    return windows


def equity_segment_metrics(curve: pd.DataFrame, start_date: str, end_date: str) -> dict[str, Any]:
    """Calculate rebased metrics for a date slice of an equity curve."""

    if curve.empty:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": None,
            "average_exposure": 0.0,
            "trading_days": 0,
        }
    data = curve.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["equity"] = pd.to_numeric(data["equity"], errors="coerce")
    data = data.dropna(subset=["date", "equity"]).sort_values("date")
    mask = (data["date"] >= pd.Timestamp(start_date)) & (data["date"] <= pd.Timestamp(end_date))
    segment = data.loc[mask].copy()
    if segment.empty:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": None,
            "average_exposure": 0.0,
            "trading_days": 0,
        }

    equity = segment["equity"].astype(float)
    rebased = equity / float(equity.iloc[0])
    returns = rebased.pct_change().fillna(0.0)
    drawdown = rebased / rebased.cummax() - 1.0
    exposure = (
        pd.to_numeric(segment["exposure"], errors="coerce").dropna()
        if "exposure" in segment.columns
        else pd.Series(dtype="float64")
    )
    sharpe = None
    if len(returns) > 1 and float(returns.std()) != 0.0:
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std())
    return {
        "total_return": float(rebased.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe": sharpe,
        "average_exposure": float(exposure.mean()) if not exposure.empty else 0.0,
        "trading_days": int(len(segment)),
    }


def build_walk_forward_selection(
    windows: list[dict[str, str]],
    candidate_metrics: pd.DataFrame,
    *,
    max_drawdown_limit: float,
) -> pd.DataFrame:
    """Select candidates on training windows and report their next-window test metrics."""

    rows: list[dict[str, Any]] = []
    if candidate_metrics.empty:
        return pd.DataFrame(rows)

    for window in windows:
        window_name = str(window["window"])
        part = candidate_metrics[candidate_metrics["window"].astype(str) == window_name].copy()
        train = part[part["split"].astype(str) == "train"].copy()
        test = part[part["split"].astype(str) == "test"].copy()
        ranked_train = rank_filter_results(train.rename(columns={"candidate": "candidate"}), max_drawdown_limit=max_drawdown_limit)
        if ranked_train.empty:
            continue
        selected = ranked_train.iloc[0]
        selected_candidate = str(selected["candidate"])
        selected_test = test[test["candidate"].astype(str) == selected_candidate]
        test_row = selected_test.iloc[0] if not selected_test.empty else pd.Series(dtype="object")
        rows.append(
            {
                **window,
                "selected_candidate": selected_candidate,
                "train_total_return": selected.get("total_return"),
                "train_max_drawdown": selected.get("max_drawdown"),
                "train_sharpe": selected.get("sharpe"),
                "train_passes_drawdown_gate": selected.get("passes_drawdown_gate"),
                "test_total_return": test_row.get("total_return", np.nan),
                "test_max_drawdown": test_row.get("max_drawdown", np.nan),
                "test_sharpe": test_row.get("sharpe", np.nan),
                "test_average_exposure": test_row.get("average_exposure", np.nan),
                "test_trading_days": test_row.get("trading_days", 0),
                "selection_status": selected.get("selection_status"),
            }
        )
    return pd.DataFrame(rows)


def dd37_filter_specs() -> tuple[FailureFilterSpec, ...]:
    """Default H2 aggressive candidates for the 37% max-drawdown allowance."""

    return (
        FailureFilterSpec(
            name="baseline_no_filter",
            description="H2 aggressive baseline; no additional failure filter.",
        ),
        FailureFilterSpec(
            name="chip_score_gte_0p18",
            description="Keep events with chip reversal score >= 0.18.",
            min_chip_score=0.18,
        ),
        FailureFilterSpec(
            name="chip_score_gte_0p20",
            description="Keep events with chip reversal score >= 0.20.",
            min_chip_score=0.20,
        ),
        FailureFilterSpec(
            name="chip_score_gte_0p24",
            description="Keep events with chip reversal score >= 0.24.",
            min_chip_score=0.24,
        ),
        FailureFilterSpec(
            name="exclude_gain_9_11",
            description="Exclude signal-day gain in [9%, 11%), the weak bucket seen in the top-three review.",
            excluded_daily_gain_ranges=((9.0, 11.0),),
        ),
        FailureFilterSpec(
            name="chip20_exclude_gain_9_11",
            description="Require chip score >= 0.20 and exclude signal-day gain in [9%, 11%).",
            min_chip_score=0.20,
            excluded_daily_gain_ranges=((9.0, 11.0),),
        ),
        FailureFilterSpec(
            name="chip24_exclude_gain_9_11",
            description="Require chip score >= 0.24 and exclude signal-day gain in [9%, 11%).",
            min_chip_score=0.24,
            excluded_daily_gain_ranges=((9.0, 11.0),),
        ),
        FailureFilterSpec(
            name="gain_gte_11",
            description="Keep only signal-day gain >= 11%; high-conviction but lower-capacity candidate.",
            min_daily_gain_pct=11.0,
        ),
        FailureFilterSpec(
            name="chip20_gain_gte_11",
            description="Require chip score >= 0.20 and signal-day gain >= 11%.",
            min_chip_score=0.20,
            min_daily_gain_pct=11.0,
        ),
    )
