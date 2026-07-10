"""Shared next-open labels and open-execution constraints."""

from __future__ import annotations

import pandas as pd


def limit_thresholds(columns: pd.Index) -> pd.Series:
    values = {}
    for code in columns:
        text = str(code).zfill(6)
        values[code] = 0.20 if text.startswith(("300", "301")) else 0.10
    return pd.Series(values, dtype=float)


def next_open_return_label(
    open_px: pd.DataFrame,
    max_abs_daily_return: float | None = None,
) -> pd.DataFrame:
    label = open_px.shift(-2) / (open_px.shift(-1) + 1e-12) - 1.0
    if max_abs_daily_return is not None and max_abs_daily_return > 0:
        label = label.mask(label.abs().gt(float(max_abs_daily_return)))
    return label


def apply_open_constraints(
    current: pd.Series,
    target: pd.Series,
    open_row: pd.Series,
    prev_close_row: pd.Series,
    max_buy_open_gap: float | None,
    limit_buffer: float,
    block_limit_up_buys: bool = True,
    block_limit_down_sells: bool = True,
) -> pd.Series:
    adjusted = target.copy()
    if not (block_limit_up_buys or block_limit_down_sells or max_buy_open_gap is not None):
        return adjusted

    current = current.reindex(target.index).fillna(0.0)
    open_gap = open_row.reindex(target.index) / (prev_close_row.reindex(target.index) + 1e-12) - 1.0
    limit = limit_thresholds(target.index)

    increasing = target.gt(current)
    decreasing = target.lt(current)
    if block_limit_up_buys:
        buy_blocked = increasing & open_gap.ge(limit * float(limit_buffer))
        adjusted = adjusted.where(~buy_blocked, current)
    if max_buy_open_gap is not None:
        gap_blocked = increasing & open_gap.gt(float(max_buy_open_gap))
        adjusted = adjusted.where(~gap_blocked, current)
    if block_limit_down_sells:
        sell_blocked = decreasing & open_gap.le(-limit * float(limit_buffer))
        adjusted = adjusted.where(~sell_blocked, current)
    return adjusted
