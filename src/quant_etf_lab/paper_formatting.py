"""Formatting and scalar-normalization helpers for paper-account reports."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _parse_yyyymmdd(value: Any) -> pd.Timestamp:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if len(text) == 8 and text.isdigit():
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(value, errors="coerce")


def _format_yyyymmdd(value: Any) -> str:
    parsed = _parse_yyyymmdd(value)
    if pd.isna(parsed):
        return str(value)
    return pd.Timestamp(parsed).strftime("%Y%m%d")


def _format_date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _pct(value: Any) -> str:
    return f"{_as_float(value) * 100:.2f}%"


def _num(value: Any, digits: int = 2) -> str:
    return f"{_as_float(value):.{digits}f}"


def _signed_pct(value: Any) -> str:
    return f"{_as_float(value) * 100:+.2f} pp"


def _maybe_pct(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return "N/A"
    return f"{number * 100:.2f}%"


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else default


def _md_text(value: Any, max_len: int = 160) -> str:
    text = _safe_text(value, "N/A").replace("\n", " ").replace("|", "\\|")
    return text if len(text) <= max_len else f"{text[: max_len - 3]}..."


def _negative_threshold(value: float, default: float) -> float:
    number = _as_float(value, default)
    return -abs(number)
