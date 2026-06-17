"""Technical indicators used by baseline strategies."""

from __future__ import annotations

import pandas as pd


def moving_average(series: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be positive")
    return series.rolling(window=window, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be positive")
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    values = 100 - (100 / (1 + rs))
    values = values.mask((avg_loss == 0) & (avg_gain > 0), 100)
    values = values.mask((avg_loss == 0) & (avg_gain == 0), 50)
    return values

