"""Portfolio-level risk controls for research backtests."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ETFSpec, LabConfig, RiskConfig
from .data import load_cached_history


@dataclass(frozen=True)
class RiskDecision:
    exposure: float
    drawdown: float
    peak_equity: float
    protected: bool
    reason: str
    benchmark_close: float | None
    benchmark_ma: float | None
    benchmark_return: float | None
    benchmark_rsrs_beta: float | None
    benchmark_rsrs_zscore: float | None
    benchmark_rsrs_risk_on: bool
    benchmark_risk_on: bool


def clamp_exposure(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def build_benchmark_risk_frame(config: LabConfig, dates: list[pd.Timestamp]) -> pd.DataFrame:
    risk = config.risk
    if not risk.enabled:
        return pd.DataFrame()
    benchmark = ETFSpec(
        code=risk.benchmark_code,
        name=risk.benchmark_name,
        asset_type=risk.benchmark_asset_type,
    )
    frame = load_cached_history(config, benchmark).sort_values("date").copy()
    frame["benchmark_ma"] = frame["close"].rolling(
        risk.benchmark_ma_window,
        min_periods=risk.benchmark_ma_window,
    ).mean()
    frame["benchmark_return"] = frame["close"].pct_change(risk.benchmark_drop_window)
    frame["benchmark_risk_on"] = (frame["close"] > frame["benchmark_ma"]) | frame["benchmark_ma"].isna()
    frame["benchmark_exposure"] = 1.0
    frame["benchmark_reason"] = "benchmark_on"
    off_mask = ~frame["benchmark_risk_on"]
    crash_mask = off_mask & (frame["benchmark_return"] <= risk.benchmark_drop_threshold)
    frame.loc[off_mask, "benchmark_exposure"] = clamp_exposure(risk.benchmark_off_exposure)
    frame.loc[off_mask, "benchmark_reason"] = "benchmark_below_ma"
    frame.loc[crash_mask, "benchmark_exposure"] = clamp_exposure(risk.benchmark_crash_exposure)
    frame.loc[crash_mask, "benchmark_reason"] = "benchmark_crash"
    frame["benchmark_rsrs_beta"] = pd.NA
    frame["benchmark_rsrs_zscore"] = pd.NA
    frame["benchmark_rsrs_risk_on"] = True
    if risk.benchmark_rsrs_enabled:
        low = pd.to_numeric(frame["low"], errors="coerce")
        high = pd.to_numeric(frame["high"], errors="coerce")
        covariance = low.rolling(risk.benchmark_rsrs_window, min_periods=risk.benchmark_rsrs_window).cov(high)
        variance = low.rolling(risk.benchmark_rsrs_window, min_periods=risk.benchmark_rsrs_window).var()
        beta = covariance / variance.replace(0, pd.NA)
        beta_mean = beta.rolling(
            risk.benchmark_rsrs_zscore_window,
            min_periods=risk.benchmark_rsrs_zscore_window,
        ).mean()
        beta_std = beta.rolling(
            risk.benchmark_rsrs_zscore_window,
            min_periods=risk.benchmark_rsrs_zscore_window,
        ).std()
        rsrs_zscore = (beta - beta_mean) / beta_std.replace(0, pd.NA)
        # Missing RSRS values are treated as risk-on so the filter only reduces
        # exposure when the rolling regression has enough confirmed data.
        rsrs_risk_on = rsrs_zscore.isna() | (rsrs_zscore > risk.benchmark_rsrs_threshold)
        rsrs_off_mask = ~rsrs_risk_on
        frame["benchmark_rsrs_beta"] = beta
        frame["benchmark_rsrs_zscore"] = rsrs_zscore
        frame["benchmark_rsrs_risk_on"] = rsrs_risk_on
        frame["benchmark_risk_on"] = frame["benchmark_risk_on"] & rsrs_risk_on
        frame.loc[rsrs_off_mask, "benchmark_exposure"] = frame.loc[
            rsrs_off_mask, "benchmark_exposure"
        ].clip(upper=clamp_exposure(risk.benchmark_rsrs_off_exposure))
        frame.loc[rsrs_off_mask, "benchmark_reason"] = frame.loc[rsrs_off_mask, "benchmark_reason"].map(
            lambda reason: "benchmark_rsrs_off" if reason == "benchmark_on" else f"{reason};benchmark_rsrs_off"
        )

    indexed = frame.set_index("date")[
        [
            "close",
            "benchmark_ma",
            "benchmark_return",
            "benchmark_rsrs_beta",
            "benchmark_rsrs_zscore",
            "benchmark_rsrs_risk_on",
            "benchmark_risk_on",
            "benchmark_exposure",
            "benchmark_reason",
        ]
    ].rename(columns={"close": "benchmark_close"})
    indexed = indexed.reindex(pd.DatetimeIndex(dates)).ffill()
    indexed["benchmark_rsrs_risk_on"] = indexed["benchmark_rsrs_risk_on"].fillna(True).astype(bool)
    indexed["benchmark_risk_on"] = indexed["benchmark_risk_on"].fillna(True).astype(bool)
    indexed["benchmark_exposure"] = indexed["benchmark_exposure"].fillna(1.0)
    indexed["benchmark_reason"] = indexed["benchmark_reason"].fillna("benchmark_unknown")
    return indexed


def drawdown_exposure(risk: RiskConfig, drawdown: float) -> tuple[float, str]:
    exposure = 1.0
    reason = "drawdown_ok"
    for threshold, level_exposure in risk.drawdown_levels:
        if drawdown >= threshold:
            exposure = clamp_exposure(level_exposure)
            reason = f"drawdown_{threshold:.0%}"
    return exposure, reason


def decide_next_risk(
    risk: RiskConfig,
    equity: float,
    peak_equity: float,
    protected: bool,
    benchmark_row: pd.Series | None,
) -> RiskDecision:
    peak = max(float(peak_equity), float(equity))
    drawdown = 0.0 if peak <= 0 else max(0.0, 1 - float(equity) / peak)

    benchmark_exposure = 1.0
    benchmark_reason = "benchmark_unknown"
    benchmark_close = None
    benchmark_ma = None
    benchmark_return = None
    benchmark_rsrs_beta = None
    benchmark_rsrs_zscore = None
    benchmark_rsrs_risk_on = True
    benchmark_risk_on = True
    if benchmark_row is not None:
        benchmark_exposure = clamp_exposure(float(benchmark_row.get("benchmark_exposure", 1.0)))
        benchmark_reason = str(benchmark_row.get("benchmark_reason", "benchmark_unknown"))
        benchmark_close = _optional_float(benchmark_row.get("benchmark_close"))
        benchmark_ma = _optional_float(benchmark_row.get("benchmark_ma"))
        benchmark_return = _optional_float(benchmark_row.get("benchmark_return"))
        benchmark_rsrs_beta = _optional_float(benchmark_row.get("benchmark_rsrs_beta"))
        benchmark_rsrs_zscore = _optional_float(benchmark_row.get("benchmark_rsrs_zscore"))
        benchmark_rsrs_risk_on = bool(benchmark_row.get("benchmark_rsrs_risk_on", True))
        benchmark_risk_on = bool(benchmark_row.get("benchmark_risk_on", True))

    threshold_tolerance = 1e-12
    if protected and drawdown <= risk.recovery_drawdown + threshold_tolerance and benchmark_risk_on:
        protected = False
    if drawdown >= risk.protection_drawdown - threshold_tolerance:
        protected = True

    dd_exposure, dd_reason = drawdown_exposure(risk, drawdown)
    exposure = min(benchmark_exposure, dd_exposure)
    reasons = [benchmark_reason, dd_reason]
    if protected:
        exposure = 0.0
        reasons.append("protection")

    return RiskDecision(
        exposure=clamp_exposure(exposure),
        drawdown=drawdown,
        peak_equity=peak,
        protected=protected,
        reason=";".join(reasons),
        benchmark_close=benchmark_close,
        benchmark_ma=benchmark_ma,
        benchmark_return=benchmark_return,
        benchmark_rsrs_beta=benchmark_rsrs_beta,
        benchmark_rsrs_zscore=benchmark_rsrs_zscore,
        benchmark_rsrs_risk_on=benchmark_rsrs_risk_on,
        benchmark_risk_on=benchmark_risk_on,
    )


def _optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
