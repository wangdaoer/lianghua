"""Baseline strategy signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig
from .indicators import moving_average, rsi


def _state_from_entry_exit(data: pd.DataFrame) -> list[int]:
    holding = False
    signals: list[int] = []
    for row in data.itertuples(index=False):
        if holding and bool(row.exit_risk):
            holding = False
        elif not holding and bool(row.entry_candidate):
            holding = True
        signals.append(1 if holding else 0)
    return signals


def _apply_volatility_stop_loss(data: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    if not config.volatility_stop_loss_enabled:
        return data
    frame = data.copy()
    returns = pd.to_numeric(frame["close"], errors="coerce").pct_change()
    stop = returns.rolling(
        config.volatility_stop_window,
        min_periods=config.volatility_stop_window,
    ).std() * config.volatility_stop_multiplier
    lower = max(float(config.volatility_stop_min_pct), 0.0)
    upper = max(float(config.volatility_stop_max_pct), lower)
    frame["volatility_stop_loss_pct"] = stop.clip(lower=lower, upper=upper)
    frame["trade_stop_loss_pct"] = frame["volatility_stop_loss_pct"].shift(1)
    return frame


def trend_signals(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    data = frame.sort_values("date").copy()
    data["ma_fast"] = moving_average(data["close"], config.fast_window)
    data["ma_slow"] = moving_average(data["close"], config.slow_window)
    data["signal"] = ((data["close"] > data["ma_fast"]) & (data["ma_fast"] > data["ma_slow"])).astype(int)
    data["score"] = (data["close"] / data["ma_slow"]) - 1
    data["trade_signal"] = data["signal"].shift(1).fillna(0).astype(int)
    data["trade_score"] = data["score"].shift(1)
    return _apply_volatility_stop_loss(data, config)


def mean_reversion_signals(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    data = frame.sort_values("date").copy()
    data["ma_trend"] = moving_average(data["close"], config.trend_window)
    data["rsi"] = rsi(data["close"], config.rsi_window)

    holding = False
    signals: list[int] = []
    previous_rsi: float | None = None
    for row in data.itertuples(index=False):
        current_rsi = getattr(row, "rsi")
        in_uptrend = pd.notna(row.ma_trend) and row.close > row.ma_trend
        recovered = (
            previous_rsi is not None
            and pd.notna(previous_rsi)
            and pd.notna(current_rsi)
            and previous_rsi < config.rsi_oversold
            and current_rsi >= config.rsi_recover
        )
        if not holding and in_uptrend and recovered:
            holding = True
        elif holding and (not in_uptrend or (pd.notna(current_rsi) and current_rsi >= config.rsi_exit)):
            holding = False
        signals.append(1 if holding else 0)
        previous_rsi = current_rsi

    data["signal"] = signals
    data["score"] = (data["close"] / data["ma_trend"]) - 1
    data["trade_signal"] = data["signal"].shift(1).fillna(0).astype(int)
    data["trade_score"] = data["score"].shift(1)
    return _apply_volatility_stop_loss(data, config)


def thsdk_monitor_signals(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    data = frame.sort_values("date").copy()
    data["signal_age"] = range(1, len(data) + 1)
    recent_high = data["high"].shift(1).rolling(5, min_periods=5).max()
    recent_low = data["low"].shift(1).rolling(5, min_periods=5).min()
    avg_amount = data["amount"].shift(1).rolling(5, min_periods=5).mean()
    price_range = (data["high"] - data["low"]).mask((data["high"] - data["low"]) <= 0)

    data["pressure"] = recent_high
    data["support"] = recent_low
    data["pct"] = data["close"].pct_change() * 100
    data["volume_ratio"] = data["amount"] / avg_amount.mask(avg_amount <= 0)
    data["recovered_from_low"] = (((data["close"] - data["low"]) / price_range) >= 0.45).fillna(False)
    data["breakout"] = (
        (data["close"] > data["pressure"])
        & (data["volume_ratio"] >= 1.2)
    ).fillna(False)
    data["pullback_hold"] = (
        (data["low"] <= data["support"] * 1.025)
        & (data["close"] >= data["support"] * 1.02)
        & data["recovered_from_low"]
    ).fillna(False)
    data["hard_drop"] = (data["pct"] <= -7).fillna(False)
    data["weak_break"] = (
        (data["pct"] <= -5)
        & (data["close"] < data["support"])
    ).fillna(False)

    score = pd.Series(50.0, index=data.index)
    score += data["breakout"].astype(float) * 18
    score += data["pullback_hold"].astype(float) * 16
    risk_signal = data["hard_drop"] | data["weak_break"]
    score -= risk_signal.astype(float) * 24
    score += (data["volume_ratio"] >= 1.5).fillna(False).astype(float) * 10
    score += ((data["volume_ratio"] >= 1.2) & (data["volume_ratio"] < 1.5)).fillna(False).astype(float) * 6
    score -= (data["volume_ratio"] < 0.8).fillna(False).astype(float) * 6
    score -= (data["pct"] <= -7).fillna(False).astype(float) * 12
    score -= ((data["pct"] <= -5) & (data["pct"] > -7)).fillna(False).astype(float) * 8
    score -= (data["pct"] >= 9).fillna(False).astype(float) * 8
    score += ((data["pct"] >= 1) & (data["pct"] <= 5)).fillna(False).astype(float) * 4
    score += data["recovered_from_low"].astype(float) * 8

    stop_loss = data["low"].where(data["pullback_hold"], data["pressure"])
    stop_risk = (data["close"] / stop_loss.mask(stop_loss <= 0) - 1) * 100
    score += (stop_risk <= 3).fillna(False).astype(float) * 8
    score -= (stop_risk >= 8).fillna(False).astype(float) * 8
    upside = (data["pressure"] / data["close"].mask(data["close"] <= 0) - 1) * 100
    downside = (data["close"] / data["support"].mask(data["support"] <= 0) - 1) * 100
    score += (upside > downside * 1.5).fillna(False).astype(float) * 6
    score -= (upside < downside).fillna(False).astype(float) * 6

    data["score"] = score.clip(lower=0, upper=100)
    data["entry_candidate"] = (
        (data["breakout"] | data["pullback_hold"])
        & ~risk_signal
        & (data["score"] >= config.min_score)
        & (data["signal_age"] >= config.signal_min_history)
    )
    data["exit_risk"] = (risk_signal | (data["close"] < data["support"])).fillna(False)

    data["signal"] = _state_from_entry_exit(data)
    data["trade_signal"] = data["signal"].shift(1).fillna(0).astype(int)
    data["trade_exit_risk"] = data["exit_risk"].shift(1).fillna(False).astype(int)
    data["trade_score"] = data["score"].shift(1)
    data["market_breadth_ratio"] = 1.0
    data["market_breadth_count"] = 1
    data["market_breadth_ok"] = True
    data["trade_entry_allowed"] = 1
    return _apply_volatility_stop_loss(data, config)


def build_signals(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    name = config.name.lower().replace("-", "_")
    if name == "trend":
        return trend_signals(frame, config)
    if name in {"mean_reversion", "rsi_mean_reversion"}:
        return mean_reversion_signals(frame, config)
    if name in {"thsdk_monitor", "ths_monitor", "tonghuashun"}:
        return thsdk_monitor_signals(frame, config)
    if name == "multi_factor":
        raise ValueError("Use build_signal_frames for multi_factor because it needs cross-sectional ranking.")
    raise ValueError(f"Unknown strategy: {config.name}")


def _market_breadth_frame(histories: dict[str, pd.DataFrame], config: StrategyConfig) -> pd.DataFrame:
    frames = []
    for code, frame in histories.items():
        data = frame.sort_values("date").copy()
        data["market_breadth_ma"] = moving_average(data["close"], config.market_breadth_window)
        data["available"] = data["close"].notna() & data["market_breadth_ma"].notna()
        data["above_ma"] = (data["close"] > data["market_breadth_ma"]) & data["available"]
        data["code"] = code
        frames.append(data[["date", "code", "available", "above_ma"]])
    if not frames:
        return pd.DataFrame(columns=["date", "market_breadth_count", "market_breadth_ratio", "market_breadth_ok"])

    panel = pd.concat(frames, ignore_index=True)
    grouped = panel.groupby("date", sort=True).agg(
        market_breadth_count=("available", "sum"),
        market_breadth_above=("above_ma", "sum"),
    )
    grouped["market_breadth_ratio"] = (
        grouped["market_breadth_above"] / grouped["market_breadth_count"].replace(0, pd.NA)
    ).fillna(0.0)
    grouped["market_breadth_ok"] = (
        (grouped["market_breadth_count"] >= config.market_breadth_min_count)
        & (grouped["market_breadth_ratio"] >= config.market_breadth_min_ratio)
    )
    return grouped.reset_index()[["date", "market_breadth_count", "market_breadth_ratio", "market_breadth_ok"]]


def _apply_market_breadth_filter(
    signal_frames: dict[str, pd.DataFrame],
    histories: dict[str, pd.DataFrame],
    config: StrategyConfig,
) -> dict[str, pd.DataFrame]:
    if not (config.market_breadth_enabled or config.market_breadth_exposure_enabled):
        return signal_frames
    breadth = _market_breadth_frame(histories, config)
    result: dict[str, pd.DataFrame] = {}
    for code, frame in signal_frames.items():
        data = frame.drop(
            columns=["market_breadth_count", "market_breadth_ratio", "market_breadth_ok"],
            errors="ignore",
        ).merge(breadth, on="date", how="left")
        data["market_breadth_count"] = data["market_breadth_count"].fillna(0).astype(int)
        data["market_breadth_ratio"] = data["market_breadth_ratio"].fillna(0.0)
        data["market_breadth_ok"] = data["market_breadth_ok"].fillna(False).astype(bool)
        base_entry_allowed = data["trade_entry_allowed"].astype(bool) if "trade_entry_allowed" in data else True
        if config.market_breadth_enabled:
            if "entry_candidate" in data and "exit_risk" in data:
                data["entry_candidate"] = data["entry_candidate"] & data["market_breadth_ok"]
                data["signal"] = _state_from_entry_exit(data)
                data["trade_signal"] = data["signal"].shift(1).fillna(0).astype(int)
            else:
                entry_candidate = data["entry_candidate"] if "entry_candidate" in data else data["signal"].astype(bool)
                data["entry_candidate"] = entry_candidate & data["market_breadth_ok"]
            data["trade_entry_allowed"] = (
                pd.Series(base_entry_allowed, index=data.index).astype(bool)
                & data["market_breadth_ok"].shift(1).fillna(False).astype(bool)
            ).astype(int)
        else:
            data["trade_entry_allowed"] = pd.Series(base_entry_allowed, index=data.index).astype(bool).astype(int)
        result[code] = data
    return result


def _factor_frame(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    data = frame.sort_values("date").copy()
    returns = data["close"].pct_change()
    data["factor_momentum"] = data["close"].pct_change(config.factor_momentum_window)
    data["factor_reversal"] = -data["close"].pct_change(config.factor_reversal_window)
    data["factor_volatility"] = -returns.rolling(config.factor_volatility_window, min_periods=config.factor_volatility_window).std()
    liquidity = data["amount"].where(data["amount"] > 0, data["volume"] * data["close"])
    data["factor_liquidity"] = liquidity.rolling(
        config.factor_liquidity_window,
        min_periods=config.factor_liquidity_window,
    ).mean()
    data["factor_trend"] = (data["close"] / moving_average(data["close"], config.factor_trend_window)) - 1
    close_base = data["close"].where(data["close"] > 0)
    convergence_components = [pd.Series(1.0, index=data.index, dtype="float64")]
    convergence_components.extend(
        moving_average(data["close"], window) / close_base
        for window in config.factor_convergence_windows
    )
    convergence_stack = pd.concat(convergence_components, axis=1)
    data["factor_convergence"] = -np.log1p(convergence_stack.std(axis=1, ddof=1))
    volume = pd.to_numeric(data["volume"], errors="coerce").where(data["volume"] > 0)
    data["factor_volume_price_corr"] = -data["close"].rolling(
        config.factor_volume_price_window,
        min_periods=config.factor_volume_price_window,
    ).corr(volume)
    volume_change = volume.pct_change().replace([np.inf, -np.inf], pd.NA)
    intraday_return = (data["close"] / data["open"].where(data["open"] > 0) - 1).replace([np.inf, -np.inf], pd.NA)
    data["factor_volume_price_divergence"] = -volume_change.rolling(
        config.factor_volume_price_window,
        min_periods=config.factor_volume_price_window,
    ).corr(intraday_return)
    price_range = (data["high"] - data["low"]).where((data["high"] - data["low"]) > 0)
    upper_shadow = (data["high"] - data[["open", "close"]].max(axis=1)) / price_range
    lower_shadow = (data[["open", "close"]].min(axis=1) - data["low"]) / price_range
    data["factor_shadow_support"] = (lower_shadow - upper_shadow).rolling(
        config.factor_shadow_window,
        min_periods=config.factor_shadow_window,
    ).mean()
    chip_prior_high = data["high"].rolling(
        config.factor_chip_reversal_drawdown_window,
        min_periods=2,
    ).max().shift(1)
    chip_drawdown = data["close"] / chip_prior_high - 1.0
    chip_cost_ma = data["close"].rolling(
        config.factor_chip_reversal_cost_window,
        min_periods=config.factor_chip_reversal_cost_window,
    ).mean()
    chip_cost_gap = data["close"] / chip_cost_ma - 1.0
    chip_score = (-chip_drawdown).clip(lower=0.0) * 0.6 + (-chip_cost_gap).clip(lower=0.0) * 0.4
    chip_drawdown_threshold = -float(config.factor_chip_reversal_min_drawdown_pct) / 100.0
    chip_event = (
        (chip_drawdown <= chip_drawdown_threshold)
        & (chip_cost_gap < 0.0)
        & (chip_score >= float(config.factor_chip_reversal_min_score))
    )
    data["factor_chip_reversal"] = chip_score.where(chip_event)
    data["factor_age"] = range(1, len(data) + 1)
    data["factor_entry_trend"] = (
        data["close"] / moving_average(data["close"], config.factor_entry_trend_window)
    ) - 1
    data["factor_entry_momentum"] = data["close"].pct_change(config.factor_entry_momentum_window)
    if config.factor_entry_filter_enabled:
        data["factor_entry_ok"] = (
            data["factor_entry_trend"].notna()
            & (data["factor_entry_trend"] >= config.factor_entry_min_trend)
            & data["factor_entry_momentum"].notna()
            & (data["factor_entry_momentum"] >= config.factor_entry_min_momentum)
        )
    else:
        data["factor_entry_ok"] = True
    return data


def _apply_cross_sectional_score_overlay(
    signal_frames: dict[str, pd.DataFrame],
    histories: dict[str, pd.DataFrame],
    config: StrategyConfig,
) -> dict[str, pd.DataFrame]:
    if not config.cross_sectional_score_enabled:
        return signal_frames

    factor_columns = {
        "momentum": "factor_momentum",
        "trend": "factor_trend",
        "reversal": "factor_reversal",
        "volatility": "factor_volatility",
        "liquidity": "factor_liquidity",
        "convergence": "factor_convergence",
        "volume_price_corr": "factor_volume_price_corr",
        "volume_price_divergence": "factor_volume_price_divergence",
        "shadow_support": "factor_shadow_support",
        "chip_reversal": "factor_chip_reversal",
    }
    frames = []
    for code, signal_frame in signal_frames.items():
        factors = _factor_frame(histories[code], config)[
            [
                "date",
                "factor_momentum",
                "factor_reversal",
                "factor_volatility",
                "factor_liquidity",
                "factor_trend",
                "factor_convergence",
                "factor_volume_price_corr",
                "factor_volume_price_divergence",
                "factor_shadow_support",
                "factor_chip_reversal",
                "factor_age",
            ]
        ]
        data = signal_frame.copy()
        if "code" not in data:
            data["code"] = code
        data["base_score"] = pd.to_numeric(data.get("score", 0.0), errors="coerce")
        data = data.merge(factors, on="date", how="left")
        frames.append(data)

    if not frames:
        return signal_frames

    panel = pd.concat(frames, ignore_index=True)
    panel["cross_sectional_score_raw"] = 0.0
    used_weight = 0.0
    for score_name, weight in config.cross_sectional_score_weights.items():
        if weight == 0:
            continue
        column = "base_score" if score_name == "signal" else factor_columns.get(score_name)
        if column is None or column not in panel:
            continue
        rank_column = f"cross_sectional_{score_name}_rank"
        panel[rank_column] = panel.groupby("date")[column].rank(pct=True, method="average")
        panel["cross_sectional_score_raw"] += panel[rank_column].fillna(0.0) * float(weight)
        used_weight += abs(float(weight))

    if used_weight == 0:
        raise ValueError("cross_sectional_score_enabled requires at least one valid non-zero score weight.")

    available = pd.to_numeric(panel["factor_age"], errors="coerce").fillna(0) >= config.factor_min_history
    panel["cross_sectional_score"] = (panel["cross_sectional_score_raw"] / used_weight * 100).clip(lower=0, upper=100)
    panel.loc[~available, "cross_sectional_score"] = panel.loc[~available, "base_score"]

    score_panel = panel[["date", "code", "cross_sectional_score"]]
    result: dict[str, pd.DataFrame] = {}
    for code, frame in signal_frames.items():
        data = frame.copy()
        if "code" not in data:
            data["code"] = code
        data["base_score"] = pd.to_numeric(data.get("score", 0.0), errors="coerce")
        data = data.drop(columns=["cross_sectional_score"], errors="ignore").merge(
            score_panel[score_panel["code"].astype(str) == str(code)],
            on=["date", "code"],
            how="left",
        )
        data["cross_sectional_score"] = data["cross_sectional_score"].fillna(data["base_score"])
        data["score"] = data["cross_sectional_score"]
        data["trade_score"] = data["cross_sectional_score"].shift(1)
        result[str(code)] = data
    return result


def _momentum_focus_board(raw_code: str) -> str:
    code = "".join(ch for ch in str(raw_code) if ch.isdigit())
    if len(code) >= 6:
        code = code[-6:]
    code = code.zfill(6)
    if code.startswith(("4", "8", "920")):
        return "bse"
    if code.startswith(("300", "301")):
        return "chinext"
    if code.startswith(("688", "689")):
        return "star"
    if code.startswith(("000", "001", "002", "003", "600", "601", "603", "605")):
        return "main"
    return "unknown"


def _momentum_focus_in_scope(board: str, scope: str) -> bool:
    if scope == "all":
        return board != "unknown"
    if scope == "main_chinext":
        return board in {"main", "chinext"}
    raise ValueError("momentum_focus_board_scope must be one of: main_chinext, all")


def _momentum_focus_limit_up_threshold(board: str) -> float:
    if board in {"chinext", "star"}:
        return 19.5
    if board == "bse":
        return 29.0
    return 9.8


def _annotate_momentum_focus(
    frame: pd.DataFrame,
    code: str,
    config: StrategyConfig,
) -> pd.DataFrame:
    data = frame.copy()
    board = _momentum_focus_board(code)
    data["momentum_change_pct"] = data["close"].pct_change() * 100
    data["momentum_focus_board"] = board
    if not _momentum_focus_in_scope(board, config.momentum_focus_board_scope):
        data["momentum_focus_flag"] = False
        data["momentum_focus_signal"] = "normal"
        data["momentum_focus_boost"] = 0.0
        return data

    threshold = float(config.momentum_focus_threshold_pct)
    limit_up_threshold = _momentum_focus_limit_up_threshold(board)
    momentum_change_pct = pd.to_numeric(data["momentum_change_pct"], errors="coerce")
    momentum_focus_flag = momentum_change_pct >= threshold
    data["momentum_focus_flag"] = momentum_focus_flag
    data["momentum_focus_signal"] = np.where(
        momentum_focus_flag,
        np.where(
            momentum_change_pct >= limit_up_threshold,
            "limit_up",
            "strong_gain",
        ),
        "normal",
    )
    data["momentum_focus_boost"] = np.where(
        momentum_focus_flag & (momentum_change_pct >= limit_up_threshold),
        float(config.momentum_focus_limit_up_boost),
        np.where(
            momentum_focus_flag,
            float(config.momentum_focus_strong_gain_boost),
            0.0,
        ),
    )
    return data


def multi_factor_signal_frames(histories: dict[str, pd.DataFrame], config: StrategyConfig) -> dict[str, pd.DataFrame]:
    factor_columns = {
        "momentum": "factor_momentum",
        "trend": "factor_trend",
        "reversal": "factor_reversal",
        "volatility": "factor_volatility",
        "liquidity": "factor_liquidity",
        "convergence": "factor_convergence",
        "volume_price_corr": "factor_volume_price_corr",
        "volume_price_divergence": "factor_volume_price_divergence",
        "shadow_support": "factor_shadow_support",
        "chip_reversal": "factor_chip_reversal",
    }
    frames = []
    for code, frame in histories.items():
        data = _factor_frame(frame, config)
        if config.momentum_focus_enabled:
            data = _annotate_momentum_focus(data, code, config)
        data["code"] = code
        frames.append(data)
    if not frames:
        return {}

    panel = pd.concat(frames, ignore_index=True)
    panel["factor_score"] = 0.0
    panel["factor_available"] = panel["factor_age"] >= config.factor_min_history
    used_weight = 0.0
    for factor_name, weight in config.factor_weights.items():
        column = factor_columns.get(factor_name)
        if column is None or column not in panel.columns or weight == 0:
            continue
        rank_column = f"{column}_rank"
        panel[rank_column] = panel.groupby("date")[column].rank(pct=True, method="average")
        panel["factor_score"] += panel[rank_column].fillna(0.0) * weight
        panel["factor_available"] &= panel[column].notna()
        used_weight += abs(weight)

    if used_weight == 0:
        raise ValueError("multi_factor strategy requires at least one valid factor weight.")

    panel.loc[~panel["factor_available"], "factor_score"] = pd.NA
    if config.momentum_focus_enabled:
        if "momentum_focus_boost" not in panel:
            panel["momentum_focus_boost"] = 0.0
        if not _momentum_focus_in_scope("main", config.momentum_focus_board_scope):
            raise ValueError("Unsupported momentum_focus_board_scope configuration value.")
        if config.momentum_focus_only:
            panel["factor_score"] = panel["factor_score"].where(
                pd.Series(panel.get("momentum_focus_flag", False), index=panel.index).astype(bool),
                other=pd.NA,
            )
            panel["factor_available"] = panel["factor_score"].notna()
        panel["factor_score"] = panel["factor_score"] + panel["momentum_focus_boost"].fillna(0.0)

    dates = sorted(panel["date"].dropna().unique())
    rebalance_interval = max(config.factor_rebalance_interval, 1)
    selected_by_date: dict[pd.Timestamp, set[str]] = {}
    current_selection: set[str] = set()
    for idx, date in enumerate(dates):
        daily = panel[(panel["date"] == date) & panel["factor_score"].notna()].copy()
        if config.factor_entry_filter_enabled:
            daily = daily[daily["factor_entry_ok"].fillna(False)]
        if idx % rebalance_interval == 0 and not daily.empty:
            daily = daily.sort_values("factor_score", ascending=False)
            current_selection = set(daily.head(max(config.max_positions, 1))["code"].astype(str))
        selected_by_date[pd.Timestamp(date)] = set(current_selection)

    panel["signal"] = [
        1
        if str(row.code) in selected_by_date.get(pd.Timestamp(row.date), set())
        and (not config.factor_entry_filter_enabled or bool(row.factor_entry_ok))
        else 0
        for row in panel.itertuples(index=False)
    ]
    panel["score"] = panel["factor_score"]
    result: dict[str, pd.DataFrame] = {}
    for code, data in panel.groupby("code", sort=False):
        out = data.sort_values("date").copy()
        out["entry_candidate"] = out["signal"].astype(bool)
        out["trade_signal"] = out["signal"].shift(1).fillna(0).astype(int)
        out["trade_score"] = out["score"].shift(1)
        if config.factor_entry_filter_enabled:
            out["trade_entry_allowed"] = out["factor_entry_ok"].shift(1).fillna(False).astype(int)
        else:
            out["trade_entry_allowed"] = 1
        result[str(code)] = _apply_volatility_stop_loss(out.reset_index(drop=True), config)
    return result


def build_signal_frames(histories: dict[str, pd.DataFrame], config: StrategyConfig) -> dict[str, pd.DataFrame]:
    name = config.name.lower().replace("-", "_")
    if name == "multi_factor":
        return _apply_market_breadth_filter(multi_factor_signal_frames(histories, config), histories, config)
    signal_frames = {code: build_signals(frame, config) for code, frame in histories.items()}
    if name in {"thsdk_monitor", "ths_monitor", "tonghuashun"}:
        signal_frames = _apply_market_breadth_filter(signal_frames, histories, config)
        signal_frames = _apply_cross_sectional_score_overlay(signal_frames, histories, config)
    return signal_frames
