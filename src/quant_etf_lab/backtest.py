"""Event-style portfolio backtester."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml

from .config import ETFSpec, LabConfig
from .data import load_universe_history
from .risk import build_benchmark_risk_frame, decide_next_risk
from .market_cap import load_stock_market_cap_cache
from .strategies import build_signal_frames


@dataclass(frozen=True)
class BacktestResult:
    run_id: str
    run_dir: Path
    equity: pd.DataFrame
    trades: pd.DataFrame
    benchmark: pd.DataFrame
    monthly_returns: pd.DataFrame
    metrics: dict[str, Any]
    risk_curve: pd.DataFrame
    risk_events: pd.DataFrame
    cooldown_events: pd.DataFrame


@dataclass(frozen=True)
class BuyOrder:
    idx: int
    exec_price: float
    quantity: float
    required_cash: float


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _max_drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return (equity / running_max) - 1


def _monthly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    series = equity.set_index("date")["equity"]
    try:
        monthly_equity = series.resample("ME").last()
    except ValueError:
        monthly_equity = series.resample("M").last()
    monthly = monthly_equity.pct_change().dropna().reset_index()
    monthly.columns = ["date", "monthly_return"]
    return monthly


def _benchmark_curve(histories: dict[str, pd.DataFrame], dates: pd.Series, initial_cash: float) -> pd.DataFrame:
    date_index = pd.DatetimeIndex(dates)
    if len(date_index) == 0:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "benchmark_equity": pd.Series(dtype="float64"),
            }
        )
    if not histories:
        return pd.DataFrame(
            {
                "date": date_index,
                "benchmark_equity": np.full(len(date_index), float(initial_cash), dtype="float64"),
            }
        )
    close_panel = pd.DataFrame(
        {code: frame.set_index("date")["close"].sort_index() for code, frame in histories.items()}
    )
    close_panel = close_panel.sort_index().reindex(date_index).ffill()
    returns = close_panel.pct_change().mean(axis=1, skipna=True).fillna(0)
    benchmark = (1 + returns).cumprod() * initial_cash
    if not benchmark.empty:
        benchmark.iloc[0] = initial_cash
    return pd.DataFrame({"date": benchmark.index, "benchmark_equity": benchmark.to_numpy()})


def _compute_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    benchmark: pd.DataFrame,
    initial_cash: float,
    risk_curve: pd.DataFrame | None = None,
    cooldown_events: pd.DataFrame | None = None,
) -> dict[str, Any]:
    equity_series = equity.set_index("date")["equity"]
    returns = equity_series.pct_change().dropna()
    final_equity = float(equity_series.iloc[-1]) if not equity_series.empty else initial_cash
    total_return = (final_equity / initial_cash) - 1
    days = max((equity_series.index[-1] - equity_series.index[0]).days, 1) if len(equity_series) > 1 else 1
    years = days / 365.25
    cagr = (final_equity / initial_cash) ** (1 / years) - 1 if years > 0 else 0.0
    drawdown = _max_drawdown(equity_series)
    sharpe = 0.0
    if len(returns) > 1 and float(returns.std()) != 0:
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std())

    sell_trades = trades[trades["side"] == "SELL"] if not trades.empty else pd.DataFrame()
    realized = pd.to_numeric(sell_trades.get("realized_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    wins = realized[realized > 0]
    losses = realized[realized < 0]
    win_rate = float(len(wins) / len(realized)) if len(realized) else 0.0
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    payoff_ratio = float(avg_win / abs(avg_loss)) if avg_loss else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else 0.0

    benchmark_final = (
        float(benchmark["benchmark_equity"].iloc[-1])
        if not benchmark.empty and "benchmark_equity" in benchmark
        else initial_cash
    )
    benchmark_return = (benchmark_final / initial_cash) - 1

    metrics = {
        "initial_cash": float(initial_cash),
        "final_equity": final_equity,
        "total_return": float(total_return),
        "benchmark_return": float(benchmark_return),
        "excess_return": float(total_return - benchmark_return),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "sharpe": float(sharpe),
        "trade_count": int(len(trades)),
        "sell_trade_count": int(len(realized)),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
        "profit_factor": profit_factor,
    }
    if not equity.empty and "positions_value" in equity:
        exposure = equity["positions_value"] / equity["equity"].replace(0, np.nan)
        metrics["average_position_exposure"] = float(exposure.replace([np.inf, -np.inf], np.nan).fillna(0).mean())
    if risk_curve is not None and not risk_curve.empty:
        risk_exposure = pd.to_numeric(risk_curve["risk_exposure"], errors="coerce").fillna(1.0)
        metrics["average_risk_exposure"] = float(risk_exposure.mean())
        metrics["risk_off_day_ratio"] = float((risk_exposure <= 0.01).mean())
        metrics["risk_reduced_day_ratio"] = float((risk_exposure < 0.999).mean())
    if not equity.empty and "cooldown_active_count" in equity:
        active_counts = pd.to_numeric(equity["cooldown_active_count"], errors="coerce").fillna(0)
        metrics["average_cooldown_count"] = float(active_counts.mean())
        metrics["cooldown_active_day_ratio"] = float((active_counts > 0).mean())
    if not equity.empty and "market_breadth_exposure" in equity:
        breadth_exposure = pd.to_numeric(equity["market_breadth_exposure"], errors="coerce").fillna(1.0)
        metrics["average_market_breadth_exposure"] = float(breadth_exposure.mean())
        metrics["market_breadth_reduced_day_ratio"] = float((breadth_exposure < 0.999).mean())
    if not equity.empty and "rebalance_loss_guarded_count" in equity:
        guarded_counts = pd.to_numeric(equity["rebalance_loss_guarded_count"], errors="coerce").fillna(0)
        metrics["average_rebalance_loss_guarded_count"] = float(guarded_counts.mean())
        metrics["rebalance_loss_guarded_day_ratio"] = float((guarded_counts > 0).mean())
    if not equity.empty and "stock_cap_blocked_count" in equity:
        blocked_counts = pd.to_numeric(equity["stock_cap_blocked_count"], errors="coerce").fillna(0)
        metrics["average_stock_cap_blocked_count"] = float(blocked_counts.mean())
        metrics["stock_cap_blocked_days"] = int((blocked_counts > 0).sum())
    if cooldown_events is not None:
        metrics["cooldown_event_count"] = int(len(cooldown_events))
    return metrics


def _plot_charts(equity: pd.DataFrame, benchmark: pd.DataFrame, run_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return

    merged = equity.merge(benchmark, on="date", how="left")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(merged["date"], merged["equity"], label="strategy")
    if "benchmark_equity" in merged:
        ax.plot(merged["date"], merged["benchmark_equity"], label="buy_hold")
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(run_dir / "equity_curve.png", dpi=150)
    plt.close(fig)

    drawdown = _max_drawdown(equity.set_index("date")["equity"])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(drawdown.index, drawdown.to_numpy(), 0, color="#d94841", alpha=0.35)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(run_dir / "drawdown.png", dpi=150)
    plt.close(fig)


def _stock_cap_exclusion_matrix(
    config: LabConfig,
    codes: list[str],
    date_index: pd.DatetimeIndex,
) -> np.ndarray:
    date_count = len(date_index)
    if date_count == 0:
        return np.zeros((0, 0), dtype=bool)
    exclusions = np.zeros((date_count, len(codes)), dtype=bool)
    if config.stock_market_cap_path is None:
        return exclusions
    if config.stock_tracking_max_market_cap_yi is None or config.stock_tracking_max_market_cap_yi <= 0:
        return exclusions
    try:
        cache = load_stock_market_cap_cache(config.stock_market_cap_path)
    except Exception:
        return exclusions
    if cache.empty or "code" not in cache.columns:
        return exclusions

    normalized = cache.copy()
    normalized["code"] = (
        normalized["code"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .str[-6:]
        .str.zfill(6)
    )
    normalized["snapshot_date"] = pd.to_datetime(normalized.get("snapshot_date"), errors="coerce")
    normalized["market_cap_yi"] = pd.to_numeric(normalized.get("market_cap_yi"), errors="coerce")
    normalized = normalized.dropna(subset=["code", "snapshot_date", "market_cap_yi"])
    if normalized.empty:
        return exclusions
    normalized = normalized.sort_values(["code", "snapshot_date"]).drop_duplicates(
        subset=["code", "snapshot_date"],
        keep="last",
    )
    dates = pd.DataFrame({"date": date_index}).sort_values("date")

    threshold = float(config.stock_tracking_max_market_cap_yi)
    for col_index, code in enumerate(codes):
        frame = normalized[normalized["code"] == str(code).zfill(6)][["snapshot_date", "market_cap_yi"]]
        if frame.empty:
            continue
        frame = frame.sort_values("snapshot_date")
        merged = pd.merge_asof(
            dates,
            frame,
            left_on="date",
            right_on="snapshot_date",
            direction="backward",
            allow_exact_matches=True,
        )
        market_caps = pd.to_numeric(merged["market_cap_yi"], errors="coerce").to_numpy()
        present = np.isfinite(market_caps)
        exclusions[present, col_index] = market_caps[present] > threshold
    return exclusions


def _write_outputs(config: LabConfig, result: BacktestResult) -> None:
    result.run_dir.mkdir(parents=True, exist_ok=True)
    result.equity.to_csv(result.run_dir / "equity_curve.csv", index=False, encoding="utf-8")
    result.trades.to_csv(result.run_dir / "trades.csv", index=False, encoding="utf-8")
    result.benchmark.to_csv(result.run_dir / "benchmark.csv", index=False, encoding="utf-8")
    result.monthly_returns.to_csv(result.run_dir / "monthly_returns.csv", index=False, encoding="utf-8")
    if not result.risk_curve.empty:
        result.risk_curve.to_csv(result.run_dir / "risk_curve.csv", index=False, encoding="utf-8")
    if not result.risk_events.empty:
        result.risk_events.to_csv(result.run_dir / "risk_events.csv", index=False, encoding="utf-8")
    if not result.cooldown_events.empty:
        result.cooldown_events.to_csv(result.run_dir / "cooldown_events.csv", index=False, encoding="utf-8")
    (result.run_dir / "metrics.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config_dump = {
        "project": {"name": config.project.name, "initial_cash": config.project.initial_cash},
        "data": {
            "start_date": config.data.start_date,
            "end_date": config.data.end_date,
            "period": config.data.period,
            "adjust": config.data.adjust,
        },
        "strategy": config.strategy.__dict__,
        "costs": config.costs.__dict__,
        "risk": config.risk.__dict__,
        "stock_market_cap_path": str(config.stock_market_cap_path) if config.stock_market_cap_path is not None else None,
        "stock_tracking_max_market_cap_yi": config.stock_tracking_max_market_cap_yi,
        "universe": [item.__dict__ for item in config.universe],
    }
    if config.universe_file is not None:
        config_dump["universe_file"] = str(config.universe_file)
    if config.universe_source is not None:
        config_dump["universe_source"] = config.universe_source.__dict__
    (result.run_dir / "config_used.yaml").write_text(yaml.safe_dump(config_dump, sort_keys=False), encoding="utf-8")
    _plot_charts(result.equity, result.benchmark, result.run_dir)


def _round_buy_quantity(quantity: float, lot_size: int) -> float:
    if lot_size <= 1:
        return max(quantity, 0.0)
    return float(int(quantity // lot_size) * lot_size)


def _score_weighted_position_weights(
    selected: list[int],
    score_column: np.ndarray,
    target_exposure: float,
    method: str = "proportional",
    tilt_strength: float = 0.5,
    min_multiplier: float = 0.5,
    max_multiplier: float = 1.5,
) -> np.ndarray:
    if not selected:
        return np.zeros(0, dtype="float64")

    if target_exposure <= 0:
        return np.zeros(len(selected), dtype="float64")

    raw_scores = np.asarray([score_column[idx] for idx in selected], dtype="float64")
    raw_scores = np.where(~np.isfinite(raw_scores), 0.0, raw_scores)
    method = str(method).lower().replace("-", "_")
    if method in {"tilt", "bounded_tilt", "rank_tilt"}:
        base_weight = target_exposure / len(selected)
        if len(selected) == 1 or float(np.nanmax(raw_scores) - np.nanmin(raw_scores)) <= 1e-12:
            return np.full(len(selected), base_weight, dtype="float64")
        ranks = pd.Series(raw_scores).rank(method="average").to_numpy(dtype="float64")
        percentile = (ranks - 1.0) / max(len(selected) - 1, 1)
        centered = (percentile - 0.5) * 2.0
        multipliers = 1.0 + centered * max(float(tilt_strength), 0.0)
        min_multiplier = max(float(min_multiplier), 0.0)
        max_multiplier = max(float(max_multiplier), min_multiplier)
        multipliers = np.clip(multipliers, min_multiplier, max_multiplier)
        total_multiplier = float(multipliers.sum())
        if total_multiplier <= 0:
            return np.full(len(selected), base_weight, dtype="float64")
        return multipliers / total_multiplier * target_exposure
    if method not in {"proportional", "raw", "score"}:
        raise ValueError(f"Unknown score allocation method: {method}")

    raw_scores = np.maximum(raw_scores, 0.0)
    score_sum = float(raw_scores.sum())
    if score_sum <= 0:
        return np.full(len(selected), target_exposure / len(selected), dtype="float64")
    return raw_scores / score_sum * target_exposure


def _scaled_buy_orders(
    deltas: dict[int, float],
    open_row: np.ndarray,
    open_valid: np.ndarray,
    cash: float,
    commission_rate: float,
    slippage_rate: float,
    lot_size: int,
    order_scores: dict[int, float] | None = None,
) -> list[BuyOrder]:
    if cash <= 0:
        return []

    order_items = list(deltas.items())
    if order_scores is not None:
        order_items.sort(
            key=lambda item: (float(order_scores.get(item[0], 0.0)), item[0]),
            reverse=True,
        )

    desired: list[BuyOrder] = []
    for idx, delta_value in order_items:
        if delta_value <= 1e-8 or not bool(open_valid[idx]):
            continue
        open_price = float(open_row[idx])
        if not np.isfinite(open_price) or open_price <= 0:
            continue
        exec_price = open_price * (1 + slippage_rate)
        if not np.isfinite(exec_price) or exec_price <= 0:
            continue
        quantity = _round_buy_quantity(delta_value / exec_price, lot_size)
        required_cash = quantity * exec_price * (1 + commission_rate)
        if quantity > 0 and required_cash > 0:
            desired.append(BuyOrder(int(idx), float(exec_price), float(quantity), float(required_cash)))

    total_required = sum(order.required_cash for order in desired)
    if total_required <= cash + 1e-6:
        return desired
    if total_required <= 0:
        return []

    scale = max(0.0, float(cash) / total_required)
    scaled: list[BuyOrder] = []
    for order in desired:
        target_cash = order.required_cash * scale
        quantity = _round_buy_quantity(target_cash / (order.exec_price * (1 + commission_rate)), lot_size)
        required_cash = quantity * order.exec_price * (1 + commission_rate)
        if quantity > 0 and required_cash > 0:
            scaled.append(
                BuyOrder(order.idx, order.exec_price, float(quantity), float(required_cash))
            )

    total_scaled = sum(order.required_cash for order in scaled)
    if total_scaled <= cash + 1e-6:
        return scaled

    remaining_cash = float(cash)
    bounded: list[BuyOrder] = []
    for order in scaled:
        if remaining_cash <= 0:
            break
        quantity = order.quantity
        required_cash = order.required_cash
        if required_cash > remaining_cash + 1e-6:
            quantity = _round_buy_quantity(
                remaining_cash / (order.exec_price * (1 + commission_rate)),
                lot_size,
            )
            required_cash = quantity * order.exec_price * (1 + commission_rate)
        if quantity <= 0 or required_cash > remaining_cash + 1e-6:
            continue
        bounded.append(BuyOrder(order.idx, order.exec_price, float(quantity), float(required_cash)))
        remaining_cash -= required_cash
    return bounded


def run_backtest(
    config: LabConfig,
    histories: dict[str, pd.DataFrame] | None = None,
    signal_frames: dict[str, pd.DataFrame] | None = None,
    run_id: str | None = None,
    write_outputs: bool = True,
    skip_missing: bool = False,
) -> BacktestResult:
    if histories is None:
        histories = load_universe_history(config, allow_fetch=not skip_missing, skip_missing=skip_missing)
    if signal_frames is None:
        signal_frames = build_signal_frames(histories, config.strategy)
    signal_frames = {code: frame.set_index("date") for code, frame in signal_frames.items()}
    date_index = pd.DatetimeIndex(sorted({date for frame in signal_frames.values() for date in frame.index}))
    if date_index.empty:
        raise ValueError("No dates available for backtest.")

    spec_by_code = {item.code: item for item in config.universe}
    for code, frame in histories.items():
        if code not in spec_by_code:
            name = str(frame["name"].iloc[0]) if "name" in frame and not frame.empty else code
            spec_by_code[code] = ETFSpec(code=code, name=name, asset_type="stock")
    codes = list(signal_frames)
    names = [spec_by_code.get(code).name if code in spec_by_code else code for code in codes]

    def aligned_array(column: str, fill_value: float | None = None) -> np.ndarray:
        columns: dict[str, pd.Series] = {}
        for code in codes:
            frame = signal_frames[code]
            if column in frame:
                series = pd.to_numeric(frame[column], errors="coerce")
            else:
                series = pd.Series(fill_value, index=frame.index, dtype="float64")
            columns[code] = series
        panel = pd.DataFrame(columns, index=date_index)
        if fill_value is not None:
            panel = panel.fillna(fill_value)
        return panel.to_numpy(dtype="float64", copy=True)

    open_values = aligned_array("open")
    close_values = aligned_array("close")
    trade_signal_values = aligned_array("trade_signal", 0.0)
    trade_score_values = aligned_array("trade_score", 0.0)
    trade_exit_risk_values = aligned_array("trade_exit_risk", 0.0)
    trade_entry_allowed_values = aligned_array("trade_entry_allowed", 1.0)
    trade_stop_loss_pct_values = aligned_array("trade_stop_loss_pct")
    market_breadth_ratio_values = aligned_array("market_breadth_ratio", 1.0)
    market_breadth_count_values = aligned_array("market_breadth_count", 0.0)
    stock_cap_blocked_values = _stock_cap_exclusion_matrix(config, codes, date_index)
    del signal_frames

    symbol_count = len(codes)
    cash = float(config.project.initial_cash)
    positions = np.zeros(symbol_count, dtype="float64")
    avg_cost = np.zeros(symbol_count, dtype="float64")
    entry_day_index = np.full(symbol_count, -1, dtype="int32")
    highest_close_since_entry = np.full(symbol_count, np.nan, dtype="float64")
    consecutive_loss_exits = np.zeros(symbol_count, dtype="int16")
    loss_cooldown_until = np.full(symbol_count, -1, dtype="int32")
    last_close = np.full(symbol_count, np.nan, dtype="float64")
    equity_records: list[dict[str, Any]] = []
    trade_records: list[dict[str, Any]] = []
    risk_records: list[dict[str, Any]] = []
    risk_event_records: list[dict[str, Any]] = []
    cooldown_records: list[dict[str, Any]] = []
    benchmark_risk = build_benchmark_risk_frame(config, list(date_index)) if config.risk.enabled else pd.DataFrame()
    risk_exposure = 1.0
    risk_reason = "risk_disabled" if not config.risk.enabled else "initial"
    risk_peak_equity = float(config.project.initial_cash)
    risk_protected = False

    for day_index, current_date in enumerate(date_index):
        open_row = open_values[day_index]
        close_row = close_values[day_index]
        open_valid = np.isfinite(open_row)
        close_valid = np.isfinite(close_row)
        breadth_ratios = market_breadth_ratio_values[day_index]
        breadth_counts = market_breadth_count_values[day_index]
        valid_breadth = np.isfinite(breadth_ratios) & np.isfinite(breadth_counts) & (breadth_counts > 0)
        market_breadth_ratio = float(breadth_ratios[valid_breadth][0]) if np.any(valid_breadth) else 1.0
        market_breadth_count = int(breadth_counts[valid_breadth][0]) if np.any(valid_breadth) else 0
        market_breadth_exposure = 1.0
        if (
            config.strategy.market_breadth_exposure_enabled
            and market_breadth_count >= config.strategy.market_breadth_min_count
            and market_breadth_ratio < config.strategy.market_breadth_weak_ratio
        ):
            market_breadth_exposure = max(0.0, min(float(config.strategy.market_breadth_weak_exposure), 1.0))

        open_value_prices = np.where(open_valid, open_row, last_close)
        held_mask = positions > 0
        valued_open_mask = held_mask & np.isfinite(open_value_prices)
        equity_open = cash + float(np.sum(positions[valued_open_mask] * open_value_prices[valued_open_mask]))

        candidate_mask = (
            (trade_signal_values[day_index] == 1)
            & open_valid
            & (day_index > loss_cooldown_until)
            & (~stock_cap_blocked_values[day_index])
            & ((positions > 0) | (trade_entry_allowed_values[day_index] == 1))
        )
        candidate_indices = np.flatnonzero(candidate_mask)
        candidate_scores = np.nan_to_num(trade_score_values[day_index, candidate_indices], nan=0.0)
        if len(candidate_indices):
            candidate_order = list(candidate_indices[np.argsort(-candidate_scores, kind="stable")])
        else:
            candidate_order = []
        max_positions = max(config.strategy.max_positions, 1)
        selected: list[int] = []
        if config.strategy.prefer_existing_positions:
            held_candidate_indices = [
                int(idx)
                for idx in np.flatnonzero((positions > 0) & candidate_mask)
                if not bool(trade_exit_risk_values[day_index, idx])
            ]
            held_candidate_indices.sort(
                key=lambda idx: (-float(np.nan_to_num(trade_score_values[day_index, idx], nan=0.0)), codes[idx])
            )
            selected.extend(held_candidate_indices[:max_positions])
        if len(selected) < max_positions:
            selected_set = set(selected)
            selected.extend([idx for idx in candidate_order if idx not in selected_set][: max_positions - len(selected)])

        forced_exit_codes = np.zeros(symbol_count, dtype="int8")
        for idx in np.flatnonzero(held_mask & open_valid & (avg_cost > 0)):
            if stock_cap_blocked_values[day_index, idx]:
                forced_exit_codes[idx] = 5
                continue
            cost = avg_cost[idx]
            if cost <= 0:
                continue
            open_return = open_row[idx] / cost - 1
            stop_loss_pct = _as_float(config.strategy.stop_loss_pct)
            dynamic_stop_loss_pct = _as_float(trade_stop_loss_pct_values[day_index, idx])
            if dynamic_stop_loss_pct is not None and dynamic_stop_loss_pct > 0:
                stop_loss_pct = dynamic_stop_loss_pct if stop_loss_pct is None else min(stop_loss_pct, dynamic_stop_loss_pct)
            if stop_loss_pct is not None and stop_loss_pct > 0 and open_return <= -stop_loss_pct:
                forced_exit_codes[idx] = 1
                continue
            if config.strategy.take_profit_pct is not None and open_return >= config.strategy.take_profit_pct:
                forced_exit_codes[idx] = 2
                continue
            if config.strategy.trailing_stop_pct is not None:
                peak = highest_close_since_entry[idx] if np.isfinite(highest_close_since_entry[idx]) else cost
                if peak > cost and open_row[idx] <= peak * (1 - config.strategy.trailing_stop_pct):
                    forced_exit_codes[idx] = 3
                    continue
            if config.strategy.time_stop_enabled:
                held_days = day_index - entry_day_index[idx] if entry_day_index[idx] >= 0 else 0
                if (
                    held_days >= config.strategy.time_stop_min_hold_days
                    and open_return <= config.strategy.time_stop_return_threshold_pct
                ):
                    forced_exit_codes[idx] = 4

        locked_indices: set[int] = set()
        if config.strategy.min_hold_days > 0 and risk_exposure >= 0.999:
            selected_set = set(selected)
            for idx in np.flatnonzero(held_mask & open_valid & (forced_exit_codes == 0)):
                if idx in selected_set:
                    continue
                held_days = day_index - entry_day_index[idx] if entry_day_index[idx] >= 0 else 0
                risk_exit = bool(trade_exit_risk_values[day_index, idx])
                if held_days < config.strategy.min_hold_days and not risk_exit:
                    locked_indices.add(int(idx))
            if locked_indices:
                selected = sorted(locked_indices, key=lambda idx: codes[idx]) + [
                    idx for idx in selected if idx not in locked_indices
                ]
                if len(selected) < max_positions:
                    selected_set = set(selected)
                    selected += [idx for idx in candidate_order if idx not in selected_set][: max_positions - len(selected)]
        target_gross_exposure = config.strategy.allocation_buffer * risk_exposure * market_breadth_exposure
        target_weights = np.zeros(symbol_count, dtype="float64")
        if selected:
            if config.strategy.score_weighted_allocation:
                target_weights[selected] = _score_weighted_position_weights(
                    selected,
                    trade_score_values[day_index],
                    target_gross_exposure,
                    method=config.strategy.score_allocation_method,
                    tilt_strength=config.strategy.score_allocation_tilt_strength,
                    min_multiplier=config.strategy.score_allocation_min_multiplier,
                    max_multiplier=config.strategy.score_allocation_max_multiplier,
                )
            else:
                weight = target_gross_exposure / len(selected)
                target_weights[selected] = weight

        trade_indices = sorted(set(selected) | set(np.flatnonzero(positions > 0)), key=lambda idx: codes[idx])
        deltas: dict[int, float] = {}
        rebalance_loss_guarded_indices: set[int] = set()
        for idx in trade_indices:
            if not open_valid[idx]:
                continue
            current_value = positions[idx] * open_row[idx]
            target_value = equity_open * target_weights[idx]
            if forced_exit_codes[idx] != 0:
                target_value = 0.0
            delta_value = target_value - current_value
            if idx in locked_indices and delta_value < 0:
                delta_value = 0.0
            if (
                config.strategy.rebalance_loss_guard_pct is not None
                and delta_value < -1e-8
                and positions[idx] > 0
                and forced_exit_codes[idx] == 0
                and avg_cost[idx] > 0
                and not bool(trade_exit_risk_values[day_index, idx])
            ):
                open_return = open_row[idx] / avg_cost[idx] - 1
                held_days = day_index - entry_day_index[idx] if entry_day_index[idx] >= 0 else 0
                within_guard_days = (
                    config.strategy.rebalance_loss_guard_max_days == 0
                    or held_days <= config.strategy.rebalance_loss_guard_max_days
                )
                if -config.strategy.rebalance_loss_guard_pct <= open_return < 0 and within_guard_days:
                    delta_value = 0.0
                    rebalance_loss_guarded_indices.add(int(idx))
            deltas[idx] = delta_value

        exit_reason_names = {
            1: "stop_loss",
            2: "take_profit",
            3: "trailing_stop",
            4: "time_stop",
            5: "market_cap_blocked",
        }
        for idx, delta_value in sorted(deltas.items(), key=lambda item: (item[1], codes[item[0]])):
            if delta_value >= -1e-8 or positions[idx] <= 0:
                continue
            exec_price = open_row[idx] * (1 - config.costs.slippage_rate)
            quantity = min(positions[idx], abs(delta_value) / exec_price)
            if quantity <= 0:
                continue
            gross = quantity * exec_price
            fee = gross * (config.costs.commission_rate + config.costs.stamp_tax_rate)
            cash += gross - fee
            positions[idx] -= quantity
            realized_pnl = (exec_price - avg_cost[idx]) * quantity - fee
            position_closed = positions[idx] <= 1e-8
            if position_closed:
                positions[idx] = 0.0
                avg_cost[idx] = 0.0
                entry_day_index[idx] = -1
                highest_close_since_entry[idx] = np.nan
            if position_closed:
                if realized_pnl < 0:
                    consecutive_loss_exits[idx] += 1
                    if (
                        config.strategy.loss_cooldown_days > 0
                        and consecutive_loss_exits[idx] >= config.strategy.loss_cooldown_min_losses
                    ):
                        cooldown_until_index = day_index + config.strategy.loss_cooldown_days
                        cooldown_until_display_index = min(cooldown_until_index, len(date_index) - 1)
                        loss_cooldown_until[idx] = max(loss_cooldown_until[idx], cooldown_until_index)
                        cooldown_records.append(
                            {
                                "date": current_date,
                                "code": codes[idx],
                                "name": names[idx],
                                "trigger": "loss_exit",
                                "exit_reason": exit_reason_names.get(int(forced_exit_codes[idx]), "rebalance"),
                                "realized_pnl": realized_pnl,
                                "consecutive_loss_exits": int(consecutive_loss_exits[idx]),
                                "cooldown_days": config.strategy.loss_cooldown_days,
                                "cooldown_until": date_index[cooldown_until_display_index],
                                "cooldown_until_index": int(cooldown_until_index),
                                "cooldown_beyond_backtest_end": bool(cooldown_until_index >= len(date_index)),
                            }
                        )
                        consecutive_loss_exits[idx] = 0
                else:
                    consecutive_loss_exits[idx] = 0
            trade_records.append(
                {
                    "date": current_date,
                    "code": codes[idx],
                    "name": names[idx],
                    "side": "SELL",
                    "price": exec_price,
                    "quantity": quantity,
                    "gross_amount": gross,
                    "fee": fee,
                    "realized_pnl": realized_pnl,
                    "cash_after": cash,
                    "exit_reason": exit_reason_names.get(int(forced_exit_codes[idx]), "rebalance"),
                }
            )

        buy_orders = _scaled_buy_orders(
            deltas,
            open_row,
            open_valid,
            cash,
            config.costs.commission_rate,
            config.costs.slippage_rate,
            config.strategy.lot_size,
            {
                int(idx): float(trade_score_values[day_index, idx])
                for idx in trade_indices
                if idx in deltas and np.isfinite(trade_score_values[day_index, idx])
            },
        )
        for order in sorted(buy_orders, key=lambda item: codes[item.idx]):
            idx = order.idx
            exec_price = order.exec_price
            quantity = order.quantity
            required_cash = order.required_cash
            if required_cash > cash and cash > 0:
                quantity = _round_buy_quantity(
                    cash / (exec_price * (1 + config.costs.commission_rate)),
                    config.strategy.lot_size,
                )
                required_cash = quantity * exec_price * (1 + config.costs.commission_rate)
            if quantity <= 0 or required_cash > cash + 1e-6:
                continue
            gross = quantity * exec_price
            fee = gross * config.costs.commission_rate
            old_quantity = positions[idx]
            new_quantity = old_quantity + quantity
            avg_cost[idx] = ((avg_cost[idx] * old_quantity) + (exec_price * quantity)) / new_quantity
            positions[idx] = new_quantity
            if old_quantity <= 1e-8:
                entry_day_index[idx] = day_index
                highest_close_since_entry[idx] = close_row[idx] if close_valid[idx] else exec_price
            cash -= gross + fee
            trade_records.append(
                {
                    "date": current_date,
                    "code": codes[idx],
                    "name": names[idx],
                    "side": "BUY",
                    "price": exec_price,
                    "quantity": quantity,
                    "gross_amount": gross,
                    "fee": fee,
                    "realized_pnl": np.nan,
                    "cash_after": cash,
                    "exit_reason": "",
                }
            )

        last_close[close_valid] = close_row[close_valid]
        held_close_indices = np.flatnonzero((positions > 0) & close_valid)
        for idx in held_close_indices:
            previous_peak = highest_close_since_entry[idx] if np.isfinite(highest_close_since_entry[idx]) else close_row[idx]
            highest_close_since_entry[idx] = max(previous_peak, close_row[idx])

        close_value_prices = np.where(close_valid, close_row, last_close)
        valued_close_mask = (positions > 0) & np.isfinite(close_value_prices)
        positions_value = float(np.sum(positions[valued_close_mask] * close_value_prices[valued_close_mask]))
        equity_records.append(
            {
                "date": current_date,
                "equity": cash + positions_value,
                "cash": cash,
                "positions_value": positions_value,
                "active_positions": int(np.sum(positions > 0)),
                "risk_exposure": risk_exposure,
                "stock_cap_blocked_count": int(np.sum(stock_cap_blocked_values[day_index])),
                "market_breadth_ratio": market_breadth_ratio,
                "market_breadth_count": market_breadth_count,
                "market_breadth_exposure": market_breadth_exposure,
                "cooldown_active_count": int(np.sum(day_index <= loss_cooldown_until)),
                "rebalance_loss_guarded_count": len(rebalance_loss_guarded_indices),
            }
        )
        equity_close = cash + positions_value
        if config.risk.enabled:
            benchmark_row = benchmark_risk.loc[current_date] if current_date in benchmark_risk.index else None
            decision = decide_next_risk(
                config.risk,
                equity=equity_close,
                peak_equity=risk_peak_equity,
                protected=risk_protected,
                benchmark_row=benchmark_row,
            )
            risk_records.append(
                {
                    "date": current_date,
                    "risk_exposure": risk_exposure,
                    "next_risk_exposure": decision.exposure,
                    "portfolio_drawdown": decision.drawdown,
                    "peak_equity": decision.peak_equity,
                    "protected": decision.protected,
                    "benchmark_close": decision.benchmark_close,
                    "benchmark_ma": decision.benchmark_ma,
                    "benchmark_return": decision.benchmark_return,
                    "benchmark_rsrs_beta": decision.benchmark_rsrs_beta,
                    "benchmark_rsrs_zscore": decision.benchmark_rsrs_zscore,
                    "benchmark_rsrs_risk_on": decision.benchmark_rsrs_risk_on,
                    "benchmark_risk_on": decision.benchmark_risk_on,
                    "risk_reason": risk_reason,
                    "next_risk_reason": decision.reason,
                }
            )
            if decision.exposure != risk_exposure or decision.protected != risk_protected or decision.reason != risk_reason:
                risk_event_records.append(
                    {
                        "date": current_date,
                        "previous_exposure": risk_exposure,
                        "next_exposure": decision.exposure,
                        "previous_protected": risk_protected,
                        "next_protected": decision.protected,
                        "portfolio_drawdown": decision.drawdown,
                        "reason": decision.reason,
                    }
                )
            risk_exposure = decision.exposure
            risk_reason = decision.reason
            risk_peak_equity = decision.peak_equity
            risk_protected = decision.protected

    equity = pd.DataFrame(equity_records)
    trades = pd.DataFrame(trade_records)
    if not trades.empty:
        trades["date"] = pd.to_datetime(trades["date"])
    benchmark = _benchmark_curve(histories, equity["date"], config.project.initial_cash)
    monthly = _monthly_returns(equity)
    risk_curve = pd.DataFrame(risk_records)
    risk_events = pd.DataFrame(risk_event_records)
    cooldown_events = pd.DataFrame(cooldown_records)
    if not cooldown_events.empty:
        cooldown_events["date"] = pd.to_datetime(cooldown_events["date"])
        cooldown_events["cooldown_until"] = pd.to_datetime(cooldown_events["cooldown_until"])
    metrics = _compute_metrics(equity, trades, benchmark, config.project.initial_cash, risk_curve, cooldown_events)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    result = BacktestResult(
        run_id=run_id,
        run_dir=config.project.output_dir / run_id,
        equity=equity,
        trades=trades,
        benchmark=benchmark,
        monthly_returns=monthly,
        metrics=metrics,
        risk_curve=risk_curve,
        risk_events=risk_events,
        cooldown_events=cooldown_events,
    )
    if write_outputs:
        _write_outputs(config, result)
    return result
