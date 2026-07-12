"""Walk-forward satellite strategy for strict strong-stock pullbacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from execution_rules import apply_open_constraints, next_open_return_label
from generate_strong_pullback_candidates import filter_strong_pullback_candidates, reason
from run_backtest import load_prices, max_drawdown, pivot_prices, sharpe_like
from train_next_open_rank_model import (
    build_breadth_exposure,
    build_features,
    clean_matrix,
    daily_ic,
    load_market_exposure,
    normalize_weights,
)


def build_raw_metric_frames(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    close_range = (high - low).replace(0, np.nan)
    return {
        "close": close,
        "return_5d": close.pct_change(5, fill_method=None),
        "return_20d": close.pct_change(20, fill_method=None),
        "return_60d": close.pct_change(60, fill_method=None),
        "pullback_5d": -close.pct_change(5, fill_method=None),
        "prior_return_20_before_pullback": close.shift(5) / (close.shift(25) + 1e-12) - 1.0,
        "prior_return_60_before_pullback": close.shift(5) / (close.shift(65) + 1e-12) - 1.0,
        "raw_distance_ma20": close / (ma20 + 1e-12) - 1.0,
        "raw_distance_ma60": close / (ma60 + 1e-12) - 1.0,
        "breakout_20d": close / (high.rolling(20).max().shift(1) + 1e-12) - 1.0,
        "raw_intraday_return": close / (open_px + 1e-12) - 1.0,
        "raw_close_position": (close - low) / (close_range + 1e-12),
        "avg_amount_20d": amount.replace(0, np.nan).rolling(20).median(),
    }


def select_target_weights(
    candidates: pd.DataFrame,
    all_symbols: pd.Index,
    top_n: int,
    leverage: float,
    max_position_weight: float,
    min_score: float | None = None,
    basket_guard_return_20d_min: float | None = None,
    basket_guard_distance_ma60_min: float | None = None,
    basket_guard_scale: float = 1.0,
) -> pd.Series:
    target = pd.Series(0.0, index=all_symbols, dtype=float)
    if candidates.empty or top_n <= 0 or leverage <= 0:
        return target
    pool = candidates
    if min_score is not None:
        pool = pool[pool["score"].ge(min_score)]
    selected = pool.sort_values("score", ascending=False).head(top_n)
    if selected.empty:
        return target

    guard = basket_guard_status(
        selected,
        basket_guard_return_20d_min=basket_guard_return_20d_min,
        basket_guard_distance_ma60_min=basket_guard_distance_ma60_min,
        basket_guard_scale=basket_guard_scale,
    )
    effective_leverage = leverage * float(guard["scale"])
    if effective_leverage <= 0:
        return target

    weight = min(max_position_weight, effective_leverage / len(selected))
    target.loc[selected.index] = weight
    gross = float(target.abs().sum())
    if gross > effective_leverage:
        target = target / gross * effective_leverage
    return target


def apply_rebound_exit(
    current: pd.Series,
    target: pd.Series,
    close_row: pd.Series,
    entry_price: pd.Series,
    rebound_exit_return: float | None = None,
    rebound_exit_scale: float = 0.0,
) -> tuple[pd.Series, dict[str, str]]:
    adjusted = target.copy()
    hits: dict[str, str] = {}
    if rebound_exit_return is None:
        return adjusted, hits

    current_aligned = current.reindex(adjusted.index).fillna(0.0)
    close_aligned = close_row.reindex(adjusted.index)
    entry_aligned = entry_price.reindex(adjusted.index)
    rebound = close_aligned / (entry_aligned + 1e-12) - 1.0
    held = current_aligned.gt(0.0) & adjusted.gt(0.0)
    valid_entry = entry_aligned.replace([np.inf, -np.inf], np.nan).gt(0.0)
    hit_mask = held & valid_entry & rebound.ge(float(rebound_exit_return))
    if not bool(hit_mask.any()):
        return adjusted, hits

    adjusted.loc[hit_mask] = adjusted.loc[hit_mask] * float(rebound_exit_scale)
    hits = {str(symbol): "rebound_exit" for symbol in adjusted.index[hit_mask]}
    return adjusted, hits


def should_apply_rebound_exit(
    rebound_exit_return: float | None,
    market_exposure: float,
    market_exposure_max: float | None = None,
    market_exposure_min: float | None = None,
) -> bool:
    if rebound_exit_return is None:
        return False
    exposure = float(market_exposure)
    if market_exposure_max is not None and exposure > float(market_exposure_max):
        return False
    if market_exposure_min is not None and exposure < float(market_exposure_min):
        return False
    return True


def basket_guard_status(
    selected: pd.DataFrame,
    *,
    basket_guard_return_20d_min: float | None = None,
    basket_guard_distance_ma60_min: float | None = None,
    basket_guard_scale: float = 1.0,
) -> dict[str, object]:
    scale = 1.0
    reason = "risk_on"
    avg_return_20d = np.nan
    avg_distance_ma60 = np.nan
    checks = []

    if selected.empty:
        return {
            "scale": scale,
            "reason": "empty",
            "avg_return_20d": avg_return_20d,
            "avg_distance_ma60": avg_distance_ma60,
        }

    if basket_guard_return_20d_min is not None and "return_20d" in selected:
        avg_return_20d = float(pd.to_numeric(selected["return_20d"], errors="coerce").mean())
        checks.append(avg_return_20d < float(basket_guard_return_20d_min))

    if basket_guard_distance_ma60_min is not None and "raw_distance_ma60" in selected:
        avg_distance_ma60 = float(pd.to_numeric(selected["raw_distance_ma60"], errors="coerce").mean())
        checks.append(avg_distance_ma60 < float(basket_guard_distance_ma60_min))

    if checks and all(checks):
        scale = float(basket_guard_scale)
        reason = "basket_weak"

    return {
        "scale": scale,
        "reason": reason,
        "avg_return_20d": avg_return_20d,
        "avg_distance_ma60": avg_distance_ma60,
    }


def build_daily_candidates(
    features: dict[str, pd.DataFrame],
    raw_metrics: dict[str, pd.DataFrame],
    weights: pd.Series,
    row_index: int,
    **filter_kwargs: float,
) -> pd.DataFrame:
    score = None
    for name, weight in weights.items():
        if name not in features:
            continue
        part = features[name].iloc[row_index].fillna(0.0) * weight
        score = part if score is None else score + part
    if score is None:
        return pd.DataFrame()

    feature_snapshot = pd.DataFrame({name: frame.iloc[row_index] for name, frame in features.items()})
    raw_snapshot = pd.DataFrame({name: frame.iloc[row_index] for name, frame in raw_metrics.items()})
    candidates = feature_snapshot.join(raw_snapshot)
    candidates["score"] = score
    candidates = candidates.dropna(subset=["score", "close"])
    return filter_strong_pullback_candidates(candidates, **filter_kwargs).sort_values(
        "score", ascending=False
    )


def run_satellite_walk_forward(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
    train_days: int,
    retrain_frequency: int,
    top_n: int,
    rebalance_frequency: int,
    max_position_weight: float,
    leverage: float,
    min_score: float | None,
    commission_bps: float,
    impact_bps: float,
    max_buy_open_gap: float,
    limit_buffer: float,
    market_exposure: pd.Series,
    initial_capital: float,
    filter_kwargs: dict[str, float],
    basket_guard_return_20d_min: float | None = None,
    basket_guard_distance_ma60_min: float | None = None,
    basket_guard_scale: float = 1.0,
    rebound_exit_return: float | None = None,
    rebound_exit_scale: float = 0.0,
    rebound_exit_market_exposure_max: float | None = None,
    rebound_exit_market_exposure_min: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = build_features(close, open_px, high, low, amount)
    raw_metrics = build_raw_metric_frames(close, open_px, high, low, amount)
    label = next_open_return_label(open_px)
    ic = daily_ic(features, label)

    equity = initial_capital
    positions = pd.Series(0.0, index=close.columns)
    entry_price = pd.Series(np.nan, index=close.columns, dtype=float)
    current_weights = pd.Series(1.0 / len(features), index=list(features))
    nav_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    guard_state = {
        "scale": 1.0,
        "reason": "risk_on",
        "avg_return_20d": np.nan,
        "avg_distance_ma60": np.nan,
    }

    for i in range(train_days, len(close.index) - 2):
        date = close.index[i]
        rebound_hits: dict[str, str] = {}
        if (i - train_days) % retrain_frequency == 0:
            current_weights = normalize_weights(ic.iloc[i - train_days : i].mean())
            weight_rows.append({"date": date.strftime("%Y-%m-%d"), **current_weights.to_dict()})

        candidates = pd.DataFrame()
        if (i - train_days) % rebalance_frequency == 0:
            candidates = build_daily_candidates(
                features,
                raw_metrics,
                current_weights,
                i,
                **filter_kwargs,
            )
            selected_for_guard = candidates
            if min_score is not None and not selected_for_guard.empty:
                selected_for_guard = selected_for_guard[selected_for_guard["score"].ge(min_score)]
            selected_for_guard = selected_for_guard.sort_values("score", ascending=False).head(top_n)
            guard_state = basket_guard_status(
                selected_for_guard,
                basket_guard_return_20d_min=basket_guard_return_20d_min,
                basket_guard_distance_ma60_min=basket_guard_distance_ma60_min,
                basket_guard_scale=basket_guard_scale,
            )
            target = select_target_weights(
                candidates,
                all_symbols=close.columns,
                top_n=top_n,
                leverage=leverage,
                max_position_weight=max_position_weight,
                min_score=min_score,
                basket_guard_return_20d_min=basket_guard_return_20d_min,
                basket_guard_distance_ma60_min=basket_guard_distance_ma60_min,
                basket_guard_scale=basket_guard_scale,
            )
            exposure = float(market_exposure.reindex([date]).iloc[0])
            target = target * exposure
        else:
            target = positions.copy()
            exposure = float(market_exposure.reindex([date]).iloc[0])

        rebound_exit_enabled = should_apply_rebound_exit(
            rebound_exit_return,
            market_exposure=exposure,
            market_exposure_max=rebound_exit_market_exposure_max,
            market_exposure_min=rebound_exit_market_exposure_min,
        )
        target, rebound_hits = apply_rebound_exit(
            positions,
            target,
            close.iloc[i],
            entry_price,
            rebound_exit_return=rebound_exit_return if rebound_exit_enabled else None,
            rebound_exit_scale=rebound_exit_scale,
        )
        target = apply_open_constraints(
            positions,
            target,
            open_px.iloc[i + 1],
            close.iloc[i],
            max_buy_open_gap=max_buy_open_gap,
            limit_buffer=limit_buffer,
        )
        active_rebound_hits = {
            str(symbol): reason
            for symbol, reason in rebound_hits.items()
            if symbol in target.index and float(target.loc[symbol]) < float(positions.loc[symbol])
        }

        selected_symbols = target[target.gt(0)].index
        if not candidates.empty and len(selected_symbols) > 0:
            selected_candidates = candidates.reindex(selected_symbols).dropna(subset=["score"])
            for symbol, row in selected_candidates.iterrows():
                candidate_rows.append(
                    {
                        "signal_date": date.strftime("%Y-%m-%d"),
                        "symbol": str(symbol).zfill(6),
                        "target_weight": float(target.loc[symbol]),
                        "score": float(row["score"]),
                        "reason": reason(row),
                        "close": float(row["close"]),
                        "return_5d": float(row["return_5d"]),
                        "return_20d": float(row["return_20d"]),
                        "return_60d": float(row["return_60d"]),
                        "prior_return_20_before_pullback": float(row["prior_return_20_before_pullback"]),
                        "prior_return_60_before_pullback": float(row["prior_return_60_before_pullback"]),
                        "raw_distance_ma60": float(row["raw_distance_ma60"]),
                        "avg_amount_20d": float(row["avg_amount_20d"]),
                        "basket_guard_scale": float(guard_state["scale"]),
                        "basket_guard_reason": str(guard_state["reason"]),
                    }
                )

        turnover = float((target - positions).abs().sum())
        cost = turnover * (commission_bps + impact_bps) / 1e4
        realized = label.iloc[i].reindex(close.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        gross_return = float((target * realized).sum())
        symbol_contributions = (target * realized).replace(0.0, np.nan).dropna()
        equity *= 1.0 + gross_return - cost

        new_entries = target.gt(0.0) & positions.le(0.0)
        exits = target.le(0.0)
        next_open = open_px.iloc[i + 1].reindex(close.columns)
        entry_price.loc[new_entries] = next_open.loc[new_entries]
        entry_price.loc[exits] = np.nan
        positions = target

        nav_rows.append(
            {
                "date": close.index[i + 2].strftime("%Y-%m-%d"),
                "equity": equity,
                "gross_return": gross_return,
                "cost": cost,
                "turnover": turnover,
                "gross_exposure": float(positions.abs().sum()),
                "market_exposure": exposure,
                "positions_count": int(positions.ne(0).sum()),
                "candidate_count": int(len(candidates)) if not candidates.empty else 0,
                "basket_guard_scale": float(guard_state["scale"]),
                "basket_guard_reason": str(guard_state["reason"]),
                "basket_avg_return_20d": float(guard_state["avg_return_20d"]),
                "basket_avg_distance_ma60": float(guard_state["avg_distance_ma60"]),
                "rebound_exit_enabled": bool(rebound_exit_enabled),
                "rebound_exit_count": int(len(active_rebound_hits)),
                "rebound_exit_symbols": ",".join(sorted(active_rebound_hits)),
            }
        )
        trade_rows.append(
            {
                "signal_date": date.strftime("%Y-%m-%d"),
                "realize_date": close.index[i + 2].strftime("%Y-%m-%d"),
                "turnover": turnover,
                "gross_return": gross_return,
                "symbol_contributions_json": json.dumps(
                    {str(symbol).zfill(6): float(value) for symbol, value in symbol_contributions.items()},
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                "cost": cost,
                "positions_count": int(positions.ne(0).sum()),
                "candidate_count": int(len(candidates)) if not candidates.empty else 0,
                "basket_guard_scale": float(guard_state["scale"]),
                "basket_guard_reason": str(guard_state["reason"]),
                "basket_avg_return_20d": float(guard_state["avg_return_20d"]),
                "basket_avg_distance_ma60": float(guard_state["avg_distance_ma60"]),
                "rebound_exit_enabled": bool(rebound_exit_enabled),
                "rebound_exit_count": int(len(active_rebound_hits)),
                "rebound_exit_symbols": ",".join(sorted(active_rebound_hits)),
            }
        )

    return (
        pd.DataFrame(nav_rows),
        pd.DataFrame(weight_rows),
        pd.DataFrame(trade_rows),
        pd.DataFrame(candidate_rows),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest strict strong-pullback satellite strategy.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/high_return_v2/strong_pullback_satellite")
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--retrain-frequency", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--rebalance-frequency", type=int, default=5)
    parser.add_argument("--max-position-weight", type=float, default=0.08)
    parser.add_argument("--leverage", type=float, default=0.60)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--impact-bps", type=float, default=0.7)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--max-buy-open-gap", type=float, default=0.05)
    parser.add_argument("--limit-buffer", type=float, default=0.995)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--market-ma-window", type=int, default=120)
    parser.add_argument("--market-risk-off-drawdown-20d", type=float, default=-0.08)
    parser.add_argument("--market-below-ma-exposure", type=float, default=0.60)
    parser.add_argument("--market-crash-exposure", type=float, default=0.0)
    parser.add_argument("--breadth-filter", action="store_true")
    parser.add_argument("--breadth-ma-window", type=int, default=60)
    parser.add_argument("--breadth-threshold", type=float, default=0.45)
    parser.add_argument("--breadth-below-exposure", type=float, default=0.55)
    parser.add_argument("--breadth-crash-threshold", type=float, default=0.32)
    parser.add_argument("--breadth-crash-exposure", type=float, default=0.20)
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--min-close", type=float, default=2.0)
    parser.add_argument("--min-avg-amount-20d", type=float, default=30_000_000.0)
    parser.add_argument("--min-pullback-5d", type=float, default=0.03)
    parser.add_argument("--max-pullback-5d", type=float, default=0.18)
    parser.add_argument("--min-prior-return-20", type=float, default=0.08)
    parser.add_argument("--min-prior-return-60", type=float, default=0.18)
    parser.add_argument("--min-return-20d", type=float, default=-0.12)
    parser.add_argument("--min-return-60d", type=float, default=0.0)
    parser.add_argument("--min-distance-ma60", type=float, default=-0.10)
    parser.add_argument("--max-intraday-return", type=float, default=0.05)
    parser.add_argument("--basket-risk-guard", action="store_true")
    parser.add_argument("--basket-guard-return-20d-min", type=float, default=-0.08)
    parser.add_argument("--basket-guard-distance-ma60-min", type=float, default=-0.03)
    parser.add_argument("--basket-guard-scale", type=float, default=0.0)
    parser.add_argument("--rebound-exit-return", type=float, default=None)
    parser.add_argument("--rebound-exit-scale", type=float, default=0.0)
    parser.add_argument("--rebound-exit-market-exposure-max", type=float, default=None)
    parser.add_argument("--rebound-exit-market-exposure-min", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_prices(Path(args.data), None, None)
    close = clean_matrix(pivot_prices(raw, "close"), args.max_abs_daily_return)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), args.max_abs_daily_return)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), args.max_abs_daily_return)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), args.max_abs_daily_return)
    amount = pivot_prices(raw, "amount").reindex_like(close)

    market_exposure = load_market_exposure(
        args.benchmark,
        close.index,
        ma_window=args.market_ma_window,
        risk_off_drawdown_20d=args.market_risk_off_drawdown_20d,
        below_ma_exposure=args.market_below_ma_exposure,
        crash_exposure=args.market_crash_exposure,
    )
    if args.breadth_filter:
        breadth_exposure = build_breadth_exposure(
            close,
            ma_window=args.breadth_ma_window,
            threshold=args.breadth_threshold,
            below_exposure=args.breadth_below_exposure,
            crash_threshold=args.breadth_crash_threshold,
            crash_exposure=args.breadth_crash_exposure,
        )
        market_exposure = pd.concat([market_exposure, breadth_exposure], axis=1).min(axis=1)

    filter_kwargs = {
        "min_close": args.min_close,
        "min_avg_amount_20d": args.min_avg_amount_20d,
        "min_pullback_5d": args.min_pullback_5d,
        "max_pullback_5d": args.max_pullback_5d,
        "min_prior_return_20": args.min_prior_return_20,
        "min_prior_return_60": args.min_prior_return_60,
        "min_return_20d": args.min_return_20d,
        "min_return_60d": args.min_return_60d,
        "min_distance_ma60": args.min_distance_ma60,
        "max_intraday_return": args.max_intraday_return,
    }
    equity, weights, trades, candidates = run_satellite_walk_forward(
        close=close,
        open_px=open_px,
        high=high,
        low=low,
        amount=amount,
        train_days=args.train_days,
        retrain_frequency=args.retrain_frequency,
        top_n=args.top_n,
        rebalance_frequency=args.rebalance_frequency,
        max_position_weight=args.max_position_weight,
        leverage=args.leverage,
        min_score=args.min_score,
        commission_bps=args.commission_bps,
        impact_bps=args.impact_bps,
        max_buy_open_gap=args.max_buy_open_gap,
        limit_buffer=args.limit_buffer,
        market_exposure=market_exposure,
        initial_capital=args.initial_capital,
        filter_kwargs=filter_kwargs,
        basket_guard_return_20d_min=args.basket_guard_return_20d_min if args.basket_risk_guard else None,
        basket_guard_distance_ma60_min=args.basket_guard_distance_ma60_min if args.basket_risk_guard else None,
        basket_guard_scale=args.basket_guard_scale,
        rebound_exit_return=args.rebound_exit_return,
        rebound_exit_scale=args.rebound_exit_scale,
        rebound_exit_market_exposure_max=args.rebound_exit_market_exposure_max,
        rebound_exit_market_exposure_min=args.rebound_exit_market_exposure_min,
    )
    if equity.empty:
        raise ValueError("No backtest rows generated. Check train-days and input history length.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    equity.to_csv(output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(output_dir / "rolling_feature_weights.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(output_dir / "trade_audit.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(output_dir / "selected_candidates.csv", index=False, encoding="utf-8-sig")

    nav = pd.Series(equity["equity"].to_numpy(), index=pd.to_datetime(equity["date"]))
    returns = nav.pct_change(fill_method=None).fillna(0.0)
    total_return = float(nav.iloc[-1] / args.initial_capital - 1.0)
    metrics = {
        "strategy": "strong_pullback_satellite",
        "initial_capital": args.initial_capital,
        "final_equity": float(nav.iloc[-1]),
        "total_return": total_return,
        "annualized_return": float((1.0 + total_return) ** (252 / max(len(nav), 1)) - 1.0),
        "max_drawdown": float(max_drawdown(nav)),
        "sharpe_like": float(sharpe_like(returns)),
        "trade_days": int(len(nav)),
        "avg_turnover": float(equity["turnover"].mean()),
        "avg_gross_exposure": float(equity["gross_exposure"].mean()),
        "avg_market_exposure": float(equity["market_exposure"].mean()),
        "avg_positions_count": float(equity["positions_count"].mean()),
        "avg_candidate_count": float(equity["candidate_count"].mean()),
        "rebalance_frequency": args.rebalance_frequency,
        "top_n": args.top_n,
        "leverage": args.leverage,
        "max_position_weight": args.max_position_weight,
        "min_score": args.min_score,
        "filter": filter_kwargs,
        "basket_risk_guard": {
            "enabled": bool(args.basket_risk_guard),
            "return_20d_min": args.basket_guard_return_20d_min,
            "distance_ma60_min": args.basket_guard_distance_ma60_min,
            "scale": args.basket_guard_scale,
            "trigger_count": int(equity["basket_guard_reason"].eq("basket_weak").sum()),
            "avg_scale": float(equity["basket_guard_scale"].mean()),
        },
        "rebound_exit": {
            "enabled": args.rebound_exit_return is not None,
            "return": args.rebound_exit_return,
            "scale": args.rebound_exit_scale,
            "market_exposure_max": args.rebound_exit_market_exposure_max,
            "market_exposure_min": args.rebound_exit_market_exposure_min,
            "enabled_day_count": int(equity["rebound_exit_enabled"].sum()),
            "trigger_count": int(equity["rebound_exit_count"].sum()),
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Strong-pullback satellite outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
