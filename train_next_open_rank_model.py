"""Walk-forward rank model for next-open tradable returns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from execution_rules import apply_open_constraints, next_open_return_label
from run_backtest import annualized_return, clean_symbol, load_prices, max_drawdown, pivot_prices, sharpe_like


MAIN_CHINEXT_PREFIXES = ("000", "001", "002", "003", "300", "301", "600", "601", "603", "605")


def clean_matrix(frame: pd.DataFrame, max_abs_return: float) -> pd.DataFrame:
    if max_abs_return <= 0:
        return frame
    ret = frame.pct_change(fill_method=None)
    return frame.mask(ret.abs().gt(max_abs_return))


def rank_pct(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, pct=True)


def build_features(close: pd.DataFrame, open_px: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, amount: pd.DataFrame) -> dict[str, pd.DataFrame]:
    returns = close.pct_change(fill_method=None)
    ma20 = close.rolling(20).mean()
    prev_high_20 = high.rolling(20).max().shift(1)
    close_range = (high - low).replace(0, np.nan)
    raw = {
        "momentum_5": close.pct_change(5, fill_method=None),
        "momentum_20": close.pct_change(20, fill_method=None),
        "momentum_60": close.pct_change(60, fill_method=None),
        "reversal_5": -close.pct_change(5, fill_method=None),
        "breakout_20": close / (prev_high_20 + 1e-12) - 1.0,
        "distance_ma20": close / (ma20 + 1e-12) - 1.0,
        "volatility_20": returns.rolling(20).std(),
        "liquidity_20": np.log1p(amount.replace(0, np.nan).rolling(20).median()),
        "intraday_return": close / (open_px + 1e-12) - 1.0,
        "close_position": (close - low) / (close_range + 1e-12),
    }
    ranked = {name: rank_pct(frame) for name, frame in raw.items()}

    # Distill the close-to-close advantage into next-open tradable patterns:
    # keep intermediate-term strength, but avoid buying immediately after short-term exhaustion.
    ranked["strong_pullback_20_5"] = rank_pct(ranked["momentum_20"] * ranked["reversal_5"])
    ranked["strong_pullback_60_5"] = rank_pct(ranked["momentum_60"] * ranked["reversal_5"])
    ranked["breakout_pullback_20_5"] = rank_pct(ranked["breakout_20"] * ranked["reversal_5"])
    ranked["anti_chase_intraday"] = rank_pct((1.0 - ranked["intraday_return"]) * (1.0 - ranked["close_position"]))
    ranked["liquid_pullback"] = rank_pct(ranked["liquidity_20"] * ranked["reversal_5"])
    return ranked


def daily_ic(features: dict[str, pd.DataFrame], label: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date in label.index:
        y = label.loc[date]
        row = {"date": date}
        for name, feature in features.items():
            x = feature.loc[date]
            both = pd.concat([x, y], axis=1, keys=["x", "y"]).dropna()
            row[name] = both["x"].corr(both["y"], method="spearman") if len(both) >= 30 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def normalize_weights(weights: pd.Series) -> pd.Series:
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    denom = weights.abs().sum()
    if denom <= 1e-12:
        return pd.Series(1.0 / len(weights), index=weights.index)
    return weights / denom


def load_market_exposure(
    benchmark_path: str | None,
    trade_dates: pd.Index,
    ma_window: int,
    risk_off_drawdown_20d: float,
    below_ma_exposure: float,
    crash_exposure: float,
) -> pd.Series:
    if not benchmark_path:
        return pd.Series(1.0, index=trade_dates)
    bench = pd.read_csv(benchmark_path, parse_dates=["date"])
    bench["close"] = pd.to_numeric(bench["close"], errors="coerce")
    bench = bench.dropna(subset=["date", "close"]).sort_values("date")
    s = bench.set_index("date")["close"].reindex(trade_dates).ffill()
    ma = s.rolling(ma_window).mean()
    ret20 = s.pct_change(20, fill_method=None)
    exposure = pd.Series(1.0, index=trade_dates)
    exposure = exposure.where(~s.lt(ma), below_ma_exposure)
    exposure = exposure.where(~ret20.le(risk_off_drawdown_20d), crash_exposure)
    return exposure.fillna(1.0).clip(lower=0.0, upper=1.0)


def build_breadth_exposure(
    close: pd.DataFrame,
    ma_window: int,
    threshold: float,
    below_exposure: float,
    crash_threshold: float,
    crash_exposure: float,
) -> pd.Series:
    ma = close.rolling(ma_window).mean()
    breadth = close.gt(ma).mean(axis=1)
    exposure = pd.Series(1.0, index=close.index)
    exposure = exposure.where(~breadth.lt(threshold), below_exposure)
    exposure = exposure.where(~breadth.lt(crash_threshold), crash_exposure)
    return exposure.fillna(1.0).clip(lower=0.0, upper=1.0)


def run_walk_forward(
    close: pd.DataFrame,
    open_px: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    label: pd.DataFrame,
    ic: pd.DataFrame,
    train_days: int,
    retrain_frequency: int,
    top_n: int,
    rebalance_frequency: int,
    max_position_weight: float,
    leverage: float,
    commission_bps: float,
    impact_bps: float,
    max_buy_open_gap: float,
    limit_buffer: float,
    market_exposure: pd.Series,
    initial_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    equity = initial_capital
    positions = pd.Series(0.0, index=close.columns)
    current_weights = pd.Series(1.0 / len(features), index=list(features))
    nav_rows = []
    weight_rows = []
    trade_rows = []

    for i in range(train_days, len(close.index) - 2):
        date = close.index[i]
        if (i - train_days) % retrain_frequency == 0:
            current_weights = normalize_weights(ic.iloc[i - train_days : i].mean())
            weight_rows.append({"date": date.strftime("%Y-%m-%d"), **current_weights.to_dict()})

        if (i - train_days) % rebalance_frequency != 0:
            target = positions.copy()
        else:
            score = None
            for name, weight in current_weights.items():
                frame = features[name]
                score = frame.iloc[i] * weight if score is None else score + frame.iloc[i] * weight

            candidates = score.dropna()
            if candidates.empty:
                target = pd.Series(0.0, index=close.columns)
            else:
                selected = candidates.nlargest(top_n).index
                raw_weight = min(max_position_weight, leverage / max(len(selected), 1))
                target = pd.Series(0.0, index=close.columns)
                target.loc[selected] = raw_weight
                gross = target.abs().sum()
                if gross > leverage:
                    target = target / gross * leverage
            exposure = float(market_exposure.reindex([date]).iloc[0])
            target = target * exposure

        target = apply_open_constraints(
            positions,
            target,
            open_px.iloc[i + 1],
            close.iloc[i],
            max_buy_open_gap=max_buy_open_gap,
            limit_buffer=limit_buffer,
        )

        turnover = float((target - positions).abs().sum())
        cost = turnover * (commission_bps + impact_bps) / 1e4
        realized = label.iloc[i].reindex(close.columns).fillna(0.0)
        gross_return = float((target * realized).sum())
        equity *= 1.0 + gross_return - cost
        positions = target

        nav_rows.append(
            {
                "date": close.index[i + 2].strftime("%Y-%m-%d"),
                "equity": equity,
                "gross_return": gross_return,
                "cost": cost,
                "turnover": turnover,
                "gross_exposure": float(positions.abs().sum()),
                "market_exposure": float(market_exposure.reindex([date]).iloc[0]),
                "positions_count": int(positions.ne(0).sum()),
            }
        )
        trade_rows.append(
            {
                "signal_date": date.strftime("%Y-%m-%d"),
                "realize_date": close.index[i + 2].strftime("%Y-%m-%d"),
                "turnover": turnover,
                "gross_return": gross_return,
                "cost": cost,
            }
        )

    return pd.DataFrame(nav_rows), pd.DataFrame(weight_rows), pd.DataFrame(trade_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a walk-forward next-open rank model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/high_return_v2/next_open_rank_model")
    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--retrain-frequency", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--rebalance-frequency", type=int, default=1)
    parser.add_argument("--max-position-weight", type=float, default=0.04)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--impact-bps", type=float, default=0.7)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--max-buy-open-gap", type=float, default=0.06)
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
    parser.add_argument("--initial-capital", type=float, default=1000000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_prices(Path(args.data), None, None)
    close = clean_matrix(pivot_prices(raw, "close"), args.max_abs_daily_return)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), args.max_abs_daily_return)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), args.max_abs_daily_return)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), args.max_abs_daily_return)
    amount = pivot_prices(raw, "amount").reindex_like(close)

    features = build_features(close, open_px, high, low, amount)
    label = next_open_return_label(open_px, max_abs_daily_return=args.max_abs_daily_return)
    ic = daily_ic(features, label)
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
    equity, weights, trades = run_walk_forward(
        close=close,
        open_px=open_px,
        features=features,
        label=label,
        ic=ic,
        train_days=args.train_days,
        retrain_frequency=args.retrain_frequency,
        top_n=args.top_n,
        rebalance_frequency=args.rebalance_frequency,
        max_position_weight=args.max_position_weight,
        leverage=args.leverage,
        commission_bps=args.commission_bps,
        impact_bps=args.impact_bps,
        max_buy_open_gap=args.max_buy_open_gap,
        limit_buffer=args.limit_buffer,
        market_exposure=market_exposure,
        initial_capital=args.initial_capital,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    equity.to_csv(output_dir / "equity_curve.csv", index=False, encoding="utf-8")
    weights.to_csv(output_dir / "rolling_feature_weights.csv", index=False, encoding="utf-8")
    trades.to_csv(output_dir / "trade_audit.csv", index=False, encoding="utf-8")
    ic.to_csv(output_dir / "daily_feature_ic.csv", encoding="utf-8")

    nav = pd.Series(equity["equity"].to_numpy(), index=pd.to_datetime(equity["date"]))
    returns = nav.pct_change(fill_method=None).fillna(0.0)
    metrics = {
        "initial_capital": args.initial_capital,
        "final_equity": float(nav.iloc[-1]),
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1),
        "annualized_return": float(annualized_return(nav)),
        "max_drawdown": float(max_drawdown(nav)),
        "sharpe_like": float(sharpe_like(returns)),
        "trade_days": int(len(nav)),
        "avg_turnover": float(equity["turnover"].mean()),
        "avg_gross_exposure": float(equity["gross_exposure"].mean()),
        "avg_market_exposure": float(equity["market_exposure"].mean()),
        "avg_positions_count": float(equity["positions_count"].mean()),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Rank model outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
