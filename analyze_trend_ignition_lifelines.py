"""Extract ignition points and lifelines for long trend runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import load_prices, pivot_prices
from train_next_open_rank_model import clean_matrix


def find_ignition_candidates(
    close: pd.Series,
    high: pd.Series,
    amount: pd.Series,
    *,
    breakout_window: int = 60,
    amount_window: int = 20,
    amount_multiplier: float = 1.5,
    breakout_buffer: float = 0.01,
) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce").sort_index()
    high = pd.to_numeric(high, errors="coerce").reindex(close.index)
    amount = pd.to_numeric(amount, errors="coerce").reindex(close.index)

    prev_high = high.rolling(
        breakout_window, min_periods=max(20, breakout_window // 2)
    ).max().shift(1)
    amount_median = amount.replace(0, np.nan).rolling(
        amount_window, min_periods=max(5, amount_window // 2)
    ).median().shift(1)
    ma20 = close.rolling(20, min_periods=20).mean()
    ma60 = close.rolling(60, min_periods=60).mean()
    amount_ratio = amount / (amount_median + 1e-12)

    breakout = close.gt(prev_high * (1.0 + breakout_buffer))
    volume_ok = amount_ratio.ge(amount_multiplier)
    trend_ok = close.gt(ma60) & ma20.ge(ma60 * 0.98)
    valid = breakout & volume_ok & trend_ok

    out = pd.DataFrame(
        {
            "close": close,
            "prev_high": prev_high,
            "breakout_pct": close / (prev_high + 1e-12) - 1.0,
            "amount": amount,
            "amount_ratio": amount_ratio,
            "ma20": ma20,
            "ma60": ma60,
        }
    )
    return out.loc[valid].dropna(subset=["close", "prev_high"])


def choose_lifeline(
    close: pd.Series,
    ignition_date: pd.Timestamp,
    peak_date: pd.Timestamp,
    *,
    windows: tuple[int, ...] = (20, 30, 60, 120),
    breach_buffer: float = 0.03,
    max_breach_days: int = 3,
    end_date: pd.Timestamp | None = None,
) -> dict[str, object]:
    close = pd.to_numeric(close, errors="coerce").sort_index()
    segment = close.loc[ignition_date:peak_date].dropna()
    candidates: list[dict[str, object]] = []

    for window in windows:
        line = close.rolling(window, min_periods=max(5, window // 2)).mean()
        aligned = pd.concat(
            [segment.rename("close"), line.reindex(segment.index).rename("line")],
            axis=1,
        ).dropna()
        if aligned.empty:
            continue
        distance = aligned["close"] / (aligned["line"] + 1e-12) - 1.0
        breach = distance.lt(-breach_buffer)
        candidates.append(
            {
                "lifeline_ma": window,
                "breach_days_to_peak": int(breach.sum()),
                "min_distance_to_lifeline": float(distance.min()),
                "avg_distance_to_lifeline": float(distance.mean()),
                "line": line,
            }
        )

    if not candidates:
        return {
            "lifeline_ma": None,
            "breach_days_to_peak": None,
            "min_distance_to_lifeline": None,
            "avg_distance_to_lifeline": None,
            "lifeline_break_date": None,
            "lifeline_status": "no_lifeline",
        }

    valid = [row for row in candidates if row["breach_days_to_peak"] <= max_breach_days]
    if valid:
        chosen = sorted(valid, key=lambda row: int(row["lifeline_ma"]))[0]
    else:
        chosen = sorted(
            candidates,
            key=lambda row: (
                int(row["breach_days_to_peak"]),
                -float(row["min_distance_to_lifeline"]),
                int(row["lifeline_ma"]),
            ),
        )[0]

    line = chosen.pop("line")
    break_date = None
    if end_date is not None:
        after_peak = close.loc[peak_date:end_date].iloc[1:].dropna()
        if not after_peak.empty:
            after_line = line.reindex(after_peak.index)
            broken = after_peak.lt(after_line * (1.0 - breach_buffer))
            if broken.any():
                break_date = broken[broken].index[0]

    chosen["lifeline_break_date"] = break_date.strftime("%Y-%m-%d") if break_date is not None else None
    chosen["lifeline_status"] = "broken" if break_date is not None else "alive_or_unbroken"
    return chosen


def _max_drawdown(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return 0.0
    return float((values / values.cummax() - 1.0).min())


def summarize_trends_for_symbol(
    symbol: str,
    close: pd.Series,
    high: pd.Series,
    amount: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    min_trend_return: float = 0.80,
    ignition_lookback_days: int = 120,
    breakout_window: int = 60,
    amount_window: int = 20,
    amount_multiplier: float = 1.5,
    breakout_buffer: float = 0.01,
    lifeline_breach_buffer: float = 0.03,
    lifeline_max_breach_days: int = 3,
    lifeline_windows: tuple[int, ...] = (20, 30, 60, 120),
) -> list[dict[str, object]]:
    close = pd.to_numeric(close, errors="coerce").sort_index()
    high = pd.to_numeric(high, errors="coerce").reindex(close.index)
    amount = pd.to_numeric(amount, errors="coerce").reindex(close.index)
    if close.loc[:end].dropna().empty:
        return []

    ignition_start = start - pd.Timedelta(days=ignition_lookback_days)
    ignitions = find_ignition_candidates(
        close,
        high,
        amount,
        breakout_window=breakout_window,
        amount_window=amount_window,
        amount_multiplier=amount_multiplier,
        breakout_buffer=breakout_buffer,
    )
    ignitions = ignitions.loc[(ignitions.index >= ignition_start) & (ignitions.index <= end)]

    records: list[dict[str, object]] = []
    occupied_until: pd.Timestamp | None = None
    for ignition_date, ignition in ignitions.iterrows():
        if occupied_until is not None and ignition_date <= occupied_until:
            continue
        future_window = close.loc[max(ignition_date, start):end].dropna()
        if future_window.empty:
            continue
        peak_date = future_window.idxmax()
        peak_close = float(future_window.loc[peak_date])
        ignition_close = float(close.loc[ignition_date])
        if ignition_close <= 0:
            continue
        peak_return = peak_close / ignition_close - 1.0
        if peak_return < min_trend_return:
            continue

        period_start_close = close.loc[close.index >= start].dropna()
        period_start_price = float(period_start_close.iloc[0]) if not period_start_close.empty else np.nan
        close_to_end = close.loc[:end].dropna()
        end_close = float(close_to_end.iloc[-1]) if not close_to_end.empty else np.nan
        trend_path = close.loc[ignition_date:peak_date]
        lifeline = choose_lifeline(
            close,
            ignition_date,
            peak_date,
            windows=lifeline_windows,
            breach_buffer=lifeline_breach_buffer,
            max_breach_days=lifeline_max_breach_days,
            end_date=end,
        )
        records.append(
            {
                "symbol": str(symbol).zfill(6),
                "ignition_date": ignition_date.strftime("%Y-%m-%d"),
                "ignition_close": ignition_close,
                "peak_date": peak_date.strftime("%Y-%m-%d"),
                "peak_close": peak_close,
                "peak_return": peak_return,
                "period_return_to_peak": peak_close / period_start_price - 1.0 if period_start_price > 0 else np.nan,
                "return_to_end_from_ignition": end_close / ignition_close - 1.0 if end_close > 0 else np.nan,
                "days_to_peak": int((peak_date - ignition_date).days),
                "max_drawdown_to_peak": _max_drawdown(trend_path),
                "breakout_pct": float(ignition["breakout_pct"]),
                "amount_ratio": float(ignition["amount_ratio"]),
                **lifeline,
            }
        )
        occupied_until = peak_date
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze long trend ignition points and lifelines.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="outputs/high_return_v2/trend_ignition_lifelines_202506_202606")
    parser.add_argument("--start", default="2025-06-01")
    parser.add_argument("--end", default="2026-06-26")
    parser.add_argument("--min-trend-return", type=float, default=0.80)
    parser.add_argument("--ignition-lookback-days", type=int, default=120)
    parser.add_argument("--breakout-window", type=int, default=60)
    parser.add_argument("--amount-window", type=int, default=20)
    parser.add_argument("--amount-multiplier", type=float, default=1.5)
    parser.add_argument("--breakout-buffer", type=float, default=0.01)
    parser.add_argument("--lifeline-breach-buffer", type=float, default=0.03)
    parser.add_argument("--lifeline-max-breach-days", type=int, default=3)
    parser.add_argument("--max-abs-daily-return", type=float, default=0.22)
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def to_chinese_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "symbol": "股票代码",
            "ignition_date": "起爆日",
            "ignition_close": "起爆收盘价",
            "peak_date": "峰值日",
            "peak_close": "峰值收盘价",
            "peak_return": "起爆至峰值收益",
            "period_return_to_peak": "区间起点至峰值收益",
            "return_to_end_from_ignition": "起爆至区间末收益",
            "days_to_peak": "起爆到峰值天数",
            "max_drawdown_to_peak": "起爆到峰值最大回撤",
            "breakout_pct": "突破幅度",
            "amount_ratio": "成交额放大倍数",
            "lifeline_ma": "生命线均线",
            "breach_days_to_peak": "峰值前跌破生命线天数",
            "min_distance_to_lifeline": "相对生命线最小距离",
            "avg_distance_to_lifeline": "相对生命线平均距离",
            "lifeline_break_date": "生命线跌破日",
            "lifeline_status": "生命线状态",
        }
    )


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    raw = load_prices(Path(args.data), None, None)
    close = clean_matrix(pivot_prices(raw, "close"), args.max_abs_daily_return)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), args.max_abs_daily_return)
    amount = pivot_prices(raw, "amount").reindex_like(close)

    records: list[dict[str, object]] = []
    for symbol in close.columns:
        records.extend(
            summarize_trends_for_symbol(
                symbol=str(symbol).zfill(6),
                close=close[symbol],
                high=high[symbol],
                amount=amount[symbol],
                start=start,
                end=end,
                min_trend_return=args.min_trend_return,
                ignition_lookback_days=args.ignition_lookback_days,
                breakout_window=args.breakout_window,
                amount_window=args.amount_window,
                amount_multiplier=args.amount_multiplier,
                breakout_buffer=args.breakout_buffer,
                lifeline_breach_buffer=args.lifeline_breach_buffer,
                lifeline_max_breach_days=args.lifeline_max_breach_days,
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trends = pd.DataFrame(records)
    if not trends.empty:
        trends = trends.sort_values(["peak_return", "amount_ratio"], ascending=[False, False])
    trends.to_csv(output_dir / "trend_ignition_lifelines.csv", index=False, encoding="utf-8-sig")
    cn = to_chinese_columns(trends)
    cn.to_csv(output_dir / "trend_ignition_lifelines_cn.csv", index=False, encoding="utf-8-sig")
    top = cn.head(args.top_n)
    top.to_csv(output_dir / "top_trends_cn.csv", index=False, encoding="utf-8-sig")

    summary = {
        "start": args.start,
        "end": args.end,
        "min_trend_return": args.min_trend_return,
        "trend_count": int(len(trends)),
        "symbol_count": int(trends["symbol"].nunique()) if not trends.empty else 0,
        "median_peak_return": float(trends["peak_return"].median()) if not trends.empty else None,
        "median_days_to_peak": float(trends["days_to_peak"].median()) if not trends.empty else None,
        "lifeline_distribution": trends["lifeline_ma"].value_counts(dropna=False).to_dict()
        if not trends.empty
        else {},
        "broken_lifeline_ratio": float(trends["lifeline_status"].eq("broken").mean()) if not trends.empty else None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Trend ignition/lifeline outputs saved to: {output_dir}")
    if not top.empty:
        print(top.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
