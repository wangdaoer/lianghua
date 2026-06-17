"""Drawdown attribution for completed backtest runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrawdownAttributionResult:
    run_dir: Path
    output_dir: Path
    periods: pd.DataFrame
    attribution: pd.DataFrame
    losing_symbols: pd.DataFrame
    exit_reasons: pd.DataFrame
    metrics: dict[str, Any]
    report_path: Path


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def load_equity_drawdown(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "equity_curve.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing equity curve: {path}")
    equity = pd.read_csv(path)
    missing = {"date", "equity"} - set(equity.columns)
    if missing:
        raise ValueError(f"Equity curve missing columns: {sorted(missing)}")
    data = equity.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["equity"] = pd.to_numeric(data["equity"], errors="coerce")
    data = data.dropna(subset=["date", "equity"]).sort_values("date").reset_index(drop=True)
    data["running_peak"] = data["equity"].cummax()
    data["drawdown"] = data["equity"] / data["running_peak"].replace(0, np.nan) - 1.0
    data["drawdown"] = data["drawdown"].fillna(0.0)
    return data


def find_drawdown_periods(equity: pd.DataFrame, min_depth: float = 0.05) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame(
            columns=[
                "episode",
                "peak_date",
                "start_date",
                "trough_date",
                "recovery_date",
                "recovered",
                "peak_equity",
                "trough_equity",
                "drawdown",
                "equity_loss",
                "trading_days",
                "calendar_days",
                "recovery_trading_days",
            ]
        )

    periods: list[dict[str, Any]] = []
    in_drawdown = False
    peak_date = pd.Timestamp(equity.iloc[0]["date"])
    peak_equity = float(equity.iloc[0]["running_peak"])
    start_date = peak_date
    trough_date = peak_date
    trough_equity = peak_equity
    trough_drawdown = 0.0
    start_index = 0
    peak_index = 0
    trough_index = 0

    for idx, row in enumerate(equity.itertuples(index=False)):
        date = pd.Timestamp(row.date)
        value = float(row.equity)
        running_peak = float(row.running_peak)
        drawdown = float(row.drawdown)

        if not in_drawdown and drawdown < 0:
            in_drawdown = True
            start_date = date
            start_index = idx
            peak_rows = equity.iloc[: idx + 1]
            peak_index = int(peak_rows["equity"].idxmax())
            peak_date = pd.Timestamp(equity.loc[peak_index, "date"])
            peak_equity = float(equity.loc[peak_index, "equity"])
            trough_date = date
            trough_equity = value
            trough_drawdown = drawdown
            trough_index = idx
        elif not in_drawdown and running_peak >= peak_equity:
            peak_date = date
            peak_equity = running_peak
            peak_index = idx

        if in_drawdown and drawdown < trough_drawdown:
            trough_date = date
            trough_equity = value
            trough_drawdown = drawdown
            trough_index = idx

        if in_drawdown and drawdown >= 0:
            if abs(trough_drawdown) >= min_depth:
                periods.append(
                    {
                        "episode": len(periods) + 1,
                        "peak_date": peak_date,
                        "start_date": start_date,
                        "trough_date": trough_date,
                        "recovery_date": date,
                        "recovered": True,
                        "peak_equity": peak_equity,
                        "trough_equity": trough_equity,
                        "drawdown": trough_drawdown,
                        "equity_loss": trough_equity - peak_equity,
                        "trading_days": trough_index - peak_index + 1,
                        "calendar_days": (trough_date - peak_date).days,
                        "recovery_trading_days": idx - trough_index,
                    }
                )
            in_drawdown = False
            peak_date = date
            peak_equity = value
            peak_index = idx

    if in_drawdown and abs(trough_drawdown) >= min_depth:
        last_date = pd.Timestamp(equity.iloc[-1]["date"])
        periods.append(
            {
                "episode": len(periods) + 1,
                "peak_date": peak_date,
                "start_date": start_date,
                "trough_date": trough_date,
                "recovery_date": pd.NaT,
                "recovered": False,
                "peak_equity": peak_equity,
                "trough_equity": trough_equity,
                "drawdown": trough_drawdown,
                "equity_loss": trough_equity - peak_equity,
                "trading_days": trough_index - peak_index + 1,
                "calendar_days": (trough_date - peak_date).days,
                "recovery_trading_days": len(equity) - trough_index - 1,
                "last_date": last_date,
            }
        )

    result = pd.DataFrame(periods)
    if result.empty:
        return result
    return result.sort_values("drawdown", ascending=True).reset_index(drop=True)


def _benchmark_period_return(benchmark: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if benchmark.empty or "benchmark_equity" not in benchmark:
        return 0.0
    data = benchmark.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["benchmark_equity"] = pd.to_numeric(data["benchmark_equity"], errors="coerce")
    data = data.dropna(subset=["date", "benchmark_equity"]).sort_values("date")
    window = data[(data["date"] >= start) & (data["date"] <= end)]
    if len(window) < 2:
        return 0.0
    return float(window["benchmark_equity"].iloc[-1] / window["benchmark_equity"].iloc[0] - 1.0)


def _period_risk_summary(risk_curve: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    if risk_curve.empty:
        return {
            "average_risk_exposure": 1.0,
            "risk_off_day_ratio": 0.0,
            "risk_reduced_day_ratio": 0.0,
            "dominant_risk_reason": "",
        }
    data = risk_curve.copy()
    data["date"] = pd.to_datetime(data["date"])
    window = data[(data["date"] >= start) & (data["date"] <= end)]
    if window.empty:
        return {
            "average_risk_exposure": 1.0,
            "risk_off_day_ratio": 0.0,
            "risk_reduced_day_ratio": 0.0,
            "dominant_risk_reason": "",
        }
    exposure = pd.to_numeric(window.get("risk_exposure", 1.0), errors="coerce").fillna(1.0)
    reason = ""
    if "risk_reason" in window:
        counts = window["risk_reason"].fillna("").astype(str).value_counts()
        reason = str(counts.index[0]) if not counts.empty else ""
    return {
        "average_risk_exposure": float(exposure.mean()),
        "risk_off_day_ratio": float((exposure <= 0.01).mean()),
        "risk_reduced_day_ratio": float((exposure < 0.999).mean()),
        "dominant_risk_reason": reason,
    }


def _equity_period_summary(equity: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    window = equity[(equity["date"] >= start) & (equity["date"] <= end)]
    if window.empty:
        return {
            "average_active_positions": 0.0,
            "max_active_positions": 0.0,
            "average_position_exposure": 0.0,
        }
    active = pd.to_numeric(window.get("active_positions", 0), errors="coerce").fillna(0.0)
    if "positions_value" in window:
        exposure = pd.to_numeric(window["positions_value"], errors="coerce") / pd.to_numeric(
            window["equity"],
            errors="coerce",
        ).replace(0, np.nan)
        exposure = exposure.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        exposure = pd.Series(0.0, index=window.index)
    return {
        "average_active_positions": float(active.mean()),
        "max_active_positions": float(active.max()),
        "average_position_exposure": float(exposure.mean()),
    }


def attribute_drawdowns(
    run_dir: Path,
    min_depth: float = 0.05,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    equity = load_equity_drawdown(run_dir)
    periods = find_drawdown_periods(equity, min_depth=min_depth)
    selected = periods.head(max(top_n, 1)).copy() if not periods.empty else periods.copy()
    trades = _read_csv_or_empty(run_dir / "trades.csv")
    risk_curve = _read_csv_or_empty(run_dir / "risk_curve.csv")
    benchmark = _read_csv_or_empty(run_dir / "benchmark.csv")

    if not trades.empty and "date" in trades:
        trades = trades.copy()
        trades["date"] = pd.to_datetime(trades["date"])
        trades["realized_pnl"] = pd.to_numeric(trades.get("realized_pnl", 0.0), errors="coerce")
        trades["fee"] = pd.to_numeric(trades.get("fee", 0.0), errors="coerce").fillna(0.0)
        trades["gross_amount"] = pd.to_numeric(trades.get("gross_amount", 0.0), errors="coerce").fillna(0.0)

    attribution_rows: list[dict[str, Any]] = []
    losing_rows: list[dict[str, Any]] = []
    reason_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        peak_date = pd.Timestamp(row.peak_date)
        trough_date = pd.Timestamp(row.trough_date)
        period_trades = trades[(trades["date"] >= peak_date) & (trades["date"] <= trough_date)] if not trades.empty else pd.DataFrame()
        sells = period_trades[period_trades["side"] == "SELL"] if not period_trades.empty and "side" in period_trades else pd.DataFrame()
        buys = period_trades[period_trades["side"] == "BUY"] if not period_trades.empty and "side" in period_trades else pd.DataFrame()
        realized = pd.to_numeric(sells.get("realized_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        realized_loss = float(realized[realized < 0].sum()) if not realized.empty else 0.0
        realized_gain = float(realized[realized > 0].sum()) if not realized.empty else 0.0
        fees = float(period_trades["fee"].sum()) if not period_trades.empty and "fee" in period_trades else 0.0
        equity_loss = float(row.equity_loss)
        risk_summary = _period_risk_summary(risk_curve, peak_date, trough_date)
        equity_summary = _equity_period_summary(equity, peak_date, trough_date)
        attribution_rows.append(
            {
                "episode": int(row.episode),
                "peak_date": peak_date.date().isoformat(),
                "trough_date": trough_date.date().isoformat(),
                "recovery_date": "" if pd.isna(row.recovery_date) else pd.Timestamp(row.recovery_date).date().isoformat(),
                "recovered": bool(row.recovered),
                "drawdown": float(row.drawdown),
                "equity_loss": equity_loss,
                "benchmark_return_peak_to_trough": _benchmark_period_return(benchmark, peak_date, trough_date),
                "trade_count": int(len(period_trades)),
                "buy_count": int(len(buys)),
                "sell_count": int(len(sells)),
                "buy_gross_amount": float(buys["gross_amount"].sum()) if not buys.empty else 0.0,
                "sell_gross_amount": float(sells["gross_amount"].sum()) if not sells.empty else 0.0,
                "realized_pnl": float(realized.sum()) if not realized.empty else 0.0,
                "realized_loss": realized_loss,
                "realized_gain": realized_gain,
                "fees": fees,
                "realized_pnl_to_equity_loss_ratio": float(realized.sum() / equity_loss) if equity_loss else 0.0,
                **risk_summary,
                **equity_summary,
            }
        )

        if not sells.empty and {"code", "name", "realized_pnl"}.issubset(sells.columns):
            grouped = sells.groupby(["code", "name"], dropna=False).agg(
                realized_pnl=("realized_pnl", "sum"),
                sell_count=("side", "count"),
                fees=("fee", "sum"),
                gross_amount=("gross_amount", "sum"),
            ).reset_index()
            grouped = grouped.sort_values("realized_pnl", ascending=True).head(10)
            for item in grouped.itertuples(index=False):
                losing_rows.append(
                    {
                        "episode": int(row.episode),
                        "code": str(item.code).zfill(6),
                        "name": str(item.name),
                        "realized_pnl": float(item.realized_pnl),
                        "sell_count": int(item.sell_count),
                        "fees": float(item.fees),
                        "gross_amount": float(item.gross_amount),
                    }
                )
        if not sells.empty and "exit_reason" in sells:
            reason_grouped = sells.groupby(sells["exit_reason"].fillna("unknown")).agg(
                realized_pnl=("realized_pnl", "sum"),
                sell_count=("side", "count"),
            ).reset_index(names="exit_reason")
            for item in reason_grouped.sort_values("realized_pnl", ascending=True).itertuples(index=False):
                reason_rows.append(
                    {
                        "episode": int(row.episode),
                        "exit_reason": str(item.exit_reason),
                        "realized_pnl": float(item.realized_pnl),
                        "sell_count": int(item.sell_count),
                    }
                )

    attribution = pd.DataFrame(attribution_rows)
    losing_symbols = pd.DataFrame(losing_rows)
    exit_reasons = pd.DataFrame(reason_rows)
    metrics = {
        "drawdown_episode_count": int(len(periods)),
        "reported_episode_count": int(len(selected)),
        "max_drawdown": float(periods["drawdown"].min()) if not periods.empty else 0.0,
        "max_drawdown_episode": int(periods.iloc[0]["episode"]) if not periods.empty else 0,
        "min_depth": float(min_depth),
    }
    return periods, attribution, losing_symbols, exit_reasons, metrics


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 10) -> str:
    if frame.empty:
        return "No rows."
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for item in frame.head(max_rows).itertuples(index=False):
        values = []
        data = item._asdict()
        for column in columns:
            value = data.get(column, "")
            if isinstance(value, float):
                if "drawdown" in column or "return" in column or "ratio" in column or "exposure" in column:
                    values.append(_pct(value))
                else:
                    values.append(_num(value))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _write_report(result: DrawdownAttributionResult) -> Path:
    largest = result.attribution.iloc[0].to_dict() if not result.attribution.empty else {}
    body = f"""# Drawdown Attribution

Run directory: `{result.run_dir}`

This report attributes backtest drawdowns using local equity curves, trades, benchmark, and risk-control records. It is research analysis only and does not place trades or provide investment advice.

## Summary

| Metric | Value |
| --- | ---: |
| Drawdown episodes over threshold | {result.metrics.get("drawdown_episode_count", 0)} |
| Reported episodes | {result.metrics.get("reported_episode_count", 0)} |
| Minimum depth threshold | {_pct(result.metrics.get("min_depth", 0))} |
| Max drawdown | {_pct(result.metrics.get("max_drawdown", 0))} |

## Largest Episode

| Metric | Value |
| --- | ---: |
| Peak date | {largest.get("peak_date", "")} |
| Trough date | {largest.get("trough_date", "")} |
| Recovery date | {largest.get("recovery_date", "")} |
| Drawdown | {_pct(largest.get("drawdown", 0))} |
| Equity loss | {_num(largest.get("equity_loss", 0))} |
| Benchmark return peak-to-trough | {_pct(largest.get("benchmark_return_peak_to_trough", 0))} |
| Realized PnL in period | {_num(largest.get("realized_pnl", 0))} |
| Fees in period | {_num(largest.get("fees", 0))} |
| Average risk exposure | {_pct(largest.get("average_risk_exposure", 0))} |
| Average position exposure | {_pct(largest.get("average_position_exposure", 0))} |
| Average active positions | {_num(largest.get("average_active_positions", 0))} |

## Episode Attribution

{_markdown_table(result.attribution, ["episode", "peak_date", "trough_date", "drawdown", "equity_loss", "benchmark_return_peak_to_trough", "realized_pnl", "fees", "average_risk_exposure"], max_rows=10)}

## Top Losing Symbols

{_markdown_table(result.losing_symbols, ["episode", "code", "name", "realized_pnl", "sell_count", "fees"], max_rows=15)}

## Exit Reasons

{_markdown_table(result.exit_reasons, ["episode", "exit_reason", "realized_pnl", "sell_count"], max_rows=15)}

## Files

- `drawdown_periods.csv`: all drawdown episodes over the threshold.
- `drawdown_attribution.csv`: top episode attribution summary.
- `top_losing_symbols.csv`: realized PnL by symbol inside reported drawdowns.
- `exit_reason_attribution.csv`: realized PnL by exit reason inside reported drawdowns.
- `drawdown_attribution_metrics.json`: machine-readable summary.
"""
    report_path = result.output_dir / "drawdown_attribution_report.md"
    report_path.write_text(body, encoding="utf-8")
    return report_path


def run_drawdown_attribution(
    run_dir: Path,
    min_depth: float = 0.05,
    top_n: int = 5,
    output_dir: Path | None = None,
) -> DrawdownAttributionResult:
    run_dir = run_dir.resolve()
    output = output_dir.resolve() if output_dir is not None else run_dir / "attribution"
    output.mkdir(parents=True, exist_ok=True)
    periods, attribution, losing_symbols, exit_reasons, metrics = attribute_drawdowns(
        run_dir,
        min_depth=min_depth,
        top_n=top_n,
    )
    periods.to_csv(output / "drawdown_periods.csv", index=False, encoding="utf-8")
    attribution.to_csv(output / "drawdown_attribution.csv", index=False, encoding="utf-8")
    losing_symbols.to_csv(output / "top_losing_symbols.csv", index=False, encoding="utf-8")
    exit_reasons.to_csv(output / "exit_reason_attribution.csv", index=False, encoding="utf-8")
    (output / "drawdown_attribution_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    placeholder = output / "drawdown_attribution_report.md"
    result = DrawdownAttributionResult(
        run_dir=run_dir,
        output_dir=output,
        periods=periods,
        attribution=attribution,
        losing_symbols=losing_symbols,
        exit_reasons=exit_reasons,
        metrics=metrics,
        report_path=placeholder,
    )
    report_path = _write_report(result)
    return DrawdownAttributionResult(
        run_dir=result.run_dir,
        output_dir=result.output_dir,
        periods=result.periods,
        attribution=result.attribution,
        losing_symbols=result.losing_symbols,
        exit_reasons=result.exit_reasons,
        metrics=result.metrics,
        report_path=report_path,
    )
