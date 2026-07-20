"""Markdown report generation for completed backtest runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ._compat import read_text


def resolve_run_dir(output_dir: Path, run_id: str) -> Path:
    output_dir = output_dir.resolve()
    if run_id != "latest":
        run_dir = output_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run not found: {run_dir}")
        return run_dir
    runs = [path for path in output_dir.iterdir() if path.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No backtest runs found under {output_dir}")
    return max(runs, key=lambda path: path.stat().st_mtime)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _monthly_table(monthly: pd.DataFrame) -> str:
    if monthly.empty:
        return "No monthly returns available."
    rows = ["| date | monthly_return |", "| --- | ---: |"]
    for item in monthly.tail(12).itertuples(index=False):
        rows.append(f"| {item.date} | {_pct(float(item.monthly_return))} |")
    return "\n".join(rows)


def _risk_metric_rows(metrics: dict) -> str:
    rows = []
    if "average_position_exposure" in metrics:
        rows.append(f"| Average position exposure | {_pct(metrics.get('average_position_exposure', 0))} |")
    if "average_risk_exposure" in metrics:
        rows.append(f"| Average risk exposure | {_pct(metrics.get('average_risk_exposure', 0))} |")
    if "risk_off_day_ratio" in metrics:
        rows.append(f"| Risk-off days | {_pct(metrics.get('risk_off_day_ratio', 0))} |")
    if "risk_reduced_day_ratio" in metrics:
        rows.append(f"| Risk-reduced days | {_pct(metrics.get('risk_reduced_day_ratio', 0))} |")
    if "cooldown_event_count" in metrics:
        rows.append(f"| Cooldown events | {int(metrics.get('cooldown_event_count', 0))} |")
    if "average_cooldown_count" in metrics:
        rows.append(f"| Average cooldown count | {metrics.get('average_cooldown_count', 0):.2f} |")
    if "cooldown_active_day_ratio" in metrics:
        rows.append(f"| Cooldown-active days | {_pct(metrics.get('cooldown_active_day_ratio', 0))} |")
    if "average_market_breadth_exposure" in metrics:
        rows.append(f"| Average breadth exposure | {_pct(metrics.get('average_market_breadth_exposure', 0))} |")
    if "market_breadth_reduced_day_ratio" in metrics:
        rows.append(f"| Breadth-reduced days | {_pct(metrics.get('market_breadth_reduced_day_ratio', 0))} |")
    if "average_rebalance_loss_guarded_count" in metrics:
        rows.append(f"| Average rebalance-loss guarded count | {metrics.get('average_rebalance_loss_guarded_count', 0):.2f} |")
    if "rebalance_loss_guarded_day_ratio" in metrics:
        rows.append(f"| Rebalance-loss guarded days | {_pct(metrics.get('rebalance_loss_guarded_day_ratio', 0))} |")
    return "\n".join(rows)


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_report(run_dir: Path) -> Path:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    metrics = json.loads(read_text(metrics_path))
    trades_path = run_dir / "trades.csv"
    trades = _read_csv_or_empty(trades_path)
    monthly_path = run_dir / "monthly_returns.csv"
    monthly = _read_csv_or_empty(monthly_path)

    recent_months = _monthly_table(monthly)
    risk_rows = _risk_metric_rows(metrics)

    trade_summary = "No trades generated."
    if not trades.empty:
        trade_summary = (
            f"Total trades: {len(trades)}; buys: {(trades['side'] == 'BUY').sum()}; "
            f"sells: {(trades['side'] == 'SELL').sum()}."
        )

    body = f"""# Backtest Report

Run directory: `{run_dir}`

## Key Metrics

| Metric | Value |
| --- | ---: |
| Initial cash | {metrics.get('initial_cash', 0):,.2f} |
| Final equity | {metrics.get('final_equity', 0):,.2f} |
| Total return | {_pct(metrics.get('total_return', 0))} |
| Benchmark return | {_pct(metrics.get('benchmark_return', 0))} |
| Excess return | {_pct(metrics.get('excess_return', 0))} |
| CAGR | {_pct(metrics.get('cagr', 0))} |
| Max drawdown | {_pct(metrics.get('max_drawdown', 0))} |
| Sharpe | {metrics.get('sharpe', 0):.3f} |
| Win rate | {_pct(metrics.get('win_rate', 0))} |
| Payoff ratio | {metrics.get('payoff_ratio', 0):.3f} |
| Profit factor | {metrics.get('profit_factor', 0):.3f} |
{risk_rows}

## Charts

![Equity curve](equity_curve.png)

![Drawdown](drawdown.png)

## Trades

{trade_summary}

Detailed trades are saved in `trades.csv`.

## Recent Monthly Returns

{recent_months}

## Research Boundary

This report is for research and review only. It does not place trades, connect to a broker, or provide investment advice.
"""
    report_path = run_dir / "report.md"
    report_path.write_text(body, encoding="utf-8")
    return report_path
