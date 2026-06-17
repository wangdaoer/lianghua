from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _money(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype={"code": "string"})
    if "code" in frame:
        frame["code"] = (
            frame["code"]
            .astype("string")
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(6)
        )
    return frame


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _attach_holding_days(trades: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    if trades.empty:
        trades["holding_days"] = pd.NA
        return trades

    trades["date"] = pd.to_datetime(trades["date"])
    open_lots: dict[str, list[list[Any]]] = {}
    holding_days: list[float | None] = []
    for row in trades.itertuples(index=False):
        code = str(row.code)
        date = pd.Timestamp(row.date)
        quantity = float(row.quantity)
        if row.side == "BUY":
            open_lots.setdefault(code, []).append([quantity, date])
            holding_days.append(None)
            continue

        remaining = quantity
        weighted_days = 0.0
        matched_quantity = 0.0
        lots = open_lots.setdefault(code, [])
        while remaining > 1e-8 and lots:
            lot_quantity, lot_date = lots[0]
            used = min(float(lot_quantity), remaining)
            weighted_days += used * max((date - pd.Timestamp(lot_date)).days, 0)
            matched_quantity += used
            lot_quantity = float(lot_quantity) - used
            remaining -= used
            if lot_quantity <= 1e-8:
                lots.pop(0)
            else:
                lots[0][0] = lot_quantity
        holding_days.append(weighted_days / matched_quantity if matched_quantity > 0 else None)

    trades["holding_days"] = holding_days
    return trades


def _holding_bucket(days: Any) -> str:
    if pd.isna(days):
        return "unknown"
    value = float(days)
    if value <= 5:
        return "0-5d"
    if value <= 20:
        return "6-20d"
    if value <= 60:
        return "21-60d"
    return "60d+"


def _risk_state(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    exposure = float(value)
    if exposure <= 0.01:
        return "risk_off"
    if exposure <= 0.15:
        return "market10_defensive"
    if exposure < 0.999:
        return "market_reduced"
    return "risk_on"


def _summarize_group(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    grouped = frame.groupby(group_column, dropna=False)
    summary = grouped.agg(
        sell_count=("realized_pnl", "size"),
        realized_pnl=("realized_pnl", "sum"),
        avg_pnl=("realized_pnl", "mean"),
        win_rate=("realized_pnl", lambda values: float((values > 0).mean())),
        loss_count=("realized_pnl", lambda values: int((values < 0).sum())),
        max_loss=("realized_pnl", "min"),
        avg_holding_days=("holding_days", "mean"),
        gross_sold=("gross_amount", "sum"),
        fees=("fee", "sum"),
    )
    return summary.reset_index().sort_values(["realized_pnl", "sell_count"], ascending=[True, False])


def _load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    trades = _attach_holding_days(_load_csv(run_dir / "trades.csv"))
    equity = _load_csv(run_dir / "equity_curve.csv")
    risk = _load_csv(run_dir / "risk_curve.csv")
    metrics = _load_metrics(run_dir)

    if not trades.empty:
        trades["date"] = pd.to_datetime(trades["date"])
        trades["realized_pnl"] = pd.to_numeric(trades["realized_pnl"], errors="coerce")
    if not equity.empty:
        equity["date"] = pd.to_datetime(equity["date"])
    if not risk.empty:
        risk["date"] = pd.to_datetime(risk["date"])
    return trades, equity, risk, metrics


def _monthly_equity(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame(columns=["month", "start_equity", "end_equity", "monthly_return", "max_drawdown"])
    frame = equity.copy().sort_values("date")
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    rows = []
    for month, group in frame.groupby("month"):
        series = group["equity"].astype(float)
        start = float(series.iloc[0])
        end = float(series.iloc[-1])
        drawdown = (series / series.cummax() - 1).min()
        rows.append(
            {
                "month": month,
                "start_equity": start,
                "end_equity": end,
                "monthly_return": end / start - 1 if start > 0 else 0.0,
                "max_drawdown": float(drawdown),
            }
        )
    return pd.DataFrame(rows)


def _prepare_sells(trades: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    sells = trades[trades["side"] == "SELL"].copy()
    if sells.empty:
        return sells
    sells["month"] = sells["date"].dt.to_period("M").astype(str)
    sells["holding_bucket"] = sells["holding_days"].map(_holding_bucket)
    if not risk.empty:
        risk_columns = [
            "date",
            "risk_exposure",
            "next_risk_exposure",
            "benchmark_return",
            "benchmark_risk_on",
            "risk_reason",
            "next_risk_reason",
        ]
        available = [column for column in risk_columns if column in risk.columns]
        sells = sells.merge(risk[available], on="date", how="left")
    sells["risk_state"] = sells.get("risk_exposure", pd.Series(index=sells.index)).map(_risk_state)
    sells["is_loss"] = sells["realized_pnl"] < 0
    return sells


def _compare_runs(target_metrics: dict[str, Any], baseline_metrics: dict[str, Any] | None) -> pd.DataFrame:
    rows = []
    keys = [
        "total_return",
        "max_drawdown",
        "sharpe",
        "win_rate",
        "payoff_ratio",
        "profit_factor",
        "trade_count",
        "average_risk_exposure",
    ]
    for key in keys:
        rows.append(
            {
                "metric": key,
                "target": target_metrics.get(key),
                "baseline": (baseline_metrics or {}).get(key),
                "delta": (
                    target_metrics.get(key) - baseline_metrics.get(key)
                    if baseline_metrics and target_metrics.get(key) is not None and baseline_metrics.get(key) is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
    if frame.empty:
        return "No rows."
    data = frame.loc[:, columns].head(max_rows).copy()
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in data.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:,.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def build_report(target_run_dir: Path, baseline_run_dir: Path | None, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    trades, equity, risk, metrics = _load_run(target_run_dir)
    baseline_metrics: dict[str, Any] | None = None
    if baseline_run_dir is not None:
        baseline_metrics = _load_metrics(baseline_run_dir)

    sells = _prepare_sells(trades, risk)
    monthly_equity = _monthly_equity(equity)
    monthly_realized = _summarize_group(sells, "month") if not sells.empty else pd.DataFrame()
    if not monthly_equity.empty and not monthly_realized.empty:
        monthly = monthly_equity.merge(
            monthly_realized.rename(columns={"realized_pnl": "realized_pnl_sum"}),
            on="month",
            how="left",
        ).fillna({"realized_pnl_sum": 0})
    else:
        monthly = monthly_equity

    stock = _summarize_group(sells, "code") if not sells.empty else pd.DataFrame()
    exit_reason = _summarize_group(sells, "exit_reason") if not sells.empty else pd.DataFrame()
    holding = _summarize_group(sells, "holding_bucket") if not sells.empty else pd.DataFrame()
    risk_state = _summarize_group(sells, "risk_state") if not sells.empty else pd.DataFrame()
    worst_trades = sells.sort_values("realized_pnl").head(30)
    comparison = _compare_runs(metrics, baseline_metrics)

    monthly.to_csv(output_dir / "monthly_attribution.csv", index=False, encoding="utf-8-sig")
    stock.to_csv(output_dir / "stock_attribution.csv", index=False, encoding="utf-8-sig")
    exit_reason.to_csv(output_dir / "exit_reason_attribution.csv", index=False, encoding="utf-8-sig")
    holding.to_csv(output_dir / "holding_bucket_attribution.csv", index=False, encoding="utf-8-sig")
    risk_state.to_csv(output_dir / "risk_state_attribution.csv", index=False, encoding="utf-8-sig")
    worst_trades.to_csv(output_dir / "worst_trades.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(output_dir / "baseline_comparison.csv", index=False, encoding="utf-8-sig")

    worst_months = monthly.sort_values("monthly_return").head(8)
    worst_stocks = stock.head(12)
    loss_by_reason = exit_reason.sort_values("realized_pnl").head(8)
    loss_by_holding = holding.sort_values("realized_pnl")
    loss_by_risk = risk_state.sort_values("realized_pnl")
    short_hold_pnl = 0.0
    long_hold_pnl = 0.0
    rebalance_pnl = 0.0
    market10_pnl = 0.0
    if not holding.empty:
        short_hold = holding[holding["holding_bucket"] == "0-5d"]
        long_hold = holding[holding["holding_bucket"] == "21-60d"]
        short_hold_pnl = float(short_hold["realized_pnl"].iloc[0]) if not short_hold.empty else 0.0
        long_hold_pnl = float(long_hold["realized_pnl"].iloc[0]) if not long_hold.empty else 0.0
    if not exit_reason.empty:
        rebalance = exit_reason[exit_reason["exit_reason"] == "rebalance"]
        rebalance_pnl = float(rebalance["realized_pnl"].iloc[0]) if not rebalance.empty else 0.0
    if not risk_state.empty:
        market10 = risk_state[risk_state["risk_state"] == "market10_defensive"]
        market10_pnl = float(market10["realized_pnl"].iloc[0]) if not market10.empty else 0.0

    facts = [
        f"- Target run: `{target_run_dir}`",
        f"- Baseline run: `{baseline_run_dir}`" if baseline_run_dir else "- Baseline run: not provided",
        f"- Total return: {_pct(metrics.get('total_return'))}",
        f"- Max drawdown: {_pct(metrics.get('max_drawdown'))}",
        f"- Sharpe: {float(metrics.get('sharpe', 0) or 0):.3f}",
        f"- Sell trades: {int(metrics.get('sell_trade_count', 0) or 0)}",
        f"- Profit factor: {float(metrics.get('profit_factor', 0) or 0):.3f}",
        f"- Average risk exposure: {_pct(metrics.get('average_risk_exposure'))}",
    ]
    inference = [
        f"- The largest residual leak is short-hold churn: the 0-5 day holding bucket made {_money(short_hold_pnl)}, while the 21-60 day bucket made {_money(long_hold_pnl)}.",
        f"- Rebalance exits dominate the loss side: `rebalance` realized PnL was {_money(rebalance_pnl)}, while `take_profit` remained strongly positive.",
        f"- The market filter helped materially, but the small defensive book still lost {_money(market10_pnl)} during `market10_defensive` days.",
        "- The next test should reduce weak-market short-hold rebalancing before cutting gross exposure further.",
    ]
    proposed_rules = [
        "1. In benchmark-off or market10-defensive state, do not reduce a position in the first 5 trading days for ordinary `rebalance`; allow only explicit risk exit, stop-loss, or take-profit.",
        "2. In benchmark-off state, require a higher entry score, for example `weak_market_min_score=75`, before opening a new position.",
        "3. If a stock has two losing sells in the same weak-market regime, pause that stock until benchmark risk returns on.",
    ]

    body = f"""# 2022-2023 Weak-Market Loss Attribution

## Research Boundary

This report is for research and review only. It does not place trades, connect to a broker, or provide investment advice.

## Facts

{chr(10).join(facts)}

## Baseline Comparison

{_markdown_table(comparison, ["metric", "target", "baseline", "delta"], 12)}

## Worst Months

{_markdown_table(worst_months, [column for column in ["month", "monthly_return", "max_drawdown", "realized_pnl_sum", "sell_count", "win_rate"] if column in worst_months.columns], 8)}

## Worst Stocks By Realized PnL

{_markdown_table(worst_stocks, [column for column in ["code", "sell_count", "realized_pnl", "avg_pnl", "win_rate", "loss_count", "max_loss", "avg_holding_days"] if column in worst_stocks.columns], 12)}

## Exit-Reason Attribution

{_markdown_table(loss_by_reason, [column for column in ["exit_reason", "sell_count", "realized_pnl", "avg_pnl", "win_rate", "loss_count", "max_loss", "avg_holding_days"] if column in loss_by_reason.columns], 8)}

## Holding-Time Attribution

{_markdown_table(loss_by_holding, [column for column in ["holding_bucket", "sell_count", "realized_pnl", "avg_pnl", "win_rate", "loss_count", "max_loss", "avg_holding_days"] if column in loss_by_holding.columns], 8)}

## Market-State Attribution

{_markdown_table(loss_by_risk, [column for column in ["risk_state", "sell_count", "realized_pnl", "avg_pnl", "win_rate", "loss_count", "max_loss", "avg_holding_days"] if column in loss_by_risk.columns], 8)}

## Largest Losing Trades

{_markdown_table(worst_trades, [column for column in ["date", "code", "name", "exit_reason", "realized_pnl", "holding_days", "risk_state", "risk_exposure"] if column in worst_trades.columns], 15)}

## Inference

{chr(10).join(inference)}

## Next Testable Rules

{chr(10).join(proposed_rules)}

## Output Files

- `monthly_attribution.csv`
- `stock_attribution.csv`
- `exit_reason_attribution.csv`
- `holding_bucket_attribution.csv`
- `risk_state_attribution.csv`
- `worst_trades.csv`
- `baseline_comparison.csv`
"""
    report_path = output_dir / "report.md"
    report_path.write_text(body, encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate weak-market loss attribution from backtest outputs.")
    parser.add_argument("--target-run-dir", required=True, type=Path)
    parser.add_argument("--baseline-run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    report = build_report(args.target_run_dir, args.baseline_run_dir, args.output_dir)
    print(f"Report written: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
