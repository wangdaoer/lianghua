"""Walk-forward style validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .backtest import run_backtest
from .config import LabConfig
from .data import load_universe_history
from .report import write_report


@dataclass(frozen=True)
class ValidationSplit:
    name: str
    start_date: str
    end_date: str | None


DEFAULT_SPLITS = (
    ValidationSplit("train_2018_2021", "20180101", "20211231"),
    ValidationSplit("validation_2022_2023", "20220101", "20231231"),
    ValidationSplit("out_of_sample_2024_latest", "20240101", None),
)


def _window_config(config: LabConfig, split: ValidationSplit) -> LabConfig:
    return replace(
        config,
        data=replace(config.data, start_date=split.start_date, end_date=split.end_date),
    )


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _number(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _actual_date(frame: pd.DataFrame, fn: str) -> str:
    if frame.empty or "date" not in frame:
        return ""
    value = getattr(frame["date"], fn)()
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _write_summary(validation_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    summary = pd.DataFrame(rows)
    csv_path = validation_dir / "summary.csv"
    md_path = validation_dir / "summary.md"
    summary.to_csv(csv_path, index=False, encoding="utf-8")

    table_rows = [
        "| split | actual_start | actual_end | total_return | max_drawdown | sharpe | trades | avg_risk | risk_off | run_dir |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        table_rows.append(
            "| {split} | {actual_start} | {actual_end} | {total_return} | {max_drawdown} | "
            "{sharpe:.3f} | {trade_count} | {avg_risk} | {risk_off} | `{run_dir}` |".format(
                split=row["split"],
                actual_start=row["actual_start"],
                actual_end=row["actual_end"],
                total_return=_pct(row["total_return"]),
                max_drawdown=_pct(row["max_drawdown"]),
                sharpe=float(row.get("sharpe", 0.0)),
                trade_count=int(row.get("trade_count", 0)),
                avg_risk=_pct(row.get("average_risk_exposure")),
                risk_off=_pct(row.get("risk_off_day_ratio")),
                run_dir=row["run_dir"],
            )
        )

    body = f"""# Validation Summary

Validation directory: `{validation_dir}`

This validation re-runs the same locked configuration over separate date windows.
It is a research stability check only; it does not connect to brokers or provide investment advice.

## Split Results

{chr(10).join(table_rows)}

## Notes

- Each split starts with the configured initial cash and only uses data inside that date window.
- Multi-factor indicators need warm-up history, so early days in each split may have no trades until enough rows exist.
- Detailed equity curves, trades, risk audit files, and reports are saved in each `run_dir`.
"""
    md_path.write_text(body, encoding="utf-8")
    return csv_path, md_path


def run_validation(
    config: LabConfig,
    splits: tuple[ValidationSplit, ...] = DEFAULT_SPLITS,
    output_dir: Path | None = None,
    run_id_prefix: str | None = None,
    allow_fetch: bool = False,
) -> tuple[Path, pd.DataFrame]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = output_dir or (config.project_root / "outputs" / "validation")
    validation_name = run_id_prefix or f"{stamp}_{config.project.name}"
    validation_dir = root / validation_name
    validation_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for split in splits:
        split_config = _window_config(config, split)
        histories = load_universe_history(split_config, allow_fetch=allow_fetch)
        run_id = f"validation_{stamp}_{split.name}"
        result = run_backtest(split_config, histories=histories, run_id=run_id, write_outputs=True)
        report_path = write_report(result.run_dir)
        metrics = result.metrics
        rows.append(
            {
                "split": split.name,
                "requested_start": split.start_date,
                "requested_end": split.end_date or "latest",
                "actual_start": _actual_date(result.equity, "min"),
                "actual_end": _actual_date(result.equity, "max"),
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "report_path": str(report_path),
                "initial_cash": metrics.get("initial_cash"),
                "final_equity": metrics.get("final_equity"),
                "total_return": metrics.get("total_return"),
                "benchmark_return": metrics.get("benchmark_return"),
                "excess_return": metrics.get("excess_return"),
                "cagr": metrics.get("cagr"),
                "max_drawdown": metrics.get("max_drawdown"),
                "sharpe": metrics.get("sharpe"),
                "trade_count": metrics.get("trade_count"),
                "win_rate": metrics.get("win_rate"),
                "payoff_ratio": metrics.get("payoff_ratio"),
                "profit_factor": metrics.get("profit_factor"),
                "average_position_exposure": metrics.get("average_position_exposure"),
                "average_risk_exposure": metrics.get("average_risk_exposure"),
                "risk_off_day_ratio": metrics.get("risk_off_day_ratio"),
                "risk_reduced_day_ratio": metrics.get("risk_reduced_day_ratio"),
            }
        )

    csv_path, md_path = _write_summary(validation_dir, rows)
    summary = pd.read_csv(csv_path)
    summary.attrs["summary_csv"] = str(csv_path)
    summary.attrs["summary_md"] = str(md_path)
    return validation_dir, summary
