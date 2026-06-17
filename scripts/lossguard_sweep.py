from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_etf_lab.backtest import run_backtest
from quant_etf_lab.config import LabConfig, load_config
from quant_etf_lab.data import load_universe_history
from quant_etf_lab.report import write_report


def _candidate_specs(mode: str) -> list[dict[str, Any]]:
    baseline = {"name": "baseline_no_lossguard", "guard_pct": None, "guard_days": 0}
    quick = [
        baseline,
        {"name": "lossguard3_d20", "guard_pct": 0.03, "guard_days": 20},
        {"name": "lossguard5_d10", "guard_pct": 0.05, "guard_days": 10},
        {"name": "lossguard5_d20", "guard_pct": 0.05, "guard_days": 20},
        {"name": "lossguard5_d40", "guard_pct": 0.05, "guard_days": 40},
        {"name": "lossguard7_d20", "guard_pct": 0.07, "guard_days": 20},
    ]
    if mode == "quick":
        return quick

    full = [baseline]
    for pct in (0.03, 0.05, 0.07, 0.10):
        for days in (10, 20, 40):
            full.append({"name": f"lossguard{int(pct * 100)}_d{days}", "guard_pct": pct, "guard_days": days})
    return full


def _variant_config(config: LabConfig, spec: dict[str, Any]) -> LabConfig:
    strategy = replace(
        config.strategy,
        rebalance_loss_guard_pct=spec["guard_pct"],
        rebalance_loss_guard_max_days=int(spec["guard_days"]),
    )
    return replace(config, strategy=strategy)


def _format_pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return ""


def _score_row(row: dict[str, Any]) -> float:
    total_return = float(row.get("total_return", 0.0) or 0.0)
    sharpe = float(row.get("sharpe", 0.0) or 0.0)
    max_drawdown = abs(float(row.get("max_drawdown", 0.0) or 0.0))
    drawdown_penalty = max(0.0, max_drawdown - 0.35) * 4.0
    return sharpe + 0.25 * total_return - drawdown_penalty


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["selection_score"] = summary.apply(lambda row: _score_row(row.to_dict()), axis=1)
        summary = summary.sort_values(["selection_score", "sharpe", "total_return"], ascending=False)
    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    table = [
        "| candidate | guard | return | max_dd | sharpe | pf | trades | guarded_days | avg_guarded | score | run_dir |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary.to_dict("records"):
        guard = "off" if pd.isna(row.get("guard_pct")) else f"{float(row['guard_pct']) * 100:.1f}%/{int(row['guard_days'])}d"
        table.append(
            "| {candidate} | {guard} | {total_return} | {max_drawdown} | {sharpe:.3f} | "
            "{profit_factor:.3f} | {trade_count} | {guarded_days} | {avg_guarded:.2f} | {score:.3f} | `{run_dir}` |".format(
                candidate=row["candidate"],
                guard=guard,
                total_return=_format_pct(row.get("total_return")),
                max_drawdown=_format_pct(row.get("max_drawdown")),
                sharpe=float(row.get("sharpe", 0.0) or 0.0),
                profit_factor=float(row.get("profit_factor", 0.0) or 0.0),
                trade_count=int(row.get("trade_count", 0) or 0),
                guarded_days=_format_pct(row.get("rebalance_loss_guarded_day_ratio")),
                avg_guarded=float(row.get("average_rebalance_loss_guarded_count", 0.0) or 0.0),
                score=float(row.get("selection_score", 0.0) or 0.0),
                run_dir=row["run_dir"],
            )
        )

    best_name = str(summary.iloc[0]["candidate"]) if not summary.empty else ""
    body = f"""# Rebalance Loss Guard Sweep

Generated at: `{datetime.now().isoformat(timespec="seconds")}`

Best by risk-adjusted score: `{best_name}`

This is a research-only parameter sensitivity check. It does not connect to brokers, place orders, or provide investment advice.

The score is `Sharpe + 0.25 * total_return - 4 * max(0, abs(max_drawdown) - 35%)`.

## Results

{chr(10).join(table)}
"""
    md_path.write_text(body, encoding="utf-8")
    return csv_path, md_path


def run_sweep(config_path: Path, output_dir: Path, run_prefix: str, mode: str) -> pd.DataFrame:
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = load_universe_history(config, allow_fetch=False, skip_missing=True)
    rows: list[dict[str, Any]] = []
    specs = _candidate_specs(mode)
    for index, spec in enumerate(specs, start=1):
        variant = _variant_config(config, spec)
        run_id = f"{run_prefix}_{index:02d}_{spec['name']}"
        print(f"[{index}/{len(specs)}] running {spec['name']} -> {run_id}", flush=True)
        result = run_backtest(variant, histories=histories, run_id=run_id, write_outputs=True, skip_missing=True)
        report_path = write_report(result.run_dir)
        metrics = result.metrics
        row = {
            "candidate": spec["name"],
            "run_id": run_id,
            "run_dir": str(result.run_dir),
            "report_path": str(report_path),
            "guard_pct": spec["guard_pct"],
            "guard_days": spec["guard_days"],
            "parameters": json.dumps(spec, ensure_ascii=False),
            **metrics,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / "summary.partial.csv", index=False, encoding="utf-8-sig")
        print(
            f"[{index}/{len(specs)}] done return={metrics['total_return'] * 100:.2f}% "
            f"max_dd={metrics['max_drawdown'] * 100:.2f}% sharpe={metrics['sharpe']:.3f}",
            flush=True,
        )
    csv_path, md_path = _write_summary(output_dir, rows)
    summary = pd.read_csv(csv_path)
    summary.attrs["summary_csv"] = str(csv_path)
    summary.attrs["summary_md"] = str(md_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a rebalance-loss guard sensitivity sweep.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/sensitivity/lossguard_latest"), type=Path)
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = args.run_prefix or f"lossguard_sweep_{stamp}"
    summary = run_sweep(args.config, args.output_dir, run_prefix, args.mode)
    print(f"Lossguard sweep completed: {args.output_dir}")
    print(f"Summary CSV: {summary.attrs.get('summary_csv')}")
    print(f"Summary report: {summary.attrs.get('summary_md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
