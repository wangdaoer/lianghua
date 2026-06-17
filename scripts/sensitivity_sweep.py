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


LIGHT_DD = {
    "drawdown_levels": ((0.20, 0.70), (0.30, 0.35), (0.40, 0.0)),
    "protection_drawdown": 0.40,
    "recovery_drawdown": 0.15,
}

GUARD_DD = {
    "drawdown_levels": ((0.10, 0.50), (0.20, 0.20), (0.30, 0.0)),
    "protection_drawdown": 0.30,
    "recovery_drawdown": 0.10,
}

NO_DD = {
    "drawdown_levels": ((0.99, 1.0),),
    "protection_drawdown": 0.99,
    "recovery_drawdown": 0.15,
}


def _candidate_specs(mode: str) -> list[dict[str, Any]]:
    base = [
        {"name": "score65_warm120_light", "min_score": 65, "signal_min_history": 120, "risk": LIGHT_DD},
        {"name": "score70_warm120_light", "min_score": 70, "signal_min_history": 120, "risk": LIGHT_DD},
        {"name": "score75_warm120_light", "min_score": 75, "signal_min_history": 120, "risk": LIGHT_DD},
        {"name": "score65_warm180_light", "min_score": 65, "signal_min_history": 180, "risk": LIGHT_DD},
        {"name": "score70_warm180_light", "min_score": 70, "signal_min_history": 180, "risk": LIGHT_DD},
        {"name": "score65_warm120_guard", "min_score": 65, "signal_min_history": 120, "risk": GUARD_DD},
        {"name": "score65_warm120_no_dd", "min_score": 65, "signal_min_history": 120, "risk": NO_DD},
    ]
    if mode == "quick":
        return base[:4]
    return base


def _variant_config(config: LabConfig, spec: dict[str, Any]) -> LabConfig:
    strategy = replace(
        config.strategy,
        min_score=float(spec["min_score"]),
        signal_min_history=int(spec["signal_min_history"]),
        prefer_existing_positions=True,
    )
    risk_overrides = dict(spec["risk"])
    risk = replace(config.risk, **risk_overrides)
    return replace(config, strategy=strategy, risk=risk)


def _format_pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return ""


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    summary = pd.DataFrame(rows)
    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    table = [
        "| candidate | return | max_dd | sharpe | trades | win_rate | profit_factor | avg_exposure | run_dir |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        table.append(
            "| {candidate} | {total_return} | {max_drawdown} | {sharpe:.3f} | {trade_count} | "
            "{win_rate} | {profit_factor:.3f} | {average_position_exposure} | `{run_dir}` |".format(
                candidate=row["candidate"],
                total_return=_format_pct(row.get("total_return")),
                max_drawdown=_format_pct(row.get("max_drawdown")),
                sharpe=float(row.get("sharpe", 0.0) or 0.0),
                trade_count=int(row.get("trade_count", 0) or 0),
                win_rate=_format_pct(row.get("win_rate")),
                profit_factor=float(row.get("profit_factor", 0.0) or 0.0),
                average_position_exposure=_format_pct(row.get("average_position_exposure")),
                run_dir=row["run_dir"],
            )
        )

    best = summary.sort_values(["sharpe", "total_return"], ascending=False).head(1)
    best_name = str(best.iloc[0]["candidate"]) if not best.empty else ""
    body = f"""# Strategy Sensitivity Sweep

Generated at: `{datetime.now().isoformat(timespec="seconds")}`

Best by Sharpe then return: `{best_name}`

This is a research-only parameter sensitivity check. It does not connect to brokers, place orders, or provide investment advice.

## Results

{chr(10).join(table)}
"""
    md_path.write_text(body, encoding="utf-8")
    return csv_path, md_path


def run_sweep(config_path: Path, output_dir: Path, run_prefix: str, mode: str) -> pd.DataFrame:
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = load_universe_history(config, allow_fetch=False)
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(_candidate_specs(mode), start=1):
        variant = _variant_config(config, spec)
        run_id = f"{run_prefix}_{index:02d}_{spec['name']}"
        print(f"[{index}] running {spec['name']} -> {run_id}", flush=True)
        result = run_backtest(variant, histories=histories, run_id=run_id, write_outputs=True)
        report_path = write_report(result.run_dir)
        metrics = result.metrics
        row = {
            "candidate": spec["name"],
            "run_id": run_id,
            "run_dir": str(result.run_dir),
            "report_path": str(report_path),
            "min_score": spec["min_score"],
            "signal_min_history": spec["signal_min_history"],
            "risk_profile": json.dumps(spec["risk"], ensure_ascii=False),
            **metrics,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / "summary.partial.csv", index=False, encoding="utf-8-sig")
        print(
            f"[{index}] done return={metrics['total_return'] * 100:.2f}% "
            f"max_dd={metrics['max_drawdown'] * 100:.2f}% sharpe={metrics['sharpe']:.3f}",
            flush=True,
        )
    csv_path, md_path = _write_summary(output_dir, rows)
    summary = pd.read_csv(csv_path)
    summary.attrs["summary_csv"] = str(csv_path)
    summary.attrs["summary_md"] = str(md_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small A-share strategy parameter sensitivity sweep.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/sensitivity/latest"), type=Path)
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = args.run_prefix or f"sensitivity_{stamp}"
    summary = run_sweep(args.config, args.output_dir, run_prefix, args.mode)
    print(f"Sensitivity completed: {args.output_dir}")
    print(f"Summary CSV: {summary.attrs.get('summary_csv')}")
    print(f"Summary report: {summary.attrs.get('summary_md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
