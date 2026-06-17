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
from quant_etf_lab.config import ETFSpec, LabConfig, load_config
from quant_etf_lab.data import load_cached_universe, load_universe_history, resolve_universe
from quant_etf_lab.report import write_report


def _candidate_specs(mode: str) -> list[dict[str, Any]]:
    baseline = {"name": "baseline_rsrs_off", "rsrs": None}
    quick = [
        baseline,
        {"name": "rsrs18_z120_t0_off35", "rsrs": {"zscore_window": 120, "threshold": 0.0, "off_exposure": 0.35}},
        {"name": "rsrs18_z120_t-05_off35", "rsrs": {"zscore_window": 120, "threshold": -0.5, "off_exposure": 0.35}},
        {"name": "rsrs18_z240_t-05_off35", "rsrs": {"zscore_window": 240, "threshold": -0.5, "off_exposure": 0.35}},
        {"name": "rsrs18_z240_t-07_off20", "rsrs": {"zscore_window": 240, "threshold": -0.7, "off_exposure": 0.20}},
    ]
    if mode == "quick":
        return quick

    full = [baseline]
    for zscore_window in (120, 240, 480):
        for threshold in (0.0, -0.5, -0.7, -1.0):
            for off_exposure in (0.10, 0.20, 0.35, 0.50):
                suffix = str(threshold).replace("-", "m").replace(".", "")
                full.append(
                    {
                        "name": f"rsrs18_z{zscore_window}_t{suffix}_off{int(off_exposure * 100)}",
                        "rsrs": {
                            "zscore_window": zscore_window,
                            "threshold": threshold,
                            "off_exposure": off_exposure,
                        },
                    }
                )
    return full


def _sample_config(config: LabConfig, max_symbols: int | None) -> LabConfig:
    if max_symbols is None or max_symbols <= 0:
        return config

    if config.universe_source is not None:
        try:
            frame = load_cached_universe(config, config.universe_source)
            instruments = tuple(
                ETFSpec(code=str(row.code).zfill(6), name=str(row.name), asset_type=str(row.asset_type).lower())
                for row in frame.head(max_symbols).itertuples(index=False)
            )
        except Exception:
            instruments = resolve_universe(config)[:max_symbols]
    else:
        instruments = resolve_universe(config)[:max_symbols]
    return replace(config, universe=instruments, universe_source=None)


def _variant_config(config: LabConfig, spec: dict[str, Any]) -> LabConfig:
    rsrs = spec["rsrs"]
    if rsrs is None:
        risk = replace(config.risk, benchmark_rsrs_enabled=False)
    else:
        risk = replace(
            config.risk,
            benchmark_rsrs_enabled=True,
            benchmark_rsrs_window=18,
            benchmark_rsrs_zscore_window=int(rsrs["zscore_window"]),
            benchmark_rsrs_threshold=float(rsrs["threshold"]),
            benchmark_rsrs_off_exposure=float(rsrs["off_exposure"]),
        )
    return replace(config, risk=risk)


def _score_row(row: dict[str, Any]) -> float:
    total_return = float(row.get("total_return", 0.0) or 0.0)
    sharpe = float(row.get("sharpe", 0.0) or 0.0)
    profit_factor = min(float(row.get("profit_factor", 0.0) or 0.0), 2.0)
    max_drawdown = abs(float(row.get("max_drawdown", 0.0) or 0.0))
    drawdown_penalty = max(0.0, max_drawdown - 0.35) * 4.0
    return sharpe + 0.20 * total_return + 0.10 * profit_factor - drawdown_penalty


def _format_pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return ""


def _write_summary(output_dir: Path, rows: list[dict[str, Any]], max_symbols: int | None) -> tuple[Path, Path]:
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["selection_score"] = summary.apply(lambda row: _score_row(row.to_dict()), axis=1)
        summary = summary.sort_values(["selection_score", "sharpe", "total_return"], ascending=False)
    csv_path = output_dir / "summary.csv"
    md_path = output_dir / "summary.md"
    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")

    table = [
        "| candidate | return | max_dd | sharpe | pf | risk_off | risk_reduced | avg_risk | score | run_dir |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary.to_dict("records"):
        table.append(
            "| {candidate} | {total_return} | {max_drawdown} | {sharpe:.3f} | {profit_factor:.3f} | "
            "{risk_off} | {risk_reduced} | {avg_risk} | {score:.3f} | `{run_dir}` |".format(
                candidate=row["candidate"],
                total_return=_format_pct(row.get("total_return")),
                max_drawdown=_format_pct(row.get("max_drawdown")),
                sharpe=float(row.get("sharpe", 0.0) or 0.0),
                profit_factor=float(row.get("profit_factor", 0.0) or 0.0),
                risk_off=_format_pct(row.get("risk_off_day_ratio")),
                risk_reduced=_format_pct(row.get("risk_reduced_day_ratio")),
                avg_risk=_format_pct(row.get("average_risk_exposure")),
                score=float(row.get("selection_score", 0.0) or 0.0),
                run_dir=row["run_dir"],
            )
        )

    best_name = str(summary.iloc[0]["candidate"]) if not summary.empty else ""
    body = f"""# RSRS Risk Sweep

Generated at: `{datetime.now().isoformat(timespec="seconds")}`

Best by quick research score: `{best_name}`

Universe sample size: `{max_symbols or "full"}`

This is a research-only benchmark risk-filter sweep. It does not connect to brokers, place orders, or provide investment advice.

The score is `Sharpe + 0.20 * total_return + 0.10 * min(profit_factor, 2) - 4 * max(0, abs(max_drawdown) - 35%)`.

## Results

{chr(10).join(table)}
"""
    md_path.write_text(body, encoding="utf-8")
    return csv_path, md_path


def run_sweep(config_path: Path, output_dir: Path, run_prefix: str, mode: str, max_symbols: int | None) -> pd.DataFrame:
    base_config = _sample_config(load_config(config_path), max_symbols)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = load_universe_history(base_config, allow_fetch=False, skip_missing=True)
    rows: list[dict[str, Any]] = []
    specs = _candidate_specs(mode)
    for index, spec in enumerate(specs, start=1):
        variant = _variant_config(base_config, spec)
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
            "parameters": json.dumps(spec, ensure_ascii=False),
            "history_count": len(histories),
            **metrics,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / "summary.partial.csv", index=False, encoding="utf-8-sig")
        print(
            f"[{index}/{len(specs)}] done return={metrics['total_return'] * 100:.2f}% "
            f"max_dd={metrics['max_drawdown'] * 100:.2f}% sharpe={metrics['sharpe']:.3f}",
            flush=True,
        )
    csv_path, md_path = _write_summary(output_dir, rows, max_symbols)
    summary = pd.read_csv(csv_path)
    summary.attrs["summary_csv"] = str(csv_path)
    summary.attrs["summary_md"] = str(md_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an RSRS benchmark risk-filter sensitivity sweep.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/sensitivity/rsrs_risk_latest"), type=Path)
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--max-symbols", type=int, default=800)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = args.run_prefix or f"rsrs_risk_sweep_{stamp}"
    summary = run_sweep(args.config, args.output_dir, run_prefix, args.mode, args.max_symbols)
    print(f"RSRS risk sweep completed: {args.output_dir}")
    print(f"Summary CSV: {summary.attrs.get('summary_csv')}")
    print(f"Summary report: {summary.attrs.get('summary_md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
