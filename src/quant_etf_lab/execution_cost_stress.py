"""Execution-cost stress review for allocator promotion evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .paper_account import run_paper_account


DEFAULT_COST_RATES = (0.0, 0.0003, 0.001)


@dataclass(frozen=True)
class ExecutionCostStressResult:
    output_dir: Path
    table_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _format_rate(rate: float) -> str:
    return f"{rate:.6f}".replace(".", "p").rstrip("0").rstrip("p") or "0"


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    return float(value or 0.0)


def _support_passes(
    edges: dict[str, float],
    min_return_edge: float,
    min_sharpe_edge: float,
    max_drawdown_worsening: float,
) -> bool:
    return (
        edges["return_edge"] >= float(min_return_edge)
        and edges["sharpe_edge"] >= float(min_sharpe_edge)
        and edges["drawdown_change"] >= -float(max_drawdown_worsening)
    )


def _render_report(snapshot: dict[str, Any], edge_rows: list[dict[str, Any]]) -> str:
    edge_table = [
        "| 成本率 | 收益边际 | Sharpe边际 | 回撤变化 | 支持候选 |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in edge_rows:
        edge_table.append(
            "| {rate} | {ret} | {sharpe} | {drawdown} | {supports} |".format(
                rate=_format_pct(row["cost_rate"]),
                ret=_format_pct(row["return_edge"]),
                sharpe=_format_float(row["sharpe_edge"]),
                drawdown=_format_pct(row["drawdown_change"]),
                supports="是" if row["supports_candidate"] else "否",
            )
        )
    return f"""# 执行成本压力测试复核

<!-- internal: Execution Cost Stress Review -->

生成时间：`{snapshot["generated_at"]}`

本复核会在配置级再平衡成本下重跑纸面账户配置器重建。它只用于研究分析，不连接券商，也不下单。

## 摘要

| 项目 | 数值 |
| --- | ---: |
| 状态 | `{snapshot["status"]}` |
| 所有成本率均支持候选 | {snapshot["all_cost_rates_support_candidate"]} |
| 最小收益边际 | {_format_pct(snapshot["min_return_edge"])} |
| 最小 Sharpe 边际 | {_format_float(snapshot["min_sharpe_edge"])} |
| 最小回撤变化 | {_format_pct(snapshot["min_drawdown_change"])} |

## 成本边际

{chr(10).join(edge_table)}

## 输出文件

- `execution_cost_stress.csv`：不同成本率下的基准和候选纸面账户指标。
- `execution_cost_stress_edges.csv`：不同成本率下的候选减基准边际。
- `execution_cost_stress_snapshot.json`：晋级闸门证据快照。
"""


def run_execution_cost_stress(
    project_root: str | Path = Path("."),
    config_path: str | Path = "configs/portfolio_core_satellite_quality_v2_guarded.yaml",
    baseline_dir: str | Path = (
        "outputs/portfolio_source_selection/main_chinext_source_selection_chip_deep_h5_cap20_margin03_validation6_v1"
    ),
    candidate_dir: str | Path = (
        "outputs/portfolio_source_selection/main_chinext_source_selection_chip_deep_h5_cap20_margin03_validation6_v1"
    ),
    output_dir: str | Path = "outputs/research/execution_cost_stress_latest",
    cost_rates: tuple[float, ...] | list[float] = DEFAULT_COST_RATES,
    min_return_edge: float = 0.03,
    min_sharpe_edge: float = 0.05,
    max_drawdown_worsening: float = 0.0,
) -> ExecutionCostStressResult:
    root = Path(project_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rates = tuple(float(rate) for rate in cost_rates)
    if not rates:
        raise ValueError("At least one execution cost rate is required.")
    negative = [rate for rate in rates if rate < 0.0]
    if negative:
        raise ValueError(f"Execution cost rates cannot be negative: {negative}")

    rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for rate in rates:
        rate_label = _format_rate(rate)
        baseline_result = run_paper_account(
            project_root=root,
            config_path=config_path,
            allocator_dir=baseline_dir,
            output_dir=out_dir / f"baseline_cost_{rate_label}",
            rebalance_cost_rate=rate,
        )
        candidate_result = run_paper_account(
            project_root=root,
            config_path=config_path,
            allocator_dir=candidate_dir,
            output_dir=out_dir / f"candidate_cost_{rate_label}",
            rebalance_cost_rate=rate,
        )
        for role, result in (("baseline", baseline_result), ("candidate", candidate_result)):
            rows.append(
                {
                    "cost_rate": rate,
                    "role": role,
                    "output_dir": str(result.output_dir),
                    "total_return": _metric(result.metrics, "total_return"),
                    "max_drawdown": _metric(result.metrics, "max_drawdown"),
                    "sharpe": _metric(result.metrics, "sharpe"),
                    "final_equity": _metric(result.metrics, "final_equity"),
                    "rebalance_cost_rate": _metric(result.metrics, "rebalance_cost_rate"),
                }
            )
        edges = {
            "cost_rate": rate,
            "return_edge": _metric(candidate_result.metrics, "total_return")
            - _metric(baseline_result.metrics, "total_return"),
            "sharpe_edge": _metric(candidate_result.metrics, "sharpe")
            - _metric(baseline_result.metrics, "sharpe"),
            "drawdown_change": _metric(candidate_result.metrics, "max_drawdown")
            - _metric(baseline_result.metrics, "max_drawdown"),
        }
        edge_rows.append(
            {
                **edges,
                "supports_candidate": _support_passes(
                    edges,
                    min_return_edge=min_return_edge,
                    min_sharpe_edge=min_sharpe_edge,
                    max_drawdown_worsening=max_drawdown_worsening,
                ),
            }
        )

    table_path = out_dir / "execution_cost_stress.csv"
    edges_path = out_dir / "execution_cost_stress_edges.csv"
    snapshot_path = out_dir / "execution_cost_stress_snapshot.json"
    report_path = out_dir / "execution_cost_stress_review.md"
    pd.DataFrame(rows).to_csv(table_path, index=False, encoding="utf-8-sig")
    edges = pd.DataFrame(edge_rows)
    edges.to_csv(edges_path, index=False, encoding="utf-8-sig")

    all_support = bool(edges["supports_candidate"].all()) if not edges.empty else False
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "supports_candidate" if all_support else "watch_candidate",
        "supports_candidate": all_support,
        "all_cost_rates_support_candidate": all_support,
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "config_path": str(config_path),
        "cost_rates": list(rates),
        "min_return_edge": round(float(edges["return_edge"].min()), 12) if not edges.empty else 0.0,
        "min_sharpe_edge": round(float(edges["sharpe_edge"].min()), 12) if not edges.empty else 0.0,
        "min_drawdown_change": round(float(edges["drawdown_change"].min()), 12) if not edges.empty else 0.0,
        "required_return_edge": float(min_return_edge),
        "required_sharpe_edge": float(min_sharpe_edge),
        "max_drawdown_worsening": float(max_drawdown_worsening),
        "table_path": str(table_path),
        "edges_path": str(edges_path),
    }
    snapshot_path.write_text(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")

    report_path.write_text(_render_report(snapshot, edge_rows), encoding="utf-8")

    return ExecutionCostStressResult(
        output_dir=out_dir,
        table_path=table_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
    )
