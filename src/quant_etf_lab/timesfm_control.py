"""TimesFM research-control registration outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONTROL_GROUP_ID = "timesfm_research_control_v1"
DEFAULT_CANDIDATE_SCOPE = ["source_curves", "etf_index_curves", "liquid_stock_sample"]
DEFAULT_HORIZONS = [1, 5, 20]


@dataclass(frozen=True)
class TimesFMControlResult:
    output_dir: Path
    snapshot_path: Path
    protocol_path: Path
    report_path: Path
    snapshot: dict[str, Any]


def _positive_ints(values: Iterable[int]) -> list[int]:
    parsed = sorted({int(value) for value in values})
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError("TimesFM control horizons must be positive integers.")
    return parsed


def _non_empty_scope(values: Iterable[str]) -> list[str]:
    parsed = [str(value).strip() for value in values if str(value).strip()]
    if not parsed:
        raise ValueError("TimesFM control candidate_scope cannot be empty.")
    return parsed


def build_timesfm_control_snapshot(
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    candidate_scope: Iterable[str] = DEFAULT_CANDIDATE_SCOPE,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the non-trading control-group contract for TimesFM."""
    horizon_values = _positive_ints(horizons)
    scope_values = _non_empty_scope(candidate_scope)
    return {
        "status": "timesfm_control_registered",
        "control_group_id": DEFAULT_CONTROL_GROUP_ID,
        "generated_at": generated_at or datetime.now().isoformat(timespec="seconds"),
        "model_family": "TimesFM",
        "model_source": "google-research/timesfm",
        "experiment_role": "control_group",
        "integration_status": "research_control_only",
        "position_effect": "none",
        "broker_action": "none",
        "dependency_policy": "optional_sidecar_only",
        "horizons": horizon_values,
        "candidate_scope": scope_values,
        "allowed_outputs": [
            "forecast_feature_csv",
            "uncertainty_reference_csv",
            "anomaly_reference_csv",
            "factor_lab_control_metrics",
            "source_selection_comparison_notes",
        ],
        "prohibited_integrations": [
            "default_allocator",
            "paper_account_target_weights",
            "daily_pipeline_position_sizing",
            "live_preflight_orders",
            "broker_order_generation",
        ],
        "comparison_protocol": {
            "primary_question": "Does TimesFM add predictive or risk-control information beyond current local factors?",
            "compare_against": [
                "current_factor_lab_baselines",
                "current_source_selection_candidates",
                "current_market_timing_state",
            ],
            "required_evidence": [
                "point_in_time_features_only",
                "rolling_oos_ic_or_spread",
                "source_selection_ablation",
                "drawdown_and_uncertainty_review",
            ],
        },
        "promotion_gate": {
            "requires_walk_forward_evidence": True,
            "requires_positive_oos_edge": True,
            "requires_no_future_leak_audit": True,
            "requires_dependency_isolation": True,
            "requires_user_approval_before_allocator_use": True,
        },
        "risk_notes": [
            "Do not rank all A-shares by raw TimesFM price forecasts without cost and liquidity tests.",
            "Treat quantile width as uncertainty evidence before considering any exposure change.",
            "Keep model weights and torch/jax dependencies outside base requirements until promotion gates pass.",
        ],
    }


def _render_protocol(snapshot: dict[str, Any]) -> str:
    scope = ", ".join(snapshot["candidate_scope"])
    horizons = ", ".join(f"{value}d" for value in snapshot["horizons"])
    prohibited = "\n".join(f"- `{item}`" for item in snapshot["prohibited_integrations"])
    evidence = "\n".join(f"- {item}" for item in snapshot["comparison_protocol"]["required_evidence"])
    return f"""# TimesFM 对照组协议

## 定位

- 对照组编号：`{snapshot["control_group_id"]}`
- 实验身份：`{snapshot["experiment_role"]}`
- 集成状态：`{snapshot["integration_status"]}`
- 依赖策略：`{snapshot["dependency_policy"]}`
- 观察范围：{scope}
- 观察周期：{horizons}

## 硬边界

TimesFM 当前只作为研究对照组，不生成目标仓位，不生成券商订单，不改变 paper-account、daily-pipeline 或正式 allocator 的默认行为。

禁止接入：
{prohibited}

## 对照问题

{snapshot["comparison_protocol"]["primary_question"]}

## 晋级前证据

{evidence}

## 风险备注

""" + "\n".join(f"- {item}" for item in snapshot["risk_notes"]) + "\n"


def _render_report(snapshot: dict[str, Any]) -> str:
    return f"""# TimesFM 对照组登记

| 项目 | 值 |
| --- | --- |
| 对照组编号 | `{snapshot["control_group_id"]}` |
| 状态 | `{snapshot["integration_status"]}` |
| 仓位影响 | `{snapshot["position_effect"]}` |
| 券商动作 | `{snapshot["broker_action"]}` |
| 依赖策略 | `{snapshot["dependency_policy"]}` |
| 观察范围 | `{", ".join(snapshot["candidate_scope"])}` |
| 观察周期 | `{", ".join(str(value) for value in snapshot["horizons"])}` |

## 禁止接入

""" + "\n".join(f"- `{item}`" for item in snapshot["prohibited_integrations"]) + """

## 下一步

- 只生成 TimesFM 预测特征、分位数不确定性和异常参考。
- 与当前 factor-lab、source-selection、market-timing-state 做旁路比较。
- 在走完 walk-forward、无未来函数审计和人工确认前，不参与资金分配。
"""


def run_timesfm_control_group(
    output_dir: str | Path = "outputs/research/timesfm_control_latest",
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    candidate_scope: Iterable[str] = DEFAULT_CANDIDATE_SCOPE,
    generated_at: str | None = None,
) -> TimesFMControlResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    snapshot = build_timesfm_control_snapshot(
        horizons=horizons,
        candidate_scope=candidate_scope,
        generated_at=generated_at,
    )
    snapshot_path = output_path / "timesfm_control_snapshot.json"
    protocol_path = output_path / "timesfm_control_protocol.md"
    report_path = output_path / "timesfm_control_report.md"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol_path.write_text(_render_protocol(snapshot), encoding="utf-8")
    report_path.write_text(_render_report(snapshot), encoding="utf-8")
    return TimesFMControlResult(
        output_dir=output_path,
        snapshot_path=snapshot_path,
        protocol_path=protocol_path,
        report_path=report_path,
        snapshot=snapshot,
    )
