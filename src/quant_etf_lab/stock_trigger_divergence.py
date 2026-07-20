"""Explain divergence between trigger-monitor candidates and portfolio stock targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DIVERGENCE_COLUMNS = [
    "divergence_bucket",
    "code",
    "name",
    "primary_diagnosis",
    "diagnosis_text",
    "recommended_next_step",
    "portfolio_target_weight",
    "trigger_signal_type",
    "trigger_action",
    "trigger_score",
    "trigger_pct",
    "trigger_reason",
    "source_strategy_profile",
    "risk_filter_status",
    "execution_gate_action",
    "review_bucket",
    "review_stage",
    "review_reason",
    "manual_review_state",
    "in_momentum_focus",
    "momentum_focus_score",
    "momentum_research_priority",
    "in_source_trades",
    "latest_source_trade_date",
    "latest_source_trade_side",
    "broker_action",
    "research_only",
]


@dataclass(frozen=True)
class StockTriggerDivergenceResult:
    output_dir: Path
    table_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    table: pd.DataFrame


def _clean_code(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6) if digits else text.zfill(6)


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    return text


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    resolved = Path(path)
    if not resolved.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(resolved, dtype={"code": str}, encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(resolved, dtype={"code": str}, encoding="gbk")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    if frame.empty:
        return frame
    if "code" in frame.columns:
        frame = frame.copy()
        frame["code"] = frame["code"].map(_clean_code)
        frame = frame[frame["code"] != ""].copy()
    return frame


def _records_by_code(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "code" not in frame.columns:
        return {}
    records: dict[str, dict[str, Any]] = {}
    for raw in frame.to_dict(orient="records"):
        code = _clean_code(raw.get("code"))
        if code:
            records[code] = raw
    return records


def _latest_trade_by_code(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty or "code" not in frame.columns:
        return {}
    data = frame.copy()
    if "date" in data.columns:
        data["_sort_date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.sort_values(["code", "_sort_date"])
    records: dict[str, dict[str, Any]] = {}
    for raw in data.to_dict(orient="records"):
        code = _clean_code(raw.get("code"))
        if code:
            records[code] = raw
    return records


def _first_existing_path_from_column(frame: pd.DataFrame, column: str) -> Path | None:
    if frame.empty or column not in frame.columns:
        return None
    for value in frame[column].dropna().tolist():
        text = _text(value)
        if text:
            path = Path(text)
            if path.exists():
                return path
    return None


def _bool_text(value: bool) -> str:
    return "是" if value else "否"


def _target_waiting_row(alignment: dict[str, Any], target: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    review_bucket = _text(review.get("review_bucket"), "not_in_review_queue")
    diagnosis = "portfolio_target_waiting_for_short_term_retrigger"
    if review_bucket in {"drawdown_review", "watch_review"}:
        diagnosis_text = "组合模型仍给目标仓位，但个股处于复核/观察语境，且短线触发监控未确认；等待短线再触发。"
    else:
        diagnosis_text = "组合模型给出目标仓位，但最新触发监控没有 fresh 信号；这是长期/多因子目标等待短线再触发。"
    return {
        "divergence_bucket": "target_without_fresh_trigger",
        "code": _clean_code(alignment.get("code")),
        "name": _text(alignment.get("name") or target.get("name")),
        "primary_diagnosis": diagnosis,
        "diagnosis_text": diagnosis_text,
        "recommended_next_step": "继续观察组合目标；实盘前置买入必须等待 fresh 触发或人工复核重新确认。",
        "portfolio_target_weight": _number(alignment.get("portfolio_target_weight") or target.get("portfolio_target_weight")),
        "trigger_signal_type": _text(alignment.get("signal_type")),
        "trigger_action": _text(alignment.get("action")),
        "trigger_score": _number(alignment.get("score")),
        "trigger_pct": _number(alignment.get("pct")),
        "trigger_reason": "",
        "source_strategy_profile": _text(target.get("source_strategy_profile")),
        "risk_filter_status": _text(target.get("risk_filter_status")),
        "execution_gate_action": _text(target.get("execution_gate_action"), "wait_for_retrigger_signal"),
        "review_bucket": review_bucket,
        "review_stage": _text(review.get("review_stage")),
        "review_reason": _text(review.get("review_reason")),
        "manual_review_state": _text(review.get("manual_review_state")),
        "in_momentum_focus": False,
        "momentum_focus_score": 0.0,
        "momentum_research_priority": "",
        "in_source_trades": True,
        "latest_source_trade_date": _text(target.get("last_trade_date")),
        "latest_source_trade_side": "TARGET",
        "broker_action": "none",
        "research_only": True,
    }


def _external_trigger_row(
    alignment: dict[str, Any],
    signal: dict[str, Any],
    momentum: dict[str, Any],
    trade: dict[str, Any],
) -> dict[str, Any]:
    code = _clean_code(alignment.get("code"))
    in_momentum = bool(momentum)
    in_trades = bool(trade)
    if in_momentum and not in_trades:
        diagnosis = "research_pool_not_promoted_to_portfolio"
        diagnosis_text = "触发候选也在强势研究池，但未进入当前组合目标，说明研究池尚未被组合层晋级。"
    elif in_trades:
        diagnosis = "source_trade_inactive_or_closed"
        diagnosis_text = "触发候选曾出现在核心源交易记录，但当前不是目标持仓，需复核是否已被源模型调出。"
    else:
        diagnosis = "external_trigger_only_not_confirmed_by_model"
        diagnosis_text = "外部触发监控给出 fresh 候选，但强势研究池、核心多因子目标和源交易记录均未确认。"
    return {
        "divergence_bucket": "fresh_trigger_not_in_target",
        "code": code,
        "name": _text(alignment.get("name") or signal.get("name")),
        "primary_diagnosis": diagnosis,
        "diagnosis_text": diagnosis_text,
        "recommended_next_step": "只进入人工观察；不要绕过组合模型和实盘前置门禁直接买入。",
        "portfolio_target_weight": 0.0,
        "trigger_signal_type": _text(alignment.get("signal_type") or signal.get("signal_type")),
        "trigger_action": _text(alignment.get("action") or signal.get("action")),
        "trigger_score": _number(alignment.get("score") or signal.get("score")),
        "trigger_pct": _number(alignment.get("pct") or signal.get("pct")),
        "trigger_reason": _text(signal.get("reason") or signal.get("score_reason")),
        "source_strategy_profile": "",
        "risk_filter_status": "not_in_current_portfolio_target",
        "execution_gate_action": "manual_review_only",
        "review_bucket": "trigger_outside_target_review",
        "review_stage": "review_required",
        "review_reason": "fresh trigger candidate is outside current model target list",
        "manual_review_state": "",
        "in_momentum_focus": in_momentum,
        "momentum_focus_score": _number(momentum.get("focus_score")),
        "momentum_research_priority": _text(momentum.get("research_priority")),
        "in_source_trades": in_trades,
        "latest_source_trade_date": _text(trade.get("date")),
        "latest_source_trade_side": _text(trade.get("side")),
        "broker_action": "none",
        "research_only": True,
    }


def _render_report(snapshot: dict[str, Any], table: pd.DataFrame) -> str:
    if table.empty:
        rows = "| - | - | - | - | - | - | - |\n"
    else:
        lines = []
        for row in table.to_dict(orient="records"):
            lines.append(
                "| {bucket} | `{code}` | {name} | `{diagnosis}` | {target_weight:.2f}% | {trigger_score:.1f} | {next_step} |".format(
                    bucket=_text(row.get("divergence_bucket")),
                    code=_text(row.get("code")),
                    name=_text(row.get("name")),
                    diagnosis=_text(row.get("primary_diagnosis")),
                    target_weight=_number(row.get("portfolio_target_weight")) * 100,
                    trigger_score=_number(row.get("trigger_score")),
                    next_step=_text(row.get("recommended_next_step")),
                )
            )
        rows = "\n".join(lines) + "\n"
    summary_lines = []
    if not table.empty:
        for row in table.to_dict(orient="records"):
            summary_lines.append(
                "- `{code}` {name}：{diagnosis_text} 动作：{next_step}".format(
                    code=_text(row.get("code")),
                    name=_text(row.get("name")),
                    diagnosis_text=_text(row.get("diagnosis_text")),
                    next_step=_text(row.get("recommended_next_step")),
                )
            )
    summaries = "\n".join(summary_lines) if summary_lines else "- 暂无分叉行。"
    return f"""# 个股目标-触发分叉归因
<!-- stock_trigger_divergence -->

生成时间：`{snapshot.get("generated_at")}`

本文件只解释模型目标和触发候选之间的分叉，不连接券商、不自动下单。

## 摘要

| 项目 | 值 |
| --- | ---: |
| 截止日期 | {snapshot.get("as_of_date")} |
| 归因状态 | `{snapshot.get("divergence_status")}` |
| 分叉行数 | {snapshot.get("divergence_count")} |
| 外部触发独有 | {snapshot.get("external_trigger_only_count")} |
| 研究池未晋级组合 | {snapshot.get("research_pool_not_promoted_count")} |
| 源模型已调出/未激活 | {snapshot.get("source_trade_inactive_count")} |
| 目标等待再触发 | {snapshot.get("target_waiting_retrigger_count")} |
| 券商动作 | `{snapshot.get("broker_action")}` |

## 归因表

| 分组 | 代码 | 名称 | 主诊断 | 目标仓位 | 触发分数 | 下一步 |
| --- | --- | --- | --- | ---: | ---: | --- |
{rows}
## 逐项说明

{summaries}

## 来源

- 对齐审计：`{snapshot.get("alignment_path")}`
- 个股目标：`{snapshot.get("stock_targets_path")}`
- 最新触发信号：`{snapshot.get("trigger_signal_path")}`
- 强势股研究池：`{snapshot.get("momentum_focus_path")}`
- 源交易记录：`{snapshot.get("source_trades_path")}`
- 明细 CSV：`{snapshot.get("table_path")}`
"""


def run_stock_trigger_divergence(
    *,
    alignment_path: str | Path,
    stock_targets_path: str | Path,
    trigger_signal_path: str | Path,
    output_dir: str | Path,
    stock_target_review_path: str | Path | None = None,
    momentum_focus_path: str | Path | None = None,
    source_trades_path: str | Path | None = None,
    as_of_date: str | pd.Timestamp,
) -> StockTriggerDivergenceResult:
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    alignment = _read_csv(alignment_path)
    targets = _read_csv(stock_targets_path)
    signals = _read_csv(trigger_signal_path)
    review = _read_csv(stock_target_review_path)
    momentum = _read_csv(momentum_focus_path)
    resolved_source_trades_path = Path(source_trades_path) if source_trades_path is not None else None
    if resolved_source_trades_path is None:
        resolved_source_trades_path = _first_existing_path_from_column(targets, "source_trades_path")
    trades = _read_csv(resolved_source_trades_path)

    targets_by_code = _records_by_code(targets)
    signals_by_code = _records_by_code(signals)
    review_by_code = _records_by_code(review)
    momentum_by_code = _records_by_code(momentum)
    trades_by_code = _latest_trade_by_code(trades)

    rows: list[dict[str, Any]] = []
    if not alignment.empty:
        for raw in alignment.to_dict(orient="records"):
            code = _clean_code(raw.get("code"))
            bucket = _text(raw.get("alignment_bucket"))
            if bucket == "target_without_fresh_trigger":
                rows.append(_target_waiting_row(raw, targets_by_code.get(code, {}), review_by_code.get(code, {})))
            elif bucket == "fresh_trigger_not_in_target":
                rows.append(
                    _external_trigger_row(
                        raw,
                        signals_by_code.get(code, {}),
                        momentum_by_code.get(code, {}),
                        trades_by_code.get(code, {}),
                    )
                )

    table = pd.DataFrame(rows, columns=DIVERGENCE_COLUMNS)
    diagnosis = table["primary_diagnosis"] if not table.empty else pd.Series(dtype=str)
    external_count = int((diagnosis == "external_trigger_only_not_confirmed_by_model").sum())
    research_pool_count = int((diagnosis == "research_pool_not_promoted_to_portfolio").sum())
    source_inactive_count = int((diagnosis == "source_trade_inactive_or_closed").sum())
    target_wait_count = int(diagnosis.astype(str).str.startswith("portfolio_target_waiting").sum())
    status = "aligned" if table.empty else "review_required"

    table_path = resolved_output / "stock_trigger_divergence.csv"
    snapshot_path = resolved_output / "stock_trigger_divergence_snapshot.json"
    report_path = resolved_output / "stock_trigger_divergence.md"
    snapshot: dict[str, Any] = {
        "schema_version": "stock-trigger-divergence-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": pd.Timestamp(pd.to_datetime(as_of_date)).strftime("%Y-%m-%d"),
        "divergence_status": status,
        "divergence_count": int(len(table)),
        "external_trigger_only_count": external_count,
        "research_pool_not_promoted_count": research_pool_count,
        "source_trade_inactive_count": source_inactive_count,
        "target_waiting_retrigger_count": target_wait_count,
        "alignment_path": str(alignment_path),
        "stock_targets_path": str(stock_targets_path),
        "trigger_signal_path": str(trigger_signal_path),
        "stock_target_review_path": str(stock_target_review_path) if stock_target_review_path is not None else None,
        "momentum_focus_path": str(momentum_focus_path) if momentum_focus_path is not None else None,
        "source_trades_path": str(resolved_source_trades_path) if resolved_source_trades_path is not None else None,
        "table_path": str(table_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "research_only": True,
        "broker_action": "none",
    }
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, table), encoding="utf-8")
    return StockTriggerDivergenceResult(
        output_dir=resolved_output,
        table_path=table_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        table=table,
    )
