"""Audit alignment between paper stock targets and fresh trigger candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TRIGGER_SIGNAL_PATH = Path("D:/codex/outputs/signal_history/signals_latest.csv")
DEFAULT_TRIGGER_VALID_DAYS = 5

ALIGNMENT_COLUMNS = [
    "alignment_bucket",
    "code",
    "name",
    "portfolio_target_weight",
    "trigger_run_time",
    "trigger_signal_age_days",
    "trigger_signal_validity_status",
    "signal_type",
    "action",
    "score",
    "score_level",
    "pct",
    "last",
    "review_note",
]


@dataclass(frozen=True)
class StockTriggerAlignmentResult:
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


def _parse_signal_date(value: Any) -> pd.Timestamp | None:
    text = _text(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d_%H%M%S", "%Y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed).normalize()
    parsed = pd.to_datetime(text.replace("_", " "), errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _read_csv(path: Path, *, code_required: bool = True) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(path, dtype={"code": str}, encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, dtype={"code": str}, encoding="gbk")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    if frame.empty:
        return frame
    if "code" not in frame.columns:
        return pd.DataFrame() if code_required else frame
    frame = frame.copy()
    frame["code"] = frame["code"].map(_clean_code)
    return frame[frame["code"] != ""].copy()


def _signal_validity_fields(run_time: Any, as_of_date: Any, valid_days: int) -> dict[str, Any]:
    signal_date = _parse_signal_date(run_time)
    as_of = pd.to_datetime(as_of_date, errors="coerce")
    if signal_date is None or pd.isna(as_of):
        return {
            "trigger_signal_age_days": pd.NA,
            "trigger_signal_validity_status": "trigger_date_unknown",
        }
    age_days = int((pd.Timestamp(as_of).normalize() - signal_date).days)
    return {
        "trigger_signal_age_days": age_days,
        "trigger_signal_validity_status": "fresh_trigger_signal" if age_days <= valid_days else "stale_trigger_signal",
    }


def _row_from_target(target: dict[str, Any], bucket: str, note: str) -> dict[str, Any]:
    return {
        "alignment_bucket": bucket,
        "code": _clean_code(target.get("code")),
        "name": _text(target.get("name")),
        "portfolio_target_weight": _number(target.get("portfolio_target_weight")),
        "trigger_run_time": _text(target.get("trigger_run_time")),
        "trigger_signal_age_days": target.get("trigger_signal_age_days"),
        "trigger_signal_validity_status": _text(target.get("trigger_signal_validity_status"), "no_current_trigger"),
        "signal_type": _text(target.get("trigger_signal_type")),
        "action": _text(target.get("trigger_action")),
        "score": target.get("trigger_score"),
        "score_level": _text(target.get("trigger_score_level")),
        "pct": target.get("trigger_pct"),
        "last": target.get("trigger_last"),
        "review_note": note,
    }


def _row_from_signal(signal: dict[str, Any], as_of_date: Any, valid_days: int, bucket: str, note: str) -> dict[str, Any]:
    validity = _signal_validity_fields(signal.get("run_time"), as_of_date, valid_days)
    return {
        "alignment_bucket": bucket,
        "code": _clean_code(signal.get("code")),
        "name": _text(signal.get("name")),
        "portfolio_target_weight": 0.0,
        "trigger_run_time": _text(signal.get("run_time")),
        "trigger_signal_age_days": validity["trigger_signal_age_days"],
        "trigger_signal_validity_status": validity["trigger_signal_validity_status"],
        "signal_type": _text(signal.get("signal_type")),
        "action": _text(signal.get("action")),
        "score": signal.get("score"),
        "score_level": _text(signal.get("score_level")),
        "pct": signal.get("pct"),
        "last": signal.get("last"),
        "review_note": note,
    }


def _render_report(snapshot: dict[str, Any], table: pd.DataFrame) -> str:
    if table.empty:
        rows = "| - | - | - | - | - | - |\n"
    else:
        row_lines = []
        for row in table.to_dict(orient="records"):
            weight = _number(row.get("portfolio_target_weight")) * 100
            row_lines.append(
                "| {bucket} | `{code}` | {name} | {weight:.2f}% | `{validity}` | {note} |".format(
                    bucket=_text(row.get("alignment_bucket")),
                    code=_text(row.get("code")),
                    name=_text(row.get("name")),
                    weight=weight,
                    validity=_text(row.get("trigger_signal_validity_status")),
                    note=_text(row.get("review_note")),
                )
            )
        rows = "\n".join(row_lines) + "\n"
    return f"""# 个股目标-触发候选对齐审计
<!-- stock_trigger_alignment -->

生成时间：`{snapshot.get("generated_at")}`

本文件只做实盘前置研究审计，不连接券商、不自动下单。

## 摘要

| 项目 | 值 |
| --- | ---: |
| 截止日期 | {snapshot.get("as_of_date")} |
| 对齐状态 | `{snapshot.get("alignment_status")}` |
| 审计决策 | `{snapshot.get("alignment_decision")}` |
| 目标个股数 | {snapshot.get("target_count")} |
| fresh 触发信号数 | {snapshot.get("fresh_trigger_signal_count")} |
| 目标中有 fresh 触发 | {snapshot.get("target_with_fresh_trigger_count")} |
| 目标缺 fresh 触发 | {snapshot.get("target_without_fresh_trigger_count")} |
| fresh 触发但不在目标 | {snapshot.get("fresh_trigger_not_in_target_count")} |

## 明细

| 分组 | 代码 | 名称 | 目标仓位 | 触发有效性 | 复核说明 |
| --- | --- | --- | ---: | --- | --- |
{rows}
## 来源

- 个股目标：`{snapshot.get("stock_targets_path")}`
- 最新触发信号：`{snapshot.get("trigger_signal_path")}`
- 明细 CSV：`{snapshot.get("table_path")}`
"""


def run_stock_trigger_alignment(
    *,
    stock_targets_path: str | Path,
    trigger_signal_path: str | Path | None = None,
    output_dir: str | Path,
    as_of_date: str | pd.Timestamp,
    valid_days: int = DEFAULT_TRIGGER_VALID_DAYS,
) -> StockTriggerAlignmentResult:
    targets_path = Path(stock_targets_path)
    signal_path = Path(trigger_signal_path) if trigger_signal_path is not None else DEFAULT_TRIGGER_SIGNAL_PATH
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    targets = _read_csv(targets_path)
    signals = _read_csv(signal_path)

    if not targets.empty and "portfolio_target_weight" in targets.columns:
        weights = pd.to_numeric(targets["portfolio_target_weight"], errors="coerce").fillna(0.0)
        target_actions = (
            targets.get("target_action", pd.Series("", index=targets.index)).fillna("").astype(str).str.strip()
        )
        trigger_watch_candidates = target_actions == "trigger_watch_candidate"
        active_targets = targets[(weights > 0) | trigger_watch_candidates].copy()
    else:
        active_targets = targets.copy()

    if not signals.empty:
        signal_rows = []
        for raw in signals.to_dict(orient="records"):
            row = dict(raw)
            row.update(_signal_validity_fields(row.get("run_time"), as_of_date, valid_days))
            signal_rows.append(row)
        signal_frame = pd.DataFrame(signal_rows)
        fresh_signals = signal_frame[signal_frame["trigger_signal_validity_status"] == "fresh_trigger_signal"].copy()
    else:
        signal_frame = signals
        fresh_signals = signals

    target_codes = set(active_targets["code"].tolist()) if "code" in active_targets.columns else set()
    fresh_codes = set(fresh_signals["code"].tolist()) if "code" in fresh_signals.columns else set()
    rows: list[dict[str, Any]] = []

    for raw in active_targets.to_dict(orient="records"):
        code = _clean_code(raw.get("code"))
        if code in fresh_codes:
            rows.append(_row_from_target(raw, "target_with_fresh_trigger", "目标股已有 fresh 触发，可继续人工复核。"))
        else:
            rows.append(_row_from_target(raw, "target_without_fresh_trigger", "目标股缺少当前 fresh 触发，实盘前置不得作为买入依据。"))

    for raw in fresh_signals.to_dict(orient="records"):
        code = _clean_code(raw.get("code"))
        if code and code not in target_codes:
            rows.append(
                _row_from_signal(
                    raw,
                    as_of_date,
                    valid_days,
                    "fresh_trigger_not_in_target",
                    "触发系统给出 fresh 候选，但组合目标未纳入；需要复核选股/组合模型是否分叉。",
                )
            )

    table = pd.DataFrame(rows, columns=ALIGNMENT_COLUMNS)
    target_with_fresh = int((table["alignment_bucket"] == "target_with_fresh_trigger").sum()) if not table.empty else 0
    target_without_fresh = int((table["alignment_bucket"] == "target_without_fresh_trigger").sum()) if not table.empty else 0
    fresh_not_target = int((table["alignment_bucket"] == "fresh_trigger_not_in_target").sum()) if not table.empty else 0
    if target_without_fresh > 0 and fresh_not_target > 0:
        status = "misaligned"
        decision = "blocked_review_required"
    elif target_without_fresh > 0:
        status = "targets_missing_fresh_trigger"
        decision = "blocked"
    elif fresh_not_target > 0:
        status = "fresh_trigger_outside_targets"
        decision = "review_required"
    else:
        status = "aligned"
        decision = "ready_for_manual_review"

    table_path = resolved_output / "stock_trigger_alignment.csv"
    snapshot_path = resolved_output / "stock_trigger_alignment_snapshot.json"
    report_path = resolved_output / "stock_trigger_alignment.md"
    snapshot: dict[str, Any] = {
        "schema_version": "stock-trigger-alignment-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "as_of_date": pd.Timestamp(pd.to_datetime(as_of_date)).strftime("%Y-%m-%d"),
        "alignment_status": status,
        "alignment_decision": decision,
        "stock_targets_path": str(targets_path),
        "trigger_signal_path": str(signal_path),
        "table_path": str(table_path),
        "report_path": str(report_path),
        "snapshot_path": str(snapshot_path),
        "target_count": int(len(active_targets)),
        "trigger_signal_count": int(len(signals)),
        "fresh_trigger_signal_count": int(len(fresh_signals)),
        "target_with_fresh_trigger_count": target_with_fresh,
        "target_without_fresh_trigger_count": target_without_fresh,
        "fresh_trigger_not_in_target_count": fresh_not_target,
        "valid_days": int(valid_days),
        "research_only": True,
        "broker_action": "none",
    }
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, table), encoding="utf-8")
    return StockTriggerAlignmentResult(
        output_dir=resolved_output,
        table_path=table_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        table=table,
    )
