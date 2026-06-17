"""Outcome backfill for observation-only chip-reversal candidates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .market_data_source import (
    DEFAULT_DAILY_MARKET_DATA_DIR,
    DEFAULT_EXCHANGE_INGEST_DIR,
    load_market_snapshot_rows,
)


DEFAULT_CANDIDATES_PATH = Path("outputs/research/chip_reversal_daily_candidates_latest/chip_reversal_daily_candidates.csv")
DEFAULT_DATA_DIR = Path("data/processed/stocks")
DEFAULT_OUTPUT_DIR = Path("outputs/research/chip_reversal_candidate_outcomes_latest")


@dataclass(frozen=True)
class ChipReversalCandidateOutcomesResult:
    output_dir: Path
    outcomes_path: Path
    summary_path: Path
    group_summary_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    outcomes: pd.DataFrame
    summary: pd.DataFrame
    group_summary: pd.DataFrame


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = Path(path) if path is not None else default
    return raw if raw.is_absolute() else project_root / raw


def _clean_code(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[-6:].zfill(6) if digits else text


def _normalize_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _row_value(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def _float_or_default(value: Any, default: float | None = None) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return default
    return float(parsed)


def _snapshot_row_to_history(row: dict[str, Any]) -> dict[str, Any] | None:
    code = _clean_code(_row_value(row, "security_code", "code"))
    date = _normalize_date(_row_value(row, "trade_date", "date"))
    close = _float_or_default(_row_value(row, "close_price", "close", "last_price"))
    if not code or not date or close is None or close <= 0:
        return None
    open_price = _float_or_default(_row_value(row, "open_price", "open"), close)
    high = _float_or_default(_row_value(row, "high_price", "high"), close)
    low = _float_or_default(_row_value(row, "low_price", "low"), close)
    return {
        "date": date,
        "code": code,
        "name": _row_value(row, "security_name", "name") or code,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": _float_or_default(_row_value(row, "volume"), 0.0),
        "amount": _float_or_default(_row_value(row, "turnover", "amount"), 0.0),
    }


def _overlay_market_snapshot_rows(
    histories: dict[str, pd.DataFrame],
    rows: list[dict[str, Any]],
    codes: Iterable[str],
) -> dict[str, pd.DataFrame]:
    wanted = {_clean_code(code) for code in codes}
    snapshot_by_code: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        history_row = _snapshot_row_to_history(row)
        if history_row is None:
            continue
        code = str(history_row["code"])
        if code not in wanted:
            continue
        snapshot_by_code.setdefault(code, []).append(history_row)

    overlaid = dict(histories)
    for code, price_rows in snapshot_by_code.items():
        existing = overlaid.get(code, pd.DataFrame())
        data = existing.copy()
        if not data.empty and "name" in data.columns and not data["name"].dropna().empty:
            prior_name = str(data["name"].dropna().iloc[-1])
            price_rows = [
                {**row, "name": prior_name} if row.get("name") == code else row
                for row in price_rows
            ]
        patched = pd.concat([data, pd.DataFrame(price_rows)], ignore_index=True)
        patched["date"] = pd.to_datetime(patched["date"], errors="coerce")
        patched["code"] = patched["code"].map(_clean_code)
        for column in ["open", "close"]:
            patched[column] = pd.to_numeric(patched[column], errors="coerce")
        patched = patched.dropna(subset=["date", "code", "open", "close"])
        patched = patched[(patched["open"] > 0) & (patched["close"] > 0)]
        overlaid[code] = (
            patched.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
    return overlaid


def _read_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str})
    missing = {"date", "open", "close"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    data = frame.copy()
    if "code" not in data.columns:
        data["code"] = path.stem[-6:].zfill(6)
    if "name" not in data.columns:
        data["name"] = data["code"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].map(_clean_code)
    for column in ["open", "close"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "code", "open", "close"])
    data = data[(data["open"] > 0) & (data["close"] > 0)]
    return data.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def _normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ["open", "close"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["date", "open", "close"]).sort_values("date").reset_index(drop=True)


def _history_map(data_dir: Path, codes: Iterable[str]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    histories: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for code in sorted({_clean_code(code) for code in codes}):
        path = data_dir / f"{code}.csv"
        if not path.exists():
            missing.append(code)
            continue
        histories[code] = _read_history(path)
    return histories, missing


def _load_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str})
    required = {"date", "code", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].map(_clean_code)
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    return data.dropna(subset=["date", "code", "close"]).reset_index(drop=True)


def _candidate_value(row: pd.Series, column: str, default: Any = None) -> Any:
    return row[column] if column in row and pd.notna(row[column]) else default


def _outcome_rows_for_candidate(
    candidate: pd.Series,
    history: pd.DataFrame | None,
    horizons: list[int],
    success_threshold: float,
) -> list[dict[str, Any]]:
    code = _clean_code(candidate.get("code"))
    signal_date = pd.Timestamp(candidate.get("date"))
    signal_close = float(candidate.get("close"))
    rows: list[dict[str, Any]] = []
    if history is None or history.empty:
        for horizon in horizons:
            rows.append(_pending_row(candidate, horizon, "missing_history"))
        return rows

    hist = _normalize_history(history)
    signal_positions = hist.index[hist["date"] == signal_date].tolist()
    if signal_positions:
        signal_pos = int(signal_positions[-1])
        target_pos_for_horizon = lambda value: signal_pos + int(value)
        next_bar_pos = signal_pos + 1
    else:
        future_positions = hist.index[hist["date"] > signal_date].tolist()
        if not future_positions:
            for horizon in horizons:
                rows.append(_pending_row(candidate, horizon, "future_bar_not_available"))
            return rows
        first_future_pos = int(future_positions[0])
        target_pos_for_horizon = lambda value: first_future_pos + int(value) - 1
        next_bar_pos = first_future_pos
    for horizon in horizons:
        target_pos = target_pos_for_horizon(horizon)
        if target_pos >= len(hist):
            rows.append(_pending_row(candidate, horizon, "future_bar_not_available"))
            continue
        target = hist.iloc[target_pos]
        next_bar = hist.iloc[next_bar_pos] if next_bar_pos < len(hist) else None
        target_close = float(target["close"])
        next_open = float(next_bar["open"]) if next_bar is not None else pd.NA
        return_close = target_close / signal_close - 1.0
        trade_return = target_close / float(next_open) - 1.0 if next_bar is not None and float(next_open) > 0 else pd.NA
        rows.append(
            {
                "candidate_rank": _candidate_value(candidate, "candidate_rank"),
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "code": code,
                "name": _candidate_value(candidate, "name", code),
                "board": _candidate_value(candidate, "board"),
                "score_bucket": _candidate_value(candidate, "score_bucket"),
                "horizon": f"{int(horizon)}d",
                "outcome_status": "ready",
                "pending_reason": "",
                "signal_close": signal_close,
                "next_open": next_open,
                "target_date": pd.Timestamp(target["date"]).strftime("%Y-%m-%d"),
                "target_close": target_close,
                "return_close": return_close,
                "trade_return_open_to_close": trade_return,
                "success_threshold": float(success_threshold),
                "success": bool(return_close >= float(success_threshold)),
                "priority_score": _candidate_value(candidate, "priority_score"),
                "watch_posture": _candidate_value(candidate, "watch_posture"),
                "broker_action": "none",
                "research_only": True,
            }
        )
    return rows


def _pending_row(candidate: pd.Series, horizon: int, reason: str) -> dict[str, Any]:
    signal_date = pd.Timestamp(candidate.get("date"))
    code = _clean_code(candidate.get("code"))
    return {
        "candidate_rank": _candidate_value(candidate, "candidate_rank"),
        "signal_date": signal_date.strftime("%Y-%m-%d") if pd.notna(signal_date) else None,
        "code": code,
        "name": _candidate_value(candidate, "name", code),
        "board": _candidate_value(candidate, "board"),
        "score_bucket": _candidate_value(candidate, "score_bucket"),
        "horizon": f"{int(horizon)}d",
        "outcome_status": "pending",
        "pending_reason": reason,
        "signal_close": _candidate_value(candidate, "close"),
        "next_open": pd.NA,
        "target_date": None,
        "target_close": pd.NA,
        "return_close": pd.NA,
        "trade_return_open_to_close": pd.NA,
        "success_threshold": pd.NA,
        "success": pd.NA,
        "priority_score": _candidate_value(candidate, "priority_score"),
        "watch_posture": _candidate_value(candidate, "watch_posture"),
        "broker_action": "none",
        "research_only": True,
    }


def _build_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if outcomes.empty:
        return pd.DataFrame(columns=["horizon", "ready_count", "pending_count", "success_count", "success_rate", "avg_return_close"])
    for horizon, group in outcomes.groupby("horizon", sort=True):
        ready = group[group["outcome_status"] == "ready"].copy()
        success = ready["success"].fillna(False).astype(bool) if "success" in ready else pd.Series(dtype=bool)
        rows.append(
            {
                "horizon": horizon,
                "ready_count": int(len(ready)),
                "pending_count": int((group["outcome_status"] == "pending").sum()),
                "success_count": int(success.sum()),
                "success_rate": float(success.mean()) if len(success) else pd.NA,
                "avg_return_close": float(pd.to_numeric(ready.get("return_close"), errors="coerce").mean()) if len(ready) else pd.NA,
                "avg_trade_return_open_to_close": (
                    float(pd.to_numeric(ready.get("trade_return_open_to_close"), errors="coerce").mean()) if len(ready) else pd.NA
                ),
            }
        )
    return pd.DataFrame(rows)


def _promotion_gate_snapshot(
    summary: pd.DataFrame,
    min_ready_per_horizon: int,
    min_success_rate: float,
) -> dict[str, Any]:
    if min_ready_per_horizon < 0:
        raise ValueError("min_ready_per_horizon must be non-negative.")
    if not 0 <= min_success_rate <= 1:
        raise ValueError("min_success_rate must be between 0 and 1.")

    reasons: list[str] = []
    if summary.empty:
        reasons.append("no_outcome_summary")
    for _, row in summary.iterrows():
        horizon = str(row.get("horizon"))
        ready_count = int(row.get("ready_count") or 0)
        success_rate = pd.to_numeric(pd.Series([row.get("success_rate")]), errors="coerce").iloc[0]
        if ready_count < int(min_ready_per_horizon):
            reasons.append(f"{horizon}:ready_count_below_minimum")
        if pd.isna(success_rate):
            reasons.append(f"{horizon}:success_rate_unavailable")
        elif float(success_rate) < float(min_success_rate):
            reasons.append(f"{horizon}:success_rate_below_minimum")
    return {
        "promotion_gate_status": "blocked" if reasons else "pass",
        "promotion_gate_reasons": reasons,
        "promotion_gate_min_ready_per_horizon": int(min_ready_per_horizon),
        "promotion_gate_min_success_rate": float(min_success_rate),
    }


def _horizon_sort_key(value: Any) -> tuple[int, str]:
    text = str(value or "")
    number = "".join(ch for ch in text if ch.isdigit())
    return (int(number) if number else 9999, text)


def _outcome_readiness_snapshot(
    candidates: pd.DataFrame,
    outcomes: pd.DataFrame,
    market_trade_date: str | None,
) -> dict[str, Any]:
    ready_horizons: list[str] = []
    pending_horizons: list[str] = []
    pending_reasons: list[str] = []
    next_horizon: str | None = None
    next_reason: str | None = None

    latest_signal_date: str | None = None
    if not candidates.empty and "date" in candidates.columns:
        signal_dates = pd.to_datetime(candidates["date"], errors="coerce").dropna()
        if not signal_dates.empty:
            latest_signal_date = pd.Timestamp(signal_dates.max()).strftime("%Y-%m-%d")

    if not outcomes.empty:
        ready = outcomes[outcomes["outcome_status"] == "ready"].copy()
        pending = outcomes[outcomes["outcome_status"] == "pending"].copy()
        ready_horizons = sorted({str(value) for value in ready.get("horizon", [])}, key=_horizon_sort_key)
        pending_horizons = sorted({str(value) for value in pending.get("horizon", [])}, key=_horizon_sort_key)
        if "pending_reason" in pending.columns:
            pending_reasons = sorted({str(value) for value in pending["pending_reason"].dropna() if str(value)})
        if pending_horizons:
            next_horizon = pending_horizons[0]
            horizon_pending = pending[pending["horizon"].astype(str) == next_horizon]
            if "pending_reason" in horizon_pending.columns and not horizon_pending.empty:
                reasons = [str(value) for value in horizon_pending["pending_reason"].dropna() if str(value)]
                next_reason = reasons[0] if reasons else None

    if outcomes.empty and candidates.empty:
        status = "no_candidates"
    elif outcomes.empty:
        status = "no_outcomes"
    elif not pending_horizons:
        status = "ready"
    elif ready_horizons:
        status = "partial"
    elif set(pending_reasons) == {"future_bar_not_available"}:
        status = "waiting_for_future_bar"
    elif "missing_history" in pending_reasons:
        status = "blocked_missing_history"
    else:
        status = "pending"

    warning = None
    if status == "waiting_for_future_bar":
        warning = "waiting_for_future_bar"
    elif status == "partial":
        warning = "partial_outcome_samples"
    elif status == "blocked_missing_history":
        warning = "missing_history"

    return {
        "outcome_readiness_status": status,
        "outcome_analysis_status": status,
        "outcome_ready_horizons": ready_horizons,
        "outcome_pending_horizons": pending_horizons,
        "outcome_ready_horizon_count": int(len(ready_horizons)),
        "outcome_pending_horizon_count": int(len(pending_horizons)),
        "next_outcome_review_horizon": next_horizon,
        "next_outcome_review_reason": next_reason,
        "outcome_pending_reasons": pending_reasons,
        "latest_signal_date": latest_signal_date,
        "latest_available_market_trade_date": market_trade_date,
        "outcome_sample_warning": warning,
    }


def _priority_bucket(value: Any) -> str:
    try:
        rank = int(float(value))
    except (TypeError, ValueError):
        return "unknown"
    if rank <= 10:
        return "top_10"
    if rank <= 20:
        return "top_20"
    return "tail"


def _group_metric_rows(outcomes: pd.DataFrame, group_type: str, group_column: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if outcomes.empty:
        return rows
    source = outcomes.copy()
    if group_type == "all":
        source["_group_value"] = "all"
    elif group_column and group_column in source.columns:
        source["_group_value"] = source[group_column].fillna("unknown").astype(str)
    else:
        return rows

    for (horizon, group_value), group in source.groupby(["horizon", "_group_value"], sort=True):
        ready = group[group["outcome_status"] == "ready"].copy()
        success = ready["success"].fillna(False).astype(bool) if "success" in ready else pd.Series(dtype=bool)
        rows.append(
            {
                "horizon": horizon,
                "group_type": group_type,
                "group_value": group_value,
                "ready_count": int(len(ready)),
                "pending_count": int((group["outcome_status"] == "pending").sum()),
                "success_count": int(success.sum()),
                "success_rate": float(success.mean()) if len(success) else pd.NA,
                "avg_return_close": float(pd.to_numeric(ready.get("return_close"), errors="coerce").mean()) if len(ready) else pd.NA,
                "avg_trade_return_open_to_close": (
                    float(pd.to_numeric(ready.get("trade_return_open_to_close"), errors="coerce").mean()) if len(ready) else pd.NA
                ),
            }
        )
    return rows


def build_outcome_group_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "horizon",
        "group_type",
        "group_value",
        "ready_count",
        "pending_count",
        "success_count",
        "success_rate",
        "avg_return_close",
        "avg_trade_return_open_to_close",
    ]
    if outcomes.empty:
        return pd.DataFrame(columns=columns)
    source = outcomes.copy()
    source["priority_bucket"] = source["candidate_rank"].map(_priority_bucket)
    rows: list[dict[str, Any]] = []
    rows.extend(_group_metric_rows(source, "all"))
    rows.extend(_group_metric_rows(source, "board", "board"))
    rows.extend(_group_metric_rows(source, "score_bucket", "score_bucket"))
    rows.extend(_group_metric_rows(source, "priority_bucket", "priority_bucket"))
    return pd.DataFrame(rows).reindex(columns=columns)


def build_chip_reversal_candidate_outcomes(
    candidates: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    horizons: Iterable[int] = (1, 2),
    success_threshold: float = 0.03,
    min_ready_per_horizon: int = 30,
    min_success_rate: float = 0.55,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    horizon_values = [int(value) for value in horizons]
    if not horizon_values or any(value <= 0 for value in horizon_values):
        raise ValueError("horizons must contain positive integers.")
    source = candidates.copy()
    if source.empty:
        empty = pd.DataFrame()
        empty_summary = _build_summary(empty)
        snapshot = {
            "status": "no_candidates",
            "candidate_count": 0,
            "ready_outcome_count": 0,
            "pending_outcome_count": 0,
            "broker_action": "none",
            "research_only": True,
            **_promotion_gate_snapshot(empty_summary, min_ready_per_horizon, min_success_rate),
        }
        return empty, empty_summary, snapshot
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["code"] = source["code"].map(_clean_code)
    source["close"] = pd.to_numeric(source["close"], errors="coerce")
    source = source.dropna(subset=["date", "code", "close"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for _, candidate in source.iterrows():
        rows.extend(
            _outcome_rows_for_candidate(
                candidate,
                histories.get(_clean_code(candidate.get("code"))),
                horizon_values,
                success_threshold,
            )
        )
    outcomes = pd.DataFrame(rows)
    if not outcomes.empty:
        outcomes["research_only"] = outcomes["research_only"].astype(object)
    summary = _build_summary(outcomes)
    snapshot = {
        "status": "ok" if not outcomes.empty else "no_outcomes",
        "candidate_count": int(len(source)),
        "horizons": horizon_values,
        "success_threshold": float(success_threshold),
        "ready_outcome_count": int((outcomes.get("outcome_status") == "ready").sum()) if not outcomes.empty else 0,
        "pending_outcome_count": int((outcomes.get("outcome_status") == "pending").sum()) if not outcomes.empty else 0,
        "broker_action": "none",
        "research_only": True,
        **_promotion_gate_snapshot(summary, min_ready_per_horizon, min_success_rate),
    }
    return outcomes, summary, snapshot


def _render_report(snapshot: dict[str, Any], summary: pd.DataFrame) -> str:
    if summary.empty:
        rows = "| N/A | 0 | 0 | 0 | N/A | N/A |"
    else:
        rows = "\n".join(
            "| {horizon} | {ready} | {pending} | {success} | {rate} | {avg} |".format(
                horizon=row.get("horizon"),
                ready=int(row.get("ready_count") or 0),
                pending=int(row.get("pending_count") or 0),
                success=int(row.get("success_count") or 0),
                rate=_pct(row.get("success_rate")),
                avg=_pct(row.get("avg_return_close")),
            )
            for _, row in summary.iterrows()
        )
    return f"""# 候选结果回填

生成时间：`{snapshot.get("generated_at")}`

这是对筹码反转观察候选的结果成熟度和后续收益统计。它不是买卖建议，不生成目标仓位、数量、券商订单或实盘执行命令。

## 快照

| 项目 | 数值 |
| --- | ---: |
| 状态 | `{snapshot.get("status")}` |
| 候选数量 | {snapshot.get("candidate_count", 0)} |
| 已成熟结果 | {snapshot.get("ready_outcome_count", 0)} |
| 待成熟结果 | {snapshot.get("pending_outcome_count", 0)} |
| 结果成熟度 | `{snapshot.get("outcome_readiness_status")}` |
| 已成熟周期 | `{snapshot.get("outcome_ready_horizons")}` |
| 待成熟周期 | `{snapshot.get("outcome_pending_horizons")}` |
| 最新行情日期 | {snapshot.get("latest_available_market_trade_date") or "N/A"} |
| 下一回看周期 | `{snapshot.get("next_outcome_review_horizon")}` |
| 下一回看原因 | `{snapshot.get("next_outcome_review_reason")}` |
| 成功阈值 | {_pct(snapshot.get("success_threshold"))} |
| 晋级门槛状态 | `{snapshot.get("promotion_gate_status")}` |
| 晋级门槛原因 | `{snapshot.get("promotion_gate_reasons")}` |
| 每周期最低成熟样本 | {snapshot.get("promotion_gate_min_ready_per_horizon")} |
| 最低成功率 | {_pct(snapshot.get("promotion_gate_min_success_rate"))} |
| 券商动作 | `{snapshot.get("broker_action")}` |

## 汇总

| 周期 | 已成熟 | 待成熟 | 成功数 | 成功率 | 平均收盘收益 |
| --- | ---: | ---: | ---: | ---: | ---: |
{rows}

## 文件

- 逐票结果 CSV：`{snapshot.get("outcomes_path")}`
- 汇总 CSV：`{snapshot.get("summary_path")}`
- 分组汇总 CSV：`{snapshot.get("group_summary_path")}`
- 快照 JSON：`{snapshot.get("snapshot_path")}`
"""


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not pd.notna(number):
        return "N/A"
    return f"{number * 100:.2f}%"


def run_chip_reversal_candidate_outcomes(
    project_root: str | Path = Path("."),
    candidates_path: str | Path | None = DEFAULT_CANDIDATES_PATH,
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    horizons: Iterable[int] = (1, 2),
    success_threshold: float = 0.03,
    min_ready_per_horizon: int = 30,
    min_success_rate: float = 0.55,
    market_snapshot_overlay: bool = False,
    daily_data_dir: str | Path | None = DEFAULT_DAILY_MARKET_DATA_DIR,
    ingest_project_dir: str | Path | None = DEFAULT_EXCHANGE_INGEST_DIR,
) -> ChipReversalCandidateOutcomesResult:
    root = Path(project_root).resolve()
    resolved_candidates = _resolve(root, candidates_path, DEFAULT_CANDIDATES_PATH)
    resolved_data = _resolve(root, data_dir, DEFAULT_DATA_DIR)
    resolved_output = _resolve(root, output_dir, DEFAULT_OUTPUT_DIR)
    candidates = _load_candidates(resolved_candidates)
    histories, missing_histories = _history_map(resolved_data, candidates["code"].tolist())
    market_payload: dict[str, Any] = {
        "market_snapshot_overlay": bool(market_snapshot_overlay),
        "market_source_kind": None,
        "market_source_path": None,
        "market_trade_date": None,
        "market_row_count": 0,
        "market_fetch_status": None,
        "market_fetch_message": None,
    }
    if market_snapshot_overlay:
        market_result = load_market_snapshot_rows(
            trade_date=None,
            daily_data_dir=daily_data_dir,
            ingest_project_dir=ingest_project_dir,
            require_success=True,
        )
        histories = _overlay_market_snapshot_rows(histories, market_result.rows, candidates["code"].tolist())
        missing_histories = [code for code in missing_histories if _clean_code(code) not in histories]
        fetch_status = market_result.fetch_status
        market_payload.update(
            {
                "market_source_kind": market_result.source_kind,
                "market_source_path": str(market_result.source_path) if market_result.source_path else None,
                "market_trade_date": market_result.trade_date,
                "market_row_count": len(market_result.rows),
                "market_fetch_status": getattr(fetch_status, "status", None),
                "market_fetch_message": getattr(fetch_status, "message", None),
            }
        )
    outcomes, summary, snapshot = build_chip_reversal_candidate_outcomes(
        candidates,
        histories,
        horizons=horizons,
        success_threshold=success_threshold,
        min_ready_per_horizon=min_ready_per_horizon,
        min_success_rate=min_success_rate,
    )
    readiness_payload = _outcome_readiness_snapshot(
        candidates,
        outcomes,
        market_payload.get("market_trade_date"),
    )
    resolved_output.mkdir(parents=True, exist_ok=True)
    outcomes_path = resolved_output / "chip_reversal_candidate_outcomes.csv"
    summary_path = resolved_output / "chip_reversal_candidate_outcome_summary.csv"
    group_summary_path = resolved_output / "chip_reversal_candidate_outcome_group_summary.csv"
    snapshot_path = resolved_output / "chip_reversal_candidate_outcomes_snapshot.json"
    report_path = resolved_output / "chip_reversal_candidate_outcomes.md"
    group_summary = build_outcome_group_summary(outcomes)
    outcomes.to_csv(outcomes_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    group_summary.to_csv(group_summary_path, index=False, encoding="utf-8-sig")
    snapshot = {
        **snapshot,
        **market_payload,
        **readiness_payload,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidates_path": str(resolved_candidates),
        "data_dir": str(resolved_data),
        "output_dir": str(resolved_output),
        "missing_history_count": int(len(missing_histories)),
        "missing_histories": missing_histories[:50],
        "outcomes_path": str(outcomes_path),
        "summary_path": str(summary_path),
        "group_summary_path": str(group_summary_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, summary), encoding="utf-8")
    return ChipReversalCandidateOutcomesResult(
        output_dir=resolved_output,
        outcomes_path=outcomes_path,
        summary_path=summary_path,
        group_summary_path=group_summary_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        outcomes=outcomes,
        summary=summary,
        group_summary=group_summary,
    )
