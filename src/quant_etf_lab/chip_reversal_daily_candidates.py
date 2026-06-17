"""Latest-date observation candidates for chip-reversal research."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .chip_reversal_lab import (
    DEFAULT_DATA_DIR,
    _clean_code,
    _csv_paths,
    _read_price_csv,
    _resolve,
    _validate_score_bucket,
    build_chip_reversal_events,
)
from .market_data_source import (
    DEFAULT_DAILY_MARKET_DATA_DIR,
    DEFAULT_EXCHANGE_INGEST_DIR,
    load_market_snapshot_rows,
)


DEFAULT_OUTPUT_DIR = Path("outputs/research/chip_reversal_daily_candidates_latest")
FUTURE_LABEL_PREFIXES = ("return_", "trade_return_")
FUTURE_LABEL_COLUMNS = {"next_open", "open_gap_next"}


@dataclass(frozen=True)
class ChipReversalDailyCandidatesResult:
    output_dir: Path
    candidates_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    candidates: pd.DataFrame


def _normalize_date(value: str | pd.Timestamp | None) -> str | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid as_of_date: {value}")
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _latest_price_date(prices: pd.DataFrame) -> str | None:
    if prices.empty or "date" not in prices.columns:
        return None
    dates = pd.to_datetime(prices["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).strftime("%Y-%m-%d")


def _future_label_columns(events: pd.DataFrame) -> list[str]:
    return [
        column
        for column in events.columns
        if column in FUTURE_LABEL_COLUMNS or column.startswith(FUTURE_LABEL_PREFIXES)
    ]


def _first_existing_column(frame: pd.DataFrame, prefix: str) -> str | None:
    return next((column for column in frame.columns if column.startswith(prefix)), None)


def _as_numeric(frame: pd.DataFrame, column: str | None, default: float = 0.0) -> pd.Series:
    if column is None or column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


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


def _snapshot_row_to_price(row: dict[str, Any]) -> dict[str, Any] | None:
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


def _overlay_market_snapshot_rows(price_frames: list[pd.DataFrame], rows: list[dict[str, Any]]) -> list[pd.DataFrame]:
    snapshot_by_code: dict[str, dict[str, Any]] = {}
    for row in rows:
        price_row = _snapshot_row_to_price(row)
        if price_row is not None:
            snapshot_by_code[str(price_row["code"])] = price_row

    overlaid: list[pd.DataFrame] = []
    for frame in price_frames:
        if frame.empty or "code" not in frame.columns:
            overlaid.append(frame)
            continue
        code = _clean_code(frame["code"].dropna().iloc[0]) if not frame["code"].dropna().empty else ""
        price_row = snapshot_by_code.get(code)
        if price_row is None:
            overlaid.append(frame)
            continue
        if price_row.get("name") == code and "name" in frame.columns and not frame["name"].dropna().empty:
            price_row = {**price_row, "name": str(frame["name"].dropna().iloc[-1])}
        data = frame.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        row_date = pd.Timestamp(price_row["date"])
        data = data[data["date"] != row_date].copy()
        patched = pd.concat([data, pd.DataFrame([price_row])], ignore_index=True)
        patched["date"] = pd.to_datetime(patched["date"], errors="coerce")
        overlaid.append(patched.sort_values("date").reset_index(drop=True))
    return overlaid


def _rank_selector_only(source: pd.DataFrame) -> pd.DataFrame:
    ranked = source.copy()
    drawdown_column = _first_existing_column(ranked, "drawdown_")
    cost_gap_column = _first_existing_column(ranked, "cost_gap_")
    score = _as_numeric(ranked, "chip_reversal_score")
    drawdown_depth = (-_as_numeric(ranked, drawdown_column)).clip(lower=0.0)
    cost_gap_depth = (-_as_numeric(ranked, cost_gap_column)).clip(lower=0.0)
    amount_yi = _as_numeric(ranked, "amount_yi")
    liquidity_component = amount_yi.clip(lower=0.0, upper=20.0) / 20.0
    theme_component = pd.Series(0.0, index=ranked.index)
    if "theme_state" in ranked.columns:
        theme_component = ranked["theme_state"].map({"second_activation": 0.08, "healthy": 0.05, "neutral": 0.02}).fillna(0.0)

    ranked["priority_score"] = (
        score * 100.0
        + drawdown_depth * 45.0
        + cost_gap_depth * 25.0
        + liquidity_component * 10.0
        + theme_component * 100.0
    )
    return ranked.sort_values(["priority_score", "chip_reversal_score", "amount_yi"], ascending=[False, False, False])


def _empty_snapshot(
    status: str,
    as_of_date: str | None,
    source_symbol_count: int,
    raw_event_count: int,
    future_columns: list[str],
) -> dict[str, Any]:
    return {
        "status": status,
        "as_of_date": as_of_date,
        "source_symbol_count": int(source_symbol_count),
        "raw_event_count": int(raw_event_count),
        "as_of_event_count": 0,
        "candidate_count": 0,
        "future_label_columns_removed": future_columns,
        "ranking_uses_forward_labels": False,
        "broker_action": "none",
        "research_only": True,
    }


def build_chip_reversal_daily_candidates(
    price_frames: Iterable[pd.DataFrame],
    as_of_date: str | None = None,
    horizons: Iterable[int] = (1, 2),
    drawdown_window: int = 20,
    cost_window: int = 20,
    min_drawdown_pct: float = 12.0,
    min_score: float = 0.08,
    score_bucket: str = "deep",
    min_amount_yi: float = 1.5,
    board_scope: str = "main_chinext",
    max_candidates: int = 50,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build latest-date, non-mature, observation-only candidates.

    The selection may use today's close-based proxy fields, but it never keeps
    next-open or future-return labels in the candidate output.
    """
    normalized_bucket = _validate_score_bucket(score_bucket)
    if min_amount_yi < 0:
        raise ValueError("min_amount_yi must be non-negative.")
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative.")

    frames = list(price_frames)
    latest_dates = [date for date in (_latest_price_date(frame) for frame in frames) if date is not None]
    selected_date = _normalize_date(as_of_date) if as_of_date else (max(latest_dates) if latest_dates else None)
    events_parts: list[pd.DataFrame] = []
    for prices in frames:
        events = build_chip_reversal_events(
            prices,
            horizons=horizons,
            drawdown_window=drawdown_window,
            cost_window=cost_window,
            min_drawdown_pct=min_drawdown_pct,
            min_score=min_score,
            board_scope=board_scope,
            require_mature=False,
        )
        if not events.empty:
            events_parts.append(events)

    raw_events = pd.concat(events_parts, ignore_index=True) if events_parts else pd.DataFrame()
    future_columns = _future_label_columns(raw_events)
    if raw_events.empty or selected_date is None:
        return pd.DataFrame(), _empty_snapshot("no_events", selected_date, len(frames), len(raw_events), future_columns)

    data = raw_events.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["code"] = data["code"].map(_clean_code) if "code" in data.columns else ""
    source = data[data["date"] == selected_date].copy()
    as_of_event_count = int(len(source))
    if source.empty:
        return pd.DataFrame(), {
            **_empty_snapshot("no_candidates", selected_date, len(frames), len(raw_events), future_columns),
            "as_of_event_count": as_of_event_count,
        }

    if normalized_bucket != "all":
        source = source[source["score_bucket"].astype(str) == normalized_bucket].copy()
    after_score_bucket = int(len(source))
    if not source.empty and min_amount_yi > 0:
        source = source[_as_numeric(source, "amount_yi") >= float(min_amount_yi)].copy()
    after_min_amount_yi = int(len(source))
    if source.empty:
        return pd.DataFrame(), {
            **_empty_snapshot("no_candidates", selected_date, len(frames), len(raw_events), future_columns),
            "as_of_event_count": as_of_event_count,
            "after_score_bucket": after_score_bucket,
            "after_min_amount_yi": after_min_amount_yi,
            "score_bucket_filter": normalized_bucket,
            "min_amount_yi": float(min_amount_yi),
        }

    source = _rank_selector_only(source)
    if max_candidates > 0:
        source = source.head(int(max_candidates)).copy()
    source["candidate_rank"] = range(1, len(source) + 1)
    source["source_sleeve"] = "chip_reversal_daily_proxy"
    source["watch_posture"] = "watch_only"
    source["review_status"] = "pending_observation"
    source["paper_queue_status"] = "observation_only_not_order"
    source["label_status"] = "not_available_non_mature"
    source["timing_contract"] = "close_signal_next_session_observation"
    source["recommended_review"] = (
        "Review next-session open strength, intraday recovery, and 1D/2D follow-up after labels become available."
    )
    source["invalidation_condition"] = "Next session opens weak and fails to reclaim signal-day close."
    source["broker_action"] = "none"
    source["research_only"] = True
    source["future_label_columns_removed"] = ",".join(future_columns)

    drawdown_column = _first_existing_column(source, "drawdown_")
    cost_gap_column = _first_existing_column(source, "cost_gap_")
    keep = [
        "candidate_rank",
        "date",
        "code",
        "name",
        "board",
        "source_sleeve",
        "watch_posture",
        "review_status",
        "paper_queue_status",
        "label_status",
        "priority_score",
        "score_bucket",
        "chip_reversal_score",
        drawdown_column,
        cost_gap_column,
        "amount_yi",
        "close",
        "theme_group",
        "theme_state",
        "auction_confirmation_status",
        "auction_data_required",
        "proxy_confirmation",
        "timing_contract",
        "recommended_review",
        "invalidation_condition",
        "future_label_columns_removed",
        "broker_action",
        "research_only",
    ]
    keep = [column for column in keep if column is not None and column in source.columns]
    candidates = source.reindex(columns=keep).reset_index(drop=True)
    if not candidates.empty:
        candidates["research_only"] = candidates["research_only"].astype(object)

    snapshot = {
        "status": "ok",
        "as_of_date": selected_date,
        "source_symbol_count": int(len(frames)),
        "raw_event_count": int(len(raw_events)),
        "as_of_event_count": as_of_event_count,
        "after_score_bucket": after_score_bucket,
        "after_min_amount_yi": after_min_amount_yi,
        "candidate_count": int(len(candidates)),
        "max_candidates": int(max_candidates),
        "horizons": [int(horizon) for horizon in horizons],
        "drawdown_window": int(drawdown_window),
        "cost_window": int(cost_window),
        "min_drawdown_pct": float(min_drawdown_pct),
        "min_score": float(min_score),
        "score_bucket_filter": normalized_bucket,
        "min_amount_yi": float(min_amount_yi),
        "board_scope": board_scope,
        "future_label_columns_removed": future_columns,
        "ranking_uses_forward_labels": False,
        "auction_confirmation_status": "unknown",
        "proxy_confirmation": "daily_only",
        "broker_action": "none",
        "research_only": True,
    }
    return candidates, snapshot


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def _render_report(snapshot: dict[str, Any], candidates: pd.DataFrame) -> str:
    if candidates.empty:
        rows = "| N/A | N/A | N/A | N/A | N/A |"
    else:
        rows = "\n".join(
            "| {rank} | `{code}` | {name} | {score:.2f} | `{posture}` |".format(
                rank=int(row.get("candidate_rank") or 0),
                code=row.get("code"),
                name=row.get("name"),
                score=float(row.get("priority_score") or 0.0),
                posture=row.get("watch_posture"),
            )
            for _, row in candidates.iterrows()
        )
    return f"""# 筹码反转每日候选

生成时间：`{snapshot.get("generated_at")}`

这是基于收盘后日线代理信号生成的研究候选清单，标签尚未成熟。它不是买卖建议，不写入目标仓位、数量、券商订单或任何实盘执行命令。

## 快照

| 项目 | 数值 |
| --- | ---: |
| 状态 | `{snapshot.get("status")}` |
| 截止日期 | `{snapshot.get("as_of_date")}` |
| 候选数量 | {snapshot.get("candidate_count", 0)} |
| 来源股票数 | {snapshot.get("source_symbol_count", 0)} |
| 原始事件数 | {snapshot.get("raw_event_count", 0)} |
| 当日事件数 | {snapshot.get("as_of_event_count", 0)} |
| 已移除未来标签 | `{snapshot.get("future_label_columns_removed")}` |
| 排名使用未来标签 | `{snapshot.get("ranking_uses_forward_labels")}` |
| 券商动作 | `{snapshot.get("broker_action")}` |

## 候选清单

| 排名 | 代码 | 名称 | 优先级 | 姿态 |
| ---: | --- | --- | ---: | --- |
{rows}

## 时点约束

- 信号来源：收盘后日线代理信号。
- 标签状态：`not_available_non_mature`。
- 集合竞价确认：`unknown`。
- 券商/订单动作：`none`。

## 文件

- 候选 CSV：`{snapshot.get("candidates_path")}`
- 快照 JSON：`{snapshot.get("snapshot_path")}`
"""


def run_chip_reversal_daily_candidates(
    project_root: str | Path = Path("."),
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    as_of_date: str | None = None,
    horizons: Iterable[int] = (1, 2),
    drawdown_window: int = 20,
    cost_window: int = 20,
    min_drawdown_pct: float = 12.0,
    min_score: float = 0.08,
    score_bucket: str = "deep",
    min_amount_yi: float = 1.5,
    board_scope: str = "main_chinext",
    max_candidates: int = 50,
    max_symbols: int | None = None,
    recursive: bool = False,
    market_snapshot_overlay: bool = False,
    daily_data_dir: str | Path | None = DEFAULT_DAILY_MARKET_DATA_DIR,
    ingest_project_dir: str | Path | None = DEFAULT_EXCHANGE_INGEST_DIR,
) -> ChipReversalDailyCandidatesResult:
    root = Path(project_root).resolve()
    resolved_data = _resolve(root, data_dir, DEFAULT_DATA_DIR)
    resolved_output = _resolve(root, output_dir, DEFAULT_OUTPUT_DIR)
    paths = _csv_paths(resolved_data, recursive=recursive)
    if max_symbols is not None and max_symbols > 0:
        paths = paths[: int(max_symbols)]

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            frames.append(_read_price_csv(path))
        except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            failures.append({"path": str(path), "error": str(error)})

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
            trade_date=as_of_date,
            daily_data_dir=daily_data_dir,
            ingest_project_dir=ingest_project_dir,
            require_success=True,
        )
        frames = _overlay_market_snapshot_rows(frames, market_result.rows)
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

    candidates, snapshot = build_chip_reversal_daily_candidates(
        frames,
        as_of_date=as_of_date,
        horizons=horizons,
        drawdown_window=drawdown_window,
        cost_window=cost_window,
        min_drawdown_pct=min_drawdown_pct,
        min_score=min_score,
        score_bucket=score_bucket,
        min_amount_yi=min_amount_yi,
        board_scope=board_scope,
        max_candidates=max_candidates,
    )

    resolved_output.mkdir(parents=True, exist_ok=True)
    candidates_path = resolved_output / "chip_reversal_daily_candidates.csv"
    snapshot_path = resolved_output / "chip_reversal_daily_candidates_snapshot.json"
    report_path = resolved_output / "chip_reversal_daily_candidates.md"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    snapshot = {
        **snapshot,
        **market_payload,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(resolved_data),
        "output_dir": str(resolved_output),
        "file_count": len(paths),
        "failure_count": len(failures),
        "failures": failures[:20],
        "candidates_path": str(candidates_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "candidates": _json_records(candidates.head(100)),
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, candidates), encoding="utf-8")
    return ChipReversalDailyCandidatesResult(
        output_dir=resolved_output,
        candidates_path=candidates_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        candidates=candidates,
    )
