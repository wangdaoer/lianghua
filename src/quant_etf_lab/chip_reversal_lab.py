"""Research-only daily proxy lab for enhanced chip reversal setups.

The original idea depends on real chip distribution and 9:20-9:25 call-auction
order-flow confirmation. Those inputs are not available in the daily cache, so
this module keeps them explicit as unknown and evaluates only a daily proxy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_DATA_DIR = Path("data/processed/stocks")
DEFAULT_OUTPUT_DIR = Path("outputs/research/chip_reversal_lab_latest")


@dataclass(frozen=True)
class ChipReversalLabResult:
    output_dir: Path
    events_path: Path
    summary_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    events: pd.DataFrame
    summary: pd.DataFrame


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = Path(path) if path is not None else default
    return raw if raw.is_absolute() else project_root / raw


def _clean_code(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[-6:].zfill(6) if digits else text


def _board(code: str) -> str:
    clean = _clean_code(code)
    if clean.startswith(("300", "301")):
        return "chinext"
    if clean.startswith(("688", "689")):
        return "star"
    if clean.startswith(("4", "8", "920")):
        return "bse"
    if clean.startswith(("000", "001", "002", "003", "600", "601", "603", "605")):
        return "main"
    return "unknown"


def _include_board(board: str, board_scope: str) -> bool:
    if board_scope == "all":
        return board != "unknown"
    if board_scope == "main_chinext":
        return board in {"main", "chinext"}
    raise ValueError("board_scope must be one of: main_chinext, all")


def _read_price_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str})
    missing = {"date", "close"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    data = frame.copy()
    if "code" not in data.columns:
        data["code"] = path.stem[-6:].zfill(6)
    if "name" not in data.columns:
        data["name"] = data["code"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].map(_clean_code)
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        else:
            data[column] = pd.NA
    data["high"] = data["high"].fillna(data["close"])
    data["low"] = data["low"].fillna(data["close"])
    data["open"] = data["open"].fillna(data["close"])
    data["amount"] = data["amount"].fillna(0.0)
    data = data.dropna(subset=["date", "code", "close"])
    data = data[(data["close"] > 0) & (data["open"] > 0) & (data["high"] > 0)]
    data = data.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return data.reset_index(drop=True)


def _csv_paths(data_dir: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(path for path in data_dir.glob(pattern) if path.is_file())


def _score_bucket(score: Any, min_score: float) -> str:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "unknown"
    if not pd.notna(value):
        return "unknown"
    if value >= min_score * 2.0:
        return "deep"
    if value >= min_score * 1.5:
        return "strong"
    return "base"


def _validate_score_bucket(score_bucket: str) -> str:
    normalized = str(score_bucket or "all").strip().lower()
    if normalized not in {"all", "base", "strong", "deep"}:
        raise ValueError("score_bucket must be one of: all, base, strong, deep")
    return normalized


def _normalize_allowed_theme_states(allowed_theme_states: Iterable[str] | str | None) -> list[str]:
    if allowed_theme_states in (None, ""):
        raw_values = ["healthy", "second_activation"]
    elif isinstance(allowed_theme_states, str):
        raw_values = [item.strip() for item in allowed_theme_states.split(",")]
    else:
        raw_values = [str(item).strip() for item in allowed_theme_states]
    normalized = [value for value in raw_values if value]
    if not normalized:
        raise ValueError("allowed_theme_states must contain at least one state.")
    return normalized


def _read_theme_state_gate(theme_state_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(theme_state_path)
    required = {"date", "theme_group", "theme_state"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"theme_state_path missing columns: {sorted(missing)}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["theme_group"] = data["theme_group"].astype(str)
    data["theme_state"] = data["theme_state"].astype(str)
    keep_columns = [
        column
        for column in [
            "date",
            "theme_group",
            "theme_state",
            "theme_second_activation",
            "breadth_ma20_smooth",
            "breadth_slope",
            "theme_rs_20d",
            "weighted_breadth_ma20",
            "relative_return",
        ]
        if column in data.columns
    ]
    return data.dropna(subset=["date", "theme_group", "theme_state"]).reindex(columns=keep_columns)


def _read_theme_symbol_panel(theme_symbol_panel_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(theme_symbol_panel_path, dtype={"code": str})
    required = {"date", "code", "theme_group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"theme_symbol_panel_path missing columns: {sorted(missing)}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    data["code"] = data["code"].map(_clean_code)
    data["theme_group"] = data["theme_group"].astype(str)
    return data.dropna(subset=["date", "code", "theme_group"])[["date", "code", "theme_group"]].drop_duplicates()


def _state_rank(value: Any) -> int:
    return {"second_activation": 3, "healthy": 2, "neutral": 1, "weak": 0}.get(str(value), -1)


def _apply_theme_state_gate(
    events: pd.DataFrame,
    theme_state_path: str | Path | None = None,
    theme_symbol_panel_path: str | Path | None = None,
    allowed_theme_states: Iterable[str] | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    allowed = _normalize_allowed_theme_states(allowed_theme_states)
    payload: dict[str, Any] = {
        "applied": False,
        "allowed_theme_states": allowed,
        "theme_state_path": None,
        "theme_symbol_panel_path": None,
        "input_count": int(len(events)),
        "matched_count": int(len(events)),
        "passed_count": int(len(events)),
        "blocked_count": 0,
        "missing_count": 0,
    }
    if theme_state_path in (None, ""):
        return events.reset_index(drop=True), payload

    state_path = Path(theme_state_path)
    theme_state = _read_theme_state_gate(state_path)
    payload["applied"] = True
    payload["theme_state_path"] = str(state_path)
    if events.empty:
        payload["matched_count"] = 0
        payload["passed_count"] = 0
        return events.reset_index(drop=True), payload

    base = events.reset_index(drop=True).copy()
    base["event_row_id"] = base.index
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    base["code"] = base["code"].map(_clean_code)
    if theme_symbol_panel_path not in (None, ""):
        symbol_path = Path(theme_symbol_panel_path)
        symbol_panel = _read_theme_symbol_panel(symbol_path)
        payload["theme_symbol_panel_path"] = str(symbol_path)
        merged = base.merge(symbol_panel, on=["date", "code"], how="left", suffixes=("", "_theme_panel"))
    else:
        merged = base.copy()
        merged["theme_group"] = merged["board"].astype(str)

    merged = merged.merge(theme_state, on=["date", "theme_group"], how="left", suffixes=("", "_theme_state"))
    matched_ids = set(merged.loc[merged["theme_state"].notna(), "event_row_id"].astype(int).tolist())
    allowed_set = set(allowed)
    passed = merged.loc[merged["theme_state"].isin(allowed_set)].copy()
    passed_ids = set(passed["event_row_id"].astype(int).tolist())
    missing_count = int(len(base) - len(matched_ids))
    blocked_count = int(len(base) - len(passed_ids) - missing_count)
    payload.update(
        {
            "matched_count": int(len(matched_ids)),
            "passed_count": int(len(passed_ids)),
            "blocked_count": blocked_count,
            "missing_count": missing_count,
        }
    )
    if passed.empty:
        return pd.DataFrame(columns=[column for column in base.columns if column != "event_row_id"]), payload

    passed["theme_state_rank"] = passed["theme_state"].map(_state_rank)
    if "breadth_ma20_smooth" in passed.columns:
        passed["theme_breadth_rank"] = pd.to_numeric(passed["breadth_ma20_smooth"], errors="coerce").fillna(-1.0)
    else:
        passed["theme_breadth_rank"] = -1.0
    passed = passed.sort_values(["event_row_id", "theme_state_rank", "theme_breadth_rank"], ascending=[True, False, False])
    passed = passed.drop_duplicates(subset=["event_row_id"], keep="first").copy()
    passed["theme_gate_status"] = "passed"
    passed["allowed_theme_states"] = ",".join(allowed)
    passed = passed.drop(columns=["event_row_id", "theme_state_rank", "theme_breadth_rank"], errors="ignore")
    return passed.reset_index(drop=True), payload


def _apply_event_filters(
    events: pd.DataFrame,
    score_bucket: str = "all",
    min_amount_yi: float = 0.0,
    max_next_open_gap_pct: float | None = None,
    theme_state_path: str | Path | None = None,
    theme_symbol_panel_path: str | Path | None = None,
    allowed_theme_states: Iterable[str] | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized_bucket = _validate_score_bucket(score_bucket)
    if min_amount_yi < 0:
        raise ValueError("min_amount_yi must be non-negative.")
    if max_next_open_gap_pct is not None and max_next_open_gap_pct < 0:
        raise ValueError("max_next_open_gap_pct must be non-negative when provided.")

    filtered = events.copy()
    counts = {"raw_event_count": int(len(filtered))}
    if not filtered.empty and normalized_bucket != "all":
        filtered = filtered[filtered["score_bucket"].astype(str) == normalized_bucket].copy()
    counts["after_score_bucket"] = int(len(filtered))

    if not filtered.empty and min_amount_yi > 0:
        amount_yi = pd.to_numeric(filtered["amount_yi"], errors="coerce")
        filtered = filtered[amount_yi >= float(min_amount_yi)].copy()
    counts["after_min_amount_yi"] = int(len(filtered))

    if not filtered.empty and max_next_open_gap_pct is not None:
        open_gap = pd.to_numeric(filtered["open_gap_next"], errors="coerce")
        filtered = filtered[open_gap <= float(max_next_open_gap_pct) / 100.0].copy()
    counts["after_max_next_open_gap_pct"] = int(len(filtered))
    filtered, theme_gate = _apply_theme_state_gate(
        filtered,
        theme_state_path=theme_state_path,
        theme_symbol_panel_path=theme_symbol_panel_path,
        allowed_theme_states=allowed_theme_states,
    )
    counts["after_theme_state_gate"] = int(len(filtered))
    counts["retained_count"] = int(len(filtered))
    counts["filtered_out_count"] = int(counts["raw_event_count"] - counts["retained_count"])

    if not filtered.empty:
        filtered["execution_filter_status"] = "passed"
        filtered["score_bucket_filter"] = normalized_bucket
        filtered["min_amount_yi_filter"] = float(min_amount_yi)
        filtered["max_next_open_gap_pct_filter"] = max_next_open_gap_pct
    counts["theme_state_gate"] = theme_gate
    return filtered.reset_index(drop=True), counts


def build_chip_reversal_events(
    prices: pd.DataFrame,
    horizons: Iterable[int] = (1, 2),
    drawdown_window: int = 20,
    cost_window: int = 20,
    min_drawdown_pct: float = 12.0,
    min_score: float = 0.08,
    board_scope: str = "main_chinext",
    require_mature: bool = True,
) -> pd.DataFrame:
    """Build daily-proxy chip reversal events and attach mature forward labels.

    The signal columns use only data available at the signal day's close. Future
    returns are attached after event selection as labels for research.
    """
    if prices.empty:
        return pd.DataFrame()
    horizons_list = [int(horizon) for horizon in horizons]
    if not horizons_list or any(horizon <= 0 for horizon in horizons_list):
        raise ValueError("horizons must be positive integers.")
    if drawdown_window <= 1:
        raise ValueError("drawdown_window must be greater than 1.")
    if cost_window <= 1:
        raise ValueError("cost_window must be greater than 1.")
    if min_drawdown_pct <= 0:
        raise ValueError("min_drawdown_pct must be positive.")
    if min_score < 0:
        raise ValueError("min_score must be non-negative.")

    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].map(_clean_code) if "code" in data.columns else ""
    if "name" not in data.columns:
        data["name"] = data["code"]
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        else:
            data[column] = pd.NA
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["open"] = data["open"].fillna(data["close"])
    data["high"] = data["high"].fillna(data["close"])
    data["low"] = data["low"].fillna(data["close"])
    data["amount"] = data["amount"].fillna(0.0)
    data = data.dropna(subset=["date", "code", "close", "open", "high"])
    data = data[(data["close"] > 0) & (data["open"] > 0) & (data["high"] > 0)]
    data = data.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    if data.empty:
        return pd.DataFrame()

    code = str(data["code"].iloc[0])
    board = _board(code)
    if not _include_board(board, board_scope):
        return pd.DataFrame()

    drawdown_column = f"drawdown_{drawdown_window}d"
    cost_gap_column = f"cost_gap_ma{cost_window}"
    ma_column = f"ma{cost_window}"
    prior_high_column = f"prior_high_{drawdown_window}d"

    data[prior_high_column] = data["high"].rolling(drawdown_window, min_periods=2).max().shift(1)
    data[drawdown_column] = data["close"] / data[prior_high_column] - 1.0
    data[ma_column] = data["close"].rolling(cost_window, min_periods=cost_window).mean()
    data[cost_gap_column] = data["close"] / data[ma_column] - 1.0
    drawdown_depth = (-data[drawdown_column]).clip(lower=0.0)
    cost_depth = (-data[cost_gap_column]).clip(lower=0.0)
    data["chip_reversal_score"] = drawdown_depth * 0.6 + cost_depth * 0.4
    data["next_open"] = data["open"].shift(-1)
    data["open_gap_next"] = data["next_open"] / data["close"] - 1.0
    for horizon in horizons_list:
        data[f"return_{horizon}d"] = data["close"].shift(-horizon) / data["close"] - 1.0
        data[f"trade_return_{horizon}d"] = data["close"].shift(-horizon) / data["next_open"] - 1.0

    max_horizon = max(horizons_list)
    mature_mask = data.index.to_series() + max_horizon < len(data) if require_mature else pd.Series(True, index=data.index)
    drawdown_threshold = -float(min_drawdown_pct) / 100.0
    signal_mask = (
        (data[drawdown_column] <= drawdown_threshold)
        & (data[cost_gap_column] < 0.0)
        & (data["chip_reversal_score"] >= float(min_score))
        & mature_mask
    )
    events = data.loc[signal_mask].copy()
    if events.empty:
        return pd.DataFrame()

    events["date"] = events["date"].dt.strftime("%Y-%m-%d")
    events["board"] = board
    events["signal_type"] = "chip_reversal_daily_proxy"
    events["score_bucket"] = events["chip_reversal_score"].map(lambda value: _score_bucket(value, float(min_score)))
    events["amount_yi"] = events["amount"] / 100_000_000.0
    events["auction_confirmation_status"] = "unknown"
    events["auction_data_required"] = "9:20-9:25 call auction order imbalance unavailable"
    events["proxy_confirmation"] = "daily_only"
    events["hold_horizon_preference"] = "1d_2d_test_only"
    events["broker_action"] = "none"
    events["research_only"] = True

    ordered_columns = [
        "date",
        "code",
        "name",
        "board",
        "signal_type",
        "score_bucket",
        "chip_reversal_score",
        prior_high_column,
        drawdown_column,
        ma_column,
        cost_gap_column,
        "close",
        "next_open",
        "open_gap_next",
        "amount",
        "amount_yi",
        "auction_confirmation_status",
        "auction_data_required",
        "proxy_confirmation",
        "hold_horizon_preference",
        "broker_action",
        "research_only",
    ]
    for horizon in horizons_list:
        ordered_columns.extend([f"return_{horizon}d", f"trade_return_{horizon}d"])
    events = events.reindex(columns=ordered_columns).reset_index(drop=True)
    events["research_only"] = events["research_only"].astype(object)
    return events


def _summarize_events(events: pd.DataFrame, horizons: list[int], min_events: int) -> pd.DataFrame:
    columns = [
        "signal_type",
        "board",
        "score_bucket",
        "horizon",
        "event_count",
        "win_rate",
        "avg_return",
        "median_return",
        "avg_trade_return",
        "payoff_ratio",
        "profit_factor",
        "avg_score",
        "avg_drawdown",
        "avg_amount_yi",
        "broker_action",
        "research_only",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    group_specs = [
        ["signal_type"],
        ["signal_type", "board"],
        ["signal_type", "score_bucket"],
        ["signal_type", "board", "score_bucket"],
    ]
    drawdown_columns = [column for column in events.columns if column.startswith("drawdown_")]
    drawdown_column = drawdown_columns[0] if drawdown_columns else None
    for group_columns in group_specs:
        for group_key, group in events.groupby(group_columns, dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_values = dict(zip(group_columns, group_key))
            for horizon in horizons:
                close_returns = pd.to_numeric(group.get(f"return_{horizon}d"), errors="coerce").dropna()
                trade_returns = pd.to_numeric(group.get(f"trade_return_{horizon}d"), errors="coerce").dropna()
                if len(trade_returns) < int(min_events):
                    continue
                gains = trade_returns[trade_returns > 0]
                losses = trade_returns[trade_returns < 0]
                avg_gain = float(gains.mean()) if not gains.empty else 0.0
                avg_loss = float(losses.mean()) if not losses.empty else 0.0
                rows.append(
                    {
                        "signal_type": str(group_values.get("signal_type", "all")),
                        "board": str(group_values.get("board", "all")),
                        "score_bucket": str(group_values.get("score_bucket", "all")),
                        "horizon": int(horizon),
                        "event_count": int(len(trade_returns)),
                        "win_rate": float((trade_returns > 0).mean()),
                        "avg_return": float(close_returns.mean()) if not close_returns.empty else 0.0,
                        "median_return": float(close_returns.median()) if not close_returns.empty else 0.0,
                        "avg_trade_return": float(trade_returns.mean()),
                        "payoff_ratio": avg_gain / abs(avg_loss) if avg_loss else 0.0,
                        "profit_factor": float(gains.sum()) / abs(float(losses.sum())) if not losses.empty else 0.0,
                        "avg_score": float(pd.to_numeric(group["chip_reversal_score"], errors="coerce").mean()),
                        "avg_drawdown": float(pd.to_numeric(group[drawdown_column], errors="coerce").mean())
                        if drawdown_column
                        else 0.0,
                        "avg_amount_yi": float(pd.to_numeric(group["amount_yi"], errors="coerce").mean()),
                        "broker_action": "none",
                        "research_only": True,
                    }
                )
    summary = pd.DataFrame(rows, columns=columns)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["horizon", "avg_trade_return", "win_rate", "event_count"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def _pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not pd.notna(number):
        return "N/A"
    return f"{number * 100:.2f}%"


def _render_report(snapshot: dict[str, Any], summary: pd.DataFrame) -> str:
    if summary.empty:
        rows = "| `chip_reversal_daily_proxy` | `all` | `all` | 0 | 0 | N/A | N/A | N/A |"
    else:
        rows = "\n".join(
            "| `{signal}` | `{board}` | `{bucket}` | {horizon} | {count} | {win} | {avg} | {score:.4f} |".format(
                signal=row.get("signal_type"),
                board=row.get("board"),
                bucket=row.get("score_bucket"),
                horizon=int(row.get("horizon") or 0),
                count=int(row.get("event_count") or 0),
                win=_pct(row.get("win_rate")),
                avg=_pct(row.get("avg_trade_return")),
                score=float(row.get("avg_score") or 0.0),
            )
            for _, row in summary.iterrows()
        )
    return f"""# Enhanced Chip Reversal Daily Proxy Lab

Generated at: `{snapshot.get("generated_at")}`

This is a research-only daily proxy for `chip_reversal_daily_proxy`. Real chip distribution and 9:20-9:25 call-auction confirmation are not available in the current daily cache, so auction fields remain `unknown`.

## Snapshot

| Item | Value |
| --- | ---: |
| Status | `{snapshot.get("status")}` |
| Event count | {snapshot.get("event_count", 0)} |
| Summary rows | {snapshot.get("summary_row_count", 0)} |
| Horizons | `{snapshot.get("horizons")}` |
| Drawdown window | {snapshot.get("drawdown_window")} |
| Cost window | {snapshot.get("cost_window")} |
| Min drawdown pct | {snapshot.get("min_drawdown_pct")} |
| Min score | {snapshot.get("min_score")} |
| Score bucket filter | `{snapshot.get("score_bucket_filter")}` |
| Min amount(Yi) | {snapshot.get("min_amount_yi")} |
| Max next open gap pct | {snapshot.get("max_next_open_gap_pct")} |
| Theme gate applied | `{snapshot.get("theme_state_gate", {}).get("applied")}` |
| Theme gate passed | {snapshot.get("theme_state_gate", {}).get("passed_count", "N/A")} |
| Allowed theme states | `{snapshot.get("allowed_theme_states")}` |
| Broker action | `{snapshot.get("broker_action")}` |

## Group Outcomes

| Signal | Board | Score bucket | Horizon | Events | Win rate | Avg trade return | Avg score |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
{rows}

## Data Limits

- Auction confirmation status: `unknown`
- Proxy confirmation: `daily_only`
- Broker/order action: `none`

## Files

- Events CSV: `{snapshot.get("events_path")}`
- Summary CSV: `{snapshot.get("summary_path")}`
- Snapshot JSON: `{snapshot.get("snapshot_path")}`
"""


def run_chip_reversal_lab(
    project_root: str | Path = Path("."),
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    horizons: Iterable[int] = (1, 2),
    drawdown_window: int = 20,
    cost_window: int = 20,
    min_drawdown_pct: float = 12.0,
    min_score: float = 0.08,
    score_bucket: str = "all",
    min_amount_yi: float = 0.0,
    max_next_open_gap_pct: float | None = None,
    theme_state_path: str | Path | None = None,
    theme_symbol_panel_path: str | Path | None = None,
    allowed_theme_states: Iterable[str] | str | None = None,
    board_scope: str = "main_chinext",
    min_events: int = 20,
    max_symbols: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    recursive: bool = False,
) -> ChipReversalLabResult:
    root = Path(project_root).resolve()
    resolved_data = _resolve(root, data_dir, DEFAULT_DATA_DIR)
    resolved_output = _resolve(root, output_dir, DEFAULT_OUTPUT_DIR)
    horizons_list = [int(horizon) for horizon in horizons]
    start = pd.to_datetime(start_date, errors="coerce") if start_date else None
    end = pd.to_datetime(end_date, errors="coerce") if end_date else None
    if start is not None and pd.isna(start):
        raise ValueError(f"Invalid start_date: {start_date}")
    if end is not None and pd.isna(end):
        raise ValueError(f"Invalid end_date: {end_date}")
    normalized_score_bucket = _validate_score_bucket(score_bucket)
    if min_amount_yi < 0:
        raise ValueError("min_amount_yi must be non-negative.")
    if max_next_open_gap_pct is not None and max_next_open_gap_pct < 0:
        raise ValueError("max_next_open_gap_pct must be non-negative when provided.")
    normalized_allowed_theme_states = _normalize_allowed_theme_states(allowed_theme_states)

    paths = _csv_paths(resolved_data, recursive=recursive)
    if max_symbols is not None and max_symbols > 0:
        paths = paths[: int(max_symbols)]

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            prices = _read_price_csv(path)
            events = build_chip_reversal_events(
                prices,
                horizons=horizons_list,
                drawdown_window=drawdown_window,
                cost_window=cost_window,
                min_drawdown_pct=min_drawdown_pct,
                min_score=min_score,
                board_scope=board_scope,
            )
            if start is not None and not events.empty:
                events = events[pd.to_datetime(events["date"], errors="coerce") >= pd.Timestamp(start)]
            if end is not None and not events.empty:
                events = events[pd.to_datetime(events["date"], errors="coerce") <= pd.Timestamp(end)]
            if not events.empty:
                frames.append(events)
        except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
            failures.append({"path": str(path), "error": str(error)})

    raw_events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not raw_events.empty:
        raw_events = raw_events.sort_values(["date", "code"]).reset_index(drop=True)
    events, filter_counts = _apply_event_filters(
        raw_events,
        score_bucket=normalized_score_bucket,
        min_amount_yi=float(min_amount_yi),
        max_next_open_gap_pct=max_next_open_gap_pct,
        theme_state_path=theme_state_path,
        theme_symbol_panel_path=theme_symbol_panel_path,
        allowed_theme_states=normalized_allowed_theme_states,
    )
    if not events.empty:
        events = events.sort_values(["date", "code"]).reset_index(drop=True)
    summary = _summarize_events(events, horizons=horizons_list, min_events=min_events)

    resolved_output.mkdir(parents=True, exist_ok=True)
    events_path = resolved_output / "chip_reversal_events.csv"
    summary_path = resolved_output / "chip_reversal_summary.csv"
    snapshot_path = resolved_output / "chip_reversal_snapshot.json"
    report_path = resolved_output / "chip_reversal_lab.md"
    events.to_csv(events_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok" if not events.empty else "no_events",
        "data_dir": str(resolved_data),
        "output_dir": str(resolved_output),
        "file_count": len(paths),
        "loaded_symbol_count": int(events["code"].nunique()) if not events.empty else 0,
        "raw_event_count": int(len(raw_events)),
        "event_count": int(len(events)),
        "summary_row_count": int(len(summary)),
        "start_date": pd.Timestamp(start).strftime("%Y-%m-%d") if start is not None else None,
        "end_date": pd.Timestamp(end).strftime("%Y-%m-%d") if end is not None else None,
        "horizons": horizons_list,
        "drawdown_window": int(drawdown_window),
        "cost_window": int(cost_window),
        "min_drawdown_pct": float(min_drawdown_pct),
        "min_score": float(min_score),
        "score_bucket_filter": normalized_score_bucket,
        "min_amount_yi": float(min_amount_yi),
        "max_next_open_gap_pct": None if max_next_open_gap_pct is None else float(max_next_open_gap_pct),
        "filter_counts": filter_counts,
        "theme_state_gate": filter_counts.get("theme_state_gate", {}),
        "theme_state_path": None if theme_state_path in (None, "") else str(Path(theme_state_path)),
        "theme_symbol_panel_path": None
        if theme_symbol_panel_path in (None, "")
        else str(Path(theme_symbol_panel_path)),
        "allowed_theme_states": normalized_allowed_theme_states,
        "board_scope": board_scope,
        "min_events": int(min_events),
        "failure_count": len(failures),
        "failures": failures[:20],
        "events_path": str(events_path),
        "summary_path": str(summary_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "events": _json_records(events.head(100)),
        "auction_confirmation_status": "unknown",
        "proxy_confirmation": "daily_only",
        "research_only": True,
        "broker_action": "none",
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, summary), encoding="utf-8")

    return ChipReversalLabResult(
        output_dir=resolved_output,
        events_path=events_path,
        summary_path=summary_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        events=events,
        summary=summary,
    )
