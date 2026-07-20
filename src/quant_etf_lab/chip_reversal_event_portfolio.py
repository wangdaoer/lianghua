"""Research-only event portfolio curve for chip-reversal proxy events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EVENTS_PATH = Path("outputs/research/chip_reversal_lab_latest/chip_reversal_events.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/chip_reversal_event_portfolio_latest")
DEFAULT_INITIAL_CASH = 1_000_000.0
DEFAULT_ROUND_TRIP_COST = 0.0016

SELECTOR_COLUMNS = (
    "chip_reversal_score",
    "score_bucket",
    "board",
    "close",
    "prior_high_20d",
    "drawdown_20d",
    "ma20",
    "cost_gap_ma20",
)
FUTURE_LABEL_PREFIXES = ("return_", "trade_return_")
FUTURE_LABEL_COLUMNS = {"next_open", "open_gap_next"}


@dataclass(frozen=True)
class ChipReversalEventPortfolioBuild:
    curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]
    selector_columns: list[str]
    future_label_columns_excluded: list[str]


@dataclass(frozen=True)
class ChipReversalEventPortfolioResult:
    output_dir: Path
    curve_path: Path
    trades_path: Path
    daily_path: Path
    metrics_path: Path
    snapshot_path: Path
    report_path: Path
    curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]


def _clean_code(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        return ""
    return digits.zfill(6)[-6:]


def _trade_return_column(horizon: int) -> str:
    return f"trade_return_{int(horizon)}d"


def _normalize_date(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid date: {value}")
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _include_board(board: Any, board_scope: str) -> bool:
    text = str(board)
    if board_scope == "main_chinext":
        return text in {"main", "chinext"}
    if board_scope == "all":
        return text not in {"", "unknown", "nan"}
    raise ValueError("board_scope must be one of: main_chinext, all")


def _apply_signal_quality_filters(
    data: pd.DataFrame,
    *,
    max_cost_gap_ma20: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_count = int(len(data))
    filtered = data.copy()
    params: dict[str, Any] = {
        "cost_gap_ma20_max": None if max_cost_gap_ma20 is None else float(max_cost_gap_ma20),
        "signal_quality_filter_input_count": input_count,
    }
    if max_cost_gap_ma20 is not None:
        if "cost_gap_ma20" not in filtered.columns:
            raise ValueError("max_cost_gap_ma20 requires a cost_gap_ma20 column.")
        filtered["cost_gap_ma20"] = pd.to_numeric(filtered["cost_gap_ma20"], errors="coerce")
        filtered = filtered[filtered["cost_gap_ma20"] <= float(max_cost_gap_ma20)].copy()
    kept_count = int(len(filtered))
    params["signal_quality_filter_kept_count"] = kept_count
    params["signal_quality_filter_removed_count"] = input_count - kept_count
    return filtered, params


def _future_label_columns(columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if column in FUTURE_LABEL_COLUMNS or column.startswith(FUTURE_LABEL_PREFIXES)
    ]


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    drawdown = series / series.cummax() - 1.0
    return float(drawdown.min())


def _prepare_price_history(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "date" not in data.columns:
        return pd.DataFrame(columns=["date", "open", "close"])
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ["open", "close"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        else:
            data[column] = pd.NA
    data = data.dropna(subset=["date", "open", "close"])
    data = data[(data["open"] > 0) & (data["close"] > 0)].copy()
    if data.empty:
        return pd.DataFrame(columns=["date", "open", "close"])
    return data.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def _load_price_histories(codes: pd.Series, data_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(data_dir)
    histories: dict[str, pd.DataFrame] = {}
    for code in sorted({_clean_code(value) for value in codes.dropna().tolist()}):
        if not code:
            continue
        path = root / f"{code}.csv"
        if not path.exists():
            continue
        try:
            histories[code] = pd.read_csv(path, usecols=["date", "open", "close"], dtype={"code": str})
        except ValueError:
            histories[code] = pd.read_csv(path, dtype={"code": str})
    return histories


def _position_metrics(
    curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    if curve.empty:
        return {
            **params,
            "initial_cash": float(initial_cash),
            "final_equity": float(initial_cash),
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": None,
            "active_day_ratio": 0.0,
            "average_exposure": 0.0,
            "selected_event_count": 0,
            "completed_trade_count": 0,
            "research_only": True,
            "broker_action": "none",
        }
    equity = pd.to_numeric(curve["equity"], errors="coerce").dropna()
    returns = pd.to_numeric(curve["daily_return"], errors="coerce").fillna(0.0)
    dates = pd.to_datetime(curve["date"], errors="coerce").dropna()
    years = max((dates.max() - dates.min()).days / 365.25, 0.0) if not dates.empty else 0.0
    final_equity = float(equity.iloc[-1]) if not equity.empty else float(initial_cash)
    total_return = final_equity / float(initial_cash) - 1.0 if initial_cash > 0 else 0.0
    cagr = (final_equity / initial_cash) ** (1.0 / years) - 1.0 if years > 0 and final_equity > 0 else 0.0
    sharpe = None
    if len(returns) > 1 and float(returns.std()) != 0.0:
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std())
    active = pd.to_numeric(curve["open_position_count"], errors="coerce").fillna(0) > 0
    exposure = pd.to_numeric(curve["exposure"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    completed = trades[trades.get("status") == "exited"] if "status" in trades.columns and not trades.empty else trades
    return {
        **params,
        "initial_cash": float(initial_cash),
        "final_equity": final_equity,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": _max_drawdown(equity),
        "sharpe": sharpe,
        "annual_volatility": float(returns.std() * np.sqrt(252)) if len(returns) > 1 else None,
        "win_day_ratio": float((returns > 0).mean()) if len(returns) else 0.0,
        "active_day_ratio": float(active.mean()) if len(active) else 0.0,
        "average_exposure": float(exposure.mean()) if not exposure.empty else 0.0,
        "selected_event_count": int(len(trades)),
        "completed_trade_count": int(len(completed)),
        "average_trade_return_after_cost": (
            float(pd.to_numeric(completed["trade_return_after_cost"], errors="coerce").mean())
            if not completed.empty and "trade_return_after_cost" in completed.columns
            else 0.0
        ),
        "research_only": True,
        "broker_action": "none",
    }


def _metrics(curve: pd.DataFrame, trades: pd.DataFrame, initial_cash: float, params: dict[str, Any]) -> dict[str, Any]:
    if curve.empty:
        return {
            **params,
            "initial_cash": float(initial_cash),
            "final_equity": float(initial_cash),
            "total_return": 0.0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": None,
            "active_day_ratio": 0.0,
            "selected_event_count": 0,
            "research_only": True,
            "broker_action": "none",
        }
    equity = pd.to_numeric(curve["equity"], errors="coerce").dropna()
    returns = pd.to_numeric(curve["daily_proxy_return"], errors="coerce").fillna(0.0)
    dates = pd.to_datetime(curve["date"], errors="coerce").dropna()
    years = max((dates.max() - dates.min()).days / 365.25, 0.0) if not dates.empty else 0.0
    final_equity = float(equity.iloc[-1]) if not equity.empty else float(initial_cash)
    total_return = final_equity / float(initial_cash) - 1.0 if initial_cash > 0 else 0.0
    cagr = (final_equity / initial_cash) ** (1.0 / years) - 1.0 if years > 0 and final_equity > 0 else 0.0
    sharpe = None
    if len(returns) > 1 and float(returns.std()) != 0.0:
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std())
    active = pd.to_numeric(curve["selected_count"], errors="coerce").fillna(0) > 0
    return {
        **params,
        "initial_cash": float(initial_cash),
        "final_equity": final_equity,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": _max_drawdown(equity),
        "sharpe": sharpe,
        "annual_volatility": float(returns.std() * np.sqrt(252)) if len(returns) > 1 else None,
        "win_day_ratio": float((returns > 0).mean()) if len(returns) else 0.0,
        "active_day_ratio": float(active.mean()) if len(active) else 0.0,
        "average_selected_per_active_day": (
            float(pd.to_numeric(curve.loc[active, "selected_count"], errors="coerce").mean()) if active.any() else 0.0
        ),
        "selected_event_count": int(len(trades)),
        "average_trade_return_after_cost": (
            float(pd.to_numeric(trades["trade_return_after_cost"], errors="coerce").mean()) if not trades.empty else 0.0
        ),
        "research_only": True,
        "broker_action": "none",
    }


def build_chip_reversal_event_portfolio(
    events: pd.DataFrame,
    *,
    horizon: int = 5,
    max_positions: int = 10,
    score_bucket: str = "deep",
    board_scope: str = "main_chinext",
    round_trip_cost: float = DEFAULT_ROUND_TRIP_COST,
    exposure: float = 1.0,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    start_date: str | None = None,
    end_date: str | None = None,
    max_cost_gap_ma20: float | None = None,
) -> ChipReversalEventPortfolioBuild:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if max_positions <= 0:
        raise ValueError("max_positions must be positive.")
    if not 0.0 <= exposure <= 1.0:
        raise ValueError("exposure must be between 0 and 1.")
    if round_trip_cost < 0:
        raise ValueError("round_trip_cost must be non-negative.")
    return_column = _trade_return_column(horizon)
    required = {"date", "code", "board", "score_bucket", "chip_reversal_score", return_column}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")

    data = events.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["board"] = data["board"].astype(str)
    data["score_bucket"] = data["score_bucket"].astype(str)
    data["chip_reversal_score"] = pd.to_numeric(data["chip_reversal_score"], errors="coerce")
    data["amount_yi"] = pd.to_numeric(data["amount_yi"], errors="coerce").fillna(0.0) if "amount_yi" in data.columns else 0.0
    data[return_column] = pd.to_numeric(data[return_column], errors="coerce")
    data = data.dropna(subset=["date", "code", "chip_reversal_score", return_column])
    if start_date:
        data = data[data["date"] >= pd.Timestamp(_normalize_date(start_date))]
    if end_date:
        data = data[data["date"] <= pd.Timestamp(_normalize_date(end_date))]
    data = data[data["board"].map(lambda value: _include_board(value, board_scope))].copy()
    calendar_dates = pd.Index(sorted(data["date"].dropna().unique()))
    if score_bucket != "all":
        data = data[data["score_bucket"] == str(score_bucket)].copy()
    data, quality_filter_params = _apply_signal_quality_filters(data, max_cost_gap_ma20=max_cost_gap_ma20)

    selector_columns = [column for column in SELECTOR_COLUMNS if column in data.columns]
    future_columns = _future_label_columns(list(events.columns))
    if data.empty:
        curve = pd.DataFrame(columns=["date", "equity", "daily_proxy_return", "basket_trade_return", "selected_count"])
        trades = pd.DataFrame()
        params = {
            "horizon": int(horizon),
            "max_positions": int(max_positions),
            "score_bucket": score_bucket,
            "board_scope": board_scope,
            "round_trip_cost": float(round_trip_cost),
            "exposure": float(exposure),
            "execution_model": "overlap_scaled_event_proxy",
            **quality_filter_params,
        }
        return ChipReversalEventPortfolioBuild(
            curve=curve,
            trades=trades,
            metrics=_metrics(curve, trades, initial_cash, params),
            selector_columns=selector_columns,
            future_label_columns_excluded=future_columns,
        )

    selected_parts: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []
    equity = float(initial_cash)
    for raw_date in calendar_dates:
        day = data[data["date"] == raw_date].copy()
        sort_columns = ["chip_reversal_score"]
        ascending = [False]
        if "code" in day.columns:
            sort_columns.append("code")
            ascending.append(True)
        day = day.sort_values(sort_columns, ascending=ascending)
        selected = day.head(max_positions).copy()
        if not selected.empty:
            selected["candidate_rank"] = range(1, len(selected) + 1)
            selected["trade_return_before_cost"] = selected[return_column].astype(float)
            selected["trade_return_after_cost"] = selected["trade_return_before_cost"] - float(round_trip_cost)
        basket_trade_return = float(selected["trade_return_after_cost"].mean()) if not selected.empty else 0.0
        daily_proxy_return = float(exposure) * basket_trade_return / float(horizon)
        equity *= 1.0 + daily_proxy_return
        daily_rows.append(
            {
                "date": pd.Timestamp(raw_date).strftime("%Y-%m-%d"),
                "equity": float(equity),
                "daily_proxy_return": daily_proxy_return,
                "basket_trade_return": basket_trade_return,
                "selected_count": int(len(selected)),
            }
        )
        selected_parts.append(selected)

    curve = pd.DataFrame(daily_rows)
    trades = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    if not trades.empty:
        keep = [
            "date",
            "candidate_rank",
            "code",
            "name",
            "board",
            "score_bucket",
            "chip_reversal_score",
            "amount_yi",
            "cost_gap_ma20",
            return_column,
            "trade_return_before_cost",
            "trade_return_after_cost",
        ]
        keep.extend(column for column in ["close", "next_open", "open_gap_next"] if column in trades.columns)
        trades = trades[[column for column in keep if column in trades.columns]].copy()
        trades["date"] = pd.to_datetime(trades["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        trades["research_only"] = True
        trades["broker_action"] = "none"

    params = {
        "horizon": int(horizon),
        "max_positions": int(max_positions),
        "score_bucket": score_bucket,
        "board_scope": board_scope,
        "round_trip_cost": float(round_trip_cost),
        "exposure": float(exposure),
        "execution_model": "overlap_scaled_event_proxy",
        **quality_filter_params,
    }
    return ChipReversalEventPortfolioBuild(
        curve=curve,
        trades=trades,
        metrics=_metrics(curve, trades, initial_cash, params),
        selector_columns=selector_columns,
        future_label_columns_excluded=future_columns,
    )


def build_chip_reversal_position_backtest(
    events: pd.DataFrame,
    *,
    histories: dict[str, pd.DataFrame],
    horizon: int = 5,
    max_new_positions_per_day: int = 5,
    max_total_positions: int | None = None,
    score_bucket: str = "deep",
    board_scope: str = "main_chinext",
    round_trip_cost: float = DEFAULT_ROUND_TRIP_COST,
    target_exposure: float = 0.5,
    per_position_weight: float | None = None,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    start_date: str | None = None,
    end_date: str | None = None,
    max_cost_gap_ma20: float | None = None,
) -> ChipReversalEventPortfolioBuild:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if max_new_positions_per_day <= 0:
        raise ValueError("max_new_positions_per_day must be positive.")
    if max_total_positions is None:
        max_total_positions = max_new_positions_per_day * int(horizon)
    if max_total_positions <= 0:
        raise ValueError("max_total_positions must be positive.")
    if not 0.0 <= target_exposure <= 1.0:
        raise ValueError("target_exposure must be between 0 and 1.")
    if per_position_weight is None:
        per_position_weight = target_exposure / float(max_total_positions) if max_total_positions else target_exposure
    if per_position_weight <= 0:
        raise ValueError("per_position_weight must be positive.")
    if round_trip_cost < 0:
        raise ValueError("round_trip_cost must be non-negative.")

    required = {"date", "code", "board", "score_bucket", "chip_reversal_score"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")

    data = events.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].map(_clean_code)
    data["board"] = data["board"].astype(str)
    data["score_bucket"] = data["score_bucket"].astype(str)
    data["chip_reversal_score"] = pd.to_numeric(data["chip_reversal_score"], errors="coerce")
    data["amount_yi"] = pd.to_numeric(data["amount_yi"], errors="coerce").fillna(0.0) if "amount_yi" in data.columns else 0.0
    data = data.dropna(subset=["date", "code", "chip_reversal_score"])
    data = data[data["code"] != ""].copy()
    if start_date:
        data = data[data["date"] >= pd.Timestamp(_normalize_date(start_date))]
    if end_date:
        data = data[data["date"] <= pd.Timestamp(_normalize_date(end_date))]
    data = data[data["board"].map(lambda value: _include_board(value, board_scope))].copy()
    if score_bucket != "all":
        data = data[data["score_bucket"] == str(score_bucket)].copy()
    data, quality_filter_params = _apply_signal_quality_filters(data, max_cost_gap_ma20=max_cost_gap_ma20)

    selector_columns = [column for column in SELECTOR_COLUMNS if column in data.columns]
    future_columns = _future_label_columns(list(events.columns))
    prepared_histories = {
        _clean_code(code): _prepare_price_history(history)
        for code, history in histories.items()
        if _clean_code(code)
    }
    prepared_histories = {code: history for code, history in prepared_histories.items() if not history.empty}
    date_index_maps = {
        code: {
            pd.Timestamp(value).strftime("%Y-%m-%d"): int(index)
            for index, value in enumerate(history["date"].tolist())
        }
        for code, history in prepared_histories.items()
    }

    planned_rows: list[dict[str, Any]] = []
    skipped_event_count = 0
    for _, event in data.iterrows():
        code = str(event["code"])
        history = prepared_histories.get(code)
        if history is None or history.empty:
            skipped_event_count += 1
            continue
        signal_date = pd.Timestamp(event["date"])
        signal_idx = date_index_maps.get(code, {}).get(signal_date.strftime("%Y-%m-%d"))
        if signal_idx is None:
            skipped_event_count += 1
            continue
        entry_idx = signal_idx + 1
        exit_idx = signal_idx + int(horizon)
        if entry_idx >= len(history) or exit_idx >= len(history):
            skipped_event_count += 1
            continue
        entry = history.iloc[entry_idx]
        exit_row = history.iloc[exit_idx]
        entry_price = float(entry["open"])
        exit_price = float(exit_row["close"])
        if entry_price <= 0 or exit_price <= 0:
            skipped_event_count += 1
            continue
        row = event.to_dict()
        row.update(
            {
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "entry_date": pd.Timestamp(entry["date"]).strftime("%Y-%m-%d"),
                "exit_date": pd.Timestamp(exit_row["date"]).strftime("%Y-%m-%d"),
                "entry_price": entry_price,
                "planned_exit_price": exit_price,
            }
        )
        planned_rows.append(row)

    if not planned_rows:
        curve = pd.DataFrame(
            columns=[
                "date",
                "cash",
                "invested_value",
                "equity",
                "daily_return",
                "exposure",
                "entry_count",
                "exit_count",
                "open_position_count",
            ]
        )
        trades = pd.DataFrame()
        params = {
            "horizon": int(horizon),
            "max_new_positions_per_day": int(max_new_positions_per_day),
            "max_total_positions": int(max_total_positions),
            "score_bucket": score_bucket,
            "board_scope": board_scope,
            "round_trip_cost": float(round_trip_cost),
            "target_exposure": float(target_exposure),
            "per_position_weight": float(per_position_weight),
            "execution_model": "cash_position_backtest",
            "skipped_event_count": int(skipped_event_count),
            **quality_filter_params,
        }
        return ChipReversalEventPortfolioBuild(
            curve=curve,
            trades=trades,
            metrics=_position_metrics(curve, trades, initial_cash, params),
            selector_columns=selector_columns,
            future_label_columns_excluded=future_columns,
        )

    planned = pd.DataFrame(planned_rows)
    planned["entry_date_ts"] = pd.to_datetime(planned["entry_date"], errors="coerce")
    planned["exit_date_ts"] = pd.to_datetime(planned["exit_date"], errors="coerce")
    planned = planned.dropna(subset=["entry_date_ts", "exit_date_ts"])
    min_date = min(pd.to_datetime(planned["signal_date"]).min(), planned["entry_date_ts"].min())
    max_date = planned["exit_date_ts"].max()
    calendar_values: set[pd.Timestamp] = set()
    for code in planned["code"].dropna().unique().tolist():
        history = prepared_histories.get(str(code))
        if history is None or history.empty:
            continue
        mask = (history["date"] >= min_date) & (history["date"] <= max_date)
        calendar_values.update(pd.Timestamp(value) for value in history.loc[mask, "date"].tolist())
    calendar = sorted(calendar_values)
    if not calendar:
        calendar = sorted(set(planned["entry_date_ts"].tolist()) | set(planned["exit_date_ts"].tolist()))

    close_maps = {
        code: dict(zip(history["date"].dt.strftime("%Y-%m-%d"), history["close"].astype(float), strict=False))
        for code, history in prepared_histories.items()
    }
    entries_by_date = {key: group.copy() for key, group in planned.groupby("entry_date")}
    cash = float(initial_cash)
    last_equity = float(initial_cash)
    open_positions: list[dict[str, Any]] = []
    trade_records: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []

    for raw_date in calendar:
        date_key = pd.Timestamp(raw_date).strftime("%Y-%m-%d")
        held_codes = {position["code"] for position in open_positions}
        invested_before_entry = float(sum(float(position["last_value"]) for position in open_positions))
        entry_count = 0
        day_candidates = entries_by_date.get(date_key, pd.DataFrame()).copy()
        if not day_candidates.empty:
            day_candidates = day_candidates[~day_candidates["code"].isin(held_codes)].copy()
            day_candidates = day_candidates.sort_values(["chip_reversal_score", "code"], ascending=[False, True])
            open_slots = max(0, int(max_total_positions) - len(open_positions))
            selected_count = min(int(max_new_positions_per_day), open_slots, len(day_candidates))
            if selected_count > 0:
                selected = day_candidates.head(selected_count).copy()
                capacity_value = max(0.0, last_equity * float(target_exposure) - invested_before_entry)
                per_position_value = min(
                    last_equity * float(per_position_weight),
                    capacity_value / float(selected_count) if selected_count else 0.0,
                    cash / float(selected_count) if selected_count else 0.0,
                )
                if per_position_value > 0:
                    for rank, (_, event) in enumerate(selected.iterrows(), start=1):
                        if cash < per_position_value:
                            break
                        entry_value = float(per_position_value)
                        cash -= entry_value
                        position = {
                            "code": str(event["code"]),
                            "entry_date": str(event["entry_date"]),
                            "exit_date": str(event["exit_date"]),
                            "entry_price": float(event["entry_price"]),
                            "entry_value": entry_value,
                            "last_value": entry_value,
                        }
                        record = {
                            "signal_date": str(event["signal_date"]),
                            "entry_date": str(event["entry_date"]),
                            "exit_date": str(event["exit_date"]),
                            "candidate_rank": int(rank),
                            "code": str(event["code"]),
                            "name": event.get("name"),
                            "board": event.get("board"),
                            "score_bucket": event.get("score_bucket"),
                            "chip_reversal_score": float(event["chip_reversal_score"]),
                            "amount_yi": float(event.get("amount_yi", 0.0)),
                            "cost_gap_ma20": event.get("cost_gap_ma20"),
                            "entry_price": float(event["entry_price"]),
                            "entry_value": entry_value,
                            "exit_price": np.nan,
                            "exit_value": np.nan,
                            "trade_return_after_cost": np.nan,
                            "status": "open",
                            "research_only": True,
                            "broker_action": "none",
                        }
                        trade_records.append(record)
                        position["record_index"] = len(trade_records) - 1
                        open_positions.append(position)
                        entry_count += 1

        exit_count = 0
        remaining_positions: list[dict[str, Any]] = []
        for position in open_positions:
            close_price = close_maps.get(position["code"], {}).get(date_key)
            if close_price is not None and close_price > 0:
                position["last_value"] = float(position["entry_value"]) * (
                    close_price / float(position["entry_price"]) - float(round_trip_cost)
                )
            if position["exit_date"] == date_key:
                cash += float(position["last_value"])
                record = trade_records[int(position["record_index"])]
                record["exit_price"] = close_price
                record["exit_value"] = float(position["last_value"])
                record["trade_return_after_cost"] = float(position["last_value"]) / float(position["entry_value"]) - 1.0
                record["status"] = "exited"
                exit_count += 1
            else:
                remaining_positions.append(position)
        open_positions = remaining_positions
        invested_value = float(sum(float(position["last_value"]) for position in open_positions))
        equity = float(cash + invested_value)
        daily_return = equity / last_equity - 1.0 if last_equity > 0 else 0.0
        exposure = invested_value / equity if equity > 0 else 0.0
        daily_rows.append(
            {
                "date": date_key,
                "cash": float(cash),
                "invested_value": invested_value,
                "equity": equity,
                "daily_return": float(daily_return),
                "exposure": float(exposure),
                "entry_count": int(entry_count),
                "exit_count": int(exit_count),
                "open_position_count": int(len(open_positions)),
            }
        )
        last_equity = equity

    curve = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_records)
    if not trades.empty:
        trades["research_only"] = trades["research_only"].astype(object)
    params = {
        "horizon": int(horizon),
        "max_new_positions_per_day": int(max_new_positions_per_day),
        "max_total_positions": int(max_total_positions),
        "score_bucket": score_bucket,
        "board_scope": board_scope,
        "round_trip_cost": float(round_trip_cost),
        "target_exposure": float(target_exposure),
        "per_position_weight": float(per_position_weight),
        "execution_model": "cash_position_backtest",
        "skipped_event_count": int(skipped_event_count),
        **quality_filter_params,
    }
    return ChipReversalEventPortfolioBuild(
        curve=curve,
        trades=trades,
        metrics=_position_metrics(curve, trades, initial_cash, params),
        selector_columns=selector_columns,
        future_label_columns_excluded=future_columns,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame, limit: int = 50) -> list[dict[str, Any]]:
    return [{key: _json_safe(value) for key, value in row.items()} for row in frame.head(limit).to_dict("records")]


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _fmt_num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def _write_report_legacy(metrics: dict[str, Any], output_dir: Path) -> Path:
    report_path = output_dir / "chip_reversal_event_portfolio.md"
    lines = [
        "# 筹码反转事件组合研究代理曲线",
        "",
        f"- 生成时间：`{metrics.get('generated_at')}`",
        "- 说明：这是用事件收益标签构造的研究代理曲线，不连接券商，不下单，不作为实盘执行曲线。",
        "- 选择器只使用信号日可见字段；未来收益列仅用于入选后的研究标签兑现。",
        "",
        "## 指标",
        "",
        "| 项目 | 值 |",
        "| --- | ---: |",
        f"| 执行模型 | `{metrics.get('execution_model')}` |",
        f"| 持有期 | {metrics.get('horizon')} |",
        f"| 每日最大入选 | {metrics.get('max_positions')} |",
        f"| 暴露比例 | {_fmt_pct(metrics.get('exposure'))} |",
        f"| 往返成本 | {_fmt_pct(metrics.get('round_trip_cost'))} |",
        f"| 总收益 | {_fmt_pct(metrics.get('total_return'))} |",
        f"| CAGR | {_fmt_pct(metrics.get('cagr'))} |",
        f"| 最大回撤 | {_fmt_pct(metrics.get('max_drawdown'))} |",
        f"| Sharpe | {_fmt_num(metrics.get('sharpe'))} |",
        f"| 活跃日占比 | {_fmt_pct(metrics.get('active_day_ratio'))} |",
        f"| 入选事件数 | {metrics.get('selected_event_count')} |",
        f"| 平均入选交易收益(扣成本后) | {_fmt_pct(metrics.get('average_trade_return_after_cost'))} |",
        f"| 券商动作 | `{metrics.get('broker_action')}` |",
        "",
        "## 约束",
        "",
        "- 这是 `overlap_scaled_event_proxy`：把持有期事件收益按持有期摊入滚动资金曲线，用于卫星候选筛选，不替代逐笔撮合回测。",
        "- 下一步若要进入组合层，需要继续和稳定核心曲线合成，并检查组合回撤、资金利用率和换手压力。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _write_report(metrics: dict[str, Any], output_dir: Path) -> Path:
    report_path = output_dir / "chip_reversal_event_portfolio.md"
    execution_model = str(metrics.get("execution_model"))
    if execution_model == "cash_position_backtest":
        title = "# 事件组合真实持仓回测"
        model_note = "- 说明：按现金、持仓、下一交易日开盘入场、持有期收盘退出构造研究回测；不连接券商，不下单，broker_action=none。"
        constraint_note = "- 该模型使用历史复权 OHLC 做逐日持仓估值，仍属于研究回测，不代表实盘可成交结果。"
        max_position_label = "每日最大新开仓"
        exposure_value = metrics.get("target_exposure")
    else:
        title = "# 筹码反转事件组合研究代理曲线"
        model_note = "- 说明：这是用事件收益标签构造的研究代理曲线，不连接券商，不下单，broker_action=none。"
        constraint_note = "- 这是 `overlap_scaled_event_proxy`：把持有期事件收益按持有期摊入滚动资金曲线，用于卫星候选筛选，不替代逐笔持仓回测。"
        max_position_label = "每日最大入选"
        exposure_value = metrics.get("exposure")
    lines = [
        title,
        "",
        f"- 生成时间：`{metrics.get('generated_at')}`",
        model_note,
        "- 选择器只使用信号日可见字段；未来收益列仅用于研究标签或事后对照，不参与排序。",
        "",
        "## 指标",
        "",
        "| 项目 | 值 |",
        "| --- | ---: |",
        f"| 执行模型 | `{metrics.get('execution_model')}` |",
        f"| 持有期 | {metrics.get('horizon')} |",
        f"| {max_position_label} | {metrics.get('max_positions', metrics.get('max_new_positions_per_day'))} |",
        f"| 最大总持仓 | {metrics.get('max_total_positions', 'N/A')} |",
        f"| 目标仓位 | {_fmt_pct(exposure_value)} |",
        f"| 单票目标权重 | {_fmt_pct(metrics.get('per_position_weight'))} |",
        f"| 往返成本 | {_fmt_pct(metrics.get('round_trip_cost'))} |",
        f"| 总收益 | {_fmt_pct(metrics.get('total_return'))} |",
        f"| CAGR | {_fmt_pct(metrics.get('cagr'))} |",
        f"| 最大回撤 | {_fmt_pct(metrics.get('max_drawdown'))} |",
        f"| Sharpe | {_fmt_num(metrics.get('sharpe'))} |",
        f"| 活跃日占比 | {_fmt_pct(metrics.get('active_day_ratio'))} |",
        f"| 平均仓位 | {_fmt_pct(metrics.get('average_exposure'))} |",
        f"| 入选事件数 | {metrics.get('selected_event_count')} |",
        f"| 完成交易数 | {metrics.get('completed_trade_count', 'N/A')} |",
        f"| 跳过事件数 | {metrics.get('skipped_event_count', 'N/A')} |",
        f"| 平均交易收益(扣成本后) | {_fmt_pct(metrics.get('average_trade_return_after_cost'))} |",
        f"| 券商动作 | `{metrics.get('broker_action')}` |",
        "",
        "## 约束",
        "",
        constraint_note,
        "- 下一步若要进入组合层，需要继续和稳定核心曲线合成，并检查组合回撤、资金利用率和换手压力。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_chip_reversal_event_portfolio(
    *,
    events_path: str | Path = DEFAULT_EVENTS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    horizon: int = 5,
    max_positions: int = 10,
    score_bucket: str = "deep",
    board_scope: str = "main_chinext",
    round_trip_cost: float = DEFAULT_ROUND_TRIP_COST,
    exposure: float = 1.0,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    start_date: str | None = None,
    end_date: str | None = None,
    execution_model: str = "proxy",
    price_data_dir: str | Path = Path("data/processed/stocks"),
    max_total_positions: int | None = None,
    per_position_weight: float | None = None,
    max_cost_gap_ma20: float | None = None,
) -> ChipReversalEventPortfolioResult:
    resolved_events = Path(events_path)
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(resolved_events, dtype={"code": str})
    model = str(execution_model).replace("_", "-").lower()
    if model in {"proxy", "overlap-scaled-event-proxy"}:
        build = build_chip_reversal_event_portfolio(
            events,
            horizon=horizon,
            max_positions=max_positions,
            score_bucket=score_bucket,
            board_scope=board_scope,
            round_trip_cost=round_trip_cost,
            exposure=exposure,
            initial_cash=initial_cash,
            start_date=start_date,
            end_date=end_date,
            max_cost_gap_ma20=max_cost_gap_ma20,
        )
    elif model in {"true-position", "cash-position-backtest"}:
        histories = _load_price_histories(events["code"], price_data_dir)
        build = build_chip_reversal_position_backtest(
            events,
            histories=histories,
            horizon=horizon,
            max_new_positions_per_day=max_positions,
            max_total_positions=max_total_positions,
            score_bucket=score_bucket,
            board_scope=board_scope,
            round_trip_cost=round_trip_cost,
            target_exposure=exposure,
            per_position_weight=per_position_weight,
            initial_cash=initial_cash,
            start_date=start_date,
            end_date=end_date,
            max_cost_gap_ma20=max_cost_gap_ma20,
        )
    else:
        raise ValueError("execution_model must be one of: proxy, true-position.")
    curve_path = resolved_output / "equity_curve.csv"
    daily_filename = (
        "daily_position_curve.csv"
        if build.metrics.get("execution_model") == "cash_position_backtest"
        else "daily_proxy_returns.csv"
    )
    daily_path = resolved_output / daily_filename
    trades_path = resolved_output / "selected_events.csv"
    metrics_path = resolved_output / "metrics.json"
    snapshot_path = resolved_output / "chip_reversal_event_portfolio_snapshot.json"

    build.curve.to_csv(curve_path, index=False, encoding="utf-8-sig")
    build.curve.to_csv(daily_path, index=False, encoding="utf-8-sig")
    build.trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    metrics = {
        **build.metrics,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "events_path": str(resolved_events),
        "output_dir": str(resolved_output),
        "curve_path": str(curve_path),
        "daily_path": str(daily_path),
        "trades_path": str(trades_path),
        "metrics_path": str(metrics_path),
        "snapshot_path": str(snapshot_path),
        "selector_columns": build.selector_columns,
        "future_label_columns_excluded": build.future_label_columns_excluded,
    }
    metrics_path.write_text(json.dumps({k: _json_safe(v) for k, v in metrics.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(metrics, resolved_output)
    snapshot = {
        **metrics,
        "report_path": str(report_path),
        "curve_preview": _records(build.curve),
        "selected_event_preview": _records(build.trades),
    }
    snapshot_path.write_text(json.dumps({k: _json_safe(v) for k, v in snapshot.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return ChipReversalEventPortfolioResult(
        output_dir=resolved_output,
        curve_path=curve_path,
        trades_path=trades_path,
        daily_path=daily_path,
        metrics_path=metrics_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        curve=build.curve,
        trades=build.trades,
        metrics=metrics,
    )
