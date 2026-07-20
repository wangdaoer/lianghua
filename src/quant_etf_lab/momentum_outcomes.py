"""Research-only after-event outcome analysis for strong-gain stocks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_DATA_DIR = Path("data/processed/stocks")
DEFAULT_OUTPUT_DIR = Path("outputs/research/momentum_outcomes_latest")


@dataclass(frozen=True)
class MomentumOutcomeResult:
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


def _limit_up_threshold_pct(board: str) -> float:
    if board in {"chinext", "star"}:
        return 19.5
    if board == "bse":
        return 29.0
    return 9.8


def _read_price_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str})
    required = {"date", "close"}
    missing = required - set(frame.columns)
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
    data = data.dropna(subset=["date", "code", "close"])
    data = data[data["close"] > 0].sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return data.reset_index(drop=True)


def build_momentum_events(
    prices: pd.DataFrame,
    horizons: Iterable[int] = (1, 3, 5, 10),
    strong_gain_threshold_pct: float = 5.0,
    board_scope: str = "main_chinext",
) -> pd.DataFrame:
    """Label strong-gain events and attach future returns."""
    if prices.empty:
        return pd.DataFrame()
    horizons_list = [int(horizon) for horizon in horizons]
    if any(horizon <= 0 for horizon in horizons_list):
        raise ValueError("horizons must be positive integers.")
    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].map(_clean_code) if "code" in data.columns else ""
    data["open"] = pd.to_numeric(data["open"], errors="coerce") if "open" in data.columns else pd.NA
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce") if "amount" in data.columns else 0.0
    data = data.dropna(subset=["date", "code", "close"]).sort_values("date").reset_index(drop=True)
    if data.empty:
        return pd.DataFrame()
    code = str(data["code"].iloc[0])
    board = _board(code)
    if not _include_board(board, board_scope):
        return pd.DataFrame()
    threshold = _limit_up_threshold_pct(board)
    data["daily_return_pct"] = data["close"].pct_change() * 100.0
    for horizon in horizons_list:
        data[f"return_{horizon}d"] = data["close"].shift(-horizon) / data["close"] - 1.0
        data[f"trade_return_{horizon}d"] = data["close"].shift(-horizon) / data["open"].shift(-1) - 1.0

    max_horizon = max(horizons_list)
    if len(data) <= max_horizon:
        return pd.DataFrame()
    mature_mask = data.index.to_series() + max_horizon < len(data)
    signal_mask = (data["daily_return_pct"] >= float(strong_gain_threshold_pct)) & mature_mask
    events = data.loc[signal_mask].copy()
    if events.empty:
        return pd.DataFrame()
    events["date"] = events["date"].dt.strftime("%Y-%m-%d")
    events["board"] = board
    events["limit_up"] = events["daily_return_pct"] >= threshold
    events["strong_gain"] = True
    events["signal_type"] = f"strong_gain_{float(strong_gain_threshold_pct):g}"
    events["next_open"] = data["open"].shift(-1).loc[events.index]
    events["capital_efficiency"] = events["amount"] / 100_000_000.0
    events["amount_bucket"] = events["capital_efficiency"].map(_amount_bucket)
    events["broker_action"] = "none"
    events["research_only"] = True
    ordered_columns = [
        "date",
        "code",
        "name",
        "board",
        "signal_type",
        "limit_up",
        "strong_gain",
        "daily_return_pct",
        "close",
        "next_open",
        "amount",
        "capital_efficiency",
        "amount_bucket",
        "broker_action",
        "research_only",
    ]
    for horizon in horizons_list:
        ordered_columns.extend([f"return_{horizon}d", f"trade_return_{horizon}d"])
    events = events.reindex(columns=ordered_columns).reset_index(drop=True)
    if not events.empty:
        events["limit_up"] = events["limit_up"].astype(object)
        events["strong_gain"] = events["strong_gain"].astype(object)
        events["research_only"] = events["research_only"].astype(object)
    return events


def _csv_paths(data_dir: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(path for path in data_dir.glob(pattern) if path.is_file())


def _amount_bucket(amount_yi: Any) -> str:
    try:
        value = float(amount_yi)
    except (TypeError, ValueError):
        return "unknown"
    if not pd.notna(value):
        return "unknown"
    if value < 1.0:
        return "lt_1y"
    if value < 5.0:
        return "gte_1y"
    if value < 20.0:
        return "gte_5y"
    return "gte_20y"


def _summarize_events(events: pd.DataFrame, horizons: list[int], min_events: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if events.empty:
        return pd.DataFrame(
            columns=[
                "signal_type",
                "board",
                "amount_bucket",
                "horizon",
                "event_count",
                "win_rate",
                "avg_return",
                "median_return",
                "capital_efficiency",
                "avg_amount_yi",
            ]
        )
    events = events.copy()
    if "amount_bucket" not in events.columns:
        events["amount_bucket"] = events["capital_efficiency"].map(_amount_bucket)

    group_specs: list[tuple[list[str], pd.DataFrame]] = [
        (["signal_type"], events),
        (["signal_type", "board"], events),
        (["signal_type", "amount_bucket"], events),
        (["signal_type", "board", "amount_bucket"], events),
    ]
    for group_columns, source in group_specs:
        for group_key, group in source.groupby(group_columns, dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_values = dict(zip(group_columns, group_key))
            for horizon in horizons:
                return_column = f"trade_return_{horizon}d" if f"trade_return_{horizon}d" in group.columns else f"return_{horizon}d"
                returns = pd.to_numeric(group[return_column], errors="coerce").dropna()
                if len(returns) < int(min_events):
                    continue
                gains = returns[returns > 0]
                losses = returns[returns < 0]
                avg_gain = float(gains.mean()) if not gains.empty else 0.0
                avg_loss = float(losses.mean()) if not losses.empty else 0.0
                avg_return = float(returns.mean())
                avg_return_per_day = avg_return / float(horizon)
                rows.append(
                    {
                        "signal_type": str(group_values.get("signal_type", "all")),
                        "board": str(group_values.get("board", "all")),
                        "amount_bucket": str(group_values.get("amount_bucket", "all")),
                        "horizon": horizon,
                        "event_count": int(len(returns)),
                        "win_rate": float((returns > 0).mean()),
                        "avg_return": avg_return,
                        "median_return": float(returns.median()),
                        "avg_gain": avg_gain,
                        "avg_loss": avg_loss,
                        "payoff_ratio": avg_gain / abs(avg_loss) if avg_loss else 0.0,
                        "profit_factor": float(gains.sum()) / abs(float(losses.sum())) if not losses.empty else 0.0,
                        "avg_return_per_day": avg_return_per_day,
                        "capital_efficiency": avg_return_per_day,
                        "avg_amount_yi": float(pd.to_numeric(group.get("capital_efficiency"), errors="coerce").mean()),
                        "return_column": return_column,
                        "broker_action": "none",
                        "research_only": True,
                    }
                )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        ["horizon", "capital_efficiency", "win_rate", "event_count"],
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
    title = "\u6da8\u505c\u4e0e\u5f3a\u52bf\u80a1\u540e\u9a8c\u8bc4\u4f30"
    if summary.empty:
        rows = "| N/A | N/A | 0 | N/A | N/A | N/A |"
    else:
        rows = "\n".join(
            "| `{signal}` | `{board}` | `{bucket}` | {horizon} | {count} | {win} | {avg} | {eff} | {amount:.2f} |".format(
                signal=row.get("signal_type"),
                board=row.get("board"),
                bucket=row.get("amount_bucket"),
                horizon=int(row.get("horizon") or 0),
                count=int(row.get("event_count") or 0),
                win=_pct(row.get("win_rate")),
                avg=_pct(row.get("avg_return")),
                eff=_pct(row.get("capital_efficiency")),
                amount=float(row.get("avg_amount_yi") or 0.0),
            )
            for _, row in summary.iterrows()
        )
    return f"""# {title}

Generated at: `{snapshot.get("generated_at")}`

This is a research-only after-event outcome analysis for limit-up and strong-gain stocks. It does not connect to brokers, place orders, or provide investment advice.

## Summary

| Item | Value |
| --- | ---: |
| Status | `{snapshot.get("status")}` |
| Event count | {snapshot.get("event_count", 0)} |
| Summary rows | {snapshot.get("summary_row_count", 0)} |
| Horizons | `{snapshot.get("horizons")}` |
| Broker action | `{snapshot.get("broker_action")}` |

## Group Outcomes

| Signal | Board | Amount bucket | Horizon | Events | Win rate | Avg return | Return/day | Avg amount(Yi) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

## Files

- Events CSV: `{snapshot.get("events_path")}`
- Summary CSV: `{snapshot.get("summary_path")}`
- Snapshot JSON: `{snapshot.get("snapshot_path")}`
"""


def run_momentum_outcome_analysis(
    project_root: str | Path = Path("."),
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    horizons: Iterable[int] = (1, 3, 5, 10),
    strong_gain_threshold_pct: float = 5.0,
    board_scope: str = "main_chinext",
    min_events: int = 5,
    max_symbols: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    recursive: bool = False,
) -> MomentumOutcomeResult:
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
    paths = _csv_paths(resolved_data, recursive=recursive)
    if max_symbols is not None and max_symbols > 0:
        paths = paths[: int(max_symbols)]
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            prices = _read_price_csv(path)
            if start is not None:
                prices = prices[prices["date"] >= pd.Timestamp(start)]
            if end is not None:
                prices = prices[prices["date"] <= pd.Timestamp(end)]
            events = build_momentum_events(
                prices,
                horizons=horizons_list,
                strong_gain_threshold_pct=strong_gain_threshold_pct,
                board_scope=board_scope,
            )
            if not events.empty:
                frames.append(events)
        except (OSError, ValueError, pd.errors.EmptyDataError) as error:
            failures.append({"path": str(path), "error": str(error)})
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not events.empty:
        events = events.sort_values(["date", "code"]).reset_index(drop=True)
    summary = _summarize_events(events, horizons=horizons_list, min_events=min_events)

    resolved_output.mkdir(parents=True, exist_ok=True)
    events_path = resolved_output / "momentum_events.csv"
    summary_path = resolved_output / "momentum_outcome_summary.csv"
    snapshot_path = resolved_output / "momentum_outcomes_snapshot.json"
    report_path = resolved_output / "momentum_outcomes.md"
    events.to_csv(events_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok" if not events.empty else "no_events",
        "data_dir": str(resolved_data),
        "output_dir": str(resolved_output),
        "file_count": len(paths),
        "loaded_symbol_count": int(events["code"].nunique()) if not events.empty else 0,
        "event_count": int(len(events)),
        "summary_row_count": int(len(summary)),
        "start_date": pd.Timestamp(start).strftime("%Y-%m-%d") if start is not None else None,
        "end_date": pd.Timestamp(end).strftime("%Y-%m-%d") if end is not None else None,
        "horizons": horizons_list,
        "strong_gain_threshold_pct": float(strong_gain_threshold_pct),
        "board_scope": board_scope,
        "min_events": int(min_events),
        "failure_count": len(failures),
        "failures": failures[:20],
        "events_path": str(events_path),
        "summary_path": str(summary_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "events": _json_records(events.head(100)),
        "research_only": True,
        "broker_action": "none",
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, summary), encoding="utf-8")

    return MomentumOutcomeResult(
        output_dir=resolved_output,
        events_path=events_path,
        summary_path=summary_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        events=events,
        summary=summary,
    )
