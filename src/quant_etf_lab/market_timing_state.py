"""Research-only market timing state for allocator observation.

This module absorbs the video-derived "five-dimensional timing" idea as an
observation layer only. The first version computes local technical, breadth,
and turnover evidence. Valuation, funding, sentiment, and fundamentals remain
explicitly unknown until audited local sources exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .market_data_source import DEFAULT_DAILY_MARKET_DATA_DIR, DEFAULT_EXCHANGE_INGEST_DIR, load_market_snapshot_rows


DEFAULT_DATA_DIR = Path("data/processed/stocks")
DEFAULT_OUTPUT_DIR = Path("outputs/research/market_timing_state_latest")
UNKNOWN_REASON = "local audited source not connected"


@dataclass(frozen=True)
class MarketTimingStateResult:
    output_dir: Path
    state_path: Path
    components_path: Path
    symbol_state_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    state: pd.DataFrame
    components: pd.DataFrame
    symbol_state: pd.DataFrame


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = default if path is None else Path(path)
    return raw if raw.is_absolute() else project_root / raw


def _parse_date(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _clean_code(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        return digits[-6:]
    return digits.zfill(6) if digits else ""


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not pd.notna(number):
        return default
    return number


def _csv_paths(data_dir: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(path for path in data_dir.glob(pattern) if path.is_file())


def _limit_paths(paths: list[Path], max_symbols: int | None) -> list[Path]:
    if max_symbols is None or max_symbols <= 0:
        return paths
    return paths[:max_symbols]


def _snapshot_codes(rows: list[dict[str, Any]]) -> set[str]:
    codes = {_clean_code(row.get("security_code") or row.get("code") or row.get("symbol")) for row in rows}
    return {code for code in codes if code}


def _read_history_latest(
    path: Path,
    trade_date: pd.Timestamp | None,
    lookback_days: int,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "code", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].map(_clean_code)
    data["name"] = data.get("name", data["code"]).astype(str)
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "code", "close"]).sort_values("date")
    data = data.loc[data["close"] > 0].drop_duplicates(subset=["date"], keep="last")
    if trade_date is not None:
        data = data.loc[data["date"] <= trade_date]
    if lookback_days > 0:
        data = data.tail(max(lookback_days, 61))
    if len(data) < 2:
        return pd.DataFrame()
    close = pd.to_numeric(data["close"], errors="coerce")
    latest = data.iloc[-1]
    previous = data.iloc[-2]
    latest_close = _as_float(latest.get("close"), default=float("nan"))
    previous_close = _as_float(previous.get("close"), default=float("nan"))
    if not pd.notna(latest_close) or not pd.notna(previous_close) or previous_close <= 0:
        return pd.DataFrame()
    row = {
        "date": pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
        "code": str(latest["code"]).zfill(6),
        "name": str(latest.get("name") or latest["code"]),
        "close": latest_close,
        "amount": _as_float(latest.get("amount"), default=0.0),
        "daily_return": latest_close / previous_close - 1.0,
        "above_ma20": bool(latest_close > close.rolling(20, min_periods=20).mean().iloc[-1]),
        "above_ma60": bool(latest_close > close.rolling(60, min_periods=60).mean().iloc[-1]),
    }
    return pd.DataFrame([row])


def build_symbol_state_panel(
    data_dir: str | Path,
    snapshot_rows: list[dict[str, Any]] | None = None,
    trade_date: str | None = None,
    lookback_days: int = 90,
    max_symbols: int | None = None,
    recursive: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved_dir = Path(data_dir)
    paths = _csv_paths(resolved_dir, recursive=recursive)
    allowed_codes = _snapshot_codes(snapshot_rows or [])
    if allowed_codes:
        paths = [path for path in paths if _clean_code(path.stem) in allowed_codes]
    paths = _limit_paths(paths, max_symbols)
    target_date = _parse_date(trade_date)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            frame = _read_history_latest(path, target_date, lookback_days=lookback_days)
            if not frame.empty:
                frames.append(frame)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            failures.append({"path": str(path), "error": str(error)})
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not panel.empty:
        panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    meta = {
        "data_dir": str(resolved_dir),
        "file_count": len(paths),
        "loaded_symbol_count": int(panel["code"].nunique()) if not panel.empty else 0,
        "row_count": int(len(panel)),
        "failure_count": len(failures),
        "failures": failures[:20],
        "snapshot_code_count": len(allowed_codes),
    }
    return panel, meta


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & (weights > 0)
    if bool(valid.any()):
        return float((values.loc[valid] * weights.loc[valid]).sum() / weights.loc[valid].sum())
    clean = values.dropna()
    return float(clean.mean()) if not clean.empty else 0.0


def _regime_from_metrics(
    above_ma20_ratio: float,
    above_ma60_ratio: float,
    up_turnover_share: float,
    down_turnover_share: float,
    market_return: float,
) -> tuple[str, float]:
    if (
        above_ma20_ratio >= 0.55
        and above_ma60_ratio >= 0.50
        and up_turnover_share >= 0.50
        and market_return >= 0.0
    ):
        return "risk_on", 0.65
    if (
        above_ma20_ratio < 0.35
        or above_ma60_ratio < 0.35
        or (down_turnover_share > 0.60 and market_return < 0.0)
    ):
        return "risk_off", 0.25
    return "neutral", 0.50


def _technical_status(above_ma20_ratio: float, above_ma60_ratio: float) -> str:
    if above_ma20_ratio >= 0.55 and above_ma60_ratio >= 0.50:
        return "risk_on"
    if above_ma20_ratio < 0.35 or above_ma60_ratio < 0.35:
        return "risk_off"
    return "neutral"


def _breadth_status(advancer_ratio: float) -> str:
    if advancer_ratio >= 0.55:
        return "risk_on"
    if advancer_ratio < 0.35:
        return "risk_off"
    return "neutral"


def _turnover_status(up_turnover_share: float, down_turnover_share: float) -> str:
    if up_turnover_share >= 0.55:
        return "risk_on"
    if down_turnover_share > 0.60:
        return "risk_off"
    return "neutral"


def build_market_timing_state(latest_rows: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    if latest_rows.empty:
        raise ValueError("No latest rows available for market timing state.")
    data = latest_rows.copy()
    amount = pd.to_numeric(data["amount"], errors="coerce").fillna(0.0)
    returns = pd.to_numeric(data["daily_return"], errors="coerce")
    above_ma20 = data["above_ma20"].astype(bool)
    above_ma60 = data["above_ma60"].astype(bool)
    up_mask = returns > 0.0
    down_mask = returns < 0.0
    total_amount = float(amount.sum())
    above_ma20_ratio = float(above_ma20.mean())
    above_ma60_ratio = float(above_ma60.mean())
    advancer_ratio = float(up_mask.mean())
    decliner_ratio = float(down_mask.mean())
    up_turnover_share = float(amount.loc[up_mask].sum() / total_amount) if total_amount > 0 else 0.0
    down_turnover_share = float(amount.loc[down_mask].sum() / total_amount) if total_amount > 0 else 0.0
    market_return = _weighted_average(returns, amount)
    market_regime, position_reference = _regime_from_metrics(
        above_ma20_ratio=above_ma20_ratio,
        above_ma60_ratio=above_ma60_ratio,
        up_turnover_share=up_turnover_share,
        down_turnover_share=down_turnover_share,
        market_return=market_return,
    )
    technical_status = _technical_status(above_ma20_ratio, above_ma60_ratio)
    breadth_status = _breadth_status(advancer_ratio)
    turnover_status = _turnover_status(up_turnover_share, down_turnover_share)
    latest_date = str(data["date"].max())
    state = {
        "status": "ok",
        "trade_date": latest_date,
        "market_regime": market_regime,
        "position_reference": position_reference,
        "position_reference_note": "observation_only_not_applied",
        "symbol_count": int(data["code"].nunique()),
        "market_return": market_return,
        "above_ma20_ratio": above_ma20_ratio,
        "above_ma60_ratio": above_ma60_ratio,
        "advancer_ratio": advancer_ratio,
        "decliner_ratio": decliner_ratio,
        "up_turnover_share": up_turnover_share,
        "down_turnover_share": down_turnover_share,
        "total_turnover_yi": total_amount / 100_000_000.0,
        "valuation_status": "unknown",
        "funding_status": "unknown",
        "sentiment_status": "unknown",
        "fundamentals_status": "unknown",
        "unknown_reason": UNKNOWN_REASON,
    }
    components = pd.DataFrame(
        [
            {
                "dimension": "technical",
                "status": technical_status,
                "score": position_reference,
                "evidence": f"above_ma20={above_ma20_ratio:.4f}; above_ma60={above_ma60_ratio:.4f}",
            },
            {
                "dimension": "breadth",
                "status": breadth_status,
                "score": advancer_ratio,
                "evidence": f"advancer_ratio={advancer_ratio:.4f}; decliner_ratio={decliner_ratio:.4f}",
            },
            {
                "dimension": "turnover_structure",
                "status": turnover_status,
                "score": up_turnover_share,
                "evidence": f"up_turnover_share={up_turnover_share:.4f}; down_turnover_share={down_turnover_share:.4f}",
            },
            {"dimension": "valuation", "status": "unknown", "score": pd.NA, "evidence": UNKNOWN_REASON},
            {"dimension": "funding", "status": "unknown", "score": pd.NA, "evidence": UNKNOWN_REASON},
            {"dimension": "sentiment", "status": "unknown", "score": pd.NA, "evidence": UNKNOWN_REASON},
            {"dimension": "fundamentals", "status": "unknown", "score": pd.NA, "evidence": UNKNOWN_REASON},
        ]
    )
    return state, components


def _fmt_pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not pd.notna(number):
        return "N/A"
    return f"{number * 100:.2f}%"


def _markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "No rows."
    subset = frame.loc[:, [column for column in columns if column in frame.columns]].head(max_rows)
    lines = ["| " + " | ".join(subset.columns) + " |", "| " + " | ".join(["---"] * len(subset.columns)) + " |"]
    for row in subset.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _render_report(snapshot: dict[str, Any], components: pd.DataFrame) -> str:
    return f"""# Market Timing State

Generated at: `{snapshot.get("generated_at")}`

This report is research-only. It does not connect to brokers, place orders, or change live allocation.

## Summary

| Item | Value |
| --- | ---: |
| Status | `{snapshot.get("status")}` |
| Trade date | `{snapshot.get("trade_date")}` |
| Market regime | `{snapshot.get("market_regime")}` |
| Position reference | {snapshot.get("position_reference")} |
| Position note | `{snapshot.get("position_reference_note")}` |
| Market return | {_fmt_pct(snapshot.get("market_return"))} |
| Above MA20 ratio | {_fmt_pct(snapshot.get("above_ma20_ratio"))} |
| Above MA60 ratio | {_fmt_pct(snapshot.get("above_ma60_ratio"))} |
| Up turnover share | {_fmt_pct(snapshot.get("up_turnover_share"))} |
| Down turnover share | {_fmt_pct(snapshot.get("down_turnover_share"))} |
| Market source | `{snapshot.get("market_source_kind")}` |
| Fetch status | `{snapshot.get("fetch_status")}` |
| Research only | `{snapshot.get("research_only")}` |

## Components

{_markdown_table(components, ["dimension", "status", "score", "evidence"], 20)}

## Interpretation

- This is an observation layer for the five-dimensional timing idea.
- Current version only uses audited local technical, breadth, and turnover evidence.
- Valuation, funding, sentiment, and fundamentals remain `unknown` until local sources are connected.
- `position_reference` is not applied to allocator, paper account, or live trading.

## Files

- State CSV: `{snapshot.get("state_path")}`
- Components CSV: `{snapshot.get("components_path")}`
- Symbol state CSV: `{snapshot.get("symbol_state_path")}`
- Snapshot JSON: `{snapshot.get("snapshot_path")}`
"""


def run_market_timing_state(
    project_root: str | Path = Path("."),
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    daily_data_dir: str | Path | None = DEFAULT_DAILY_MARKET_DATA_DIR,
    ingest_project_dir: str | Path | None = DEFAULT_EXCHANGE_INGEST_DIR,
    trade_date: str | None = None,
    lookback_days: int = 90,
    max_symbols: int | None = None,
    recursive: bool = False,
    require_success: bool = True,
) -> MarketTimingStateResult:
    root = Path(project_root).resolve()
    resolved_data = _resolve(root, data_dir, DEFAULT_DATA_DIR)
    resolved_output = _resolve(root, output_dir, DEFAULT_OUTPUT_DIR)
    market_result = load_market_snapshot_rows(
        trade_date=trade_date,
        market="all",
        daily_data_dir=daily_data_dir,
        ingest_project_dir=ingest_project_dir or DEFAULT_EXCHANGE_INGEST_DIR,
        require_success=require_success,
    )
    panel, meta = build_symbol_state_panel(
        data_dir=resolved_data,
        snapshot_rows=market_result.rows,
        trade_date=market_result.trade_date or trade_date,
        lookback_days=lookback_days,
        max_symbols=max_symbols,
        recursive=recursive,
    )
    if panel.empty:
        raise ValueError(f"No market timing rows built from {resolved_data}")
    state, components = build_market_timing_state(panel)

    resolved_output.mkdir(parents=True, exist_ok=True)
    state_path = resolved_output / "market_timing_state.csv"
    components_path = resolved_output / "market_timing_components.csv"
    symbol_state_path = resolved_output / "market_timing_symbol_state.csv"
    snapshot_path = resolved_output / "market_timing_state_snapshot.json"
    report_path = resolved_output / "market_timing_state.md"

    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **state,
        "research_only": True,
        "broker_action": "none",
        "allocator_action": "none",
        "data_dir": str(resolved_data),
        "output_dir": str(resolved_output),
        "market_source_kind": market_result.source_kind,
        "market_source_path": str(market_result.source_path) if market_result.source_path else None,
        "market_snapshot_trade_date": market_result.trade_date,
        "market_snapshot_rows": int(len(market_result.rows)),
        "fetch_status": getattr(market_result.fetch_status, "status", ""),
        "fetch_run_id": getattr(market_result.fetch_status, "run_id", ""),
        "file_count": meta["file_count"],
        "loaded_symbol_count": meta["loaded_symbol_count"],
        "failure_count": meta["failure_count"],
        "failures": meta["failures"],
        "snapshot_code_count": meta["snapshot_code_count"],
        "lookback_days": int(lookback_days),
        "state_path": str(state_path),
        "components_path": str(components_path),
        "symbol_state_path": str(symbol_state_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "promotion_note": "Observation-only market regime evidence; not connected to allocator or broker actions.",
    }
    state_frame = pd.DataFrame([state])
    state_frame.to_csv(state_path, index=False, encoding="utf-8-sig")
    components.to_csv(components_path, index=False, encoding="utf-8-sig")
    panel.to_csv(symbol_state_path, index=False, encoding="utf-8-sig")
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, components), encoding="utf-8")
    return MarketTimingStateResult(
        output_dir=resolved_output,
        state_path=state_path,
        components_path=components_path,
        symbol_state_path=symbol_state_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        state=state_frame,
        components=components,
        symbol_state=panel,
    )
