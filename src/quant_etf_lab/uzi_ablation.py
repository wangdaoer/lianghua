"""Ablation checks for UZI-Skill auxiliary sidecar signals."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_UZI_SNAPSHOT_PATH = Path("outputs/research/uzi_auxiliary_signals_latest/uzi_auxiliary_snapshot.json")
DEFAULT_STOCK_PRICE_DIR = Path("data/processed/stocks")
DEFAULT_OUTCOME_HISTORY_PATH = Path("outputs/research/stock_target_review_outcomes_history.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/uzi_sidecar_ablation_latest")
DEFAULT_HORIZONS = [1, 5, 20]

ABLATION_FIELDS = [
    "ticker",
    "code",
    "risk_flag",
    "integration_recommendation",
    "valuation_overheat",
    "bearish_ratio",
    "bottleneck_consensus",
    "entry_date",
    "entry_close",
    "price_source",
    "price_cache_status",
    "return_1d",
    "max_drawdown_1d",
    "status_1d",
    "return_5d",
    "max_drawdown_5d",
    "status_5d",
    "return_20d",
    "max_drawdown_20d",
    "status_20d",
    "latest_available_date",
    "available_forward_days",
    "ablation_status",
    "position_effect",
    "broker_action",
]


@dataclass(frozen=True)
class UZISidecarAblationResult:
    output_dir: Path
    snapshot_path: Path
    summary_path: Path
    report_path: Path
    snapshot: dict[str, Any]


def _float_or_none(value: Any) -> float | None:
    if value in (None, "", "-", "--"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _round_or_none(value: Any, digits: int = 6) -> float | None:
    number = _float_or_none(value)
    if number is None:
        return None
    return round(number, digits)


def _normalize_code(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[-6:].zfill(6)


def _read_uzi_snapshot(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("signals"), list):
        raise ValueError(f"Invalid UZI auxiliary snapshot: {resolved}")
    return payload


def _read_price_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str})
    if frame.empty:
        return pd.DataFrame(columns=["date", "close", "low"])
    required = {"date", "close", "low"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Price file {path} missing columns: {sorted(missing)}")
    prices = frame.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices["low"] = pd.to_numeric(prices["low"], errors="coerce")
    for column in ("open", "high", "volume", "amount"):
        if column in prices.columns:
            prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices = prices.dropna(subset=["date", "close", "low"]).sort_values("date")
    duplicate_columns = [column for column in ("open", "high", "low", "close", "volume", "amount") if column in prices.columns]
    if len(duplicate_columns) == 6:
        stale_duplicate = prices[duplicate_columns].eq(prices[duplicate_columns].shift()).all(axis=1)
        prices = prices[~stale_duplicate]
    return prices.reset_index(drop=True)


def _read_outcome_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype={"code": str})
    if frame.empty or "code" not in frame.columns:
        return pd.DataFrame()
    outcomes = frame.copy()
    outcomes["code"] = outcomes["code"].map(_normalize_code)
    for column in ("date", "entry_date"):
        if column in outcomes.columns:
            outcomes[column] = pd.to_datetime(outcomes[column], errors="coerce")
    return outcomes


def _latest_outcome_for_code(outcomes: pd.DataFrame, code: str) -> pd.Series | None:
    if outcomes.empty:
        return None
    matched = outcomes[outcomes["code"] == code]
    if matched.empty:
        return None
    sort_columns = [column for column in ("entry_date", "date") if column in matched.columns]
    if sort_columns:
        matched = matched.sort_values(sort_columns)
    return matched.iloc[-1]


def _generated_date(snapshot: dict[str, Any]) -> pd.Timestamp | None:
    parsed = pd.to_datetime(snapshot.get("generated_at"), errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _infer_entry_index(prices: pd.DataFrame, signal: dict[str, Any], generated_date: pd.Timestamp | None) -> int | None:
    if prices.empty:
        return None
    eligible = prices
    if generated_date is not None:
        eligible = prices[prices["date"] <= generated_date]
    if eligible.empty:
        return None
    current_price = _float_or_none(signal.get("current_price"))
    if current_price is not None:
        matches = eligible[(eligible["close"] - current_price).abs() <= max(0.01, abs(current_price) * 0.0001)]
        if not matches.empty:
            last_idx = int(matches.index[-1])
            first_idx = last_idx
            while first_idx > 0 and abs(float(prices.loc[first_idx - 1, "close"]) - current_price) <= max(0.01, abs(current_price) * 0.0001):
                first_idx -= 1
            return first_idx
    return int(eligible.index[-1])


def _horizon_metrics(prices: pd.DataFrame, entry_idx: int, horizon: int) -> dict[str, Any]:
    entry_close = float(prices.loc[entry_idx, "close"])
    future_idx = entry_idx + int(horizon)
    if future_idx >= len(prices):
        return {
            f"return_{horizon}d": None,
            f"max_drawdown_{horizon}d": None,
            f"status_{horizon}d": "pending",
        }
    future_close = float(prices.loc[future_idx, "close"])
    window = prices.iloc[entry_idx + 1 : future_idx + 1]
    min_low = float(window["low"].min()) if not window.empty else entry_close
    return {
        f"return_{horizon}d": round(future_close / entry_close - 1.0, 6),
        f"max_drawdown_{horizon}d": round(min_low / entry_close - 1.0, 6),
        f"status_{horizon}d": "available",
    }


def _risk_flag(signal: dict[str, Any], bearish_threshold: float, bottleneck_threshold: float) -> bool:
    bearish_ratio = _float_or_none(signal.get("bearish_ratio"))
    bottleneck = _float_or_none(signal.get("bottleneck_consensus"))
    return bool(
        signal.get("valuation_overheat") is True
        and bearish_ratio is not None
        and bearish_ratio >= bearish_threshold
        and bottleneck is not None
        and bottleneck <= bottleneck_threshold
    )


def _ablation_status(row: dict[str, Any], horizons: Iterable[int]) -> str:
    if row["price_cache_status"] not in {"ok", "outcome_history"}:
        return "missing_price_cache"
    available_returns = [
        _float_or_none(row.get(f"return_{horizon}d"))
        for horizon in horizons
        if row.get(f"status_{horizon}d") == "available"
    ]
    if not available_returns:
        return "pending_future_data"
    if not row["risk_flag"]:
        return "control_observe"
    if any(value is not None and value < 0 for value in available_returns):
        return "loss_reduction_candidate"
    if any(value is not None and value > 0 for value in available_returns):
        return "opportunity_cost_candidate"
    return "neutral"


def build_uzi_sidecar_ablation_snapshot(
    uzi_snapshot_path: str | Path = DEFAULT_UZI_SNAPSHOT_PATH,
    stock_price_dir: str | Path = DEFAULT_STOCK_PRICE_DIR,
    outcome_history_path: str | Path = DEFAULT_OUTCOME_HISTORY_PATH,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    bearish_threshold: float = 0.50,
    bottleneck_threshold: float = 20.0,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build an ablation snapshot without changing trading behavior."""
    horizon_values = sorted({int(value) for value in horizons})
    if not horizon_values or any(value <= 0 for value in horizon_values):
        raise ValueError("Ablation horizons must be positive integers.")
    source = _read_uzi_snapshot(uzi_snapshot_path)
    price_root = Path(stock_price_dir)
    outcome_path = Path(outcome_history_path)
    outcomes = _read_outcome_frame(outcome_path)
    gen_date = _generated_date(source)
    rows: list[dict[str, Any]] = []
    for signal in source["signals"]:
        code = _normalize_code(signal.get("code") or signal.get("ticker"))
        ticker = str(signal.get("ticker") or code)
        price_path = price_root / f"{code}.csv"
        row: dict[str, Any] = {
            "ticker": ticker,
            "code": code,
            "risk_flag": _risk_flag(signal, bearish_threshold, bottleneck_threshold),
            "integration_recommendation": signal.get("integration_recommendation", ""),
            "valuation_overheat": bool(signal.get("valuation_overheat")),
            "bearish_ratio": _round_or_none(signal.get("bearish_ratio"), 4),
            "bottleneck_consensus": _round_or_none(signal.get("bottleneck_consensus"), 2),
            "entry_date": None,
            "entry_close": None,
            "price_source": str(price_path),
            "price_cache_status": "missing",
            "latest_available_date": None,
            "available_forward_days": 0,
            "position_effect": "none",
            "broker_action": "none",
        }
        for horizon in horizon_values:
            row[f"return_{horizon}d"] = None
            row[f"max_drawdown_{horizon}d"] = None
            row[f"status_{horizon}d"] = "pending"

        outcome_row = _latest_outcome_for_code(outcomes, code)
        if outcome_row is not None:
            row["price_cache_status"] = "outcome_history"
            entry_date = outcome_row.get("entry_date")
            if pd.notna(entry_date):
                row["entry_date"] = pd.Timestamp(entry_date).strftime("%Y-%m-%d")
            row["entry_close"] = _round_or_none(outcome_row.get("entry_close"), 4)
            row["price_source"] = str(outcome_path)
            latest_dates: list[pd.Timestamp] = []
            available_forward_days = 0
            for horizon in horizon_values:
                status_value = str(outcome_row.get(f"outcome_status_{horizon}d") or "pending")
                return_value = _float_or_none(outcome_row.get(f"return_{horizon}d"))
                future_date = pd.to_datetime(outcome_row.get(f"future_date_{horizon}d"), errors="coerce")
                if pd.notna(future_date):
                    latest_dates.append(pd.Timestamp(future_date))
                if status_value == "available":
                    available_forward_days = max(available_forward_days, horizon)
                row[f"return_{horizon}d"] = _round_or_none(return_value, 6)
                row[f"max_drawdown_{horizon}d"] = None
                row[f"status_{horizon}d"] = status_value
            row["available_forward_days"] = available_forward_days
            if latest_dates:
                row["latest_available_date"] = max(latest_dates).strftime("%Y-%m-%d")
        elif price_path.exists():
            prices = _read_price_frame(price_path)
            entry_idx = _infer_entry_index(prices, signal, gen_date)
            if entry_idx is not None:
                row["price_cache_status"] = "ok"
                row["entry_date"] = pd.Timestamp(prices.loc[entry_idx, "date"]).strftime("%Y-%m-%d")
                row["entry_close"] = _round_or_none(prices.loc[entry_idx, "close"], 4)
                row["latest_available_date"] = pd.Timestamp(prices["date"].max()).strftime("%Y-%m-%d")
                row["available_forward_days"] = max(0, int(len(prices) - entry_idx - 1))
                for horizon in horizon_values:
                    row.update(_horizon_metrics(prices, entry_idx, horizon))
            else:
                row["price_cache_status"] = "empty_or_unusable"
        row["ablation_status"] = _ablation_status(row, horizon_values)
        rows.append(row)

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["ablation_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "status": "uzi_sidecar_ablation_built",
        "generated_at": generated_at or datetime.now().isoformat(timespec="seconds"),
        "source_snapshot_path": str(Path(uzi_snapshot_path)),
        "stock_price_dir": str(price_root),
        "outcome_history_path": str(outcome_path),
        "outcome_history_used": outcome_path.exists(),
        "integration_status": "research_ablation_only",
        "position_effect": "none",
        "broker_action": "none",
        "horizons": horizon_values,
        "risk_rule": {
            "valuation_overheat": True,
            "bearish_ratio_min": bearish_threshold,
            "bottleneck_consensus_max": bottleneck_threshold,
        },
        "summary": {
            "sample_count": len(rows),
            "risk_flag_count": sum(1 for row in rows if row["risk_flag"]),
            "missing_price_cache_count": sum(1 for row in rows if row["price_cache_status"] not in {"ok", "outcome_history"}),
            "status_counts": status_counts,
        },
        "rows": rows,
        "prohibited_integrations": [
            "default_allocator",
            "paper_account_target_weights",
            "daily_pipeline_position_sizing",
            "live_preflight_orders",
            "broker_order_generation",
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ABLATION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _render_report(snapshot: dict[str, Any]) -> str:
    rows = snapshot["rows"]
    table = "\n".join(
        "| {ticker} | {risk_flag} | {entry_date} | {entry_close} | {return_1d} | {max_drawdown_1d} | {status_1d} | {ablation_status} |".format(**row)
        for row in rows
    )
    prohibited = "\n".join(f"- `{item}`" for item in snapshot["prohibited_integrations"])
    return f"""# UZI Sidecar Ablation

| Item | Value |
| --- | --- |
| Status | `{snapshot["integration_status"]}` |
| Position effect | `{snapshot["position_effect"]}` |
| Broker action | `{snapshot["broker_action"]}` |
| Sample count | `{snapshot["summary"]["sample_count"]}` |
| Risk-flag count | `{snapshot["summary"]["risk_flag_count"]}` |
| Missing price cache | `{snapshot["summary"]["missing_price_cache_count"]}` |

## Rule Under Test

Flag a target when `valuation_overheat=True`, `bearish_ratio >= {snapshot["risk_rule"]["bearish_ratio_min"]}`, and `bottleneck_consensus <= {snapshot["risk_rule"]["bottleneck_consensus_max"]}`.

## Rows

| Ticker | Risk flag | Entry date | Entry close | Return 1D | Max DD 1D | Status 1D | Ablation status |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
{table}

## Interpretation

This is a research-only sidecar check. Current evidence is preliminary when 5D/20D windows are still pending or price cache is missing.

## Prohibited Direct Integrations

{prohibited}
"""


def run_uzi_sidecar_ablation(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    uzi_snapshot_path: str | Path = DEFAULT_UZI_SNAPSHOT_PATH,
    stock_price_dir: str | Path = DEFAULT_STOCK_PRICE_DIR,
    outcome_history_path: str | Path = DEFAULT_OUTCOME_HISTORY_PATH,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> UZISidecarAblationResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    snapshot = build_uzi_sidecar_ablation_snapshot(
        uzi_snapshot_path=uzi_snapshot_path,
        stock_price_dir=stock_price_dir,
        outcome_history_path=outcome_history_path,
        horizons=horizons,
    )
    snapshot_path = output_path / "uzi_sidecar_ablation_snapshot.json"
    summary_path = output_path / "uzi_sidecar_ablation_summary.csv"
    report_path = output_path / "uzi_sidecar_ablation_report.md"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary_path, list(snapshot["rows"]))
    report_path.write_text(_render_report(snapshot), encoding="utf-8")
    return UZISidecarAblationResult(
        output_dir=output_path,
        snapshot_path=snapshot_path,
        summary_path=summary_path,
        report_path=report_path,
        snapshot=snapshot,
    )
