"""Research-only focus pool for limit-up and strong-gain A-share candidates."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .market_data_source import DEFAULT_DAILY_MARKET_DATA_DIR, DEFAULT_EXCHANGE_INGEST_DIR, load_market_snapshot_rows


DEFAULT_OUTPUT_DIR = Path("outputs/research/momentum_focus_latest")
DEFAULT_OUTCOME_SUMMARY_PATH = Path("outputs/research/momentum_outcomes_latest/momentum_outcome_summary.csv")
DEFAULT_NAME_MAP_PATH = Path("data/processed/stock_name_map.csv")
MOMENTUM_FOCUS_COLUMNS = [
    "as_of_date",
    "trade_date",
    "code",
    "name",
    "market",
    "board",
    "signal_type",
    "limit_up",
    "strong_gain",
    "change_pct",
    "close_price",
    "turnover",
    "turnover_yi",
    "volume",
    "amount_bucket",
    "capital_utilization_score",
    "outcome_prior_match_level",
    "outcome_prior_win_rate",
    "outcome_prior_avg_return",
    "outcome_prior_profit_factor",
    "outcome_prior_event_count",
    "outcome_prior_avg_amount_yi",
    "focus_score",
    "research_priority",
    "source",
    "broker_action",
    "research_only",
]


@dataclass(frozen=True)
class MomentumFocusResult:
    output_dir: Path
    candidates_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    candidates: pd.DataFrame


def _parse_date(value: str | date | None) -> str:
    if value is None:
        return datetime.now().date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid as_of_date: {value}")
    return pd.Timestamp(parsed).date().isoformat()


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = default if path is None else Path(path)
    return raw if raw.is_absolute() else project_root / raw


def _clean_code(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6) if digits else text


def _clean_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"0", "none", "nan"}:
        return "未知"
    return text


def _name_from_map(code: str, name_map: dict[str, str] | None) -> str | None:
    if not name_map:
        return None
    mapped = str(name_map.get(_clean_code(code), "") or "").strip()
    return mapped or None


def _resolve_name(raw_name: Any, code: str, name_map: dict[str, str] | None) -> str:
    cleaned = _clean_name(raw_name)
    if cleaned == _clean_name(""):
        return _name_from_map(code, name_map) or cleaned
    return cleaned


def _is_special_treatment_name(name: Any) -> bool:
    text = str(name or "").strip()
    compact = text.replace(" ", "").upper()
    return compact.startswith(("ST", "*ST", "S*ST")) or "\u9000\u5e02" in text


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not pd.notna(number):
        return default
    return number


def _change_pct(value: Any, *, fractional_ratio: bool = False) -> float:
    pct = _as_float(value, default=float("nan"))
    if not pd.notna(pct):
        return float("nan")
    if fractional_ratio and -1.0 <= pct <= 1.0:
        return pct * 100.0
    return pct


def _row_change_pct(row: dict[str, Any]) -> float:
    if row.get("change_ratio") not in (None, ""):
        return _change_pct(row.get("change_ratio"), fractional_ratio=False)
    return _change_pct(row.get("pct_chg"), fractional_ratio=True)


def _board(raw_code: Any, clean_code: str) -> str:
    raw = str(raw_code or "").strip().lower()
    if raw.startswith("bj") or clean_code.startswith(("4", "8", "920")):
        return "bse"
    if clean_code.startswith(("300", "301")):
        return "chinext"
    if clean_code.startswith(("688", "689")):
        return "star"
    if clean_code.startswith(("000", "001", "002", "003", "600", "601", "603", "605")):
        return "main"
    return "unknown"


def _amount_bucket(value: Any) -> str:
    try:
        amount_yi = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not pd.notna(amount_yi):
        return "unknown"
    if amount_yi < 1.0:
        return "lt_1y"
    if amount_yi < 5.0:
        return "gte_1y"
    if amount_yi < 20.0:
        return "gte_5y"
    return "gte_20y"


def _load_outcome_summary(summary_path: Path | None, target_horizon: int) -> pd.DataFrame:
    if summary_path is None:
        return pd.DataFrame()
    if not summary_path.exists():
        return pd.DataFrame()
    try:
        data = pd.read_csv(summary_path)
    except Exception:
        return pd.DataFrame()
    if data.empty:
        return data
    data = data.copy()
    data["horizon"] = pd.to_numeric(data.get("horizon"), errors="coerce")
    data = data.loc[data["horizon"] == float(target_horizon)]
    if data.empty:
        return data
    return data.dropna(subset=["signal_type", "board", "amount_bucket"])


def _load_name_map(path: Path | None) -> tuple[dict[str, str], str]:
    if path is None or not path.exists():
        return {}, "missing"
    try:
        frame = pd.read_csv(path, dtype=str)
    except Exception:
        return {}, "invalid"
    if frame.empty:
        return {}, "empty"
    code_column = next((column for column in ["code", "security_code", "symbol"] if column in frame.columns), None)
    name_column = next((column for column in ["name", "security_name", "stock_name"] if column in frame.columns), None)
    if code_column is None or name_column is None:
        return {}, "invalid"
    result: dict[str, str] = {}
    for row in frame[[code_column, name_column]].to_dict(orient="records"):
        code = _clean_code(row.get(code_column))
        name = str(row.get(name_column) or "").strip()
        if code and name and name.lower() not in {"0", "none", "nan"}:
            result[code] = name
    return result, "ok" if result else "empty"


def _build_outcome_prior_lookup(summary: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in summary.to_dict(orient="records"):
        key = (
            str(row.get("signal_type", "all")),
            str(row.get("board", "all")),
            str(row.get("amount_bucket", "all")),
        )
        record = {
            "win_rate": _as_float(row.get("win_rate"), default=float("nan")),
            "avg_return": _as_float(row.get("avg_return"), default=float("nan")),
            "profit_factor": _as_float(row.get("profit_factor"), default=float("nan")),
            "event_count": int(_as_float(row.get("event_count"), default=0.0)),
            "avg_amount_yi": _as_float(row.get("avg_amount_yi"), default=float("nan")),
            "board": str(row.get("board", "all")),
            "amount_bucket": str(row.get("amount_bucket", "all")),
            "signal_type": str(row.get("signal_type", "all")),
        }
        lookup[key] = record
        # Add a signal-agnostic board+bucket alias so downstream can still
        # find prior evidence for both limit-up and strong-gain candidates.
        signal_agnostic_key = ("all", key[1], key[2])
        if signal_agnostic_key not in lookup:
            signal_agnostic_record = dict(record)
            signal_agnostic_record["signal_type"] = "all"
            lookup[signal_agnostic_key] = signal_agnostic_record
    return lookup


def _find_outcome_prior(
    prior_lookup: dict[tuple[str, str, str], dict[str, Any]],
    signal_type: str,
    board: str,
    amount_bucket: str,
) -> tuple[str, dict[str, Any] | None]:
    candidates = [
        ("exact", (signal_type, board, amount_bucket)),
        ("exact", ("all", board, amount_bucket)),
        ("board_bucket", (signal_type, board, "all")),
        ("signal_bucket", (signal_type, "all", amount_bucket)),
        ("global", (signal_type, "all", "all")),
    ]
    for match_level, key in candidates:
        prior = prior_lookup.get(key)
        if prior is not None:
            return match_level, prior
    return "unmatched", None


def _prior_score(prior: dict[str, Any] | None) -> float:
    if prior is None:
        return 0.0
    win_rate = _as_float(prior.get("win_rate"), default=0.0)
    avg_return = max(_as_float(prior.get("avg_return"), default=0.0), 0.0)
    profit_factor = _as_float(prior.get("profit_factor"), default=0.0)
    event_count = int(_as_float(prior.get("event_count"), default=0.0))
    score = win_rate * 35.0 + avg_return * 200.0 + min(math.log1p(max(profit_factor, 0.0)) * 8.0, 22.0)
    score += min(event_count / 250.0, 0.3) * 20.0
    if event_count < 20:
        score *= 0.5
    return score


def _limit_up_threshold_pct(board: str) -> float:
    if board in {"chinext", "star"}:
        return 19.5
    if board == "bse":
        return 29.0
    return 9.8


def _include_board(board: str, board_scope: str) -> bool:
    if board_scope == "all":
        return board != "unknown"
    if board_scope == "main_chinext":
        return board in {"main", "chinext"}
    raise ValueError("board_scope must be one of: main_chinext, all")


def _research_priority(limit_up: bool, change_pct: float, capital_utilization_score: float) -> str:
    if limit_up and capital_utilization_score >= 6.0:
        return "high"
    if limit_up or change_pct >= 9.0:
        return "medium"
    return "watch"


def build_momentum_focus_candidates(
    rows: list[dict[str, Any]],
    as_of_date: str | date | None = None,
    strong_gain_threshold_pct: float = 7.0,
    board_scope: str = "main_chinext",
    outcome_summary: pd.DataFrame | None = None,
    target_horizon: int = 5,
    name_map: dict[str, str] | None = None,
    exclude_special_treatment: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a research-only candidate pool from daily snapshot rows."""
    as_of = _parse_date(as_of_date)
    candidate_rows: list[dict[str, Any]] = []
    excluded_non_scope = 0
    below_threshold = 0
    malformed = 0
    excluded_special_treatment = 0
    prior_lookup = _build_outcome_prior_lookup(
        pd.DataFrame() if outcome_summary is None or outcome_summary.empty else outcome_summary
    )
    matched_prior_count = 0

    for raw in rows:
        raw_code = raw.get("security_code") or raw.get("code")
        code = _clean_code(raw_code)
        board = _board(raw_code, code)
        if not code or board == "unknown":
            malformed += 1
            continue
        if not _include_board(board, board_scope):
            excluded_non_scope += 1
            continue
        pct = _row_change_pct(raw)
        if not pd.notna(pct):
            malformed += 1
            continue
        strong_gain = pct >= strong_gain_threshold_pct
        limit_up = pct >= _limit_up_threshold_pct(board)
        if not strong_gain:
            below_threshold += 1
            continue
        name = _resolve_name(raw.get("security_name") or raw.get("name"), code, name_map)
        if exclude_special_treatment and _is_special_treatment_name(name):
            excluded_special_treatment += 1
            continue
        turnover = _as_float(raw.get("turnover"))
        turnover_yi = turnover / 100_000_000.0 if turnover > 0 else 0.0
        amount_bucket = _amount_bucket(turnover_yi)
        signal_type = "limit_up" if limit_up else "strong_gain_7pct"
        match_level, prior = _find_outcome_prior(
            prior_lookup=prior_lookup,
            signal_type=signal_type,
            board=board,
            amount_bucket=amount_bucket,
        )
        if prior is not None:
            matched_prior_count += 1
        capital_utilization_score = min(max(turnover_yi, 0.0), 20.0) / 20.0 * 10.0
        focus_score = (
            pct * 2.0
            + (20.0 if limit_up else 0.0)
            + capital_utilization_score
            + _prior_score(prior)
            + (3.0 if match_level == "exact" else 0.0)
        )
        if match_level in {"board_bucket", "signal_bucket", "global"}:
            focus_score += 1.0
        candidate_rows.append(
            {
                "as_of_date": as_of,
                "trade_date": _parse_date(raw.get("trade_date") or as_of),
                "code": code,
                "name": name,
                "market": str(raw.get("market") or "").strip(),
                "board": board,
                "signal_type": signal_type,
                "limit_up": bool(limit_up),
                "strong_gain": bool(strong_gain),
                "change_pct": pct,
                "close_price": _as_float(raw.get("close_price", raw.get("close"))),
                "turnover": turnover,
                "turnover_yi": turnover_yi,
                "volume": _as_float(raw.get("volume")),
                "amount_bucket": amount_bucket,
                "capital_utilization_score": capital_utilization_score,
                "outcome_prior_match_level": match_level,
                "outcome_prior_win_rate": prior.get("win_rate") if prior is not None else float("nan"),
                "outcome_prior_avg_return": prior.get("avg_return") if prior is not None else float("nan"),
                "outcome_prior_profit_factor": prior.get("profit_factor") if prior is not None else float("nan"),
                "outcome_prior_event_count": prior.get("event_count") if prior is not None else float("nan"),
                "outcome_prior_avg_amount_yi": prior.get("avg_amount_yi") if prior is not None else float("nan"),
                "focus_score": focus_score,
                "research_priority": _research_priority(limit_up, pct, capital_utilization_score),
                "source": str(raw.get("source") or "").strip(),
                "broker_action": "none",
                "research_only": True,
            }
        )

    candidates = pd.DataFrame(candidate_rows, columns=MOMENTUM_FOCUS_COLUMNS)
    if not candidates.empty:
        candidates = candidates.sort_values(
            ["focus_score", "change_pct", "turnover"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    payload = {
        "status": "ok",
        "as_of_date": as_of,
        "input_row_count": int(len(rows)),
        "candidate_count": int(len(candidates)),
        "limit_up_count": int(candidates["limit_up"].sum()) if not candidates.empty else 0,
        "strong_gain_count": int(candidates["strong_gain"].sum()) if not candidates.empty else 0,
        "excluded_non_main_chinext_count": int(excluded_non_scope),
        "excluded_special_treatment_count": int(excluded_special_treatment),
        "below_threshold_count": int(below_threshold),
        "malformed_row_count": int(malformed),
        "strong_gain_threshold_pct": float(strong_gain_threshold_pct),
        "board_scope": board_scope,
        "outcome_target_horizon": int(target_horizon),
        "outcome_prior_rows": int(0 if outcome_summary is None else len(outcome_summary)),
        "outcome_prior_matched_count": int(matched_prior_count),
        "research_objective": "Improve candidate-generation success rate, return, and capital utilization through limit-up and >=7% strong-gain research pools.",
        "broker_action": "none",
        "research_only": True,
    }
    return candidates, payload


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def _pct(value: Any) -> str:
    return f"{_as_float(value):.2f}%"


def _render_report(snapshot: dict[str, Any], candidates: pd.DataFrame) -> str:
    if candidates.empty:
        rows = "| N/A | N/A | N/A | N/A | 0.00% | 0.00 | 0.00 | N/A |"
    else:
        lines = []
        for row in candidates.head(40).to_dict(orient="records"):
            lines.append(
                "| `{code}` | {name} | {board} | `{signal}` | {change} | {turnover:.2f} | {score:.2f} | {priority} |".format(
                    code=str(row.get("code", "")),
                    name=str(row.get("name", "")),
                    board=str(row.get("board", "")),
                    signal=str(row.get("signal_type", "")),
                    change=_pct(row.get("change_pct")),
                    turnover=_as_float(row.get("turnover_yi")),
                    score=_as_float(row.get("focus_score")),
                    priority=str(row.get("research_priority", "")),
                )
            )
        rows = "\n".join(lines)
    return f"""# 涨停与强势股研究池

生成时间: `{snapshot.get("generated_at")}`

本文件只用于研究候选池生成，不连接券商、不下单、不撤单，也不构成投资建议。

## 摘要

| 项目 | 值 |
| --- | ---: |
| 状态 | `{snapshot.get("status")}` |
| 分析日期 | {snapshot.get("as_of_date")} |
| 行情日期 | {snapshot.get("trade_date")} |
| 数据来源 | `{snapshot.get("source_kind")}` |
| 输入行数 | {snapshot.get("input_row_count")} |
| 候选数量 | {snapshot.get("candidate_count")} |
| 涨停数量 | {snapshot.get("limit_up_count")} |
| 涨幅大于等于7%数量 | {snapshot.get("strong_gain_count")} |
| 非主板/创业板排除数量 | {snapshot.get("excluded_non_main_chinext_count")} |
| ST/退市排除数量 | {snapshot.get("excluded_special_treatment_count")} |
| 未达阈值数量 | {snapshot.get("below_threshold_count")} |
| 券商动作 | `{snapshot.get("broker_action")}` |

## 候选明细

| 股票代码 | 股票名称 | 板块 | 信号 | 涨幅 | 成交额(亿元) | 评分 | 优先级 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
{rows}

## 文件

- 候选CSV: `{snapshot.get("candidates_path")}`
- 快照JSON: `{snapshot.get("snapshot_path")}`
"""


def run_momentum_focus(
    project_root: str | Path = Path("."),
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    daily_data_dir: str | Path | None = None,
    ingest_project_dir: str | Path | None = None,
    as_of_date: str | date | None = None,
    trade_date: str | date | None = None,
    strong_gain_threshold_pct: float = 7.0,
    board_scope: str = "main_chinext",
    outcome_summary_path: str | Path | None = None,
    name_map_path: str | Path | None = DEFAULT_NAME_MAP_PATH,
    target_horizon: int = 5,
) -> MomentumFocusResult:
    root = Path(project_root).resolve()
    resolved_output = _resolve(root, output_dir, DEFAULT_OUTPUT_DIR)
    as_of = _parse_date(as_of_date)
    snapshot_trade_date = _parse_date(trade_date) if trade_date is not None else as_of
    resolved_summary_path = _resolve(root, outcome_summary_path, DEFAULT_OUTCOME_SUMMARY_PATH)
    outcome_summary = _load_outcome_summary(resolved_summary_path, target_horizon=target_horizon)
    resolved_name_map_path = _resolve(root, name_map_path, DEFAULT_NAME_MAP_PATH) if name_map_path is not None else None
    name_map, name_map_status = _load_name_map(resolved_name_map_path)
    result = load_market_snapshot_rows(
        trade_date=snapshot_trade_date,
        market="all",
        daily_data_dir=daily_data_dir,
        ingest_project_dir=ingest_project_dir or DEFAULT_EXCHANGE_INGEST_DIR,
        require_success=True,
    )
    candidates, payload = build_momentum_focus_candidates(
        result.rows,
        as_of_date=as_of,
        strong_gain_threshold_pct=strong_gain_threshold_pct,
        board_scope=board_scope,
        outcome_summary=outcome_summary,
        target_horizon=target_horizon,
        name_map=name_map,
    )
    resolved_output.mkdir(parents=True, exist_ok=True)
    candidates_path = resolved_output / "momentum_focus_candidates.csv"
    snapshot_path = resolved_output / "momentum_focus_snapshot.json"
    report_path = resolved_output / "momentum_focus.md"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **payload,
        "trade_date": result.trade_date,
        "source_kind": result.source_kind,
        "source_path": str(result.source_path) if result.source_path else None,
        "fetch_status": getattr(result.fetch_status, "status", ""),
        "fetch_run_id": getattr(result.fetch_status, "run_id", ""),
        "daily_data_dir": str(daily_data_dir or DEFAULT_DAILY_MARKET_DATA_DIR),
        "ingest_project_dir": str(ingest_project_dir or DEFAULT_EXCHANGE_INGEST_DIR),
        "outcome_summary_path": str(resolved_summary_path),
        "outcome_summary_rows": int(len(outcome_summary)),
        "name_map_path": str(resolved_name_map_path) if resolved_name_map_path else None,
        "name_map_status": name_map_status,
        "name_map_rows": int(len(name_map)),
        "target_horizon": int(target_horizon),
        "candidates_path": str(candidates_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "candidates": _json_records(candidates),
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, candidates), encoding="utf-8")
    return MomentumFocusResult(
        output_dir=resolved_output,
        candidates_path=candidates_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        candidates=candidates,
    )
