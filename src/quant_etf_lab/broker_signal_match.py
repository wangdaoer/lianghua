"""Research-only reconciliation between real broker trades and model signals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


MATCH_COLUMNS = [
    "statement_row_id",
    "trade_date",
    "symbol",
    "name",
    "side",
    "quantity",
    "price",
    "gross_amount",
    "settlement_amount",
    "known_signal_at_trade",
    "match_status",
    "known_signal_date",
    "signal_age_days",
    "signal_age_bucket",
    "signal_valid_until",
    "signal_validity_status",
    "execution_review_action",
    "target_match_timing",
    "target_signal_date",
    "target_layer",
    "target_weight",
    "target_action",
    "target_trigger_status",
    "target_trigger_summary",
    "review_match_timing",
    "review_signal_date",
    "matched_review_bucket",
    "matched_review_stage",
    "matched_manual_status",
    "matched_recommended_review",
    "history_match_timing",
    "history_signal_date",
    "history_review_bucket",
    "history_review_stage",
    "history_return_1d",
    "symbol_realized_pnl",
    "notes",
]

MATCH_CN_COLUMNS = {
    "statement_row_id": "脱敏流水ID",
    "trade_date": "成交日期",
    "symbol": "证券代码",
    "name": "证券名称",
    "side": "方向",
    "quantity": "成交数量",
    "price": "成交价格",
    "gross_amount": "成交金额",
    "settlement_amount": "清算金额",
    "known_signal_at_trade": "成交时已有信号",
    "match_status": "匹配状态",
    "known_signal_date": "已知信号日期",
    "signal_age_days": "信号距成交天数",
    "signal_age_bucket": "信号新鲜度分层",
    "signal_valid_until": "信号有效截止日",
    "signal_validity_status": "信号有效性状态",
    "execution_review_action": "执行复盘动作",
    "target_match_timing": "目标池匹配时点",
    "target_signal_date": "目标池信号日期",
    "target_layer": "目标层",
    "target_weight": "目标权重",
    "target_action": "目标动作",
    "target_trigger_status": "触发状态",
    "target_trigger_summary": "触发摘要",
    "review_match_timing": "复核池匹配时点",
    "review_signal_date": "复核信号日期",
    "matched_review_bucket": "复核桶",
    "matched_review_stage": "复核阶段",
    "matched_manual_status": "人工状态",
    "matched_recommended_review": "建议复核",
    "history_match_timing": "历史结果匹配时点",
    "history_signal_date": "历史信号日期",
    "history_review_bucket": "历史复核桶",
    "history_review_stage": "历史复核阶段",
    "history_return_1d": "历史1日收益",
    "symbol_realized_pnl": "标的已实现盈亏",
    "notes": "说明",
}

AGE_BUCKET_ORDER = ("0_1d", "2_5d", "6_10d", "gt10d")
SIGNAL_VALID_DAYS = 5


@dataclass(frozen=True)
class BrokerSignalMatchReviewResult:
    output_dir: Path
    match_path: Path
    match_cn_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _clean_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    return text.zfill(6) if text else ""


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _normalize_signal_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = frame.copy()
    if "symbol" not in data.columns and "code" in data.columns:
        data["symbol"] = data["code"]
    if "symbol" not in data.columns:
        return pd.DataFrame()
    data["symbol"] = data["symbol"].map(_clean_code)
    signal_date = data["signal_date"] if "signal_date" in data.columns else data.get("date")
    data["signal_date"] = pd.to_datetime(signal_date, errors="coerce")
    return data.sort_values(["symbol", "signal_date"]).reset_index(drop=True)


def _load_broker_trades(broker_dir: Path) -> pd.DataFrame:
    path = broker_dir / "broker_trades_normalized.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing broker normalized trades: {path}")
    trades = pd.read_csv(path, encoding="utf-8-sig")
    if "symbol" not in trades.columns:
        raise ValueError(f"Broker trades missing symbol column: {path}")
    trades["symbol"] = trades["symbol"].map(_clean_code)
    trades["trade_date_ts"] = pd.to_datetime(trades["trade_date"], errors="coerce")
    return trades.sort_values(["trade_date_ts", "statement_row_id"]).reset_index(drop=True)


def _load_realized_by_symbol(broker_dir: Path) -> dict[str, float]:
    realized = _read_optional_csv(broker_dir / "broker_realized_trades.csv")
    if realized.empty or "symbol" not in realized.columns or "realized_pnl" not in realized.columns:
        return {}
    realized["symbol"] = realized["symbol"].map(_clean_code)
    realized["realized_pnl"] = pd.to_numeric(realized["realized_pnl"], errors="coerce").fillna(0.0)
    return realized.groupby("symbol")["realized_pnl"].sum().to_dict()


def _pick_signal(frame: pd.DataFrame, symbol: str, trade_date: pd.Timestamp) -> tuple[pd.Series | None, str]:
    if frame.empty or not symbol or pd.isna(trade_date):
        return None, "missing_source"
    rows = frame[frame["symbol"] == symbol].copy()
    if rows.empty:
        return None, "no_match"
    known = rows[rows["signal_date"].notna() & (rows["signal_date"] <= trade_date)]
    if not known.empty:
        return known.sort_values("signal_date").iloc[-1], "known_at_trade"
    dated = rows[rows["signal_date"].notna()]
    if not dated.empty:
        return dated.sort_values("signal_date").iloc[0], "after_trade_snapshot"
    return rows.iloc[-1], "undated_snapshot"


def _value(row: pd.Series | None, key: str, default: Any = "") -> Any:
    if row is None or key not in row.index:
        return default
    value = row[key]
    if pd.isna(value):
        return default
    return value


def _signal_date(row: pd.Series | None) -> str:
    value = _value(row, "signal_date", pd.NaT)
    if pd.isna(value):
        return ""
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _known_signal_date(signal_matches: list[tuple[pd.Series | None, str]]) -> pd.Timestamp | None:
    dates: list[pd.Timestamp] = []
    for row, timing in signal_matches:
        if timing != "known_at_trade":
            continue
        value = _value(row, "signal_date", pd.NaT)
        if pd.isna(value):
            continue
        dates.append(pd.to_datetime(value).normalize())
    return max(dates) if dates else None


def _signal_age_days(trade_date: pd.Timestamp, signal_date: pd.Timestamp | None) -> Any:
    if signal_date is None or pd.isna(trade_date):
        return pd.NA
    return int((pd.to_datetime(trade_date).normalize() - signal_date).days)


def _signal_age_bucket(age_days: Any) -> str:
    if age_days is None or pd.isna(age_days):
        return "not_known"
    age = int(age_days)
    if age <= 1:
        return "0_1d"
    if age <= 5:
        return "2_5d"
    if age <= 10:
        return "6_10d"
    return "gt10d"


def _signal_valid_until(signal_date: pd.Timestamp | None) -> str:
    if signal_date is None:
        return ""
    return (signal_date + pd.Timedelta(days=SIGNAL_VALID_DAYS)).strftime("%Y-%m-%d")


def _signal_validity_status(match_status: str, age_days: Any) -> str:
    if match_status == "known_at_trade":
        if age_days is None or pd.isna(age_days):
            return "unknown_known_signal_age"
        return "fresh_known_signal" if int(age_days) <= SIGNAL_VALID_DAYS else "stale_known_signal"
    if match_status == "current_snapshot_only":
        return "current_snapshot_only"
    return "no_model_signal"


def _execution_review_action(validity_status: str) -> str:
    if validity_status == "fresh_known_signal":
        return "usable_for_trade_review"
    if validity_status == "stale_known_signal":
        return "requires_retrigger_before_trade"
    if validity_status == "current_snapshot_only":
        return "cannot_backfill_signal"
    return "needs_manual_case_review"


def build_broker_signal_match(
    trades: pd.DataFrame,
    target_signals: pd.DataFrame,
    review_signals: pd.DataFrame,
    history_signals: pd.DataFrame,
    realized_by_symbol: dict[str, float] | None = None,
) -> pd.DataFrame:
    realized_by_symbol = realized_by_symbol or {}
    target_signals = _normalize_signal_frame(target_signals)
    review_signals = _normalize_signal_frame(review_signals)
    history_signals = _normalize_signal_frame(history_signals)
    rows: list[dict[str, Any]] = []
    for _, trade in trades.iterrows():
        symbol = str(trade["symbol"])
        trade_date = pd.to_datetime(trade["trade_date_ts"])
        target, target_timing = _pick_signal(target_signals, symbol, trade_date)
        review, review_timing = _pick_signal(review_signals, symbol, trade_date)
        history, history_timing = _pick_signal(history_signals, symbol, trade_date)
        signal_matches = [(target, target_timing), (review, review_timing), (history, history_timing)]
        known_date = _known_signal_date(signal_matches)
        signal_age_days = _signal_age_days(trade_date, known_date)
        known = any(timing == "known_at_trade" for timing in [target_timing, review_timing, history_timing])
        future_only = any(timing in {"after_trade_snapshot", "undated_snapshot"} for timing in [target_timing, review_timing, history_timing])
        if known:
            match_status = "known_at_trade"
            notes = "至少一个模型信号日期不晚于真实成交日，可作为成交时已知信号对照。"
        elif future_only:
            match_status = "current_snapshot_only"
            notes = "仅在成交日之后或无日期的截面中匹配；不能把当前截面倒推解释为当日信号。"
        else:
            match_status = "no_model_signal"
            notes = "未在当前已接入的模型目标池、复核池或结果历史中找到该标的。"
        validity_status = _signal_validity_status(match_status, signal_age_days)
        rows.append(
            {
                "statement_row_id": _value(trade, "statement_row_id"),
                "trade_date": _value(trade, "trade_date"),
                "symbol": symbol,
                "name": _value(trade, "name"),
                "side": _value(trade, "side"),
                "quantity": _value(trade, "quantity", 0.0),
                "price": _value(trade, "price", 0.0),
                "gross_amount": _value(trade, "gross_amount", 0.0),
                "settlement_amount": _value(trade, "settlement_amount", 0.0),
                "known_signal_at_trade": bool(known),
                "match_status": match_status,
                "known_signal_date": known_date.strftime("%Y-%m-%d") if known_date is not None else "",
                "signal_age_days": signal_age_days,
                "signal_age_bucket": _signal_age_bucket(signal_age_days),
                "signal_valid_until": _signal_valid_until(known_date),
                "signal_validity_status": validity_status,
                "execution_review_action": _execution_review_action(validity_status),
                "target_match_timing": target_timing,
                "target_signal_date": _signal_date(target),
                "target_layer": _value(target, "layer"),
                "target_weight": _value(target, "portfolio_target_weight", 0.0),
                "target_action": _value(target, "target_action"),
                "target_trigger_status": _value(target, "trigger_monitor_status"),
                "target_trigger_summary": _value(target, "trigger_summary"),
                "review_match_timing": review_timing,
                "review_signal_date": _signal_date(review),
                "matched_review_bucket": _value(review, "review_bucket", _value(history, "review_bucket")),
                "matched_review_stage": _value(review, "review_stage", _value(history, "review_stage")),
                "matched_manual_status": _value(review, "manual_status_normalized"),
                "matched_recommended_review": _value(review, "recommended_review"),
                "history_match_timing": history_timing,
                "history_signal_date": _signal_date(history),
                "history_review_bucket": _value(history, "review_bucket", _value(history, "signal_source")),
                "history_review_stage": _value(history, "review_stage", _value(history, "score_bucket")),
                "history_return_1d": _value(history, "return_1d", 0.0),
                "symbol_realized_pnl": realized_by_symbol.get(symbol, 0.0),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=MATCH_COLUMNS)


def _summarize_match(match: pd.DataFrame, broker_dir: Path, paper_account_dir: Path) -> dict[str, Any]:
    buy = match[match["side"] == "buy"] if not match.empty else match
    sell = match[match["side"] == "sell"] if not match.empty else match
    buy_known = buy[buy["match_status"] == "known_at_trade"]
    buy_future = buy[buy["match_status"] == "current_snapshot_only"]
    buy_none = buy[buy["match_status"] == "no_model_signal"]
    age_counts: dict[str, int] = {}
    if not buy_known.empty and "signal_age_bucket" in buy_known.columns:
        raw_age_counts = buy_known["signal_age_bucket"].fillna("not_known").astype(str).value_counts().to_dict()
        age_counts = {bucket: int(raw_age_counts[bucket]) for bucket in AGE_BUCKET_ORDER if bucket in raw_age_counts}
    fresh_known = sum(age_counts.get(bucket, 0) for bucket in ("0_1d", "2_5d"))
    stale_known = sum(age_counts.get(bucket, 0) for bucket in ("6_10d", "gt10d"))
    validity_counts: dict[str, int] = {}
    if not buy.empty and "signal_validity_status" in buy.columns:
        validity_counts = {
            str(key): int(value)
            for key, value in buy["signal_validity_status"].fillna("missing").astype(str).value_counts().to_dict().items()
        }
    total_buy = len(buy)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "matched" if not match.empty else "empty",
        "broker_action": "none",
        "data_usage": "research_only_actual_vs_model_signal_match",
        "broker_dir": str(broker_dir),
        "paper_account_dir": str(paper_account_dir),
        "row_count": int(len(match)),
        "buy_rows": int(total_buy),
        "sell_rows": int(len(sell)),
        "buy_known_signal_rows": int(len(buy_known)),
        "buy_current_snapshot_only_rows": int(len(buy_future)),
        "buy_no_model_signal_rows": int(len(buy_none)),
        "buy_known_signal_age_buckets": age_counts,
        "buy_fresh_known_signal_rows": int(fresh_known),
        "buy_stale_known_signal_rows": int(stale_known),
        "buy_signal_validity_status_counts": validity_counts,
        "buy_valid_fresh_signal_rows": int(validity_counts.get("fresh_known_signal", 0)),
        "buy_requires_retrigger_rows": int(validity_counts.get("stale_known_signal", 0)),
        "buy_cannot_backfill_signal_rows": int(validity_counts.get("current_snapshot_only", 0)),
        "buy_known_signal_ratio": float(len(buy_known) / total_buy) if total_buy else 0.0,
        "buy_current_snapshot_only_ratio": float(len(buy_future) / total_buy) if total_buy else 0.0,
        "buy_no_model_signal_ratio": float(len(buy_none) / total_buy) if total_buy else 0.0,
        "matched_symbol_count": int(match.loc[match["match_status"] != "no_model_signal", "symbol"].nunique())
        if not match.empty
        else 0,
        "known_signal_symbol_count": int(match.loc[match["match_status"] == "known_at_trade", "symbol"].nunique())
        if not match.empty
        else 0,
        "research_conclusion": (
            "已建立真实成交与模型信号的只读对账。known_at_trade 可用于成交时点复盘；"
            "current_snapshot_only 只能说明当前模型仍关注该标的，不能倒推为当日信号。"
        ),
    }


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _money(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 20) -> str:
    if not rows:
        return "暂无数据。"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows[:max_rows]:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, sep, *body])


def _render_report(snapshot: dict[str, Any], match: pd.DataFrame) -> str:
    summary_rows = [
        {"项目": "真实成交流水", "数值": snapshot["row_count"]},
        {"项目": "买入 / 卖出流水", "数值": f"{snapshot['buy_rows']} / {snapshot['sell_rows']}"},
        {
            "项目": "买入成交时已有模型信号",
            "数值": f"{snapshot['buy_known_signal_rows']} ({_pct(snapshot['buy_known_signal_ratio'])})",
        },
        {
            "项目": "买入仅当前截面匹配",
            "数值": f"{snapshot['buy_current_snapshot_only_rows']} ({_pct(snapshot['buy_current_snapshot_only_ratio'])})",
        },
        {
            "项目": "买入未匹配模型信号",
            "数值": f"{snapshot['buy_no_model_signal_rows']} ({_pct(snapshot['buy_no_model_signal_ratio'])})",
        },
        {
            "项目": "买入已知信号新鲜 / 偏旧",
            "数值": f"{snapshot['buy_fresh_known_signal_rows']} / {snapshot['buy_stale_known_signal_rows']}",
        },
        {
            "项目": "买入可复盘 / 需重新触发",
            "数值": f"{snapshot['buy_valid_fresh_signal_rows']} / {snapshot['buy_requires_retrigger_rows']}",
        },
    ]
    age_labels = {
        "0_1d": "0-1天",
        "2_5d": "2-5天",
        "6_10d": "6-10天",
        "gt10d": "10天以上",
    }
    age_counts = snapshot.get("buy_known_signal_age_buckets", {})
    age_rows = [
        {"信号新鲜度": age_labels[bucket], "买入笔数": age_counts.get(bucket, 0)} for bucket in AGE_BUCKET_ORDER
    ]
    buy_rows = []
    if not match.empty:
        buys = match[match["side"] == "buy"].copy()
        for _, row in buys.head(30).iterrows():
            buy_rows.append(
                {
                    "日期": row["trade_date"],
                    "证券": f"{row['symbol']} {row['name']}",
                    "状态": row["match_status"],
                    "信号日": row.get("known_signal_date", ""),
                    "新鲜度": row.get("signal_age_bucket", ""),
                    "有效性": row.get("signal_validity_status", ""),
                    "处理建议": row.get("execution_review_action", ""),
                    "目标时点": row["target_match_timing"],
                    "复核桶": row["matched_review_bucket"],
                    "成交额": _money(row["gross_amount"]),
                    "说明": row["notes"],
                }
            )
    return f"""# 真实成交与模型信号对账

<!-- internal: Broker Signal Match Review -->

生成时间：`{snapshot["generated_at"]}`

<span class="badge">研究模式</span> broker_action=none，不连接券商，不自动下单。

## 结论

{snapshot["research_conclusion"]}

## 匹配口径

- `known_at_trade`：至少一个模型信号日期不晚于真实成交日。
- `current_snapshot_only`：只在成交日之后的最新截面或无日期截面里匹配，不能把当前截面倒推解释为当日信号。
- `no_model_signal`：当前接入的目标池、复核池和结果历史都没有匹配。

## 摘要

{_markdown_table(summary_rows, ["项目", "数值"], max_rows=20)}

## 信号新鲜度

{_markdown_table(age_rows, ["信号新鲜度", "买入笔数"], max_rows=20)}

## 买入成交逐笔对账

{_markdown_table(buy_rows, ["日期", "证券", "状态", "信号日", "新鲜度", "有效性", "处理建议", "目标时点", "复核桶", "成交额", "说明"], max_rows=30)}

## 输出文件

- `actual_trade_signal_match.csv`：真实成交与模型信号逐笔匹配。
- `actual_trade_signal_match_cn.csv`：中文表头版本。
- `broker_signal_match_snapshot.json`：每日流水线可读取的摘要快照。
"""


def run_broker_signal_match_review(
    broker_dir: str | Path = "outputs/research/broker_statement_latest",
    paper_account_dir: str | Path = "outputs/research/paper_account_latest",
    historical_signal_path: str | Path | None = None,
    output_dir: str | Path = "outputs/research/broker_signal_match_latest",
) -> BrokerSignalMatchReviewResult:
    broker_path = Path(broker_dir)
    paper_path = Path(paper_account_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = _load_broker_trades(broker_path)
    realized_by_symbol = _load_realized_by_symbol(broker_path)
    target_signals = _read_optional_csv(paper_path / "stock_targets.csv")
    review_signals = _read_optional_csv(paper_path / "stock_target_review.csv")
    history_signals = _read_optional_csv(paper_path / "stock_target_review_outcomes_history.csv")
    if historical_signal_path:
        historical_signals = _read_optional_csv(Path(historical_signal_path))
        if not historical_signals.empty:
            history_signals = pd.concat([history_signals, historical_signals], ignore_index=True)
    match = build_broker_signal_match(
        trades,
        target_signals=target_signals,
        review_signals=review_signals,
        history_signals=history_signals,
        realized_by_symbol=realized_by_symbol,
    )
    snapshot = _summarize_match(match, broker_path, paper_path)

    match_path = out_dir / "actual_trade_signal_match.csv"
    match_cn_path = out_dir / "actual_trade_signal_match_cn.csv"
    snapshot_path = out_dir / "broker_signal_match_snapshot.json"
    report_path = out_dir / "actual_execution_attribution.md"

    match.to_csv(match_path, index=False, encoding="utf-8-sig")
    match.rename(columns={key: value for key, value in MATCH_CN_COLUMNS.items() if key in match.columns}).to_csv(
        match_cn_path,
        index=False,
        encoding="utf-8-sig",
    )
    snapshot_path.write_text(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, match), encoding="utf-8")

    return BrokerSignalMatchReviewResult(
        output_dir=out_dir,
        match_path=match_path,
        match_cn_path=match_cn_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
    )
