"""Broker statement import and research-only execution audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


GF_COLUMNS = [
    "业务日期",
    "发生时间",
    "流水序号",
    "资金账号",
    "证券代码",
    "证券名称",
    "业务标志名称",
    "成交数量",
    "成交价格",
    "净佣金",
    "印花税",
    "过户费",
    "证管费",
    "经手费",
    "其他费",
    "清算金额",
    "货币名称",
    "委托编号",
    "应计利息",
]

NUMERIC_COLUMNS = ["成交数量", "成交价格", "净佣金", "印花税", "过户费", "证管费", "经手费", "其他费", "清算金额", "应计利息"]
FEE_COLUMNS = ["净佣金", "印花税", "过户费", "证管费", "经手费", "其他费"]

CN_TRADE_COLUMNS = {
    "trade_date": "交易日期",
    "trade_time": "成交时间",
    "statement_row_id": "脱敏流水ID",
    "symbol": "证券代码",
    "name": "证券名称",
    "business": "业务类型",
    "side": "方向",
    "quantity": "成交数量",
    "signed_quantity": "带方向数量",
    "price": "成交价格",
    "gross_amount": "成交金额",
    "commission": "净佣金",
    "stamp_tax": "印花税",
    "transfer_fee": "过户费",
    "regulatory_fee": "证管费",
    "handling_fee": "经手费",
    "other_fee": "其他费",
    "total_fee": "费用合计",
    "settlement_amount": "清算金额",
    "currency": "货币",
    "interest": "应计利息",
    "trade_datetime": "成交日期时间",
}

CN_REALIZED_COLUMNS = {
    "sell_date": "卖出日期",
    "symbol": "证券代码",
    "name": "证券名称",
    "sell_quantity": "卖出数量",
    "matched_quantity": "已匹配数量",
    "net_proceeds": "卖出净回款",
    "matched_cost": "匹配成本",
    "realized_pnl": "已实现盈亏",
    "return_on_cost": "成本收益率",
    "hold_days": "持有天数",
    "unmatched_quantity": "未匹配卖出数量",
}

CN_OPEN_COLUMNS = {
    "symbol": "证券代码",
    "name": "证券名称",
    "quantity": "剩余数量",
    "cost_total": "剩余成本",
    "avg_cost": "平均成本",
    "first_buy_date": "最早买入日期",
}

CN_SYMBOL_COLUMNS = {
    "symbol": "证券代码",
    "name": "证券名称",
    "rows": "流水条数",
    "buy_quantity": "买入数量",
    "sell_quantity": "卖出数量",
    "turnover": "成交额",
    "fees": "费用",
    "realized_pnl": "已实现盈亏",
}


@dataclass(frozen=True)
class BrokerStatementReviewResult:
    output_dir: Path
    normalized_path: Path
    normalized_cn_path: Path
    realized_path: Path
    realized_cn_path: Path
    open_positions_path: Path
    open_positions_cn_path: Path
    symbol_summary_path: Path
    symbol_summary_cn_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\t", "").strip()


def _to_float_series(series: pd.Series) -> pd.Series:
    cleaned = series.map(_clean_text)
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def _normalize_symbol(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    if "." in text:
        text = text.split(".", 1)[0]
    return text.zfill(6)


def _row_id(row: pd.Series) -> str:
    seed = "|".join(
        [
            _clean_text(row.get("业务日期")),
            _clean_text(row.get("发生时间")),
            _clean_text(row.get("流水序号")),
            _clean_text(row.get("证券代码")),
            _clean_text(row.get("成交数量")),
            _clean_text(row.get("清算金额")),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


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


def _format_money(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_num(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 10) -> str:
    if not rows:
        return "暂无数据。"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows[:max_rows]:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, sep, *body])


def _with_chinese_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    return frame.rename(columns={key: value for key, value in mapping.items() if key in frame.columns})


def normalize_broker_statement(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Guangfa PC statement export without retaining account identifiers."""

    if len(raw.columns) < len(GF_COLUMNS):
        raise ValueError(f"Broker statement requires at least {len(GF_COLUMNS)} columns, got {len(raw.columns)}.")

    data = raw.copy()
    if not set(GF_COLUMNS).issubset(set(data.columns)):
        rename_map = {old: new for old, new in zip(list(data.columns)[: len(GF_COLUMNS)], GF_COLUMNS)}
        data = data.rename(columns=rename_map)

    for column in data.columns:
        if data[column].dtype == object:
            data[column] = data[column].map(_clean_text)
    for column in NUMERIC_COLUMNS:
        data[column] = _to_float_series(data[column])

    signed_quantity = data["成交数量"].astype(float)
    trade_date = pd.to_datetime(data["业务日期"], errors="coerce")
    trade_time = data["发生时间"].map(_clean_text)
    trade_datetime = pd.to_datetime(
        trade_date.dt.strftime("%Y-%m-%d").fillna("") + " " + trade_time,
        errors="coerce",
    )
    gross_amount = (signed_quantity.abs() * data["成交价格"].astype(float)).round(6)
    total_fee = data[FEE_COLUMNS].sum(axis=1).round(6)
    side = signed_quantity.map(lambda value: "buy" if value > 0 else ("sell" if value < 0 else "other"))

    normalized = pd.DataFrame(
        {
            "trade_date": trade_date.dt.strftime("%Y-%m-%d"),
            "trade_time": trade_time,
            "statement_row_id": data.apply(_row_id, axis=1),
            "symbol": data["证券代码"].map(_normalize_symbol),
            "name": data["证券名称"].map(_clean_text),
            "business": data["业务标志名称"].map(_clean_text),
            "side": side,
            "quantity": signed_quantity.abs(),
            "signed_quantity": signed_quantity,
            "price": data["成交价格"].astype(float),
            "gross_amount": gross_amount,
            "commission": data["净佣金"].astype(float),
            "stamp_tax": data["印花税"].astype(float),
            "transfer_fee": data["过户费"].astype(float),
            "regulatory_fee": data["证管费"].astype(float),
            "handling_fee": data["经手费"].astype(float),
            "other_fee": data["其他费"].astype(float),
            "total_fee": total_fee,
            "settlement_amount": data["清算金额"].astype(float),
            "currency": data["货币名称"].map(_clean_text),
            "interest": data["应计利息"].astype(float),
            "trade_datetime": trade_datetime,
        }
    )
    return normalized.sort_values(["trade_datetime", "statement_row_id"]).reset_index(drop=True)


def load_broker_statement(path: str | Path, encoding: str = "utf-8-sig") -> pd.DataFrame:
    return normalize_broker_statement(pd.read_csv(path, encoding=encoding))


def build_fifo_analysis(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    open_lots: dict[str, list[dict[str, Any]]] = {}
    realized_rows: list[dict[str, Any]] = []
    sorted_trades = trades.sort_values(["trade_datetime", "statement_row_id"]).reset_index(drop=True)

    for _, row in sorted_trades.iterrows():
        symbol = str(row["symbol"])
        side = str(row["side"])
        quantity = float(row["quantity"] or 0.0)
        if quantity <= 0.0 or side == "other":
            continue
        if side == "buy":
            open_lots.setdefault(symbol, []).append(
                {
                    "symbol": symbol,
                    "name": row["name"],
                    "date": pd.to_datetime(row["trade_date"]),
                    "quantity": quantity,
                    "cost_total": abs(float(row["settlement_amount"] or 0.0)),
                }
            )
            continue
        if side != "sell":
            continue

        remaining = quantity
        matched_quantity = 0.0
        matched_cost = 0.0
        weighted_date = 0.0
        lots = open_lots.setdefault(symbol, [])
        while remaining > 1e-9 and lots:
            lot = lots[0]
            take = min(remaining, float(lot["quantity"]))
            cost_part = float(lot["cost_total"]) * take / float(lot["quantity"]) if lot["quantity"] else 0.0
            matched_quantity += take
            matched_cost += cost_part
            weighted_date += pd.Timestamp(lot["date"]).toordinal() * take
            lot["quantity"] = float(lot["quantity"]) - take
            lot["cost_total"] = float(lot["cost_total"]) - cost_part
            remaining -= take
            if float(lot["quantity"]) <= 1e-9:
                lots.pop(0)

        proceeds_per_share = float(row["settlement_amount"] or 0.0) / quantity if quantity else 0.0
        net_proceeds = proceeds_per_share * matched_quantity
        realized_pnl = net_proceeds - matched_cost if matched_quantity > 0.0 else float("nan")
        buy_date = (
            pd.Timestamp.fromordinal(int(round(weighted_date / matched_quantity)))
            if matched_quantity > 0.0
            else pd.NaT
        )
        sell_date = pd.to_datetime(row["trade_date"])
        hold_days = (sell_date - buy_date).days if pd.notna(buy_date) else pd.NA
        realized_rows.append(
            {
                "sell_date": sell_date.strftime("%Y-%m-%d") if pd.notna(sell_date) else "",
                "symbol": symbol,
                "name": row["name"],
                "sell_quantity": quantity,
                "matched_quantity": matched_quantity,
                "net_proceeds": round(net_proceeds, 6),
                "matched_cost": round(matched_cost, 6),
                "realized_pnl": round(realized_pnl, 6) if matched_quantity > 0.0 else float("nan"),
                "return_on_cost": realized_pnl / matched_cost if matched_cost else float("nan"),
                "hold_days": hold_days,
                "unmatched_quantity": round(max(remaining, 0.0), 6),
            }
        )

    open_rows: list[dict[str, Any]] = []
    for lots in open_lots.values():
        for lot in lots:
            quantity = float(lot["quantity"])
            if quantity <= 1e-9:
                continue
            open_rows.append(
                {
                    "symbol": lot["symbol"],
                    "name": lot["name"],
                    "quantity": round(quantity, 6),
                    "cost_total": round(float(lot["cost_total"]), 6),
                    "avg_cost": float(lot["cost_total"]) / quantity if quantity else float("nan"),
                    "first_buy_date": pd.Timestamp(lot["date"]).strftime("%Y-%m-%d"),
                }
            )

    realized = pd.DataFrame(realized_rows, columns=list(CN_REALIZED_COLUMNS))
    open_positions = pd.DataFrame(open_rows, columns=list(CN_OPEN_COLUMNS))
    if not open_positions.empty:
        open_positions = (
            open_positions.groupby(["symbol", "name"], as_index=False)
            .agg(
                quantity=("quantity", "sum"),
                cost_total=("cost_total", "sum"),
                avg_cost=("avg_cost", "mean"),
                first_buy_date=("first_buy_date", "min"),
            )
            .reset_index(drop=True)
        )
    return realized, open_positions


def build_symbol_summary(trades: pd.DataFrame, realized: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=list(CN_SYMBOL_COLUMNS))
    summary = (
        trades.groupby(["symbol", "name"], as_index=False)
        .agg(
            rows=("symbol", "size"),
            buy_quantity=("signed_quantity", lambda values: float(values[values > 0].sum())),
            sell_quantity=("signed_quantity", lambda values: float(-values[values < 0].sum())),
            turnover=("gross_amount", "sum"),
            fees=("total_fee", "sum"),
        )
        .reset_index(drop=True)
    )
    if realized.empty:
        summary["realized_pnl"] = 0.0
    else:
        pnl = realized.groupby("symbol", as_index=False)["realized_pnl"].sum()
        summary = summary.merge(pnl, on="symbol", how="left")
        summary["realized_pnl"] = summary["realized_pnl"].fillna(0.0)
    return summary.sort_values("turnover", ascending=False).reset_index(drop=True)


def summarize_statement(
    trades: pd.DataFrame,
    realized: pd.DataFrame,
    open_positions: pd.DataFrame,
    symbol_summary: pd.DataFrame,
    source_path: Path,
) -> dict[str, Any]:
    buy = trades[trades["side"] == "buy"]
    sell = trades[trades["side"] == "sell"]
    valid_realized = realized[pd.to_numeric(realized["matched_quantity"], errors="coerce").fillna(0.0) > 0.0]
    realized_pnl = float(pd.to_numeric(valid_realized["realized_pnl"], errors="coerce").fillna(0.0).sum())
    win_rate = (
        float((pd.to_numeric(valid_realized["realized_pnl"], errors="coerce") > 0.0).mean())
        if not valid_realized.empty
        else 0.0
    )
    avg_return = float(pd.to_numeric(valid_realized["return_on_cost"], errors="coerce").mean()) if not valid_realized.empty else 0.0
    median_hold_days = (
        float(pd.to_numeric(valid_realized["hold_days"], errors="coerce").median()) if not valid_realized.empty else 0.0
    )
    turnover = float(trades["gross_amount"].sum()) if not trades.empty else 0.0
    top_symbols = symbol_summary.head(10)[["symbol", "name", "turnover", "realized_pnl"]].to_dict("records")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "usable_for_research_audit" if not trades.empty else "empty",
        "broker_action": "none",
        "data_usage": "research_only_actual_trade_audit",
        "source_file_name": source_path.name,
        "source_file_size": source_path.stat().st_size if source_path.exists() else None,
        "account_identifier_written": False,
        "row_count": int(len(trades)),
        "start_date": str(trades["trade_date"].min()) if not trades.empty else "",
        "end_date": str(trades["trade_date"].max()) if not trades.empty else "",
        "trade_days": int(trades["trade_date"].nunique()) if not trades.empty else 0,
        "symbol_count": int(trades["symbol"].nunique()) if not trades.empty else 0,
        "buy_rows": int(len(buy)),
        "sell_rows": int(len(sell)),
        "other_rows": int(len(trades) - len(buy) - len(sell)),
        "gross_buy": float(buy["gross_amount"].sum()),
        "gross_sell": float(sell["gross_amount"].sum()),
        "turnover": turnover,
        "total_fee": float(trades["total_fee"].sum()),
        "commission": float(trades["commission"].sum()),
        "stamp_tax": float(trades["stamp_tax"].sum()),
        "fee_rate_on_turnover": float(trades["total_fee"].sum() / turnover) if turnover else 0.0,
        "net_cashflow": float(trades["settlement_amount"].sum()),
        "closed_sell_rows": int(len(realized)),
        "matched_sell_rows": int(len(valid_realized)),
        "unmatched_sell_rows": int((pd.to_numeric(realized["unmatched_quantity"], errors="coerce").fillna(0.0) > 0.0).sum())
        if not realized.empty
        else 0,
        "realized_pnl": realized_pnl,
        "win_rate": win_rate,
        "average_return_on_cost": avg_return,
        "median_hold_days": median_hold_days,
        "open_position_count": int(len(open_positions)),
        "top_turnover_symbols": top_symbols,
        "research_conclusion": (
            "可纳入实盘前置研究体系，优先用于真实成交审计、费用校准、人工执行偏差复盘和模型建议对照；"
            "样本期较短，暂不直接作为模型训练标签。"
        ),
    }


def _render_report(
    snapshot: dict[str, Any],
    symbol_summary: pd.DataFrame,
    realized: pd.DataFrame,
    open_positions: pd.DataFrame,
) -> str:
    summary_rows = [
        {"项目": "样本区间", "数值": f"{snapshot['start_date']} 至 {snapshot['end_date']}"},
        {"项目": "流水 / 交易日 / 标的", "数值": f"{snapshot['row_count']} / {snapshot['trade_days']} / {snapshot['symbol_count']}"},
        {"项目": "买入 / 卖出流水", "数值": f"{snapshot['buy_rows']} / {snapshot['sell_rows']}"},
        {"项目": "总成交额", "数值": _format_money(snapshot["turnover"])},
        {"项目": "总费用 / 费率", "数值": f"{_format_money(snapshot['total_fee'])} / {_format_pct(snapshot['fee_rate_on_turnover'])}"},
        {"项目": "已实现盈亏", "数值": _format_money(snapshot["realized_pnl"])},
        {"项目": "胜率 / 平均成本收益率", "数值": f"{_format_pct(snapshot['win_rate'])} / {_format_pct(snapshot['average_return_on_cost'])}"},
        {"项目": "中位持有天数", "数值": _format_num(snapshot["median_hold_days"])},
        {"项目": "期末未平仓标的数", "数值": snapshot["open_position_count"]},
    ]
    top_symbols = []
    for _, row in symbol_summary.head(10).iterrows():
        top_symbols.append(
            {
                "证券代码": row["symbol"],
                "证券名称": row["name"],
                "成交额": _format_money(row["turnover"]),
                "费用": _format_money(row["fees"]),
                "已实现盈亏": _format_money(row["realized_pnl"]),
            }
        )
    best = []
    worst = []
    if not realized.empty:
        ranked = realized.copy()
        ranked["realized_pnl"] = pd.to_numeric(ranked["realized_pnl"], errors="coerce")
        for _, row in ranked.sort_values("realized_pnl", ascending=False).head(5).iterrows():
            best.append(
                {
                    "日期": row["sell_date"],
                    "证券": f"{row['symbol']} {row['name']}",
                    "盈亏": _format_money(row["realized_pnl"]),
                    "收益率": _format_pct(row["return_on_cost"]),
                    "持有天数": row["hold_days"],
                }
            )
        for _, row in ranked.sort_values("realized_pnl", ascending=True).head(5).iterrows():
            worst.append(
                {
                    "日期": row["sell_date"],
                    "证券": f"{row['symbol']} {row['name']}",
                    "盈亏": _format_money(row["realized_pnl"]),
                    "收益率": _format_pct(row["return_on_cost"]),
                    "持有天数": row["hold_days"],
                }
            )
    open_rows = []
    for _, row in open_positions.head(10).iterrows():
        open_rows.append(
            {
                "证券": f"{row['symbol']} {row['name']}",
                "剩余数量": _format_num(row["quantity"]),
                "剩余成本": _format_money(row["cost_total"]),
                "平均成本": _format_money(row["avg_cost"]),
                "最早买入": row["first_buy_date"],
            }
        )
    return f"""# 券商交割单研究审计

<!-- internal: Broker Statement Research Audit -->

生成时间：`{snapshot["generated_at"]}`

<span class="badge">研究模式</span> broker_action=none，不连接券商，不自动下单。原始资金账号、流水序号和委托编号不会写入标准化输出。

## 结论

{snapshot["research_conclusion"]}

## 数据摘要

{_markdown_table(summary_rows, ["项目", "数值"], max_rows=20)}

## 成交额最高标的

{_markdown_table(top_symbols, ["证券代码", "证券名称", "成交额", "费用", "已实现盈亏"], max_rows=10)}

## 已实现交易表现

### 收益最高

{_markdown_table(best, ["日期", "证券", "盈亏", "收益率", "持有天数"], max_rows=5)}

### 回撤最大

{_markdown_table(worst, ["日期", "证券", "盈亏", "收益率", "持有天数"], max_rows=5)}

## 期末未平仓

{_markdown_table(open_rows, ["证券", "剩余数量", "剩余成本", "平均成本", "最早买入"], max_rows=10)}

## 可纳入体系的位置

- 真实成交审计：对比每日纸面账户和 live-shadow 计划，衡量人工执行偏差。
- 成本校准：用真实佣金、印花税、过户费、证管费、经手费估计执行成本。
- 行为复盘：观察持有天数、追涨/止损、单票集中度和高换手标的。
- 信号借鉴：只作为候选信号的事后对照样本，不直接作为训练标签。

## 输出文件

- `broker_trades_normalized.csv`：脱敏标准化成交流水，英文字段供程序读取。
- `broker_trades_normalized_cn.csv`：脱敏标准化成交流水，中文表头供人工查看。
- `broker_realized_trades.csv`：FIFO 已实现盈亏明细。
- `broker_open_positions.csv`：按交割单 FIFO 估计的期末未平仓成本。
- `broker_symbol_summary.csv`：按标的汇总的成交额、费用和已实现盈亏。
- `broker_statement_snapshot.json`：供后续每日流水线读取的摘要快照。
"""


def run_broker_statement_review(
    input_path: str | Path,
    output_dir: str | Path = "outputs/research/broker_statement_latest",
    encoding: str = "utf-8-sig",
) -> BrokerStatementReviewResult:
    source = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = load_broker_statement(source, encoding=encoding)
    realized, open_positions = build_fifo_analysis(trades)
    symbol_summary = build_symbol_summary(trades, realized)
    snapshot = summarize_statement(trades, realized, open_positions, symbol_summary, source)

    normalized_path = out_dir / "broker_trades_normalized.csv"
    normalized_cn_path = out_dir / "broker_trades_normalized_cn.csv"
    realized_path = out_dir / "broker_realized_trades.csv"
    realized_cn_path = out_dir / "broker_realized_trades_cn.csv"
    open_positions_path = out_dir / "broker_open_positions.csv"
    open_positions_cn_path = out_dir / "broker_open_positions_cn.csv"
    symbol_summary_path = out_dir / "broker_symbol_summary.csv"
    symbol_summary_cn_path = out_dir / "broker_symbol_summary_cn.csv"
    snapshot_path = out_dir / "broker_statement_snapshot.json"
    report_path = out_dir / "broker_statement_review.md"

    trades.to_csv(normalized_path, index=False, encoding="utf-8-sig")
    _with_chinese_columns(trades, CN_TRADE_COLUMNS).to_csv(normalized_cn_path, index=False, encoding="utf-8-sig")
    realized.to_csv(realized_path, index=False, encoding="utf-8-sig")
    _with_chinese_columns(realized, CN_REALIZED_COLUMNS).to_csv(realized_cn_path, index=False, encoding="utf-8-sig")
    open_positions.to_csv(open_positions_path, index=False, encoding="utf-8-sig")
    _with_chinese_columns(open_positions, CN_OPEN_COLUMNS).to_csv(open_positions_cn_path, index=False, encoding="utf-8-sig")
    symbol_summary.to_csv(symbol_summary_path, index=False, encoding="utf-8-sig")
    _with_chinese_columns(symbol_summary, CN_SYMBOL_COLUMNS).to_csv(
        symbol_summary_cn_path,
        index=False,
        encoding="utf-8-sig",
    )
    snapshot_path.write_text(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, symbol_summary, realized, open_positions), encoding="utf-8")

    return BrokerStatementReviewResult(
        output_dir=out_dir,
        normalized_path=normalized_path,
        normalized_cn_path=normalized_cn_path,
        realized_path=realized_path,
        realized_cn_path=realized_cn_path,
        open_positions_path=open_positions_path,
        open_positions_cn_path=open_positions_cn_path,
        symbol_summary_path=symbol_summary_path,
        symbol_summary_cn_path=symbol_summary_cn_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
    )
