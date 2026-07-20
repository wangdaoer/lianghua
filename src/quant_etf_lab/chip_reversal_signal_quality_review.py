"""Signal-quality review for chip-reversal satellite trades."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TRADES_PATH = Path(
    "outputs/research/gain_gt5_chip_confirm_2018_2025_20260618/true_position_h5_new5_total25_exp80/selected_events.csv"
)
DEFAULT_EVENTS_PATH = Path(
    "outputs/research/gain_gt5_chip_confirm_2018_2025_20260618/gain_gt5_chip_deep_confirm_events_sanity.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/research/chip_reversal_signal_quality_latest")


@dataclass(frozen=True)
class ChipReversalSignalQualityReviewResult:
    output_dir: Path
    rank_path: Path
    feature_bins_path: Path
    worst_trades_path: Path
    summary_path: Path
    report_path: Path
    rank: pd.DataFrame
    feature_bins: pd.DataFrame
    worst_trades: pd.DataFrame
    summary: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return value


def _clean_code(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    digits = "".join(char for char in text if char.isdigit())
    return digits.zfill(6)[-6:] if digits else ""


def _profit_factor(returns: pd.Series) -> float | None:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    gross_profit = float(clean[clean > 0.0].sum())
    gross_loss = abs(float(clean[clean < 0.0].sum()))
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0.0 else None
    return gross_profit / gross_loss


def _rank_bucket(value: Any) -> str:
    rank = int(float(value))
    if rank == 1:
        return "rank_1"
    if rank <= 3:
        return "rank_2_3"
    return "rank_4_plus"


def _score_bucket(value: Any) -> str:
    number = float(value)
    if number < 0.10:
        return "score_lt10"
    if number < 0.15:
        return "score_10_15"
    return "score_gte15"


def _amount_bucket(value: Any) -> str:
    number = float(value)
    if number < 5.0:
        return "amount_lt5yi"
    if number < 20.0:
        return "amount_5_20yi"
    return "amount_gte20yi"


def _drawdown_bucket(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    number = float(value)
    if number <= -0.20:
        return "deep_gte20pct"
    if number <= -0.10:
        return "mid_10_20pct"
    return "shallow_lt10pct"


def _gain_bucket(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    number = float(value)
    if number < 7.0:
        return "gain_5_7pct"
    if number < 10.0:
        return "gain_7_10pct"
    return "gain_gte10pct"


def _gain_fine_bucket(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    number = float(value)
    if number < 7.0:
        return "gain_5_7pct"
    if number < 10.0:
        return "gain_7_10pct"
    if number < 15.0:
        return "gain_10_15pct"
    if number < 20.0:
        return "gain_15_20pct"
    return "gain_20pct_plus"


def _cost_gap_bucket(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    number = float(value)
    if number < -0.03:
        return "cost_gap_below_ma3pct"
    if number <= 0.03:
        return "cost_gap_near_ma"
    return "cost_gap_above_ma3pct"


def _prepare_trades(trades: pd.DataFrame, events: pd.DataFrame | None) -> pd.DataFrame:
    required = {"signal_date", "code", "candidate_rank", "chip_reversal_score", "amount_yi", "trade_return_after_cost"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"trades missing columns: {sorted(missing)}")
    data = trades.copy()
    data["signal_date"] = pd.to_datetime(data["signal_date"], errors="coerce")
    data["code"] = data["code"].map(_clean_code)
    data["candidate_rank"] = pd.to_numeric(data["candidate_rank"], errors="coerce")
    data["chip_reversal_score"] = pd.to_numeric(data["chip_reversal_score"], errors="coerce")
    data["amount_yi"] = pd.to_numeric(data["amount_yi"], errors="coerce")
    data["trade_return_after_cost"] = pd.to_numeric(data["trade_return_after_cost"], errors="coerce")
    if "status" in data.columns:
        data = data[data["status"].astype(str).eq("exited")].copy()
    data = data.dropna(
        subset=["signal_date", "code", "candidate_rank", "chip_reversal_score", "amount_yi", "trade_return_after_cost"]
    )
    data = data[data["code"] != ""].copy()
    if events is not None and not events.empty:
        event_features = events.copy()
        if "date" not in event_features.columns or "code" not in event_features.columns:
            raise ValueError("events must contain date and code columns.")
        event_features["signal_date"] = pd.to_datetime(event_features["date"], errors="coerce")
        event_features["code"] = event_features["code"].map(_clean_code)
        keep = [
            column
            for column in [
                "signal_date",
                "code",
                "drawdown_20d",
                "cost_gap_ma20",
                "daily_return_pct",
                "signal_type",
            ]
            if column in event_features.columns
        ]
        event_features = event_features[keep].drop_duplicates(subset=["signal_date", "code"], keep="last")
        data = data.merge(event_features, on=["signal_date", "code"], how="left")
    for column in ("drawdown_20d", "cost_gap_ma20", "daily_return_pct"):
        if column not in data.columns:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if "board" not in data.columns:
        data["board"] = "unknown"
    data["rank_bucket"] = data["candidate_rank"].map(_rank_bucket)
    data["chip_score_bucket"] = data["chip_reversal_score"].map(_score_bucket)
    data["amount_bucket"] = data["amount_yi"].map(_amount_bucket)
    data["drawdown_bucket"] = data["drawdown_20d"].map(_drawdown_bucket)
    data["signal_gain_bucket"] = data["daily_return_pct"].map(_gain_bucket)
    data["signal_gain_fine_bucket"] = data["daily_return_pct"].map(_gain_fine_bucket)
    data["cost_gap_bucket"] = data["cost_gap_ma20"].map(_cost_gap_bucket)
    data["signal_month"] = data["signal_date"].dt.strftime("%Y-%m")
    data["win"] = data["trade_return_after_cost"] > 0.0
    return data.reset_index(drop=True)


def _bucket_summary(data: pd.DataFrame, group_columns: list[str], min_bucket_trades: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in data.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        trade_count = int(len(group))
        if trade_count < min_bucket_trades:
            continue
        returns = pd.to_numeric(group["trade_return_after_cost"], errors="coerce")
        row = {column: str(value) for column, value in zip(group_columns, keys, strict=False)}
        row.update(
            {
                "trade_count": trade_count,
                "win_rate": float((returns > 0.0).mean()),
                "avg_trade_return": float(returns.mean()),
                "median_trade_return": float(returns.median()),
                "sum_trade_return": float(returns.sum()),
                "profit_factor": _profit_factor(returns),
                "avg_candidate_rank": float(pd.to_numeric(group["candidate_rank"], errors="coerce").mean()),
                "avg_chip_reversal_score": float(pd.to_numeric(group["chip_reversal_score"], errors="coerce").mean()),
                "avg_amount_yi": float(pd.to_numeric(group["amount_yi"], errors="coerce").mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_summaries(data: pd.DataFrame, min_bucket_trades: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for feature in [
        "board",
        "rank_bucket",
        "chip_score_bucket",
        "amount_bucket",
        "drawdown_bucket",
        "signal_gain_bucket",
        "signal_gain_fine_bucket",
        "cost_gap_bucket",
    ]:
        summary = _bucket_summary(data, [feature], min_bucket_trades)
        if summary.empty:
            continue
        summary = summary.rename(columns={feature: "bucket"})
        summary.insert(0, "feature", feature)
        frames.append(summary)
    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True)
    return output.sort_values(["avg_trade_return", "trade_count"], ascending=[True, False]).reset_index(drop=True)


def _summary(data: pd.DataFrame, feature_bins: pd.DataFrame) -> dict[str, Any]:
    worst_feature_bucket = None
    if not feature_bins.empty:
        priority = {
            "drawdown_bucket": 0,
            "chip_score_bucket": 1,
            "signal_gain_bucket": 2,
            "signal_gain_fine_bucket": 3,
            "amount_bucket": 4,
            "cost_gap_bucket": 5,
            "rank_bucket": 6,
            "board": 7,
        }
        ranked = feature_bins.copy()
        ranked["_feature_priority"] = ranked["feature"].map(priority).fillna(99)
        row = ranked.sort_values(
            ["avg_trade_return", "_feature_priority", "trade_count"],
            ascending=[True, True, False],
        ).iloc[0]
        worst_feature_bucket = f"{row['feature']}={row['bucket']}"
    returns = pd.to_numeric(data["trade_return_after_cost"], errors="coerce")
    extreme_signal_gain = data[pd.to_numeric(data["daily_return_pct"], errors="coerce") > 20.0]
    extreme_returns = pd.to_numeric(extreme_signal_gain["trade_return_after_cost"], errors="coerce")
    return {
        "status": "ok",
        "research_only": True,
        "broker_action": "none",
        "completed_trade_count": int(len(data)),
        "win_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
        "avg_trade_return": float(returns.mean()) if len(returns) else 0.0,
        "median_trade_return": float(returns.median()) if len(returns) else 0.0,
        "sum_trade_return": float(returns.sum()) if len(returns) else 0.0,
        "profit_factor": _profit_factor(returns),
        "worst_feature_bucket": worst_feature_bucket,
        "extreme_signal_gain_count": int(len(extreme_signal_gain)),
        "extreme_signal_gain_ratio": float(len(extreme_signal_gain) / len(data)) if len(data) else 0.0,
        "extreme_signal_gain_avg_trade_return": float(extreme_returns.mean()) if len(extreme_returns) else 0.0,
    }


def build_chip_reversal_signal_quality_review(
    trades: pd.DataFrame,
    *,
    events: pd.DataFrame | None = None,
    min_bucket_trades: int = 30,
    worst_trade_count: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if min_bucket_trades < 1:
        raise ValueError("min_bucket_trades must be at least 1.")
    data = _prepare_trades(trades, events)
    rank = _bucket_summary(data, ["rank_bucket"], min_bucket_trades)
    if not rank.empty:
        rank = rank.sort_values("rank_bucket").reset_index(drop=True)
    feature_bins = _feature_summaries(data, min_bucket_trades)
    worst_columns = [
        "signal_date",
        "code",
        "name",
        "board",
        "candidate_rank",
        "chip_reversal_score",
        "amount_yi",
        "drawdown_20d",
        "cost_gap_ma20",
        "daily_return_pct",
        "trade_return_after_cost",
        "rank_bucket",
        "chip_score_bucket",
        "amount_bucket",
        "drawdown_bucket",
        "signal_gain_bucket",
        "signal_gain_fine_bucket",
        "cost_gap_bucket",
    ]
    worst_trades = data[data["trade_return_after_cost"] < 0.0].sort_values("trade_return_after_cost", ascending=True).head(
        int(worst_trade_count)
    )
    worst_trades = worst_trades[[column for column in worst_columns if column in worst_trades.columns]].reset_index(drop=True)
    return rank, feature_bins, worst_trades, _summary(data, feature_bins)


_RANK_CN_COLUMNS = {
    "rank_bucket": "候选排名分组",
    "trade_count": "交易数",
    "win_rate": "胜率",
    "avg_trade_return": "平均交易收益",
    "median_trade_return": "中位交易收益",
    "sum_trade_return": "交易收益合计",
    "profit_factor": "盈亏比",
    "avg_candidate_rank": "平均候选排名",
    "avg_chip_reversal_score": "平均筹码反包分数",
    "avg_amount_yi": "平均成交额亿元",
}

_FEATURE_CN_COLUMNS = {
    "feature": "特征",
    "bucket": "分组",
    "trade_count": "交易数",
    "win_rate": "胜率",
    "avg_trade_return": "平均交易收益",
    "median_trade_return": "中位交易收益",
    "sum_trade_return": "交易收益合计",
    "profit_factor": "盈亏比",
    "avg_candidate_rank": "平均候选排名",
    "avg_chip_reversal_score": "平均筹码反包分数",
    "avg_amount_yi": "平均成交额亿元",
}

_WORST_CN_COLUMNS = {
    "signal_date": "信号日期",
    "code": "代码",
    "name": "名称",
    "board": "板块",
    "candidate_rank": "候选排名",
    "chip_reversal_score": "筹码反包分数",
    "amount_yi": "成交额亿元",
    "drawdown_20d": "20日回撤",
    "cost_gap_ma20": "相对20日均成本",
    "daily_return_pct": "信号日涨幅",
    "trade_return_after_cost": "交易收益",
    "rank_bucket": "候选排名分组",
    "chip_score_bucket": "筹码分数组",
    "amount_bucket": "成交额分组",
    "drawdown_bucket": "回撤分组",
    "signal_gain_bucket": "涨幅分组",
    "signal_gain_fine_bucket": "细分涨幅分组",
    "cost_gap_bucket": "成本缺口分组",
}


def _to_chinese_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    output = frame.copy()
    if "signal_date" in output.columns:
        output["signal_date"] = pd.to_datetime(output["signal_date"]).dt.strftime("%Y-%m-%d")
    ordered = [column for column in mapping if column in output.columns]
    rest = [column for column in output.columns if column not in ordered]
    output = output[ordered + rest]
    return output.rename(columns=mapping)


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _fmt_num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, float) and np.isinf(value):
        return "inf"
    return f"{float(value):.3f}"


def _write_report(
    *,
    name: str,
    rank_cn: pd.DataFrame,
    feature_bins_cn: pd.DataFrame,
    worst_trades_cn: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    trades_path: Path,
    events_path: Path | None,
) -> Path:
    report_path = output_dir / "chip_reversal_signal_quality_report.html"
    rank_table = rank_cn.to_html(index=False, border=0, classes="data-table")
    feature_table = feature_bins_cn.head(60).to_html(index=False, border=0, classes="data-table")
    worst_table = worst_trades_cn.head(30).to_html(index=False, border=0, classes="data-table")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{name} 卫星事件质量诊断</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #1f2933; }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    h2 {{ font-size: 18px; margin-top: 28px; }}
    .meta {{ color: #52616b; line-height: 1.7; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 6px; background: #e8f5e9; color: #1b5e20; font-size: 12px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f0f4f8; }}
  </style>
</head>
<body>
  <h1>{name} 卫星事件质量诊断</h1>
  <div class="meta">
    <div><span class="badge">研究模式</span> broker_action=none，不开券商动作，不自动下单。</div>
    <div>交易文件：{trades_path}</div>
    <div>事件文件：{events_path if events_path is not None else "未提供"}</div>
    <div>完成交易数：{summary.get("completed_trade_count")}；胜率：{_fmt_pct(summary.get("win_rate"))}；平均交易收益：{_fmt_pct(summary.get("avg_trade_return"))}</div>
    <div>盈亏比：{_fmt_num(summary.get("profit_factor"))}；最弱分组：{summary.get("worst_feature_bucket")}</div>
    <div>信号日涨幅&gt;20%样本：{summary.get("extreme_signal_gain_count")}；占比：{_fmt_pct(summary.get("extreme_signal_gain_ratio"))}；平均交易收益：{_fmt_pct(summary.get("extreme_signal_gain_avg_trade_return"))}</div>
  </div>
  <h2>候选排名质量</h2>
  {rank_table}
  <h2>信号特征分组</h2>
  {feature_table}
  <h2>最差交易</h2>
  {worst_table}
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return report_path


def run_chip_reversal_signal_quality_review(
    *,
    trades_path: str | Path = DEFAULT_TRADES_PATH,
    events_path: str | Path | None = DEFAULT_EVENTS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    name: str = "chip_reversal_signal_quality",
    min_bucket_trades: int = 30,
    worst_trade_count: int = 50,
) -> ChipReversalSignalQualityReviewResult:
    resolved_trades_path = Path(trades_path)
    resolved_events_path = Path(events_path) if events_path is not None else None
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(resolved_trades_path, dtype={"code": str})
    events = pd.read_csv(resolved_events_path, dtype={"code": str}) if resolved_events_path is not None else None
    rank, feature_bins, worst_trades, summary = build_chip_reversal_signal_quality_review(
        trades,
        events=events,
        min_bucket_trades=min_bucket_trades,
        worst_trade_count=worst_trade_count,
    )
    summary = {
        **summary,
        "name": name,
        "trades_path": str(resolved_trades_path),
        "events_path": str(resolved_events_path) if resolved_events_path is not None else None,
    }
    rank_cn = _to_chinese_columns(rank, _RANK_CN_COLUMNS)
    feature_bins_cn = _to_chinese_columns(feature_bins, _FEATURE_CN_COLUMNS)
    worst_trades_cn = _to_chinese_columns(worst_trades, _WORST_CN_COLUMNS)
    rank_path = resolved_output / "chip_reversal_signal_quality_rank_cn.csv"
    feature_bins_path = resolved_output / "chip_reversal_signal_quality_feature_bins_cn.csv"
    worst_trades_path = resolved_output / "chip_reversal_signal_quality_worst_trades_cn.csv"
    summary_path = resolved_output / "chip_reversal_signal_quality_summary.json"
    rank_cn.to_csv(rank_path, index=False, encoding="utf-8-sig")
    feature_bins_cn.to_csv(feature_bins_path, index=False, encoding="utf-8-sig")
    worst_trades_cn.to_csv(worst_trades_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(
        name=name,
        rank_cn=rank_cn,
        feature_bins_cn=feature_bins_cn,
        worst_trades_cn=worst_trades_cn,
        summary=summary,
        output_dir=resolved_output,
        trades_path=resolved_trades_path,
        events_path=resolved_events_path,
    )
    return ChipReversalSignalQualityReviewResult(
        output_dir=resolved_output,
        rank_path=rank_path,
        feature_bins_path=feature_bins_path,
        worst_trades_path=worst_trades_path,
        summary_path=summary_path,
        report_path=report_path,
        rank=rank,
        feature_bins=feature_bins,
        worst_trades=worst_trades,
        summary=summary,
    )
