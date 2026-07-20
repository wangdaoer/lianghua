"""Diagnostics for satellite_healthy underperformance days."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EQUITY_PATH = Path("outputs/portfolios/chip_true_sat30_guarded_rel20_20260618/portfolio_equity.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/core_true_position_satellite_healthy_underperformance_latest")


@dataclass(frozen=True)
class PortfolioHealthyUnderperformanceReviewResult:
    output_dir: Path
    daily_path: Path
    patterns_path: Path
    months_path: Path
    summary_path: Path
    report_path: Path
    daily: pd.DataFrame
    patterns: pd.DataFrame
    months: pd.DataFrame
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


def _clean_equity(equity: pd.DataFrame, reason: str) -> pd.DataFrame:
    required = {"date", "core_return", "satellite_return", "satellite_weight", "satellite_effective_reason"}
    missing = required - set(equity.columns)
    if missing:
        raise ValueError(f"equity missing columns: {sorted(missing)}")
    optional = [
        column
        for column in ("satellite_momentum", "satellite_relative_momentum", "satellite_drawdown")
        if column in equity.columns
    ]
    data = equity[list(required) + optional].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in required - {"date", "satellite_effective_reason"}:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in optional:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["satellite_effective_reason"] = data["satellite_effective_reason"].fillna("unknown").astype(str)
    data = data.dropna(subset=["date", "core_return", "satellite_return", "satellite_weight"])
    data = data.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    data = data[data["satellite_effective_reason"] == reason].reset_index(drop=True)
    if data.empty:
        raise ValueError(f"equity contains no rows for reason: {reason}")
    data["satellite_relative_contribution"] = data["satellite_weight"] * (data["satellite_return"] - data["core_return"])
    data["negative_contribution"] = data["satellite_relative_contribution"] < 0.0
    data["return_gap"] = data["core_return"] - data["satellite_return"]
    data["month"] = data["date"].dt.strftime("%Y-%m")
    if "satellite_relative_momentum" not in data.columns:
        data["satellite_relative_momentum"] = np.nan
    if "satellite_momentum" not in data.columns:
        data["satellite_momentum"] = np.nan
    if "satellite_drawdown" not in data.columns:
        data["satellite_drawdown"] = np.nan
    data["underperformance_pattern"] = data.apply(_classify_pattern, axis=1)
    data["relative_momentum_state"] = data["satellite_relative_momentum"].apply(_relative_momentum_state)
    return data


def _classify_pattern(row: pd.Series) -> str:
    contribution = float(row["satellite_relative_contribution"])
    if contribution >= 0.0:
        return "positive_or_neutral"
    core_return = float(row["core_return"])
    satellite_return = float(row["satellite_return"])
    if core_return > 0.0 and satellite_return < 0.0:
        return "core_up_satellite_down"
    if core_return > 0.0 and satellite_return >= 0.0:
        return "core_up_satellite_lag"
    if core_return <= 0.0 and satellite_return < core_return:
        return "core_down_satellite_worse"
    return "other_negative"


def _relative_momentum_state(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    return "relative_weak_after_close" if float(value) < 0.0 else "relative_not_weak_after_close"


def _summary_rows(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    negative = data[data["negative_contribution"]].copy()
    if negative.empty:
        return pd.DataFrame(
            columns=[
                group_column,
                "trading_days",
                "satellite_relative_contribution",
                "average_core_return",
                "average_satellite_return",
                "average_return_gap",
                "average_satellite_weight",
                "average_relative_momentum",
                "negative_day_ratio",
                "worst_day",
                "worst_day_contribution",
            ]
        )
    total_days = data.groupby(group_column).size()
    rows: list[dict[str, Any]] = []
    for key, group in negative.groupby(group_column, sort=True):
        rows.append(
            {
                group_column: str(key),
                "trading_days": int(len(group)),
                "satellite_relative_contribution": float(group["satellite_relative_contribution"].sum()),
                "average_core_return": float(group["core_return"].mean()),
                "average_satellite_return": float(group["satellite_return"].mean()),
                "average_return_gap": float(group["return_gap"].mean()),
                "average_satellite_weight": float(group["satellite_weight"].mean()),
                "average_relative_momentum": float(group["satellite_relative_momentum"].mean(skipna=True)),
                "negative_day_ratio": float(len(group) / int(total_days.loc[key])),
                "worst_day": pd.Timestamp(group.loc[group["satellite_relative_contribution"].idxmin(), "date"]),
                "worst_day_contribution": float(group["satellite_relative_contribution"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values("satellite_relative_contribution", ascending=True).reset_index(drop=True)


def _month_summary(data: pd.DataFrame) -> pd.DataFrame:
    return _summary_rows(data, "month")


def _pattern_summary(data: pd.DataFrame) -> pd.DataFrame:
    return _summary_rows(data, "underperformance_pattern")


def _build_summary(data: pd.DataFrame, patterns: pd.DataFrame, months: pd.DataFrame, reason: str) -> dict[str, Any]:
    negative = data[data["negative_contribution"]]
    worst_day = None
    worst_day_contribution = None
    if not negative.empty:
        row = negative.loc[negative["satellite_relative_contribution"].idxmin()]
        worst_day = pd.Timestamp(row["date"])
        worst_day_contribution = float(row["satellite_relative_contribution"])
    worst_pattern = str(patterns.iloc[0]["underperformance_pattern"]) if not patterns.empty else None
    worst_month = str(months.iloc[0]["month"]) if not months.empty else None
    core_up = negative[negative["core_return"] > 0.0]
    satellite_down = negative[negative["satellite_return"] < 0.0]
    return {
        "status": "ok",
        "research_only": True,
        "broker_action": "none",
        "reason": reason,
        "start_date": pd.Timestamp(data["date"].iloc[0]),
        "end_date": pd.Timestamp(data["date"].iloc[-1]),
        "healthy_row_count": int(len(data)),
        "healthy_negative_day_count": int(len(negative)),
        "healthy_negative_day_ratio": float(len(negative) / len(data)) if len(data) else 0.0,
        "healthy_total_contribution": float(data["satellite_relative_contribution"].sum()),
        "healthy_negative_contribution": float(negative["satellite_relative_contribution"].sum()),
        "core_up_negative_day_ratio": float(len(core_up) / len(negative)) if len(negative) else 0.0,
        "satellite_down_negative_day_ratio": float(len(satellite_down) / len(negative)) if len(negative) else 0.0,
        "worst_day": worst_day,
        "worst_day_contribution": worst_day_contribution,
        "worst_pattern": worst_pattern,
        "worst_month": worst_month,
    }


def build_portfolio_healthy_underperformance_review(
    equity: pd.DataFrame,
    *,
    reason: str = "satellite_healthy",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = _clean_equity(equity, reason)
    patterns = _pattern_summary(data)
    months = _month_summary(data)
    summary = _build_summary(data, patterns, months, reason)
    return data.reset_index(drop=True), patterns.reset_index(drop=True), months.reset_index(drop=True), summary


_DAILY_CN_COLUMNS = {
    "date": "日期",
    "month": "月份",
    "core_return": "核心日收益",
    "satellite_return": "卫星日收益",
    "satellite_weight": "卫星仓位",
    "satellite_relative_contribution": "卫星相对核心贡献",
    "return_gap": "核心领先幅度",
    "underperformance_pattern": "负贡献模式",
    "relative_momentum_state": "相对动量状态",
    "satellite_momentum": "卫星动量",
    "satellite_relative_momentum": "相对核心动量",
    "satellite_drawdown": "卫星回撤",
}

_PATTERN_CN_COLUMNS = {
    "underperformance_pattern": "负贡献模式",
    "trading_days": "交易日数",
    "satellite_relative_contribution": "卫星相对核心贡献",
    "average_core_return": "平均核心日收益",
    "average_satellite_return": "平均卫星日收益",
    "average_return_gap": "平均核心领先幅度",
    "average_satellite_weight": "平均卫星仓位",
    "average_relative_momentum": "平均相对核心动量",
    "negative_day_ratio": "负贡献天数占比",
    "worst_day": "最差日期",
    "worst_day_contribution": "最差日贡献",
}

_MONTH_CN_COLUMNS = {
    "month": "月份",
    "trading_days": "交易日数",
    "satellite_relative_contribution": "卫星相对核心贡献",
    "average_core_return": "平均核心日收益",
    "average_satellite_return": "平均卫星日收益",
    "average_return_gap": "平均核心领先幅度",
    "average_satellite_weight": "平均卫星仓位",
    "average_relative_momentum": "平均相对核心动量",
    "negative_day_ratio": "负贡献天数占比",
    "worst_day": "最差日期",
    "worst_day_contribution": "最差日贡献",
}


def _to_chinese_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    output = frame.copy()
    for column in ("date", "worst_day"):
        if column in output.columns:
            output[column] = pd.to_datetime(output[column]).dt.strftime("%Y-%m-%d")
    ordered = [column for column in mapping if column in output.columns]
    rest = [column for column in output.columns if column not in ordered]
    output = output[ordered + rest]
    return output.rename(columns=mapping)


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _write_report(
    *,
    name: str,
    daily_cn: pd.DataFrame,
    patterns_cn: pd.DataFrame,
    months_cn: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    equity_path: Path,
) -> Path:
    report_path = output_dir / "portfolio_healthy_underperformance_report.html"
    negative_daily = daily_cn[daily_cn["卫星相对核心贡献"] < 0].sort_values("卫星相对核心贡献").head(30)
    daily_table = negative_daily.to_html(index=False, border=0, classes="data-table")
    pattern_table = patterns_cn.to_html(index=False, border=0, classes="data-table")
    month_table = months_cn.head(20).to_html(index=False, border=0, classes="data-table")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{name} 卫星健康负贡献诊断</title>
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
  <h1>{name} 卫星健康负贡献诊断</h1>
  <div class="meta">
    <div><span class="badge">研究模式</span> broker_action=none，不开券商动作，不自动下单。</div>
    <div>输入曲线：{equity_path}</div>
    <div>诊断原因：{summary.get("reason")}</div>
    <div>健康日：{summary.get("healthy_row_count")}；负贡献日：{summary.get("healthy_negative_day_count")}；占比：{_fmt_pct(summary.get("healthy_negative_day_ratio"))}</div>
    <div>健康日总贡献：{_fmt_pct(summary.get("healthy_total_contribution"))}；负贡献合计：{_fmt_pct(summary.get("healthy_negative_contribution"))}</div>
    <div>核心上涨负贡献日占比：{_fmt_pct(summary.get("core_up_negative_day_ratio"))}；卫星下跌负贡献日占比：{_fmt_pct(summary.get("satellite_down_negative_day_ratio"))}</div>
    <div>最差模式：{summary.get("worst_pattern")}；最差月份：{summary.get("worst_month")}；最差日：{_json_ready(summary.get("worst_day"))}</div>
  </div>
  <h2>负贡献模式</h2>
  {pattern_table}
  <h2>月份压力</h2>
  {month_table}
  <h2>最差健康日</h2>
  {daily_table}
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return report_path


def run_portfolio_healthy_underperformance_review(
    *,
    equity_path: str | Path = DEFAULT_EQUITY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    name: str = "portfolio",
    reason: str = "satellite_healthy",
) -> PortfolioHealthyUnderperformanceReviewResult:
    resolved_equity_path = Path(equity_path)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    equity = pd.read_csv(resolved_equity_path)
    daily, patterns, months, summary = build_portfolio_healthy_underperformance_review(equity, reason=reason)
    summary = {**summary, "name": name, "equity_path": str(resolved_equity_path)}

    daily_cn = _to_chinese_columns(daily, _DAILY_CN_COLUMNS)
    patterns_cn = _to_chinese_columns(patterns, _PATTERN_CN_COLUMNS)
    months_cn = _to_chinese_columns(months, _MONTH_CN_COLUMNS)
    daily_path = resolved_output_dir / "portfolio_healthy_underperformance_daily_cn.csv"
    patterns_path = resolved_output_dir / "portfolio_healthy_underperformance_patterns_cn.csv"
    months_path = resolved_output_dir / "portfolio_healthy_underperformance_months_cn.csv"
    summary_path = resolved_output_dir / "portfolio_healthy_underperformance_summary.json"
    daily_cn.to_csv(daily_path, index=False, encoding="utf-8-sig")
    patterns_cn.to_csv(patterns_path, index=False, encoding="utf-8-sig")
    months_cn.to_csv(months_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(
        name=name,
        daily_cn=daily_cn,
        patterns_cn=patterns_cn,
        months_cn=months_cn,
        summary=summary,
        output_dir=resolved_output_dir,
        equity_path=resolved_equity_path,
    )
    return PortfolioHealthyUnderperformanceReviewResult(
        output_dir=resolved_output_dir,
        daily_path=daily_path,
        patterns_path=patterns_path,
        months_path=months_path,
        summary_path=summary_path,
        report_path=report_path,
        daily=daily,
        patterns=patterns,
        months=months,
        summary=summary,
    )
