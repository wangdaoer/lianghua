"""Risk-on satellite failure diagnostics for core-satellite portfolios."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EQUITY_PATH = Path("outputs/portfolios/chip_true_sat30_20260618/portfolio_equity.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/core_true_position_satellite_failure_latest")


@dataclass(frozen=True)
class PortfolioSatelliteFailureReviewResult:
    output_dir: Path
    monthly_path: Path
    clusters_path: Path
    reasons_path: Path
    summary_path: Path
    report_path: Path
    monthly: pd.DataFrame
    clusters: pd.DataFrame
    reasons: pd.DataFrame
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


def _clean_equity(equity: pd.DataFrame, regime: str) -> pd.DataFrame:
    required = {
        "date",
        "core_return",
        "satellite_return",
        "satellite_weight",
        "effective_regime",
        "satellite_effective_reason",
    }
    missing = required - set(equity.columns)
    if missing:
        raise ValueError(f"equity missing columns: {sorted(missing)}")
    data = equity[list(required)].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in ("core_return", "satellite_return", "satellite_weight"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["effective_regime"] = data["effective_regime"].fillna("unknown").astype(str)
    data["satellite_effective_reason"] = data["satellite_effective_reason"].fillna("unknown").astype(str)
    data = data.dropna(subset=["date", "core_return", "satellite_return", "satellite_weight"])
    data = data.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    data = data[data["effective_regime"] == regime].reset_index(drop=True)
    if data.empty:
        raise ValueError(f"equity contains no valid {regime} rows.")
    data["satellite_contribution"] = data["satellite_weight"] * (data["satellite_return"] - data["core_return"])
    data["month"] = data["date"].dt.strftime("%Y-%m")
    data["year"] = data["date"].dt.year.astype(int)
    data["negative_contribution"] = data["satellite_contribution"] < 0.0
    return data


def _compound_return(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    return float((1.0 + clean).prod() - 1.0)


def _monthly_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for month, group in data.groupby("month", sort=True):
        rows.append(
            {
                "month": str(month),
                "start_date": pd.Timestamp(group["date"].iloc[0]),
                "end_date": pd.Timestamp(group["date"].iloc[-1]),
                "trading_days": int(len(group)),
                "satellite_contribution": float(group["satellite_contribution"].sum()),
                "core_return": _compound_return(group["core_return"]),
                "satellite_return": _compound_return(group["satellite_return"]),
                "average_satellite_weight": float(group["satellite_weight"].mean()),
                "negative_day_ratio": float(group["negative_contribution"].mean()),
                "worst_day": pd.Timestamp(group.loc[group["satellite_contribution"].idxmin(), "date"]),
                "worst_day_contribution": float(group["satellite_contribution"].min()),
            }
        )
    return pd.DataFrame(rows)


def _failure_clusters(data: pd.DataFrame) -> pd.DataFrame:
    negative = data[data["negative_contribution"]].copy()
    if negative.empty:
        return pd.DataFrame(
            columns=[
                "start_date",
                "end_date",
                "trading_days",
                "satellite_contribution",
                "average_satellite_weight",
                "worst_day",
                "worst_day_contribution",
                "primary_reason",
            ]
        )
    negative["_break"] = negative.index.to_series().diff().fillna(1).ne(1)
    negative["_cluster_id"] = negative["_break"].cumsum()
    rows: list[dict[str, Any]] = []
    for _, group in negative.groupby("_cluster_id", sort=True):
        reason_counts = group["satellite_effective_reason"].value_counts()
        rows.append(
            {
                "start_date": pd.Timestamp(group["date"].iloc[0]),
                "end_date": pd.Timestamp(group["date"].iloc[-1]),
                "trading_days": int(len(group)),
                "satellite_contribution": float(group["satellite_contribution"].sum()),
                "average_satellite_weight": float(group["satellite_weight"].mean()),
                "worst_day": pd.Timestamp(group.loc[group["satellite_contribution"].idxmin(), "date"]),
                "worst_day_contribution": float(group["satellite_contribution"].min()),
                "primary_reason": str(reason_counts.index[0]) if not reason_counts.empty else "unknown",
            }
        )
    clusters = pd.DataFrame(rows)
    return clusters.sort_values(["satellite_contribution", "trading_days"], ascending=[True, False]).reset_index(drop=True)


def _reason_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for reason, group in data.groupby("satellite_effective_reason", sort=True):
        rows.append(
            {
                "satellite_effective_reason": str(reason),
                "trading_days": int(len(group)),
                "satellite_contribution": float(group["satellite_contribution"].sum()),
                "average_satellite_weight": float(group["satellite_weight"].mean()),
                "negative_day_ratio": float(group["negative_contribution"].mean()),
                "worst_day": pd.Timestamp(group.loc[group["satellite_contribution"].idxmin(), "date"]),
                "worst_day_contribution": float(group["satellite_contribution"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values("satellite_contribution", ascending=True).reset_index(drop=True)


def _build_summary(
    data: pd.DataFrame,
    monthly: pd.DataFrame,
    clusters: pd.DataFrame,
    reasons: pd.DataFrame,
    regime: str,
) -> dict[str, Any]:
    worst_month = None
    worst_month_contribution = None
    if not monthly.empty:
        row = monthly.sort_values("satellite_contribution", ascending=True).iloc[0]
        worst_month = str(row["month"])
        worst_month_contribution = float(row["satellite_contribution"])
    worst_cluster = None
    worst_cluster_contribution = None
    if not clusters.empty:
        row = clusters.iloc[0]
        worst_cluster = f"{pd.Timestamp(row['start_date']).date()}~{pd.Timestamp(row['end_date']).date()}"
        worst_cluster_contribution = float(row["satellite_contribution"])
    worst_reason = None
    worst_reason_contribution = None
    if not reasons.empty:
        row = reasons.iloc[0]
        worst_reason = str(row["satellite_effective_reason"])
        worst_reason_contribution = float(row["satellite_contribution"])
    return {
        "status": "ok",
        "research_only": True,
        "broker_action": "none",
        "regime": regime,
        "start_date": pd.Timestamp(data["date"].iloc[0]),
        "end_date": pd.Timestamp(data["date"].iloc[-1]),
        "risk_on_row_count": int(len(data)),
        "total_satellite_contribution": float(data["satellite_contribution"].sum()),
        "negative_day_ratio": float(data["negative_contribution"].mean()),
        "average_satellite_weight": float(data["satellite_weight"].mean()),
        "worst_month": worst_month,
        "worst_month_contribution": worst_month_contribution,
        "worst_cluster": worst_cluster,
        "worst_cluster_contribution": worst_cluster_contribution,
        "worst_reason": worst_reason,
        "worst_reason_contribution": worst_reason_contribution,
    }


def build_portfolio_satellite_failure_review(
    equity: pd.DataFrame,
    *,
    regime: str = "risk_on",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = _clean_equity(equity, regime)
    monthly = _monthly_summary(data)
    clusters = _failure_clusters(data)
    reasons = _reason_summary(data)
    summary = _build_summary(data, monthly, clusters, reasons, regime)
    return monthly.reset_index(drop=True), clusters.reset_index(drop=True), reasons.reset_index(drop=True), summary


_MONTHLY_CN_COLUMNS = {
    "month": "月份",
    "start_date": "开始日期",
    "end_date": "结束日期",
    "trading_days": "交易日数",
    "satellite_contribution": "卫星相对核心贡献",
    "core_return": "核心收益率",
    "satellite_return": "卫星收益率",
    "average_satellite_weight": "平均卫星仓位",
    "negative_day_ratio": "负贡献天数占比",
    "worst_day": "最差日期",
    "worst_day_contribution": "最差日贡献",
}

_CLUSTER_CN_COLUMNS = {
    "start_date": "开始日期",
    "end_date": "结束日期",
    "trading_days": "交易日数",
    "satellite_contribution": "卫星相对核心贡献",
    "average_satellite_weight": "平均卫星仓位",
    "worst_day": "最差日期",
    "worst_day_contribution": "最差日贡献",
    "primary_reason": "主要原因",
}

_REASON_CN_COLUMNS = {
    "satellite_effective_reason": "卫星风控原因",
    "trading_days": "交易日数",
    "satellite_contribution": "卫星相对核心贡献",
    "average_satellite_weight": "平均卫星仓位",
    "negative_day_ratio": "负贡献天数占比",
    "worst_day": "最差日期",
    "worst_day_contribution": "最差日贡献",
}


def _to_chinese_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    output = frame.copy()
    for column in ("start_date", "end_date", "worst_day"):
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
    monthly_cn: pd.DataFrame,
    clusters_cn: pd.DataFrame,
    reasons_cn: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    equity_path: Path,
) -> Path:
    report_path = output_dir / "portfolio_satellite_failure_report.html"
    monthly_table = monthly_cn.sort_values("卫星相对核心贡献", ascending=True).head(20).to_html(
        index=False, border=0, classes="data-table"
    )
    clusters_table = clusters_cn.head(20).to_html(index=False, border=0, classes="data-table")
    reasons_table = reasons_cn.head(20).to_html(index=False, border=0, classes="data-table")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{name} 卫星失败诊断</title>
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
  <h1>{name} 卫星失败诊断</h1>
  <div class="meta">
    <div><span class="badge">研究模式</span> broker_action=none，不开券商动作，不自动下单。</div>
    <div>输入曲线：{equity_path}</div>
    <div>诊断状态：{summary.get("regime")}</div>
    <div>样本区间：{_json_ready(summary.get("start_date"))} 至 {_json_ready(summary.get("end_date"))}</div>
    <div>卫星总贡献：{_fmt_pct(summary.get("total_satellite_contribution"))}</div>
    <div>最差月份：{summary.get("worst_month")}，贡献：{_fmt_pct(summary.get("worst_month_contribution"))}</div>
    <div>最差原因：{summary.get("worst_reason")}，贡献：{_fmt_pct(summary.get("worst_reason_contribution"))}</div>
  </div>
  <h2>最差月份</h2>
  {monthly_table}
  <h2>连续负贡献段</h2>
  {clusters_table}
  <h2>按卫星原因汇总</h2>
  {reasons_table}
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return report_path


def run_portfolio_satellite_failure_review(
    *,
    equity_path: str | Path = DEFAULT_EQUITY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    name: str = "portfolio",
    regime: str = "risk_on",
) -> PortfolioSatelliteFailureReviewResult:
    resolved_equity_path = Path(equity_path)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    equity = pd.read_csv(resolved_equity_path)
    monthly, clusters, reasons, summary = build_portfolio_satellite_failure_review(equity, regime=regime)
    summary = {**summary, "name": name, "equity_path": str(resolved_equity_path)}

    monthly_cn = _to_chinese_columns(monthly, _MONTHLY_CN_COLUMNS)
    clusters_cn = _to_chinese_columns(clusters, _CLUSTER_CN_COLUMNS)
    reasons_cn = _to_chinese_columns(reasons, _REASON_CN_COLUMNS)
    monthly_path = resolved_output_dir / "portfolio_satellite_failure_monthly_cn.csv"
    clusters_path = resolved_output_dir / "portfolio_satellite_failure_clusters_cn.csv"
    reasons_path = resolved_output_dir / "portfolio_satellite_failure_reasons_cn.csv"
    summary_path = resolved_output_dir / "portfolio_satellite_failure_summary.json"
    monthly_cn.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    clusters_cn.to_csv(clusters_path, index=False, encoding="utf-8-sig")
    reasons_cn.to_csv(reasons_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(
        name=name,
        monthly_cn=monthly_cn,
        clusters_cn=clusters_cn,
        reasons_cn=reasons_cn,
        summary=summary,
        output_dir=resolved_output_dir,
        equity_path=resolved_equity_path,
    )
    return PortfolioSatelliteFailureReviewResult(
        output_dir=resolved_output_dir,
        monthly_path=monthly_path,
        clusters_path=clusters_path,
        reasons_path=reasons_path,
        summary_path=summary_path,
        report_path=report_path,
        monthly=monthly,
        clusters=clusters,
        reasons=reasons,
        summary=summary,
    )
