"""Excess-return attribution for core-satellite portfolio curves."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EQUITY_PATH = Path("outputs/portfolios/chip_true_sat30_20260618/portfolio_equity.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/core_true_position_satellite_attribution_latest")


@dataclass(frozen=True)
class PortfolioExcessAttributionResult:
    output_dir: Path
    annual_path: Path
    regime_path: Path
    daily_path: Path
    summary_path: Path
    report_path: Path
    annual: pd.DataFrame
    by_regime: pd.DataFrame
    daily: pd.DataFrame
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


def _clean_equity(equity: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "portfolio_return",
        "core_return",
        "satellite_return",
        "core_weight",
        "satellite_weight",
        "cash_weight",
    }
    missing = required - set(equity.columns)
    if missing:
        raise ValueError(f"equity missing columns: {sorted(missing)}")
    optional = [column for column in ("effective_regime", "satellite_effective_reason") if column in equity.columns]
    data = equity[list(required) + optional].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in required - {"date"}:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=list(required))
    data = data.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    if data.empty:
        raise ValueError("equity contains no valid rows.")
    if "effective_regime" not in data.columns:
        data["effective_regime"] = "unknown"
    if "satellite_effective_reason" not in data.columns:
        data["satellite_effective_reason"] = "unknown"
    data["effective_regime"] = data["effective_regime"].fillna("unknown").astype(str)
    data["satellite_effective_reason"] = data["satellite_effective_reason"].fillna("unknown").astype(str)
    return data


def _annotate_daily(data: pd.DataFrame) -> pd.DataFrame:
    daily = data.copy()
    daily["daily_excess_return"] = daily["portfolio_return"] - daily["core_return"]
    daily["satellite_contribution"] = daily["satellite_weight"] * (daily["satellite_return"] - daily["core_return"])
    daily["cash_guard_contribution"] = -daily["cash_weight"] * daily["core_return"]
    daily["residual_contribution"] = (
        daily["daily_excess_return"] - daily["satellite_contribution"] - daily["cash_guard_contribution"]
    )
    daily["year"] = daily["date"].dt.year.astype(int)
    daily["satellite_active"] = daily["satellite_weight"] > 0
    daily["cash_active"] = daily["cash_weight"] > 0
    daily["risk_on"] = daily["effective_regime"].eq("risk_on")
    return daily


def _compound_return(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    return float((1.0 + clean).prod() - 1.0)


def _summarize_group(group: pd.DataFrame, labels: dict[str, Any]) -> dict[str, Any]:
    return {
        **labels,
        "start_date": pd.Timestamp(group["date"].iloc[0]),
        "end_date": pd.Timestamp(group["date"].iloc[-1]),
        "trading_days": int(len(group)),
        "portfolio_return": _compound_return(group["portfolio_return"]),
        "core_return": _compound_return(group["core_return"]),
        "geometric_excess_return": _compound_return(group["portfolio_return"]) - _compound_return(group["core_return"]),
        "daily_excess_sum": float(group["daily_excess_return"].sum()),
        "satellite_relative_contribution": float(group["satellite_contribution"].sum()),
        "cash_guard_contribution": float(group["cash_guard_contribution"].sum()),
        "residual_contribution": float(group["residual_contribution"].sum()),
        "average_core_weight": float(group["core_weight"].mean()),
        "average_satellite_weight": float(group["satellite_weight"].mean()),
        "average_cash_weight": float(group["cash_weight"].mean()),
        "satellite_active_day_ratio": float(group["satellite_active"].mean()),
        "cash_active_day_ratio": float(group["cash_active"].mean()),
        "risk_on_day_ratio": float(group["risk_on"].mean()),
    }


def _summarize_annual(daily: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _summarize_group(group.reset_index(drop=True), {"year": int(year)})
        for year, group in daily.groupby("year", sort=True)
    ]
    return pd.DataFrame(rows)


def _summarize_by_regime(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, regime), group in daily.groupby(["year", "effective_regime"], sort=True):
        rows.append(_summarize_group(group.reset_index(drop=True), {"year": int(year), "effective_regime": str(regime)}))
    return pd.DataFrame(rows)


def _summary_from_annual(annual: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    worst_year: int | None = None
    worst_excess: float | None = None
    if not annual.empty:
        worst = annual.sort_values("daily_excess_sum", ascending=True).iloc[0]
        worst_year = int(worst["year"])
        worst_excess = float(worst["daily_excess_sum"])
    negative_satellite_days = daily[daily["satellite_contribution"] < 0]
    negative_cash_days = daily[daily["cash_guard_contribution"] < 0]
    return {
        "status": "ok",
        "research_only": True,
        "broker_action": "none",
        "start_date": pd.Timestamp(daily["date"].iloc[0]),
        "end_date": pd.Timestamp(daily["date"].iloc[-1]),
        "row_count": int(len(daily)),
        "annual_row_count": int(len(annual)),
        "worst_annual_excess_year": worst_year,
        "worst_annual_daily_excess_sum": worst_excess,
        "total_daily_excess_sum": float(daily["daily_excess_return"].sum()),
        "total_satellite_relative_contribution": float(daily["satellite_contribution"].sum()),
        "total_cash_guard_contribution": float(daily["cash_guard_contribution"].sum()),
        "total_residual_contribution": float(daily["residual_contribution"].sum()),
        "negative_satellite_day_ratio": float(len(negative_satellite_days) / len(daily)),
        "negative_cash_day_ratio": float(len(negative_cash_days) / len(daily)),
        "average_satellite_weight": float(daily["satellite_weight"].mean()),
        "average_cash_weight": float(daily["cash_weight"].mean()),
    }


def build_portfolio_excess_attribution(
    equity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = _clean_equity(equity)
    daily = _annotate_daily(data)
    annual = _summarize_annual(daily)
    by_regime = _summarize_by_regime(daily)
    summary = _summary_from_annual(annual, daily)
    return annual.reset_index(drop=True), by_regime.reset_index(drop=True), daily.reset_index(drop=True), summary


_ANNUAL_CN_COLUMNS = {
    "year": "年份",
    "start_date": "开始日期",
    "end_date": "结束日期",
    "trading_days": "交易日数",
    "portfolio_return": "组合收益率",
    "core_return": "核心收益率",
    "geometric_excess_return": "几何超额收益率",
    "daily_excess_sum": "日度超额合计",
    "satellite_relative_contribution": "卫星相对核心贡献",
    "cash_guard_contribution": "现金/风控贡献",
    "residual_contribution": "残差贡献",
    "average_core_weight": "平均核心仓位",
    "average_satellite_weight": "平均卫星仓位",
    "average_cash_weight": "平均现金仓位",
    "satellite_active_day_ratio": "卫星启用天数占比",
    "cash_active_day_ratio": "现金启用天数占比",
    "risk_on_day_ratio": "风险开启天数占比",
}

_REGIME_CN_COLUMNS = {
    **_ANNUAL_CN_COLUMNS,
    "effective_regime": "市场状态",
}

_DAILY_CN_COLUMNS = {
    "date": "日期",
    "year": "年份",
    "portfolio_return": "组合日收益",
    "core_return": "核心日收益",
    "satellite_return": "卫星日收益",
    "daily_excess_return": "日度超额收益",
    "satellite_contribution": "卫星相对核心贡献",
    "cash_guard_contribution": "现金/风控贡献",
    "residual_contribution": "残差贡献",
    "core_weight": "核心仓位",
    "satellite_weight": "卫星仓位",
    "cash_weight": "现金仓位",
    "effective_regime": "市场状态",
    "satellite_effective_reason": "卫星风控原因",
}


def _to_chinese_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    output = frame.copy()
    for column in ("date", "start_date", "end_date"):
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
    annual_cn: pd.DataFrame,
    regime_cn: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    equity_path: Path,
) -> Path:
    report_path = output_dir / "portfolio_excess_attribution_report.html"
    annual_table = annual_cn.to_html(index=False, border=0, classes="data-table")
    regime_table = regime_cn.head(40).to_html(index=False, border=0, classes="data-table")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{name} 超额收益归因</title>
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
  <h1>{name} 超额收益归因</h1>
  <div class="meta">
    <div><span class="badge">研究模式</span> broker_action=none，不开券商动作，不自动下单。</div>
    <div>输入曲线：{equity_path}</div>
    <div>样本区间：{_json_ready(summary.get("start_date"))} 至 {_json_ready(summary.get("end_date"))}</div>
    <div>最弱年度：{summary.get("worst_annual_excess_year")}，日度超额合计：{_fmt_pct(summary.get("worst_annual_daily_excess_sum"))}</div>
    <div>总卫星相对核心贡献：{_fmt_pct(summary.get("total_satellite_relative_contribution"))}</div>
    <div>总现金/风控贡献：{_fmt_pct(summary.get("total_cash_guard_contribution"))}</div>
  </div>
  <h2>年度归因</h2>
  {annual_table}
  <h2>按市场状态归因</h2>
  {regime_table}
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return report_path


def run_portfolio_excess_attribution_review(
    *,
    equity_path: str | Path = DEFAULT_EQUITY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    name: str = "portfolio",
) -> PortfolioExcessAttributionResult:
    resolved_equity_path = Path(equity_path)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    equity = pd.read_csv(resolved_equity_path)
    annual, by_regime, daily, summary = build_portfolio_excess_attribution(equity)
    summary = {**summary, "name": name, "equity_path": str(resolved_equity_path)}

    annual_cn = _to_chinese_columns(annual, _ANNUAL_CN_COLUMNS)
    regime_cn = _to_chinese_columns(by_regime, _REGIME_CN_COLUMNS)
    daily_cn = _to_chinese_columns(daily, _DAILY_CN_COLUMNS)
    annual_path = resolved_output_dir / "portfolio_excess_attribution_annual_cn.csv"
    regime_path = resolved_output_dir / "portfolio_excess_attribution_regime_cn.csv"
    daily_path = resolved_output_dir / "portfolio_excess_attribution_daily_cn.csv"
    summary_path = resolved_output_dir / "portfolio_excess_attribution_summary.json"
    annual_cn.to_csv(annual_path, index=False, encoding="utf-8-sig")
    regime_cn.to_csv(regime_path, index=False, encoding="utf-8-sig")
    daily_cn.to_csv(daily_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(
        name=name,
        annual_cn=annual_cn,
        regime_cn=regime_cn,
        summary=summary,
        output_dir=resolved_output_dir,
        equity_path=resolved_equity_path,
    )
    return PortfolioExcessAttributionResult(
        output_dir=resolved_output_dir,
        annual_path=annual_path,
        regime_path=regime_path,
        daily_path=daily_path,
        summary_path=summary_path,
        report_path=report_path,
        annual=annual,
        by_regime=by_regime,
        daily=daily,
        summary=summary,
    )
