"""Rolling-window weakness diagnostics for core-satellite portfolios."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EQUITY_PATH = Path("outputs/portfolios/chip_true_sat30_guarded_rel20_20260618/portfolio_equity.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/core_true_position_satellite_rolling_weakness_latest")


@dataclass(frozen=True)
class PortfolioRollingWeaknessReviewResult:
    output_dir: Path
    windows_path: Path
    weak_windows_path: Path
    reasons_path: Path
    months_path: Path
    summary_path: Path
    report_path: Path
    windows: pd.DataFrame
    weak_windows: pd.DataFrame
    reasons: pd.DataFrame
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


def _clean_equity(equity: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "portfolio_return",
        "core_return",
        "satellite_return",
        "satellite_weight",
        "cash_weight",
        "satellite_effective_reason",
    }
    missing = required - set(equity.columns)
    if missing:
        raise ValueError(f"equity missing columns: {sorted(missing)}")
    optional = [column for column in ("portfolio_equity", "core_equity") if column in equity.columns]
    data = equity[list(required) + optional].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    for column in required - {"date", "satellite_effective_reason"}:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in optional:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["satellite_effective_reason"] = data["satellite_effective_reason"].fillna("unknown").astype(str)
    data = data.dropna(subset=["date", "portfolio_return", "core_return", "satellite_return", "satellite_weight", "cash_weight"])
    if {"portfolio_equity", "core_equity"} <= set(data.columns):
        data = data.dropna(subset=["portfolio_equity", "core_equity"])
        data = data[(data["portfolio_equity"] > 0) & (data["core_equity"] > 0)]
    data = data.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    if data.empty:
        raise ValueError("equity contains no valid rows.")
    data["satellite_relative_contribution"] = data["satellite_weight"] * (data["satellite_return"] - data["core_return"])
    data["cash_guard_contribution"] = -data["cash_weight"] * data["core_return"]
    data["daily_excess_return"] = data["portfolio_return"] - data["core_return"]
    data["month"] = data["date"].dt.strftime("%Y-%m")
    data["_month_period"] = data["date"].dt.to_period("M")
    data["negative_satellite_day"] = data["satellite_relative_contribution"] < 0.0
    return data


def _compound_return(returns: pd.Series) -> float:
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return 0.0
    return float((1.0 + clean).prod() - 1.0)


def _window_row(group: pd.DataFrame, window_months: int) -> dict[str, Any]:
    if {"portfolio_equity", "core_equity"} <= set(group.columns):
        portfolio_return = float(group["portfolio_equity"].iloc[-1] / group["portfolio_equity"].iloc[0] - 1.0)
        core_return = float(group["core_equity"].iloc[-1] / group["core_equity"].iloc[0] - 1.0)
    else:
        portfolio_return = _compound_return(group["portfolio_return"])
        core_return = _compound_return(group["core_return"])
    return {
        "window_months": int(window_months),
        "start_date": pd.Timestamp(group["date"].iloc[0]),
        "end_date": pd.Timestamp(group["date"].iloc[-1]),
        "start_month": str(group["month"].iloc[0]),
        "end_month": str(group["month"].iloc[-1]),
        "trading_days": int(len(group)),
        "portfolio_return": portfolio_return,
        "core_return": core_return,
        "excess_return": float(portfolio_return - core_return),
        "daily_excess_sum": float(group["daily_excess_return"].sum()),
        "satellite_relative_contribution": float(group["satellite_relative_contribution"].sum()),
        "cash_guard_contribution": float(group["cash_guard_contribution"].sum()),
        "average_satellite_weight": float(group["satellite_weight"].mean()),
        "negative_satellite_day_ratio": float(group["negative_satellite_day"].mean()),
    }


def _rolling_windows(data: pd.DataFrame, window_months: int, min_window_rows: int) -> pd.DataFrame:
    months = sorted(data["_month_period"].drop_duplicates().tolist())
    rows: list[dict[str, Any]] = []
    if len(months) < window_months:
        return pd.DataFrame()
    for end_index in range(window_months - 1, len(months)):
        window_keys = months[end_index - window_months + 1 : end_index + 1]
        group = data[data["_month_period"].isin(window_keys)].reset_index(drop=True)
        if len(group) < min_window_rows:
            continue
        rows.append(_window_row(group, window_months))
    return pd.DataFrame(rows)


def _weak_days(data: pd.DataFrame, weak_windows: pd.DataFrame) -> pd.DataFrame:
    if weak_windows.empty:
        return data.iloc[0:0].copy()
    mask = pd.Series(False, index=data.index)
    for row in weak_windows.itertuples(index=False):
        start = pd.Timestamp(row.start_date)
        end = pd.Timestamp(row.end_date)
        mask = mask | ((data["date"] >= start) & (data["date"] <= end))
    return data[mask].copy()


def _reason_summary(days: pd.DataFrame) -> pd.DataFrame:
    if days.empty:
        return pd.DataFrame(
            columns=[
                "satellite_effective_reason",
                "trading_days",
                "satellite_relative_contribution",
                "cash_guard_contribution",
                "average_satellite_weight",
                "negative_day_ratio",
                "worst_day",
                "worst_day_contribution",
            ]
        )
    rows: list[dict[str, Any]] = []
    for reason, group in days.groupby("satellite_effective_reason", sort=True):
        rows.append(
            {
                "satellite_effective_reason": str(reason),
                "trading_days": int(len(group)),
                "satellite_relative_contribution": float(group["satellite_relative_contribution"].sum()),
                "cash_guard_contribution": float(group["cash_guard_contribution"].sum()),
                "average_satellite_weight": float(group["satellite_weight"].mean()),
                "negative_day_ratio": float(group["negative_satellite_day"].mean()),
                "worst_day": pd.Timestamp(group.loc[group["satellite_relative_contribution"].idxmin(), "date"]),
                "worst_day_contribution": float(group["satellite_relative_contribution"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values("satellite_relative_contribution", ascending=True).reset_index(drop=True)


def _month_pressure(days: pd.DataFrame) -> pd.DataFrame:
    if days.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "trading_days",
                "satellite_relative_contribution",
                "cash_guard_contribution",
                "average_satellite_weight",
                "negative_day_ratio",
                "worst_day",
                "worst_day_contribution",
            ]
        )
    rows: list[dict[str, Any]] = []
    for month, group in days.groupby("month", sort=True):
        rows.append(
            {
                "month": str(month),
                "trading_days": int(len(group)),
                "satellite_relative_contribution": float(group["satellite_relative_contribution"].sum()),
                "cash_guard_contribution": float(group["cash_guard_contribution"].sum()),
                "average_satellite_weight": float(group["satellite_weight"].mean()),
                "negative_day_ratio": float(group["negative_satellite_day"].mean()),
                "worst_day": pd.Timestamp(group.loc[group["satellite_relative_contribution"].idxmin(), "date"]),
                "worst_day_contribution": float(group["satellite_relative_contribution"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values("satellite_relative_contribution", ascending=True).reset_index(drop=True)


def _summary(
    data: pd.DataFrame,
    windows: pd.DataFrame,
    weak_windows: pd.DataFrame,
    reasons: pd.DataFrame,
    months: pd.DataFrame,
    window_months: int,
) -> dict[str, Any]:
    worst_window = None
    worst_window_excess = None
    if not weak_windows.empty:
        row = weak_windows.sort_values("excess_return", ascending=True).iloc[0]
        worst_window = f"{pd.Timestamp(row['start_date']).date()}~{pd.Timestamp(row['end_date']).date()}"
        worst_window_excess = float(row["excess_return"])
    worst_reason = None
    if not reasons.empty:
        worst_reason = str(reasons.iloc[0]["satellite_effective_reason"])
    worst_month = None
    if not months.empty:
        worst_month = str(months.iloc[0]["month"])
    excess = pd.to_numeric(windows["excess_return"], errors="coerce") if "excess_return" in windows else pd.Series(dtype=float)
    positive_count = int((excess > 0.0).sum())
    negative_count = int((excess < 0.0).sum())
    flat_count = int((excess == 0.0).sum())
    window_count = int(len(windows))
    return {
        "status": "ok",
        "research_only": True,
        "broker_action": "none",
        "window_months": int(window_months),
        "start_date": pd.Timestamp(data["date"].iloc[0]),
        "end_date": pd.Timestamp(data["date"].iloc[-1]),
        "window_count": window_count,
        "positive_window_count": positive_count,
        "negative_window_count": negative_count,
        "flat_window_count": flat_count,
        "positive_window_ratio": float(positive_count / window_count) if window_count else 0.0,
        "negative_window_ratio": float(negative_count / window_count) if window_count else 0.0,
        "flat_window_ratio": float(flat_count / window_count) if window_count else 0.0,
        "underperform_window_count": int(len(weak_windows)),
        "underperform_window_ratio": float(len(weak_windows) / len(windows)) if len(windows) else 0.0,
        "worst_window": worst_window,
        "worst_window_excess_return": worst_window_excess,
        "worst_reason": worst_reason,
        "worst_month": worst_month,
    }


def build_portfolio_rolling_weakness_review(
    equity: pd.DataFrame,
    *,
    window_months: int = 3,
    min_window_rows: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if window_months < 1:
        raise ValueError("window_months must be at least 1.")
    if min_window_rows < 1:
        raise ValueError("min_window_rows must be at least 1.")
    data = _clean_equity(equity)
    windows = _rolling_windows(data, int(window_months), int(min_window_rows))
    weak_windows = windows[windows["excess_return"] <= 0.0].copy() if not windows.empty else windows.copy()
    if not weak_windows.empty:
        weak_windows = weak_windows.sort_values("excess_return", ascending=True).reset_index(drop=True)
    weak_days = _weak_days(data, weak_windows)
    reasons = _reason_summary(weak_days)
    months = _month_pressure(weak_days)
    summary = _summary(data, windows, weak_windows, reasons, months, int(window_months))
    return (
        windows.reset_index(drop=True),
        weak_windows.reset_index(drop=True),
        reasons.reset_index(drop=True),
        months.reset_index(drop=True),
        summary,
    )


_WINDOW_CN_COLUMNS = {
    "window_months": "窗口月数",
    "start_date": "开始日期",
    "end_date": "结束日期",
    "start_month": "开始月份",
    "end_month": "结束月份",
    "trading_days": "交易日数",
    "portfolio_return": "组合收益率",
    "core_return": "核心收益率",
    "excess_return": "超额收益率",
    "daily_excess_sum": "日度超额合计",
    "satellite_relative_contribution": "卫星相对核心贡献",
    "cash_guard_contribution": "现金/风控贡献",
    "average_satellite_weight": "平均卫星仓位",
    "negative_satellite_day_ratio": "卫星负贡献天数占比",
}

_REASON_CN_COLUMNS = {
    "satellite_effective_reason": "卫星风控原因",
    "trading_days": "交易日数",
    "satellite_relative_contribution": "卫星相对核心贡献",
    "cash_guard_contribution": "现金/风控贡献",
    "average_satellite_weight": "平均卫星仓位",
    "negative_day_ratio": "负贡献天数占比",
    "worst_day": "最差日期",
    "worst_day_contribution": "最差日贡献",
}

_MONTH_CN_COLUMNS = {
    "month": "月份",
    "trading_days": "交易日数",
    "satellite_relative_contribution": "卫星相对核心贡献",
    "cash_guard_contribution": "现金/风控贡献",
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
    windows_cn: pd.DataFrame,
    weak_windows_cn: pd.DataFrame,
    reasons_cn: pd.DataFrame,
    months_cn: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    equity_path: Path,
) -> Path:
    report_path = output_dir / "portfolio_rolling_weakness_report.html"
    weak_table = weak_windows_cn.head(20).to_html(index=False, border=0, classes="data-table")
    reason_table = reasons_cn.head(20).to_html(index=False, border=0, classes="data-table")
    month_table = months_cn.head(20).to_html(index=False, border=0, classes="data-table")
    window_table = windows_cn.head(20).to_html(index=False, border=0, classes="data-table")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{name} 滚动弱点诊断</title>
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
  <h1>{name} 滚动弱点诊断</h1>
  <div class="meta">
    <div><span class="badge">研究模式</span> broker_action=none，不开券商动作，不自动下单。</div>
    <div>输入曲线：{equity_path}</div>
    <div>窗口月数：{summary.get("window_months")}</div>
    <div>样本区间：{_json_ready(summary.get("start_date"))} 至 {_json_ready(summary.get("end_date"))}</div>
    <div>跑输窗口占比：{_fmt_pct(summary.get("underperform_window_ratio"))}</div>
    <div>跑赢/真实负超额/持平窗口：{summary.get("positive_window_count")} / {summary.get("negative_window_count")} / {summary.get("flat_window_count")}</div>
    <div>最差窗口：{summary.get("worst_window")}，超额：{_fmt_pct(summary.get("worst_window_excess_return"))}</div>
    <div>最差原因：{summary.get("worst_reason")}；最差月份：{summary.get("worst_month")}</div>
  </div>
  <h2>跑输窗口</h2>
  {weak_table}
  <h2>跑输窗口内的原因归因</h2>
  {reason_table}
  <h2>跑输窗口覆盖月份压力</h2>
  {month_table}
  <h2>全部滚动窗口</h2>
  {window_table}
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return report_path


def run_portfolio_rolling_weakness_review(
    *,
    equity_path: str | Path = DEFAULT_EQUITY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    name: str = "portfolio",
    window_months: int = 3,
    min_window_rows: int = 40,
) -> PortfolioRollingWeaknessReviewResult:
    resolved_equity_path = Path(equity_path)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    equity = pd.read_csv(resolved_equity_path)
    windows, weak_windows, reasons, months, summary = build_portfolio_rolling_weakness_review(
        equity,
        window_months=window_months,
        min_window_rows=min_window_rows,
    )
    summary = {**summary, "name": name, "equity_path": str(resolved_equity_path)}

    windows_cn = _to_chinese_columns(windows, _WINDOW_CN_COLUMNS)
    weak_windows_cn = _to_chinese_columns(weak_windows, _WINDOW_CN_COLUMNS)
    reasons_cn = _to_chinese_columns(reasons, _REASON_CN_COLUMNS)
    months_cn = _to_chinese_columns(months, _MONTH_CN_COLUMNS)
    windows_path = resolved_output_dir / "portfolio_rolling_windows_cn.csv"
    weak_windows_path = resolved_output_dir / "portfolio_rolling_weak_windows_cn.csv"
    reasons_path = resolved_output_dir / "portfolio_rolling_weak_reasons_cn.csv"
    months_path = resolved_output_dir / "portfolio_rolling_weak_months_cn.csv"
    summary_path = resolved_output_dir / "portfolio_rolling_weakness_summary.json"
    windows_cn.to_csv(windows_path, index=False, encoding="utf-8-sig")
    weak_windows_cn.to_csv(weak_windows_path, index=False, encoding="utf-8-sig")
    reasons_cn.to_csv(reasons_path, index=False, encoding="utf-8-sig")
    months_cn.to_csv(months_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(
        name=name,
        windows_cn=windows_cn,
        weak_windows_cn=weak_windows_cn,
        reasons_cn=reasons_cn,
        months_cn=months_cn,
        summary=summary,
        output_dir=resolved_output_dir,
        equity_path=resolved_equity_path,
    )
    return PortfolioRollingWeaknessReviewResult(
        output_dir=resolved_output_dir,
        windows_path=windows_path,
        weak_windows_path=weak_windows_path,
        reasons_path=reasons_path,
        months_path=months_path,
        summary_path=summary_path,
        report_path=report_path,
        windows=windows,
        weak_windows=weak_windows,
        reasons=reasons,
        months=months,
        summary=summary,
    )
