"""Portfolio stability review against a core baseline curve."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_EQUITY_PATH = Path("outputs/portfolios/chip_true_sat30_20260618/portfolio_equity.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/core_true_position_satellite_stability_latest")


@dataclass(frozen=True)
class PortfolioStabilityResult:
    output_dir: Path
    annual_path: Path
    rolling_path: Path
    summary_path: Path
    report_path: Path
    annual: pd.DataFrame
    rolling: pd.DataFrame
    summary: dict[str, Any]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
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


def _clean_equity(
    equity: pd.DataFrame,
    *,
    portfolio_col: str,
    baseline_col: str,
    date_col: str,
) -> pd.DataFrame:
    required = {date_col, portfolio_col, baseline_col}
    missing = required - set(equity.columns)
    if missing:
        raise ValueError(f"equity missing columns: {sorted(missing)}")
    data = equity[[date_col, portfolio_col, baseline_col]].copy()
    data = data.rename(columns={date_col: "date", portfolio_col: "portfolio_equity", baseline_col: "baseline_equity"})
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["portfolio_equity"] = pd.to_numeric(data["portfolio_equity"], errors="coerce")
    data["baseline_equity"] = pd.to_numeric(data["baseline_equity"], errors="coerce")
    data = data.dropna(subset=["date", "portfolio_equity", "baseline_equity"])
    data = data[(data["portfolio_equity"] > 0) & (data["baseline_equity"] > 0)]
    data = data.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(data) < 2:
        raise ValueError("equity must contain at least two valid rows.")
    return data


def _max_drawdown(equity: pd.Series) -> float:
    curve = pd.to_numeric(equity, errors="coerce").dropna()
    if curve.empty:
        return float("nan")
    drawdown = curve / curve.cummax() - 1.0
    return float(drawdown.min())


def _annualized_sharpe(equity: pd.Series) -> float | None:
    returns = pd.to_numeric(equity, errors="coerce").pct_change().dropna()
    if len(returns) < 2:
        return None
    std = float(returns.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return None
    return float(returns.mean() / std * np.sqrt(252.0))


def _period_row(frame: pd.DataFrame, *, label: dict[str, Any]) -> dict[str, Any]:
    portfolio_return = float(frame["portfolio_equity"].iloc[-1] / frame["portfolio_equity"].iloc[0] - 1.0)
    baseline_return = float(frame["baseline_equity"].iloc[-1] / frame["baseline_equity"].iloc[0] - 1.0)
    portfolio_drawdown = _max_drawdown(frame["portfolio_equity"])
    baseline_drawdown = _max_drawdown(frame["baseline_equity"])
    excess_return = portfolio_return - baseline_return
    row = {
        **label,
        "start_date": pd.Timestamp(frame["date"].iloc[0]),
        "end_date": pd.Timestamp(frame["date"].iloc[-1]),
        "trading_days": int(len(frame)),
        "portfolio_return": portfolio_return,
        "baseline_return": baseline_return,
        "excess_return": excess_return,
        "portfolio_max_drawdown": portfolio_drawdown,
        "baseline_max_drawdown": baseline_drawdown,
        "drawdown_edge": portfolio_drawdown - baseline_drawdown,
        "portfolio_sharpe": _annualized_sharpe(frame["portfolio_equity"]),
        "baseline_sharpe": _annualized_sharpe(frame["baseline_equity"]),
        "positive_excess": bool(excess_return > 0.0),
    }
    portfolio_sharpe = row["portfolio_sharpe"]
    baseline_sharpe = row["baseline_sharpe"]
    row["sharpe_edge"] = (
        float(portfolio_sharpe) - float(baseline_sharpe)
        if portfolio_sharpe is not None and baseline_sharpe is not None
        else None
    )
    return row


def _build_annual(data: pd.DataFrame, *, min_year_trading_days: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, group in data.groupby(data["date"].dt.year):
        if len(group) < min_year_trading_days:
            continue
        rows.append(_period_row(group.reset_index(drop=True), label={"year": int(year)}))
    return pd.DataFrame(rows)


def _build_rolling(
    data: pd.DataFrame,
    *,
    rolling_months: Iterable[int],
    min_window_trading_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = data.copy()
    work["_month"] = work["date"].dt.to_period("M")
    months = sorted(work["_month"].drop_duplicates().tolist())
    for raw_months in rolling_months:
        window_months = max(int(raw_months), 1)
        if len(months) < window_months:
            continue
        for end_index in range(window_months - 1, len(months)):
            window_keys = months[end_index - window_months + 1 : end_index + 1]
            frame = work[work["_month"].isin(window_keys)].drop(columns=["_month"]).reset_index(drop=True)
            if len(frame) < min_window_trading_days:
                continue
            rows.append(_period_row(frame, label={"window_months": window_months}))
    return pd.DataFrame(rows)


def _mean_bool(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.astype(bool).mean())


def _safe_min(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.min())


def _safe_mean(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _classify_evidence(
    annual_positive_ratio: float | None,
    rolling_positive_ratios: dict[str, float | None],
    worst_annual_excess: float | None,
    worst_rolling_excess: dict[str, float | None],
) -> str:
    annual_ratio = annual_positive_ratio if annual_positive_ratio is not None else 0.0
    rolling_values = [value for value in rolling_positive_ratios.values() if value is not None]
    rolling_pass = bool(rolling_values) and all(value >= 0.60 for value in rolling_values)
    annual_loss_ok = worst_annual_excess is None or worst_annual_excess >= -0.03
    rolling_loss_ok = all(value is None or value >= -0.05 for value in worst_rolling_excess.values())
    if annual_ratio >= 0.75 and rolling_pass and annual_loss_ok and rolling_loss_ok:
        return "stable_improvement"
    if annual_ratio >= 0.50 and any(value >= 0.50 for value in rolling_values):
        return "mixed_improvement"
    return "weak"


def build_portfolio_stability(
    equity: pd.DataFrame,
    *,
    portfolio_col: str = "portfolio_equity",
    baseline_col: str = "core_equity",
    date_col: str = "date",
    rolling_months: Iterable[int] = (3, 6, 12),
    min_year_trading_days: int = 2,
    min_window_trading_days: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = _clean_equity(equity, portfolio_col=portfolio_col, baseline_col=baseline_col, date_col=date_col)
    rolling_month_tuple = tuple(max(int(months), 1) for months in rolling_months)
    annual = _build_annual(data, min_year_trading_days=max(int(min_year_trading_days), 2))
    rolling = _build_rolling(
        data,
        rolling_months=rolling_month_tuple,
        min_window_trading_days=max(int(min_window_trading_days), 2),
    )

    annual_positive_ratio = _mean_bool(annual["positive_excess"]) if "positive_excess" in annual else None
    rolling_positive_ratios: dict[str, float | None] = {}
    worst_rolling_excess: dict[str, float | None] = {}
    for months in rolling_month_tuple:
        subset = rolling[rolling["window_months"] == months] if not rolling.empty else pd.DataFrame()
        rolling_positive_ratios[str(months)] = _mean_bool(subset["positive_excess"]) if "positive_excess" in subset else None
        worst_rolling_excess[str(months)] = _safe_min(subset["excess_return"]) if "excess_return" in subset else None

    summary = {
        "status": "ok",
        "research_only": True,
        "broker_action": "none",
        "start_date": pd.Timestamp(data["date"].iloc[0]),
        "end_date": pd.Timestamp(data["date"].iloc[-1]),
        "row_count": int(len(data)),
        "annual_row_count": int(len(annual)),
        "rolling_row_count": int(len(rolling)),
        "annual_positive_excess_ratio": annual_positive_ratio,
        "rolling_positive_excess_ratio": rolling_positive_ratios,
        "worst_annual_excess_return": _safe_min(annual["excess_return"]) if "excess_return" in annual else None,
        "worst_rolling_excess_return": worst_rolling_excess,
        "average_drawdown_edge": _safe_mean(annual["drawdown_edge"]) if "drawdown_edge" in annual else None,
        "evidence_status": _classify_evidence(
            annual_positive_ratio,
            rolling_positive_ratios,
            _safe_min(annual["excess_return"]) if "excess_return" in annual else None,
            worst_rolling_excess,
        ),
    }
    return annual.reset_index(drop=True), rolling.reset_index(drop=True), summary


_ANNUAL_CN_COLUMNS = {
    "year": "年份",
    "start_date": "开始日期",
    "end_date": "结束日期",
    "trading_days": "交易日数",
    "portfolio_return": "组合收益率",
    "baseline_return": "核心收益率",
    "excess_return": "超额收益率",
    "portfolio_max_drawdown": "组合最大回撤",
    "baseline_max_drawdown": "核心最大回撤",
    "drawdown_edge": "回撤优势",
    "portfolio_sharpe": "组合Sharpe",
    "baseline_sharpe": "核心Sharpe",
    "positive_excess": "是否跑赢核心",
    "sharpe_edge": "Sharpe优势",
}

_ROLLING_CN_COLUMNS = {
    "window_months": "窗口月数",
    "start_date": "开始日期",
    "end_date": "结束日期",
    "trading_days": "交易日数",
    "portfolio_return": "组合收益率",
    "baseline_return": "核心收益率",
    "excess_return": "超额收益率",
    "portfolio_max_drawdown": "组合最大回撤",
    "baseline_max_drawdown": "核心最大回撤",
    "drawdown_edge": "回撤优势",
    "portfolio_sharpe": "组合Sharpe",
    "baseline_sharpe": "核心Sharpe",
    "positive_excess": "是否跑赢核心",
    "sharpe_edge": "Sharpe优势",
}


def _to_chinese_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[mapping.get(column, column) for column in mapping])
    output = frame.copy()
    for column in ("start_date", "end_date"):
        if column in output.columns:
            output[column] = pd.to_datetime(output[column]).dt.strftime("%Y-%m-%d")
    return output.rename(columns=mapping)


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _fmt_num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def _write_report(
    *,
    name: str,
    annual_cn: pd.DataFrame,
    rolling_cn: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    equity_path: Path,
) -> Path:
    report_path = output_dir / "portfolio_stability_report.html"
    annual_preview = annual_cn.head(20).to_html(index=False, border=0, classes="data-table")
    rolling_preview = rolling_cn.head(40).to_html(index=False, border=0, classes="data-table")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{name} 组合稳定性审计</title>
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
  <h1>{name} 组合稳定性审计</h1>
  <div class="meta">
    <div><span class="badge">研究模式</span> broker_action=none，不开券商动作，不自动下单。</div>
    <div>输入曲线：{equity_path}</div>
    <div>样本区间：{_json_ready(summary.get("start_date"))} 至 {_json_ready(summary.get("end_date"))}</div>
    <div>年度跑赢占比：{_fmt_pct(summary.get("annual_positive_excess_ratio"))}</div>
    <div>证据状态：{summary.get("evidence_status")}</div>
  </div>
  <h2>年度稳定性</h2>
  {annual_preview}
  <h2>滚动窗口稳定性</h2>
  {rolling_preview}
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    return report_path


def run_portfolio_stability_review(
    *,
    equity_path: str | Path = DEFAULT_EQUITY_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    name: str = "portfolio",
    portfolio_col: str = "portfolio_equity",
    baseline_col: str = "core_equity",
    rolling_months: Iterable[int] = (3, 6, 12),
    min_year_trading_days: int = 2,
    min_window_trading_days: int = 40,
) -> PortfolioStabilityResult:
    resolved_equity_path = Path(equity_path)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    equity = pd.read_csv(resolved_equity_path)
    annual, rolling, summary = build_portfolio_stability(
        equity,
        portfolio_col=portfolio_col,
        baseline_col=baseline_col,
        rolling_months=rolling_months,
        min_year_trading_days=min_year_trading_days,
        min_window_trading_days=min_window_trading_days,
    )
    summary = {
        **summary,
        "name": name,
        "equity_path": str(resolved_equity_path),
        "portfolio_column": portfolio_col,
        "baseline_column": baseline_col,
    }

    annual_cn = _to_chinese_columns(annual, _ANNUAL_CN_COLUMNS)
    rolling_cn = _to_chinese_columns(rolling, _ROLLING_CN_COLUMNS)
    annual_path = resolved_output_dir / "portfolio_stability_annual_cn.csv"
    rolling_path = resolved_output_dir / "portfolio_stability_rolling_cn.csv"
    summary_path = resolved_output_dir / "portfolio_stability_summary.json"
    annual_cn.to_csv(annual_path, index=False, encoding="utf-8-sig")
    rolling_cn.to_csv(rolling_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(
        name=name,
        annual_cn=annual_cn,
        rolling_cn=rolling_cn,
        summary=summary,
        output_dir=resolved_output_dir,
        equity_path=resolved_equity_path,
    )
    return PortfolioStabilityResult(
        output_dir=resolved_output_dir,
        annual_path=annual_path,
        rolling_path=rolling_path,
        summary_path=summary_path,
        report_path=report_path,
        annual=annual,
        rolling=rolling,
        summary=summary,
    )
