"""Historical out-of-sample review helpers for stitched walk-forward curves."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_START = "2018-01-01"
DEFAULT_END = "2025-12-31"


@dataclass(frozen=True)
class CurveSpec:
    label: str
    path: Path
    category: str = "candidate"
    equity_column: str = "stitched_equity"


@dataclass(frozen=True)
class HistoricalOosReviewResult:
    output_dir: Path
    summary_path: Path
    chinese_summary_path: Path
    snapshot_path: Path
    report_path: Path


DEFAULT_CURVE_SPECS: tuple[CurveSpec, ...] = (
    CurveSpec(
        label="稳定核心_v2",
        category="core",
        path=Path("outputs/walk_forward/main_chinext_stable_v2_full_20260613_011322/oos_equity_stitched.csv"),
    ),
    CurveSpec(
        label="卫星质量_v2",
        category="satellite",
        path=Path("outputs/walk_forward/main_chinext_satellite_quality_v2_full/oos_equity_stitched.csv"),
    ),
    CurveSpec(
        label="组合源选择_validation6",
        category="portfolio",
        path=Path(
            "outputs/portfolio_source_selection/main_chinext_source_selection_chip_deep_h5_cap20_margin03_validation6_v1/oos_equity_stitched.csv"
        ),
    ),
    CurveSpec(
        label="高收益pos10风险倾斜_md25_capital",
        category="highgain_candidate",
        path=Path("outputs/walk_forward/wf_focus_highgain_pos10_mainch_risklean_full_md25_capital/oos_equity_stitched.csv"),
    ),
    CurveSpec(
        label="源选择pos10高收益margin02",
        category="portfolio_candidate",
        path=Path(
            "outputs/portfolio_source_selection/main_chinext_source_selection_pos10_highgain_source_margin02_validation6_v1/oos_equity_stitched.csv"
        ),
    ),
    CurveSpec(
        label="源选择pos10资本cap10_margin03",
        category="portfolio_candidate",
        path=Path("outputs/portfolio_source_selection/main_chinext_source_selection_pos10_cap10_margin03_validation6_v1/oos_equity_stitched.csv"),
    ),
)


CHINESE_COLUMNS = {
    "label": "策略/曲线",
    "category": "类别",
    "status": "状态",
    "coverage_status": "覆盖状态",
    "requested_start": "请求开始",
    "requested_end": "请求结束",
    "actual_start": "实际开始",
    "actual_end": "实际结束",
    "rows": "行数",
    "window_count": "窗口数",
    "start_equity": "期初权益",
    "end_equity": "期末权益",
    "total_return": "总收益",
    "cagr": "CAGR",
    "max_drawdown": "最大回撤",
    "sharpe": "Sharpe",
    "annual_volatility": "年化波动",
    "win_day_ratio": "日胜率",
    "return_to_drawdown": "收益回撤比",
    "path": "文件路径",
    "equity_column": "权益列",
    "coverage_note": "备注",
}


def _date_text(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid date value: {value}")
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _empty_summary_row(
    *,
    label: str,
    category: str = "",
    path: Path,
    equity_column: str,
    start: str,
    end: str,
    status: str,
    coverage_status: str,
    coverage_note: str,
) -> dict[str, Any]:
    return {
        "label": label,
        "category": category,
        "status": status,
        "coverage_status": coverage_status,
        "requested_start": _date_text(start),
        "requested_end": _date_text(end),
        "actual_start": None,
        "actual_end": None,
        "rows": 0,
        "window_count": 0,
        "start_equity": None,
        "end_equity": None,
        "total_return": None,
        "cagr": None,
        "max_drawdown": None,
        "sharpe": None,
        "annual_volatility": None,
        "win_day_ratio": None,
        "return_to_drawdown": None,
        "path": str(path),
        "equity_column": equity_column,
        "coverage_note": coverage_note,
    }


def _coverage_status(actual_start: pd.Timestamp, actual_end: pd.Timestamp, requested_start: pd.Timestamp, requested_end: pd.Timestamp) -> str:
    start_ok = actual_start <= requested_start + pd.Timedelta(days=10)
    end_ok = actual_end >= requested_end - pd.Timedelta(days=10)
    return "full_window" if start_ok and end_ok else "partial_window"


def _coverage_note(actual_start: pd.Timestamp, actual_end: pd.Timestamp, requested_start: pd.Timestamp, requested_end: pd.Timestamp) -> str:
    notes: list[str] = []
    if actual_start > requested_start + pd.Timedelta(days=10):
        notes.append(f"覆盖不足：起始晚于请求窗口，实际从 {actual_start.strftime('%Y-%m-%d')} 开始")
    if actual_end < requested_end - pd.Timedelta(days=10):
        notes.append(f"覆盖不足：结束早于请求窗口，实际到 {actual_end.strftime('%Y-%m-%d')} 截止")
    return "；".join(notes) if notes else "覆盖请求窗口"


def summarize_curve_frame(
    *,
    label: str,
    path: Path,
    frame: pd.DataFrame,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    equity_column: str = "stitched_equity",
    category: str = "",
) -> dict[str, Any]:
    requested_start = pd.Timestamp(_date_text(start))
    requested_end = pd.Timestamp(_date_text(end))
    if requested_start > requested_end:
        raise ValueError("start must be on or before end.")
    if "date" not in frame.columns or equity_column not in frame.columns:
        return _empty_summary_row(
            label=label,
            category=category,
            path=path,
            equity_column=equity_column,
            start=start,
            end=end,
            status="invalid_schema",
            coverage_status="no_coverage",
            coverage_note=f"缺少 date 或 {equity_column} 列",
        )

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data[equity_column] = pd.to_numeric(data[equity_column], errors="coerce")
    data = data.dropna(subset=["date", equity_column])
    data = data[(data["date"] >= requested_start) & (data["date"] <= requested_end)]
    data = data[data[equity_column] > 0].sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if data.empty:
        return _empty_summary_row(
            label=label,
            category=category,
            path=path,
            equity_column=equity_column,
            start=start,
            end=end,
            status="no_data_in_window",
            coverage_status="no_coverage",
            coverage_note="覆盖不足：请求窗口内没有有效权益数据",
        )

    series = data.set_index("date")[equity_column].astype(float).sort_index()
    returns = series.pct_change().dropna()
    running_max = series.cummax()
    drawdown = series / running_max - 1.0
    start_equity = float(series.iloc[0])
    end_equity = float(series.iloc[-1])
    total_return = end_equity / start_equity - 1.0 if start_equity > 0 else None
    years = max((series.index[-1] - series.index[0]).days / 365.25, 0.0)
    cagr = (end_equity / start_equity) ** (1.0 / years) - 1.0 if years > 0 and start_equity > 0 else None
    volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else None
    sharpe = float(np.sqrt(252) * returns.mean() / returns.std()) if len(returns) > 1 and float(returns.std()) != 0.0 else None
    max_drawdown = float(drawdown.min()) if not drawdown.empty else None
    win_day_ratio = float((returns > 0).mean()) if not returns.empty else None
    return_to_drawdown = (
        float(total_return / abs(max_drawdown))
        if total_return is not None and max_drawdown is not None and max_drawdown < 0
        else None
    )
    actual_start = pd.Timestamp(series.index[0])
    actual_end = pd.Timestamp(series.index[-1])
    window_count = int(data["window"].nunique()) if "window" in data.columns else 0

    return {
        "label": label,
        "category": category,
        "status": "ok",
        "coverage_status": _coverage_status(actual_start, actual_end, requested_start, requested_end),
        "requested_start": requested_start.strftime("%Y-%m-%d"),
        "requested_end": requested_end.strftime("%Y-%m-%d"),
        "actual_start": actual_start.strftime("%Y-%m-%d"),
        "actual_end": actual_end.strftime("%Y-%m-%d"),
        "rows": int(len(data)),
        "window_count": window_count,
        "start_equity": start_equity,
        "end_equity": end_equity,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "annual_volatility": volatility,
        "win_day_ratio": win_day_ratio,
        "return_to_drawdown": return_to_drawdown,
        "path": str(path),
        "equity_column": equity_column,
        "coverage_note": _coverage_note(actual_start, actual_end, requested_start, requested_end),
    }


def summarize_curve_file(project_root: Path, spec: CurveSpec, start: str = DEFAULT_START, end: str = DEFAULT_END) -> dict[str, Any]:
    path = spec.path if spec.path.is_absolute() else project_root / spec.path
    if not path.exists():
        return _empty_summary_row(
            label=spec.label,
            category=spec.category,
            path=path,
            equity_column=spec.equity_column,
            start=start,
            end=end,
            status="missing_file",
            coverage_status="no_coverage",
            coverage_note="覆盖不足：曲线文件不存在",
        )
    if path.stat().st_size == 0:
        return _empty_summary_row(
            label=spec.label,
            category=spec.category,
            path=path,
            equity_column=spec.equity_column,
            start=start,
            end=end,
            status="empty_file",
            coverage_status="no_coverage",
            coverage_note="覆盖不足：曲线文件为空",
        )
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        return _empty_summary_row(
            label=spec.label,
            category=spec.category,
            path=path,
            equity_column=spec.equity_column,
            start=start,
            end=end,
            status="read_error",
            coverage_status="no_coverage",
            coverage_note=f"读取失败：{exc}",
        )
    return summarize_curve_frame(
        label=spec.label,
        category=spec.category,
        path=path,
        frame=frame,
        start=start,
        end=end,
        equity_column=spec.equity_column,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return value


def _records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _json_safe(value) for key, value in row.items()} for row in frame.to_dict("records")]


def _summary_for_chinese(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = [column for column in CHINESE_COLUMNS if column in frame.columns]
    return frame[ordered].rename(columns=CHINESE_COLUMNS)


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _fmt_num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def _write_report(summary: pd.DataFrame, snapshot: dict[str, Any], output_dir: Path) -> Path:
    report_path = output_dir / "historical_oos_review.md"
    ok = summary[summary["status"] == "ok"].copy()
    if not ok.empty:
        ok = ok.sort_values(["total_return", "sharpe"], ascending=[False, False], na_position="last")
    lines = [
        "# 2018-2025 历史 OOS 曲线复盘",
        "",
        f"- 窗口：{snapshot['requested_start']} 至 {snapshot['requested_end']}",
        f"- 曲线数量：{snapshot['curve_count']}；有效曲线：{snapshot['ok_curve_count']}；覆盖不足/缺失：{snapshot['coverage_shortfall_count']}",
        f"- 总收益最高：{snapshot.get('best_by_total_return') or 'n/a'}",
        f"- 收益回撤比最高：{snapshot.get('best_by_return_to_drawdown') or 'n/a'}",
        "",
        "## 指标对比",
        "",
        "| 策略/曲线 | 状态 | 覆盖 | 实际区间 | 总收益 | CAGR | 最大回撤 | Sharpe | 收益回撤比 | 备注 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    ordered = ok if not ok.empty else summary
    for _, row in ordered.iterrows():
        actual = "n/a"
        if pd.notna(row.get("actual_start")) and pd.notna(row.get("actual_end")):
            actual = f"{row.get('actual_start')} ~ {row.get('actual_end')}"
        lines.append(
            "| {label} | {status} | {coverage} | {actual} | {total_return} | {cagr} | {max_drawdown} | {sharpe} | {rtd} | {note} |".format(
                label=row.get("label", ""),
                status=row.get("status", ""),
                coverage=row.get("coverage_status", ""),
                actual=actual,
                total_return=_fmt_pct(row.get("total_return")),
                cagr=_fmt_pct(row.get("cagr")),
                max_drawdown=_fmt_pct(row.get("max_drawdown")),
                sharpe=_fmt_num(row.get("sharpe")),
                rtd=_fmt_num(row.get("return_to_drawdown")),
                note=row.get("coverage_note", ""),
            )
        )
    lines.extend(
        [
            "",
            "## 读数提示",
            "",
            "- 本报告只读取已有滚动训练/OOS 曲线，不重新训练、不改参数、不连接券商。",
            "- 若实际开始日期晚于 2018-01-01，说明该曲线受训练窗口约束，仅能评价现有 OOS 覆盖段。",
            "- 后续优化优先看：最大回撤、收益回撤比、Sharpe、覆盖区间是否一致，再决定是否扩展训练或调整资金分配。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_historical_oos_review(
    *,
    project_root: Path,
    output_dir: Path,
    curve_specs: Iterable[CurveSpec] = DEFAULT_CURVE_SPECS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> HistoricalOosReviewResult:
    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [summarize_curve_file(project_root, spec, start=start, end=end) for spec in curve_specs]
    summary = pd.DataFrame(rows)
    summary_path = output_dir / "historical_oos_summary.csv"
    chinese_summary_path = output_dir / "historical_oos_summary_cn.csv"
    snapshot_path = output_dir / "historical_oos_snapshot.json"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    _summary_for_chinese(summary).to_csv(chinese_summary_path, index=False, encoding="utf-8")

    ok = summary[summary["status"] == "ok"].copy()
    best_by_total_return = None
    best_by_return_to_drawdown = None
    if not ok.empty:
        best_return = ok.sort_values("total_return", ascending=False, na_position="last").iloc[0]
        best_by_total_return = str(best_return["label"])
        rtd_ok = ok.dropna(subset=["return_to_drawdown"])
        if not rtd_ok.empty:
            best_rtd = rtd_ok.sort_values("return_to_drawdown", ascending=False, na_position="last").iloc[0]
            best_by_return_to_drawdown = str(best_rtd["label"])

    coverage_shortfall = summary[(summary["status"] != "ok") | (summary["coverage_status"] != "full_window")]
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "requested_start": _date_text(start),
        "requested_end": _date_text(end),
        "curve_count": int(len(summary)),
        "ok_curve_count": int(len(ok)),
        "coverage_shortfall_count": int(len(coverage_shortfall)),
        "best_by_total_return": best_by_total_return,
        "best_by_return_to_drawdown": best_by_return_to_drawdown,
        "summary_path": str(summary_path),
        "chinese_summary_path": str(chinese_summary_path),
        "records": _records_for_json(summary),
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(summary, snapshot, output_dir)

    return HistoricalOosReviewResult(
        output_dir=output_dir,
        summary_path=summary_path,
        chinese_summary_path=chinese_summary_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
    )
