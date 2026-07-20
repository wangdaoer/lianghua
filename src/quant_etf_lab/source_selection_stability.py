"""Stability review for portfolio source-selection walk-forward runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SourceSelectionStabilityResult:
    output_dir: Path
    metrics_path: Path
    window_path: Path
    source_path: Path
    summary_path: Path
    report_path: Path
    metrics: pd.DataFrame
    windows: pd.DataFrame
    sources: pd.DataFrame
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


def _run_dir_from_value(value: str | Path) -> Path:
    path = Path(value)
    return path


def _read_curve(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "oos_equity_stitched.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing source-selection curve: {path}")
    curve = pd.read_csv(path)
    if "stitched_equity" not in curve.columns:
        raise ValueError(f"{path} missing stitched_equity column")
    if "date" not in curve.columns:
        raise ValueError(f"{path} missing date column")
    data = curve[["date", "stitched_equity"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["stitched_equity"] = pd.to_numeric(data["stitched_equity"], errors="coerce")
    data = data.dropna(subset=["date", "stitched_equity"])
    data = data[data["stitched_equity"] > 0.0]
    data = data.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(data) < 2:
        raise ValueError(f"{path} must contain at least two valid equity rows")
    return data


def _read_windows(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "portfolio_walk_forward_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "window" not in frame.columns:
        return pd.DataFrame()
    output = frame.copy()
    for column in (
        "test_total_return",
        "test_max_drawdown",
        "test_sharpe",
        "train_total_return",
        "train_max_drawdown",
        "train_sharpe",
    ):
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    for column in ("selected_source", "selected_candidate", "selected_source_group"):
        if column not in output.columns:
            output[column] = ""
        output[column] = output[column].fillna("").astype(str)
    return output.reset_index(drop=True)


def _max_drawdown(equity: pd.Series) -> float:
    clean = pd.to_numeric(equity, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float((clean / clean.cummax() - 1.0).min())


def _sharpe(equity: pd.Series) -> float | None:
    returns = pd.to_numeric(equity, errors="coerce").pct_change().dropna()
    if len(returns) < 2:
        return None
    std = float(returns.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return None
    return float(returns.mean() / std * np.sqrt(252.0))


def _curve_metrics(label: str, run_dir: Path, curve: pd.DataFrame) -> dict[str, Any]:
    start_equity = float(curve["stitched_equity"].iloc[0])
    end_equity = float(curve["stitched_equity"].iloc[-1])
    returns = curve["stitched_equity"].pct_change().dropna()
    return {
        "label": label,
        "run_dir": str(run_dir),
        "start_date": pd.Timestamp(curve["date"].iloc[0]),
        "end_date": pd.Timestamp(curve["date"].iloc[-1]),
        "trading_days": int(len(curve)),
        "total_return": end_equity / start_equity - 1.0,
        "max_drawdown": _max_drawdown(curve["stitched_equity"]),
        "sharpe": _sharpe(curve["stitched_equity"]),
        "positive_day_ratio": float((returns > 0.0).mean()) if not returns.empty else None,
        "worst_day_return": float(returns.min()) if not returns.empty else None,
        "best_day_return": float(returns.max()) if not returns.empty else None,
        "final_equity": end_equity,
    }


def _window_contribution(label: str, windows: pd.DataFrame) -> pd.DataFrame:
    if windows.empty or "test_total_return" not in windows.columns:
        return pd.DataFrame(
            columns=[
                "label",
                "window",
                "selected_source",
                "selected_candidate",
                "selected_source_group",
                "test_total_return",
                "test_max_drawdown",
                "test_sharpe",
                "growth_before",
                "growth_after",
                "growth_contribution",
                "contribution_share",
                "positive_window",
            ]
        )
    rows: list[dict[str, Any]] = []
    growth = 1.0
    for idx, row in windows.reset_index(drop=True).iterrows():
        test_return = float(row.get("test_total_return", 0.0) or 0.0)
        contribution = growth * test_return
        growth_after = growth * (1.0 + test_return)
        rows.append(
            {
                "label": label,
                "window_index": int(idx + 1),
                "window": row.get("window", ""),
                "selected_source": row.get("selected_source", ""),
                "selected_candidate": row.get("selected_candidate", ""),
                "selected_source_group": row.get("selected_source_group", ""),
                "test_total_return": test_return,
                "test_max_drawdown": row.get("test_max_drawdown"),
                "test_sharpe": row.get("test_sharpe"),
                "growth_before": growth,
                "growth_after": growth_after,
                "growth_contribution": contribution,
                "positive_window": bool(test_return > 0.0),
            }
        )
        growth = growth_after
    output = pd.DataFrame(rows)
    total_return = float(growth - 1.0)
    output["contribution_share"] = output["growth_contribution"] / total_return if total_return != 0.0 else np.nan
    return output


def _source_usage(windows: pd.DataFrame) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame(
            columns=[
                "label",
                "selected_source",
                "selected_count",
                "selected_ratio",
                "average_test_return",
                "positive_test_ratio",
                "average_test_drawdown",
                "average_test_sharpe",
            ]
        )
    work = windows.copy()
    work["selected_source"] = work["selected_source"].replace("", "unknown")
    grouped = work.groupby(["label", "selected_source"], dropna=False)
    rows: list[dict[str, Any]] = []
    totals = work.groupby("label")["window"].count().to_dict()
    for (label, source), group in grouped:
        returns = pd.to_numeric(group.get("test_total_return"), errors="coerce")
        drawdown = pd.to_numeric(group.get("test_max_drawdown"), errors="coerce")
        sharpe = pd.to_numeric(group.get("test_sharpe"), errors="coerce")
        rows.append(
            {
                "label": label,
                "selected_source": source,
                "selected_count": int(len(group)),
                "selected_ratio": float(len(group) / totals.get(label, len(group))),
                "average_test_return": float(returns.mean()) if not returns.dropna().empty else None,
                "positive_test_ratio": float((returns > 0.0).mean()) if not returns.dropna().empty else None,
                "average_test_drawdown": float(drawdown.mean()) if not drawdown.dropna().empty else None,
                "average_test_sharpe": float(sharpe.mean()) if not sharpe.dropna().empty else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["label", "selected_count", "average_test_return"], ascending=[True, False, False])


def _apply_baseline(metrics: pd.DataFrame, baseline_label: str | None) -> pd.DataFrame:
    output = metrics.copy()
    output["excess_return_vs_baseline"] = np.nan
    output["drawdown_edge_vs_baseline"] = np.nan
    output["sharpe_edge_vs_baseline"] = np.nan
    if not baseline_label or baseline_label not in set(output["label"]):
        return output
    baseline = output[output["label"].eq(baseline_label)].iloc[0]
    output["excess_return_vs_baseline"] = output["total_return"] - float(baseline["total_return"])
    output["drawdown_edge_vs_baseline"] = output["max_drawdown"] - float(baseline["max_drawdown"])
    if pd.notna(baseline["sharpe"]):
        output["sharpe_edge_vs_baseline"] = output["sharpe"] - float(baseline["sharpe"])
    return output


def build_source_selection_stability(
    run_dirs: Mapping[str, str | Path],
    *,
    baseline_label: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not run_dirs:
        raise ValueError("at least one run directory is required")

    metrics_rows: list[dict[str, Any]] = []
    window_frames: list[pd.DataFrame] = []
    for label, raw_path in run_dirs.items():
        clean_label = str(label).strip()
        if not clean_label:
            raise ValueError("run label cannot be empty")
        run_dir = _run_dir_from_value(raw_path)
        curve = _read_curve(run_dir)
        windows = _read_windows(run_dir)
        metrics_rows.append(_curve_metrics(clean_label, run_dir, curve))
        window_frames.append(_window_contribution(clean_label, windows))

    metrics = pd.DataFrame(metrics_rows)
    metrics = _apply_baseline(metrics, baseline_label)
    windows = pd.concat(window_frames, ignore_index=True) if window_frames else pd.DataFrame()
    sources = _source_usage(windows)

    best_return_label = str(metrics.sort_values(["total_return", "sharpe"], ascending=[False, False]).iloc[0]["label"])
    best_sharpe_label = str(metrics.sort_values(["sharpe", "total_return"], ascending=[False, False]).iloc[0]["label"])
    shallowest_drawdown_label = str(metrics.sort_values(["max_drawdown", "total_return"], ascending=[False, False]).iloc[0]["label"])
    summary = {
        "status": "ok",
        "research_only": True,
        "broker_action": "none",
        "run_count": int(len(metrics)),
        "baseline_label": baseline_label,
        "best_total_return_label": best_return_label,
        "best_sharpe_label": best_sharpe_label,
        "shallowest_drawdown_label": shallowest_drawdown_label,
        "best_total_return": float(metrics.loc[metrics["label"].eq(best_return_label), "total_return"].iloc[0]),
        "best_total_return_max_drawdown": float(
            metrics.loc[metrics["label"].eq(best_return_label), "max_drawdown"].iloc[0]
        ),
        "best_total_return_sharpe": _json_ready(metrics.loc[metrics["label"].eq(best_return_label), "sharpe"].iloc[0]),
    }
    return metrics.reset_index(drop=True), windows.reset_index(drop=True), sources.reset_index(drop=True), summary


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _format_float(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def _write_report(
    *,
    output_dir: Path,
    name: str,
    metrics: pd.DataFrame,
    windows: pd.DataFrame,
    sources: pd.DataFrame,
    summary: dict[str, Any],
) -> Path:
    report_path = output_dir / "source_selection_stability_report.md"
    metric_rows = [
        "| 标的 | 总收益 | 最大回撤 | Sharpe | 相对基准超额 | 正收益天数占比 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.itertuples(index=False):
        metric_rows.append(
            "| {label} | {ret} | {dd} | {sharpe} | {excess} | {pos} |".format(
                label=row.label,
                ret=_format_pct(row.total_return),
                dd=_format_pct(row.max_drawdown),
                sharpe=_format_float(row.sharpe),
                excess=_format_pct(row.excess_return_vs_baseline),
                pos=_format_pct(row.positive_day_ratio),
            )
        )

    top_windows = windows.sort_values(["label", "growth_contribution"], ascending=[True, False]).groupby("label").head(3)
    window_rows = [
        "| 标的 | 窗口 | 来源 | 测试收益 | 对总收益贡献 | 贡献占比 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in top_windows.itertuples(index=False):
        window_rows.append(
            "| {label} | {window} | {source} | {ret} | {contrib} | {share} |".format(
                label=row.label,
                window=row.window,
                source=row.selected_source,
                ret=_format_pct(row.test_total_return),
                contrib=_format_pct(row.growth_contribution),
                share=_format_pct(row.contribution_share),
            )
        )

    source_rows = [
        "| 标的 | 来源 | 入选次数 | 入选占比 | 平均测试收益 | 正收益窗口占比 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sources.itertuples(index=False):
        source_rows.append(
            "| {label} | {source} | {count} | {ratio} | {ret} | {pos} |".format(
                label=row.label,
                source=row.selected_source,
                count=int(row.selected_count),
                ratio=_format_pct(row.selected_ratio),
                ret=_format_pct(row.average_test_return),
                pos=_format_pct(row.positive_test_ratio),
            )
        )

    text = f"""# {name}

本报告只分析已经生成的 source-selection 滚动回测结果，不连接券商，不自动下单。

## 汇总

- 最优总收益标的：`{summary.get("best_total_return_label")}`，总收益 `{_format_pct(summary.get("best_total_return"))}`，最大回撤 `{_format_pct(summary.get("best_total_return_max_drawdown"))}`，Sharpe `{_format_float(summary.get("best_total_return_sharpe"))}`
- 最优 Sharpe 标的：`{summary.get("best_sharpe_label")}`
- 最浅回撤标的：`{summary.get("shallowest_drawdown_label")}`
- 基准标的：`{summary.get("baseline_label") or "未设置"}`

## 曲线对比

{chr(10).join(metric_rows)}

## 主要窗口贡献

{chr(10).join(window_rows)}

## 来源稳定性

{chr(10).join(source_rows)}

## 输出文件

- `source_selection_curve_metrics.csv`
- `source_selection_window_contribution.csv`
- `source_selection_source_usage.csv`
- `source_selection_stability_summary.json`
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def run_source_selection_stability_review(
    run_dirs: Mapping[str, str | Path],
    *,
    output_dir: str | Path,
    baseline_label: str | None = None,
    name: str = "Source-selection stability review",
) -> SourceSelectionStabilityResult:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    metrics, windows, sources, summary = build_source_selection_stability(
        run_dirs,
        baseline_label=baseline_label,
    )
    summary = {**summary, "name": name, "run_dirs": {str(label): str(path) for label, path in run_dirs.items()}}

    metrics_path = resolved_output_dir / "source_selection_curve_metrics.csv"
    window_path = resolved_output_dir / "source_selection_window_contribution.csv"
    source_path = resolved_output_dir / "source_selection_source_usage.csv"
    summary_path = resolved_output_dir / "source_selection_stability_summary.json"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    windows.to_csv(window_path, index=False, encoding="utf-8-sig")
    sources.to_csv(source_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(
        output_dir=resolved_output_dir,
        name=name,
        metrics=metrics,
        windows=windows,
        sources=sources,
        summary=summary,
    )

    return SourceSelectionStabilityResult(
        output_dir=resolved_output_dir,
        metrics_path=metrics_path,
        window_path=window_path,
        source_path=source_path,
        summary_path=summary_path,
        report_path=report_path,
        metrics=metrics,
        windows=windows,
        sources=sources,
        summary=summary,
    )
