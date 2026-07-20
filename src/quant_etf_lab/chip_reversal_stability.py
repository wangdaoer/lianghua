"""Year-by-year stability review for chip-reversal research events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_EVENTS_PATH = Path("outputs/research/chip_reversal_lab_latest/chip_reversal_events.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/chip_reversal_stability_latest")
EVIDENCE_STATUS_RANK = {"stable_positive": 0, "mixed": 1, "weak": 2}


@dataclass(frozen=True)
class ChipReversalStabilityResult:
    output_dir: Path
    annual_path: Path
    summary_path: Path
    snapshot_path: Path
    report_path: Path
    annual: pd.DataFrame
    summary: pd.DataFrame
    snapshot: dict[str, Any]


def _trade_return_column(horizon: int) -> str:
    return f"trade_return_{int(horizon)}d"


def _profit_factor(returns: pd.Series) -> float | None:
    gross_profit = float(returns[returns > 0].sum())
    gross_loss = abs(float(returns[returns < 0].sum()))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else None
    return gross_profit / gross_loss


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if pd.isna(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: _json_safe(value) for key, value in row.items()} for row in frame.to_dict("records")]


def build_chip_reversal_stability(
    events: pd.DataFrame,
    horizons: Iterable[int] = (1, 2, 5),
    min_events: int = 30,
    min_years: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if min_events < 1:
        raise ValueError("min_events must be at least 1.")
    if min_years < 1:
        raise ValueError("min_years must be at least 1.")
    required = {"date", "board", "score_bucket"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    base = events.copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base = base.dropna(subset=["date"])
    base["year"] = base["date"].dt.year.astype(int)
    base["board"] = base["board"].astype(str)
    base["score_bucket"] = base["score_bucket"].astype(str)

    for horizon in horizons:
        column = _trade_return_column(int(horizon))
        if column not in base.columns:
            continue
        data = base[["year", "board", "score_bucket", column]].copy()
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=[column])
        for (year, board, bucket), group in data.groupby(["year", "board", "score_bucket"], dropna=False):
            returns = group[column].astype(float)
            event_count = int(len(returns))
            if event_count < min_events:
                continue
            profit_factor = _profit_factor(returns)
            avg_trade_return = float(returns.mean())
            win_rate = float((returns > 0).mean())
            positive_edge = bool(avg_trade_return > 0 and profit_factor is not None and profit_factor > 1.0)
            rows.append(
                {
                    "year": int(year),
                    "board": str(board),
                    "score_bucket": str(bucket),
                    "horizon": int(horizon),
                    "event_count": event_count,
                    "win_rate": win_rate,
                    "avg_trade_return": avg_trade_return,
                    "median_trade_return": float(returns.median()),
                    "profit_factor": profit_factor,
                    "positive_edge": positive_edge,
                }
            )

    annual = pd.DataFrame(rows)
    if annual.empty:
        return annual, pd.DataFrame(
            columns=[
                "board",
                "score_bucket",
                "horizon",
                "year_count",
                "positive_years",
                "positive_year_ratio",
                "min_event_count",
                "avg_trade_return_mean",
                "worst_year_trade_return",
                "best_year_trade_return",
                "avg_win_rate",
                "avg_profit_factor",
                "evidence_status",
            ]
        )

    summary_rows: list[dict[str, Any]] = []
    for (board, bucket, horizon), group in annual.groupby(["board", "score_bucket", "horizon"], dropna=False):
        positive_years = int(group["positive_edge"].sum())
        year_count = int(group["year"].nunique())
        positive_year_ratio = positive_years / year_count if year_count else 0.0
        worst_year = float(group["avg_trade_return"].min())
        if year_count >= min_years and positive_year_ratio >= 0.75 and worst_year > 0:
            evidence_status = "stable_positive"
        elif positive_year_ratio >= 0.5:
            evidence_status = "mixed"
        else:
            evidence_status = "weak"
        summary_rows.append(
            {
                "board": str(board),
                "score_bucket": str(bucket),
                "horizon": int(horizon),
                "year_count": year_count,
                "positive_years": positive_years,
                "positive_year_ratio": float(positive_year_ratio),
                "min_event_count": int(group["event_count"].min()),
                "avg_trade_return_mean": float(group["avg_trade_return"].mean()),
                "worst_year_trade_return": worst_year,
                "best_year_trade_return": float(group["avg_trade_return"].max()),
                "avg_win_rate": float(group["win_rate"].mean()),
                "avg_profit_factor": float(group["profit_factor"].replace(np.inf, np.nan).mean()),
                "evidence_status": evidence_status,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["_evidence_rank"] = summary["evidence_status"].map(EVIDENCE_STATUS_RANK).fillna(99)
    summary = summary.sort_values(
        ["_evidence_rank", "positive_year_ratio", "avg_trade_return_mean", "avg_win_rate"],
        ascending=[True, False, False, False],
    ).drop(columns=["_evidence_rank"])
    return annual.reset_index(drop=True), summary.reset_index(drop=True)


def _load_events_subset(events_path: Path, horizons: Iterable[int]) -> pd.DataFrame:
    header = pd.read_csv(events_path, nrows=0)
    wanted = ["date", "board", "score_bucket"]
    wanted.extend(_trade_return_column(int(horizon)) for horizon in horizons)
    usecols = [column for column in wanted if column in header.columns]
    return pd.read_csv(events_path, usecols=usecols)


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _fmt_num(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def _write_report(summary: pd.DataFrame, snapshot: dict[str, Any], output_dir: Path) -> Path:
    report_path = output_dir / "chip_reversal_stability.md"
    ranked = summary.copy()
    if not ranked.empty:
        ranked["_evidence_rank"] = ranked["evidence_status"].map(EVIDENCE_STATUS_RANK).fillna(99)
        ranked = ranked.sort_values(
            ["_evidence_rank", "positive_year_ratio", "avg_trade_return_mean"],
            ascending=[True, False, False],
        ).drop(columns=["_evidence_rank"])
    lines = [
        "# 筹码反转年度稳定性复盘",
        "",
        f"- 事件文件：`{snapshot['events_path']}`",
        f"- 年度行数：{snapshot['annual_row_count']}；组合行数：{snapshot['summary_row_count']}",
        f"- stable_positive 组合数：{snapshot['stable_positive_group_count']}",
        "",
        "## 年度稳定性",
        "",
        "| 板块 | 桶 | 持有期 | 年份数 | 正收益年份占比 | 年均交易收益 | 最差年份 | 平均胜率 | 状态 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in ranked.head(30).iterrows():
        lines.append(
            "| {board} | {bucket} | {horizon} | {years} | {ratio} | {avg} | {worst} | {win} | {status} |".format(
                board=row.get("board"),
                bucket=row.get("score_bucket"),
                horizon=int(row.get("horizon") or 0),
                years=int(row.get("year_count") or 0),
                ratio=_fmt_pct(row.get("positive_year_ratio")),
                avg=_fmt_pct(row.get("avg_trade_return_mean")),
                worst=_fmt_pct(row.get("worst_year_trade_return")),
                win=_fmt_pct(row.get("avg_win_rate")),
                status=row.get("evidence_status"),
            )
        )
    lines.extend(
        [
            "",
            "## 判断规则",
            "",
            "- `stable_positive`：年份数达到要求、正收益年份占比不低于 75%，且最差年份平均交易收益仍为正。",
            "- `mixed`：正收益年份占比不低于 50%，但稳定性仍不足。",
            "- 这仍是事件层统计，不等同于组合资金曲线；下一步要进入卫星组合回测验证。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_chip_reversal_stability_review(
    *,
    events_path: str | Path = DEFAULT_EVENTS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    horizons: Iterable[int] = (1, 2, 5),
    min_events: int = 30,
    min_years: int = 2,
) -> ChipReversalStabilityResult:
    resolved_events = Path(events_path)
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)

    events = _load_events_subset(resolved_events, horizons)
    annual, summary = build_chip_reversal_stability(events, horizons=horizons, min_events=min_events, min_years=min_years)

    annual_path = resolved_output / "chip_reversal_annual_stability.csv"
    summary_path = resolved_output / "chip_reversal_stability_summary.csv"
    snapshot_path = resolved_output / "chip_reversal_stability_snapshot.json"
    annual.to_csv(annual_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    stable_count = int((summary["evidence_status"] == "stable_positive").sum()) if not summary.empty else 0
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok" if not summary.empty else "no_summary",
        "events_path": str(resolved_events),
        "output_dir": str(resolved_output),
        "horizons": [int(horizon) for horizon in horizons],
        "min_events": int(min_events),
        "min_years": int(min_years),
        "annual_row_count": int(len(annual)),
        "summary_row_count": int(len(summary)),
        "stable_positive_group_count": stable_count,
        "annual_path": str(annual_path),
        "summary_path": str(summary_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(resolved_output / "chip_reversal_stability.md"),
        "top_groups": _records(summary.head(20)),
    }
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(summary, snapshot, resolved_output)
    snapshot["report_path"] = str(report_path)
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return ChipReversalStabilityResult(
        output_dir=resolved_output,
        annual_path=annual_path,
        summary_path=summary_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        annual=annual,
        summary=summary,
        snapshot=snapshot,
    )
