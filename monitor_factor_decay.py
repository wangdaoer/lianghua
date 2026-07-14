"""Monitor point-in-time factor decay without changing model parameters."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from analyze_factor_selection_alpha import (
    DEFAULT_HORIZONS,
    attach_forward_open_returns,
    compute_healthy_breadth_state,
)
from multifactor_observation_evolution import prepare_factor_panel
from run_multifactor_observation_evolution import load_market_data


FACTOR = "liquidity_stability_20"


def _rank_ic(group: pd.DataFrame, factor: str, return_column: str) -> float:
    values = pd.DataFrame(
        {
            "factor": pd.to_numeric(group[factor], errors="coerce"),
            "forward": pd.to_numeric(group[return_column], errors="coerce"),
        }
    ).dropna()
    if len(values) < 20 or values["factor"].nunique() < 2 or values["forward"].nunique() < 2:
        return math.nan
    return float(values["factor"].rank().corr(values["forward"].rank()))


def compute_daily_factor_observations(
    factor_panel: pd.DataFrame,
    breadth_state: pd.DataFrame,
    factor: str = FACTOR,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Compute matured daily RankIC only when the hard breadth gate is open."""
    if factor not in factor_panel:
        raise ValueError(f"missing monitored factor: {factor}")
    enriched = attach_forward_open_returns(factor_panel, horizons)
    enriched = enriched.merge(
        breadth_state.reset_index()[["date", "breadth_gap", "breadth_healthy"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    eligible = enriched.get("score_eligible", pd.Series(True, index=enriched.index))
    enriched = enriched.loc[
        eligible.fillna(False) & enriched["breadth_healthy"].fillna(False)
    ]
    rows: list[dict[str, object]] = []
    for date, group in enriched.groupby("date", sort=True):
        factor_rank = pd.to_numeric(group[factor], errors="coerce").rank(pct=True)
        for horizon in horizons:
            return_column = f"forward_open_return_{int(horizon)}d"
            forward = pd.to_numeric(group[return_column], errors="coerce")
            valid = factor_rank.notna() & forward.notna()
            top = valid & factor_rank.ge(0.8)
            bottom = valid & factor_rank.le(0.2)
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "factor": factor,
                    "horizon": int(horizon),
                    "observations": int(valid.sum()),
                    "rank_ic": _rank_ic(group.loc[valid], factor, return_column),
                    "top_bottom_spread": (
                        float(forward.loc[top].mean() - forward.loc[bottom].mean())
                        if top.any() and bottom.any()
                        else math.nan
                    ),
                    "breadth_gap": float(group["breadth_gap"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def summarize_factor_decay(
    daily: pd.DataFrame,
    *,
    selection_start: pd.Timestamp,
    selection_end: pd.Timestamp,
    asof_date: pd.Timestamp,
    recent_signal_days: int,
) -> pd.DataFrame:
    """Compare a fixed selection reference with the latest matured observations."""
    if recent_signal_days < 1:
        raise ValueError("recent_signal_days must be positive")
    rows: list[dict[str, object]] = []
    for horizon, group in daily.groupby("horizon", sort=True):
        matured = group.loc[group["rank_ic"].notna() & group["date"].le(asof_date)]
        reference = matured.loc[
            matured["date"].between(selection_start, selection_end)
        ]
        recent = matured.loc[matured["date"].gt(selection_end)].tail(recent_signal_days)
        reference_ic = float(reference["rank_ic"].mean()) if not reference.empty else math.nan
        recent_ic = float(recent["rank_ic"].mean()) if not recent.empty else math.nan
        if len(recent) < min(10, recent_signal_days):
            status = "insufficient_history"
        elif reference_ic > 0.0 and recent_ic < 0.0:
            status = "direction_reversal"
        elif reference_ic > 0.0 and recent_ic < reference_ic * 0.5:
            status = "weakened"
        else:
            status = "stable"
        rows.append(
            {
                "factor": str(group["factor"].iloc[0]),
                "horizon": int(horizon),
                "selection_start": selection_start.date().isoformat(),
                "selection_end": selection_end.date().isoformat(),
                "reference_days": int(len(reference)),
                "reference_mean_rank_ic": reference_ic,
                "reference_positive_ic_ratio": (
                    float(reference["rank_ic"].gt(0.0).mean())
                    if not reference.empty
                    else math.nan
                ),
                "recent_days": int(len(recent)),
                "recent_start": (
                    pd.Timestamp(recent["date"].min()).date().isoformat()
                    if not recent.empty
                    else None
                ),
                "recent_end": (
                    pd.Timestamp(recent["date"].max()).date().isoformat()
                    if not recent.empty
                    else None
                ),
                "recent_mean_rank_ic": recent_ic,
                "recent_positive_ic_ratio": (
                    float(recent["rank_ic"].gt(0.0).mean())
                    if not recent.empty
                    else math.nan
                ),
                "recent_mean_top_bottom_spread": (
                    float(recent["top_bottom_spread"].mean())
                    if not recent.empty
                    else math.nan
                ),
                "rank_ic_delta": recent_ic - reference_ic,
                "status": status,
                "latest_mature_signal_date": (
                    pd.Timestamp(matured["date"].max()).date().isoformat()
                    if not matured.empty
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def classify_overall_status(summary: pd.DataFrame) -> str:
    statuses = summary.get("status", pd.Series(dtype="string")).astype("string")
    if statuses.empty or statuses.eq("insufficient_history").all():
        return "insufficient_history"
    reversals = int(statuses.eq("direction_reversal").sum())
    if reversals >= 2:
        return "direction_reversal"
    if reversals == 1 or statuses.eq("weakened").any():
        return "weakened"
    return "stable"


def build_current_observation(
    factor_panel: pd.DataFrame,
    asof_date: pd.Timestamp,
    top_n: int,
    factor: str = FACTOR,
) -> pd.DataFrame:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    rows = factor_panel.loc[factor_panel["date"].eq(asof_date)].copy()
    eligible = rows.get("score_eligible", pd.Series(True, index=rows.index))
    rows = rows.loc[eligible.fillna(False)]
    rows["factor_pct_rank"] = pd.to_numeric(rows[factor], errors="coerce").rank(
        method="average", pct=True
    )
    rows["median_amount_20"] = pd.to_numeric(
        rows.get("liquidity_20"), errors="coerce"
    ).map(lambda value: math.expm1(value) if pd.notna(value) else math.nan)
    columns = [
        name
        for name in (
            "date",
            "symbol",
            "stock_name",
            "close",
            factor,
            "factor_pct_rank",
            "median_amount_20",
            "momentum_20",
            "momentum_60",
            "breakout_distance_20",
        )
        if name in rows
    ]
    out = rows.sort_values(
        ["factor_pct_rank", "symbol"], ascending=[False, True], kind="mergesort"
    ).head(top_n)[columns]
    out.insert(len(out.columns), "observation_only", True)
    out.insert(len(out.columns), "trade_instruction", False)
    return out.reset_index(drop=True)


def update_monitor_history(
    history_path: Path,
    asof_date: pd.Timestamp,
    summary: pd.DataFrame,
    overall_status: str,
) -> pd.DataFrame:
    current = summary.copy()
    current.insert(0, "asof_date", asof_date.date().isoformat())
    current.insert(1, "overall_status", overall_status)
    if history_path.exists():
        history = pd.read_csv(history_path, dtype={"asof_date": "string"})
        current = pd.concat([history, current], ignore_index=True, sort=False)
    current = current.drop_duplicates(
        subset=["asof_date", "factor", "horizon"], keep="last"
    ).sort_values(["asof_date", "factor", "horizon"], kind="mergesort")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = history_path.with_suffix(history_path.suffix + ".tmp")
    current.to_csv(temporary, index=False)
    temporary.replace(history_path)
    return current


def _pct(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "n/a" if pd.isna(number) else f"{float(number):.2%}"


def _number(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "n/a" if pd.isna(number) else f"{float(number):.4f}"


def build_report(
    summary: pd.DataFrame,
    overall_status: str,
    asof_date: pd.Timestamp,
    breadth_row: pd.Series,
    observation: pd.DataFrame,
) -> str:
    lines = [
        "# 选股因子衰减观察",
        "",
        "> 研究/模拟盘用途，不构成交易指令。本报告不会自动修改模型参数。",
        "",
        f"- 数据截至：{asof_date.date().isoformat()}",
        f"- 监测因子：`{FACTOR}`",
        f"- 总体状态：`{overall_status}`",
        f"- 当日动量广度：{_pct(breadth_row.get('breadth_20'))}",
        f"- 广度相对 20 日均线：{_pct(breadth_row.get('breadth_gap'))}",
        f"- 硬广度门控：{'开放' if bool(breadth_row.get('breadth_healthy')) else '关闭'}",
        "",
        "## 前瞻 RankIC",
        "",
        "| 周期 | 固定参考期 | 最近成熟样本 | 变化 | 最近正IC比例 | 状态 | 最新成熟信号日 |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {int(row['horizon'])}日 | {_number(row['reference_mean_rank_ic'])} | "
            f"{_number(row['recent_mean_rank_ic'])} | {_number(row['rank_ic_delta'])} | "
            f"{_pct(row['recent_positive_ic_ratio'])} | {row['status']} | "
            f"{row['latest_mature_signal_date'] or 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## 当日高稳定性观察",
            "",
            "以下仅按因子排序，不是买入名单。",
            "",
            "| 代码 | 名称 | 因子分位 | 20日动量 | 60日动量 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in observation.head(10).to_dict("records"):
        lines.append(
            f"| {row.get('symbol', '')} | {row.get('stock_name', '') or ''} | "
            f"{_pct(row.get('factor_pct_rank'))} | {_pct(row.get('momentum_20'))} | "
            f"{_pct(row.get('momentum_60'))} |"
        )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 状态只用于观察因子是否失效，不触发调仓、训练、候选晋级或模拟盘写入。",
            "- 前瞻收益按信号日后的下一开盘进入，达到对应持有期后的开盘退出。",
            "- 最近样本只使用已经成熟的信号日，尾部未实现收益不会被填充。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_factor_decay_monitor(
    panel: pd.DataFrame,
    *,
    asof_date: pd.Timestamp,
    output_dir: Path,
    selection_start: pd.Timestamp,
    selection_end: pd.Timestamp,
    recent_signal_days: int = 20,
    top_n: int = 40,
) -> dict[str, object]:
    panel = panel.loc[pd.to_datetime(panel["date"]).le(asof_date)].copy()
    factor_panel, metadata = prepare_factor_panel(panel)
    breadth = compute_healthy_breadth_state(factor_panel, ma_window=20)
    daily = compute_daily_factor_observations(factor_panel, breadth)
    summary = summarize_factor_decay(
        daily,
        selection_start=selection_start,
        selection_end=selection_end,
        asof_date=asof_date,
        recent_signal_days=recent_signal_days,
    )
    overall_status = classify_overall_status(summary)
    observation = build_current_observation(factor_panel, asof_date, top_n)
    breadth_row = breadth.loc[asof_date]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = asof_date.strftime("%Y%m%d")
    csv_path = output_dir / f"factor_decay_monitor_{token}.csv"
    json_path = output_dir / f"factor_decay_monitor_{token}.json"
    report_path = output_dir / f"factor_decay_monitor_{token}.md"
    observation_path = output_dir / f"liquidity_stability_observation_{token}.csv"
    history_path = output_dir / "factor_decay_monitor_history.csv"
    summary.to_csv(csv_path, index=False)
    observation.to_csv(observation_path, index=False)
    history = update_monitor_history(history_path, asof_date, summary, overall_status)
    report_path.write_text(
        build_report(summary, overall_status, asof_date, breadth_row, observation),
        encoding="utf-8",
    )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "asof_date": asof_date.date().isoformat(),
        "factor": FACTOR,
        "overall_status": overall_status,
        "research_only": True,
        "trade_instruction": False,
        "automatic_model_change": False,
        "selection_reference": {
            "start": selection_start.date().isoformat(),
            "end": selection_end.date().isoformat(),
        },
        "breadth": {
            "value": breadth_row.get("breadth_20"),
            "moving_average": breadth_row.get("breadth_20_ma"),
            "gap": breadth_row.get("breadth_gap"),
            "healthy": bool(breadth_row.get("breadth_healthy")),
        },
        "horizons": summary.to_dict("records"),
        "observation_count": int(len(observation)),
        "history_rows": int(len(history)),
        "factor_metadata": metadata,
        "artifacts": {
            "summary_csv": str(csv_path),
            "report": str(report_path),
            "observation": str(observation_path),
            "history": str(history_path),
        },
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return {**payload, "json": str(json_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor liquidity-stability factor decay without model changes."
    )
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--asof-date", required=True)
    parser.add_argument("--output-dir", default="outputs/high_return_v2")
    parser.add_argument("--selection-start", default="2023-01-01")
    parser.add_argument("--selection-end", default="2025-12-31")
    parser.add_argument("--recent-signal-days", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=40)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_factor_decay_monitor(
        load_market_data([Path(value) for value in args.data]),
        asof_date=pd.Timestamp(args.asof_date),
        output_dir=Path(args.output_dir),
        selection_start=pd.Timestamp(args.selection_start),
        selection_end=pd.Timestamp(args.selection_end),
        recent_signal_days=args.recent_signal_days,
        top_n=args.top_n,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
