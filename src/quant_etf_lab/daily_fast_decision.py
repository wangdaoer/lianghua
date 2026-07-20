"""Fast, read-only daily decision summary built from completed research artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .artifact_io import publish_json_if_semantically_changed, write_text_if_changed


@dataclass(frozen=True)
class DailyFastDecisionResult:
    output_dir: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]


def _resolve(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        return _clean(value.item())
    return value


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    selected = [column for column in columns if column in frame.columns]
    return [
        {key: _clean(value) for key, value in row.items()}
        for row in frame.loc[:, selected].to_dict(orient="records")
    ]


def _truthy(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _fingerprint(paths: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    for path in paths:
        stat = path.stat()
        entries.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    encoded = json.dumps(entries, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), entries


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return ["无。"]
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "") or "") for key, _ in columns) + " |")
    return lines


def _build_report(snapshot: dict[str, Any]) -> str:
    observation_rows = []
    for row in snapshot.get("observation_targets", []):
        observation_rows.append(
            {
                **row,
                "portfolio_target_weight": _fmt_pct(row.get("portfolio_target_weight")),
                "unrealized_return": _fmt_pct(row.get("unrealized_return")),
            }
        )
    candidate_rows = snapshot.get("momentum_candidates", [])
    blockers = snapshot.get("blocking_items") or []
    lines = [
        "# 每日快速决策摘要",
        "",
        f"- 结果状态：`{snapshot.get('status')}`",
        f"- 数据日期：`{snapshot.get('as_of_date')}`",
        f"- 缓存命中：`{snapshot.get('cache_hit')}`",
        f"- 市场状态：`{snapshot.get('sentiment_state')}` / `{snapshot.get('latest_regime')}`",
        f"- 决策姿态：`{snapshot.get('dashboard_posture')}`",
        f"- 实盘前检：`{snapshot.get('preflight_decision')}`",
        f"- 阻断项：{'；'.join(str(item) for item in blockers) if blockers else '无'}",
        f"- 核心 / 卫星 / 现金：{_fmt_pct(snapshot.get('core_weight'))} / "
        f"{_fmt_pct(snapshot.get('satellite_weight'))} / {_fmt_pct(snapshot.get('cash_weight'))}",
        f"- 当前回撤：{_fmt_pct(snapshot.get('current_drawdown'))}",
        f"- 观察标的：{snapshot.get('observation_target_count', 0)} 只，合计目标权重 "
        f"{_fmt_pct(snapshot.get('observation_total_weight'))}",
        f"- 复核硬排除：{snapshot.get('review_excluded_count', 0)} 只",
        "",
        "## 当前观察清单",
        "",
        *_markdown_table(
            observation_rows,
            [
                ("code", "代码"),
                ("name", "名称"),
                ("layer", "层级"),
                ("portfolio_target_weight", "目标权重"),
                ("unrealized_return", "未实现收益"),
                ("target_action", "调整动作"),
            ],
        ),
        "",
        "## 强势候选",
        "",
        *_markdown_table(
            candidate_rows,
            [
                ("code", "代码"),
                ("name", "名称"),
                ("change_pct", "涨跌幅%"),
                ("focus_score", "关注分"),
                ("research_priority", "优先级"),
            ],
        ),
        "",
        "## 运行耗时",
        "",
    ]
    for stage, elapsed in snapshot.get("stage_timings_ms", {}).items():
        lines.append(f"- {stage}: `{elapsed:.3f} ms`")
    lines.extend(["", "> 本报告仅用于研究与人工复核，不连接券商、不自动下单。", ""])
    return "\n".join(lines)


def run_daily_fast_decision(
    project_root: Path = Path("."),
    output_dir: Path = Path("outputs/research/daily_fast_decision_latest"),
    research_dir: Path = Path("outputs/research"),
    top_candidates: int = 20,
    force: bool = False,
) -> DailyFastDecisionResult:
    """Build a cached decision summary without training, backtesting, or broker actions."""

    started = time.perf_counter()
    project_root = project_root.resolve()
    research_root = _resolve(project_root, research_dir)
    resolved_output = _resolve(project_root, output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    snapshot_path = resolved_output / "fast_decision_snapshot.json"
    report_path = resolved_output / "fast_decision.md"

    source_paths = {
        "pipeline": research_root / "daily_pipeline_latest" / "daily_pipeline_snapshot.json",
        "dashboard": research_root / "latest_dashboard" / "latest_dashboard_snapshot.json",
        "paper_metrics": research_root / "paper_account_latest" / "metrics.json",
        "stock_targets": research_root / "paper_account_latest" / "stock_targets.csv",
        "stock_review": research_root / "paper_account_latest" / "stock_target_review.csv",
        "momentum_snapshot": research_root / "momentum_focus_latest" / "momentum_focus_snapshot.json",
        "momentum_candidates": research_root / "momentum_focus_latest" / "momentum_focus_candidates.csv",
        "preflight": research_root / "live_preflight_latest" / "live_preflight_snapshot.json",
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Fast decision inputs are missing: " + ", ".join(missing))

    stage_timings: dict[str, float] = {}
    tick = time.perf_counter()
    compact_json = {
        key: _read_json(path)
        for key, path in source_paths.items()
        if path.suffix.lower() == ".json"
    }
    market_source_text = compact_json["momentum_snapshot"].get("source_path")
    market_source = Path(market_source_text).resolve() if market_source_text else None
    fingerprint_paths = list(source_paths.values())
    if market_source is not None and market_source.exists():
        fingerprint_paths.append(market_source)
    fingerprint, fingerprint_entries = _fingerprint(fingerprint_paths)
    stage_timings["输入定位与指纹"] = (time.perf_counter() - tick) * 1000

    if not force and snapshot_path.exists():
        cached = _read_json(snapshot_path)
        if cached.get("input_fingerprint") == fingerprint and cached.get("top_candidates") == top_candidates:
            cached["cache_hit"] = True
            cached["generated_at"] = datetime.now().isoformat(timespec="seconds")
            cached["stage_timings_ms"] = {
                **stage_timings,
                "总耗时": (time.perf_counter() - started) * 1000,
            }
            return DailyFastDecisionResult(resolved_output, snapshot_path, report_path, cached)

    tick = time.perf_counter()
    stock_targets = pd.read_csv(source_paths["stock_targets"], dtype={"code": str})
    stock_review = pd.read_csv(source_paths["stock_review"], dtype={"code": str})
    momentum = pd.read_csv(source_paths["momentum_candidates"], dtype={"code": str})
    for frame in (stock_targets, stock_review, momentum):
        if "code" in frame:
            frame["code"] = frame["code"].astype(str).str.zfill(6)
    stage_timings["读取决策表"] = (time.perf_counter() - tick) * 1000

    tick = time.perf_counter()
    excluded_mask = (
        _truthy(stock_review["observation_excluded"])
        if "observation_excluded" in stock_review
        else pd.Series(False, index=stock_review.index)
    )
    excluded_codes = set(stock_review.loc[excluded_mask, "code"].dropna())
    observation = stock_targets.loc[~stock_targets["code"].isin(excluded_codes)].copy()
    if "tracking_excluded" in observation:
        observation = observation.loc[~_truthy(observation["tracking_excluded"])]
    momentum = momentum.loc[~momentum["code"].isin(excluded_codes)].copy()
    if "focus_score" in momentum:
        momentum = momentum.sort_values("focus_score", ascending=False, na_position="last")
    momentum = momentum.head(max(top_candidates, 0))
    stage_timings["硬排除与候选筛选"] = (time.perf_counter() - tick) * 1000

    pipeline = compact_json["pipeline"]
    dashboard = compact_json["dashboard"]
    metrics = compact_json["paper_metrics"]
    momentum_snapshot = compact_json["momentum_snapshot"]
    preflight = compact_json["preflight"]
    dates = {
        "pipeline": pipeline.get("as_of_date"),
        "dashboard": dashboard.get("as_of_date"),
        "paper": metrics.get("latest_date"),
        "momentum": momentum_snapshot.get("trade_date") or momentum_snapshot.get("as_of_date"),
        "preflight": preflight.get("as_of_date"),
    }
    known_dates = {str(value)[:10] for value in dates.values() if value}
    dates_aligned = len(known_dates) == 1
    as_of_date = max(known_dates) if known_dates else None
    blocking_items = list(preflight.get("blocking_items") or [])
    if not dates_aligned:
        blocking_items.insert(0, "fast_decision_input_dates_not_aligned")
    status = "ready" if dates_aligned else "stale_inputs"
    if preflight.get("decision") == "blocked":
        status = "blocked" if dates_aligned else status

    tick = time.perf_counter()
    snapshot: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "research_only": True,
        "broker_action": "none",
        "cache_hit": False,
        "input_fingerprint": fingerprint,
        "input_fingerprint_entries": fingerprint_entries,
        "input_dates": dates,
        "input_dates_aligned": dates_aligned,
        "as_of_date": as_of_date,
        "data_freshness_status": dashboard.get("data_freshness_status") or pipeline.get("data_freshness_status"),
        "market_source_kind": momentum_snapshot.get("source_kind") or dashboard.get("market_cache_source_kind"),
        "market_source_path": market_source_text,
        "sentiment_state": dashboard.get("sentiment_state") or pipeline.get("sentiment_state"),
        "dashboard_posture": dashboard.get("dashboard_posture"),
        "latest_regime": metrics.get("latest_regime"),
        "core_weight": metrics.get("latest_core_weight"),
        "satellite_weight": metrics.get("latest_satellite_weight"),
        "cash_weight": metrics.get("latest_cash_weight"),
        "current_drawdown": metrics.get("current_drawdown"),
        "paper_total_return": metrics.get("total_return"),
        "paper_cagr": metrics.get("cagr"),
        "paper_sharpe": metrics.get("sharpe"),
        "preflight_decision": preflight.get("decision"),
        "blocking_items": blocking_items,
        "monitor_items": preflight.get("monitor_items") or [],
        "review_excluded_count": len(excluded_codes),
        "review_excluded_codes": sorted(excluded_codes),
        "observation_target_count": len(observation),
        "observation_total_weight": float(
            pd.to_numeric(observation.get("portfolio_target_weight", pd.Series(dtype=float)), errors="coerce").sum()
        ),
        "observation_targets": _records(
            observation,
            [
                "date", "layer", "code", "name", "portfolio_target_weight", "unrealized_return",
                "target_action", "risk_filter_status", "execution_gate_action",
            ],
        ),
        "momentum_candidate_source_count": int(momentum_snapshot.get("candidate_count") or 0),
        "momentum_candidate_count": len(momentum),
        "momentum_candidates": _records(
            momentum,
            [
                "as_of_date", "code", "name", "board", "signal_type", "change_pct", "close_price",
                "turnover_yi", "focus_score", "research_priority", "broker_action",
            ],
        ),
        "top_candidates": top_candidates,
        "source_paths": {key: str(path) for key, path in source_paths.items()},
    }
    stage_timings["组装决策摘要"] = (time.perf_counter() - tick) * 1000
    stage_timings["总耗时"] = (time.perf_counter() - started) * 1000
    snapshot["stage_timings_ms"] = stage_timings
    snapshot, snapshot_changed = publish_json_if_semantically_changed(
        snapshot_path,
        snapshot,
        ignored_fields=("generated_at", "cache_hit", "stage_timings_ms"),
    )
    if snapshot_changed or not report_path.exists():
        write_text_if_changed(report_path, _build_report(snapshot), encoding="utf-8")
    return DailyFastDecisionResult(resolved_output, snapshot_path, report_path, snapshot)
