"""Build a daily Chinese report for the personal trade-habit overlay."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shadow_account_signals import (
    apply_shadow_account_signals,
    load_shadow_account_review,
    shadow_account_summary_lines,
)


ACTION_CN = {"allow": "可保留", "reduce": "降仓", "watch_only": "只观察"}
CODE_COLUMNS = [
    "symbol",
    "security_code",
    "股票代码",
    "证券代码",
    "代码",
    "code",
    "raw_security_code",
    "prefixed_security_code",
]
NAME_COLUMNS = ["stock_name", "name", "security_name", "股票名称", "证券简称", "简称", "名称"]


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _text(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=object)
    return frame[column].fillna(default).astype(str)


def _symbols(frame: pd.DataFrame) -> pd.Series:
    if "symbol" not in frame:
        return pd.Series("", index=frame.index, dtype=object)
    return _normalize_symbol_series(frame["symbol"])


def _normalize_symbol_series(values: pd.Series) -> pd.Series:
    text = values.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    extracted = text.str.extract(r"(\d{6})", expand=False)
    digits = text.str.replace(r"\D", "", regex=True)
    normalized = extracted.fillna(digits).str[-6:].str.zfill(6)
    return normalized.mask(normalized.eq("000000"), "")


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    exact = {str(column): column for column in frame.columns}
    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _read_text_name_source(path: Path, prefer_tab: bool = False) -> pd.DataFrame:
    """Read CSV-like name sources, including THS tab text exported with .xls suffix."""

    separators = ["\t", ","] if prefer_tab else [",", "\t"]
    last_error: Exception | None = None
    for encoding in ["utf-8-sig", "utf-8", "gb18030"]:
        for separator in separators:
            try:
                frame = pd.read_csv(path, dtype=str, encoding=encoding, sep=separator)
            except Exception as exc:  # pragma: no cover - only used for messy external files.
                last_error = exc
                continue
            if len(frame.columns) > 1:
                return frame
    raise last_error or ValueError(f"Cannot read name source: {path}")


def load_name_map(source: pd.DataFrame | str | Path | None) -> dict[str, str]:
    """Load stock code -> Chinese name mapping from a DataFrame or local file."""

    if source is None:
        return {}
    if isinstance(source, pd.DataFrame):
        frame = source.copy()
    else:
        path = Path(source)
        if not path.exists():
            return {}
        if path.suffix.lower() in {".xls", ".xlsx"}:
            try:
                frame = pd.read_excel(path, dtype=str)
            except Exception:
                frame = _read_text_name_source(path, prefer_tab=True)
        else:
            frame = _read_text_name_source(path)

    code_column = _first_existing_column(frame, CODE_COLUMNS)
    name_column = _first_existing_column(frame, NAME_COLUMNS)
    if code_column is None or name_column is None:
        return {}

    out = pd.DataFrame(
        {
            "symbol": _normalize_symbol_series(frame[code_column]),
            "stock_name": frame[name_column].fillna("").astype(str).str.strip(),
        }
    )
    out = out[out["symbol"].ne("") & out["stock_name"].ne("")]
    out = out.drop_duplicates("symbol", keep="last")
    return dict(zip(out["symbol"], out["stock_name"], strict=False))


def _name_series(frame: pd.DataFrame, name_map: dict[str, str] | None = None) -> pd.Series:
    name_column = _first_existing_column(frame, NAME_COLUMNS)
    if name_column is not None:
        existing = frame[name_column].fillna("").astype(str).str.strip()
    else:
        existing = pd.Series("", index=frame.index, dtype=object)
    if name_map:
        mapped = _symbols(frame).map(name_map).fillna("")
        existing = existing.mask(existing.eq(""), mapped)
    return existing


def _format_symbol_names(symbols: list[str], name_map: dict[str, str] | None = None, limit: int | None = None) -> str:
    items = symbols[:limit] if limit is not None else symbols
    formatted: list[str] = []
    for symbol in items:
        name = (name_map or {}).get(symbol, "")
        formatted.append(f"{symbol}（{name}）" if name else symbol)
    if not formatted:
        return "无"
    suffix = ""
    if limit is not None and len(symbols) > limit:
        suffix = f" 等 {len(symbols)} 只"
    return "、".join(formatted) + suffix


def _bool_column(frame: pd.DataFrame, column: str, fallback_weight_column: str | None = None) -> pd.Series:
    if column in frame:
        values = frame[column]
        if pd.api.types.is_bool_dtype(values):
            return values.fillna(False).astype(bool)
        normalized = values.fillna(False).astype(str).str.strip().str.lower()
        return normalized.isin({"1", "true", "t", "yes", "y", "是", "入选"})
    if fallback_weight_column:
        return _numeric(frame, fallback_weight_column, 0.0).gt(0.0)
    return pd.Series(False, index=frame.index, dtype=bool)


def _selected_symbol_set(frame: pd.DataFrame, selected_column: str, weight_column: str) -> set[str]:
    if frame.empty:
        return set()
    selected = _bool_column(frame, selected_column, weight_column)
    return set(_symbols(frame.loc[selected]))


def _overlay_selected(overlay: pd.DataFrame) -> pd.Series:
    return _bool_column(overlay, "personal_selected", "personal_adjusted_target_weight")


def _before_weight(overlay: pd.DataFrame) -> pd.Series:
    if "target_weight_before_behavior" in overlay:
        return _numeric(overlay, "target_weight_before_behavior", 0.0)
    return _numeric(overlay, "target_weight", 0.0)


def _after_weight(overlay: pd.DataFrame) -> pd.Series:
    if "personal_adjusted_target_weight" in overlay:
        return _numeric(overlay, "personal_adjusted_target_weight", 0.0)
    if "target_weight_after_behavior" in overlay:
        return _numeric(overlay, "target_weight_after_behavior", 0.0)
    return _numeric(overlay, "target_weight", 0.0)


def summarize_overlay_changes(base: pd.DataFrame, overlay: pd.DataFrame) -> dict[str, Any]:
    """Summarize how the personal overlay changed the base candidate list."""

    base = base.copy()
    overlay = overlay.copy()
    if not base.empty:
        base["symbol"] = _symbols(base)
    if not overlay.empty:
        overlay["symbol"] = _symbols(overlay)

    base_selected = _selected_symbol_set(base, "selected", "target_weight")
    overlay_selected = _selected_symbol_set(
        overlay,
        "personal_selected",
        "personal_adjusted_target_weight",
    )
    before = _before_weight(overlay)
    after = _after_weight(overlay)
    selected = _overlay_selected(overlay)
    reduced_selected_symbols = set(
        overlay.loc[selected & before.gt(0.0) & after.lt(before - 1e-12), "symbol"]
    )
    action_counts = (
        _text(overlay, "personal_action", "")
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .to_dict()
    )

    return {
        "base_selected_count": len(base_selected),
        "overlay_selected_count": len(overlay_selected),
        "kept_count": len(base_selected & overlay_selected),
        "added_count": len(overlay_selected - base_selected),
        "removed_count": len(base_selected - overlay_selected),
        "reduced_selected_count": len(reduced_selected_symbols),
        "added_symbols": sorted(overlay_selected - base_selected),
        "removed_symbols": sorted(base_selected - overlay_selected),
        "reduced_selected_symbols": sorted(reduced_selected_symbols),
        "base_gross_weight": float(_numeric(base.loc[_bool_column(base, "selected", "target_weight")], "target_weight", 0.0).sum()),
        "overlay_gross_weight": float(after[selected].sum()),
        "action_counts": action_counts,
    }


def build_selected_view(overlay: pd.DataFrame, name_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Return selected rows with Chinese headers for daily review."""

    frame = overlay.copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "股票代码",
                "股票名称",
                "原模型排名",
                "个人层排名",
                "原模型权重",
                "个人习惯层权重",
                "权重变化",
                "动作",
                "调整原因",
                "趋势状态",
                "收盘价",
                "20日收益",
                "收盘位置",
                "模型分数",
                "个人调整后分数",
            ]
        )
    frame["symbol"] = _symbols(frame)
    selected = frame.loc[_overlay_selected(frame)].copy()
    if "personal_rank" in selected:
        selected = selected.sort_values("personal_rank")
    elif "rank" in selected:
        selected = selected.sort_values("rank")

    before = _before_weight(selected)
    after = _after_weight(selected)
    action = _text(selected, "personal_action_cn", "")
    if action.eq("").any() and "personal_action" in selected:
        action = action.mask(action.eq(""), _text(selected, "personal_action", "").map(ACTION_CN).fillna(""))

    view = pd.DataFrame(
        {
            "股票代码": _symbols(selected).to_numpy(),
            "股票名称": _name_series(selected, name_map).to_numpy(),
            "原模型排名": _numeric(selected, "rank", np.nan).to_numpy(),
            "个人层排名": _numeric(selected, "personal_rank", np.nan).to_numpy(),
            "原模型权重": before.to_numpy(),
            "个人习惯层权重": after.to_numpy(),
            "权重变化": (after - before).to_numpy(),
            "动作": action.to_numpy(),
            "调整原因": _text(selected, "personal_reasons_cn", "").to_numpy(),
            "趋势状态": _text(selected, "trend_state", "").to_numpy(),
            "收盘价": _numeric(selected, "close", np.nan).to_numpy(),
            "20日收益": _numeric(selected, "return_20d", np.nan).to_numpy(),
            "收盘位置": _numeric(selected, "close_position", np.nan).to_numpy(),
            "模型分数": _numeric(selected, "score", np.nan).to_numpy(),
            "个人调整后分数": _numeric(selected, "personal_adjusted_score", np.nan).to_numpy(),
        }
    )
    if "shadow_account_signal_cn" in selected or "shadow_account_notes" in selected:
        view["影子账户提示"] = _text(
            selected, "shadow_account_signal_cn", "无额外提示"
        ).to_numpy()
        view["影子账户说明"] = _text(
            selected, "shadow_account_notes", ""
        ).to_numpy()
    return view.reset_index(drop=True)


def build_change_view(
    base: pd.DataFrame,
    overlay: pd.DataFrame,
    name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return added, removed and reduced rows with Chinese headers."""

    summary = summarize_overlay_changes(base, overlay)
    frame = overlay.copy()
    if frame.empty:
        return pd.DataFrame()
    frame["symbol"] = _symbols(frame)
    before = _before_weight(frame)
    after = _after_weight(frame)
    frame["_before_weight"] = before
    frame["_after_weight"] = after
    frame["_change_type"] = ""
    frame.loc[frame["symbol"].isin(summary["added_symbols"]), "_change_type"] = "补入"
    frame.loc[frame["symbol"].isin(summary["removed_symbols"]), "_change_type"] = "移出观察"
    reduced_mask = frame["symbol"].isin(summary["reduced_selected_symbols"]) & frame["_change_type"].eq("")
    frame.loc[reduced_mask, "_change_type"] = "降仓"
    changed = frame.loc[frame["_change_type"].ne("")].copy()
    if changed.empty:
        return pd.DataFrame(
            columns=[
                "调整类型",
                "股票代码",
                "股票名称",
                "原模型排名",
                "个人层排名",
                "原模型权重",
                "个人习惯层权重",
                "权重变化",
                "动作",
                "调整原因",
                "趋势状态",
                "20日收益",
                "收盘位置",
            ]
        )
    order = {"移出观察": 0, "降仓": 1, "补入": 2}
    changed["_order"] = changed["_change_type"].map(order).fillna(9)
    changed = changed.sort_values(["_order", "rank", "personal_rank"], na_position="last")
    action = _text(changed, "personal_action_cn", "")
    if action.eq("").any() and "personal_action" in changed:
        action = action.mask(action.eq(""), _text(changed, "personal_action", "").map(ACTION_CN).fillna(""))
    return pd.DataFrame(
        {
            "调整类型": changed["_change_type"].to_numpy(),
            "股票代码": _symbols(changed).to_numpy(),
            "股票名称": _name_series(changed, name_map).to_numpy(),
            "原模型排名": _numeric(changed, "rank", np.nan).to_numpy(),
            "个人层排名": _numeric(changed, "personal_rank", np.nan).to_numpy(),
            "原模型权重": changed["_before_weight"].to_numpy(),
            "个人习惯层权重": changed["_after_weight"].to_numpy(),
            "权重变化": (changed["_after_weight"] - changed["_before_weight"]).to_numpy(),
            "动作": action.to_numpy(),
            "调整原因": _text(changed, "personal_reasons_cn", "").to_numpy(),
            "趋势状态": _text(changed, "trend_state", "").to_numpy(),
            "20日收益": _numeric(changed, "return_20d", np.nan).to_numpy(),
            "收盘位置": _numeric(changed, "close_position", np.nan).to_numpy(),
        }
    ).reset_index(drop=True)


def build_watchlist_view(
    watchlist: pd.DataFrame,
    name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    if watchlist is None or watchlist.empty:
        return pd.DataFrame(
            columns=[
                "股票代码",
                "股票名称",
                "观察模式",
                "观察评分",
                "观察理由",
                "收盘价",
                "20日收益",
                "60日收益",
                "15日振幅",
                "20日高点回撤",
                "成交额放大倍数",
            ]
        )
    frame = watchlist.copy()
    if "symbol" not in frame and "股票代码" in frame:
        frame["symbol"] = frame["股票代码"]
    frame["symbol"] = _symbols(frame)
    if "pattern_score" in frame:
        frame = frame.sort_values("pattern_score", ascending=False)
    return pd.DataFrame(
        {
            "股票代码": _symbols(frame).to_numpy(),
            "股票名称": _name_series(frame, name_map).to_numpy(),
            "观察模式": _text(frame, "pattern_type", "").to_numpy(),
            "观察评分": _numeric(frame, "pattern_score", np.nan).to_numpy(),
            "观察理由": _text(frame, "pattern_reason", "").to_numpy(),
            "收盘价": _numeric(frame, "close", np.nan).to_numpy(),
            "20日收益": _numeric(frame, "return_20d", np.nan).to_numpy(),
            "60日收益": _numeric(frame, "return_60d", np.nan).to_numpy(),
            "15日振幅": _numeric(frame, "range_15d", np.nan).to_numpy(),
            "20日高点回撤": _numeric(frame, "drawdown_20d", np.nan).to_numpy(),
            "成交额放大倍数": _numeric(frame, "amount_ratio", np.nan).to_numpy(),
        }
    ).reset_index(drop=True)


def _format_percent(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.2%}"


def _format_number(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(number):
        return ""
    return f"{number:.{digits}f}"


def _markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "无"
    display = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in ["原模型权重", "个人习惯层权重", "权重变化", "20日收益", "60日收益", "收盘位置", "15日振幅", "20日高点回撤"]:
        if column in display:
            display[column] = display[column].map(_format_percent)
    for column in ["收盘价", "模型分数", "个人调整后分数", "观察评分", "成交额放大倍数"]:
        if column in display:
            display[column] = display[column].map(_format_number)
    for column in ["原模型排名", "个人层排名"]:
        if column in display:
            display[column] = display[column].map(lambda x: "" if pd.isna(x) else str(int(float(x))))
    try:
        return display.to_markdown(index=False, disable_numparse=True)
    except TypeError:
        return display.to_markdown(index=False)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rules(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return json.loads(path.read_text(encoding="utf-8"))


def _metrics_lines(metrics: dict[str, Any]) -> list[str]:
    if not metrics:
        return ["- 未提供回测指标文件，本报告只汇总今日候选调整。"]
    base = metrics.get("base", {})
    overlay = metrics.get("personal_overlay", {})
    inc = metrics.get("incremental", {})
    return [
        f"- 原模型总收益: {_format_percent(base.get('total_return'))}，最大回撤: {_format_percent(base.get('max_drawdown'))}，Sharpe-like: {_format_number(base.get('sharpe_like'), 3)}",
        f"- 个人习惯层总收益: {_format_percent(overlay.get('total_return'))}，最大回撤: {_format_percent(overlay.get('max_drawdown'))}，Sharpe-like: {_format_number(overlay.get('sharpe_like'), 3)}",
        f"- 增量收益: {_format_percent(inc.get('total_return_diff'))}，回撤改善: {_format_percent(inc.get('max_drawdown_diff'))}，Sharpe-like 改善: {_format_number(inc.get('sharpe_like_diff'), 3)}",
        f"- 平均仓位: 原模型 {_format_percent(metrics.get('avg_base_gross_exposure'))}，个人习惯层 {_format_percent(metrics.get('avg_overlay_gross_exposure'))}",
    ]


def build_risk_warnings(summary: dict[str, Any], metrics: dict[str, Any] | None = None) -> list[str]:
    """Build reader-facing daily risk warnings from overlay changes and backtest metrics."""

    metrics = metrics or {}
    warnings: list[str] = []
    gross = float(summary.get("overlay_gross_weight") or 0.0)
    removed = int(summary.get("removed_count") or 0)
    reduced = int(summary.get("reduced_selected_count") or 0)
    watch_only = int((summary.get("action_counts") or {}).get("watch_only", 0))
    max_drawdown = (metrics.get("personal_overlay") or {}).get("max_drawdown")

    if gross >= 0.70:
        warnings.append(f"当前目标仓位约 {_format_percent(gross)}，属于偏积极状态，次日若大幅低开需要优先控制单日风险。")
    elif gross <= 0.40:
        warnings.append(f"当前目标仓位约 {_format_percent(gross)}，模型处在偏防守状态，注意不要为了追收益手动加满。")
    else:
        warnings.append(f"当前目标仓位约 {_format_percent(gross)}，仓位处在中等区间，按候选权重执行即可。")

    if removed or watch_only:
        warnings.append(f"今日有 {removed} 只原入选被移出观察，全候选中 {watch_only} 只标记为只观察，说明局部趋势损伤仍需要回避。")
    if reduced:
        warnings.append(f"今日入选标的中有 {reduced} 只被降仓，主要用于减少弱 20 日动量和追高后的回撤暴露。")
    try:
        drawdown_value = float(max_drawdown)
    except (TypeError, ValueError):
        drawdown_value = np.nan
    if np.isfinite(drawdown_value) and drawdown_value <= -0.20:
        warnings.append(f"个人习惯层历史最大回撤仍约 {_format_percent(drawdown_value)}，策略虽改善回撤，但仍不是低波动策略。")
    return warnings


def _rules_lines(rules: dict[str, Any]) -> list[str]:
    if not rules:
        return ["- 未提供规则配置文件。"]
    return [
        f"- 信号日收盘位置 <= {_format_percent(rules.get('low_entry_position_max'))}: 加分，保留低位/中低位介入优势。",
        f"- 信号日收盘位置 >= {_format_percent(rules.get('high_entry_position_min'))}: 降分并降仓，减少追高。",
        f"- 20日收益在 {_format_percent(rules.get('weak_20d_return_min'))} 到 {_format_percent(rules.get('weak_20d_return_max'))}: 降仓到 {_format_percent(rules.get('weak_20d_weight_multiplier'))}。",
        f"- 20日收益 <= {_format_percent(rules.get('damaged_20d_return_max'))}: 只观察，不新开仓。",
        f"- 默认观察持有下限: {rules.get('min_default_holding_days', '')} 个交易日；避免 {rules.get('avoid_discretionary_sell_before', '')} 前做主观卖出决定。",
    ]


def write_report(
    output: Path,
    base: pd.DataFrame,
    overlay: pd.DataFrame,
    selected_view: pd.DataFrame,
    change_view: pd.DataFrame,
    metrics: dict[str, Any] | None = None,
    rules: dict[str, Any] | None = None,
    asof_date: str | None = None,
    name_map: dict[str, str] | None = None,
    watchlist_view: pd.DataFrame | None = None,
    shadow_account_review: dict[str, Any] | None = None,
) -> Path:
    summary = summarize_overlay_changes(base, overlay)
    date_label = asof_date or "latest"
    action_counts_cn = {
        ACTION_CN.get(action, action): count
        for action, count in summary["action_counts"].items()
    }
    added = _format_symbol_names(summary["added_symbols"], name_map)
    removed = _format_symbol_names(summary["removed_symbols"], name_map)
    reduced = _format_symbol_names(summary["reduced_selected_symbols"], name_map, limit=20)

    lines = [
        f"# 每日个人交易习惯层报告（{date_label}）",
        "",
        "## 今日结论",
        f"- 原模型入选 {summary['base_selected_count']} 只，个人习惯层后入选 {summary['overlay_selected_count']} 只。",
        f"- 总目标仓位从 {_format_percent(summary['base_gross_weight'])} 调整到 {_format_percent(summary['overlay_gross_weight'])}。",
        f"- 保留原入选 {summary['kept_count']} 只，新增候补 {summary['added_count']} 只，移出观察 {summary['removed_count']} 只，入选标的中降仓 {summary['reduced_selected_count']} 只。",
        f"- 动作分布: {json.dumps(action_counts_cn, ensure_ascii=False)}",
        f"- 新增候补: {added}",
        f"- 移出观察: {removed}",
        f"- 降仓名单: {reduced}",
        "",
        "## 风险提示",
        *[f"- {line}" for line in build_risk_warnings(summary, metrics or {})],
        "",
        "## 影子账户提示",
        *[f"- {line}" for line in shadow_account_summary_lines(shadow_account_review)],
        "",
        "## 提前观察形态",
        "这部分只用于提前观察分析，不直接触发买入或加仓。",
        _markdown_table(watchlist_view if watchlist_view is not None else pd.DataFrame(), max_rows=30),
        "",
        "## 已验证的收益/回撤变化",
        *_metrics_lines(metrics or {}),
        "",
        "## 当前执行规则",
        *_rules_lines(rules or {}),
        "",
        "## 调整明细",
        _markdown_table(change_view),
        "",
        "## 最终入选清单",
        _markdown_table(selected_view),
        "",
        "## 备注",
        "- 这是研究用每日决策辅助输出，不连接券商，不自动下单。",
        "- 个人习惯层只做候选再排序、补位、降仓和观察过滤；主模型训练逻辑保持不变。",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _infer_asof_date(path: Path) -> str:
    matches = re.findall(r"(20\d{6})", path.stem)
    if not matches:
        return "latest"
    raw = matches[-1]
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def _check_latest_fetch_status(utils_path: Path) -> str:
    if not utils_path.exists():
        return "未找到行情状态检查脚本，跳过状态检查。"
    try:
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("market_data_utils", utils_path)
        if spec is None or spec.loader is None:
            return "行情状态检查脚本无法加载，跳过状态检查。"
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        status = module.get_latest_fetch_status(project_root=str(utils_path.parents[1]))
        return f"行情状态检查: status={getattr(status, 'status', '')}, rows={getattr(status, 'row_count', '')}, message={getattr(status, 'message', '')}"
    except Exception as exc:
        return f"行情状态检查未完成: {exc}"


def _find_default_name_source(asof_date: str, daily_data_dir: Path, fallback_data_dir: Path) -> Path | None:
    date = asof_date if asof_date != "latest" else ""
    token = date.replace("-", "")
    candidates: list[Path] = []
    if date:
        candidates.extend(
            [
                daily_data_dir / "ths_exports" / "normalized" / f"ths_hs_a_share_{date}.csv",
                daily_data_dir / "snapshots" / f"{date}_market_snapshot.csv",
                fallback_data_dir / "ths_exports" / "normalized" / f"ths_hs_a_share_{date}.csv",
                fallback_data_dir / "snapshots" / f"{date}_market_snapshot.csv",
            ]
        )
    candidates.extend(
        [
            daily_data_dir / "exports" / "latest_model2_input_with_sector.json",
            daily_data_dir / "exports" / "latest_model2_input.json",
        ]
    )
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() in {".csv", ".xls", ".xlsx"}:
            return candidate
    if token:
        files = sorted(
            daily_data_dir.glob(f"**/*{date}*.csv"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if files:
            return files[0]
    return None


def _latest_file(pattern: str, output_dir: Path) -> Path:
    files = sorted(output_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files match {output_dir / pattern}")
    return files[0]


def _default_base_for_overlay(overlay_path: Path, output_dir: Path) -> Path:
    matches = re.findall(r"(20\d{6})", overlay_path.stem)
    if matches:
        date = matches[-1]
        candidates = [
            output_dir / f"rank_model_candidates_trend_gated_bench{date}_{date}.csv",
            output_dir / f"rank_model_candidates_trend_gated_{date}.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    files = [
        item
        for item in output_dir.glob("rank_model_candidates_trend_gated_*.csv")
        if "personal_overlay" not in item.stem and not item.stem.endswith("_cn")
    ]
    files = sorted(files, key=lambda item: item.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No base candidate files found in {output_dir}")
    return files[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily Chinese report for personal overlay candidates.")
    parser.add_argument("--base-candidates", default=None)
    parser.add_argument("--overlay-candidates", default=None)
    parser.add_argument("--metrics", default="outputs/high_return_v2/personal_behavior_overlay_best_20220101_20260629/metrics.json")
    parser.add_argument("--rules", default="configs/personal_trade_habit_overlay.yaml")
    parser.add_argument("--output-dir", default="outputs/high_return_v2")
    parser.add_argument("--asof-date", default=None)
    parser.add_argument("--names-source", default=None)
    parser.add_argument("--early-watchlist", default=None)
    parser.add_argument("--shadow-account-review", default=None)
    parser.add_argument("--daily-data-dir", default="D:/codex/daily-market-data")
    parser.add_argument("--fallback-data-dir", default="D:/codex/2026-06-15-exchange-data-ingest")
    parser.add_argument(
        "--fetch-status-utils",
        default="D:/codex/2026-06-15-exchange-data-ingest/scripts/market_data_utils.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    overlay_path = (
        Path(args.overlay_candidates)
        if args.overlay_candidates
        else _latest_file("rank_model_candidates_trend_gated_personal_overlay_*.csv", output_dir)
    )
    base_path = Path(args.base_candidates) if args.base_candidates else _default_base_for_overlay(overlay_path, output_dir)
    asof = args.asof_date or _infer_asof_date(overlay_path)

    base = pd.read_csv(base_path, dtype={"symbol": str})
    overlay = pd.read_csv(overlay_path, dtype={"symbol": str})
    shadow_account_review = load_shadow_account_review(args.shadow_account_review)
    overlay = apply_shadow_account_signals(overlay, shadow_account_review)
    print(_check_latest_fetch_status(Path(args.fetch_status_utils)))
    names_source = (
        Path(args.names_source)
        if args.names_source
        else _find_default_name_source(asof, Path(args.daily_data_dir), Path(args.fallback_data_dir))
    )
    name_map = load_name_map(names_source) if names_source else {}
    selected_view = build_selected_view(overlay, name_map=name_map)
    change_view = build_change_view(base, overlay, name_map=name_map)
    watchlist = (
        pd.read_csv(args.early_watchlist, dtype={"symbol": str})
        if args.early_watchlist and Path(args.early_watchlist).exists()
        else pd.DataFrame()
    )
    watchlist_view = build_watchlist_view(watchlist, name_map=name_map)
    metrics = _load_json(Path(args.metrics) if args.metrics else None)
    rules = _load_rules(Path(args.rules) if args.rules else None)

    date_token = asof.replace("-", "") if asof != "latest" else "latest"
    selected_path = output_dir / f"daily_personal_overlay_selected_{date_token}.csv"
    changes_path = output_dir / f"daily_personal_overlay_changes_{date_token}.csv"
    report_path = output_dir / f"daily_personal_overlay_report_{date_token}.md"
    selected_view.to_csv(selected_path, index=False, encoding="utf-8-sig")
    change_view.to_csv(changes_path, index=False, encoding="utf-8-sig")
    write_report(
        report_path,
        base,
        overlay,
        selected_view,
        change_view,
        metrics,
        rules,
        asof,
        name_map,
        watchlist_view,
        shadow_account_review,
    )

    summary = summarize_overlay_changes(base, overlay)
    print(f"Report: {report_path}")
    print(f"Selected CSV: {selected_path}")
    print(f"Changes CSV: {changes_path}")
    print(f"Names source: {names_source if names_source else 'none'} names={len(name_map)}")
    print(
        "Summary: "
        f"base={summary['base_selected_count']} overlay={summary['overlay_selected_count']} "
        f"kept={summary['kept_count']} added={summary['added_count']} "
        f"removed={summary['removed_count']} reduced={summary['reduced_selected_count']}"
    )


if __name__ == "__main__":
    main()
