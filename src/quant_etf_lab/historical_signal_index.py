"""Build a unified historical model-signal index from saved research outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


INDEX_COLUMNS = [
    "signal_date",
    "symbol",
    "name",
    "signal_source",
    "signal_rank",
    "priority_score",
    "score_bucket",
    "model_score",
    "close",
    "recommended_review",
    "broker_action",
    "research_only",
    "source_dir",
    "source_file",
]

INDEX_CN_COLUMNS = {
    "signal_date": "信号日期",
    "symbol": "证券代码",
    "name": "证券名称",
    "signal_source": "信号来源",
    "signal_rank": "信号排名",
    "priority_score": "优先分",
    "score_bucket": "分数桶",
    "model_score": "模型分",
    "close": "收盘价",
    "recommended_review": "建议复核",
    "broker_action": "券商动作",
    "research_only": "仅研究",
    "source_dir": "来源目录",
    "source_file": "来源文件",
}

SUPPORTED_FILES = {
    "chip_reversal_daily_candidates.csv": "chip_reversal_daily_candidates",
    "chip_reversal_observation_queue.csv": "chip_reversal_observation_queue",
    "chip_reversal_candidate_outcomes.csv": "chip_reversal_candidate_outcomes",
}
LAB_EVENT_FILE = "chip_reversal_events.csv"
LAB_EVENT_SOURCE = "chip_reversal_lab_events"


@dataclass(frozen=True)
class HistoricalSignalIndexResult:
    output_dir: Path
    index_path: Path
    index_cn_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]


def _clean_code(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    return text.zfill(6) if text else ""


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _source_date_from_dir(path: Path) -> str:
    match = re.search(r"(20\d{6})", path.name)
    if not match:
        return ""
    raw = match.group(1)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"


def _normalize_date(value: str | pd.Timestamp | None) -> str | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid date: {value}")
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _symbol_set(symbols: list[str] | tuple[str, ...] | set[str] | None) -> set[str] | None:
    if not symbols:
        return None
    normalized = {_clean_code(symbol) for symbol in symbols if str(symbol).strip()}
    return normalized or None


def _col(data: pd.DataFrame, name: str, default: Any = "") -> pd.Series:
    if name in data.columns:
        return data[name]
    return pd.Series([default] * len(data), index=data.index)


def _first_existing(data: pd.DataFrame, names: list[str], default: Any = "") -> pd.Series:
    for name in names:
        if name in data.columns:
            return data[name]
    return pd.Series([default] * len(data), index=data.index)


def _apply_index_filters(
    frame: pd.DataFrame,
    symbols: set[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = frame.copy()
    if symbols is not None:
        data = data[data["symbol"].isin(symbols)].copy()
    if start_date is not None:
        data = data[data["signal_date"] >= start_date].copy()
    if end_date is not None:
        data = data[data["signal_date"] <= end_date].copy()
    return data.reset_index(drop=True)


def _normalize_source_frame(data: pd.DataFrame, path: Path, source: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    signal_date = _first_existing(data, ["signal_date", "date"], _source_date_from_dir(path.parent))
    rank = _first_existing(data, ["candidate_rank", "observation_rank"], 0)
    model_score = _first_existing(data, ["chip_reversal_score", "score", "model_score"], 0.0)
    out = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(signal_date, errors="coerce").dt.strftime("%Y-%m-%d"),
            "symbol": _first_existing(data, ["code", "symbol"]).map(_clean_code),
            "name": _col(data, "name"),
            "signal_source": source,
            "signal_rank": pd.to_numeric(rank, errors="coerce").fillna(0).astype(int),
            "priority_score": pd.to_numeric(_col(data, "priority_score", 0.0), errors="coerce").fillna(0.0),
            "score_bucket": _col(data, "score_bucket"),
            "model_score": pd.to_numeric(model_score, errors="coerce").fillna(0.0),
            "close": pd.to_numeric(_first_existing(data, ["close", "signal_close"], 0.0), errors="coerce").fillna(0.0),
            "recommended_review": _col(data, "recommended_review"),
            "broker_action": _col(data, "broker_action", "none"),
            "research_only": _col(data, "research_only", True),
            "source_dir": str(path.parent),
            "source_file": str(path),
        }
    )
    out = out[(out["signal_date"].notna()) & (out["signal_date"] != "NaT") & (out["symbol"] != "")]
    return out[INDEX_COLUMNS].reset_index(drop=True)


def _normalize_source_file(path: Path, source: str) -> pd.DataFrame:
    data = pd.read_csv(path, encoding="utf-8-sig")
    return _normalize_source_frame(data, path, source)


def _normalize_lab_events_file(
    path: Path,
    symbols: set[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, encoding="utf-8-sig", chunksize=chunksize):
        if chunk.empty:
            continue
        if "code" in chunk.columns:
            chunk["code"] = chunk["code"].map(_clean_code)
        date_column = "date" if "date" in chunk.columns else "signal_date"
        if date_column in chunk.columns:
            chunk[date_column] = pd.to_datetime(chunk[date_column], errors="coerce").dt.strftime("%Y-%m-%d")
        if symbols is not None and "code" in chunk.columns:
            chunk = chunk[chunk["code"].isin(symbols)].copy()
        if start_date is not None and date_column in chunk.columns:
            chunk = chunk[chunk[date_column] >= start_date].copy()
        if end_date is not None and date_column in chunk.columns:
            chunk = chunk[chunk[date_column] <= end_date].copy()
        if chunk.empty:
            continue
        frames.append(_normalize_source_frame(chunk, path, LAB_EVENT_SOURCE))
    if not frames:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def build_historical_signal_index(
    research_dir: str | Path = "outputs/research",
    include_lab_events: bool = False,
    symbols: list[str] | tuple[str, ...] | set[str] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    root = Path(research_dir)
    normalized_symbols = _symbol_set(symbols)
    normalized_start = _normalize_date(start_date)
    normalized_end = _normalize_date(end_date)
    frames: list[pd.DataFrame] = []
    for file_name, source in SUPPORTED_FILES.items():
        for path in root.rglob(file_name):
            try:
                frame = _normalize_source_file(path, source)
            except Exception:
                continue
            if not frame.empty:
                frames.append(
                    _apply_index_filters(
                        frame,
                        symbols=normalized_symbols,
                        start_date=normalized_start,
                        end_date=normalized_end,
                    )
                )
    if include_lab_events:
        for path in root.rglob(LAB_EVENT_FILE):
            try:
                frame = _normalize_lab_events_file(
                    path,
                    symbols=normalized_symbols,
                    start_date=normalized_start,
                    end_date=normalized_end,
                )
            except Exception:
                continue
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=INDEX_COLUMNS)
    index = pd.concat(frames, ignore_index=True)
    return index.drop_duplicates(
        subset=["signal_date", "symbol", "signal_source", "source_file"],
        keep="first",
    ).sort_values(["signal_date", "symbol", "signal_source", "signal_rank"]).reset_index(drop=True)


def _snapshot(index: pd.DataFrame, research_dir: Path) -> dict[str, Any]:
    by_source = index["signal_source"].value_counts().to_dict() if not index.empty else {}
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "ok" if not index.empty else "empty",
        "broker_action": "none",
        "data_usage": "research_only_historical_signal_index",
        "research_dir": str(research_dir),
        "signal_count": int(len(index)),
        "symbol_count": int(index["symbol"].nunique()) if not index.empty else 0,
        "start_date": str(index["signal_date"].min()) if not index.empty else "",
        "end_date": str(index["signal_date"].max()) if not index.empty else "",
        "source_counts": {str(key): int(value) for key, value in by_source.items()},
        "research_conclusion": "只读研究索引，可用于真实成交对账；索引本身不连接券商，不自动下单。",
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 20) -> str:
    if not rows:
        return "暂无数据。"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows[:max_rows]:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, sep, *body])


def _render_report(snapshot: dict[str, Any], index: pd.DataFrame) -> str:
    source_rows = [
        {"来源": source, "信号数": count}
        for source, count in sorted(snapshot.get("source_counts", {}).items(), key=lambda item: item[0])
    ]
    sample_rows = []
    if not index.empty:
        for _, row in index.head(20).iterrows():
            sample_rows.append(
                {
                    "日期": row["signal_date"],
                    "证券": f"{row['symbol']} {row['name']}",
                    "来源": row["signal_source"],
                    "排名": row["signal_rank"],
                    "优先分": f"{float(row['priority_score']):.2f}",
                }
            )
    return f"""# 历史模型信号索引

<!-- internal: Historical Model Signal Index -->

生成时间：`{snapshot["generated_at"]}`

<span class="badge">只读研究索引</span> broker_action=none，不连接券商，不自动下单。

## 摘要

| 项目 | 数值 |
| --- | ---: |
| 信号数 | {snapshot["signal_count"]} |
| 标的数 | {snapshot["symbol_count"]} |
| 起止日期 | {snapshot["start_date"]} 至 {snapshot["end_date"]} |

## 来源分布

{_markdown_table(source_rows, ["来源", "信号数"], max_rows=20)}

## 样例

{_markdown_table(sample_rows, ["日期", "证券", "来源", "排名", "优先分"], max_rows=20)}

## 说明

该文件只汇总已经落盘的研究输出，用于真实成交与模型信号对账。信号日期来自源文件的 `date` / `signal_date` 字段，缺失时才回退到目录名中的 `YYYYMMDD`。
"""


def run_historical_signal_index(
    research_dir: str | Path = "outputs/research",
    output_dir: str | Path = "outputs/research/historical_signal_index_latest",
    include_lab_events: bool = False,
    symbols: list[str] | tuple[str, ...] | set[str] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> HistoricalSignalIndexResult:
    research_path = Path(research_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = build_historical_signal_index(
        research_path,
        include_lab_events=include_lab_events,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )
    snapshot = _snapshot(index, research_path)

    index_path = out_dir / "historical_signal_index.csv"
    index_cn_path = out_dir / "historical_signal_index_cn.csv"
    snapshot_path = out_dir / "historical_signal_index_snapshot.json"
    report_path = out_dir / "historical_signal_index.md"

    index.to_csv(index_path, index=False, encoding="utf-8-sig")
    index.rename(columns={key: value for key, value in INDEX_CN_COLUMNS.items() if key in index.columns}).to_csv(
        index_cn_path,
        index=False,
        encoding="utf-8-sig",
    )
    snapshot_path.write_text(json.dumps(_json_ready(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, index), encoding="utf-8")

    return HistoricalSignalIndexResult(
        output_dir=out_dir,
        index_path=index_path,
        index_cn_path=index_cn_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
    )
