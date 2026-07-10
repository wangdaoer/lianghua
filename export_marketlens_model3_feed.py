from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("outputs/high_return_v2")
DEFAULT_DASHBOARD_OUTPUT = Path("D:/codex/outputs/stock-analysis-dashboard/data/quant-model3-latest.json")
LOCAL_PATH_RE = re.compile(r"[A-Za-z]:\\[^\"'\n\r]+")


def date_token(asof_date: str) -> str:
    return asof_date.replace("-", "")


def latest_asof_date(output_root: Path) -> str:
    candidates = sorted(output_root.glob("merged_priority_watchlist_*.csv"), key=lambda item: item.stat().st_mtime)
    candidates = [
        item
        for item in candidates
        if not item.stem.endswith("_cn") and re.search(r"merged_priority_watchlist_\d{8}", item.stem)
    ]
    if not candidates:
        raise FileNotFoundError(f"No merged_priority_watchlist_YYYYMMDD.csv found under {output_root}")
    match = re.search(r"(\d{8})", candidates[-1].stem)
    if not match:
        raise ValueError(f"Cannot infer date from {candidates[-1]}")
    token = match.group(1)
    return f"{token[:4]}-{token[4:6]}-{token[6:8]}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def latest_state_record(state_log: Path, asof_date: str) -> dict[str, Any]:
    if not state_log.exists():
        return {}
    records: list[dict[str, Any]] = []
    for line in state_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("asof_date") == asof_date:
            records.append(record)
    return records[-1] if records else {}


def sanitize_public_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_public_value(item) for item in value]
    if isinstance(value, str):
        if LOCAL_PATH_RE.search(value):
            return LOCAL_PATH_RE.sub("[local-path]", value)
        return value
    return value


def clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def number_or_none(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def pct_or_none(value: Any) -> float | None:
    number = number_or_none(value)
    if number is None:
        return None
    return round(number * 100.0, 4)


def bool_value(value: Any) -> bool:
    return clean_text(value).lower() in {"1", "true", "yes", "y", "是"}


def bucket_label(bucket: str) -> str:
    return {
        "action_focus": "行动关注",
        "model_focus": "模型关注",
        "pattern_watch": "形态观察",
        "risk_watch": "风险观察",
        "review_later": "稍后复盘",
    }.get(bucket, bucket or "未分层")


def shape_priority_row(row: dict[str, str]) -> dict[str, Any]:
    bucket = clean_text(row.get("priority_bucket"))
    shaped = {
        "code": clean_text(row.get("symbol") or row.get("code")),
        "name": clean_text(row.get("stock_name") or row.get("name")),
        "bucket": bucket,
        "bucketLabel": bucket_label(bucket),
        "priorityScore": number_or_none(row.get("priority_score")),
        "rank": number_or_none(row.get("personal_rank")),
        "selected": bool_value(row.get("personal_selected")),
        "targetWeightPct": pct_or_none(row.get("personal_target_weight")),
        "action": clean_text(row.get("personal_action")),
        "actionLabel": clean_text(row.get("personal_action_cn")),
        "reason": clean_text(row.get("personal_reasons_cn") or row.get("pattern_reason")),
        "trendState": clean_text(row.get("trend_state")),
        "patternType": clean_text(row.get("pattern_type")),
        "patternScore": number_or_none(row.get("pattern_score")),
        "riskFlags": clean_text(row.get("risk_flags")),
        "close": number_or_none(row.get("close")),
        "return5dPct": pct_or_none(row.get("return_5d")),
        "return20dPct": pct_or_none(row.get("return_20d")),
        "return60dPct": pct_or_none(row.get("return_60d")),
        "closePositionPct": pct_or_none(row.get("close_position")),
    }
    return {key: value for key, value in shaped.items() if value not in (None, "")}


def compact_artifacts(output_root: Path, asof_date: str) -> dict[str, str]:
    token = date_token(asof_date)
    names = {
        "priorityWatchlist": f"merged_priority_watchlist_{token}.csv",
        "priorityWatchlistCn": f"merged_priority_watchlist_{token}_cn.csv",
        "modelDecision": f"merged_model_decision_table_{token}.csv",
        "earlyWatchlist": f"early_pattern_watchlist_{token}.csv",
        "report": f"merged_daily_outputs_{token}.md",
    }
    return {key: value for key, value in names.items() if (output_root / value).exists()}


def build_feed(asof_date: str, output_root: Path, state_log: Path) -> dict[str, Any]:
    token = date_token(asof_date)
    priority_path = output_root / f"merged_priority_watchlist_{token}.csv"
    early_path = output_root / f"early_pattern_watchlist_{token}.csv"
    model_path = output_root / f"merged_model_decision_table_{token}.csv"
    priority_rows_raw = read_csv_rows(priority_path)
    early_rows = read_csv_rows(early_path)
    model_rows = read_csv_rows(model_path)
    state_record = latest_state_record(state_log, asof_date)
    priority_rows = [shape_priority_row(row) for row in priority_rows_raw]
    bucket_counts = dict(Counter(row.get("bucket", "未分层") for row in priority_rows))
    verification = {
        "tests": state_record.get("verification", {}).get("tests", "unknown"),
        "priorityRows": len(priority_rows),
        "earlyPatternRows": len(early_rows),
        "modelDecisionRows": len(model_rows),
        "missingNames": sum(1 for row in priority_rows if not row.get("name")),
    }
    feed = {
        "schemaVersion": 1,
        "generatedAt": datetime.now().isoformat(timespec="seconds"),
        "asofDate": asof_date,
        "source": {
            "model": "high_risk_quant_model3",
            "purpose": "MarketLens website research snapshot",
            "publicPathsOnly": True,
        },
        "runType": state_record.get("run_type", "unknown"),
        "verification": verification,
        "bucketCounts": bucket_counts,
        "buckets": [
            {"bucket": bucket, "label": bucket_label(bucket), "count": count}
            for bucket, count in bucket_counts.items()
        ],
        "top10": priority_rows[:10],
        "priorityRows": priority_rows,
        "artifacts": compact_artifacts(output_root, asof_date),
    }
    return sanitize_public_value(feed)


def export_feed(
    asof_date: str,
    output_root: Path,
    state_log: Path,
    output_path: Path,
    dashboard_output: Path | None = None,
) -> dict[str, Any]:
    feed = build_feed(asof_date, output_root, state_log)
    text = json.dumps(feed, ensure_ascii=False, indent=2) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    if dashboard_output:
        dashboard_output.parent.mkdir(parents=True, exist_ok=True)
        dashboard_output.write_text(text, encoding="utf-8")
    return feed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Model3 snapshot for the MarketLens website.")
    parser.add_argument("--asof-date", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--state-log", default="daily_run_state.jsonl")
    parser.add_argument("--output", default="")
    parser.add_argument("--dashboard-output", default=str(DEFAULT_DASHBOARD_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    asof_date = args.asof_date or latest_asof_date(output_root)
    output_path = Path(args.output) if args.output else output_root / "marketlens_model3_latest.json"
    dashboard_output = Path(args.dashboard_output) if args.dashboard_output else None
    feed = export_feed(asof_date, output_root, Path(args.state_log), output_path, dashboard_output)
    print(f"Exported Model3 feed: {output_path}")
    if dashboard_output:
        print(f"Dashboard copy: {dashboard_output}")
    print(f"Rows: {feed['verification']['priorityRows']} / asof: {feed['asofDate']}")


if __name__ == "__main__":
    main()
