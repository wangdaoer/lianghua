"""Sync external THSDK trigger-monitor outputs into the quant project contract."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_THSDK_OUTPUT_ROOT = Path("D:/codex/outputs")
DEFAULT_TRIGGER_REPORT_DIR = DEFAULT_THSDK_OUTPUT_ROOT / "trigger_reports"
DEFAULT_SIGNAL_HISTORY_DIR = DEFAULT_THSDK_OUTPUT_ROOT / "signal_history"
DEFAULT_SIGNAL_JOURNAL_PATH = DEFAULT_THSDK_OUTPUT_ROOT / "signal_journal" / "signal_journal.csv"

LATEST_SIGNAL_COLUMNS = [
    "run_time",
    "code",
    "name",
    "ths_code",
    "last",
    "pct",
    "high",
    "low",
    "prev_close",
    "prev_close_source",
    "prev_gap_pct",
    "reference_warning",
    "volume_ratio",
    "signal_type",
    "action",
    "reason",
    "support",
    "pressure",
    "stop_loss",
    "score",
    "score_level",
    "score_reason",
    "strategy_tags",
    "czsc_summary",
]


@dataclass(frozen=True)
class TriggerMonitorSyncResult:
    output_root: Path
    source_report_path: Path
    source_journal_path: Path
    latest_trigger_path: Path
    dated_trigger_path: Path
    latest_signal_path: Path
    dated_signal_path: Path
    snapshot_path: Path
    snapshot: dict[str, Any]


def _clean_code(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6) if digits else text.zfill(6)


def _ths_code(code: str) -> str:
    return ("USZH" if code.startswith("6") else "USZA") + code


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _number_text(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        return f"{float(text):.10g}"
    except ValueError:
        return text


def _run_time_from_row(row: dict[str, Any]) -> str:
    for key in ("last_seen_ts", "first_seen_ts", "trading_date"):
        text = _text(row.get(key))
        if not text:
            continue
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.notna(parsed):
            if key == "trading_date":
                return pd.Timestamp(parsed).strftime("%Y-%m-%d_000000")
            return pd.Timestamp(parsed).strftime("%Y-%m-%d_%H%M%S")
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _trade_date_from_rows(rows: pd.DataFrame, fallback_report: Path) -> str:
    if not rows.empty and "trading_date" in rows.columns:
        dates = pd.to_datetime(rows["trading_date"], errors="coerce").dropna()
        if not dates.empty:
            return pd.Timestamp(dates.max()).strftime("%Y-%m-%d")
    match = pd.Series([fallback_report.name]).str.extract(r"(20\d{2}-\d{2}-\d{2})", expand=False).dropna()
    if not match.empty:
        return str(match.iloc[0])
    return datetime.now().strftime("%Y-%m-%d")


def _normalize_trade_date(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return pd.Timestamp(parsed).strftime("%Y-%m-%d")
    return text


def _run_stamp_from_report(report_path: Path) -> str:
    match = pd.Series([report_path.name]).str.extract(r"(20\d{2}-\d{2}-\d{2}_\d{6})", expand=False).dropna()
    if not match.empty:
        return str(match.iloc[0])
    return datetime.fromtimestamp(report_path.stat().st_mtime).strftime("%Y-%m-%d_%H%M%S")


def _latest_report(output_root: Path) -> Path:
    candidates = sorted(output_root.glob("thsdk_strategy_monitor_*.md"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No THSDK monitor report found under {output_root}")
    return candidates[-1]


def _normalize_path(path: Any) -> str:
    return str(Path(_text(path))).lower().replace("/", "\\")


def _load_journal_rows(journal_path: Path, report_path: Path) -> pd.DataFrame:
    if not journal_path.exists():
        return pd.DataFrame()
    journal = pd.read_csv(journal_path, dtype={"code": str}, encoding="utf-8-sig")
    if journal.empty:
        return journal
    if "source_report" in journal.columns:
        source = journal["source_report"].map(_normalize_path)
        target = _normalize_path(report_path)
        matched = journal.loc[source == target].copy()
        if not matched.empty:
            return matched
    if "trading_date" in journal.columns:
        report_date = _trade_date_from_rows(pd.DataFrame(), report_path)
        dated = journal.loc[journal["trading_date"].astype(str).str[:10] == report_date].copy()
        if not dated.empty:
            return dated
    return pd.DataFrame(columns=journal.columns)


def _latest_signal_frame(rows: pd.DataFrame, *, include_risk_signals: bool) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=LATEST_SIGNAL_COLUMNS)
    data = rows.copy()
    if "action_category" in data.columns:
        allowed = {"candidate"}
        if include_risk_signals:
            allowed.add("risk")
        data = data.loc[data["action_category"].astype(str).isin(allowed)].copy()
    if data.empty:
        return pd.DataFrame(columns=LATEST_SIGNAL_COLUMNS)
    records: list[dict[str, Any]] = []
    for raw in data.to_dict(orient="records"):
        code = _clean_code(raw.get("code"))
        if not code:
            continue
        records.append(
            {
                "run_time": _run_time_from_row(raw),
                "code": code,
                "name": _text(raw.get("name")),
                "ths_code": _ths_code(code),
                "last": _number_text(raw.get("last")),
                "pct": _number_text(raw.get("pct")),
                "high": "",
                "low": "",
                "prev_close": _number_text(raw.get("prev_close")),
                "prev_close_source": _text(raw.get("prev_close_source")),
                "prev_gap_pct": _number_text(raw.get("prev_gap_pct")),
                "reference_warning": "",
                "volume_ratio": _number_text(raw.get("volume_ratio")),
                "signal_type": _text(raw.get("signal_type")),
                "action": _text(raw.get("action")),
                "reason": _text(raw.get("reason")),
                "support": _number_text(raw.get("support")),
                "pressure": _number_text(raw.get("pressure")),
                "stop_loss": _number_text(raw.get("stop_loss")),
                "score": _number_text(raw.get("score")),
                "score_level": _text(raw.get("score_level")),
                "score_reason": _text(raw.get("score_reason")),
                "strategy_tags": _text(raw.get("strategy_tags")),
                "czsc_summary": _text(raw.get("czsc_summary")),
            }
        )
    return pd.DataFrame(records, columns=LATEST_SIGNAL_COLUMNS)


def sync_thsdk_trigger_monitor_outputs(
    *,
    output_root: str | Path = DEFAULT_THSDK_OUTPUT_ROOT,
    report_path: str | Path | None = None,
    journal_path: str | Path | None = None,
    include_risk_signals: bool = False,
    expected_trade_date: str | None = None,
) -> TriggerMonitorSyncResult:
    root = Path(output_root)
    source_report = Path(report_path) if report_path is not None else _latest_report(root)
    source_journal = Path(journal_path) if journal_path is not None else root / "signal_journal" / "signal_journal.csv"
    if not source_report.exists():
        raise FileNotFoundError(f"THSDK monitor report does not exist: {source_report}")

    rows = _load_journal_rows(source_journal, source_report)
    signal_frame = _latest_signal_frame(rows, include_risk_signals=include_risk_signals)
    trade_date = _trade_date_from_rows(rows, source_report)
    run_stamp = _run_stamp_from_report(source_report)
    normalized_expected_trade_date = _normalize_trade_date(expected_trade_date)
    trade_date_matches_expected = (
        None if normalized_expected_trade_date is None else trade_date == normalized_expected_trade_date
    )
    trigger_sync_freshness_status = (
        "unchecked"
        if trade_date_matches_expected is None
        else "fresh_enough"
        if trade_date_matches_expected
        else "stale"
    )

    trigger_dir = root / "trigger_reports"
    signal_dir = root / "signal_history"
    trigger_dir.mkdir(parents=True, exist_ok=True)
    signal_dir.mkdir(parents=True, exist_ok=True)

    dated_trigger = trigger_dir / f"strategy_trigger_{run_stamp}.md"
    latest_trigger = trigger_dir / "latest_trigger.md"
    shutil.copy2(source_report, dated_trigger)
    shutil.copy2(source_report, latest_trigger)

    dated_signal = signal_dir / f"signals_{trade_date}.csv"
    latest_signal = signal_dir / "signals_latest.csv"
    signal_frame.to_csv(dated_signal, index=False, encoding="utf-8-sig")
    signal_frame.to_csv(latest_signal, index=False, encoding="utf-8-sig")

    action_counts = rows["action_category"].value_counts().to_dict() if "action_category" in rows.columns else {}
    snapshot = {
        "schema_version": "thsdk-trigger-monitor-sync-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(root),
        "source_report_path": str(source_report),
        "source_journal_path": str(source_journal),
        "latest_trigger_path": str(latest_trigger),
        "dated_trigger_path": str(dated_trigger),
        "latest_signal_path": str(latest_signal),
        "dated_signal_path": str(dated_signal),
        "trade_date": trade_date,
        "expected_trade_date": normalized_expected_trade_date,
        "trade_date_matches_expected": trade_date_matches_expected,
        "trigger_sync_freshness_status": trigger_sync_freshness_status,
        "run_stamp": run_stamp,
        "source_row_count": int(len(rows)),
        "candidate_source_row_count": int(action_counts.get("candidate", 0)),
        "risk_source_row_count": int(action_counts.get("risk", 0)),
        "latest_signal_count": int(len(signal_frame)),
        "include_risk_signals": bool(include_risk_signals),
    }
    snapshot_path = root / "trigger_reports" / "latest_trigger_sync_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    return TriggerMonitorSyncResult(
        output_root=root,
        source_report_path=source_report,
        source_journal_path=source_journal,
        latest_trigger_path=latest_trigger,
        dated_trigger_path=dated_trigger,
        latest_signal_path=latest_signal,
        dated_signal_path=dated_signal,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )
