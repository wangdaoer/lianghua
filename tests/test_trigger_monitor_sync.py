from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser
from quant_etf_lab.trigger_monitor_sync import sync_thsdk_trigger_monitor_outputs


def _journal_frame(report_path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_key": "2026-06-19|600360|candidate",
                "first_seen_ts": "2026-06-19T19:01:49",
                "last_seen_ts": "2026-06-19T19:01:49",
                "seen_count": 1,
                "trading_date": "2026-06-19",
                "code": "600360",
                "name": "Hua Micro",
                "signal_type": "breakout",
                "action_category": "candidate",
                "action": "wait_pullback",
                "score": 97,
                "score_level": "A",
                "last": 12.4,
                "pct": 1.8062,
                "support": 10.71,
                "pressure": 12.29,
                "stop_loss": 12.29,
                "volume_ratio": 1.2615,
                "prev_close": 12.18,
                "prev_close_source": "previous_row_close",
                "prev_gap_pct": "",
                "data_status": "prev_close_fallback_previous_row",
                "strategy_tags": "breakout",
                "czsc_summary": "CZSC ok",
                "reason": "candidate reason",
                "score_reason": "score reason",
                "invalidation": "break stop",
                "watch_conditions": "watch pullback",
                "source_report": str(report_path),
                "backtest_horizon": 7,
                "hist_trigger_count": 21,
                "hist_win_rate": 76.19,
                "hist_avg_return": 5.79,
                "hist_best_return": 29.58,
                "hist_worst_return": -3.89,
                "outcome_status": "pending",
                "outcome_return_pct": "",
                "outcome_evaluated_at": "",
                "schema_version": 1,
            },
            {
                "event_key": "2026-06-19|605006|risk",
                "first_seen_ts": "2026-06-19T19:01:49",
                "last_seen_ts": "2026-06-19T19:01:49",
                "seen_count": 1,
                "trading_date": "2026-06-19",
                "code": "605006",
                "name": "Risk Stock",
                "signal_type": "risk",
                "action_category": "risk",
                "action": "do_not_buy",
                "score": 0,
                "score_level": "D",
                "last": 23.4,
                "pct": -10.0,
                "support": 20.54,
                "pressure": 26.38,
                "stop_loss": 23.4,
                "volume_ratio": "",
                "prev_close": 26.0,
                "prev_close_source": "previous_row_close",
                "prev_gap_pct": "",
                "data_status": "prev_close_fallback_previous_row",
                "strategy_tags": "risk",
                "czsc_summary": "CZSC ok",
                "reason": "risk reason",
                "score_reason": "risk score",
                "invalidation": "risk recover",
                "watch_conditions": "do not buy",
                "source_report": str(report_path),
                "backtest_horizon": 7,
                "hist_trigger_count": 18,
                "hist_win_rate": 88.89,
                "hist_avg_return": 11.04,
                "hist_best_return": 44.26,
                "hist_worst_return": -6.71,
                "outcome_status": "pending",
                "outcome_return_pct": "",
                "outcome_evaluated_at": "",
                "schema_version": 1,
            },
        ]
    )


def test_trigger_sync_writes_candidate_only_latest_signals(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    report = output_root / "thsdk_strategy_monitor_2026-06-19_190149.md"
    journal = output_root / "signal_journal" / "signal_journal.csv"
    report.parent.mkdir(parents=True)
    journal.parent.mkdir(parents=True)
    report.write_text("# THSDK monitor 2026-06-19_190149\n", encoding="utf-8")
    _journal_frame(report).to_csv(journal, index=False, encoding="utf-8-sig")

    result = sync_thsdk_trigger_monitor_outputs(output_root=output_root)

    latest = pd.read_csv(result.latest_signal_path, dtype={"code": str}, encoding="utf-8-sig")
    assert latest["code"].tolist() == ["600360"]
    assert latest["run_time"].tolist() == ["2026-06-19_190149"]
    assert result.latest_trigger_path.read_text(encoding="utf-8-sig").startswith("# THSDK monitor")
    assert result.snapshot["candidate_source_row_count"] == 1
    assert result.snapshot["risk_source_row_count"] == 1
    assert result.snapshot["latest_signal_count"] == 1

    snapshot = json.loads(result.snapshot_path.read_text(encoding="utf-8-sig"))
    assert snapshot["latest_signal_path"] == str(result.latest_signal_path)


def test_trigger_sync_can_include_risk_signals_when_requested(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    report = output_root / "thsdk_strategy_monitor_2026-06-19_190149.md"
    journal = output_root / "signal_journal" / "signal_journal.csv"
    report.parent.mkdir(parents=True)
    journal.parent.mkdir(parents=True)
    report.write_text("# THSDK monitor 2026-06-19_190149\n", encoding="utf-8")
    _journal_frame(report).to_csv(journal, index=False, encoding="utf-8-sig")

    result = sync_thsdk_trigger_monitor_outputs(output_root=output_root, include_risk_signals=True)

    latest = pd.read_csv(result.latest_signal_path, dtype={"code": str}, encoding="utf-8-sig")
    assert latest["code"].tolist() == ["600360", "605006"]
    assert result.snapshot["include_risk_signals"] is True


def test_trigger_sync_records_expected_trade_date_freshness(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    report = output_root / "thsdk_strategy_monitor_2026-06-19_190149.md"
    journal = output_root / "signal_journal" / "signal_journal.csv"
    report.parent.mkdir(parents=True)
    journal.parent.mkdir(parents=True)
    report.write_text("# THSDK monitor 2026-06-19_190149\n", encoding="utf-8")
    _journal_frame(report).to_csv(journal, index=False, encoding="utf-8-sig")

    result = sync_thsdk_trigger_monitor_outputs(output_root=output_root, expected_trade_date="2026-06-20")

    assert result.snapshot["expected_trade_date"] == "2026-06-20"
    assert result.snapshot["trade_date_matches_expected"] is False
    assert result.snapshot["trigger_sync_freshness_status"] == "stale"


def test_trigger_sync_cli_parser_defaults() -> None:
    args = build_parser().parse_args(["trigger-sync"])

    assert args.command == "trigger-sync"
    assert args.output_root == "D:/codex/outputs"
    assert args.expected_trade_date is None
    assert args.include_risk_signals is False


def test_trigger_sync_cli_parser_accepts_expected_trade_date() -> None:
    args = build_parser().parse_args(["trigger-sync", "--expected-trade-date", "2026-06-29"])

    assert args.expected_trade_date == "2026-06-29"
