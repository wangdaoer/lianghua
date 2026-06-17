from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_etf_lab.chip_reversal_candidate_outcomes import (
    build_chip_reversal_candidate_outcomes,
    build_outcome_group_summary,
    run_chip_reversal_candidate_outcomes,
)
from quant_etf_lab.cli import build_parser


def _history(code: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2026-01-06", "code": code, "name": "候选", "open": 10.0, "high": 10.2, "low": 9.8, "close": 10.0},
            {"date": "2026-01-07", "code": code, "name": "候选", "open": 10.4, "high": 10.8, "low": 10.1, "close": 10.6},
            {"date": "2026-01-08", "code": code, "name": "候选", "open": 10.7, "high": 11.2, "low": 10.5, "close": 11.0},
        ]
    )


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "date": "2026-01-06",
                "code": "000001",
                "name": "强势候选",
                "board": "main",
                "score_bucket": "deep",
                "close": 10.0,
                "priority_score": 55.0,
                "watch_posture": "watch_only",
            },
            {
                "candidate_rank": 2,
                "date": "2026-01-08",
                "code": "000002",
                "name": "未成熟候选",
                "board": "chinext",
                "score_bucket": "base",
                "close": 8.0,
                "priority_score": 40.0,
                "watch_posture": "watch_only",
            },
        ]
    )


def test_build_chip_reversal_candidate_outcomes_marks_ready_and_pending() -> None:
    outcomes, summary, snapshot = build_chip_reversal_candidate_outcomes(
        _candidates(),
        {"000001": _history("000001"), "000002": _history("000002")},
        horizons=[1, 2],
        success_threshold=0.03,
    )

    ready_1d = outcomes[(outcomes["code"] == "000001") & (outcomes["horizon"] == "1d")].iloc[0]
    ready_2d = outcomes[(outcomes["code"] == "000001") & (outcomes["horizon"] == "2d")].iloc[0]
    pending = outcomes[(outcomes["code"] == "000002") & (outcomes["horizon"] == "1d")].iloc[0]

    assert ready_1d["outcome_status"] == "ready"
    assert round(float(ready_1d["return_close"]), 6) == 0.06
    assert round(float(ready_1d["trade_return_open_to_close"]), 6) == round(10.6 / 10.4 - 1.0, 6)
    assert bool(ready_1d["success"]) is True
    assert ready_2d["outcome_status"] == "ready"
    assert bool(ready_2d["success"]) is True
    assert pending["outcome_status"] == "pending"
    assert pd.isna(pending["return_close"])

    assert snapshot["candidate_count"] == 2
    assert snapshot["ready_outcome_count"] == 2
    assert snapshot["pending_outcome_count"] == 2
    assert snapshot["broker_action"] == "none"
    assert summary.loc[summary["horizon"].eq("1d"), "ready_count"].iloc[0] == 1
    assert summary.loc[summary["horizon"].eq("1d"), "success_rate"].iloc[0] == 1.0


def test_chip_reversal_candidate_outcomes_records_promotion_gate() -> None:
    _, _, passing = build_chip_reversal_candidate_outcomes(
        _candidates(),
        {"000001": _history("000001"), "000002": _history("000002")},
        horizons=[1, 2],
        success_threshold=0.03,
        min_ready_per_horizon=1,
        min_success_rate=0.55,
    )
    _, _, blocked = build_chip_reversal_candidate_outcomes(
        _candidates(),
        {"000001": _history("000001"), "000002": _history("000002")},
        horizons=[1, 2],
        success_threshold=0.03,
        min_ready_per_horizon=2,
        min_success_rate=0.55,
    )

    assert passing["promotion_gate_status"] == "pass"
    assert passing["promotion_gate_min_ready_per_horizon"] == 1
    assert passing["promotion_gate_min_success_rate"] == 0.55
    assert blocked["promotion_gate_status"] == "blocked"
    assert any("ready_count_below_minimum" in reason for reason in blocked["promotion_gate_reasons"])


def test_build_chip_reversal_candidate_outcomes_uses_candidate_close_when_signal_bar_missing() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "date": "2026-01-06",
                "code": "000001",
                "name": "快照候选",
                "close": 10.0,
                "priority_score": 55.0,
                "watch_posture": "watch_only",
            }
        ]
    )
    history = pd.DataFrame(
        [
            {"date": "2026-01-07", "code": "000001", "open": 10.2, "close": 10.5},
            {"date": "2026-01-08", "code": "000001", "open": 10.6, "close": 10.9},
        ]
    )

    outcomes, summary, snapshot = build_chip_reversal_candidate_outcomes(
        candidates,
        {"000001": history},
        horizons=[1, 2],
        success_threshold=0.03,
    )

    first = outcomes[outcomes["horizon"].eq("1d")].iloc[0]
    second = outcomes[outcomes["horizon"].eq("2d")].iloc[0]
    assert first["outcome_status"] == "ready"
    assert first["pending_reason"] == ""
    assert round(float(first["return_close"]), 6) == 0.05
    assert second["outcome_status"] == "ready"
    assert round(float(second["return_close"]), 6) == 0.09
    assert snapshot["ready_outcome_count"] == 2
    assert summary.loc[summary["horizon"].eq("2d"), "success_rate"].iloc[0] == 1.0


def test_build_outcome_group_summary_reports_board_score_and_priority_buckets() -> None:
    outcomes, _, _ = build_chip_reversal_candidate_outcomes(
        _candidates(),
        {"000001": _history("000001"), "000002": _history("000002")},
        horizons=[1],
        success_threshold=0.03,
    )

    group_summary = build_outcome_group_summary(outcomes)
    main = group_summary[
        (group_summary["horizon"] == "1d")
        & (group_summary["group_type"] == "board")
        & (group_summary["group_value"] == "main")
    ].iloc[0]
    top_10 = group_summary[
        (group_summary["horizon"] == "1d")
        & (group_summary["group_type"] == "priority_bucket")
        & (group_summary["group_value"] == "top_10")
    ].iloc[0]
    deep = group_summary[
        (group_summary["horizon"] == "1d")
        & (group_summary["group_type"] == "score_bucket")
        & (group_summary["group_value"] == "deep")
    ].iloc[0]

    assert int(main["ready_count"]) == 1
    assert float(main["success_rate"]) == 1.0
    assert int(top_10["ready_count"]) == 1
    assert int(top_10["pending_count"]) == 1
    assert int(deep["success_count"]) == 1


def test_run_chip_reversal_candidate_outcomes_writes_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    _history("000001").to_csv(data_dir / "000001.csv", index=False)
    _history("000002").to_csv(data_dir / "000002.csv", index=False)
    candidates_path = tmp_path / "candidates.csv"
    _candidates().to_csv(candidates_path, index=False)

    result = run_chip_reversal_candidate_outcomes(
        project_root=tmp_path,
        candidates_path=candidates_path,
        data_dir=data_dir,
        output_dir=tmp_path / "outcomes",
        horizons=[1, 2],
        success_threshold=0.03,
    )

    assert result.snapshot["status"] == "ok"
    assert result.snapshot["ready_outcome_count"] == 2
    assert result.outcomes_path.exists()
    assert result.summary_path.exists()
    assert result.group_summary_path.exists()
    assert result.snapshot_path.exists()
    assert result.report_path.exists()
    report_text = result.report_path.read_text(encoding="utf-8")
    assert "候选结果回填" in report_text
    assert "不是买卖建议" in report_text
    assert "晋级门槛状态" in report_text
    assert "`blocked`" in report_text


def test_run_chip_reversal_candidate_outcomes_can_overlay_unified_market_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {"date": "2026-01-06", "code": "000001", "name": "Signal", "open": 9.8, "close": 10.0},
        ]
    ).to_csv(data_dir / "000001.csv", index=False)
    candidates_path = tmp_path / "candidates.csv"
    pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "date": "2026-01-06",
                "code": "000001",
                "name": "Signal",
                "board": "main",
                "score_bucket": "deep",
                "close": 10.0,
                "priority_score": 55.0,
                "watch_posture": "watch_only",
            }
        ]
    ).to_csv(candidates_path, index=False)

    class FakeMarketSnapshot:
        rows = [
            {
                "trade_date": "2026-01-07",
                "security_code": "000001",
                "open_price": 10.4,
                "high_price": 10.8,
                "low_price": 10.1,
                "close_price": 10.6,
                "volume": 1_000_000,
                "turnover": 180_000_000,
            }
        ]
        source_kind = "daily_market_data_csv"
        source_path = Path("D:/codex/daily-market-data/snapshots/snapshot_all_20260107.csv")
        trade_date = "2026-01-07"
        fetch_status = type("FetchStatus", (), {"status": "ok", "message": ""})()

    def fake_load_market_snapshot_rows(**kwargs):
        return FakeMarketSnapshot()

    monkeypatch.setattr(
        "quant_etf_lab.chip_reversal_candidate_outcomes.load_market_snapshot_rows",
        fake_load_market_snapshot_rows,
        raising=False,
    )

    result = run_chip_reversal_candidate_outcomes(
        project_root=tmp_path,
        candidates_path=candidates_path,
        data_dir=data_dir,
        output_dir=tmp_path / "outcomes_overlay",
        horizons=[1],
        success_threshold=0.03,
        market_snapshot_overlay=True,
        daily_data_dir=tmp_path / "daily-market-data",
        ingest_project_dir=tmp_path / "exchange-ingest",
    )

    row = result.outcomes.iloc[0]
    assert row["outcome_status"] == "ready"
    assert row["target_date"] == "2026-01-07"
    assert round(float(row["return_close"]), 6) == 0.06
    assert result.snapshot["market_snapshot_overlay"] is True
    assert result.snapshot["market_source_kind"] == "daily_market_data_csv"
    assert result.snapshot["market_trade_date"] == "2026-01-07"
    assert result.snapshot["market_row_count"] == 1


def test_run_chip_reversal_candidate_outcomes_marks_waiting_for_future_bar(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    pd.DataFrame(
        [
            {"date": "2026-01-06", "code": "000001", "name": "Signal", "open": 9.8, "close": 10.0},
        ]
    ).to_csv(data_dir / "000001.csv", index=False)
    candidates_path = tmp_path / "candidates.csv"
    pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "date": "2026-01-06",
                "code": "000001",
                "name": "Signal",
                "board": "main",
                "score_bucket": "deep",
                "close": 10.0,
                "priority_score": 55.0,
                "watch_posture": "watch_only",
            }
        ]
    ).to_csv(candidates_path, index=False)

    class FakeMarketSnapshot:
        rows = [
            {
                "trade_date": "2026-01-06",
                "security_code": "000001",
                "open_price": 9.8,
                "high_price": 10.1,
                "low_price": 9.7,
                "close_price": 10.0,
            }
        ]
        source_kind = "daily_market_data_csv"
        source_path = Path("D:/codex/daily-market-data/snapshots/snapshot_all_20260106.csv")
        trade_date = "2026-01-06"
        fetch_status = type("FetchStatus", (), {"status": "ok", "message": ""})()

    monkeypatch.setattr(
        "quant_etf_lab.chip_reversal_candidate_outcomes.load_market_snapshot_rows",
        lambda **kwargs: FakeMarketSnapshot(),
    )

    result = run_chip_reversal_candidate_outcomes(
        project_root=tmp_path,
        candidates_path=candidates_path,
        data_dir=data_dir,
        output_dir=tmp_path / "outcomes_waiting",
        horizons=[1, 2],
        success_threshold=0.03,
        market_snapshot_overlay=True,
        daily_data_dir=tmp_path / "daily-market-data",
        ingest_project_dir=tmp_path / "exchange-ingest",
    )

    assert result.snapshot["outcome_readiness_status"] == "waiting_for_future_bar"
    assert result.snapshot["outcome_analysis_status"] == "waiting_for_future_bar"
    assert result.snapshot["outcome_ready_horizons"] == []
    assert result.snapshot["outcome_pending_horizons"] == ["1d", "2d"]
    assert result.snapshot["next_outcome_review_horizon"] == "1d"
    assert result.snapshot["next_outcome_review_reason"] == "future_bar_not_available"
    assert result.snapshot["latest_signal_date"] == "2026-01-06"
    assert result.snapshot["latest_available_market_trade_date"] == "2026-01-06"


def test_chip_reversal_candidate_outcomes_cli_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["chip-reversal-candidate-outcomes"])

    assert args.command == "chip-reversal-candidate-outcomes"
    assert args.candidates_path == "outputs/research/chip_reversal_daily_candidates_latest/chip_reversal_daily_candidates.csv"
    assert args.data_dir == "data/processed/stocks"
    assert args.horizons == "1,2"
    assert args.success_threshold == 0.03
    assert args.min_ready_per_horizon == 30
    assert args.min_success_rate == 0.55
    assert args.market_snapshot_overlay is True
    assert args.daily_data_dir == "D:/codex/daily-market-data"
    assert args.ingest_project_dir == "D:/codex/2026-06-15-exchange-data-ingest"
