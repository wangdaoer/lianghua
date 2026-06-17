from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_etf_lab.chip_reversal_observation_queue import (
    build_chip_reversal_observation_queue,
    run_chip_reversal_observation_queue,
)
from quant_etf_lab.cli import build_parser


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "code": "000001",
                "name": "Old",
                "board": "main",
                "score_bucket": "deep",
                "chip_reversal_score": 0.25,
                "drawdown_20d": -0.30,
                "cost_gap_ma20": -0.12,
                "amount_yi": 5.0,
                "close": 8.0,
                "next_open": 8.1,
                "return_1d": 0.20,
                "trade_return_1d": 0.19,
                "broker_action": "none",
                "research_only": True,
            },
            {
                "date": "2026-01-06",
                "code": "000002",
                "name": "HighScore",
                "board": "main",
                "score_bucket": "deep",
                "chip_reversal_score": 0.30,
                "drawdown_20d": -0.35,
                "cost_gap_ma20": -0.14,
                "amount_yi": 6.0,
                "close": 7.0,
                "next_open": 7.5,
                "return_1d": -0.10,
                "trade_return_1d": -0.12,
                "broker_action": "none",
                "research_only": True,
            },
            {
                "date": "2026-01-06",
                "code": "000003",
                "name": "HighForwardReturn",
                "board": "main",
                "score_bucket": "deep",
                "chip_reversal_score": 0.10,
                "drawdown_20d": -0.13,
                "cost_gap_ma20": -0.04,
                "amount_yi": 8.0,
                "close": 6.0,
                "next_open": 6.1,
                "return_1d": 0.50,
                "trade_return_1d": 0.48,
                "broker_action": "none",
                "research_only": True,
            },
        ]
    )


def test_build_chip_reversal_observation_queue_ignores_forward_labels_for_ranking() -> None:
    queue, snapshot = build_chip_reversal_observation_queue(_events(), as_of_date="2026-01-06", max_candidates=2)

    assert snapshot["status"] == "ok"
    assert snapshot["as_of_date"] == "2026-01-06"
    assert snapshot["candidate_count"] == 2
    assert snapshot["forward_label_columns_ignored"] == ["next_open", "return_1d", "trade_return_1d"]
    assert list(queue["code"]) == ["000002", "000003"]
    assert queue.iloc[0]["watch_posture"] == "watch_only"
    assert queue.iloc[0]["review_status"] == "pending_observation"
    assert queue.iloc[0]["broker_action"] == "none"
    assert bool(queue.iloc[0]["research_only"]) is True
    assert "target_weight" not in queue.columns
    assert "order_quantity" not in queue.columns


def test_run_chip_reversal_observation_queue_writes_outputs(tmp_path: Path) -> None:
    events_path = tmp_path / "chip_reversal_events.csv"
    _events().to_csv(events_path, index=False)

    result = run_chip_reversal_observation_queue(
        project_root=tmp_path,
        events_path=events_path,
        output_dir=tmp_path / "queue",
        as_of_date="2026-01-06",
        max_candidates=1,
    )

    assert result.snapshot["status"] == "ok"
    assert result.snapshot["candidate_count"] == 1
    assert result.snapshot["broker_action"] == "none"
    assert result.queue_path.exists()
    assert result.snapshot_path.exists()
    assert result.report_path.exists()
    assert result.queue.iloc[0]["code"] == "000002"
    assert "watch_only" in result.report_path.read_text(encoding="utf-8")


def test_chip_reversal_observation_queue_cli_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["chip-reversal-observation-queue"])

    assert args.command == "chip-reversal-observation-queue"
    assert args.events_path == "outputs/research/chip_reversal_lab_latest/chip_reversal_events.csv"
    assert args.output_dir == "outputs/research/chip_reversal_observation_queue_latest"
    assert args.max_candidates == 20
