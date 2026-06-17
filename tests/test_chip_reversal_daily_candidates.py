from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_etf_lab.chip_reversal_daily_candidates import (
    build_chip_reversal_daily_candidates,
    run_chip_reversal_daily_candidates,
)
from quant_etf_lab.cli import build_parser


def _latest_signal_prices(code: str = "000001", name: str = "LatestSignal") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=6, freq="D"),
            "code": [code] * 6,
            "name": [name] * 6,
            "open": [10.0, 10.4, 10.8, 10.6, 9.2, 8.7],
            "high": [10.2, 10.6, 11.2, 10.9, 9.4, 8.8],
            "low": [9.8, 10.2, 10.6, 10.4, 8.8, 8.4],
            "close": [10.0, 10.5, 11.0, 10.8, 9.0, 8.6],
            "volume": [1_000_000] * 6,
            "amount": [
                100_000_000,
                120_000_000,
                150_000_000,
                130_000_000,
                170_000_000,
                180_000_000,
            ],
        }
    )


def test_build_chip_reversal_daily_candidates_uses_latest_non_mature_day_without_forward_labels() -> None:
    candidates, snapshot = build_chip_reversal_daily_candidates(
        [_latest_signal_prices()],
        as_of_date=None,
        horizons=[1, 2],
        drawdown_window=3,
        cost_window=3,
        min_drawdown_pct=10.0,
        min_score=0.05,
        score_bucket="deep",
        min_amount_yi=1.5,
        max_candidates=10,
    )

    assert snapshot["status"] == "ok"
    assert snapshot["as_of_date"] == "2026-01-06"
    assert snapshot["candidate_count"] == 1
    assert snapshot["ranking_uses_forward_labels"] is False
    assert "next_open" in snapshot["future_label_columns_removed"]
    assert "return_1d" in snapshot["future_label_columns_removed"]
    assert "trade_return_2d" in snapshot["future_label_columns_removed"]

    row = candidates.iloc[0]
    assert row["date"] == "2026-01-06"
    assert row["code"] == "000001"
    assert row["watch_posture"] == "watch_only"
    assert row["review_status"] == "pending_observation"
    assert row["label_status"] == "not_available_non_mature"
    assert row["auction_confirmation_status"] == "unknown"
    assert row["proxy_confirmation"] == "daily_only"
    assert row["broker_action"] == "none"
    assert bool(row["research_only"]) is True
    assert "target_weight" not in candidates.columns
    assert "order_quantity" not in candidates.columns
    assert not any(column in candidates.columns for column in ["next_open", "open_gap_next"])
    assert not any(column.startswith(("return_", "trade_return_")) for column in candidates.columns)


def test_run_chip_reversal_daily_candidates_writes_watch_only_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    _latest_signal_prices("000001", "Main").to_csv(data_dir / "000001.csv", index=False)
    _latest_signal_prices("300001", "ChiNext").to_csv(data_dir / "300001.csv", index=False)

    result = run_chip_reversal_daily_candidates(
        project_root=tmp_path,
        data_dir=data_dir,
        output_dir=tmp_path / "daily_candidates",
        horizons=[1, 2],
        drawdown_window=3,
        cost_window=3,
        min_drawdown_pct=10.0,
        min_score=0.05,
        score_bucket="deep",
        min_amount_yi=1.5,
        max_candidates=1,
    )

    assert result.snapshot["status"] == "ok"
    assert result.snapshot["as_of_date"] == "2026-01-06"
    assert result.snapshot["candidate_count"] == 1
    assert result.snapshot["broker_action"] == "none"
    assert result.candidates_path.exists()
    assert result.snapshot_path.exists()
    assert result.report_path.exists()
    assert result.candidates.iloc[0]["watch_posture"] == "watch_only"
    report_text = result.report_path.read_text(encoding="utf-8")
    assert "不是买卖建议" in report_text
    assert "| 排名 | 代码 | 名称 | 优先级 | 姿态 |" in report_text


def test_run_chip_reversal_daily_candidates_can_overlay_unified_market_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    _latest_signal_prices("000001", "Main").iloc[:-1].to_csv(data_dir / "000001.csv", index=False)

    class FakeMarketSnapshot:
        rows = [
            {
                "trade_date": "2026-01-06",
                "security_code": "000001",
                "open_price": 8.7,
                "high_price": 8.8,
                "low_price": 8.4,
                "close_price": 8.6,
                "volume": 1_000_000,
                "turnover": 180_000_000,
            }
        ]
        source_kind = "daily_market_data_csv"
        source_path = Path("D:/codex/daily-market-data/snapshots/snapshot_all_20260106.csv")
        trade_date = "2026-01-06"
        fetch_status = type("FetchStatus", (), {"status": "ok", "message": ""})()

    def fake_load_market_snapshot_rows(**kwargs):
        return FakeMarketSnapshot()

    monkeypatch.setattr(
        "quant_etf_lab.chip_reversal_daily_candidates.load_market_snapshot_rows",
        fake_load_market_snapshot_rows,
    )

    result = run_chip_reversal_daily_candidates(
        project_root=tmp_path,
        data_dir=data_dir,
        output_dir=tmp_path / "daily_candidates_overlay",
        horizons=[1, 2],
        drawdown_window=3,
        cost_window=3,
        min_drawdown_pct=10.0,
        min_score=0.05,
        score_bucket="deep",
        min_amount_yi=1.5,
        max_candidates=10,
        market_snapshot_overlay=True,
    )

    assert result.snapshot["status"] == "ok"
    assert result.snapshot["as_of_date"] == "2026-01-06"
    assert result.snapshot["market_snapshot_overlay"] is True
    assert result.snapshot["market_source_kind"] == "daily_market_data_csv"
    assert result.snapshot["market_trade_date"] == "2026-01-06"
    assert result.candidates.iloc[0]["date"] == "2026-01-06"
    assert result.candidates.iloc[0]["name"] == "Main"


def test_chip_reversal_daily_candidates_cli_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["chip-reversal-daily-candidates"])

    assert args.command == "chip-reversal-daily-candidates"
    assert args.data_dir == "data/processed/stocks"
    assert args.output_dir == "outputs/research/chip_reversal_daily_candidates_latest"
    assert args.horizons == "1,2"
    assert args.score_bucket == "deep"
    assert args.min_amount_yi == 1.5
    assert args.board_scope == "main_chinext"
    assert args.max_candidates == 50
    assert args.market_snapshot_overlay is True
