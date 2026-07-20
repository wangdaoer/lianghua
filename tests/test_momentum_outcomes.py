from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser
from quant_etf_lab.momentum_outcomes import build_momentum_events, run_momentum_outcome_analysis


def test_build_momentum_events_labels_limit_up_and_forward_returns_without_future_leakage() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=7, freq="D"),
            "code": ["000001"] * 7,
            "name": ["Test"] * 7,
            "open": [10.0, 10.8, 11.8, 11.6, 12.0, 11.4, 11.9],
            "high": [10.2, 11.2, 12.2, 12.0, 12.4, 11.8, 12.3],
            "low": [9.8, 10.6, 11.5, 11.2, 11.5, 11.0, 11.6],
            "close": [10.0, 11.0, 12.1, 11.8, 12.2, 11.6, 12.0],
            "volume": [1_000_000] * 7,
            "amount": [100_000_000, 300_000_000, 350_000_000, 160_000_000, 180_000_000, 170_000_000, 190_000_000],
        }
    )

    events = build_momentum_events(prices, horizons=[1, 3], strong_gain_threshold_pct=7.0)

    by_date = events.set_index("date")
    assert "2026-01-02" in by_date.index
    assert by_date.loc["2026-01-02", "signal_type"] == "strong_gain_7"
    assert bool(by_date.loc["2026-01-02", "limit_up"]) is True
    assert round(float(by_date.loc["2026-01-02", "return_1d"]), 6) == 0.10
    assert round(float(by_date.loc["2026-01-02", "return_3d"]), 6) == round(12.2 / 11.0 - 1.0, 6)
    assert "2026-01-07" not in by_date.index


def test_run_momentum_outcome_analysis_writes_summary_and_report(tmp_path: Path) -> None:
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8, freq="D"),
            "code": ["000001"] * 8,
            "name": ["Main"] * 8,
            "open": [10.0, 10.8, 11.5, 11.0, 10.8, 11.3, 11.8, 12.2],
            "high": [10.2, 11.2, 11.8, 11.5, 11.2, 11.8, 12.2, 12.6],
            "low": [9.8, 10.6, 11.0, 10.5, 10.4, 11.0, 11.5, 12.0],
            "close": [10.0, 11.0, 11.5, 10.9, 11.4, 11.9, 12.3, 12.4],
            "volume": [1_000_000] * 8,
            "amount": [100_000_000, 300_000_000, 280_000_000, 100_000_000, 250_000_000, 260_000_000, 270_000_000, 280_000_000],
        }
    ).to_csv(data_dir / "000001.csv", index=False)
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8, freq="D"),
            "code": ["300001"] * 8,
            "name": ["Chi"] * 8,
            "open": [20.0, 21.2, 21.5, 21.0, 22.0, 21.6, 22.5, 23.0],
            "high": [20.5, 21.8, 22.0, 21.5, 22.5, 22.2, 23.0, 23.6],
            "low": [19.8, 20.8, 21.0, 20.6, 21.5, 21.2, 22.0, 22.7],
            "close": [20.0, 21.6, 21.0, 22.6, 21.4, 22.8, 23.3, 23.8],
            "volume": [800_000] * 8,
            "amount": [90_000_000, 220_000_000, 120_000_000, 230_000_000, 110_000_000, 240_000_000, 250_000_000, 260_000_000],
        }
    ).to_csv(data_dir / "300001.csv", index=False)

    result = run_momentum_outcome_analysis(
        project_root=tmp_path,
        data_dir=data_dir,
        output_dir=tmp_path / "momentum_outcomes",
        horizons=[1, 3],
        min_events=1,
    )

    assert result.snapshot["status"] == "ok"
    assert result.snapshot["event_count"] >= 3
    assert result.events_path.exists()
    assert result.summary_path.exists()
    assert result.report_path.exists()
    assert {"signal_type", "horizon", "win_rate", "avg_return", "capital_efficiency"}.issubset(result.summary.columns)
    assert {"board", "amount_bucket", "avg_amount_yi"}.issubset(result.summary.columns)
    assert "main" in set(result.summary["board"])
    assert "gte_1y" in set(result.events["amount_bucket"])
    assert "涨停与强势股后验评估" in result.report_path.read_text(encoding="utf-8")


def test_momentum_outcomes_cli_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["momentum-outcomes"])

    assert args.command == "momentum-outcomes"
    assert args.data_dir == "data/processed/stocks"
    assert args.output_dir == "outputs/research/momentum_outcomes_latest"
    assert args.horizons == "1,3,5,10"
    assert args.strong_gain_threshold_pct == 5.0
    assert args.board_scope == "main_chinext"
