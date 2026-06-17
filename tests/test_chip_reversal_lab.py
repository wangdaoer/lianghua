from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_etf_lab.chip_reversal_lab import build_chip_reversal_events, run_chip_reversal_lab
from quant_etf_lab.cli import build_parser


def test_build_chip_reversal_events_uses_daily_proxy_and_forward_outcomes_only() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=9, freq="D"),
            "code": ["000001"] * 9,
            "name": ["Main"] * 9,
            "open": [10.0, 10.4, 10.8, 10.6, 9.2, 8.7, 8.8, 9.1, 9.0],
            "high": [10.2, 10.6, 11.2, 10.9, 9.4, 8.8, 9.0, 9.3, 9.2],
            "low": [9.8, 10.2, 10.6, 10.4, 8.8, 8.4, 8.6, 8.9, 8.8],
            "close": [10.0, 10.5, 11.0, 10.8, 9.0, 8.6, 8.9, 9.2, 9.0],
            "volume": [1_000_000] * 9,
            "amount": [100_000_000, 120_000_000, 150_000_000, 130_000_000, 170_000_000, 180_000_000, 160_000_000, 190_000_000, 140_000_000],
        }
    )

    events = build_chip_reversal_events(
        prices,
        horizons=[1, 2],
        drawdown_window=3,
        cost_window=3,
        min_drawdown_pct=10.0,
        min_score=0.05,
    )

    by_date = events.set_index("date")
    assert "2026-01-06" in by_date.index
    event = by_date.loc["2026-01-06"]
    assert event["signal_type"] == "chip_reversal_daily_proxy"
    assert float(event["chip_reversal_score"]) > 0.05
    assert round(float(event["drawdown_3d"]), 6) == round(8.6 / 11.2 - 1.0, 6)
    assert round(float(event["return_1d"]), 6) == round(8.9 / 8.6 - 1.0, 6)
    assert round(float(event["trade_return_2d"]), 6) == round(9.2 / 8.8 - 1.0, 6)
    assert event["auction_confirmation_status"] == "unknown"
    assert event["proxy_confirmation"] == "daily_only"
    assert event["broker_action"] == "none"
    assert bool(event["research_only"]) is True
    assert "2026-01-09" not in by_date.index


def test_run_chip_reversal_lab_writes_events_summary_snapshot_and_report(tmp_path: Path) -> None:
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "code": ["000001"] * 10,
            "name": ["Main"] * 10,
            "open": [10.0, 10.4, 10.8, 10.6, 9.2, 8.7, 8.8, 9.1, 9.0, 9.2],
            "high": [10.2, 10.6, 11.2, 10.9, 9.4, 8.8, 9.0, 9.3, 9.2, 9.4],
            "low": [9.8, 10.2, 10.6, 10.4, 8.8, 8.4, 8.6, 8.9, 8.8, 9.0],
            "close": [10.0, 10.5, 11.0, 10.8, 9.0, 8.6, 8.9, 9.2, 9.0, 9.3],
            "volume": [1_000_000] * 10,
            "amount": [100_000_000, 120_000_000, 150_000_000, 130_000_000, 170_000_000, 180_000_000, 160_000_000, 190_000_000, 140_000_000, 150_000_000],
        }
    ).to_csv(data_dir / "000001.csv", index=False)
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "code": ["300001"] * 10,
            "name": ["Chi"] * 10,
            "open": [20.0, 20.4, 21.0, 20.8, 18.9, 18.1, 18.3, 18.6, 18.7, 18.9],
            "high": [20.2, 20.8, 21.5, 21.0, 19.1, 18.3, 18.6, 18.8, 18.9, 19.1],
            "low": [19.8, 20.0, 20.7, 20.4, 18.5, 17.9, 18.1, 18.4, 18.5, 18.6],
            "close": [20.0, 20.6, 21.2, 20.7, 18.7, 18.0, 18.4, 18.7, 18.6, 19.0],
            "volume": [900_000] * 10,
            "amount": [90_000_000, 110_000_000, 140_000_000, 130_000_000, 150_000_000, 160_000_000, 150_000_000, 170_000_000, 150_000_000, 160_000_000],
        }
    ).to_csv(data_dir / "300001.csv", index=False)

    result = run_chip_reversal_lab(
        project_root=tmp_path,
        data_dir=data_dir,
        output_dir=tmp_path / "chip_reversal",
        horizons=[1, 2],
        min_events=1,
        drawdown_window=3,
        cost_window=3,
        min_drawdown_pct=10.0,
        min_score=0.05,
    )

    assert result.snapshot["status"] == "ok"
    assert result.snapshot["event_count"] >= 2
    assert result.snapshot["research_only"] is True
    assert result.events_path.exists()
    assert result.summary_path.exists()
    assert result.snapshot_path.exists()
    assert result.report_path.exists()
    assert {"signal_type", "horizon", "win_rate", "avg_trade_return", "payoff_ratio"}.issubset(result.summary.columns)
    assert {"auction_confirmation_status", "proxy_confirmation", "broker_action"}.issubset(result.events.columns)
    assert "chip_reversal_daily_proxy" in result.report_path.read_text(encoding="utf-8")


def test_run_chip_reversal_lab_applies_strict_research_filters(tmp_path: Path) -> None:
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    base = {
        "date": pd.date_range("2026-01-01", periods=10, freq="D"),
        "name": ["Pass"] * 10,
        "open": [10.0, 10.4, 10.8, 10.6, 9.2, 8.7, 8.7, 10.7, 11.0, 11.2],
        "high": [10.2, 10.6, 11.2, 10.9, 9.4, 8.8, 10.7, 11.0, 11.2, 11.4],
        "low": [9.8, 10.2, 10.6, 10.4, 8.8, 8.4, 8.6, 10.5, 10.8, 11.0],
        "close": [10.0, 10.5, 11.0, 10.8, 9.0, 8.6, 10.5, 10.9, 11.0, 11.2],
        "volume": [1_000_000] * 10,
        "amount": [100_000_000, 120_000_000, 150_000_000, 130_000_000, 50_000_000, 180_000_000, 160_000_000, 190_000_000, 140_000_000, 150_000_000],
    }
    pass_frame = pd.DataFrame({**base, "code": ["000001"] * 10})
    pass_frame.to_csv(data_dir / "000001.csv", index=False)

    high_gap_frame = pass_frame.copy()
    high_gap_frame["code"] = "000002"
    high_gap_frame["name"] = "HighGap"
    high_gap_frame.loc[6, "open"] = 9.0
    high_gap_frame.to_csv(data_dir / "000002.csv", index=False)

    low_amount_frame = pass_frame.copy()
    low_amount_frame["code"] = "300001"
    low_amount_frame["name"] = "LowAmount"
    low_amount_frame.loc[5, "amount"] = 50_000_000
    low_amount_frame.to_csv(data_dir / "300001.csv", index=False)

    result = run_chip_reversal_lab(
        project_root=tmp_path,
        data_dir=data_dir,
        output_dir=tmp_path / "chip_reversal_strict",
        horizons=[1, 2],
        min_events=1,
        drawdown_window=3,
        cost_window=3,
        min_drawdown_pct=10.0,
        min_score=0.05,
        score_bucket="deep",
        min_amount_yi=1.5,
        max_next_open_gap_pct=2.0,
    )

    assert result.snapshot["raw_event_count"] > result.snapshot["event_count"]
    assert result.snapshot["score_bucket_filter"] == "deep"
    assert result.snapshot["min_amount_yi"] == 1.5
    assert result.snapshot["max_next_open_gap_pct"] == 2.0
    assert result.snapshot["filter_counts"]["after_max_next_open_gap_pct"] == 1
    assert set(result.events["code"]) == {"000001"}
    assert set(result.events["score_bucket"]) == {"deep"}
    assert float(result.events["amount_yi"].min()) >= 1.5
    assert float(result.events["open_gap_next"].max()) <= 0.02
    assert set(result.events["execution_filter_status"]) == {"passed"}


def test_run_chip_reversal_lab_applies_theme_state_gate(tmp_path: Path) -> None:
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "code": ["000001"] * 10,
            "name": ["GoodTheme"] * 10,
            "open": [10.0, 10.4, 10.8, 10.6, 9.2, 8.7, 8.7, 10.7, 11.0, 11.2],
            "high": [10.2, 10.6, 11.2, 10.9, 9.4, 8.8, 10.7, 11.0, 11.2, 11.4],
            "low": [9.8, 10.2, 10.6, 10.4, 8.8, 8.4, 8.6, 10.5, 10.8, 11.0],
            "close": [10.0, 10.5, 11.0, 10.8, 9.0, 8.6, 10.5, 10.9, 11.0, 11.2],
            "volume": [1_000_000] * 10,
            "amount": [100_000_000, 120_000_000, 150_000_000, 130_000_000, 170_000_000, 180_000_000, 160_000_000, 190_000_000, 140_000_000, 150_000_000],
        }
    )
    frame.to_csv(data_dir / "000001.csv", index=False)
    weak_frame = frame.copy()
    weak_frame["code"] = "000002"
    weak_frame["name"] = "WeakTheme"
    weak_frame.to_csv(data_dir / "000002.csv", index=False)

    symbol_panel = tmp_path / "theme_symbol_panel.csv"
    pd.DataFrame(
        [
            {"date": "2026-01-06", "code": "000001", "theme_group": "theme_good"},
            {"date": "2026-01-06", "code": "000002", "theme_group": "theme_weak"},
        ]
    ).to_csv(symbol_panel, index=False)
    theme_state = tmp_path / "theme_state.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-01-06",
                "theme_group": "theme_good",
                "theme_state": "healthy",
                "theme_second_activation": False,
                "breadth_ma20_smooth": 0.72,
                "theme_rs_20d": 0.08,
            },
            {
                "date": "2026-01-06",
                "theme_group": "theme_weak",
                "theme_state": "weak",
                "theme_second_activation": False,
                "breadth_ma20_smooth": 0.20,
                "theme_rs_20d": -0.05,
            },
        ]
    ).to_csv(theme_state, index=False)

    result = run_chip_reversal_lab(
        project_root=tmp_path,
        data_dir=data_dir,
        output_dir=tmp_path / "chip_reversal_theme_gate",
        horizons=[1, 2],
        min_events=1,
        drawdown_window=3,
        cost_window=3,
        min_drawdown_pct=10.0,
        min_score=0.05,
        start_date="2026-01-06",
        end_date="2026-01-06",
        theme_state_path=theme_state,
        theme_symbol_panel_path=symbol_panel,
        allowed_theme_states=["healthy", "second_activation"],
    )

    assert result.snapshot["raw_event_count"] == 2
    assert result.snapshot["event_count"] == 1
    assert result.snapshot["theme_state_gate"]["applied"] is True
    assert result.snapshot["theme_state_gate"]["passed_count"] == 1
    assert result.snapshot["theme_state_gate"]["blocked_count"] == 1
    assert set(result.events["code"]) == {"000001"}
    assert set(result.events["theme_state"]) == {"healthy"}
    assert set(result.events["theme_gate_status"]) == {"passed"}
    assert set(result.events["theme_group"]) == {"theme_good"}


def test_chip_reversal_lab_cli_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["chip-reversal-lab"])

    assert args.command == "chip-reversal-lab"
    assert args.data_dir == "data/processed/stocks"
    assert args.output_dir == "outputs/research/chip_reversal_lab_latest"
    assert args.horizons == "1,2"
    assert args.min_drawdown_pct == 12.0
    assert args.min_score == 0.08
    assert args.score_bucket == "all"
    assert args.min_amount_yi == 0.0
    assert args.max_next_open_gap_pct is None
    assert args.theme_state_path is None
    assert args.theme_symbol_panel_path is None
    assert args.allowed_theme_states == "healthy,second_activation"
    assert args.board_scope == "main_chinext"
