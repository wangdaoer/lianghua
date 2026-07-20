from __future__ import annotations

import pandas as pd
import pytest

from quant_etf_lab.chip_reversal_failure_filters import (
    FailureFilterSpec,
    apply_failure_filter,
    build_walk_forward_selection,
    equity_segment_metrics,
    generate_year_walk_forward_windows,
    rank_filter_results,
)


def test_failure_filter_uses_signal_day_fields_without_future_returns() -> None:
    events = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "code": ["000001", "000002", "000003"],
            "chip_reversal_score": [0.19, 0.25, 0.30],
            "daily_return_pct": [8.0, 10.0, 12.0],
            "trade_return_2d": [9.99, -9.99, -9.99],
        }
    )

    filtered, audit = apply_failure_filter(
        events,
        FailureFilterSpec(
            name="chip20_exclude_9_11",
            description="test",
            min_chip_score=0.20,
            excluded_daily_gain_ranges=((9.0, 11.0),),
        ),
    )

    assert filtered["code"].tolist() == ["000003"]
    assert audit["input_event_count"] == 3
    assert audit["kept_event_count"] == 1
    assert "trade_return_2d" not in audit["used_filter_columns"]


def test_rank_filter_results_allows_drawdown_equal_to_limit() -> None:
    rows = pd.DataFrame(
        [
            {"candidate": "at_limit", "total_return": 1.0, "max_drawdown": -0.37, "sharpe": 1.0},
            {"candidate": "over_limit", "total_return": 2.0, "max_drawdown": -0.371, "sharpe": 2.0},
        ]
    )

    ranked = rank_filter_results(rows, max_drawdown_limit=0.37)

    by_name = ranked.set_index("candidate")
    assert bool(by_name.loc["at_limit", "passes_drawdown_gate"]) is True
    assert bool(by_name.loc["over_limit", "passes_drawdown_gate"]) is False


def test_rank_filter_results_prefers_highest_return_inside_drawdown_gate() -> None:
    rows = pd.DataFrame(
        [
            {"candidate": "too_much_dd", "total_return": 5.0, "max_drawdown": -0.50, "sharpe": 2.0},
            {"candidate": "best_allowed", "total_return": 3.0, "max_drawdown": -0.36, "sharpe": 1.0},
            {"candidate": "lower_allowed", "total_return": 2.0, "max_drawdown": -0.20, "sharpe": 3.0},
        ]
    )

    ranked = rank_filter_results(rows, max_drawdown_limit=0.37)

    assert ranked.iloc[0]["candidate"] == "best_allowed"
    assert ranked.iloc[0]["selection_status"] == "selected_under_drawdown_gate"


def test_generate_year_walk_forward_windows_uses_train_then_next_test_year() -> None:
    windows = generate_year_walk_forward_windows(2018, 2022, train_years=2, test_years=1)

    assert windows == [
        {
            "window": "train_2018_2019_test_2020_2020",
            "train_start": "2018-01-01",
            "train_end": "2019-12-31",
            "test_start": "2020-01-01",
            "test_end": "2020-12-31",
        },
        {
            "window": "train_2019_2020_test_2021_2021",
            "train_start": "2019-01-01",
            "train_end": "2020-12-31",
            "test_start": "2021-01-01",
            "test_end": "2021-12-31",
        },
        {
            "window": "train_2020_2021_test_2022_2022",
            "train_start": "2020-01-01",
            "train_end": "2021-12-31",
            "test_start": "2022-01-01",
            "test_end": "2022-12-31",
        },
    ]


def test_equity_segment_metrics_rebases_window_and_calculates_drawdown() -> None:
    curve = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "equity": [100.0, 110.0, 105.0, 120.0],
            "exposure": [0.1, 0.2, 0.3, 0.4],
        }
    )

    metrics = equity_segment_metrics(curve, "2024-01-02", "2024-01-05")

    assert metrics["total_return"] == pytest.approx(0.20)
    assert metrics["max_drawdown"] == pytest.approx(-0.045454545454545414)
    assert metrics["average_exposure"] == pytest.approx(0.25)
    assert metrics["trading_days"] == 4


def test_build_walk_forward_selection_picks_train_winner_then_reports_test() -> None:
    windows = [
        {
            "window": "w1",
            "train_start": "2020-01-01",
            "train_end": "2020-12-31",
            "test_start": "2021-01-01",
            "test_end": "2021-12-31",
        }
    ]
    candidate_metrics = pd.DataFrame(
        [
            {"candidate": "overfit", "window": "w1", "split": "train", "total_return": 2.0, "max_drawdown": -0.50, "sharpe": 2.0},
            {"candidate": "stable", "window": "w1", "split": "train", "total_return": 1.0, "max_drawdown": -0.20, "sharpe": 1.0},
            {"candidate": "stable", "window": "w1", "split": "test", "total_return": 0.30, "max_drawdown": -0.10, "sharpe": 0.8},
            {"candidate": "overfit", "window": "w1", "split": "test", "total_return": 0.80, "max_drawdown": -0.40, "sharpe": 1.2},
        ]
    )

    selected = build_walk_forward_selection(windows, candidate_metrics, max_drawdown_limit=0.37)

    assert selected.iloc[0]["selected_candidate"] == "stable"
    assert selected.iloc[0]["train_total_return"] == 1.0
    assert selected.iloc[0]["test_total_return"] == 0.30
