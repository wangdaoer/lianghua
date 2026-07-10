from __future__ import annotations

import pandas as pd

from summarize_core_risk_filter_stability import (
    build_segment_metrics,
    compare_segments_to_baseline,
    compute_returns,
    rolling_window_stats,
)


def test_rolling_window_stats_reports_worst_return_and_drawdown() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "equity": [100.0, 110.0, 99.0, 105.0, 84.0],
        }
    )
    returns = compute_returns(equity, initial_capital=100.0)

    stats = rolling_window_stats(returns, window=3)

    assert stats["rolling_3_positive_rate"] == 1 / 3
    assert round(stats["rolling_3_worst_return"], 6) == round(84.0 / 110.0 - 1.0, 6)
    assert stats["rolling_3_worst_end_date"] == "2026-01-05"
    assert stats["rolling_3_worst_drawdown"] < -0.19


def test_compare_segments_to_baseline_counts_period_wins() -> None:
    rows = [
        {"scheme": "baseline", "period_type": "year", "period": "2025", "return": 0.10, "max_drawdown": -0.20},
        {"scheme": "candidate", "period_type": "year", "period": "2025", "return": 0.12, "max_drawdown": -0.15},
        {"scheme": "baseline", "period_type": "year", "period": "2026", "return": -0.05, "max_drawdown": -0.18},
        {"scheme": "candidate", "period_type": "year", "period": "2026", "return": -0.07, "max_drawdown": -0.12},
    ]
    comparison = compare_segments_to_baseline(pd.DataFrame(rows), baseline_scheme="baseline")

    candidate = comparison.loc[comparison["scheme"] == "candidate"].iloc[0]

    assert candidate["period_count"] == 2
    assert candidate["return_win_count"] == 1
    assert round(candidate["avg_excess_return"], 6) == 0.0
    assert candidate["drawdown_improved_count"] == 2


def test_build_segment_metrics_uses_calendar_years_and_half_years() -> None:
    equity = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-12-31", "2026-01-02", "2026-06-30", "2026-07-01"]),
            "equity": [100.0, 110.0, 121.0, 108.9],
        }
    )
    returns = compute_returns(equity, initial_capital=100.0)

    segments = build_segment_metrics(returns, scheme="candidate")

    assert set(segments["period_type"]) == {"year", "half_year"}
    assert set(segments.loc[segments["period_type"] == "year", "period"]) == {"2025", "2026"}
    assert set(segments.loc[segments["period_type"] == "half_year", "period"]) == {"2025H2", "2026H1", "2026H2"}
