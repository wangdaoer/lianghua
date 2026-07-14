from __future__ import annotations

import pandas as pd
import pytest

from analyze_momentum_breadth_regime import (
    compute_daily_momentum_breadth,
    summarize_signal_buckets,
)


def breadth_panel(periods: int = 30) -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2026-01-01", periods=periods)
    for index, date in enumerate(dates):
        positive_count = 1 if index < 10 else 3
        for symbol_index in range(4):
            positive = symbol_index < positive_count
            rows.append(
                {
                    "date": date,
                    "symbol": f"{symbol_index:06d}",
                    "momentum_20": 0.20 if positive else -0.10,
                    "momentum_60": 0.30 if positive else -0.20,
                    "score_eligible": True,
                }
            )
    return pd.DataFrame(rows)


def test_daily_breadth_is_point_in_time_and_has_expected_levels():
    panel = breadth_panel()

    daily = compute_daily_momentum_breadth(panel)

    assert daily.iloc[0]["breadth_20"] == pytest.approx(0.25)
    assert daily.iloc[-1]["breadth_20"] == pytest.approx(0.75)
    assert pd.isna(daily.iloc[18]["breadth_20_ma20"])
    assert daily.iloc[19]["breadth_20_ma20"] == pytest.approx(0.50)

    extended = compute_daily_momentum_breadth(
        pd.concat([panel, breadth_panel(35).iloc[-20:]], ignore_index=True)
        .drop_duplicates(["date", "symbol"], keep="last")
    )
    pd.testing.assert_series_equal(
        daily.loc[: daily.index[-2], "breadth_20"],
        extended.loc[: daily.index[-2], "breadth_20"],
    )


def test_bucket_summary_reports_forward_return_direction():
    daily = compute_daily_momentum_breadth(breadth_panel())
    daily["forward_strategy_return_5d"] = daily["breadth_20"].sub(0.50)
    daily["forward_excess_return_5d"] = daily["breadth_20"].sub(0.60)

    summary = summarize_signal_buckets(
        daily, "breadth_20", (0.0, 0.50, 1.000001)
    )

    assert summary["sessions"].sum() == 30
    assert summary.iloc[0]["mean_forward_strategy_5d"] < 0.0
    assert summary.iloc[1]["mean_forward_strategy_5d"] > 0.0
