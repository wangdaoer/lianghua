from pathlib import Path

import pandas as pd

from monitor_factor_decay import (
    classify_overall_status,
    summarize_factor_decay,
    update_monitor_history,
)


def _daily_rows() -> pd.DataFrame:
    rows = []
    for horizon in (5, 10, 20):
        for date in pd.date_range("2023-01-01", periods=12, freq="D"):
            rows.append(
                {
                    "date": date,
                    "factor": "liquidity_stability_20",
                    "horizon": horizon,
                    "rank_ic": 0.10,
                    "top_bottom_spread": 0.01,
                }
            )
        for date in pd.date_range("2026-01-01", periods=12, freq="D"):
            rows.append(
                {
                    "date": date,
                    "factor": "liquidity_stability_20",
                    "horizon": horizon,
                    "rank_ic": -0.05 if horizon != 20 else 0.02,
                    "top_bottom_spread": -0.01,
                }
            )
    return pd.DataFrame(rows)


def test_summary_detects_multi_horizon_direction_reversal():
    summary = summarize_factor_decay(
        _daily_rows(),
        selection_start=pd.Timestamp("2023-01-01"),
        selection_end=pd.Timestamp("2025-12-31"),
        asof_date=pd.Timestamp("2026-02-01"),
        recent_signal_days=12,
    )

    assert summary.set_index("horizon").loc[5, "status"] == "direction_reversal"
    assert summary.set_index("horizon").loc[20, "status"] == "weakened"
    assert classify_overall_status(summary) == "direction_reversal"


def test_history_update_is_idempotent_for_same_date(tmp_path: Path):
    history_path = tmp_path / "factor_decay_monitor_history.csv"
    summary = summarize_factor_decay(
        _daily_rows(),
        selection_start=pd.Timestamp("2023-01-01"),
        selection_end=pd.Timestamp("2025-12-31"),
        asof_date=pd.Timestamp("2026-02-01"),
        recent_signal_days=12,
    )

    first = update_monitor_history(
        history_path, pd.Timestamp("2026-02-01"), summary, "direction_reversal"
    )
    second = update_monitor_history(
        history_path, pd.Timestamp("2026-02-01"), summary, "direction_reversal"
    )

    assert len(first) == 3
    assert len(second) == 3
    assert history_path.exists()
