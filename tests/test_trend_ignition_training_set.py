import pandas as pd

from build_trend_ignition_training_set import build_training_rows, classify_lifeline_quality


def test_classify_lifeline_quality_marks_survivors_and_breaks():
    assert classify_lifeline_quality("alive_or_unbroken", None) == 1
    assert classify_lifeline_quality("broken", "2025-01-10") == 0
    assert classify_lifeline_quality("no_lifeline", None) == 0


def test_build_training_rows_derives_features_and_labels():
    samples = pd.DataFrame(
        [
            {
                "period": "2024-2026",
                "symbol": "300001",
                "ignition_date": "2024-09-27",
                "peak_date": "2026-01-15",
                "analysis_end_date": "2026-06-30",
                "peak_return": 2.0,
                "return_to_end_from_ignition": 1.2,
                "days_to_peak": 475,
                "max_drawdown_to_peak": -0.30,
                "breakout_pct": 0.05,
                "amount_ratio": 4.0,
                "return_20d": 0.20,
                "return_60d": 0.35,
                "volatility_20d": 0.03,
                "ma20_over_ma60": 0.08,
                "close_over_ma20": 0.05,
                "drawdown_120d": -0.01,
                "amount_trend_5_20": 1.8,
                "breakout_count_20d": 2.0,
                "lifeline_ma": 120,
                "min_distance_to_lifeline": -0.10,
                "avg_distance_to_lifeline": 0.25,
                "lifeline_status": "alive_or_unbroken",
                "lifeline_break_date": None,
            },
            {
                "period": "2024-2026",
                "symbol": "300002",
                "ignition_date": "2024-09-27",
                "peak_date": "2024-11-15",
                "analysis_end_date": "2026-06-30",
                "peak_return": 0.9,
                "return_to_end_from_ignition": -0.2,
                "days_to_peak": 49,
                "max_drawdown_to_peak": -0.45,
                "breakout_pct": 0.02,
                "amount_ratio": 1.6,
                "return_20d": 0.08,
                "return_60d": 0.12,
                "volatility_20d": 0.02,
                "ma20_over_ma60": 0.03,
                "close_over_ma20": 0.02,
                "drawdown_120d": -0.03,
                "amount_trend_5_20": 1.2,
                "breakout_count_20d": 1.0,
                "lifeline_ma": 20,
                "min_distance_to_lifeline": -0.25,
                "avg_distance_to_lifeline": 0.05,
                "lifeline_status": "broken",
                "lifeline_break_date": "2024-12-01",
            },
        ]
    )

    rows = build_training_rows(samples, high_return_threshold=1.5, min_days_to_peak=120)

    assert rows[
        [
            "symbol",
            "period",
            "label_high_trend",
            "label_lifeline_survives",
            "feature_log_amount_ratio",
        ]
    ].to_dict("records") == [
        {
            "symbol": "300001",
            "period": "2024-2026",
            "label_high_trend": 1,
            "label_lifeline_survives": 1,
            "feature_log_amount_ratio": rows.loc[0, "feature_log_amount_ratio"],
        },
        {
            "symbol": "300002",
            "period": "2024-2026",
            "label_high_trend": 0,
            "label_lifeline_survives": 0,
            "feature_log_amount_ratio": rows.loc[1, "feature_log_amount_ratio"],
        },
    ]

    assert rows.loc[0, "feature_return_60d"] == 0.35
    assert rows.loc[0, "feature_breakout_count_20d"] == 2.0


def test_build_training_rows_drops_censored_recent_samples():
    samples = pd.DataFrame(
        [
            {
                "period": "2025",
                "symbol": "300001",
                "ignition_date": "2025-12-01",
                "peak_date": "2025-12-15",
                "analysis_end_date": "2025-12-31",
                "peak_return": 0.2,
                "return_to_end_from_ignition": 0.1,
                "days_to_peak": 14,
                "max_drawdown_to_peak": -0.1,
                "breakout_pct": 0.03,
                "amount_ratio": 2.0,
                "return_20d": 0.10,
                "return_60d": 0.20,
                "volatility_20d": 0.02,
                "ma20_over_ma60": 0.04,
                "close_over_ma20": 0.03,
                "drawdown_120d": -0.02,
                "amount_trend_5_20": 1.4,
                "breakout_count_20d": 1.0,
                "lifeline_ma": 20,
                "min_distance_to_lifeline": -0.1,
                "avg_distance_to_lifeline": 0.1,
                "lifeline_status": "broken",
                "lifeline_break_date": "2025-12-20",
            }
        ]
    )

    rows = build_training_rows(samples, min_followup_days=120)

    assert rows.empty
