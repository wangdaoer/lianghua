import pandas as pd
import pytest

from evaluate_trend_ignition_score_forward_returns import (
    assign_score_bucket,
    summarize_score_forward_returns,
    validate_scorer_temporal_contract,
)


def test_assign_score_bucket_uses_quantiles_when_levels_are_missing():
    frame = pd.DataFrame({"trend_ignition_score": [0.1, 0.2, 0.3, 0.4, 0.5]})

    out = assign_score_bucket(frame, allow_dynamic_thresholds=True)

    assert out["trend_ignition_score_bucket"].tolist() == ["low", "low", "middle", "high", "high"]


def test_assign_score_bucket_uses_fixed_training_thresholds():
    frame = pd.DataFrame({"trend_ignition_score": [0.1, 0.4, 0.7]})

    out = assign_score_bucket(frame, {"low_max": 0.2, "high_min": 0.6})

    assert out["trend_ignition_score_bucket"].tolist() == ["low", "middle", "high"]


def test_assign_score_bucket_rejects_missing_score_column():
    with pytest.raises(ValueError, match="requires trend_ignition_score"):
        assign_score_bucket(pd.DataFrame({"symbol": ["000001"]}), {"low_max": 0.2, "high_min": 0.6})


def test_static_scorer_must_predate_forward_score_dates():
    watchlists = pd.DataFrame({"asof_date": ["2025-01-02"]})

    with pytest.raises(ValueError, match="later than"):
        validate_scorer_temporal_contract(watchlists, {"training_end_date": "2025-01-02"})


def test_summarize_score_forward_returns_groups_by_score_bucket():
    samples = pd.DataFrame(
        {
            "trend_ignition_score_bucket": ["high", "high", "low"],
            "forward_return_5d": [0.10, 0.20, -0.05],
            "max_adverse_return_5d": [-0.02, -0.03, -0.08],
        }
    )

    summary = summarize_score_forward_returns(samples, horizons=(5,))

    high = summary[summary["trend_ignition_score_bucket"].eq("high")].iloc[0]
    low = summary[summary["trend_ignition_score_bucket"].eq("low")].iloc[0]
    assert high["completed_count"] == 2
    assert round(high["avg_return"], 6) == 0.15
    assert high["win_rate"] == 1.0
    assert low["avg_return"] == -0.05
