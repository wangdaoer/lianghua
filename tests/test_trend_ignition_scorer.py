import pandas as pd
import pytest

from train_trend_ignition_scorer import (
    fit_binned_feature_scorer,
    passes_research_gate,
    score_with_binned_feature_scorer,
    validate_feature_contract,
    walk_forward_feature_diagnostics,
    walk_forward_period_validation,
)


def test_binned_feature_scorer_orders_positive_rate_by_feature():
    train = pd.DataFrame(
        {
            "feature_log_amount_ratio": [0.1, 0.2, 0.8, 0.9],
            "feature_breakout_pct": [0.01, 0.02, 0.08, 0.09],
            "label_high_trend": [0, 0, 1, 1],
        }
    )

    scorer = fit_binned_feature_scorer(
        train,
        feature_columns=["feature_log_amount_ratio", "feature_breakout_pct"],
        label_column="label_high_trend",
        bins=2,
    )
    scored = score_with_binned_feature_scorer(train, scorer)

    assert scored["score"].iloc[0] < scored["score"].iloc[-1]
    assert scorer["baseline_rate"] == 0.5
    assert scorer["feature_contract"] == "ignition_close_point_in_time_v2"
    assert set(scorer["score_thresholds"]) == {"low_max", "high_min"}


def test_missing_values_use_a_separate_bin_from_low_values():
    train = pd.DataFrame(
        {
            "feature": [None, None, 0.0, 0.1, 0.9, 1.0],
            "label": [1, 1, 0, 0, 0, 0],
        }
    )
    scorer = fit_binned_feature_scorer(
        train,
        feature_columns=["feature"],
        label_column="label",
        bins=2,
    )
    scored = score_with_binned_feature_scorer(train, scorer)

    assert "-1" in scorer["features"]["feature"]["bin_lift"]
    assert scored.loc[0, "score"] != scored.loc[2, "score"]


def test_feature_contract_rejects_silent_partial_training():
    with pytest.raises(ValueError, match="feature_b"):
        validate_feature_contract(pd.DataFrame({"feature_a": [1.0]}), ["feature_a", "feature_b"])


def test_walk_forward_constant_scores_report_missing_correlation_without_warning():
    training = pd.DataFrame(
        {
            "period": ["p1"] * 4 + ["p2"] * 4,
            "constant_feature": [1.0] * 8,
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    _, summary = walk_forward_period_validation(
        training,
        feature_columns=["constant_feature"],
        label_column="label",
        bins=3,
    )

    assert pd.isna(summary.loc[0, "score_label_corr"])


def test_research_gate_requires_material_strength_in_every_fold():
    base = {
        "fixed_high_count": 10,
        "fixed_middle_count": 10,
        "fixed_low_count": 10,
    }
    weak = pd.DataFrame(
        [
            {**base, "fixed_bucket_spread": 0.001, "score_label_corr": 0.007},
            {**base, "fixed_bucket_spread": 0.030, "score_label_corr": 0.050},
        ]
    )
    strong = pd.DataFrame(
        [
            {**base, "fixed_bucket_spread": 0.006, "score_label_corr": 0.012},
            {**base, "fixed_bucket_spread": 0.020, "score_label_corr": 0.035},
        ]
    )

    assert not passes_research_gate(weak)
    assert passes_research_gate(strong)


def test_walk_forward_validation_uses_only_earlier_periods():
    rows = []
    for period in ["p1", "p2", "p3"]:
        for i in range(6):
            rows.append(
                {
                    "period": period,
                    "symbol": f"30000{i}",
                    "feature_log_amount_ratio": float(i),
                    "feature_breakout_pct": float(i) / 100.0,
                    "label_high_trend": int(i >= 3),
                }
            )
    training = pd.DataFrame(rows)

    scored, summary = walk_forward_period_validation(
        training,
        feature_columns=["feature_log_amount_ratio", "feature_breakout_pct"],
        label_column="label_high_trend",
        bins=3,
    )

    assert set(scored["validation_period"]) == {"p2", "p3"}
    assert scored["score"].notna().all()
    assert summary["validation_period"].tolist() == ["p2", "p3"]
    assert summary["training_periods"].tolist() == ["p1", "p1|p2"]
    assert set(scored.loc[scored["validation_period"].eq("p2"), "training_periods"]) == {"p1"}
    assert set(["top_quantile_positive_rate", "bottom_quantile_positive_rate"]).issubset(summary.columns)


def test_feature_diagnostics_rejects_features_with_empty_fixed_buckets():
    rows = []
    for period in ["p1", "p2", "p3"]:
        for i in range(20):
            rows.append(
                {
                    "period": period,
                    "stable_feature": float(i),
                    "noise_feature": float(i % 2),
                    "label": int(i >= 15),
                }
            )
    details, summary = walk_forward_feature_diagnostics(
        pd.DataFrame(rows),
        feature_columns=["stable_feature", "noise_feature"],
        label_column="label",
        bins=4,
    )

    assert set(details["feature"]) == {"stable_feature", "noise_feature"}
    assert set(summary["feature"]) == {"stable_feature", "noise_feature"}
    stable = summary.set_index("feature").loc["stable_feature"]
    assert not bool(stable["passes_exploratory_gate"])
    assert stable["min_middle_count"] == 0
