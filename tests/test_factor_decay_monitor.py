from pathlib import Path

import pandas as pd

from monitor_factor_decay import (
    classify_overall_status,
    evaluate_preregistered_replacements,
    summarize_factor_competition,
    summarize_factor_decay,
    update_monitor_history,
    update_factor_replacement_preregistration,
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


def _competition_rows() -> pd.DataFrame:
    rows = []
    for factor, reference_ic, recent_ic in (
        ("liquidity_stability_20", 0.10, -0.02),
        ("momentum_60", 0.05, 0.08),
    ):
        for horizon in (5, 10, 20):
            for date in pd.date_range("2023-01-01", periods=12, freq="D"):
                rows.append(
                    {
                        "date": date,
                        "factor": factor,
                        "direction": 1.0,
                        "horizon": horizon,
                        "observations": 100,
                        "rank_ic": reference_ic,
                        "top_bottom_spread": reference_ic / 10.0,
                    }
                )
            for date in pd.date_range("2026-01-01", periods=12, freq="D"):
                rows.append(
                    {
                        "date": date,
                        "factor": factor,
                        "direction": 1.0,
                        "horizon": horizon,
                        "observations": 100,
                        "rank_ic": recent_ic,
                        "top_bottom_spread": recent_ic / 10.0,
                    }
                )
    return pd.DataFrame(rows)


def test_factor_competition_shortlists_supported_improvement():
    result = summarize_factor_competition(
        _competition_rows(),
        selection_start=pd.Timestamp("2023-01-01"),
        selection_end=pd.Timestamp("2025-12-31"),
        asof_date=pd.Timestamp("2026-02-01"),
        recent_signal_days=12,
        factor_directions={
            "liquidity_stability_20": 1.0,
            "momentum_60": 1.0,
        },
    ).set_index("factor")

    assert result.loc["liquidity_stability_20", "candidate_status"] == "incumbent"
    assert result.loc["momentum_60", "support_status"] == "supported"
    assert result.loc["momentum_60", "candidate_status"] == "shortlist"
    assert result.loc["momentum_60", "recent_ic_advantage_vs_incumbent"] > 0.0


def test_factor_replacement_preregistration_is_frozen(tmp_path: Path):
    competition = summarize_factor_competition(
        _competition_rows(),
        selection_start=pd.Timestamp("2023-01-01"),
        selection_end=pd.Timestamp("2025-12-31"),
        asof_date=pd.Timestamp("2026-02-01"),
        recent_signal_days=12,
        factor_directions={
            "liquidity_stability_20": 1.0,
            "momentum_60": 1.0,
        },
    )
    path = tmp_path / "factor_replacement_preregistration.json"

    first = update_factor_replacement_preregistration(
        path,
        competition,
        asof_date=pd.Timestamp("2026-02-01"),
        incumbent_status="direction_reversal",
    )
    changed = competition.copy()
    changed.loc[changed["factor"].eq("momentum_60"), "candidate_status"] = "not_shortlisted"
    second = update_factor_replacement_preregistration(
        path,
        changed,
        asof_date=pd.Timestamp("2026-03-01"),
        incumbent_status="direction_reversal",
    )

    assert first is not None
    assert second is not None
    assert first == second
    assert first["candidate_factors"] == ["momentum_60"]
    assert first["validation_start_date"] == "2026-02-02"
    assert first["automatic_model_change"] is False


def test_preregistered_tracking_uses_only_future_mature_samples():
    preregistration = {
        "validation_start_date": "2026-07-15",
        "incumbent_factor": "liquidity_stability_20",
        "candidate_factors": ["momentum_60"],
        "horizons": [5, 10],
        "target_mature_signal_days": 2,
        "gates": {
            "min_mean_rank_ic": 0.01,
            "min_positive_ic_ratio": 0.55,
            "min_rank_ic_advantage_vs_incumbent": 0.0,
        },
    }
    rows = []
    for factor, rank_ic in (
        ("liquidity_stability_20", 0.02),
        ("momentum_60", 0.08),
    ):
        for horizon in (5, 10):
            rows.append(
                {
                    "date": pd.Timestamp("2026-07-14"),
                    "factor": factor,
                    "horizon": horizon,
                    "rank_ic": -1.0,
                }
            )
            for date in pd.to_datetime(["2026-07-15", "2026-07-16"]):
                rows.append(
                    {
                        "date": date,
                        "factor": factor,
                        "horizon": horizon,
                        "rank_ic": rank_ic,
                    }
                )

    tracking_rows, result = evaluate_preregistered_replacements(
        pd.DataFrame(rows), preregistration
    )

    candidate = next(
        row for row in result["factors"] if row["factor"] == "momentum_60"
    )
    assert result["status"] == "ready_for_research_review"
    assert candidate["minimum_mature_signal_days"] == 2
    assert candidate["mean_rank_ic"] == 0.08
    assert candidate["decision"] == "ready_for_research_review"
    assert len(tracking_rows) == 4
