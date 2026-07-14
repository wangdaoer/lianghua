import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from score_trend_ignition_shadow import (
    load_scorer_bundle,
    run_shadow_scoring,
    score_shadow_watchlists,
    validate_shadow_dates,
    write_shadow_outputs,
)


def _scorer() -> dict[str, object]:
    return {
        "schema_version": 2,
        "feature_contract": "ignition_close_point_in_time_v2",
        "label_column": "label_high_trend",
        "feature_columns": ["feature_close_over_ma20"],
        "baseline_rate": 0.05,
        "features": {
            "feature_close_over_ma20": {
                "edges": [-1.0, 0.0, 1.0],
                "missing_bin": -1,
                "bin_lift": {"-1": 0.0, "0": -0.01, "1": 0.02},
            }
        },
        "score_thresholds": {"low_max": 0.045, "high_min": 0.055},
        "training_end_date": "2025-06-12",
        "training_periods": ["2018_2020", "2021_2023", "2024_2026"],
    }


def _metadata() -> dict[str, object]:
    return {
        "schema_version": 2,
        "feature_contract": "ignition_close_point_in_time_v2",
        "training_end_date": "2025-06-12",
        "training_periods": ["2018_2020", "2021_2023", "2024_2026"],
        "feature_columns": ["feature_close_over_ma20"],
        "score_thresholds": {"low_max": 0.045, "high_min": 0.055},
        "passes_research_gate": True,
        "deployment_status": "research_only",
        "selection_status": "exploratory_posthoc",
    }


def _price_matrices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2025-07-01", periods=130)
    close_values = np.linspace(10.0, 11.0, len(dates))
    high_values = close_values * 1.001
    close_values[-1] = high_values[:-1].max() * 1.03
    high_values[-1] = close_values[-1] * 1.001
    amount_values = np.full(len(dates), 100_000_000.0)
    amount_values[-1] = 200_000_000.0
    close = pd.DataFrame({"000001": close_values}, index=dates)
    high = pd.DataFrame({"000001": high_values}, index=dates)
    amount = pd.DataFrame({"000001": amount_values}, index=dates)
    return close, high, amount


def test_load_scorer_bundle_preserves_research_boundary(tmp_path: Path) -> None:
    scorer_path = tmp_path / "scorer.json"
    summary_path = tmp_path / "summary.json"
    scorer_path.write_text(json.dumps(_scorer()), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "feature_columns": ["feature_close_over_ma20"],
                "passes_research_gate": True,
                "deployment_status": "research_only",
            }
        ),
        encoding="utf-8",
    )

    _, metadata = load_scorer_bundle(
        scorer_path,
        summary_path,
        selection_status="exploratory_posthoc",
    )

    assert metadata["training_end_date"] == "2025-06-12"
    assert metadata["passes_research_gate"] is True
    assert metadata["deployment_status"] == "research_only"
    assert metadata["selection_status"] == "exploratory_posthoc"


def test_load_scorer_bundle_rejects_non_point_in_time_contract(tmp_path: Path) -> None:
    scorer = _scorer()
    scorer["feature_contract"] = "future_peak_features"
    scorer_path = tmp_path / "scorer.json"
    summary_path = tmp_path / "summary.json"
    scorer_path.write_text(json.dumps(scorer), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "feature_columns": ["feature_close_over_ma20"],
                "passes_research_gate": True,
                "deployment_status": "research_only",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="point-in-time feature contract"):
        load_scorer_bundle(
            scorer_path,
            summary_path,
            selection_status="exploratory_posthoc",
        )


def test_validate_shadow_dates_rejects_training_period() -> None:
    watchlists = pd.DataFrame({"symbol": ["000001"], "asof_date": ["2025-06-12"]})

    with pytest.raises(ValueError, match="later than scorer training_end_date"):
        validate_shadow_dates(watchlists, _metadata())


def test_scores_only_rows_that_meet_frozen_ignition_contract() -> None:
    close, high, amount = _price_matrices()
    watchlists = pd.DataFrame(
        {
            "symbol": ["000001", "000001", "999999"],
            "asof_date": [close.index[-2], close.index[-1], close.index[-1]],
            "priority_score": [10.0, 20.0, 30.0],
        }
    )

    scored, coverage = score_shadow_watchlists(
        watchlists,
        close,
        high,
        amount,
        _scorer(),
        _metadata(),
    )

    assert scored["symbol"].tolist() == ["000001"]
    assert scored["asof_date"].tolist() == [close.index[-1]]
    assert scored["priority_score"].tolist() == [20.0]
    assert scored["trend_ignition_score_bucket"].tolist() == ["high"]
    assert scored["trend_ignition_selection_status"].tolist() == ["exploratory_posthoc"]
    assert scored["trend_ignition_ranking_modified"].tolist() == [False]
    assert scored["trade_instruction"].tolist() == [False]
    assert coverage["source_rows"] == 3
    assert coverage["eligible_rows"] == 1
    assert coverage["missing_history_symbols"] == ["999999"]


def test_write_shadow_outputs_creates_separate_research_watchlists(tmp_path: Path) -> None:
    close, high, amount = _price_matrices()
    source = pd.DataFrame(
        {
            "symbol": ["000001"],
            "asof_date": [close.index[-1]],
            "priority_score": [20.0],
        }
    )
    scored, coverage = score_shadow_watchlists(
        source,
        close,
        high,
        amount,
        _scorer(),
        _metadata(),
    )

    manifest = write_shadow_outputs(scored, coverage, tmp_path, _metadata())

    token = close.index[-1].strftime("%Y%m%d")
    daily_path = tmp_path / f"merged_priority_watchlist_{token}.csv"
    assert daily_path.exists()
    assert (tmp_path / "trend_ignition_shadow_scores.csv").exists()
    assert manifest["research_only"] is True
    assert manifest["trade_instruction"] is False
    assert manifest["ranking_modified"] is False
    assert manifest["source_watchlists_modified"] is False
    persisted = pd.read_csv(daily_path)
    assert persisted["trend_ignition_deployment_status"].tolist() == ["research_only"]


def test_empty_watchlist_run_does_not_require_market_data(tmp_path: Path) -> None:
    scorer_path = tmp_path / "scorer.json"
    summary_path = tmp_path / "summary.json"
    watchlist_dir = tmp_path / "watchlists"
    output_dir = tmp_path / "output"
    watchlist_dir.mkdir()
    scorer_path.write_text(json.dumps(_scorer()), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "feature_columns": ["feature_close_over_ma20"],
                "passes_research_gate": True,
                "deployment_status": "research_only",
            }
        ),
        encoding="utf-8",
    )

    manifest = run_shadow_scoring(
        tmp_path / "missing_market_data.csv",
        watchlist_dir,
        scorer_path,
        summary_path,
        output_dir,
        start="2026-07-14",
        end="2026-07-14",
    )

    assert manifest["coverage"]["source_rows"] == 0
    assert manifest["coverage"]["eligible_rows"] == 0
    assert (output_dir / "manifest.json").exists()
