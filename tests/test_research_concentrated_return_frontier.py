from __future__ import annotations

import pandas as pd

from research_concentrated_return_frontier import (
    SCENARIOS,
    build_variants,
    evaluate_research_gates,
    inactive_tail_days,
    scenario_id_from_variant,
    segment_metrics,
    target_required_cagr,
)


def _base_config():
    return {
        "execution": {},
        "signal": {"top_n_long": 10, "top_n_short": 0},
        "portfolio": {"max_position_weight": 0.14, "leverage": 1.2, "allow_short": False, "short_exposure": 0.0},
        "risk": {"max_drawdown": 0.30},
        "cost": {"commission_bps": 1.0, "impact_bps": 0.7},
    }


def test_build_variants_pairs_each_concentration_with_three_stresses():
    variants = build_variants(_base_config())
    assert len(variants) == len(SCENARIOS) * 3
    for scenario in SCENARIOS:
        matches = [item for item in variants if item[0].startswith(scenario.scenario_id)]
        assert len(matches) == 3
        assert all(item[1]["signal"]["top_n_long"] == scenario.top_n_long for item in matches)
        assert all(item[1]["portfolio"]["max_position_weight"] == scenario.max_position_weight for item in matches)
        assert all(item[1]["risk"]["max_drawdown"] == 0.65 for item in matches)
        assert {item[2] for item in matches} == {1.0, 2.0}
        assert any(item[3] == 0.03 for item in matches)


def test_segment_metrics_uses_calendar_years():
    index = pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05"])
    equity = pd.Series([1.0, 1.1, 1.1, 1.21], index=index)
    rows = segment_metrics(equity)
    assert [row["year"] for row in rows] == [2025, 2026]
    assert all(abs(row["total_return"] - 0.1) < 1e-12 for row in rows)


def test_scenario_id_preserves_concentration_number():
    assert scenario_id_from_variant("concentrated_3_base_cost") == "concentrated_3"
    assert scenario_id_from_variant("concentrated_5_double_cost") == "concentrated_5"
    assert scenario_id_from_variant("reference_10_gap3") == "reference_10"


def test_target_required_cagr_matches_tenfold_equity_target():
    index = pd.to_datetime(["2020-01-01", "2025-01-01"])
    equity = pd.Series([1.0, 1.0], index=index)
    required = target_required_cagr(equity)
    assert 0.58 < required < 0.59


def test_inactive_tail_is_not_misread_as_stability():
    equity = pd.Series([1.0, 0.8, 0.7, 0.7, 0.7, 0.7])
    assert inactive_tail_days(equity) == 3


def test_research_gates_reject_failed_concentrated_curve():
    failures = evaluate_research_gates(
        total_return=-0.61,
        max_drawdown_value=-0.657,
        double_cost_return=-0.62,
        gap3_return=-0.66,
        positive_year_ratio=0.0,
        inactive_tail=500,
    )
    assert failures == [
        "non_positive_full_period_return",
        "drawdown_fail_line_breached",
        "non_positive_double_cost_return",
        "non_positive_gap3_return",
        "insufficient_positive_year_ratio",
        "inactive_tail_after_strategy_failure",
    ]
