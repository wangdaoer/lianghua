from __future__ import annotations

import pandas as pd

from quant_etf_lab.paper_account import apply_review_required_observation_exclusion


def test_review_required_rows_are_removed_from_observation_targets() -> None:
    targets = pd.DataFrame(
        [
            {
                "layer": "core",
                "code": "000001",
                "name": "drawdown",
                "portfolio_target_weight": 0.6,
                "portfolio_target_value": 6000.0,
                "target_action": "target_hold",
                "trigger_monitor_status": "not_in_latest_trigger_report",
                "tracking_excluded": False,
            },
            {
                "layer": "core",
                "code": "000002",
                "name": "routine",
                "portfolio_target_weight": 0.4,
                "portfolio_target_value": 4000.0,
                "target_action": "target_hold",
                "trigger_monitor_status": "not_in_latest_trigger_report",
                "tracking_excluded": False,
            },
            {
                "layer": "trigger",
                "code": "000003",
                "name": "trigger",
                "portfolio_target_weight": 0.0,
                "portfolio_target_value": 0.0,
                "target_action": "trigger_watch_candidate",
                "trigger_monitor_status": "matched_latest_trigger",
                "tracking_excluded": False,
            },
        ]
    )
    review = pd.DataFrame(
        [
            {
                "review_rank": 1,
                "layer": "core",
                "code": "000001",
                "review_stage": "review_required",
                "recommended_review": "review drawdown",
            },
            {
                "review_rank": 2,
                "layer": "trigger",
                "code": "000003",
                "review_stage": "review_required",
                "recommended_review": "review trigger",
            },
            {
                "review_rank": 3,
                "layer": "core",
                "code": "000002",
                "review_stage": "routine",
                "recommended_review": "routine",
            },
        ]
    )

    observation, target_payload, audit, review_payload = apply_review_required_observation_exclusion(
        targets,
        {"status": "ok", "stock_target_count": 3},
        review,
        {"status": "ok", "review_required_count": 2, "drawdown_review_count": 1, "trigger_review_count": 1},
    )

    assert observation["code"].tolist() == ["000002"]
    assert target_payload["source_stock_target_count"] == 3
    assert target_payload["stock_target_count"] == 1
    assert target_payload["active_stock_target_count"] == 1
    assert target_payload["review_required_excluded_count"] == 2
    assert target_payload["review_required_excluded_codes"] == ["000001", "000003"]
    assert target_payload["total_portfolio_target_weight"] == 0.4
    assert review_payload["review_required_count"] == 0
    assert review_payload["excluded_review_required_count"] == 2
    assert review_payload["drawdown_review_count"] == 0
    assert review_payload["trigger_review_count"] == 0
    assert review_payload["original_drawdown_review_count"] == 1
    assert review_payload["original_trigger_review_count"] == 1
    excluded = audit.loc[audit["observation_excluded"]]
    assert set(excluded["code"]) == {"000001", "000003"}
    assert set(excluded["review_stage"]) == {"excluded"}
    assert set(excluded["original_review_stage"]) == {"review_required"}
