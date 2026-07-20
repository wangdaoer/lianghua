from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser


def test_satellite_filter_model_cli() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "satellite-filter-model",
            "--candidates-path",
            "outputs/research/paper_account_latest/stock_target_review.csv",
            "--filter-candidates-path",
            "outputs/research/satellite_filter_research_20260622/satellite_filter_candidates.csv",
            "--output-dir",
            "outputs/research/satellite_filter_model",
            "--industry-chain-scores-path",
            "outputs/research/industry_chain_factor_latest/industry_chain_factor_scores.csv",
            "--min-industry-chain-score",
            "60",
            "--min-bottleneck-score",
            "55",
        ]
    )

    assert args.command == "satellite-filter-model"
    assert args.candidates_path == "outputs/research/paper_account_latest/stock_target_review.csv"
    assert args.filter_candidates_path.endswith("satellite_filter_candidates.csv")
    assert args.output_dir == "outputs/research/satellite_filter_model"
    assert args.industry_chain_scores_path.endswith("industry_chain_factor_scores.csv")
    assert args.min_industry_chain_score == 60.0
    assert args.min_bottleneck_score == 55.0


def test_satellite_filter_model_prefers_confirmed_then_watch_rules(tmp_path: Path) -> None:
    from quant_etf_lab.satellite_filter_research import run_satellite_filter_model

    candidates_path = tmp_path / "candidates.csv"
    filters_path = tmp_path / "satellite_filter_candidates.csv"
    output_dir = tmp_path / "satellite_filter_model"
    pd.DataFrame(
        [
            {
                "date": "2026-06-22",
                "layer": "satellite",
                "code": "300001",
                "name": "Confirmed",
                "review_bucket": "trigger_review",
                "action_code": "review_required_pending",
                "manual_status_normalized": "unreviewed",
            },
            {
                "date": "2026-06-22",
                "layer": "satellite",
                "code": "300002",
                "name": "Watch",
                "review_bucket": "suppressed_layer_review",
                "action_code": "",
                "manual_status_normalized": "reviewed",
            },
            {
                "date": "2026-06-22",
                "layer": "satellite",
                "code": "300003",
                "name": "Hold",
                "review_bucket": "routine_review",
                "action_code": "",
                "manual_status_normalized": "reviewed",
            },
            {
                "date": "2026-06-22",
                "layer": "core",
                "code": "600001",
                "name": "Core",
                "review_bucket": "trigger_review",
                "action_code": "review_required_pending",
                "manual_status_normalized": "unreviewed",
            },
        ]
    ).to_csv(candidates_path, index=False)
    pd.DataFrame(
        [
            {
                "dimension": "review_bucket",
                "group_value": "trigger_review",
                "candidate_status": "confirmed",
                "passed_horizons": "5d,10d",
                "best_horizon": "10d",
                "best_sample_count": 8,
                "best_win_rate": 0.75,
                "best_avg_return": 0.08,
                "best_min_return": -0.04,
            },
            {
                "dimension": "review_bucket",
                "group_value": "suppressed_layer_review",
                "candidate_status": "watch",
                "passed_horizons": "5d",
                "best_horizon": "5d",
                "best_sample_count": 6,
                "best_win_rate": 0.70,
                "best_avg_return": 0.05,
                "best_min_return": -0.03,
            },
        ]
    ).to_csv(filters_path, index=False)

    result = run_satellite_filter_model(
        candidates_path=candidates_path,
        filter_candidates_path=filters_path,
        output_dir=output_dir,
    )

    scores = pd.read_csv(result.scores_path, dtype={"code": str})
    decisions = dict(zip(scores["code"], scores["satellite_filter_decision"]))
    assert result.snapshot["status"] == "allow_candidates"
    assert result.snapshot["satellite_candidate_count"] == 3
    assert result.snapshot["allow_count"] == 1
    assert result.snapshot["watch_count"] == 1
    assert result.snapshot["hold_count"] == 1
    assert decisions["300001"] == "allow_research_trial"
    assert decisions["300002"] == "watch_only"
    assert decisions["300003"] == "hold_satellite"
    assert "600001" not in decisions
    payload = json.loads(result.snapshot_path.read_text(encoding="utf-8-sig"))
    assert payload["broker_action"] == "none"
    assert payload["research_only"] is True
    assert result.report_path.exists()


def test_satellite_filter_model_reports_no_satellite_candidates(tmp_path: Path) -> None:
    from quant_etf_lab.satellite_filter_research import run_satellite_filter_model

    candidates_path = tmp_path / "candidates.csv"
    filters_path = tmp_path / "satellite_filter_candidates.csv"
    pd.DataFrame(
        [{"date": "2026-06-22", "layer": "core", "code": "600001", "name": "Core"}]
    ).to_csv(candidates_path, index=False)
    pd.DataFrame(
        [
            {
                "dimension": "review_bucket",
                "group_value": "trigger_review",
                "candidate_status": "confirmed",
                "passed_horizons": "5d,10d",
                "best_sample_count": 5,
                "best_win_rate": 0.7,
                "best_avg_return": 0.05,
            }
        ]
    ).to_csv(filters_path, index=False)

    result = run_satellite_filter_model(
        candidates_path=candidates_path,
        filter_candidates_path=filters_path,
        output_dir=tmp_path / "satellite_filter_model",
    )

    assert result.snapshot["status"] == "no_satellite_candidates"
    assert result.snapshot["satellite_candidate_count"] == 0
    scores = pd.read_csv(result.scores_path)
    assert scores.empty


def test_satellite_filter_model_uses_industry_chain_gate_for_confirmed_rules(tmp_path: Path) -> None:
    from quant_etf_lab.satellite_filter_research import run_satellite_filter_model

    candidates_path = tmp_path / "candidates.csv"
    filters_path = tmp_path / "satellite_filter_candidates.csv"
    chain_scores_path = tmp_path / "industry_chain_factor_scores.csv"
    output_dir = tmp_path / "satellite_filter_model"

    pd.DataFrame(
        [
            {
                "date": "2026-06-23",
                "layer": "satellite",
                "code": "300001",
                "name": "ChainPass",
                "review_bucket": "trigger_review",
            },
            {
                "date": "2026-06-23",
                "layer": "satellite",
                "code": "300002",
                "name": "ChainWeak",
                "review_bucket": "trigger_review",
            },
            {
                "date": "2026-06-23",
                "layer": "satellite",
                "code": "300003",
                "name": "ChainOnly",
                "review_bucket": "routine_review",
            },
        ]
    ).to_csv(candidates_path, index=False)
    pd.DataFrame(
        [
            {
                "dimension": "review_bucket",
                "group_value": "trigger_review",
                "candidate_status": "confirmed",
                "passed_horizons": "5d,10d",
                "best_horizon": "10d",
                "best_sample_count": 8,
                "best_win_rate": 0.75,
                "best_avg_return": 0.08,
                "best_min_return": -0.04,
            }
        ]
    ).to_csv(filters_path, index=False)
    pd.DataFrame(
        [
            {
                "code": "300001",
                "industry_chain_factor_score": 78.0,
                "serenity_bottleneck_score": 82.0,
                "factor_bucket": "core_chain_priority",
                "serenity_bottleneck_bucket": "serenity_core_bottleneck",
            },
            {
                "code": "300002",
                "industry_chain_factor_score": 45.0,
                "serenity_bottleneck_score": 38.0,
                "factor_bucket": "chain_watch",
                "serenity_bottleneck_bucket": "serenity_low_conviction",
            },
            {
                "code": "300003",
                "industry_chain_factor_score": 76.0,
                "serenity_bottleneck_score": 73.0,
                "factor_bucket": "core_chain_priority",
                "serenity_bottleneck_bucket": "serenity_core_bottleneck",
            },
        ]
    ).to_csv(chain_scores_path, index=False)

    result = run_satellite_filter_model(
        candidates_path=candidates_path,
        filter_candidates_path=filters_path,
        industry_chain_scores_path=chain_scores_path,
        min_industry_chain_score=60.0,
        min_bottleneck_score=55.0,
        output_dir=output_dir,
    )

    scores = pd.read_csv(result.scores_path, dtype={"code": str})
    by_code = scores.set_index("code")
    assert by_code.loc["300001", "satellite_filter_decision"] == "allow_research_trial"
    assert by_code.loc["300001", "industry_chain_filter_status"] == "industry_chain_pass"
    assert by_code.loc["300002", "satellite_filter_decision"] == "watch_only"
    assert by_code.loc["300002", "satellite_filter_status"] == "industry_chain_gate_pending"
    assert by_code.loc["300003", "satellite_filter_decision"] == "watch_only"
    assert by_code.loc["300003", "satellite_filter_status"] == "industry_chain_only_watch"
    assert by_code.loc["300001", "satellite_filter_score"] > by_code.loc["300002", "satellite_filter_score"]
    assert by_code.loc["300003", "satellite_filter_score"] > by_code.loc["300002", "satellite_filter_score"]
    assert result.snapshot["industry_chain_gate_enabled"] is True
    assert result.snapshot["industry_chain_pass_count"] == 2


def test_satellite_filter_model_treats_chip_reversal_source_as_satellite(tmp_path: Path) -> None:
    from quant_etf_lab.satellite_filter_research import run_satellite_filter_model

    candidates_path = tmp_path / "chip_candidates.csv"
    filters_path = tmp_path / "satellite_filter_candidates.csv"
    chain_scores_path = tmp_path / "industry_chain_factor_scores.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-23",
                "source_sleeve": "chip_reversal_daily_proxy",
                "code": "603409",
                "name": "ChipSatellite",
                "review_bucket": "routine_review",
            }
        ]
    ).to_csv(candidates_path, index=False)
    pd.DataFrame(
        [
            {
                "dimension": "review_bucket",
                "group_value": "trigger_review",
                "candidate_status": "confirmed",
            }
        ]
    ).to_csv(filters_path, index=False)
    pd.DataFrame(
        [
            {
                "code": "603409",
                "industry_chain_factor_score": 70.0,
                "serenity_bottleneck_score": 65.0,
                "factor_bucket": "core_chain_priority",
                "serenity_bottleneck_bucket": "serenity_watch",
            }
        ]
    ).to_csv(chain_scores_path, index=False)

    result = run_satellite_filter_model(
        candidates_path=candidates_path,
        filter_candidates_path=filters_path,
        industry_chain_scores_path=chain_scores_path,
        output_dir=tmp_path / "satellite_filter_model",
    )

    scores = pd.read_csv(result.scores_path, dtype={"code": str})
    assert result.snapshot["satellite_candidate_count"] == 1
    assert scores.loc[0, "layer"] == "satellite"
    assert scores.loc[0, "satellite_filter_decision"] == "watch_only"
    assert scores.loc[0, "satellite_filter_status"] == "industry_chain_only_watch"
