from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser


def test_industry_chain_factor_cli() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "industry-chain-factor",
            "--candidates-path",
            "outputs/research/chip_reversal_daily_candidates_20260622/chip_reversal_daily_candidates.csv",
            "--chain-map-path",
            "data/research/industry_chain_map.csv",
            "--market-snapshot-path",
            "D:/codex/daily-market-data/ths_exports/normalized/ths_hs_a_share_2026-06-22.csv",
            "--output-dir",
            "outputs/research/industry_chain_factor",
            "--top-n",
            "20",
        ]
    )

    assert args.command == "industry-chain-factor"
    assert args.candidates_path.endswith("chip_reversal_daily_candidates.csv")
    assert args.chain_map_path == "data/research/industry_chain_map.csv"
    assert args.market_snapshot_path.endswith("ths_hs_a_share_2026-06-22.csv")
    assert args.output_dir == "outputs/research/industry_chain_factor"
    assert args.top_n == 20


def test_industry_chain_factor_scores_core_bottleneck_above_unmapped_candidate(tmp_path: Path) -> None:
    from quant_etf_lab.industry_chain_factor import run_industry_chain_factor

    candidates = tmp_path / "candidates.csv"
    chain_map = tmp_path / "industry_chain_map.csv"
    market_snapshot = tmp_path / "market_snapshot.csv"
    output_dir = tmp_path / "industry_chain_factor"

    pd.DataFrame(
        [
            {
                "candidate_rank": 1,
                "date": "2026-06-22",
                "code": "000001",
                "name": "MappedCore",
                "priority_score": 60.0,
                "broker_action": "none",
                "research_only": True,
            },
            {
                "candidate_rank": 2,
                "date": "2026-06-22",
                "code": "000002",
                "name": "HotUnmapped",
                "priority_score": 90.0,
                "broker_action": "none",
                "research_only": True,
            },
        ]
    ).to_csv(candidates, index=False)
    pd.DataFrame(
        [
            {
                "code": "000001",
                "industry_chain": "ai_semiconductor",
                "chain_node": "advanced_packaging",
                "node_role": "bottleneck",
                "demand_certainty": 0.90,
                "supply_constraint": 0.95,
                "attention_gap": 0.80,
                "value_capture": 0.85,
                "catalyst_strength": 0.70,
                "substitution_risk": 0.10,
                "crowding_risk": 0.10,
                "valuation_risk": 0.20,
            }
        ]
    ).to_csv(chain_map, index=False)
    pd.DataFrame(
        [
            {"security_code": "000001", "market_cap": 50_000_000_000, "turnover_rate": 1.2, "change_ratio": 1.0},
            {"security_code": "000002", "market_cap": 300_000_000_000, "turnover_rate": 12.0, "change_ratio": 18.0},
        ]
    ).to_csv(market_snapshot, index=False)

    result = run_industry_chain_factor(
        candidates_path=candidates,
        chain_map_path=chain_map,
        market_snapshot_path=market_snapshot,
        output_dir=output_dir,
        top_n=10,
    )

    scores = pd.read_csv(result.scores_path, dtype={"code": str})
    assert result.snapshot["status"] == "ok"
    assert result.snapshot["research_only"] is True
    assert result.snapshot["mapped_candidate_count"] == 1
    assert scores.iloc[0]["code"] == "000001"
    assert scores.iloc[0]["factor_bucket"] == "core_chain_priority"
    assert scores.loc[scores["code"] == "000001", "industry_chain_factor_score"].iloc[0] > scores.loc[
        scores["code"] == "000002", "industry_chain_factor_score"
    ].iloc[0]
    payload = json.loads(result.snapshot_path.read_text(encoding="utf-8-sig"))
    assert payload["broker_action"] == "none"
    assert result.template_path.exists()
    assert result.report_path.exists()


def test_industry_chain_factor_scores_serenity_style_bottleneck_evidence(tmp_path: Path) -> None:
    from quant_etf_lab.industry_chain_factor import run_industry_chain_factor

    candidates = tmp_path / "candidates.csv"
    chain_map = tmp_path / "industry_chain_map.csv"
    output_dir = tmp_path / "industry_chain_factor"

    pd.DataFrame(
        [
            {"date": "2026-06-23", "code": "300001", "name": "UpstreamChoke", "priority_score": 50.0},
            {"date": "2026-06-23", "code": "300002", "name": "HotAssembler", "priority_score": 95.0},
        ]
    ).to_csv(candidates, index=False)
    pd.DataFrame(
        [
            {
                "code": "300001",
                "industry_chain": "ai_optical",
                "chain_node": "inp_substrate",
                "node_role": "bottleneck",
                "demand_certainty": 0.90,
                "supply_constraint": 0.95,
                "attention_gap": 0.75,
                "value_capture": 0.85,
                "catalyst_strength": 0.70,
                "upstream_dependency": 0.95,
                "customer_validation": 0.80,
                "qualification_cycle": 0.90,
                "pricing_power": 0.85,
                "substitution_risk": 0.05,
                "crowding_risk": 0.10,
                "valuation_risk": 0.15,
                "dilution_risk": 0.05,
            },
            {
                "code": "300002",
                "industry_chain": "ai_optical",
                "chain_node": "module_assembly",
                "node_role": "downstream",
                "demand_certainty": 0.85,
                "supply_constraint": 0.30,
                "attention_gap": 0.15,
                "value_capture": 0.35,
                "catalyst_strength": 0.50,
                "upstream_dependency": 0.20,
                "customer_validation": 0.35,
                "qualification_cycle": 0.25,
                "pricing_power": 0.20,
                "substitution_risk": 0.55,
                "crowding_risk": 0.70,
                "valuation_risk": 0.50,
                "dilution_risk": 0.20,
            },
        ]
    ).to_csv(chain_map, index=False)

    result = run_industry_chain_factor(
        candidates_path=candidates,
        chain_map_path=chain_map,
        output_dir=output_dir,
        top_n=10,
    )

    scores = pd.read_csv(result.scores_path, dtype={"code": str})
    by_code = scores.set_index("code")
    assert by_code.loc["300001", "serenity_bottleneck_bucket"] == "serenity_core_bottleneck"
    assert by_code.loc["300001", "serenity_bottleneck_score"] > by_code.loc["300002", "serenity_bottleneck_score"]
    assert by_code.loc["300001", "industry_chain_factor_score"] > by_code.loc["300002", "industry_chain_factor_score"]
    assert "serenity_bottleneck_components" in result.snapshot["factor_components"]


def test_industry_chain_factor_does_not_penalize_market_cap_above_1500_yi() -> None:
    from quant_etf_lab.industry_chain_factor import _market_fit_score

    score = _market_fit_score(
        pd.Series(
            {
                "market_cap": 1800.0,
                "turnover_rate": 1.0,
                "change_ratio": 1.0,
            }
        )
    )

    assert score >= 0.7


def test_industry_chain_factor_without_map_writes_research_template(tmp_path: Path) -> None:
    from quant_etf_lab.industry_chain_factor import run_industry_chain_factor

    candidates = tmp_path / "candidates.csv"
    output_dir = tmp_path / "industry_chain_factor"
    pd.DataFrame(
        [
            {"date": "2026-06-22", "code": "300001", "name": "NeedsMap", "priority_score": 50.0},
        ]
    ).to_csv(candidates, index=False)

    result = run_industry_chain_factor(
        candidates_path=candidates,
        output_dir=output_dir,
        top_n=10,
    )

    scores = pd.read_csv(result.scores_path, dtype={"code": str})
    template = pd.read_csv(result.template_path, dtype={"code": str})
    assert result.snapshot["status"] == "needs_chain_map"
    assert result.snapshot["mapped_candidate_count"] == 0
    assert scores["factor_bucket"].tolist() == ["needs_chain_research"]
    assert template["code"].tolist() == ["300001"]
    assert "demand_certainty" in template.columns
    assert "chain_node" in template.columns
    assert "Fill chain node" in template["research_notes"].iloc[0]
