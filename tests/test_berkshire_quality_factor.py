from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser


def test_berkshire_quality_factor_cli() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "berkshire-quality-factor",
            "--candidates-path",
            "outputs/research/paper_account_latest/stock_targets.csv",
            "--fundamentals-path",
            "data/research/berkshire_fundamentals.csv",
            "--qualitative-path",
            "data/research/berkshire_qualitative.csv",
            "--output-dir",
            "outputs/research/berkshire_quality_factor",
            "--top-n",
            "25",
        ]
    )

    assert args.command == "berkshire-quality-factor"
    assert args.candidates_path.endswith("stock_targets.csv")
    assert args.fundamentals_path == "data/research/berkshire_fundamentals.csv"
    assert args.qualitative_path == "data/research/berkshire_qualitative.csv"
    assert args.output_dir == "outputs/research/berkshire_quality_factor"
    assert args.top_n == 25


def test_berkshire_quality_factor_scores_quality_company_above_vetoed_candidate(tmp_path: Path) -> None:
    from quant_etf_lab.berkshire_quality_factor import run_berkshire_quality_factor

    candidates = tmp_path / "candidates.csv"
    fundamentals = tmp_path / "fundamentals.csv"
    qualitative = tmp_path / "qualitative.csv"
    output_dir = tmp_path / "berkshire_quality"

    pd.DataFrame(
        [
            {"code": "000001", "name": "QualityCompounder", "priority_score": 50.0},
            {"code": "000002", "name": "DilutiveStory", "priority_score": 95.0},
        ]
    ).to_csv(candidates, index=False)
    pd.DataFrame(
        [
            {
                "code": "000001",
                "roe_10y_avg": 0.18,
                "fcf_5y_cumulative": 120.0,
                "interest_coverage": 8.0,
                "gross_margin_5y_avg": 0.42,
                "ocf_to_net_income_5y_avg": 1.15,
                "net_margin_10y_avg": 0.16,
                "share_dilution_5y": 0.03,
            },
            {
                "code": "000002",
                "roe_10y_avg": 0.04,
                "fcf_5y_cumulative": -10.0,
                "interest_coverage": 1.1,
                "gross_margin_5y_avg": 0.10,
                "ocf_to_net_income_5y_avg": 0.45,
                "net_margin_10y_avg": 0.02,
                "share_dilution_5y": 0.35,
            },
        ]
    ).to_csv(fundamentals, index=False)
    pd.DataFrame(
        [
            {
                "code": "000001",
                "circle_of_competence_score": 4,
                "moat_score": 5,
                "management_score": 4,
                "safety_margin_score": 4,
                "thesis_clarity_score": 5,
                "red_flag_count": 0,
                "data_confidence": "A",
            },
            {
                "code": "000002",
                "circle_of_competence_score": 3,
                "moat_score": 2,
                "management_score": 1,
                "safety_margin_score": 3,
                "thesis_clarity_score": 2,
                "red_flag_count": 1,
                "data_confidence": "B",
            },
        ]
    ).to_csv(qualitative, index=False)

    result = run_berkshire_quality_factor(
        candidates_path=candidates,
        fundamentals_path=fundamentals,
        qualitative_path=qualitative,
        output_dir=output_dir,
        top_n=10,
    )

    scores = pd.read_csv(result.scores_path, dtype={"code": str})
    by_code = scores.set_index("code")
    assert result.snapshot["status"] == "ok"
    assert result.snapshot["research_only"] is True
    assert by_code.loc["000001", "berkshire_bucket"] == "berkshire_quality_priority"
    assert by_code.loc["000002", "berkshire_bucket"] == "berkshire_veto"
    assert by_code.loc["000001", "berkshire_quality_score"] > by_code.loc["000002", "berkshire_quality_score"]
    assert by_code.loc["000002", "berkshire_hard_veto"] is True or bool(by_code.loc["000002", "berkshire_hard_veto"])
    payload = json.loads(result.snapshot_path.read_text(encoding="utf-8-sig"))
    assert payload["broker_action"] == "none"
    assert result.report_path.exists()


def test_berkshire_quality_factor_without_fundamentals_writes_template(tmp_path: Path) -> None:
    from quant_etf_lab.berkshire_quality_factor import run_berkshire_quality_factor

    candidates = tmp_path / "candidates.csv"
    output_dir = tmp_path / "berkshire_quality"
    pd.DataFrame(
        [
            {"security_code": "300001", "security_name": "NeedsFundamentalMap", "target_weight": 0.02},
        ]
    ).to_csv(candidates, index=False)

    result = run_berkshire_quality_factor(
        candidates_path=candidates,
        output_dir=output_dir,
        top_n=10,
    )

    scores = pd.read_csv(result.scores_path, dtype={"code": str})
    template = pd.read_csv(result.template_path, dtype={"code": str})
    assert result.snapshot["status"] == "needs_fundamental_data"
    assert result.snapshot["mapped_candidate_count"] == 0
    assert scores["berkshire_bucket"].tolist() == ["needs_fundamental_research"]
    assert template["code"].tolist() == ["300001"]
    assert "roe_10y_avg" in template.columns
    assert "moat_score" in template.columns
    assert "Fill Berkshire quality" in template["research_notes"].iloc[0]
