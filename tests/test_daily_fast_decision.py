import json
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from quant_etf_lab.cli import _publish_fast_decision_for_pipeline, build_parser
from quant_etf_lab.daily_fast_decision import run_daily_fast_decision


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_inputs(root: Path) -> Path:
    research = root / "outputs" / "research"
    market = root / "market.csv"
    market.write_text("code,close\n000001,10\n", encoding="utf-8")
    _write_json(
        research / "daily_pipeline_latest" / "daily_pipeline_snapshot.json",
        {"as_of_date": "2026-07-20", "data_freshness_status": "fresh_enough"},
    )
    _write_json(
        research / "latest_dashboard" / "latest_dashboard_snapshot.json",
        {
            "as_of_date": "2026-07-20",
            "data_freshness_status": "fresh_enough",
            "sentiment_state": "warm",
            "dashboard_posture": "research_ready",
        },
    )
    _write_json(
        research / "paper_account_latest" / "metrics.json",
        {
            "latest_date": "2026-07-20",
            "latest_regime": "risk_on",
            "latest_core_weight": 0.7,
            "latest_satellite_weight": 0.3,
            "latest_cash_weight": 0.0,
            "current_drawdown": -0.05,
            "total_return": 0.4,
            "cagr": 0.2,
            "sharpe": 1.1,
        },
    )
    paper = research / "paper_account_latest"
    pd.DataFrame(
        [
            {"date": "2026-07-20", "layer": "core", "code": "000001", "name": "保留", "portfolio_target_weight": 0.2},
            {"date": "2026-07-20", "layer": "core", "code": "000002", "name": "剔除", "portfolio_target_weight": 0.3},
        ]
    ).to_csv(paper / "stock_targets.csv", index=False)
    pd.DataFrame(
        [
            {"date": "2026-07-20", "code": "000002", "name": "剔除", "original_review_stage": "review_required", "observation_excluded": True},
        ]
    ).to_csv(paper / "stock_target_review.csv", index=False)
    momentum_dir = research / "momentum_focus_latest"
    _write_json(
        momentum_dir / "momentum_focus_snapshot.json",
        {
            "as_of_date": "2026-07-20",
            "trade_date": "2026-07-20",
            "candidate_count": 2,
            "source_kind": "daily_market_data_csv",
            "source_path": str(market),
        },
    )
    pd.DataFrame(
        [
            {"as_of_date": "2026-07-20", "code": "000002", "name": "剔除", "focus_score": 99.0},
            {"as_of_date": "2026-07-20", "code": "000003", "name": "候选", "focus_score": 88.0},
        ]
    ).to_csv(momentum_dir / "momentum_focus_candidates.csv", index=False)
    _write_json(
        research / "live_preflight_latest" / "live_preflight_snapshot.json",
        {"as_of_date": "2026-07-20", "decision": "ready", "blocking_items": [], "monitor_items": []},
    )
    return research


def test_daily_fast_decision_filters_review_required_and_writes_outputs(tmp_path: Path) -> None:
    research = _build_inputs(tmp_path)
    result = run_daily_fast_decision(project_root=tmp_path, research_dir=research)

    assert result.snapshot["cache_hit"] is False
    assert result.snapshot["status"] == "ready"
    assert result.snapshot["observation_target_count"] == 1
    assert result.snapshot["review_excluded_codes"] == ["000002"]
    assert [row["code"] for row in result.snapshot["observation_targets"]] == ["000001"]
    assert [row["code"] for row in result.snapshot["momentum_candidates"]] == ["000003"]
    assert result.snapshot_path.exists()
    assert "每日快速决策摘要" in result.report_path.read_text(encoding="utf-8")


def test_daily_fast_decision_hits_cache_when_inputs_are_unchanged(tmp_path: Path) -> None:
    research = _build_inputs(tmp_path)
    first = run_daily_fast_decision(project_root=tmp_path, research_dir=research)
    second = run_daily_fast_decision(project_root=tmp_path, research_dir=research)

    assert first.snapshot["cache_hit"] is False
    assert second.snapshot["cache_hit"] is True
    assert second.snapshot["input_fingerprint"] == first.snapshot["input_fingerprint"]


def test_daily_fast_decision_invalidates_cache_when_source_changes(tmp_path: Path) -> None:
    research = _build_inputs(tmp_path)
    first = run_daily_fast_decision(project_root=tmp_path, research_dir=research)
    candidates_path = research / "momentum_focus_latest" / "momentum_focus_candidates.csv"
    with candidates_path.open("a", encoding="utf-8") as handle:
        handle.write("2026-07-20,000004,新增,77\n")
    second = run_daily_fast_decision(project_root=tmp_path, research_dir=research)

    assert second.snapshot["cache_hit"] is False
    assert second.snapshot["input_fingerprint"] != first.snapshot["input_fingerprint"]


def test_daily_fast_decision_marks_misaligned_inputs_stale(tmp_path: Path) -> None:
    research = _build_inputs(tmp_path)
    dashboard_path = research / "latest_dashboard" / "latest_dashboard_snapshot.json"
    payload = json.loads(dashboard_path.read_text(encoding="utf-8"))
    payload["as_of_date"] = "2026-07-19"
    _write_json(dashboard_path, payload)

    result = run_daily_fast_decision(project_root=tmp_path, research_dir=research)

    assert result.snapshot["status"] == "stale_inputs"
    assert result.snapshot["input_dates_aligned"] is False
    assert "fast_decision_input_dates_not_aligned" in result.snapshot["blocking_items"]


def test_daily_fast_decision_cli_defaults() -> None:
    args = build_parser().parse_args(["daily-fast-decision"])

    assert args.command == "daily-fast-decision"
    assert args.top_candidates == 20
    assert args.force is False


def test_daily_pipeline_fast_decision_defaults_and_opt_in() -> None:
    defaults = build_parser().parse_args(["daily-pipeline"])
    enabled = build_parser().parse_args(
        [
            "daily-pipeline",
            "--publish-fast-decision",
            "--fast-decision-output-dir",
            "outputs/test/fast",
            "--fast-decision-research-dir",
            "outputs/test/research",
            "--fast-decision-top-candidates",
            "7",
        ]
    )

    assert defaults.publish_fast_decision is False
    assert defaults.fast_decision_top_candidates == 20
    assert enabled.publish_fast_decision is True
    assert enabled.fast_decision_output_dir == "outputs/test/fast"
    assert enabled.fast_decision_research_dir == "outputs/test/research"
    assert enabled.fast_decision_top_candidates == 7


def test_pipeline_fast_decision_publisher_runs_once_and_is_non_blocking(tmp_path: Path) -> None:
    args = SimpleNamespace(
        fast_decision_output_dir=str(tmp_path / "fast"),
        fast_decision_research_dir=str(tmp_path / "research"),
        fast_decision_top_candidates=7,
    )
    expected = SimpleNamespace(
        output_dir=tmp_path / "fast",
        report_path=tmp_path / "fast" / "daily_fast_decision.md",
        snapshot={"cache_hit": True},
    )
    stdout = StringIO()
    stderr = StringIO()

    with patch("quant_etf_lab.cli.process_lock", return_value=nullcontext()), patch(
        "quant_etf_lab.cli.run_daily_fast_decision", return_value=expected
    ) as run_mock, redirect_stdout(stdout):
        assert _publish_fast_decision_for_pipeline(args) is expected
    run_mock.assert_called_once_with(
        project_root=Path("."),
        output_dir=tmp_path / "fast",
        research_dir=tmp_path / "research",
        top_candidates=7,
    )
    assert "Fast decision cache hit: True" in stdout.getvalue()

    with patch("quant_etf_lab.cli.process_lock", return_value=nullcontext()), patch(
        "quant_etf_lab.cli.run_daily_fast_decision", side_effect=ValueError("broken input")
    ), redirect_stderr(stderr):
        assert _publish_fast_decision_for_pipeline(args) is None
    assert "failed; continuing: broken input" in stderr.getvalue()
