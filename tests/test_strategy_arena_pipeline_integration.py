import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from run_daily_model_pipeline import PipelineConfig, PipelineStep, build_daily_run_state


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASOF_DATE = "2026-07-20"
DATE_TOKEN = "20260720"


def _write_curve(
    path: Path,
    dates: pd.DatetimeIndex,
    returns: list[float],
    exposure: float,
) -> None:
    nav = 1_000_000.0 * pd.Series(1.0 + np.asarray(returns), index=dates).cumprod()
    pd.DataFrame(
        {
            "date": dates,
            "equity": nav.to_numpy(),
            "turnover": 0.10,
            "cost": 0.0001,
            "gross_exposure": exposure,
        }
    ).to_csv(path, index=False)


def _write_metrics(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "total_return": 0.0,
                "max_drawdown": -0.10,
                "sharpe_like": 0.5,
            }
        ),
        encoding="utf-8",
    )


def test_strategy_arena_cli_outputs_feed_daily_run_state(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    output.mkdir()
    dates = pd.bdate_range(end=ASOF_DATE, periods=30)
    champion_returns = [0.0] + [0.001, -0.002] * 14 + [0.001]
    challenger_returns = [0.0] + [0.0015, -0.0005] * 14 + [0.0015]
    entrant_specs = {
        "champion": (champion_returns, 0.80),
        "breadth": (challenger_returns, 0.55),
        "baseline": (champion_returns, 0.45),
        "dynamic": (challenger_returns, 0.60),
    }
    sources: dict[str, tuple[Path, Path]] = {}
    for name, (returns, exposure) in entrant_specs.items():
        equity = tmp_path / f"{name}_equity.csv"
        metrics = tmp_path / f"{name}_metrics.json"
        _write_curve(equity, dates, returns, exposure)
        _write_metrics(metrics)
        sources[name] = (equity, metrics)

    tracking = tmp_path / "tracking.json"
    tracking.write_text(
        json.dumps(
            {
                "status": "collecting",
                "latest_asof_date": ASOF_DATE,
                "valid_observation_count": 3,
                "target_days": 20,
                "cumulative_return_delta": -0.002,
            }
        ),
        encoding="utf-8",
    )
    family_health = tmp_path / "strategy_family_health.csv"
    pd.DataFrame(
        [
            {
                "strategy_family": "trend_momentum",
                "strategy_family_cn": "趋势动量",
                "signal_count": 20,
                "completed_count": 8,
                "selected_horizon_days": 5,
                "avg_return": 0.02,
                "win_rate": 0.625,
                "worst_adverse_return": -0.08,
                "family_health_status": "normal",
                "family_health_reason": "正常观察",
            }
        ]
    ).to_csv(family_health, index=False)
    czsc_metadata = tmp_path / "czsc.json"
    czsc_metadata.write_text(
        json.dumps({"status": "partial", "candidate_count": 2, "analyzed_count": 1}),
        encoding="utf-8",
    )
    flow_metadata = tmp_path / "flow.json"
    flow_metadata.write_text(
        json.dumps({"status": "research_ready", "latest_source_rows": 2}),
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(PROJECT_ROOT / "build_strategy_arena_report.py"),
        "--asof-date",
        ASOF_DATE,
        "--champion-equity",
        str(sources["champion"][0]),
        "--champion-metrics",
        str(sources["champion"][1]),
        "--breadth-equity",
        str(sources["breadth"][0]),
        "--breadth-metrics",
        str(sources["breadth"][1]),
        "--baseline-equity",
        str(sources["baseline"][0]),
        "--baseline-metrics",
        str(sources["baseline"][1]),
        "--dynamic-equity",
        str(sources["dynamic"][0]),
        "--dynamic-metrics",
        str(sources["dynamic"][1]),
        "--tracking-summary",
        str(tracking),
        "--family-health",
        str(family_health),
        "--czsc-metadata",
        str(czsc_metadata),
        "--flow-metadata",
        str(flow_metadata),
        "--output-dir",
        str(output),
        "--history",
        str(output / "strategy_arena_history.csv"),
        "--min-common-days",
        "20",
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    expected_artifacts = (
        output / f"strategy_arena_portfolio_{DATE_TOKEN}.csv",
        output / f"strategy_arena_portfolio_{DATE_TOKEN}_cn.csv",
        output / f"strategy_arena_pairwise_{DATE_TOKEN}.csv",
        output / f"strategy_arena_signal_division_{DATE_TOKEN}.csv",
        output / f"strategy_arena_{DATE_TOKEN}.json",
        output / f"strategy_arena_{DATE_TOKEN}.md",
        output / "strategy_arena_history.csv",
    )
    for artifact in expected_artifacts:
        assert artifact.exists(), artifact

    config = PipelineConfig(
        asof_date=ASOF_DATE,
        project_root=tmp_path,
        output_root=Path("outputs"),
        daily_data_dir=tmp_path / "daily-data",
        fetch_status_utils=tmp_path / "missing_market_data_utils.py",
        include_strategy_arena=True,
        include_regime_shadow_compare=True,
        include_regime_shadow_tracking=False,
        include_research_db_sync=False,
    )
    state = build_daily_run_state(
        config,
        [PipelineStep("strategy_arena", tuple(command))],
        test_status="arena_smoke_passed",
    )
    verification = state["verification"]

    assert verification["tests"] == "arena_smoke_passed"
    assert verification["strategy_arena_status"] == "research_ready"
    assert verification["strategy_arena_portfolio_entrants"] == 4
    assert verification["strategy_arena_production_champion"] == "core_rank"
    assert verification["strategy_arena_observation_status"] == "collecting"
    assert verification["strategy_arena_observation_count"] == 3
    assert verification["strategy_arena_observation_target"] == 20
    assert verification["strategy_arena_automatic_promotion"] is False
    assert verification["strategy_arena_pareto_by_league"] == {
        "core_next_open": ["core_breadth_guard"],
        "pullback_satellite": ["pullback_dynamic"],
    }
    assert state["artifacts"]["strategy_arena_portfolio"] == str(
        expected_artifacts[0]
    )
    assert state["artifacts"]["strategy_arena_report"] == str(
        output / f"strategy_arena_{DATE_TOKEN}.md"
    )
