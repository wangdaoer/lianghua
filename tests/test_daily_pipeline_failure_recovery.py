import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from run_daily_model_pipeline import (
    PipelineConfig,
    PipelineStep,
    PipelineStepExecutionError,
    build_daily_run_state,
    execute_pipeline,
)


ASOF_DATE = "2026-07-20"
DATE_TOKEN = "20260720"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_failed_rerun_hides_same_day_stale_downstream_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    output.mkdir()

    main_csv = output / f"main_net_volume_shadow_{DATE_TOKEN}.csv"
    _write_csv(main_csv, [{"symbol": "000001"}])
    _write_json(
        main_csv.with_suffix(".json"),
        {"status": "research_ready", "selection_effect": False},
    )

    institutional_csv = output / f"institutional_accumulation_shadow_{DATE_TOKEN}.csv"
    _write_csv(institutional_csv, [{"symbol": "000001", "signal_active": True}])
    _write_json(
        institutional_csv.with_suffix(".json"),
        {"status": "research_ready", "active_signal_rows": 1},
    )
    institutional_tracking = output / f"institutional_accumulation_tracking_{DATE_TOKEN}.csv"
    _write_csv(institutional_tracking, [{"symbol": "000001", "tracking_status": "complete"}])
    _write_json(
        institutional_tracking.with_suffix(".json"),
        {"status": "complete", "primary_completed_samples": 1},
    )

    czsc_csv = output / f"czsc_structure_shadow_{DATE_TOKEN}.csv"
    _write_csv(czsc_csv, [{"symbol": "000001", "analysis_status": "analyzed"}])
    _write_json(
        czsc_csv.with_suffix(".json"),
        {"status": "research_ready", "analyzed_count": 1},
    )

    arena_csv = output / f"strategy_arena_portfolio_{DATE_TOKEN}.csv"
    _write_csv(
        arena_csv,
        [{"entrant_id": "core_rank", "role": "production_champion"}],
    )
    _write_json(
        output / f"strategy_arena_{DATE_TOKEN}.json",
        {
            "status": "research_ready",
            "asof_date": ASOF_DATE,
            "production_champion": "core_rank",
            "automatic_promotion": False,
        },
    )

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
        max_parallel_steps=1,
    )
    succeed = (sys.executable, "-c", "raise SystemExit(0)")
    fail = (sys.executable, "-c", "raise SystemExit(7)")
    steps = [
        PipelineStep("main_net_volume_shadow", succeed),
        PipelineStep("institutional_accumulation_shadow", fail),
        PipelineStep("institutional_accumulation_tracking", succeed),
        PipelineStep("czsc_structure_shadow", succeed),
        PipelineStep("strategy_arena", succeed),
    ]
    state_log = tmp_path / "daily_run_state.jsonl"

    with pytest.raises(PipelineStepExecutionError) as raised:
        execute_pipeline(config, steps, state_log=state_log)

    assert raised.value.step.name == "institutional_accumulation_shadow"
    records = [json.loads(line) for line in state_log.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    state = records[0]
    verification = state["verification"]
    artifacts = state["artifacts"]

    assert state["run_status"] == "failed"
    assert state["failure"]["step"] == "institutional_accumulation_shadow"
    assert verification["main_net_volume_shadow_status"] == "research_ready"
    assert verification["main_net_volume_shadow_rows"] == 1
    assert artifacts["main_net_volume_shadow"] == str(main_csv)

    assert verification["institutional_accumulation_status"] == "failed"
    assert verification["institutional_accumulation_rows"] == 0
    assert artifacts["institutional_accumulation_shadow"] is None
    assert artifacts["institutional_accumulation_metadata"] is None

    assert verification["institutional_accumulation_tracking_status"] == "not_run"
    assert verification["institutional_accumulation_tracking_rows"] == 0
    assert artifacts["institutional_accumulation_tracking"] is None

    assert verification["czsc_structure_shadow_status"] == "not_run"
    assert verification["czsc_structure_shadow_rows"] == 0
    assert artifacts["czsc_structure_shadow"] is None

    assert verification["strategy_arena_status"] == "not_run"
    assert verification["strategy_arena_portfolio_entrants"] == 0
    assert artifacts["strategy_arena_portfolio"] is None
    assert artifacts["strategy_arena_metadata"] is None


def test_parallel_step_recorded_success_keeps_its_current_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    output.mkdir()
    czsc_csv = output / f"czsc_structure_shadow_{DATE_TOKEN}.csv"
    _write_csv(czsc_csv, [{"symbol": "000001", "analysis_status": "analyzed"}])
    _write_json(
        czsc_csv.with_suffix(".json"),
        {"status": "research_ready", "analyzed_count": 1},
    )
    config = PipelineConfig(
        asof_date=ASOF_DATE,
        project_root=tmp_path,
        output_root=Path("outputs"),
        daily_data_dir=tmp_path / "daily-data",
        fetch_status_utils=tmp_path / "missing_market_data_utils.py",
        include_strategy_arena=False,
        include_regime_shadow_compare=False,
        include_regime_shadow_tracking=False,
        include_research_db_sync=False,
        max_parallel_steps=2,
    )
    steps = [
        PipelineStep("institutional_accumulation_shadow", ("python", "fail.py")),
        PipelineStep("czsc_structure_shadow", ("python", "success.py")),
    ]

    state = build_daily_run_state(
        config,
        steps,
        run_status="failed",
        failure={"step": "institutional_accumulation_shadow", "returncode": 7},
        step_executions=[
            {"name": "institutional_accumulation_shadow", "status": "failed"},
            {"name": "czsc_structure_shadow", "status": "success"},
        ],
    )

    assert state["verification"]["institutional_accumulation_status"] == "failed"
    assert state["verification"]["czsc_structure_shadow_status"] == "research_ready"
    assert state["verification"]["czsc_structure_shadow_rows"] == 1
    assert state["artifacts"]["czsc_structure_shadow"] == str(czsc_csv)
