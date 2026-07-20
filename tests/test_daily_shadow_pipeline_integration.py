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


def _write_panel(path: Path) -> None:
    dates = pd.bdate_range(end=ASOF_DATE, periods=120)
    rows: list[dict[str, object]] = []
    for symbol, slope, flow in (
        ("000001", 0.012, 0.004),
        ("000002", 0.005, -0.002),
    ):
        close = 10.0 + np.arange(len(dates)) * slope
        amount = np.full(len(dates), 100_000_000.0)
        if symbol == "000001":
            amount[-5:] = 140_000_000.0
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close[index] * 0.998,
                    "high": close[index] * 1.015,
                    "low": close[index] * 0.985,
                    "close": close[index],
                    "volume": 1_000_000 + index * 100,
                    "amount": amount[index],
                    "main_net_volume_ratio": flow,
                    "main_net_inflow": amount[index] * flow,
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _run_script(script: str, *args: object, env: dict[str, str]) -> None:
    command = [sys.executable, str(PROJECT_ROOT / script), *map(str, args)]
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_shadow_cli_outputs_feed_daily_run_state(tmp_path: Path) -> None:
    output = tmp_path / "outputs"
    output.mkdir()
    panel = tmp_path / "panel.parquet"
    _write_panel(panel)

    names = tmp_path / "names.csv"
    pd.DataFrame(
        {"代码": ["000001", "000002"], "名称": ["平安银行", "万科A"]}
    ).to_csv(names, index=False, encoding="utf-8-sig")
    early = output / f"early_pattern_watchlist_{DATE_TOKEN}.csv"
    pd.DataFrame(
        {
            "symbol": ["000001"],
            "stock_name": ["平安银行"],
            "pattern_type": ["隐性吸筹观察"],
            "pattern_score": [1.2],
            "hidden_accumulation_trade_watch": [True],
        }
    ).to_csv(early, index=False)
    overlay = output / f"rank_model_candidates_trend_gated_personal_overlay_{DATE_TOKEN}.csv"
    pd.DataFrame(
        {
            "symbol": ["000001"],
            "stock_name": ["平安银行"],
            "personal_rank": [1],
            "personal_selected": [True],
            "trend_state": ["strong"],
        }
    ).to_csv(overlay, index=False)

    # Force the deterministic, auditable CZSC-unavailable path on every CI host.
    stub_root = tmp_path / "stubs"
    package = stub_root / "czsc"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "unsupported-test-version"\n')
    (package / "signals.py").write_text("", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(stub_root), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    main_output = output / f"main_net_volume_shadow_{DATE_TOKEN}.csv"
    _run_script(
        "main_net_volume_shadow.py",
        "--data",
        panel,
        "--output",
        main_output,
        "--asof-date",
        ASOF_DATE,
        "--names-source",
        names,
        "--early-watchlist",
        early,
        env=env,
    )

    institutional_output = output / f"institutional_accumulation_shadow_{DATE_TOKEN}.csv"
    _run_script(
        "institutional_accumulation_shadow.py",
        "--data",
        panel,
        "--output",
        institutional_output,
        "--asof-date",
        ASOF_DATE,
        "--config",
        PROJECT_ROOT / "configs" / "institutional_accumulation_shadow.yaml",
        "--names-source",
        names,
        env=env,
    )

    czsc_output = output / f"czsc_structure_shadow_{DATE_TOKEN}.csv"
    _run_script(
        "czsc_structure_shadow.py",
        "--data",
        panel,
        "--candidates",
        overlay,
        "--candidates",
        early,
        "--output",
        czsc_output,
        "--asof-date",
        ASOF_DATE,
        "--names-source",
        names,
        env=env,
    )

    for artifact in (
        main_output,
        main_output.with_name(f"{main_output.stem}_cn.csv"),
        main_output.with_suffix(".json"),
        main_output.with_suffix(".md"),
        institutional_output,
        institutional_output.with_name(f"{institutional_output.stem}_cn.csv"),
        institutional_output.with_suffix(".json"),
        institutional_output.with_suffix(".md"),
        czsc_output,
        czsc_output.with_name(f"{czsc_output.stem}_cn.csv"),
        czsc_output.with_suffix(".json"),
        czsc_output.with_suffix(".md"),
    ):
        assert artifact.exists(), artifact

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
    )
    steps = [
        PipelineStep("main_net_volume_shadow", ("python", "main_net_volume_shadow.py")),
        PipelineStep(
            "institutional_accumulation_shadow",
            ("python", "institutional_accumulation_shadow.py"),
        ),
        PipelineStep("czsc_structure_shadow", ("python", "czsc_structure_shadow.py")),
    ]
    state = build_daily_run_state(config, steps, test_status="smoke_passed")
    verification = state["verification"]
    institutional_rows = len(pd.read_csv(institutional_output))

    assert verification["tests"] == "smoke_passed"
    assert verification["main_net_volume_shadow_status"] == "research_ready"
    assert verification["main_net_volume_shadow_rows"] == 2
    assert verification["main_net_volume_early_pattern_overlap"] == 1
    assert verification["main_net_volume_selection_effect"] is False
    assert verification["institutional_accumulation_status"] == "research_ready"
    assert institutional_rows > 0
    assert verification["institutional_accumulation_rows"] == institutional_rows
    assert verification["institutional_accumulation_selection_effect"] is False
    assert verification["czsc_structure_shadow_status"] == "unavailable"
    assert verification["czsc_structure_shadow_rows"] == 1
    assert verification["czsc_structure_selection_effect"] is False
