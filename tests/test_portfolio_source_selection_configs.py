from __future__ import annotations

import json
from pathlib import Path

from quant_etf_lab.portfolio import load_portfolio_config, load_portfolio_source_selection_config


def test_chip_reversal_truepos_h5_costgap03_config_loads_filtered_curve() -> None:
    config, sources = load_portfolio_source_selection_config(
        Path("configs/portfolio_core_source_selection_quality_reversal_chip_reversal_truepos_h5_costgap03.yaml")
    )

    assert config.name == "main_chinext_core_source_selection_quality_reversal_chip_reversal_truepos_h5_costgap03"
    assert "chip_reversal_truepos_h5_costgap03" in sources
    chip_source = sources["chip_reversal_truepos_h5_costgap03"]
    assert chip_source.equity_column == "equity"
    assert chip_source.path.exists()
    assert chip_source.path.name == "equity_curve.csv"


def test_latest_chip_reversal_truepos_h5_source_selection_config_uses_executable_curve() -> None:
    config, sources = load_portfolio_source_selection_config(
        Path("configs/portfolio_core_source_selection_quality_reversal_chip_reversal_truepos_h5_latest.yaml")
    )

    assert config.name == "main_chinext_core_source_selection_quality_reversal_chip_reversal_truepos_h5_latest"
    assert "chip_reversal_truepos_h5" in sources
    chip_source = sources["chip_reversal_truepos_h5"]
    assert chip_source.equity_column == "equity"
    assert chip_source.path.exists()
    assert chip_source.path.name == "equity_curve.csv"
    assert "2018_2026_20260618" in chip_source.path.as_posix()
    assert "deep_h5_pos10_2018_2025" not in chip_source.path.as_posix()

    snapshot_path = chip_source.path.parent / "chip_reversal_event_portfolio_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
    assert snapshot["execution_model"] == "cash_position_backtest"
    assert snapshot["broker_action"] == "none"
    assert snapshot["research_only"] is True
    assert snapshot["horizon"] == 5
    assert snapshot["max_total_positions"] == 25


def test_latest_chip_reversal_truepos_h5_guarded_config_is_research_only_single_source() -> None:
    config = load_portfolio_config(Path("configs/portfolio_core_chip_reversal_truepos_h5_exp50_latest_guarded.yaml"))

    assert config.name == "main_chinext_core_chip_reversal_truepos_h5_exp50_latest_guarded"
    assert config.satellite.equity_column == "equity"
    assert config.satellite.path.exists()
    assert config.weights["risk_on"]["core"] == 0.90
    assert config.weights["risk_on"]["satellite"] == 0.10
    assert "broker_action=none" in config.notes

    snapshot_path = config.satellite.path.parent / "chip_reversal_event_portfolio_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
    assert snapshot["execution_model"] == "cash_position_backtest"
    assert snapshot["broker_action"] == "none"
    assert snapshot["target_exposure"] == 0.5
    assert snapshot["per_position_weight"] == 0.02


def test_highyield_dd50_config_adds_protected_highgain_dd40_source() -> None:
    config, sources = load_portfolio_source_selection_config(
        Path("configs/portfolio_core_source_selection_quality_reversal_highyield_dd50_highgain_dd40.yaml")
    )

    assert config.name == "main_chinext_core_source_selection_quality_reversal_highyield_dd50_highgain_dd40"
    assert "highgain_dd40_protected" in sources
    source = sources["highgain_dd40_protected"]
    assert source.equity_column == "stitched_equity"
    assert source.path.exists()
    assert "wf_next_highgain_only_entry_highgain_dd40_grid_v2_capital_gate_full" in source.path.as_posix()


def test_highgain_dd40_only_config_loads_protected_source() -> None:
    config, sources = load_portfolio_source_selection_config(
        Path("configs/portfolio_core_source_selection_quality_reversal_highgain_dd40_only.yaml")
    )

    assert config.name == "main_chinext_core_source_selection_quality_reversal_highgain_dd40_only"
    assert "highgain_dd40_protected" in sources
    assert sources["highgain_dd40_protected"].path.exists()
