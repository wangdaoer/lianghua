from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "extend_latest_derived_curves.py"


def load_module():
    spec = importlib.util.spec_from_file_location("extend_latest_derived_curves", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stock_candidate_from_payload_restores_selected_parameters() -> None:
    module = load_module()
    payload = {
        "factor_weights": {"momentum": 0.2, "trend": 0.2, "reversal": 0.25, "volatility": 0.25, "liquidity": 0.1},
        "max_positions": 15,
        "rebalance_interval": 40,
        "risk_overrides": {"benchmark_off_exposure": 0.2},
    }

    candidate = module.stock_candidate_from_payload("stable_final", payload)

    assert candidate.name == "stable_final"
    assert candidate.rebalance_interval == 40
    assert candidate.max_positions == 15
    assert candidate.factor_weights["reversal"] == 0.25
    assert candidate.risk_overrides["benchmark_off_exposure"] == 0.2


def test_replace_last_window_row_updates_only_matching_final_window() -> None:
    module = load_module()
    rows = pd.DataFrame(
        [
            {"window": "wf_01_20250101_20250630", "test_start": "20250101", "test_end": "20250630", "value": 1},
            {"window": "wf_02_20260101_20260612", "test_start": "20260101", "test_end": "20260612", "value": 2},
        ]
    )
    new_row = {
        "window": "wf_02_20260101_20260615",
        "test_start": "20260101",
        "test_end": "20260615",
        "value": 3,
    }

    replaced = module.replace_last_window_row(rows, new_row)

    assert list(replaced["window"]) == ["wf_01_20250101_20250630", "wf_02_20260101_20260615"]
    assert replaced.iloc[-1]["test_end"] == "20260615"
    assert replaced.iloc[-1]["value"] == 3


def test_portfolio_candidate_from_payload_restores_source_and_allocation() -> None:
    module = load_module()
    payload = {
        "source_name": "quality",
        "source_path": "outputs/walk_forward/main/oos_equity_stitched.csv",
        "allocation": {
            "weights": {
                "risk_on": {"core": 0.75, "satellite": 0.25, "cash": 0.0},
                "risk_off": {"core": 1.0, "satellite": 0.0, "cash": 0.0},
                "crash": {"core": 1.0, "satellite": 0.0, "cash": 0.0},
            },
            "satellite_filter": {
                "enabled": False,
                "ma_window": 60,
                "momentum_window": 20,
                "min_momentum": 0.0,
                "max_drawdown": 0.15,
                "reduced_drawdown": 0.08,
                "reduced_scale": 0.5,
                "default_scale": 0.0,
                "require_above_ma": True,
                "reallocate_to": "core",
            },
            "regime_overrides": {"ma_window": 60, "risk_on_drop_threshold": -0.03},
        },
    }

    candidate = module.portfolio_source_candidate_from_payload("quality__sat25", payload, ROOT)

    assert candidate.name == "quality__sat25"
    assert candidate.source_name == "quality"
    assert candidate.satellite is not None
    assert candidate.satellite.path == ROOT / "outputs/walk_forward/main/oos_equity_stitched.csv"
    assert candidate.allocation.weights["risk_on"]["satellite"] == 0.25
    assert candidate.allocation.regime_overrides["ma_window"] == 60
