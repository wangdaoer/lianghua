import json
from pathlib import Path

import pandas as pd
import pytest

from analyze_factor_selection_alpha import (
    attach_forward_open_returns,
    build_factor_variants,
    compute_healthy_breadth_state,
    load_reference_parameters,
)
from multifactor_observation_evolution import DEFAULT_PARAMETERS, FACTOR_NAMES, WEIGHT_KEYS


def _factor_panel(momentum_by_date: list[tuple[float, float]]) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(momentum_by_date), freq="D")
    rows = []
    for index, (first, second) in enumerate(momentum_by_date):
        for symbol, momentum in (("000001.SZ", first), ("000002.SZ", second)):
            rows.append(
                {
                    "date": dates[index],
                    "symbol": symbol,
                    "open": 10.0 + index,
                    "momentum_20": momentum,
                    "score_eligible": True,
                }
            )
    return pd.DataFrame(rows)


def test_forward_returns_enter_at_next_open_and_skip_entry_gap():
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="D"),
            "symbol": ["000001.SZ"] * 4,
            "open": [10.0, 20.0, 22.0, 24.0],
        }
    )

    result = attach_forward_open_returns(panel, horizons=(1, 2))

    assert result.iloc[0]["forward_open_return_1d"] == pytest.approx(0.10)
    assert result.iloc[0]["forward_open_return_2d"] == pytest.approx(0.20)


def test_healthy_breadth_requires_point_in_time_ma_warmup():
    panel = _factor_panel([(0.1, -0.1), (0.1, 0.2), (0.1, 0.2)])

    state = compute_healthy_breadth_state(panel, ma_window=2)

    assert pd.isna(state.iloc[0]["breadth_20_ma"])
    assert not bool(state.iloc[0]["breadth_healthy"])
    assert state.iloc[1]["breadth_20"] == pytest.approx(1.0)
    assert state.iloc[1]["breadth_20_ma"] == pytest.approx(0.75)
    assert bool(state.iloc[1]["breadth_healthy"])


def test_factor_variants_build_reference_drop_and_single_factor_sets():
    variants = build_factor_variants(DEFAULT_PARAMETERS)

    assert len(variants) == 1 + 2 * len(FACTOR_NAMES)
    assert variants["drop_momentum_20"]["momentum_20_weight"] == 0.0
    single = variants["only_trend_acceleration"]
    assert single["trend_acceleration_weight"] == 1.0
    assert sum(float(single[key]) for key in WEIGHT_KEYS) == pytest.approx(1.0)


def test_reference_parameters_are_loaded_from_exact_candidate(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "candidate_id": "target",
                "parameters": json.dumps(DEFAULT_PARAMETERS),
            }
        ]
    ).to_csv(run_dir / "candidate_scores.csv", index=False)

    loaded = load_reference_parameters(run_dir, "target")

    assert loaded["top_n"] == DEFAULT_PARAMETERS["top_n"]
    assert loaded["breadth_ma_window"] == DEFAULT_PARAMETERS["breadth_ma_window"]
