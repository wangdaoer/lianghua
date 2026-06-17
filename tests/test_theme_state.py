from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser
from quant_etf_lab.theme_state import compute_beta_gap, compute_beta_gap_evaluation, compute_theme_state, run_theme_state


def _write_price_csv(path: Path, code: str, closes: list[float], amount: float = 100_000_000.0) -> None:
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="D")
    pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "code": code,
            "name": code,
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [1_000_000.0] * len(closes),
            "amount": [amount] * len(closes),
        }
    ).to_csv(path, index=False)


def test_theme_state_cli_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["theme-state"])

    assert args.command == "theme-state"
    assert args.data_dir == "data/processed/stocks"
    assert args.output_dir == "outputs/research/theme_state_latest"
    assert args.horizons == "1,5,20"
    assert args.beta_window == 60
    assert args.theme_active_only is True


def test_run_theme_state_uses_board_fallback_and_writes_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    _write_price_csv(data_dir / "000001.csv", "000001", [10 + idx * 0.1 for idx in range(90)])
    _write_price_csv(data_dir / "300001.csv", "300001", [20 + idx * 0.05 for idx in range(90)])

    result = run_theme_state(
        project_root=tmp_path,
        data_dir=data_dir,
        output_dir=tmp_path / "theme_state",
        max_symbols=None,
        beta_window=10,
        theme_active_only=False,
    )

    assert result.snapshot["status"] == "ok"
    assert result.snapshot["research_only"] is True
    assert result.snapshot["group_source"] == "board_fallback"
    assert set(result.symbol_panel["theme_group"].unique()) == {"main", "chinext"}
    assert result.theme_state_path.exists()
    assert result.beta_gap_path.exists()
    assert result.beta_gap_ic_path.exists()
    assert result.beta_gap_summary_path.exists()
    payload = json.loads(result.snapshot_path.read_text(encoding="utf-8"))
    assert payload["broker_action"] == "none"


def test_run_theme_state_expands_multi_theme_map_memberships(tmp_path: Path) -> None:
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    _write_price_csv(data_dir / "000001.csv", "000001", [10 + idx * 0.1 for idx in range(90)])
    _write_price_csv(data_dir / "300001.csv", "300001", [20 + idx * 0.2 for idx in range(90)])
    theme_map = tmp_path / "theme_map.csv"
    pd.DataFrame(
        [
            {"code": "000001", "concept": "AI;机器人"},
            {"code": "300001", "concept": "机器人"},
        ]
    ).to_csv(theme_map, index=False)

    result = run_theme_state(
        project_root=tmp_path,
        data_dir=data_dir,
        output_dir=tmp_path / "theme_state",
        theme_map_path=theme_map,
        beta_window=10,
        theme_active_only=False,
    )

    groups_by_code = result.symbol_panel.groupby("code")["theme_group"].apply(lambda values: set(values)).to_dict()
    assert groups_by_code["000001"] == {"AI", "机器人"}
    assert groups_by_code["300001"] == {"机器人"}
    assert result.snapshot["group_count"] == 2
    assert "theme_map" in result.snapshot["group_source"]


def test_run_theme_state_accepts_theme_group_map_column(tmp_path: Path) -> None:
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    _write_price_csv(data_dir / "000001.csv", "000001", [10 + idx * 0.1 for idx in range(90)])
    theme_map = tmp_path / "theme_map.csv"
    pd.DataFrame([{"code": "000001", "theme_group": "theme_a"}]).to_csv(theme_map, index=False)

    result = run_theme_state(
        project_root=tmp_path,
        data_dir=data_dir,
        output_dir=tmp_path / "theme_state",
        theme_map_path=theme_map,
        beta_window=10,
        theme_active_only=False,
    )

    assert set(result.symbol_panel["theme_group"].unique()) == {"theme_a"}
    assert result.snapshot["group_source"] == "mixed:board_fallback,theme_map"


def test_compute_theme_state_detects_second_activation() -> None:
    dates = pd.date_range("2026-01-01", periods=25, freq="D")
    rows = []
    for idx, date_value in enumerate(dates):
        active = idx >= 15
        rows.append(
            {
                "date": date_value,
                "code": "000001",
                "name": "A",
                "theme_group": "theme_a",
                "close": 10.0 + idx,
                "amount": 100.0,
                "daily_return": 0.02 if active else -0.01,
                "above_ma20": active,
                "above_ma60": active,
                "momentum_20d": 0.1 if active else -0.1,
            }
        )
    panel = pd.DataFrame(rows)

    state = compute_theme_state(
        panel,
        breadth_window=3,
        slope_window=1,
        quiet_lookback=3,
        activation_breadth=0.60,
        prior_breadth_ceiling=0.40,
        min_breadth_slope=0.20,
        min_rs_20d=-1.0,
    )

    assert bool(state["theme_second_activation"].any()) is True
    assert "second_activation" in set(state["theme_state"])


def test_compute_beta_gap_marks_laggard_when_theme_return_exceeds_stock_return() -> None:
    dates = pd.date_range("2026-01-01", periods=8, freq="D")
    theme_returns = [0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04]
    panel = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "code": ["000001"] * 8 + ["000002"] * 8,
            "name": ["A"] * 8 + ["B"] * 8,
            "theme_group": ["theme_a"] * 16,
            "close": [10.0] * 16,
            "amount": [100.0] * 16,
            "daily_return": [0.005, 0.010, 0.015, 0.020, 0.005, 0.010, 0.015, 0.015] + theme_returns,
            "above_ma20": [True] * 16,
            "above_ma60": [True] * 16,
            "momentum_20d": [0.1] * 16,
            "breakout_20d": [0.0] * 16,
        }
    )
    theme_state = pd.DataFrame(
        {
            "date": dates,
            "theme_group": ["theme_a"] * 8,
            "theme_return": theme_returns,
            "theme_state": ["healthy"] * 8,
            "theme_second_activation": [False] * 8,
        }
    )

    beta_gap = compute_beta_gap(panel, theme_state, beta_window=3, theme_active_only=True)

    latest_a = beta_gap.loc[(beta_gap["code"] == "000001") & (beta_gap["date"] == dates[-1])].iloc[0]
    assert latest_a["theme_state"] == "healthy"
    assert bool(latest_a["beta_gap_positive"]) is True
    assert latest_a["beta_gap"] > 0


def test_compute_beta_gap_evaluation_returns_summary() -> None:
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    rows = []
    for date_value in dates:
        for idx in range(8):
            rows.append(
                {
                    "date": date_value,
                    "theme_group": "theme_a",
                    "theme_state": "healthy",
                    "beta_gap": float(idx),
                    "fwd_return_1d": float(idx) / 100.0,
                }
            )
    beta_gap = pd.DataFrame(rows)

    by_date, summary = compute_beta_gap_evaluation(beta_gap, horizons=[1], quantiles=4, min_obs=6)

    assert len(by_date) == 5
    assert summary.iloc[0]["factor"] == "beta_gap"
    assert summary.iloc[0]["horizon"] == 1
    assert summary.iloc[0]["ic_mean"] > 0
    assert summary.iloc[0]["long_short_mean"] > 0
