from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_legacy_alpha_strict_replay as replay


def _build_research_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", "2026-06-30")
    symbols = [
        "300001",
        "300002",
        "300003",
        "300004",
        "300005",
        "300006",
        "300007",
        "300008",
        "300009",
        "300010",
        "300011",
        "300012",
    ]
    rows: list[dict[str, object]] = []
    time_index = np.arange(len(dates), dtype=float)
    for offset, symbol in enumerate(symbols):
        drift = 0.0010 + offset * 0.00005
        wave = 0.0018 * np.sin(time_index / (7.0 + offset / 3.0) + offset * 0.4)
        returns = drift + wave
        if symbol == "300001":
            for spike_day in (160, 220, 280):
                if spike_day < len(returns):
                    returns[spike_day] += 0.08
        close = 20.0 + offset + np.cumprod(1.0 + returns)
        open_px = close * (1.0 + 0.0015 * np.cos(time_index / (5.0 + offset / 4.0)))
        high = np.maximum(open_px, close) * 1.01
        low = np.minimum(open_px, close) * 0.99
        volume = 1_000_000.0 + offset * 25_000.0 + time_index * 400.0
        amount = volume * close
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "open": float(open_px[idx]),
                    "high": float(high[idx]),
                    "low": float(low[idx]),
                    "close": float(close[idx]),
                    "volume": float(volume[idx]),
                    "amount": float(amount[idx]),
                    "limit_rate": 0.20,
                }
            )

    panel = pd.DataFrame(rows)
    gap_date = dates[161]
    previous_date = dates[160]
    previous_close = float(
        panel.loc[
            panel["date"].eq(previous_date.strftime("%Y-%m-%d")) & panel["symbol"].eq("300001"),
            "close",
        ].iloc[0]
    )
    panel.loc[
        panel["date"].eq(gap_date.strftime("%Y-%m-%d")) & panel["symbol"].eq("300001"),
        "open",
    ] = previous_close * 1.05
    return panel


def _build_benchmark() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", "2026-06-30")
    leg1 = np.linspace(100.0, 130.0, 180, dtype=float)
    leg2 = np.linspace(129.0, 95.0, len(dates) - 180, dtype=float)
    close = np.concatenate([leg1, leg2])
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": close})


def _build_control_equity() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", "2026-06-30")
    daily_returns = 0.0006 + 0.0002 * np.sin(np.arange(len(dates)) / 9.0)
    equity = 1_000_000.0 * np.cumprod(1.0 + daily_returns)
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "equity": equity})


def test_run_replay_study_writes_preregistered_outputs(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.csv"
    benchmark_path = tmp_path / "benchmark.csv"
    control_path = tmp_path / "control.csv"
    output_dir = tmp_path / "study"

    _build_research_panel().to_csv(panel_path, index=False)
    _build_benchmark().to_csv(benchmark_path, index=False)
    _build_control_equity().to_csv(control_path, index=False)

    manifest = replay.run_replay_study(
        data_path=panel_path,
        benchmark_path=benchmark_path,
        control_equity_path=control_path,
        output_dir=output_dir,
        asof_date="2026-07-21",
    )

    assert manifest["research_only"] is True
    assert manifest["promotion_allowed"] is False
    assert set(manifest["preregistered_variants"]) == {
        "Control",
        "OldTop10-Strict",
        "OldTop10-Strict+Market",
        "OldTop10-Strict+Market+Breadth",
    }

    for filename in manifest["outputs"]:
        assert (output_dir / filename).exists(), filename

    segment_frame = pd.read_csv(output_dir / "segment_comparison.csv")
    full_history = segment_frame.loc[segment_frame["segment_id"].eq("full_history"), "variant_id"]
    assert set(full_history) == set(manifest["preregistered_variants"])
    assert segment_frame["segment_id"].isin(
        ["full_history", "2025H2", "2026-01-01_to_2026-06-15"]
    ).all()

    manifest_payload = json.loads((output_dir / "replay_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["factor_definitions"]["trend120"] == "rank_pct of close / MA120 - 1"
    assert manifest_payload["production_integration"] == "disabled_research_only"
    assert manifest_payload["variant_summaries"]["Control"]["blocked_orders_total"] is None
    assert all(
        "control_position_overlap_unavailable" in failures
        for failures in manifest_payload["failed_gates"].values()
    )


def test_run_strict_replay_variant_records_open_gap_blocks() -> None:
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    columns = ["300001"]
    close = pd.DataFrame([[100.0], [103.0], [104.0]], index=dates, columns=columns)
    open_px = pd.DataFrame([[100.0], [105.0], [106.0]], index=dates, columns=columns)
    score = pd.DataFrame([[1.0], [1.0], [1.0]], index=dates, columns=columns)
    exposure = pd.Series(1.0, index=dates)

    result = replay.run_strict_replay_variant(
        replay.VariantSpec("OldTop10-Strict", use_market=False, use_breadth=False),
        close,
        open_px,
        score,
        exposure,
        initial_capital=1_000_000.0,
    )

    assert result.metrics["blocked_open_gap_buys"] == 1
    assert result.metrics["blocked_orders_total"] == 1
    assert float(result.curve["equity"].iloc[-1]) == pytest.approx(1_000_000.0)


def test_strict_replay_reselects_only_on_frozen_factor_schedule(monkeypatch) -> None:
    dates = pd.bdate_range("2026-01-02", periods=12)
    columns = [f"300{i:03d}" for i in range(1, 12)]
    close = pd.DataFrame(100.0, index=dates, columns=columns)
    open_px = close.copy()
    score = pd.DataFrame(
        [np.arange(len(columns)) if i < 4 else np.arange(len(columns))[::-1] for i in range(len(dates))],
        index=dates,
        columns=columns,
    )
    exposure = pd.Series(1.0, index=dates)
    monkeypatch.setattr(replay, "MAX_BUY_OPEN_GAP", None)

    result = replay.run_strict_replay_variant(
        replay.VariantSpec("OldTop10-Strict", use_market=False, use_breadth=False),
        close,
        open_px,
        score,
        exposure,
        initial_capital=1_000_000.0,
    )

    assert result.diagnostics.loc[1:7, "turnover"].eq(0.0).all()
    assert result.diagnostics.loc[8, "turnover"] > 0.0


def test_load_control_equity_curve_rejects_duplicate_dates(tmp_path: Path) -> None:
    path = tmp_path / "control.csv"
    pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-02"],
            "equity": [1_000_000.0, 1_001_000.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate dates"):
        replay.load_control_equity_curve(path, asof_date="2026-07-21")
