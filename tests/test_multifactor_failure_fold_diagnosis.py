from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analyze_multifactor_failure_fold import (
    build_entry_events,
    build_symbol_contributions,
    load_current_snapshot_metadata,
    summarize_buckets,
)
from multifactor_observation_evolution import (
    DEFAULT_PARAMETERS,
    FACTOR_NAMES,
    FACTOR_RANK_COLUMNS,
    EvolutionPeriods,
    evaluate_parameter_set,
    run_observation_backtest,
)


def parameter_set(**overrides):
    parameters = dict(DEFAULT_PARAMETERS)
    parameters.update(
        {
            "top_n": 1,
            "rebalance_frequency": 1,
            "gross_exposure": 1.0,
            "max_position_weight": 1.0,
            "min_score": 0.60,
            "min_median_amount_20": 0.0,
            "max_daily_amount_participation": 1.0,
            "max_buy_open_gap": 0.50,
        }
    )
    parameters.update(overrides)
    return parameters


def factor_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-01", periods=12)
    prices_a = [10.0, 10.1, 10.2, 10.3, 9.5, 9.0, 8.8, 8.7, 8.6, 8.5, 8.4, 8.3]
    rows = []
    for index, date in enumerate(dates):
        for symbol, price, factor_value, rank_value in (
            ("000001", prices_a[index], 0.20, 0.5 if index < 3 else 1.0),
            ("000002", 10.0, 0.01, 0.4),
        ):
            row = {
                "date": date,
                "symbol": symbol,
                "open": price,
                "close": price,
                "score_eligible": True,
            }
            for factor in FACTOR_NAMES:
                row[factor] = (
                    np.log1p(100_000_000.0)
                    if factor == "liquidity_20"
                    else factor_value
                )
                row[FACTOR_RANK_COLUMNS[factor]] = rank_value
            rows.append(row)
    return pd.DataFrame(rows)


def periods(panel: pd.DataFrame) -> EvolutionPeriods:
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    return EvolutionPeriods(
        research_start=dates[0],
        selection_start=dates[3],
        selection_end=dates[7],
        holdout_start=dates[8],
        holdout_end=dates[-1],
        fold_sessions=5,
        step_sessions=5,
        warmup_sessions=2,
    )


def test_position_recording_is_opt_in_and_captures_after_open_weights():
    panel = factor_panel()

    ordinary = run_observation_backtest(panel, parameter_set(min_score=0.4))
    diagnosed = run_observation_backtest(
        panel, parameter_set(min_score=0.4), record_positions=True
    )

    assert ordinary.position_weights is None
    assert diagnosed.position_weights is not None
    assert diagnosed.position_weights.iloc[0]["000001"] == pytest.approx(0.0)
    assert diagnosed.position_weights.iloc[1]["000001"] == pytest.approx(1.0)


def test_snapshot_metadata_is_explicitly_non_point_in_time(tmp_path: Path):
    snapshot = tmp_path / "ths_hs_a_share_2026-07-13.xls"
    snapshot.write_bytes(
        (
            "代码\t名称\t换手\t所属行业\t细分行业\t总市值\t\n"
            "SH600000\t浦发银行\t0.23%\t银行\t股份制银行\t306080650000\t\n"
        ).encode("gb18030")
    )

    metadata = load_current_snapshot_metadata(snapshot)

    assert metadata.iloc[0]["symbol"] == "600000"
    assert metadata.iloc[0]["market_cap_current"] == pytest.approx(306080650000.0)
    assert metadata.iloc[0]["turnover_rate_current"] == pytest.approx(0.0023)
    assert metadata.iloc[0]["snapshot_asof"] == "2026-07-13"
    assert bool(metadata.iloc[0]["snapshot_is_point_in_time_for_fold"]) is False


def test_fold_diagnosis_reconciles_symbol_contributions_and_uses_signal_date():
    panel = factor_panel()
    evaluation = evaluate_parameter_set(
        panel,
        parameter_set(),
        periods(panel),
        record_positions=True,
    )
    entries = build_entry_events(
        evaluation, panel, parameter_set(), "baseline", "wf_01"
    )
    metadata = pd.DataFrame(
        [
            {
                "symbol": "000001",
                "security_name_current": "A",
                "market_cap_current": 8_000_000_000.0,
                "snapshot_is_point_in_time_for_fold": False,
            }
        ]
    )
    symbols = build_symbol_contributions(
        evaluation, entries, metadata, "baseline", "wf_01"
    )

    fold = evaluation.fold_rows[0]
    expected = evaluation.backtest.symbol_pnl.loc[
        pd.Timestamp(fold["start"]) : pd.Timestamp(fold["end"])
    ].to_numpy().sum()
    assert entries["signal_date"].lt(entries["execution_date"]).all()
    assert symbols["gross_contribution"].sum() == pytest.approx(expected)
    assert symbols.iloc[0]["historical_liquidity_bucket"] == "100m_to_300m"
    assert symbols.iloc[0]["current_cap_bucket"] == "5bn_to_10bn"

    bucket = summarize_buckets(symbols, "historical_liquidity_bucket")
    assert bool(bucket.iloc[0]["metadata_is_point_in_time_for_fold"]) is True
    assert bucket["share_of_candidate_gross_loss"].sum() == pytest.approx(1.0)
