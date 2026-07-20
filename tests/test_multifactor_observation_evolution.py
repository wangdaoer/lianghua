from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import run_multifactor_observation_evolution as evolution_runner
from multifactor_observation_evolution import (
    DEFAULT_PARAMETERS,
    EvolutionPeriods,
    SearchCandidate,
    SearchGroup,
    benchmark_total_return,
    build_factor_scores,
    build_cross_sectional_momentum_breadth,
    build_market_exposure,
    build_walk_forward_folds,
    combine_panel_frames,
    load_evolution_config,
    parse_evolution_config,
    run_observation_backtest,
)
from run_multifactor_observation_evolution import (
    _candidate_parameters,
    load_market_data,
    run_evolution,
    select_search_groups,
)
from panel_io import write_panel_atomic
from strategy_evolution_core import PromotionDecision, PromotionPolicy


CONFIG_PATH = Path("configs/evolution_multifactor_observation.yaml")


def test_load_market_data_accepts_parquet_panel(tmp_path: Path) -> None:
    path = tmp_path / "panel.parquet"
    panel = pd.DataFrame(
        {
            "date": ["2026-07-20"],
            "symbol": ["000001"],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "volume": [100.0],
            "amount": [1020.0],
        }
    )
    write_panel_atomic(panel, path)

    loaded = load_market_data([path])

    assert loaded["symbol"].tolist() == ["000001"]
    assert loaded["date"].max() == pd.Timestamp("2026-07-20")


def parameter_set(**overrides):
    params = dict(DEFAULT_PARAMETERS)
    params.update(overrides)
    return params


def scored_panel(opens_a, closes_a=None):
    dates = pd.date_range("2026-01-01", periods=len(opens_a), freq="D")
    closes_a = closes_a if closes_a is not None else opens_a
    rows = []
    for index, date in enumerate(dates):
        for symbol, open_price, close_price, factor_value in (
            ("000001", opens_a[index], closes_a[index], 2.0),
            ("000002", 10.0, 10.0, 1.0),
        ):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": open_price,
                    "close": close_price,
                    "momentum_20": factor_value,
                    "momentum_60": factor_value,
                    "breakout_distance_20": factor_value,
                    "trend_acceleration": factor_value,
                    "liquidity_20": np.log1p(100_000_000.0),
                    "liquidity_stability_20": factor_value,
                    "score_eligible": True,
                }
            )
    return pd.DataFrame(rows)


def history_panel(periods=240):
    dates = pd.bdate_range("2025-01-01", periods=periods)
    rows = []
    for symbol, slope in (("000001", 0.02), ("000002", 0.01), ("000003", -0.002)):
        for index, date in enumerate(dates):
            close = 10.0 + index * slope
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close * 0.999,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000.0,
                    "amount": 100_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_default_config_preserves_non_negotiable_execution_rules():
    config = load_evolution_config(CONFIG_PATH)

    assert config.locked_execution == {
        "signal_lag_sessions": 1,
        "execution_model": "next_open",
        "historical_universe_only": True,
        "allow_short": False,
        "max_leverage": 1.0,
        "block_limit_up_buys": True,
        "block_limit_down_sells": True,
    }
    assert config.annual_return_stretch == 10.0
    assert config.benchmark == "CSI1000"
    assert config.baseline["market_below_ma_exposure"] == 1.0
    assert {group.group_id for group in config.search_groups}.issuperset(
        {
            "portfolio_drawdown_stop",
            "overextension_guard",
            "momentum_breadth_regime",
            "momentum_breadth_refinement",
            "healthy_breadth_factor_selection",
        }
    )


def test_config_rejects_relaxed_signal_lag():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["locked_execution"]["signal_lag_sessions"] = 0

    with pytest.raises(ValueError, match="locked_execution"):
        parse_evolution_config(raw)


def test_candidate_cannot_optimize_transaction_costs():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw["search_groups"][0]["candidates"][0]["overrides"]["commission_bps"] = 0.0

    with pytest.raises(ValueError, match="locked parameters"):
        parse_evolution_config(raw)


def test_market_regime_candidate_cannot_use_invalid_exposure():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    group = next(item for item in raw["search_groups"] if item["id"] == "market_regime")
    group["candidates"][0]["overrides"][
        "market_crash_exposure"
    ] = 1.1

    with pytest.raises(ValueError, match="market_crash_exposure"):
        parse_evolution_config(raw)


def test_absolute_trend_candidate_rejects_invalid_threshold():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    group = next(
        item for item in raw["search_groups"] if item["id"] == "absolute_trend_guard"
    )
    group["candidates"][0]["overrides"][
        "min_momentum_20"
    ] = -1.1

    with pytest.raises(ValueError, match="min_momentum_20"):
        parse_evolution_config(raw)


def test_portfolio_stop_candidate_rejects_nonnegative_drawdown():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    group = next(
        item
        for item in raw["search_groups"]
        if item["id"] == "portfolio_drawdown_stop"
    )
    group["candidates"][0]["overrides"]["portfolio_stop_drawdown"] = 0.0

    with pytest.raises(ValueError, match="portfolio_stop_drawdown"):
        parse_evolution_config(raw)


def test_overextension_candidate_rejects_invalid_ceiling():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    group = next(
        item for item in raw["search_groups"] if item["id"] == "overextension_guard"
    )
    group["candidates"][0]["overrides"]["max_momentum_60"] = -0.1

    with pytest.raises(ValueError, match="max_momentum_60"):
        parse_evolution_config(raw)


def test_breadth_candidate_rejects_inverted_crash_threshold():
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    group = next(
        item
        for item in raw["search_groups"]
        if item["id"] == "momentum_breadth_regime"
    )
    candidate = group["candidates"][-1]["overrides"]
    candidate["breadth_risk_off_gap"] = -0.10
    candidate["breadth_crash_gap"] = 0.05

    with pytest.raises(ValueError, match="breadth_crash_gap"):
        parse_evolution_config(raw)


def test_candidate_identity_is_stable_after_numeric_validation():
    candidate = SearchCandidate(
        "strict_liquidity", {"min_median_amount_20": 50_000_000}
    )

    generated = _candidate_parameters(
        DEFAULT_PARAMETERS, (candidate,), max_candidates=1, seed=1
    )

    assert generated[0][0] == "strict_liquidity"
    assert generated[0][1]["min_median_amount_20"] == 50_000_000.0


def test_runner_can_limit_evaluation_to_one_research_group():
    config = load_evolution_config(CONFIG_PATH)

    selected = select_search_groups(config, ["overextension_guard"])

    assert [group.group_id for group in selected.search_groups] == [
        "overextension_guard"
    ]
    with pytest.raises(ValueError, match="unknown search groups"):
        select_search_groups(config, ["missing_group"])


def test_later_daily_source_replaces_overlapping_history_row():
    old = pd.DataFrame(
        [{"date": "2026-06-24", "symbol": "000001", "open": 10.0, "close": 10.0}]
    )
    daily = pd.DataFrame(
        [{"date": "2026-06-24", "symbol": "000001", "open": 10.5, "close": 10.6}]
    )

    combined = combine_panel_frames([old, daily])

    assert len(combined) == 1
    assert combined.iloc[0]["close"] == pytest.approx(10.6)


def test_evolution_quarantines_only_complete_zero_market_placeholders():
    frame = pd.DataFrame(
        [
            {
                "date": "2026-06-16",
                "symbol": "000001",
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.1,
                "volume": 100.0,
                "amount": 1_000.0,
            },
            {
                "date": "2026-06-17",
                "symbol": "000001",
                "open": 0.0,
                "high": 0.0,
                "low": 0.0,
                "close": 0.0,
                "volume": 0.0,
                "amount": 0.0,
            },
            {
                "date": "2026-06-18",
                "symbol": "000001",
                "open": 10.2,
                "high": 10.3,
                "low": 10.0,
                "close": 10.2,
                "volume": 120.0,
                "amount": 1_200.0,
            },
        ]
    )

    combined = combine_panel_frames([frame])

    assert combined["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-06-16",
        "2026-06-18",
    ]
    assert combined.attrs["quarantined_all_zero_market_rows"] == 1


def test_evolution_rejects_partial_zero_market_row():
    frame = pd.DataFrame(
        [
            {
                "date": "2026-06-16",
                "symbol": "000001",
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.1,
                "volume": 100.0,
                "amount": 1_000.0,
            },
            {
                "date": "2026-06-17",
                "symbol": "000001",
                "open": 10.0,
                "high": 10.0,
                "low": 0.0,
                "close": 0.0,
                "volume": 0.0,
                "amount": 0.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="terminal all-zero"):
        combine_panel_frames([frame])


def test_evolution_marks_carried_close_suspension_as_no_open_quote():
    frame = pd.DataFrame(
        [
            {
                "date": "2026-06-16",
                "symbol": "600717",
                "open": 4.31,
                "high": 4.31,
                "low": 4.31,
                "close": 4.31,
                "volume": 100.0,
                "amount": 431.0,
            },
            {
                "date": "2026-06-17",
                "symbol": "600717",
                "open": 0.0,
                "high": 0.0,
                "low": 0.0,
                "close": 4.31,
                "volume": 0.0,
                "amount": 0.0,
            },
        ]
    )

    combined = combine_panel_frames([frame])
    suspended = combined.loc[combined["date"].eq(pd.Timestamp("2026-06-17"))].iloc[0]

    assert pd.isna(suspended["open"])
    assert pd.isna(suspended["high"])
    assert pd.isna(suspended["low"])
    assert suspended["close"] == pytest.approx(4.31)
    assert combined.attrs["suspended_no_open_quote_rows"] == 1


def test_scores_use_only_approved_observation_factors():
    panel = scored_panel([10.0, 10.1, 10.2])
    baseline = build_factor_scores(panel, parameter_set(top_n=1))
    contaminated = panel.assign(forward_return_20=999.0, next_day_open=999.0)

    changed = build_factor_scores(contaminated, parameter_set(top_n=1))

    pd.testing.assert_series_equal(
        baseline["evolution_score"], changed["evolution_score"]
    )


def test_absolute_trend_filter_blocks_high_ranked_negative_momentum_stock():
    panel = scored_panel([10.0, 10.1, 10.2])
    top = panel["symbol"].eq("000001")
    panel.loc[top, ["momentum_20", "momentum_60"]] = -0.01
    scored = build_factor_scores(
        panel,
        parameter_set(
            min_score=0.0,
            min_momentum_20=0.0,
            min_momentum_60=0.0,
            min_breakout_distance_20=-1.0,
        ),
    )

    assert not scored.loc[top, "evolution_eligible"].any()
    assert scored.loc[~top, "evolution_eligible"].all()


def test_overextension_filter_blocks_high_ranked_extreme_momentum_stock():
    panel = scored_panel([10.0, 10.1, 10.2])
    top = panel["symbol"].eq("000001")
    scored = build_factor_scores(
        panel,
        parameter_set(min_score=0.0, max_momentum_60=1.5),
    )

    assert not scored.loc[top, "evolution_eligible"].any()
    assert scored.loc[~top, "evolution_eligible"].all()


def test_default_overextension_ceiling_preserves_legacy_extreme_rows():
    panel = scored_panel([10.0, 10.1, 10.2])
    top = panel["symbol"].eq("000001")
    panel.loc[top, "momentum_60"] = 11.0

    scored = build_factor_scores(panel, parameter_set(min_score=0.0))

    assert scored.loc[top, "evolution_eligible"].all()


def test_cross_sectional_momentum_breadth_is_point_in_time():
    panel = scored_panel([10.0, 10.1, 10.2])
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    second = panel["symbol"].eq("000002")
    panel.loc[second & panel["date"].eq(dates[0]), "momentum_20"] = -0.1

    breadth = build_cross_sectional_momentum_breadth(panel, dates)

    assert breadth.iloc[0] == pytest.approx(0.5)
    assert breadth.iloc[1] == pytest.approx(1.0)


def test_breadth_exposure_combines_with_market_regime_and_blocks_warmup():
    dates = pd.bdate_range("2025-01-01", periods=25)
    benchmark = pd.Series(10.0, index=dates)
    benchmark.iloc[20:] = 8.0
    breadth = pd.Series(0.5, index=dates)
    breadth.iloc[20:] = 0.2
    params = parameter_set(
        market_ma_window=20,
        market_below_ma_exposure=0.5,
        breadth_ma_window=20,
        breadth_risk_off_gap=0.0,
        breadth_risk_off_exposure=0.4,
    )

    exposure = build_market_exposure(dates, benchmark, params, breadth)

    assert exposure.iloc[:19].eq(0.0).all()
    assert exposure.iloc[19] == pytest.approx(1.0)
    assert exposure.iloc[20] == pytest.approx(0.4)
    missing = build_market_exposure(dates, benchmark, params, breadth.drop(dates[-1]))
    assert missing.iloc[-1] == pytest.approx(0.0)


def test_signal_executes_next_open_and_does_not_capture_entry_gap():
    panel = scored_panel(
        [10.0, 10.02, 20.04, 20.04],
        [10.0, 10.02, 20.04, 20.04],
    )
    params = parameter_set(
        top_n=1,
        gross_exposure=1.0,
        max_position_weight=1.0,
        rebalance_frequency=1,
        commission_bps=0.0,
        impact_bps=0.0,
    )

    result = run_observation_backtest(panel, params)

    assert result.equity.iloc[1] == pytest.approx(params["initial_capital"])
    assert result.equity.iloc[2] == pytest.approx(params["initial_capital"] * 2.0)


def test_limit_up_open_blocks_new_position():
    panel = scored_panel([10.0, 11.0, 22.0, 22.0], [10.0, 11.0, 22.0, 22.0])
    params = parameter_set(
        top_n=1,
        gross_exposure=1.0,
        max_position_weight=1.0,
        rebalance_frequency=1,
        commission_bps=0.0,
        impact_bps=0.0,
        max_buy_open_gap=1.0,
    )

    result = run_observation_backtest(panel, params)

    assert result.execution_counts["blocked_limit_up_buys"] >= 1
    assert result.equity.iloc[-1] == pytest.approx(params["initial_capital"])


def test_double_cost_run_cannot_outperform_same_trades_at_base_cost():
    panel = scored_panel([10.0, 10.01, 10.02, 10.03])
    params = parameter_set(
        top_n=1,
        gross_exposure=1.0,
        max_position_weight=1.0,
        rebalance_frequency=1,
    )

    base = run_observation_backtest(panel, params, cost_multiplier=1.0)
    stressed = run_observation_backtest(panel, params, cost_multiplier=2.0)

    assert stressed.equity.iloc[-1] <= base.equity.iloc[-1]


def test_market_exposure_is_point_in_time_and_ignores_future_benchmark_values():
    dates = pd.bdate_range("2025-01-01", periods=180)
    history = pd.Series(
        np.linspace(100.0, 120.0, 150), index=dates[:150]
    )
    future = pd.Series(np.linspace(80.0, 60.0, 30), index=dates[150:])
    params = parameter_set(
        market_ma_window=60,
        market_below_ma_exposure=0.5,
        market_risk_off_drawdown_20d=-0.08,
        market_crash_exposure=0.0,
    )

    before = build_market_exposure(dates[:150], history, params)
    after = build_market_exposure(dates, pd.concat([history, future]), params)

    pd.testing.assert_series_equal(before, after.loc[dates[:150]])


def test_market_exposure_applies_crash_floor_and_blocks_missing_quote():
    dates = pd.bdate_range("2025-01-01", periods=121)
    benchmark = pd.Series([100.0] * 120 + [80.0], index=dates)
    params = parameter_set(
        market_ma_window=120,
        market_below_ma_exposure=0.5,
        market_risk_off_drawdown_20d=-0.10,
        market_crash_exposure=0.2,
    )

    exposure = build_market_exposure(dates, benchmark, params)
    missing = build_market_exposure(dates, benchmark.drop(dates[-1]), params)

    assert exposure.loc[dates[-2]] == pytest.approx(1.0)
    assert exposure.loc[dates[-1]] == pytest.approx(0.2)
    assert missing.loc[dates[-1]] == pytest.approx(0.0)


def test_active_market_regime_requires_benchmark_data():
    params = parameter_set(market_below_ma_exposure=0.5)

    with pytest.raises(ValueError, match="require benchmark"):
        build_market_exposure(pd.bdate_range("2025-01-01", periods=30), None, params)


def test_market_regime_scales_next_open_position():
    opens = [10.0] * 122 + [20.0] * 3
    panel = scored_panel(opens)
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    benchmark = pd.Series([100.0] * 120 + [80.0] * 5, index=dates)
    common = {
        "top_n": 1,
        "gross_exposure": 1.0,
        "max_position_weight": 1.0,
        "rebalance_frequency": 1,
        "commission_bps": 0.0,
        "impact_bps": 0.0,
    }
    baseline = run_observation_backtest(panel, parameter_set(**common))
    defensive = run_observation_backtest(
        panel,
        parameter_set(
            **common,
            market_ma_window=120,
            market_below_ma_exposure=0.5,
            market_risk_off_drawdown_20d=-0.10,
            market_crash_exposure=0.2,
        ),
        benchmark=benchmark,
    )

    assert defensive.market_exposure.loc[dates[120]] == pytest.approx(0.2)
    assert defensive.equity.iloc[-1] < baseline.equity.iloc[-1]


def test_portfolio_stop_exits_next_open_and_limits_later_losses():
    opens = [10.0, 10.0]
    for _ in range(8):
        opens.append(opens[-1] * 0.94)
    panel = scored_panel(opens)
    common = {
        "top_n": 1,
        "gross_exposure": 1.0,
        "max_position_weight": 1.0,
        "rebalance_frequency": 1,
        "commission_bps": 0.0,
        "impact_bps": 0.0,
    }

    baseline = run_observation_backtest(panel, parameter_set(**common))
    stopped = run_observation_backtest(
        panel,
        parameter_set(
            **common,
            portfolio_stop_drawdown=-0.10,
            portfolio_stop_cooldown_sessions=5,
        ),
    )

    assert stopped.execution_counts["portfolio_stop_triggers"] >= 1
    assert stopped.equity.iloc[-1] > baseline.equity.iloc[-1]


def test_walk_forward_folds_require_point_in_time_warmup():
    dates = pd.bdate_range("2025-01-01", periods=180)
    periods = EvolutionPeriods(
        research_start=dates[0],
        selection_start=dates[80],
        selection_end=dates[159],
        holdout_start=dates[160],
        holdout_end=dates[-1],
        fold_sessions=20,
        step_sessions=20,
        warmup_sessions=61,
    )

    folds = build_walk_forward_folds(dates, periods)

    assert len(folds) == 4
    assert folds[0].start == dates[80]
    assert folds[-1].end == dates[159]


def test_benchmark_return_requires_full_evaluation_window_coverage():
    benchmark = pd.Series(
        [100.0, 110.0],
        index=pd.to_datetime(["2026-01-01", "2026-06-29"]),
    )

    covered = benchmark_total_return(
        benchmark, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-29")
    )
    stale = benchmark_total_return(
        benchmark, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-07-13")
    )

    assert covered == pytest.approx(0.10)
    assert stale is None


def test_dry_run_writes_research_artifacts_but_not_shadow_state(tmp_path):
    panel = history_panel()
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    config = load_evolution_config(CONFIG_PATH)
    periods = EvolutionPeriods(
        research_start=dates[0],
        selection_start=dates[80],
        selection_end=dates[179],
        holdout_start=dates[180],
        holdout_end=dates[-1],
        fold_sessions=20,
        step_sessions=20,
        warmup_sessions=61,
    )
    config = dataclasses.replace(
        config,
        periods=periods,
        promotion=PromotionPolicy(
            min_folds=1,
            min_filled_trades_per_fold=1,
            min_positive_fold_ratio=0.0,
            min_mean_return_improvement=0.0,
            max_drawdown_floor=-1.0,
            max_drawdown_worsening=1.0,
            max_turnover_ratio=10.0,
            max_pnl_concentration=1.0,
        ),
        search_groups=(
            SearchGroup(
                group_id="smoke",
                hypothesis="smoke",
                candidates=(SearchCandidate("top6", {"top_n": 6}),),
            ),
        ),
    )
    state_path = tmp_path / "state" / "multifactor.json"

    outcome = run_evolution(
        panel=panel,
        benchmark=None,
        config=config,
        output_root=tmp_path / "runs",
        state_path=state_path,
        run_id="dry-run-smoke",
        dry_run=True,
        promote_shadow=False,
    )

    run_dir = Path(outcome["run_dir"])
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "candidate_scores.csv").exists()
    assert (run_dir / "fold_metrics.csv").exists()
    assert outcome["research_only"] is True
    assert outcome["trade_instruction"] is False
    assert not state_path.exists()
    scores = pd.read_csv(run_dir / "candidate_scores.csv")
    assert scores.iloc[0]["group_id"] == "__baseline__"
    assert scores.iloc[0]["status"] == "incumbent"
    assert scores.iloc[0]["portfolio_stop_triggers"] == 0


def test_stale_benchmark_blocks_holdout_and_shadow_write(tmp_path, monkeypatch):
    panel = history_panel()
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    config = load_evolution_config(CONFIG_PATH)
    config = dataclasses.replace(
        config,
        periods=EvolutionPeriods(
            research_start=dates[0],
            selection_start=dates[80],
            selection_end=dates[179],
            holdout_start=dates[180],
            holdout_end=dates[-1],
            fold_sessions=20,
            step_sessions=20,
            warmup_sessions=61,
        ),
        search_groups=(
            SearchGroup(
                group_id="forced_winner",
                hypothesis="verify stale benchmark gate",
                candidates=(SearchCandidate("top2", {"top_n": 2}),),
            ),
        ),
    )
    benchmark = pd.Series(
        np.linspace(100.0, 120.0, 180), index=dates[:180]
    )
    monkeypatch.setattr(
        evolution_runner,
        "evaluate_candidate",
        lambda *args, **kwargs: PromotionDecision(
            "eligible_for_shadow", (), {"forced": {"passed": True}}
        ),
    )
    monkeypatch.setattr(
        evolution_runner, "_positive_excess_ratio", lambda evaluation: 1.0
    )
    monkeypatch.setattr(
        evolution_runner,
        "evaluate_holdout",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("stale benchmark must block before holdout evaluation")
        ),
    )
    state_path = tmp_path / "state" / "multifactor.json"

    outcome = run_evolution(
        panel=panel,
        benchmark=benchmark,
        config=config,
        output_root=tmp_path / "runs",
        state_path=state_path,
        run_id="stale-benchmark",
        dry_run=False,
        promote_shadow=True,
    )

    assert outcome["status"] == "holdout_blocked_benchmark_coverage"
    assert outcome["holdout_gates"]["benchmark_coverage"]["passed"] is False
    assert outcome["shadow_write_status"] == "not_authorized"
    assert not state_path.exists()
