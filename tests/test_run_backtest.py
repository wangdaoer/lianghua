import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_backtest import (
    BacktestEngine,
    StrategyConfig,
    annualized_return,
    clean_symbol,
    load_config,
    load_prices,
    prepare_prices,
)


def test_annualized_return_uses_trading_observations_not_calendar_days():
    dates = pd.bdate_range("2025-01-02", periods=253)
    equity = pd.Series(np.linspace(1.0, 1.10, len(dates)), index=dates)

    assert annualized_return(equity) == pytest.approx(0.10)


def make_config(**overrides):
    values = {
        "initial_capital": 1_000_000.0,
        "start_date": None,
        "end_date": None,
        "universe_top_n_by_mcap": 100,
        "universe_dynamic_top_n": 0,
        "universe_selection_mode": "none",
        "universe_selection_min_history": 120,
        "universe_selection_lag_days": 0,
        "max_abs_daily_return": 0.35,
        "execution_model": "close_to_close",
        "block_limit_up_buys": False,
        "block_limit_down_sells": False,
        "limit_buffer": 0.995,
        "max_buy_open_gap": None,
        "short_window": 3,
        "long_window": 5,
        "breakout_window": 5,
        "acceleration_window": 3,
        "breakout_threshold": 0.01,
        "vol_window": 3,
        "signal_type": "trend_breakout_accel",
        "top_n_long": 3,
        "top_n_short": 0,
        "signal_only_long_breakout": False,
        "breakout_up_weight": 0.55,
        "breakout_dn_weight": -0.35,
        "trend_weight": 0.25,
        "accel_weight": 0.25,
        "signal_noise_clip": 5.0,
        "rebalance_frequency": 1,
        "long_exposure": 0.60,
        "short_exposure": 0.0,
        "max_position_weight": 0.30,
        "use_net_neutral": False,
        "allow_short": False,
        "leverage": 1.0,
        "max_drawdown": 0.30,
        "drawdown_cooldown_days": 5,
        "target_annualized_vol": 0.30,
        "vol_lookback": 3,
        "min_signal_non_na_ratio": 0.0,
        "commission_bps": 0.0,
        "impact_bps": 0.0,
        "output_dir": "outputs/test",
    }
    values.update(overrides)
    return StrategyConfig(**values)


def test_strategy_config_defaults_dynamic_universe_lag_to_zero():
    cfg = load_config(Path("configs/high_risk_strategy_high_return_v2_dynamic_universe.yaml"))

    assert cfg.universe_selection_lag_days == 0


def test_dynamic_universe_lag_uses_previous_trading_day_membership():
    engine = BacktestEngine(make_config(
        universe_dynamic_top_n=1,
        universe_selection_mode="high_return",
        universe_selection_min_history=2,
        universe_selection_lag_days=1,
    ))
    dates = pd.DatetimeIndex(
        ["2026-01-02", "2026-01-05", "2026-01-09", "2026-01-14"]
    ).append(pd.bdate_range("2026-01-15", periods=61))
    assert dates[0].day_name() == "Friday"
    assert dates[1].day_name() == "Monday"
    assert dates.to_series().diff().max() >= pd.Timedelta(days=5)
    close = pd.DataFrame({
        "000001": np.linspace(10, 30, 65),
        "000002": np.r_[np.linspace(10, 11, 64), 40],
    }, index=dates)
    amount = pd.DataFrame(1_000_000.0, index=dates, columns=close.columns)

    lagged = engine._precompute_universe_panel(close, amount)
    same_day = BacktestEngine(make_config(
        universe_dynamic_top_n=1,
        universe_selection_mode="high_return",
        universe_selection_min_history=2,
        universe_selection_lag_days=0,
    ))._precompute_universe_panel(close, amount)

    pd.testing.assert_series_equal(lagged.iloc[-1], same_day.iloc[-2], check_names=False)


def test_load_prices_preserves_six_digit_symbols_across_csv_round_trip(tmp_path):
    path = tmp_path / "symbols.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="D"),
            "symbol": ["000001", "000100", 1, 100],
            "open": [10.0] * 4,
            "high": [10.0] * 4,
            "low": [10.0] * 4,
            "close": [10.0] * 4,
            "volume": [1_000.0] * 4,
            "amount": [10_000.0] * 4,
        }
    ).to_csv(path, index=False)

    loaded = load_prices(path, None, None)

    assert loaded["symbol"].tolist() == ["000001", "000100", "000001", "000100"]
    assert clean_symbol(1) == "000001"
    assert clean_symbol(100.0) == "000100"


def test_prepare_prices_accepts_a_validated_in_memory_panel():
    panel = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "symbol": [1, 100.0],
            "open": [10.0, 11.0],
            "high": [10.0, 11.0],
            "low": [10.0, 11.0],
            "close": [10.0, 11.0],
            "volume": [1_000.0, 1_000.0],
            "amount": [10_000.0, 11_000.0],
        }
    )

    prepared = prepare_prices(panel, None, None)

    assert prepared["symbol"].tolist() == ["000001", "000100"]
    assert pd.api.types.is_datetime64_any_dtype(prepared["date"])


def test_load_prices_keeps_legacy_vendor_symbol_compatibility(tmp_path):
    path = tmp_path / "vendor_symbols.csv"
    pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "symbol": ["000001.SZ", "SH600000"],
            "open": [10.0, 10.0],
            "high": [10.0, 10.0],
            "low": [10.0, 10.0],
            "close": [10.0, 10.0],
            "volume": [1_000.0, 1_000.0],
            "amount": [10_000.0, 10_000.0],
        }
    ).to_csv(path, index=False)

    loaded = load_prices(path, None, None)

    assert loaded["symbol"].tolist() == ["000001", "600000"]


def test_load_prices_preserves_legacy_zero_row_filtering(tmp_path):
    path = tmp_path / "interior_zero.csv"
    pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02"],
            "symbol": ["000001", "000001"],
            "open": [0.0, 10.0],
            "high": [0.0, 10.0],
            "low": [0.0, 10.0],
            "close": [0.0, 10.0],
            "volume": [0.0, 1_000.0],
            "amount": [0.0, 10_000.0],
        }
    ).to_csv(path, index=False)

    loaded = load_prices(path, None, None)

    assert loaded["date"].tolist() == [pd.Timestamp("2026-01-02")]


class RunBacktestEngineTest(unittest.TestCase):
    def test_engine_reuses_supplied_signal_and_universe_caches(self):
        dates = pd.date_range("2026-01-01", periods=6, freq="D")
        raw = pd.DataFrame(
            {
                "date": list(dates) * 2,
                "symbol": ["000001"] * len(dates) + ["000002"] * len(dates),
                "open": [10.0] * 12,
                "high": [10.0] * 12,
                "low": [10.0] * 12,
                "close": [10.0] * 12,
                "volume": [1_000.0] * 12,
                "amount": [10_000.0] * 12,
            }
        )
        cache = pd.DataFrame(0.0, index=dates, columns=["000001", "000002"])
        membership = pd.DataFrame(True, index=dates, columns=cache.columns)
        engine = BacktestEngine(make_config())
        engine._precompute_signal_panel = lambda close: self.fail("signal cache recomputed")
        engine._precompute_universe_panel = lambda close, amount: self.fail("universe cache recomputed")

        result = engine.run(raw, signal_cache=cache, universe_cache=membership)

        self.assertEqual(result["metrics"]["trade_days"], 6)

    def test_execution_constraint_counts_accept_new_diagnostic_keys(self):
        engine = BacktestEngine(make_config())

        engine._record_execution_constraint_counts(
            {"blocked_orders_total": 1, "gross_budget_overrun": 1}
        )

        self.assertEqual(engine.execution_constraint_counts["blocked_orders_total"], 1)
        self.assertEqual(engine.execution_constraint_counts["gross_budget_overrun"], 1)

    def test_engine_keeps_missing_amount_unavailable(self):
        engine = BacktestEngine(make_config())
        dates = pd.date_range("2026-01-01", periods=6, freq="D")
        raw = pd.DataFrame(
            {
                "date": dates,
                "symbol": ["000001"] * len(dates),
                "open": [10.0] * len(dates),
                "high": [10.0] * len(dates),
                "low": [10.0] * len(dates),
                "close": [10.0] * len(dates),
                "volume": [1_000.0] * len(dates),
            }
        )
        captured = {}
        original = engine._precompute_universe_panel

        def capture_amount(close, amount):
            captured["amount"] = amount.copy()
            return original(close, amount)

        engine._precompute_universe_panel = capture_amount

        engine.run(raw)

        self.assertTrue(captured["amount"].isna().all().all())

    def test_close_to_close_metrics_report_zero_execution_constraint_counts(self):
        dates = pd.date_range("2026-01-01", periods=6, freq="D")
        raw = pd.DataFrame(
            {
                "date": list(dates) * 2,
                "symbol": ["000001"] * len(dates) + ["000002"] * len(dates),
                "open": [10.0] * 12,
                "high": [10.0] * 12,
                "low": [10.0] * 12,
                "close": [10.0] * 12,
                "volume": [1_000.0] * 12,
                "amount": [10_000.0] * 12,
            }
        )

        result = BacktestEngine(make_config()).run(raw)

        for name in (
            "blocked_limit_up_buys",
            "blocked_limit_down_sells",
            "blocked_open_gap_buys",
            "blocked_orders_total",
        ):
            self.assertEqual(result["metrics"][name], 0)

    def test_next_open_metrics_accumulate_execution_constraint_counts(self):
        engine = BacktestEngine(
            make_config(
                execution_model="next_open",
                block_limit_up_buys=True,
                block_limit_down_sells=True,
                max_buy_open_gap=0.15,
            )
        )
        dates = pd.date_range("2026-01-01", periods=3, freq="D")
        columns = pd.Index(["000001", "000002", "300001"])
        close = pd.DataFrame(10.0, index=dates, columns=columns)
        open_px = close.copy()
        open_px.iloc[2] = [11.0, 9.0, 11.6]
        amount = pd.DataFrame(1_000_000.0, index=dates, columns=columns)
        targets = [
            pd.Series({"000001": 0.0, "000002": 0.2, "300001": 0.0}),
            pd.Series({"000001": 0.2, "000002": 0.0, "300001": 0.2}),
            pd.Series(0.0, index=columns),
        ]

        def build_target(*_args):
            return targets.pop(0), 0, "rebalance"

        engine._build_target = build_target

        result = engine._run_next_open(close, open_px, amount)

        self.assertEqual(
            {name: result["metrics"][name] for name in (
                "blocked_limit_up_buys",
                "blocked_limit_down_sells",
                "blocked_open_gap_buys",
                "blocked_orders_total",
            )},
            {
                "blocked_limit_up_buys": 1,
                "blocked_limit_down_sells": 1,
                "blocked_open_gap_buys": 1,
                "blocked_orders_total": 3,
            },
        )

    def test_next_open_uses_point_in_time_limit_rate_when_present(self):
        engine = BacktestEngine(
            make_config(
                execution_model="next_open",
                block_limit_up_buys=True,
                block_limit_down_sells=True,
            )
        )
        dates = pd.date_range("2026-01-01", periods=3, freq="D")
        columns = pd.Index(["000001"])
        close = pd.DataFrame(10.0, index=dates, columns=columns)
        open_px = close.copy()
        open_px.iloc[2, 0] = 10.5
        amount = pd.DataFrame(1_000_000.0, index=dates, columns=columns)
        limit_rate = pd.DataFrame(0.05, index=dates, columns=columns)
        targets = [
            pd.Series(0.0, index=columns),
            pd.Series(0.2, index=columns),
            pd.Series(0.0, index=columns),
        ]

        def build_target(*_args):
            return targets.pop(0), 0, "rebalance"

        engine._build_target = build_target

        result = engine._run_next_open(
            close,
            open_px,
            amount,
            limit_rate=limit_rate,
        )

        self.assertEqual(result["metrics"]["blocked_limit_up_buys"], 1)
        self.assertEqual(result["metrics"]["blocked_orders_total"], 1)
        self.assertAlmostEqual(float(result["positions"]["000001"]), 0.0)

    def test_target_weights_redistributes_clipped_long_exposure_to_available_names(self):
        engine = BacktestEngine(
            make_config(
                top_n_long=3,
                long_exposure=0.60,
                max_position_weight=0.30,
                leverage=1.0,
            )
        )
        signal = pd.Series({"000001": 100.0, "000002": 1.0, "000003": 1.0})

        target = engine._target_weights(signal)

        self.assertAlmostEqual(float(target.sum()), 0.60)
        self.assertAlmostEqual(float(target["000001"]), 0.30)
        self.assertAlmostEqual(float(target["000002"]), 0.15)
        self.assertAlmostEqual(float(target["000003"]), 0.15)

    def test_cap_and_redistribute_preserves_feasible_exposure_across_multiple_caps(self):
        engine = BacktestEngine(make_config(max_position_weight=0.14))
        signal = pd.Series(
            [100.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            index=[f"{value:06d}" for value in range(1, 11)],
        )

        target = engine._cap_and_redistribute(signal, exposure=1.0)

        self.assertAlmostEqual(float(target.sum()), 1.0)
        self.assertLessEqual(float(target.max()), 0.14 + 1e-12)

        ranked_target = target.loc[signal.sort_values(ascending=False).index]
        self.assertTrue((np.diff(ranked_target.to_numpy()) <= 1e-12).all())

    def test_target_weights_enforces_position_cap_after_leverage(self):
        engine = BacktestEngine(
            make_config(
                top_n_long=10,
                long_exposure=1.0,
                max_position_weight=0.14,
                leverage=2.0,
            )
        )
        signal = pd.Series(
            list(range(20, 10, -1)),
            index=[f"{value:06d}" for value in range(1, 11)],
            dtype=float,
        )

        target = engine._target_weights(signal)

        self.assertLessEqual(float(target.abs().max()), 0.14 + 1e-12)
        self.assertAlmostEqual(float(target.abs().sum()), 1.40)

    def test_target_weights_accepts_capacity_shortfall_when_candidates_are_few(self):
        engine = BacktestEngine(
            make_config(
                top_n_long=3,
                long_exposure=1.0,
                max_position_weight=0.14,
                leverage=2.0,
            )
        )
        signal = pd.Series({"000001": 3.0, "000002": 2.0, "000003": 1.0})

        target = engine._target_weights(signal)

        self.assertLessEqual(float(target.abs().max()), 0.14 + 1e-12)
        self.assertAlmostEqual(float(target.abs().sum()), 0.42)

    def test_target_weights_preserves_net_neutrality_when_final_cap_is_binding(self):
        engine = BacktestEngine(
            make_config(
                top_n_long=2,
                top_n_short=2,
                long_exposure=0.80,
                short_exposure=0.80,
                max_position_weight=0.30,
                leverage=2.0,
                allow_short=True,
                use_net_neutral=True,
            )
        )
        signal = pd.Series(
            {"000001": 10.0, "000002": 5.0, "000003": -5.0, "000004": -10.0}
        )

        target = engine._target_weights(signal)

        self.assertLessEqual(float(target.abs().max()), 0.30 + 1e-12)
        self.assertAlmostEqual(float(target.sum()), 0.0)
        self.assertAlmostEqual(float(target[target > 0].sum()), 0.60)
        self.assertAlmostEqual(float(target[target < 0].sum()), -0.60)

    def test_final_position_cap_balances_asymmetric_net_neutral_capacity(self):
        engine = BacktestEngine(
            make_config(max_position_weight=0.25, allow_short=True, use_net_neutral=True)
        )
        weights = pd.Series(
            {"000001": 0.60, "000002": 0.30, "000003": 0.10, "000004": -0.80}
        )

        target = engine._enforce_final_position_cap(weights)

        self.assertLessEqual(float(target.abs().max()), 0.25 + 1e-12)
        self.assertAlmostEqual(float(target[target > 0].sum()), 0.25)
        self.assertAlmostEqual(float(target[target < 0].sum()), -0.25)
        self.assertAlmostEqual(float(target.sum()), 0.0)

    def test_final_position_cap_cleans_non_finite_and_near_zero_weights(self):
        engine = BacktestEngine(make_config(max_position_weight=0.30))
        weights = pd.Series(
            {
                "000001": 0.40,
                "000002": -0.40,
                "000003": np.nan,
                "000004": np.inf,
                "000005": 1e-16,
            }
        )

        target = engine._enforce_final_position_cap(weights)

        self.assertEqual(set(target.index), {"000001", "000002"})
        self.assertAlmostEqual(float(target["000001"]), 0.30)
        self.assertAlmostEqual(float(target["000002"]), -0.30)

        zero_cap_target = BacktestEngine(
            make_config(max_position_weight=0.0)
        )._enforce_final_position_cap(pd.Series({"000001": 0.50}))
        self.assertTrue(zero_cap_target.empty)

    def test_mean_reversion_signal_does_not_forward_fill_missing_prices(self):
        engine = BacktestEngine(
            make_config(
                signal_type="mean_reversion_mr",
                long_window=2,
                vol_window=2,
            )
        )
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        close = pd.DataFrame(
            {
                "000001": [10.0, 11.0, np.nan, 13.0, 14.0],
            },
            index=dates,
        )

        signal = engine._signal_mean_reversion(close)

        self.assertTrue(pd.isna(signal["000001"]))

    def test_scale_for_risk_uses_current_simplified_uncorrelated_volatility_model(self):
        engine = BacktestEngine(make_config(target_annualized_vol=0.10, vol_lookback=2))
        weights = pd.Series({"000001": 0.50, "000002": 0.50})
        returns = pd.DataFrame(
            {
                "000001": [0.00, 0.02, -0.02, 0.02],
                "000002": [0.00, -0.02, 0.02, -0.02],
            }
        )

        scaled = engine._scale_for_risk(weights, returns, idx=3)
        ann_vol = returns.iloc[1:4].std(ddof=0) * np.sqrt(252)
        port_vol = np.sqrt((weights.abs() * ann_vol.fillna(0.0)).pow(2).sum())
        expected_scale = min(1.0, 0.10 / port_vol)

        pd.testing.assert_series_equal(scaled, weights * expected_scale)


if __name__ == "__main__":
    unittest.main()
