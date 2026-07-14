import unittest

import numpy as np
import pandas as pd

from run_backtest import BacktestEngine, StrategyConfig


def make_config(**overrides):
    values = {
        "initial_capital": 1_000_000.0,
        "start_date": None,
        "end_date": None,
        "universe_top_n_by_mcap": 100,
        "universe_dynamic_top_n": 0,
        "universe_selection_mode": "none",
        "universe_selection_min_history": 120,
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


class RunBacktestEngineTest(unittest.TestCase):
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
