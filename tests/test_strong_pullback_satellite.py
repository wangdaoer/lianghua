import json
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_strong_pullback_satellite import (
    apply_rebound_exit,
    build_regime_leverage_schedule,
    observable_ic_training_window,
    parse_args,
    run_satellite_walk_forward,
    select_target_weights,
    should_apply_rebound_exit,
)
from execution_rules import next_open_return_label
from train_next_open_rank_model import daily_ic, normalize_weights


class StrongPullbackSatelliteTest(unittest.TestCase):
    def test_cli_accepts_regime_gated_leverage_controls(self):
        argv = [
            "run_strong_pullback_satellite.py",
            "--data", "panel.csv",
            "--benchmark", "benchmark.csv",
            "--regime-strong-leverage", "0.75",
            "--regime-exceptional-leverage", "0.90",
            "--regime-strong-breadth-threshold", "0.55",
            "--regime-exceptional-breadth-threshold", "0.70",
            "--regime-strong-volatility-max", "0.28",
            "--regime-exceptional-volatility-max", "0.20",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.regime_strong_leverage, 0.75)
        self.assertEqual(args.regime_exceptional_leverage, 0.90)
        self.assertEqual(args.regime_strong_breadth_threshold, 0.55)
        self.assertEqual(args.regime_exceptional_breadth_threshold, 0.70)
        self.assertEqual(args.regime_strong_volatility_max, 0.28)
        self.assertEqual(args.regime_exceptional_volatility_max, 0.20)

    def test_cli_rejects_incoherent_regime_gated_leverage_controls(self):
        argv = [
            "run_strong_pullback_satellite.py",
            "--data", "panel.csv",
            "--benchmark", "benchmark.csv",
            "--regime-strong-leverage", "0.90",
            "--regime-exceptional-leverage", "0.75",
            "--regime-strong-breadth-threshold", "0.70",
            "--regime-exceptional-breadth-threshold", "0.55",
            "--regime-strong-volatility-max", "0.18",
            "--regime-exceptional-volatility-max", "0.28",
        ]

        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            parse_args()

    @staticmethod
    def _regime_inputs(rising_symbols: int) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
        dates = pd.bdate_range("2025-01-02", periods=140)
        steps = np.arange(len(dates), dtype=float)
        benchmark = pd.Series(100.0 * np.power(1.002, steps), index=dates)
        close = pd.DataFrame(index=dates)
        for index in range(10):
            if index < rising_symbols:
                close[f"{index + 1:06d}"] = 10.0 * np.power(1.003, steps)
            else:
                close[f"{index + 1:06d}"] = 10.0
        market_exposure = pd.Series(1.0, index=dates)
        return benchmark, close, market_exposure

    def test_regime_schedule_uses_strong_leverage_when_breadth_is_not_exceptional(self):
        benchmark, close, market_exposure = self._regime_inputs(rising_symbols=6)

        schedule = build_regime_leverage_schedule(
            benchmark,
            close,
            market_exposure,
            base_leverage=0.60,
            strong_leverage=0.75,
            exceptional_leverage=0.90,
            strong_breadth_threshold=0.55,
            exceptional_breadth_threshold=0.80,
            strong_volatility_max=0.25,
            exceptional_volatility_max=0.18,
        )

        self.assertEqual(schedule.iloc[-1]["risk_regime"], "strong")
        self.assertAlmostEqual(float(schedule.iloc[-1]["target_leverage"]), 0.75)

    def test_regime_schedule_uses_exceptional_leverage_only_for_broad_low_volatility_trend(self):
        benchmark, close, market_exposure = self._regime_inputs(rising_symbols=9)

        schedule = build_regime_leverage_schedule(
            benchmark,
            close,
            market_exposure,
            base_leverage=0.60,
            strong_leverage=0.75,
            exceptional_leverage=0.90,
            strong_breadth_threshold=0.55,
            exceptional_breadth_threshold=0.80,
            strong_volatility_max=0.25,
            exceptional_volatility_max=0.18,
        )

        self.assertEqual(schedule.iloc[-1]["risk_regime"], "exceptional")
        self.assertAlmostEqual(float(schedule.iloc[-1]["target_leverage"]), 0.90)

    def test_regime_schedule_does_not_use_prices_after_signal_date(self):
        benchmark, close, market_exposure = self._regime_inputs(rising_symbols=9)
        signal_date = benchmark.index[110]
        changed_benchmark = benchmark.copy()
        changed_close = close.copy()
        changed_benchmark.loc[changed_benchmark.index > signal_date] *= 0.50
        changed_close.loc[changed_close.index > signal_date] *= np.linspace(
            1.0, 0.20, int((changed_close.index > signal_date).sum())
        )[:, None]
        kwargs = {
            "base_leverage": 0.60,
            "strong_leverage": 0.75,
            "exceptional_leverage": 0.90,
            "strong_breadth_threshold": 0.55,
            "exceptional_breadth_threshold": 0.80,
            "strong_volatility_max": 0.25,
            "exceptional_volatility_max": 0.18,
        }

        original = build_regime_leverage_schedule(
            benchmark, close, market_exposure, **kwargs
        )
        changed = build_regime_leverage_schedule(
            changed_benchmark, changed_close, market_exposure, **kwargs
        )

        pd.testing.assert_frame_equal(
            original.loc[:signal_date], changed.loc[:signal_date]
        )

    def test_regime_breadth_ignores_symbols_without_ma60_history(self):
        benchmark, close, market_exposure = self._regime_inputs(rising_symbols=6)
        unavailable = pd.DataFrame(
            np.nan,
            index=close.index,
            columns=[f"9{index:05d}" for index in range(90)],
        )
        expanded_close = pd.concat([close, unavailable], axis=1)
        kwargs = {
            "base_leverage": 0.60,
            "strong_leverage": 0.75,
            "exceptional_leverage": 0.90,
            "strong_breadth_threshold": 0.55,
            "exceptional_breadth_threshold": 0.80,
            "strong_volatility_max": 0.25,
            "exceptional_volatility_max": 0.18,
        }

        original = build_regime_leverage_schedule(
            benchmark, close, market_exposure, **kwargs
        )
        expanded = build_regime_leverage_schedule(
            benchmark, expanded_close, market_exposure, **kwargs
        )

        self.assertAlmostEqual(
            float(original.iloc[-1]["breadth_ma60"]),
            float(expanded.iloc[-1]["breadth_ma60"]),
        )
        self.assertEqual(expanded.iloc[-1]["risk_regime"], "strong")

    def test_regime_schedule_rejects_partial_controls(self):
        benchmark, close, market_exposure = self._regime_inputs(rising_symbols=6)

        with self.assertRaisesRegex(ValueError, "together"):
            build_regime_leverage_schedule(
                benchmark,
                close,
                market_exposure,
                base_leverage=0.60,
                strong_leverage=0.75,
            )

    def test_regime_schedule_requires_benchmark_when_enabled(self):
        _, close, market_exposure = self._regime_inputs(rising_symbols=6)
        missing_benchmark = pd.Series(np.nan, index=close.index)

        with self.assertRaisesRegex(ValueError, "benchmark"):
            build_regime_leverage_schedule(
                missing_benchmark,
                close,
                market_exposure,
                base_leverage=0.60,
                strong_leverage=0.75,
                exceptional_leverage=0.90,
                strong_breadth_threshold=0.55,
                exceptional_breadth_threshold=0.70,
                strong_volatility_max=0.28,
                exceptional_volatility_max=0.20,
            )

    def test_non_rebalance_regime_change_resizes_positions_at_next_open(self):
        dates = pd.bdate_range("2025-01-02", periods=75)
        symbols = pd.Index(["000001", "000002"])
        close = pd.DataFrame(10.0, index=dates, columns=symbols)
        regime_schedule = pd.DataFrame(
            {
                "target_leverage": 0.60,
                "risk_regime": "base",
                "trend_on": False,
                "breadth_ma60": 0.60,
                "benchmark_return_20d": 0.05,
                "benchmark_volatility_20d": 0.10,
            },
            index=dates,
        )
        regime_schedule.loc[dates[67]:, "target_leverage"] = 0.90
        regime_schedule.loc[dates[67]:, "risk_regime"] = "exceptional"
        candidates = pd.DataFrame(
            {
                "score": [1.0, 0.9],
                "close": [10.0, 10.0],
                "return_5d": [0.0, 0.0],
                "return_20d": [0.1, 0.1],
                "return_60d": [0.2, 0.2],
                "prior_return_20_before_pullback": [0.1, 0.1],
                "prior_return_60_before_pullback": [0.2, 0.2],
                "raw_distance_ma60": [0.1, 0.1],
                "avg_amount_20d": [50_000_000.0, 50_000_000.0],
            },
            index=symbols,
        )

        with patch(
            "run_strong_pullback_satellite.build_daily_candidates",
            return_value=candidates,
        ):
            nav, _, _, _ = run_satellite_walk_forward(
                close=close,
                open_px=close,
                high=close,
                low=close,
                amount=pd.DataFrame(50_000_000.0, index=dates, columns=symbols),
                train_days=65,
                retrain_frequency=5,
                top_n=2,
                rebalance_frequency=5,
                max_position_weight=0.50,
                leverage=0.60,
                min_score=None,
                commission_bps=0.0,
                impact_bps=0.0,
                max_buy_open_gap=0.05,
                limit_buffer=1.0,
                market_exposure=pd.Series(1.0, index=dates),
                initial_capital=1_000_000.0,
                filter_kwargs={},
                regime_schedule=regime_schedule,
            )

        exceptional_row = nav.loc[nav["risk_regime"].eq("exceptional")].iloc[0]
        self.assertAlmostEqual(float(exceptional_row["gross_exposure"]), 0.90)
        self.assertAlmostEqual(float(exceptional_row["target_gross_exposure"]), 0.90)
        self.assertAlmostEqual(
            float(exceptional_row["post_overlay_target_gross_exposure"]), 0.90
        )
        self.assertAlmostEqual(
            float(exceptional_row["budget_gross_exposure_shortfall"]), 0.0
        )
        self.assertAlmostEqual(
            float(exceptional_row["execution_gross_exposure_shortfall"]), 0.0
        )
        self.assertAlmostEqual(
            float(exceptional_row["execution_gross_exposure_overshoot"]), 0.0
        )

    def test_rebound_exit_is_budget_shortfall_not_execution_failure(self):
        dates = pd.bdate_range("2025-01-02", periods=75)
        symbols = pd.Index(["000001", "000002"])
        close = pd.DataFrame(10.0, index=dates, columns=symbols)
        candidates = pd.DataFrame(
            {
                "score": [1.0, 0.9],
                "close": [10.0, 10.0],
                "return_5d": [0.0, 0.0],
                "return_20d": [0.1, 0.1],
                "return_60d": [0.2, 0.2],
                "prior_return_20_before_pullback": [0.1, 0.1],
                "prior_return_60_before_pullback": [0.2, 0.2],
                "raw_distance_ma60": [0.1, 0.1],
                "avg_amount_20d": [50_000_000.0, 50_000_000.0],
            },
            index=symbols,
        )

        with patch(
            "run_strong_pullback_satellite.build_daily_candidates",
            return_value=candidates,
        ):
            nav, _, _, _ = run_satellite_walk_forward(
                close=close,
                open_px=close,
                high=close,
                low=close,
                amount=pd.DataFrame(50_000_000.0, index=dates, columns=symbols),
                train_days=65,
                retrain_frequency=5,
                top_n=2,
                rebalance_frequency=5,
                max_position_weight=0.50,
                leverage=0.60,
                min_score=None,
                commission_bps=0.0,
                impact_bps=0.0,
                max_buy_open_gap=0.05,
                limit_buffer=1.0,
                market_exposure=pd.Series(1.0, index=dates),
                initial_capital=1_000_000.0,
                filter_kwargs={},
                rebound_exit_return=-0.001,
                rebound_exit_scale=0.0,
            )

        exit_row = nav.loc[nav["rebound_exit_count"].gt(0)].iloc[0]
        self.assertAlmostEqual(float(exit_row["target_gross_exposure"]), 0.60)
        self.assertAlmostEqual(
            float(exit_row["post_overlay_target_gross_exposure"]), 0.0
        )
        self.assertAlmostEqual(
            float(exit_row["budget_gross_exposure_shortfall"]), 0.60
        )
        self.assertAlmostEqual(
            float(exit_row["execution_gross_exposure_shortfall"]), 0.0
        )
        self.assertAlmostEqual(
            float(exit_row["execution_gross_exposure_overshoot"]), 0.0
        )

    def test_walk_forward_uses_regime_target_leverage_and_records_audit_fields(self):
        benchmark, close, market_exposure = self._regime_inputs(rising_symbols=6)
        schedule = build_regime_leverage_schedule(
            benchmark,
            close,
            market_exposure,
            base_leverage=0.60,
            strong_leverage=0.75,
            exceptional_leverage=0.90,
            strong_breadth_threshold=0.55,
            exceptional_breadth_threshold=0.80,
            strong_volatility_max=0.25,
            exceptional_volatility_max=0.18,
        )

        nav, _, trades, _ = run_satellite_walk_forward(
            close=close,
            open_px=close,
            high=close * 1.01,
            low=close * 0.99,
            amount=pd.DataFrame(50_000_000.0, index=close.index, columns=close.columns),
            train_days=65,
            retrain_frequency=5,
            top_n=2,
            rebalance_frequency=1,
            max_position_weight=0.50,
            leverage=0.60,
            min_score=None,
            commission_bps=0.0,
            impact_bps=0.0,
            max_buy_open_gap=0.05,
            limit_buffer=1.0,
            market_exposure=market_exposure,
            initial_capital=1_000_000.0,
            filter_kwargs={},
            regime_schedule=schedule,
        )

        self.assertEqual(nav.iloc[-1]["risk_regime"], "strong")
        self.assertAlmostEqual(float(nav.iloc[-1]["target_leverage"]), 0.75)
        self.assertAlmostEqual(float(trades.iloc[-1]["target_leverage"]), 0.75)
        self.assertIn("target_gross_exposure", nav.columns)
        self.assertIn("post_overlay_target_gross_exposure", nav.columns)
        self.assertIn("budget_gross_exposure_shortfall", nav.columns)
        self.assertIn("execution_gross_exposure_shortfall", nav.columns)
        self.assertIn("execution_gross_exposure_overshoot", nav.columns)
        self.assertIn("target_gross_exposure", trades.columns)
        self.assertIn("post_overlay_target_gross_exposure", trades.columns)
        self.assertIn("budget_gross_exposure_shortfall", trades.columns)
        self.assertIn("execution_gross_exposure_shortfall", trades.columns)
        self.assertIn("execution_gross_exposure_overshoot", trades.columns)
        self.assertIn("breadth_ma60", nav.columns)
        self.assertIn("benchmark_volatility_20d", nav.columns)

    def test_signal_close_ic_fit_excludes_unobservable_future_open_labels(self):
        dates = pd.bdate_range("2026-01-05", periods=16)
        symbols = pd.Index([f"{index + 1:06d}" for index in range(40)])
        symbol_axis = np.arange(len(symbols), dtype=float)
        open_px = pd.DataFrame(
            [10.0 + symbol_axis * 0.05 + day * (0.01 + symbol_axis * 0.0002)
             for day in range(len(dates))],
            index=dates,
            columns=symbols,
        )
        features = {
            "ascending": pd.DataFrame(
                np.tile(symbol_axis, (len(dates), 1)), index=dates, columns=symbols
            ),
            "alternating": pd.DataFrame(
                np.tile((symbol_axis * 17.0) % 41.0, (len(dates), 1)),
                index=dates,
                columns=symbols,
            ),
        }
        signal_index = 10
        train_days = 5

        changed_open = open_px.copy()
        changed_open.iloc[signal_index + 1] *= pd.Series(
            np.linspace(1.15, 0.85, len(symbols)), index=symbols
        )
        changed_open.iloc[signal_index + 2 :] *= pd.Series(
            np.linspace(0.80, 1.20, len(symbols)), index=symbols
        )
        base_ic = daily_ic(features, next_open_return_label(open_px))
        changed_ic = daily_ic(features, next_open_return_label(changed_open))

        base_window = observable_ic_training_window(
            base_ic, signal_index=signal_index, train_days=train_days
        )
        changed_window = observable_ic_training_window(
            changed_ic, signal_index=signal_index, train_days=train_days
        )
        base_weights = normalize_weights(base_window.mean())
        changed_weights = normalize_weights(changed_window.mean())
        base_signal = sum(
            features[name].iloc[signal_index] * weight
            for name, weight in base_weights.items()
        )
        changed_signal = sum(
            features[name].iloc[signal_index] * weight
            for name, weight in changed_weights.items()
        )

        self.assertEqual(len(base_window), train_days)
        self.assertEqual(base_window.index[-1], dates[signal_index - 2])
        self.assertFalse(base_ic.iloc[signal_index - 1].equals(changed_ic.iloc[signal_index - 1]))
        pd.testing.assert_frame_equal(base_window, changed_window)
        pd.testing.assert_series_equal(base_weights, changed_weights)
        pd.testing.assert_series_equal(base_signal, changed_signal)

    def test_trade_audit_symbol_contributions_sum_to_gross_return(self):
        dates = pd.bdate_range("2025-01-01", periods=90)
        symbols = pd.Index(["000001", "000002"])
        rising = 10.0 * np.power(1.02, np.arange(61))
        pullback = rising[-1] * np.power(0.99, np.arange(1, 30))
        close = pd.DataFrame(
            {"000001": np.concatenate((rising, pullback)), "000002": np.concatenate((rising * 1.01, pullback * 1.01))},
            index=dates,
        )
        open_px = close * 0.998
        high = close * 1.01
        low = close * 0.99
        amount = pd.DataFrame(50_000_000.0, index=dates, columns=symbols)
        market_exposure = pd.Series(1.0, index=dates)

        _, _, trades, _ = run_satellite_walk_forward(
            close, open_px, high, low, amount,
            train_days=65,
            retrain_frequency=5,
            top_n=1,
            rebalance_frequency=1,
            max_position_weight=0.6,
            leverage=0.6,
            min_score=None,
            commission_bps=0.0,
            impact_bps=0.0,
            max_buy_open_gap=0.05,
            limit_buffer=1.0,
            market_exposure=market_exposure,
            initial_capital=1_000_000.0,
            filter_kwargs={},
        )

        contributions = [json.loads(row.symbol_contributions_json) for row in trades.itertuples()]
        self.assertTrue(any(any(value != 0.0 for value in row.values()) for row in contributions))
        for row, contribution in zip(trades.itertuples(), contributions):
            self.assertAlmostEqual(sum(contribution.values()), row.gross_return)

    def test_selects_top_candidates_with_leverage_cap(self):
        candidates = pd.DataFrame(
            {
                "score": [0.20, 0.50, 0.10],
            },
            index=["000001", "000002", "000003"],
        )

        target = select_target_weights(
            candidates,
            all_symbols=pd.Index(["000001", "000002", "000003", "000004"]),
            top_n=2,
            leverage=0.60,
            max_position_weight=0.40,
        )

        self.assertAlmostEqual(target["000002"], 0.30)
        self.assertAlmostEqual(target["000001"], 0.30)
        self.assertAlmostEqual(target["000003"], 0.0)
        self.assertAlmostEqual(target.sum(), 0.60)

    def test_empty_candidates_return_zero_weights(self):
        target = select_target_weights(
            pd.DataFrame({"score": []}),
            all_symbols=pd.Index(["000001", "000002"]),
            top_n=5,
            leverage=0.80,
            max_position_weight=0.10,
        )

        self.assertEqual(target.to_dict(), {"000001": 0.0, "000002": 0.0})

    def test_position_cap_keeps_underfilled_target_explicit_when_candidates_are_few(self):
        candidates = pd.DataFrame(
            {"score": [0.5, 0.4]}, index=["000001", "000002"]
        )

        target = select_target_weights(
            candidates,
            all_symbols=pd.Index(["000001", "000002"]),
            top_n=8,
            leverage=0.90,
            max_position_weight=0.12,
        )

        self.assertAlmostEqual(float(target.sum()), 0.24)

    def test_min_score_filters_low_conviction_candidates(self):
        candidates = pd.DataFrame(
            {
                "score": [0.20, -0.01, 0.10],
            },
            index=["000001", "000002", "000003"],
        )

        target = select_target_weights(
            candidates,
            all_symbols=pd.Index(["000001", "000002", "000003"]),
            top_n=3,
            leverage=0.60,
            max_position_weight=0.40,
            min_score=0.0,
        )

        self.assertAlmostEqual(target["000001"], 0.30)
        self.assertAlmostEqual(target["000003"], 0.30)
        self.assertAlmostEqual(target["000002"], 0.0)

    def test_basket_guard_can_cut_exposure_for_broken_pullback_basket(self):
        candidates = pd.DataFrame(
            {
                "score": [0.20, 0.10, 0.05],
                "return_20d": [-0.10, -0.09, -0.11],
                "raw_distance_ma60": [-0.05, -0.04, -0.06],
            },
            index=["000001", "000002", "000003"],
        )

        target = select_target_weights(
            candidates,
            all_symbols=pd.Index(["000001", "000002", "000003"]),
            top_n=3,
            leverage=0.60,
            max_position_weight=0.40,
            basket_guard_return_20d_min=-0.08,
            basket_guard_distance_ma60_min=-0.03,
            basket_guard_scale=0.0,
        )

        self.assertAlmostEqual(target.sum(), 0.0)

    def test_rebound_exit_reduces_only_existing_positions_that_hit_profit_threshold(self):
        current = pd.Series({"000001": 0.08, "000002": 0.08, "000003": 0.0})
        target = pd.Series({"000001": 0.08, "000002": 0.08, "000003": 0.08})
        close_row = pd.Series({"000001": 10.7, "000002": 10.3, "000003": 12.0})
        entry_price = pd.Series({"000001": 10.0, "000002": 10.0, "000003": 11.0})

        adjusted, hits = apply_rebound_exit(
            current,
            target,
            close_row,
            entry_price,
            rebound_exit_return=0.05,
            rebound_exit_scale=0.0,
        )

        self.assertAlmostEqual(adjusted["000001"], 0.0)
        self.assertAlmostEqual(adjusted["000002"], 0.08)
        self.assertAlmostEqual(adjusted["000003"], 0.08)
        self.assertEqual(hits["000001"], "rebound_exit")
        self.assertNotIn("000002", hits)
        self.assertNotIn("000003", hits)

    def test_rebound_exit_market_exposure_gate_only_enables_in_weak_market(self):
        self.assertFalse(
            should_apply_rebound_exit(
                rebound_exit_return=0.095,
                market_exposure=1.0,
                market_exposure_max=0.99,
            )
        )
        self.assertTrue(
            should_apply_rebound_exit(
                rebound_exit_return=0.095,
                market_exposure=0.6,
                market_exposure_max=0.99,
            )
        )
        self.assertTrue(
            should_apply_rebound_exit(
                rebound_exit_return=0.095,
                market_exposure=1.0,
                market_exposure_max=None,
            )
        )

    def test_rebound_exit_market_exposure_floor_only_enables_in_strong_market(self):
        self.assertFalse(
            should_apply_rebound_exit(
                rebound_exit_return=0.095,
                market_exposure=0.6,
                market_exposure_min=0.99,
            )
        )
        self.assertTrue(
            should_apply_rebound_exit(
                rebound_exit_return=0.095,
                market_exposure=1.0,
                market_exposure_min=0.99,
            )
        )


if __name__ == "__main__":
    unittest.main()
