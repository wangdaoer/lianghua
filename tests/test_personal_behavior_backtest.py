import importlib
import unittest

import pandas as pd


class PersonalBehaviorBacktestTest(unittest.TestCase):
    def _module(self):
        try:
            return importlib.import_module("run_personal_behavior_overlay_backtest")
        except ImportError as exc:
            self.fail(f"missing personal behavior backtest module: {exc}")

    def test_build_candidate_table_marks_original_top_names(self):
        mod = self._module()
        score = pd.Series(
            {"000001": 0.5, "000002": 0.3, "000003": 0.1, "000004": -0.2}
        )
        metrics = pd.DataFrame(
            {
                "return_20d": [0.01, -0.02, 0.03, 0.04],
                "close_position": [0.2, 0.8, 0.4, 0.5],
            },
            index=["000001", "000002", "000003", "000004"],
        )

        out = mod.build_candidate_table(
            score,
            metrics,
            top_n=2,
            candidate_pool_n=3,
            raw_weight=0.10,
        )

        self.assertEqual(out["symbol"].tolist(), ["000001", "000002", "000003"])
        self.assertEqual(out["selected"].tolist(), [True, True, False])
        self.assertEqual(out["target_weight"].tolist(), [0.10, 0.10, 0.0])

    def test_targets_from_overlay_table_applies_exposure_and_zero_fills(self):
        mod = self._module()
        overlay = pd.DataFrame(
            {
                "symbol": ["000001", "000003"],
                "personal_selected": [True, True],
                "personal_adjusted_target_weight": [0.10, 0.05],
            }
        )

        target = mod.targets_from_overlay_table(
            overlay,
            all_symbols=pd.Index(["000001", "000002", "000003"]),
            exposure=0.5,
        )

        self.assertAlmostEqual(target["000001"], 0.05)
        self.assertAlmostEqual(target["000002"], 0.0)
        self.assertAlmostEqual(target["000003"], 0.025)

    def test_weights_for_signal_date_uses_last_known_row(self):
        mod = self._module()
        weights = pd.DataFrame(
            [
                {"date": "2023-01-10", "momentum_20": 0.1, "reversal_5": -0.1},
                {"date": "2023-01-20", "momentum_20": 0.2, "reversal_5": -0.2},
            ]
        )

        selected = mod.weights_for_signal_date(weights, pd.Timestamp("2023-01-15"))

        self.assertAlmostEqual(selected["momentum_20"], 0.1)
        self.assertAlmostEqual(selected["reversal_5"], -0.1)


if __name__ == "__main__":
    unittest.main()
