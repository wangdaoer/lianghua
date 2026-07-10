import importlib
import unittest

import pandas as pd


class PersonalBehaviorSweepTest(unittest.TestCase):
    def _module(self):
        try:
            return importlib.import_module("sweep_personal_behavior_overlay")
        except ImportError as exc:
            self.fail(f"missing personal behavior sweep module: {exc}")

    def test_build_rule_variants_crosses_weak_multiplier_and_damage_threshold(self):
        mod = self._module()
        base = {
            "weak_20d_weight_multiplier": 0.70,
            "damaged_20d_return_max": -0.20,
            "selection_mode": "conservative_fill",
        }

        variants = mod.build_rule_variants(
            base,
            weak_multipliers=[0.60, 0.80],
            damaged_thresholds=[-0.15, -0.25],
            selection_modes=["conservative_fill", "full_rerank"],
        )

        self.assertEqual(len(variants), 8)
        self.assertEqual(variants[0]["variant"], "conservative_fill_weak060_damage-15")
        self.assertEqual(variants[-1]["rules"]["weak_20d_weight_multiplier"], 0.80)
        self.assertEqual(variants[-1]["rules"]["damaged_20d_return_max"], -0.25)
        self.assertEqual(variants[-1]["rules"]["selection_mode"], "full_rerank")

    def test_rank_sweep_results_prefers_return_inside_drawdown_floor(self):
        mod = self._module()
        table = pd.DataFrame(
            [
                {"variant": "too_risky", "total_return": 0.30, "max_drawdown": -0.35, "sharpe_like": 0.5},
                {"variant": "balanced", "total_return": 0.22, "max_drawdown": -0.28, "sharpe_like": 0.4},
                {"variant": "safer", "total_return": 0.18, "max_drawdown": -0.20, "sharpe_like": 0.6},
            ]
        )

        ranked = mod.rank_sweep_results(table, max_drawdown_floor=-0.30)

        self.assertEqual(ranked.iloc[0]["variant"], "balanced")
        self.assertFalse(bool(ranked.loc[ranked["variant"].eq("too_risky"), "passes_drawdown"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
