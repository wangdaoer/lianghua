import copy
import unittest

from strong_pullback_evolution import (
    SearchCandidate,
    SearchGroup,
    build_group_candidates,
    parse_evolution_config,
)


def valid_raw_config() -> dict[str, object]:
    return {
        "strategy": "strong_pullback_satellite",
        "periods": {
            "research_start": "2022-01-01",
            "train_end": "2024-12-31",
            "validation_start": "2025-01-01",
            "validation_end": "2025-12-31",
            "test_start": "2026-01-01",
            "test_end": None,
        },
        "baseline": {"top_n": 8, "leverage": 0.60, "max_position_weight": 0.08},
        "search_groups": [
            {
                "id": "risk_budget",
                "hypothesis_cn": "扩大风险预算",
                "candidates": [
                    {"id": "risk_075", "overrides": {"leverage": 0.75}},
                    {"id": "risk_090", "overrides": {"leverage": 0.90}},
                ],
            }
        ],
        "selection": {
            "min_validation_days": 120,
            "min_test_days": 60,
            "max_drawdown_floor": -0.40,
            "min_annualized_return_delta": 0.01,
            "min_sharpe_delta": -0.10,
            "max_turnover_ratio": 1.50,
            "rolling_window_days": 126,
            "max_negative_window_rate": 0.60,
        },
    }


class EvolutionConfigTest(unittest.TestCase):
    def test_rejects_overlapping_validation_and_test_periods(self):
        raw = valid_raw_config()
        raw["periods"]["test_start"] = "2025-12-31"

        with self.assertRaisesRegex(ValueError, "train_end < validation_start <= validation_end < test_start"):
            parse_evolution_config(raw)

    def test_rejects_unknown_strategy_override(self):
        raw = valid_raw_config()
        raw["search_groups"][0]["candidates"][0]["overrides"] = {"future_leak": 1}

        with self.assertRaisesRegex(ValueError, "Unknown strategy parameters"):
            parse_evolution_config(raw)

    def test_rejects_duplicate_candidate_ids_across_groups(self):
        raw = valid_raw_config()
        raw["search_groups"].append(copy.deepcopy(raw["search_groups"][0]))
        raw["search_groups"][1]["id"] = "entry_depth"

        with self.assertRaisesRegex(ValueError, "Duplicate candidate id"):
            parse_evolution_config(raw)

    def test_group_candidates_share_incumbent_but_not_each_other(self):
        config = parse_evolution_config(valid_raw_config())
        generated = build_group_candidates(config.baseline, config.search_groups[0])

        self.assertEqual(generated[0][1]["leverage"], 0.75)
        self.assertEqual(generated[1][1]["leverage"], 0.90)
        self.assertEqual(config.baseline["leverage"], 0.60)
        self.assertIsNot(generated[0][1], generated[1][1])

    def test_rejects_null_empty_and_non_string_group_ids(self):
        for invalid_id in (None, "", 123):
            with self.subTest(invalid_id=invalid_id):
                raw = valid_raw_config()
                raw["search_groups"][0]["id"] = invalid_id

                with self.assertRaisesRegex(ValueError, "Group id must be a non-empty string"):
                    parse_evolution_config(raw)

    def test_rejects_null_empty_and_non_string_candidate_ids(self):
        for invalid_id in (None, "", 123):
            with self.subTest(invalid_id=invalid_id):
                raw = valid_raw_config()
                raw["search_groups"][0]["candidates"][0]["id"] = invalid_id

                with self.assertRaisesRegex(ValueError, "Candidate id must be a non-empty string"):
                    parse_evolution_config(raw)

    def test_group_candidates_are_deeply_independent(self):
        incumbent = {"nested": {"values": ["incumbent"]}}
        group = SearchGroup(
            "nested",
            "nested values",
            (
                SearchCandidate("first", {"nested": {"values": ["first"]}}),
                SearchCandidate("second", {"nested": {"values": ["second"]}}),
            ),
        )

        generated = build_group_candidates(incumbent, group)
        generated[0][1]["nested"]["values"].append("changed")

        self.assertEqual(generated[1][1]["nested"]["values"], ["second"])
        self.assertEqual(incumbent["nested"]["values"], ["incumbent"])

    def test_rejects_malformed_search_group_entries(self):
        raw = valid_raw_config()
        raw["search_groups"] = [None]

        with self.assertRaisesRegex(ValueError, "search_groups entries must be mappings"):
            parse_evolution_config(raw)

    def test_rejects_malformed_search_candidate_entries(self):
        raw = valid_raw_config()
        raw["search_groups"][0]["candidates"] = [None]

        with self.assertRaisesRegex(ValueError, "candidates entries must be mappings"):
            parse_evolution_config(raw)

    def test_rejects_unknown_selection_keys(self):
        raw = valid_raw_config()
        raw["selection"]["future_rule"] = True

        with self.assertRaisesRegex(ValueError, "Unknown selection keys"):
            parse_evolution_config(raw)

    def test_rejects_missing_selection_keys(self):
        raw = valid_raw_config()
        del raw["selection"]["min_test_days"]

        with self.assertRaisesRegex(ValueError, "Missing selection keys"):
            parse_evolution_config(raw)
