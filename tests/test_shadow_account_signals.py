import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from shadow_account_signals import (
    apply_shadow_account_signals,
    load_shadow_account_review,
    shadow_account_summary_lines,
)


class ShadowAccountSignalsTest(unittest.TestCase):
    def test_apply_shadow_account_signals_tags_symbol_position_and_momentum_rules(self):
        review = {
            "research_only": True,
            "allows_broker_orders": False,
            "rules": [
                {
                    "action": "avoid_or_watch",
                    "source": "symbol_history",
                    "value": "000002",
                    "description": "000002 has severe personal loss evidence",
                },
                {
                    "action": "avoid_or_reduce",
                    "source": "entry_position",
                    "value": "high_quarter",
                    "description": "Avoid high_quarter entries",
                },
                {
                    "action": "prefer",
                    "source": "entry_position",
                    "value": "mid_low",
                    "description": "Prefer mid_low entries",
                },
                {
                    "action": "avoid_or_reduce",
                    "source": "entry_momentum",
                    "value": ">+20%",
                    "description": "Avoid chasing after >20% 20d run",
                },
            ],
        }
        table = pd.DataFrame(
            [
                {"symbol": "000001", "close_position": 0.35, "return_20d": 0.08},
                {"symbol": "000002", "close_position": 0.82, "return_20d": 0.11},
                {"symbol": "000003", "close_position": 0.52, "return_20d": 0.26},
                {"symbol": "000004", "close_position": 0.62, "return_20d": 0.03},
            ]
        )

        enriched = apply_shadow_account_signals(table, review)

        by_symbol = enriched.set_index("symbol")
        self.assertEqual(by_symbol.loc["000001", "shadow_account_signal"], "prefer")
        self.assertIn("mid_low", by_symbol.loc["000001", "shadow_account_notes"])
        self.assertEqual(by_symbol.loc["000002", "shadow_account_signal"], "risk")
        self.assertIn("000002", by_symbol.loc["000002", "shadow_account_notes"])
        self.assertIn("high_quarter", by_symbol.loc["000002", "shadow_account_notes"])
        self.assertEqual(by_symbol.loc["000003", "shadow_account_signal"], "risk")
        self.assertIn(">+20%", by_symbol.loc["000003", "shadow_account_notes"])
        self.assertEqual(by_symbol.loc["000004", "shadow_account_signal"], "neutral")
        pd.testing.assert_frame_equal(
            enriched[table.columns].reset_index(drop=True),
            table.reset_index(drop=True),
        )

    def test_load_shadow_account_review_rejects_unmarked_or_broker_enabled_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shadow_account_review.json"
            for payload in (
                {"allows_broker_orders": False, "rules": []},
                {"research_only": True, "allows_broker_orders": True, "rules": []},
                ["not", "an", "object"],
                {
                    "research_only": True,
                    "allows_broker_orders": False,
                    "rules": ["not-an-object"],
                },
            ):
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_shadow_account_review(path)

    def test_load_shadow_account_review_and_summary_are_research_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shadow_account_review.json"
            path.write_text(
                json.dumps(
                    {
                        "research_only": True,
                        "allows_broker_orders": False,
                        "rules": [
                            {"action": "avoid_or_reduce", "source": "buy_time", "value": "09:30-10:00"},
                            {"action": "prefer", "source": "holding", "value": "4-10d"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            review = load_shadow_account_review(path)
            lines = shadow_account_summary_lines(review)

        self.assertTrue(review["research_only"])
        self.assertIn("不连接券商", "\n".join(lines))
        self.assertIn("09:30-10:00", "\n".join(lines))
        self.assertIn("4-10d", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
