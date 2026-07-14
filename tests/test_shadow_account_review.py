import csv
import json
import tempfile
import unittest
from pathlib import Path

from build_shadow_account_review import derive_shadow_account_review, write_shadow_account_review


class ShadowAccountReviewTest(unittest.TestCase):
    def test_derive_shadow_account_review_extracts_rules_without_trading_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            review_dir = Path(tmp)
            (review_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "period_start": "2023-01-03",
                        "period_end": "2026-06-08",
                        "matched_round_trips": 30,
                        "realized_pnl": -1000.0,
                        "win_rate": 0.4,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self._write_csv(
                review_dir / "by_holding_bucket.csv",
                [
                    {"holding_bucket": "0d", "trades": "12", "pnl": "-5000", "win_rate": "0.25", "avg_ret": "-0.03"},
                    {"holding_bucket": "2-3d", "trades": "10", "pnl": "3500", "win_rate": "0.70", "avg_ret": "0.04"},
                ],
            )
            self._write_csv(
                review_dir / "by_entry_position_bucket.csv",
                [
                    {
                        "entry_position_bucket": "low_quarter",
                        "trades": "9",
                        "pnl": "2200",
                        "win_rate": "0.67",
                        "avg_ret": "0.035",
                    },
                    {
                        "entry_position_bucket": "high_quarter",
                        "trades": "9",
                        "pnl": "-1800",
                        "win_rate": "0.22",
                        "avg_ret": "-0.025",
                    },
                ],
            )
            self._write_csv(
                review_dir / "by_symbol.csv",
                [
                    {"symbol": "000001", "name": "Alpha", "trades": "5", "pnl": "1200", "win_rate": "0.80", "avg_ret": "0.05"},
                    {"symbol": "000002", "name": "Beta", "trades": "6", "pnl": "-2400", "win_rate": "0.17", "avg_ret": "-0.06"},
                ],
            )
            self._write_csv(
                review_dir / "round_trips.csv",
                [
                    {
                        "symbol": "000001",
                        "name": "Alpha",
                        "pnl": "1000",
                        "return_pct": "0.05",
                        "holding_days": "2",
                        "mfe_pct": "0.08",
                        "giveback_from_mfe_pct": "0.02",
                    },
                    {
                        "symbol": "000002",
                        "name": "Beta",
                        "pnl": "-900",
                        "return_pct": "-0.06",
                        "holding_days": "0",
                        "mfe_pct": "0.02",
                        "giveback_from_mfe_pct": "0.08",
                    },
                ],
            )

            review = derive_shadow_account_review(review_dir, min_trades=5)

            self.assertTrue(review["research_only"])
            self.assertFalse(review["allows_broker_orders"])
            self.assertGreaterEqual(len(review["rules"]), 3)
            rule_text = "\n".join(rule["description"] for rule in review["rules"])
            self.assertIn("2-3d", rule_text)
            self.assertIn("low_quarter", rule_text)
            self.assertIn("000002", rule_text)
            self.assertIn("counterfactual", review)

    def test_write_shadow_account_review_outputs_json_csv_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            review_dir = Path(tmp)
            (review_dir / "summary.json").write_text("{}", encoding="utf-8")
            self._write_csv(review_dir / "by_holding_bucket.csv", [])
            self._write_csv(review_dir / "by_symbol.csv", [])

            paths = write_shadow_account_review(review_dir, generated_at="2026-06-29T16:00:00")

            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["rules_csv"].exists())
            self.assertTrue(paths["markdown"].exists())
            self.assertIn("Shadow Account Review", paths["markdown"].read_text(encoding="utf-8"))
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertTrue(payload["research_only"])

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = sorted({column for row in rows for column in row})
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
