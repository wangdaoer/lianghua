import json
import tempfile
import unittest
from pathlib import Path

from export_marketlens_model3_feed import build_feed, export_feed


class MarketLensModel3ExportTest(unittest.TestCase):
    def test_build_feed_shapes_priority_rows_for_public_website(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs" / "high_return_v2"
            token = "20260708"
            self._write_csv(
                output / f"merged_priority_watchlist_{token}.csv",
                [
                    {
                        "symbol": "000001",
                        "stock_name": "Alpha",
                        "priority_bucket": "model_focus",
                        "priority_score": "212.325",
                        "personal_rank": "1",
                        "personal_selected": "True",
                        "personal_target_weight": "0.02325",
                        "personal_action_cn": "可保留",
                        "personal_reasons_cn": "低位介入习惯",
                        "trend_state": "生命线健康",
                        "return_5d": "-0.1",
                        "return_20d": "0.25",
                        "return_60d": "",
                        "close_position": "0.3",
                    },
                    {
                        "symbol": "000002",
                        "stock_name": "Beta",
                        "priority_bucket": "risk_watch",
                        "priority_score": "150",
                        "risk_flags": "st_or_special_treatment",
                    },
                ],
            )
            self._write_csv(output / f"early_pattern_watchlist_{token}.csv", [{"symbol": "000001"}])
            self._write_csv(output / f"merged_model_decision_table_{token}.csv", [{"symbol": "000001"}])
            state_log = root / "daily_run_state.jsonl"
            state_log.write_text(
                json.dumps(
                    {
                        "asof_date": "2026-07-08",
                        "run_type": "skip_train",
                        "verification": {"tests": "76 passed", "priority_rows": 2},
                        "artifacts": {"priority": "D:\\secret\\merged_priority_watchlist_20260708.csv"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            feed = build_feed("2026-07-08", output, state_log)
            rendered = json.dumps(feed, ensure_ascii=False)

            self.assertEqual(feed["schemaVersion"], 1)
            self.assertEqual(feed["asofDate"], "2026-07-08")
            self.assertEqual(feed["runType"], "skip_train")
            self.assertEqual(feed["verification"]["priorityRows"], 2)
            self.assertEqual(feed["bucketCounts"], {"model_focus": 1, "risk_watch": 1})
            self.assertEqual(feed["top10"][0]["code"], "000001")
            self.assertEqual(feed["top10"][0]["targetWeightPct"], 2.325)
            self.assertEqual(feed["top10"][0]["return5dPct"], -10.0)
            self.assertNotIn("D:\\", rendered)

    def test_export_feed_writes_model_output_and_dashboard_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs" / "high_return_v2"
            token = "20260708"
            self._write_csv(
                output / f"merged_priority_watchlist_{token}.csv",
                [{"symbol": "000001", "stock_name": "Alpha", "priority_bucket": "model_focus"}],
            )
            self._write_csv(output / f"early_pattern_watchlist_{token}.csv", [])
            self._write_csv(output / f"merged_model_decision_table_{token}.csv", [])
            model_output = output / "marketlens_model3_latest.json"
            dashboard_output = root / "dashboard" / "data" / "quant-model3-latest.json"

            feed = export_feed("2026-07-08", output, root / "missing.jsonl", model_output, dashboard_output)

            self.assertEqual(json.loads(model_output.read_text(encoding="utf-8")), feed)
            self.assertEqual(json.loads(dashboard_output.read_text(encoding="utf-8")), feed)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        import csv

        path.parent.mkdir(parents=True, exist_ok=True)
        columns = sorted({column for row in rows for column in row}) or ["symbol"]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
