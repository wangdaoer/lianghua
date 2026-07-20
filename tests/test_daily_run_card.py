import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from daily_run_card import build_daily_run_card, write_daily_run_card


class DailyRunCardTest(unittest.TestCase):
    def test_build_daily_run_card_hashes_artifacts_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "priority.csv"
            artifact.write_text("symbol,name\n000001,Alpha\n", encoding="utf-8")
            missing = root / "missing.csv"
            record = {
                "asof_date": "2026-06-29",
                "project_root": str(root),
                "run_type": "full_train",
                "data_source": str(artifact),
                "commands": ["python run_daily_model_pipeline.py --asof-date 2026-06-29"],
                "verification": {"priority_rows": 1, "tests": "passed"},
                "artifacts": {"priority": str(artifact), "missing": str(missing)},
            }

            card = build_daily_run_card(record, generated_at="2026-06-29T16:00:00")

            self.assertEqual(card["schema_version"], 2)
            self.assertEqual(card["asof_date"], "2026-06-29")
            self.assertEqual(card["run_type"], "full_train")
            self.assertIn("config_hash", card)
            self.assertEqual(card["command_count"], 1)
            by_key = {item["key"]: item for item in card["artifacts"]}
            self.assertTrue(by_key["priority"]["exists"])
            self.assertEqual(by_key["priority"]["size_bytes"], artifact.stat().st_size)
            self.assertEqual(by_key["priority"]["sha256"], hashlib.sha256(artifact.read_bytes()).hexdigest())
            self.assertFalse(by_key["missing"]["exists"])
            self.assertIsNone(by_key["missing"]["sha256"])

    def test_write_daily_run_card_outputs_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs"
            artifact = root / "selected.csv"
            self._write_csv(artifact, [{"symbol": "000001"}])
            record = {
                "asof_date": "2026-06-29",
                "run_type": "skip_train",
                "data_source": str(artifact),
                "commands": [],
                "verification": {"selected_rows": 1},
                "artifacts": {"selected": str(artifact)},
            }

            paths = write_daily_run_card(output, record, generated_at="2026-06-29T16:00:00")

            self.assertEqual(paths["json"], output / "daily_run_card_20260629.json")
            self.assertEqual(paths["markdown"], output / "daily_run_card_20260629.md")
            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["markdown"].exists())
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["verification"]["selected_rows"], 1)
            self.assertIn("# Daily Run Card 2026-06-29", paths["markdown"].read_text(encoding="utf-8"))

    def test_run_card_preserves_step_execution_audit(self):
        execution = {
            "max_parallel_steps": 2,
            "wall_duration_seconds": 8.0,
            "summed_step_duration_seconds": 12.5,
            "cache_hits": 0,
            "status_counts": {"success": 2},
            "steps": [
                {
                    "name": "update_panel",
                    "status": "success",
                    "duration_seconds": 8.0,
                    "cache_hit": False,
                }
            ],
        }

        card = build_daily_run_card(
            {"asof_date": "2026-06-29", "execution": execution, "artifacts": {}},
            generated_at="2026-06-29T16:00:00",
        )

        self.assertEqual(card["execution"], execution)

    def test_build_daily_run_card_ignores_unavailable_optional_artifact_paths(self):
        card = build_daily_run_card(
            {
                "asof_date": "2026-06-29",
                "artifacts": {"optional": None, "also_optional": ""},
            },
            generated_at="2026-06-29T16:00:00",
        )

        self.assertEqual(card["artifacts"], [])
        self.assertNotIn("artifact_missing:optional", card["warnings"])

    def test_failed_run_card_exposes_failure_and_stale_artifact_warning(self):
        card = build_daily_run_card(
            {
                "asof_date": "2026-06-29",
                "run_status": "failed",
                "failure": {"step": "merged_daily_outputs", "returncode": 1},
                "artifacts": {},
            },
            generated_at="2026-06-29T16:00:00",
        )

        self.assertEqual(card["run_status"], "failed")
        self.assertEqual(card["failure"]["step"], "merged_daily_outputs")
        self.assertIn("run_failed", card["warnings"])
        self.assertIn(
            "artifacts_may_include_same_day_pre_failure_outputs", card["warnings"]
        )

    def test_run_card_hash_ignores_its_own_artifact_paths(self):
        record = {
            "asof_date": "2026-06-29",
            "run_status": "success",
            "artifacts": {"selected": "selected.csv"},
        }
        first = build_daily_run_card(record, generated_at="2026-06-29T16:00:00")
        record["artifacts"]["daily_run_card_json"] = "daily_run_card_20260629.json"
        record["artifacts"]["daily_run_card_markdown"] = "daily_run_card_20260629.md"
        second = build_daily_run_card(record, generated_at="2026-06-29T16:00:00")

        self.assertEqual(first["record_hash"], second["record_hash"])
        self.assertEqual(first["artifacts"], second["artifacts"])

    def test_detailed_passing_test_status_does_not_emit_warning(self):
        card = build_daily_run_card(
            {
                "asof_date": "2026-07-15",
                "verification": {"tests": "560 passed; 563 subtests passed"},
                "artifacts": {},
            }
        )

        self.assertEqual(card["warnings"], [])

    def test_detailed_failed_test_status_emits_warning(self):
        card = build_daily_run_card(
            {
                "asof_date": "2026-07-15",
                "verification": {"tests": "560 passed, 1 failed"},
                "artifacts": {},
            }
        )

        self.assertIn("tests:560 passed, 1 failed", card["warnings"])

    def test_degraded_benchmark_refresh_emits_warning(self):
        card = build_daily_run_card(
            {
                "asof_date": "2026-07-16",
                "verification": {
                    "tests": "passed",
                    "benchmark_refresh_status": "updated_degraded",
                },
                "artifacts": {},
            }
        )

        self.assertIn(
            "benchmark_refresh:updated_degraded",
            card["warnings"],
        )

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = sorted({column for row in rows for column in row})
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
