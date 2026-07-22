import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from world_event_shadow import (
    WorldEventConfig,
    build_observation,
    fetch_worldmonitor,
    load_payload_cache,
    load_config,
    write_payload_cache,
    write_outputs,
)


class WorldEventShadowTest(unittest.TestCase):
    def test_builds_research_only_observation_from_documented_contracts(self):
        payloads = {
            "macro_signals": {
                "timestamp": "2026-07-22T08:00:00Z",
                "verdict": "CASH",
                "signals": {"macroRegime": {"status": "DEFENSIVE"}},
                "unavailable": False,
            },
            "economic_stress": {
                "seededAt": "2026-07-22T08:30:00Z",
                "compositeScore": 72,
                "unavailable": False,
            },
            "fear_greed": {
                "seededAt": "2026-07-22T08:20:00Z",
                "compositeScore": 25,
                "unavailable": False,
            },
            "normalized": {
                "china_external_risk": 65,
                "energy_shock": 40,
                "shipping_disruption": 55,
                "trade_policy_pressure": 70,
            },
        }

        result = build_observation(
            payloads,
            asof_date="2026-07-22",
            now=datetime(2026, 7, 22, 9, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(result["status"], "research_ready")
        self.assertGreater(result["global_risk_score"], 70)
        self.assertEqual(result["event_source_count"], 3)
        self.assertEqual(result["data_freshness"], "current")
        self.assertTrue(result["research_only"])
        self.assertFalse(result["trade_instruction"])
        self.assertFalse(result["selection_effect"])
        self.assertEqual(result["portfolio_weight_effect"], 0.0)

    def test_missing_api_key_is_explicitly_degraded(self):
        config = WorldEventConfig(api_key_env="MODEL3_TEST_MISSING_KEY")
        with patch.dict("os.environ", {}, clear=True):
            payloads, errors = fetch_worldmonitor(config)
        result = build_observation(
            payloads,
            asof_date="2026-07-22",
            config=config,
            fetch_errors=errors,
        )

        self.assertEqual(result["status"], "degraded")
        self.assertIsNone(result["global_risk_score"])
        self.assertEqual(result["data_freshness"], "unknown")
        self.assertEqual(len(result["fetch_errors"]), 3)
        self.assertTrue(all(item["status"] == "missing_api_key" for item in errors))

    def test_unavailable_payload_is_not_treated_as_zero_risk(self):
        result = build_observation(
            {
                "macro_signals": {"unavailable": True},
                "economic_stress": {"unavailable": True, "compositeScore": 0},
                "fear_greed": {"unavailable": True, "compositeScore": 100},
            },
            asof_date="2026-07-22",
        )

        self.assertIsNone(result["global_risk_score"])
        self.assertEqual(result["risk_level"], "unknown")

    def test_outputs_include_json_csv_and_chinese_report(self):
        observation = build_observation({}, asof_date="2026-07-22")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "world_event_shadow_20260722.json"
            csv_path, report_path = write_outputs(output, observation)

            payload = json.loads(output.read_text(encoding="utf-8"))
            report = report_path.read_text(encoding="utf-8")

        self.assertFalse(payload["selection_effect"])
        self.assertTrue(csv_path.name.endswith(".csv"))
        self.assertIn("全球事件影子观察", report)
        self.assertIn("不改变 Model 3", report)

    def test_config_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("world_event_shadow:\n  surprise: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown world event settings"):
                load_config(path)

    def test_payload_cache_expires_explicitly(self):
        config = WorldEventConfig(max_cache_age_hours=72)
        cached_at = datetime(2026, 7, 20, 8, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            write_payload_cache(
                path,
                {"economic_stress": {"compositeScore": 60}},
                now=cached_at,
            )
            current, current_age = load_payload_cache(
                path,
                config,
                now=datetime(2026, 7, 22, 8, 0, tzinfo=timezone.utc),
            )
            expired, expired_age = load_payload_cache(
                path,
                config,
                now=datetime(2026, 7, 24, 9, 0, tzinfo=timezone.utc),
            )

        self.assertIn("economic_stress", current)
        self.assertEqual(current_age, 48.0)
        self.assertEqual(expired, {})
        self.assertGreater(expired_age, 72.0)


if __name__ == "__main__":
    unittest.main()
