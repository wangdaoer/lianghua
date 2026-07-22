import json
import tempfile
import unittest
from pathlib import Path

from run_daily_model_pipeline import (
    PipelineConfig,
    build_daily_pipeline_steps,
    build_daily_run_state,
    pipeline_step_dependencies,
)


class WorldEventPipelineIntegrationTest(unittest.TestCase):
    def _config(self, root: Path, **overrides) -> PipelineConfig:
        values = {
            "asof_date": "2026-07-22",
            "python_exe": "python",
            "project_root": root,
            "output_root": Path("outputs/high_return_v2"),
            "include_benchmark_refresh": False,
            "include_regime_shadow_compare": False,
            "include_regime_shadow_tracking": False,
            "include_dynamic_breadth_tracking": False,
            "include_strategy_arena": False,
            "include_research_db_sync": False,
            "include_marketlens_export": False,
            "train_model": False,
        }
        values.update(overrides)
        return PipelineConfig(**values)

    def test_daily_pipeline_runs_world_event_as_independent_shadow_step(self):
        config = self._config(Path("C:/model"))

        steps = build_daily_pipeline_steps(config)
        names = [step.name for step in steps]
        step = steps[names.index("world_event_shadow")]
        command = " ".join(str(part) for part in step.command).replace("\\", "/")

        self.assertIn("world_event_shadow.py", command)
        self.assertIn("configs/world_event_shadow.yaml", command)
        self.assertIn("world_event_shadow_20260722.json", command)
        self.assertIn("world_event_shadow_payload_cache.json", command)
        self.assertEqual(pipeline_step_dependencies(steps)["world_event_shadow"], ())

    def test_daily_pipeline_can_use_reviewed_local_snapshot(self):
        config = self._config(
            Path("C:/model"),
            world_event_snapshot=Path("data/world_event_snapshot.json"),
        )

        steps = build_daily_pipeline_steps(config)
        step = next(item for item in steps if item.name == "world_event_shadow")
        command = " ".join(str(part) for part in step.command).replace("\\", "/")

        self.assertIn("--snapshot", command)
        self.assertIn("data/world_event_snapshot.json", command)

    def test_daily_run_state_exposes_current_shadow_without_strategy_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs" / "high_return_v2"
            output.mkdir(parents=True)
            metadata = {
                "asof_date": "2026-07-22",
                "status": "partial",
                "global_risk_score": 58.5,
                "risk_level": "elevated",
                "event_source_count": 2,
                "event_confidence": 0.4,
                "data_freshness": "stale",
                "selection_effect": False,
            }
            base = output / "world_event_shadow_20260722"
            base.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
            base.with_suffix(".csv").write_text("status\npartial\n", encoding="utf-8")
            base.with_suffix(".md").write_text("report", encoding="utf-8")
            config = self._config(root)
            steps = build_daily_pipeline_steps(config)

            record = build_daily_run_state(config, steps)

        verification = record["verification"]
        self.assertEqual(verification["world_event_shadow_status"], "partial")
        self.assertEqual(verification["world_event_global_risk_score"], 58.5)
        self.assertFalse(verification["world_event_selection_effect"])
        self.assertTrue(record["artifacts"]["world_event_shadow_report"].endswith(".md"))

    def test_daily_pipeline_can_skip_world_event_shadow(self):
        config = self._config(Path("C:/model"), include_world_event_shadow=False)

        names = [step.name for step in build_daily_pipeline_steps(config)]

        self.assertNotIn("world_event_shadow", names)


if __name__ == "__main__":
    unittest.main()
