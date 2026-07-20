import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline_cache import PipelineStepCache, build_global_fingerprint
from run_daily_model_pipeline import PipelineStep, run_steps


class PipelineCacheTest(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        (root / "configs").mkdir()
        daily_dir = root / "daily"
        daily_dir.mkdir()
        paths = {
            "code": root / "worker.py",
            "config": root / "configs" / "model.yaml",
            "base": root / "base.csv",
            "benchmark": root / "benchmark.csv",
            "daily": daily_dir / "ths_hs_a_share_2026-07-20.csv",
            "daily_dir": daily_dir,
        }
        paths["code"].write_text("VALUE = 1\n", encoding="utf-8")
        paths["config"].write_text("top_n: 40\n", encoding="utf-8")
        paths["base"].write_text("date,symbol\n", encoding="utf-8")
        paths["benchmark"].write_text("date,close\n", encoding="utf-8")
        paths["daily"].write_text("date,symbol\n", encoding="utf-8")
        return paths

    def _fingerprint(self, root: Path, paths: dict[str, Path]):
        return build_global_fingerprint(
            project_root=root,
            base_panel=paths["base"],
            daily_dir=paths["daily_dir"],
            daily_start="2026-07-20",
            asof_date="2026-07-20",
            benchmark=paths["benchmark"],
        )

    def test_cache_requires_matching_command_inputs_and_output_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._fixture(root)
            output = root / "panel.parquet"
            output.write_bytes(b"panel-v1")
            command = ("python", "panel.py", "--output", str(output))
            fingerprint, inputs = self._fingerprint(root, paths)
            manifest = root / "cache.json"

            cache = PipelineStepCache(manifest, fingerprint, inputs)
            cache.record_success("update_panel", command)
            cache.save()
            restored = PipelineStepCache(manifest, fingerprint, inputs)

            self.assertTrue(restored.restore("update_panel", command, []))
            self.assertFalse(
                restored.restore(
                    "update_panel",
                    (*command[:-1], str(root / "different.parquet")),
                    [],
                )
            )
            output.write_bytes(b"panel-v2")
            self.assertFalse(restored.restore("update_panel", command, []))

            paths["code"].write_text("VALUE = 2\n", encoding="utf-8")
            changed_fingerprint, changed_inputs = self._fingerprint(root, paths)
            self.assertNotEqual(changed_fingerprint, fingerprint)
            invalidated = PipelineStepCache(manifest, changed_fingerprint, changed_inputs)
            self.assertFalse(invalidated.restore("update_panel", command, []))

    def test_directory_outputs_are_content_addressed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "trend"
            output_dir.mkdir()
            (output_dir / "state.csv").write_text("state\n", encoding="utf-8")
            command = ("python", "trend.py", "--output-dir", str(output_dir))
            cache = PipelineStepCache(root / "cache.json", "global", {})
            cache.record_success("trend_state", command)

            self.assertTrue(cache.restore("trend_state", command, [True]))
            (output_dir / "state.csv").write_text("changed\n", encoding="utf-8")
            self.assertFalse(cache.restore("trend_state", command, [True]))
            self.assertFalse(cache.restore("trend_state", command, [False]))

    def test_run_steps_restores_dependency_chain_without_subprocesses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = root / "panel.parquet"
            trend_dir = root / "trend"
            panel.write_bytes(b"panel")
            trend_dir.mkdir()
            (trend_dir / "state.csv").write_text("state\n", encoding="utf-8")
            steps = [
                PipelineStep(
                    "update_panel",
                    ("python", "panel.py", "--output", str(panel)),
                ),
                PipelineStep(
                    "trend_state",
                    ("python", "trend.py", "--output-dir", str(trend_dir)),
                ),
            ]
            cache = PipelineStepCache(root / "cache.json", "global", {})
            for step in steps:
                cache.record_success(step.name, step.command)

            with patch("run_daily_model_pipeline.subprocess.run") as subprocess_run:
                executions = run_steps(steps, root, max_parallel_steps=2, step_cache=cache)

            subprocess_run.assert_not_called()
            self.assertEqual([item["status"] for item in executions], ["cached", "cached"])
            self.assertTrue(all(item["cache_hit"] for item in executions))


if __name__ == "__main__":
    unittest.main()
