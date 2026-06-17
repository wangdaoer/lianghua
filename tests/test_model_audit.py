from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser
from quant_etf_lab.dashboard import _summarize_model_audit
from quant_etf_lab.model_audit import run_model_build_audit


class ModelAuditTests(unittest.TestCase):
    def test_model_audit_cli_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["model-audit"])
        self.assertEqual(args.config_dir, "configs")
        self.assertEqual(args.walk_forward_dir, "outputs/walk_forward")
        self.assertEqual(args.walk_forward_resolution_path, "outputs/research/walk_forward_run_resolutions.csv")
        self.assertEqual(args.output_dir, "outputs/research/model_build_audit_latest")

    def test_model_build_audit_resolutions_suppress_resolved_walk_forward_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            walk_dir = root / "walk_forward"
            lock_dir = root / "locks"
            output_dir = root / "audit"
            config_dir.mkdir()
            walk_dir.mkdir()
            lock_dir.mkdir()
            (config_dir / "example.yaml").write_text(
                """
project:
  name: example
strategy:
  name: demo
""",
                encoding="utf-8",
            )
            partial = walk_dir / "partial_run"
            partial.mkdir()
            (partial / "wf_01_selected_params.json").write_text("{}", encoding="utf-8")
            missing = walk_dir / "missing_run"
            missing.mkdir()
            resolution_path = root / "walk_forward_run_resolutions.csv"
            pd.DataFrame(
                {
                    "run_name": ["partial_run"],
                    "resolution_status": ["superseded"],
                    "replacement_run": ["complete_run"],
                    "resolved_at": ["2024-04-02"],
                    "resolved_by": ["test"],
                    "resolution_note": ["covered by later complete run"],
                }
            ).to_csv(resolution_path, index=False)

            result = run_model_build_audit(
                project_root=root,
                config_dir=config_dir,
                walk_forward_dir=walk_dir,
                lock_dir=lock_dir,
                output_dir=output_dir,
                walk_forward_resolution_path=resolution_path,
            )

            runs = pd.read_csv(result.walk_forward_runs_path)
            actions = pd.read_csv(result.walk_forward_actions_path)
            partial_row = runs.loc[runs["run_name"] == "partial_run"].iloc[0]
            self.assertEqual(partial_row["resolution_status"], "superseded")
            self.assertEqual(partial_row["replacement_run"], "complete_run")
            self.assertEqual(result.snapshot["resolved_walk_forward_runs"], 1)
            self.assertEqual(result.snapshot["incomplete_walk_forward_runs"], 2)
            self.assertEqual(result.snapshot["unresolved_incomplete_walk_forward_runs"], 1)
            self.assertEqual(result.snapshot["walk_forward_action_items"], 1)
            self.assertNotIn("partial_run", set(actions["run_name"]))
            self.assertIn("missing_run", set(actions["run_name"]))

    def test_model_audit_generated_after_as_of_is_ok_when_actions_are_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_audit = root / "model_audit"
            model_audit.mkdir()
            (model_audit / "model_build_audit_snapshot.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-06-14T18:00:00",
                        "duplicate_config_groups": 0,
                        "root_configs_without_extends": 0,
                        "walk_forward_action_items": 0,
                        "walk_forward_resume_candidates": 0,
                        "walk_forward_archive_review_candidates": 0,
                    }
                ),
                encoding="utf-8",
            )
            (model_audit / "model_build_audit.md").write_text("# Model Build Audit\n", encoding="utf-8")
            pd.DataFrame(columns=["run_id", "recommended_action"]).to_csv(
                model_audit / "walk_forward_run_actions.csv",
                index=False,
            )

            snapshot = _summarize_model_audit(root, model_audit, date(2026, 6, 12), max_staleness_days=3)

            self.assertEqual(snapshot["model_audit_status"], "ok")
            self.assertEqual(snapshot["model_audit_days_since_generated"], -2)

    def test_model_build_audit_detects_duplicate_configs_and_partial_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "configs"
            wf_dir = root / "outputs" / "walk_forward"
            base_dir = config_dir / "base"
            config_dir.mkdir(parents=True)
            base_dir.mkdir(parents=True)
            wf_dir.mkdir(parents=True)
            config_body = """
project:
  name: sample
strategy:
  name: multi_factor
  factor_weights:
    reversal: 0.5
risk:
  enabled: true
costs:
  commission_rate: 0.0003
"""
            (config_dir / "a.yaml").write_text(config_body, encoding="utf-8")
            (config_dir / "b.yaml").write_text(config_body.replace("name: sample", "name: sample_b"), encoding="utf-8")
            (base_dir / "base.yaml").write_text(
                """
project:
  name: base
strategy:
  name: trend
""",
                encoding="utf-8",
            )
            (config_dir / "c.yaml").write_text(
                """
extends: base/base.yaml
project:
  name: child
""",
                encoding="utf-8",
            )
            partial = wf_dir / "wf_partial"
            partial.mkdir()
            (partial / "wf_01_selected_params.json").write_text("{}", encoding="utf-8")
            missing = wf_dir / "wf_missing"
            missing.mkdir()
            complete = wf_dir / "wf_complete"
            complete.mkdir()
            pd.DataFrame({"window": ["wf_01"], "selected_candidate": ["c1"]}).to_csv(
                complete / "walk_forward_summary.csv",
                index=False,
            )
            pd.DataFrame({"window": ["wf_01"], "candidate": ["c1"]}).to_csv(
                complete / "candidate_results.csv",
                index=False,
            )
            pd.DataFrame({"date": ["2020-01-01"], "stitched_equity": [100000.0]}).to_csv(
                complete / "oos_equity_stitched.csv",
                index=False,
            )
            (complete / "wf_01_selected_params.json").write_text("{}", encoding="utf-8")
            (complete / "summary.md").write_text(
                "| Metric | Value |\n| --- | ---: |\n| Stitched OOS total return | 1.00% |\n| Candidate grid | compact |\n",
                encoding="utf-8",
            )
            result = run_model_build_audit(
                root,
                config_dir=config_dir,
                walk_forward_dir=wf_dir,
                lock_dir=root / "outputs" / "locks",
                output_dir=root / "outputs" / "audit",
            )
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.config_map_path.exists())
            self.assertGreaterEqual(int(result.snapshot["duplicate_config_groups"]), 1)
            self.assertEqual(int(result.snapshot["root_config_files"]), 3)
            self.assertEqual(int(result.snapshot["base_config_files"]), 1)
            self.assertEqual(int(result.snapshot["root_configs_with_extends"]), 1)
            self.assertEqual(int(result.snapshot["root_configs_without_extends"]), 2)
            self.assertEqual(int(result.snapshot["incomplete_walk_forward_runs"]), 2)
            self.assertEqual(int(result.snapshot["walk_forward_action_items"]), 2)
            self.assertEqual(int(result.snapshot["walk_forward_resume_candidates"]), 1)
            self.assertEqual(int(result.snapshot["walk_forward_archive_review_candidates"]), 1)
            config_map = pd.read_csv(result.config_map_path)
            self.assertIn("base/base.yaml", set(config_map["extends"]))
            walk_runs = pd.read_csv(result.walk_forward_runs_path)
            self.assertIn("partial_window_outputs", set(walk_runs["status"]))
            actions = pd.read_csv(result.walk_forward_actions_path)
            action_by_run = dict(zip(actions["run_name"], actions["recommended_action"]))
            self.assertEqual(action_by_run["wf_partial"], "resume_or_finalize")
            self.assertEqual(action_by_run["wf_missing"], "archive_candidate_after_review")
            self.assertTrue(bool(actions.loc[actions["run_name"] == "wf_missing", "requires_manual_confirmation"].iloc[0]))

