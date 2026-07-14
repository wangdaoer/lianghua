import json
import tempfile
import unittest
from pathlib import Path

from audit_legacy_10000pct_reproduction import (
    AuditCase,
    collect_results,
    evaluate_audit,
    write_markdown,
)


def write_metrics(path: Path, **overrides):
    payload = {
        "initial_capital": 1_000_000,
        "final_equity": 2_000_000,
        "total_return": 1.0,
        "annualized_return": 0.2,
        "max_drawdown": -0.1,
        "trade_days": 100,
        "avg_gross_exposure": 0.5,
        "avg_turnover": 0.2,
    }
    payload.update(overrides)
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


class Legacy10000PctReproductionAuditTest(unittest.TestCase):
    def test_audit_blocks_promotion_when_only_legacy_curve_exceeds_10000pct(self):
        legacy = type(
            "Row",
            (),
            {
                "status": "ok",
                "role": "legacy",
                "total_return_pct": 11623.01,
            },
        )()
        strict = type(
            "Row",
            (),
            {
                "status": "ok",
                "role": "strict",
                "total_return_pct": -30.08,
            },
        )()

        summary = evaluate_audit([legacy, strict])

        self.assertEqual(
            summary["audit_status"],
            "legacy_curve_reproducible_under_old_assumptions_only",
        )
        self.assertTrue(summary["legacy_10000pct_observed"])
        self.assertFalse(summary["strict_10000pct_observed"])
        self.assertFalse(summary["promote_to_current_simulation"])

    def test_collect_results_reads_existing_metrics_and_writes_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            legacy_dir = tmp_path / "legacy"
            strict_dir = tmp_path / "strict"
            write_metrics(legacy_dir, final_equity=117_230_117.83, total_return=116.2301)
            write_metrics(strict_dir, final_equity=699_202.72, total_return=-0.3008)
            cases = [
                AuditCase(
                    case_id="legacy",
                    group="g",
                    role="legacy",
                    label="legacy",
                    data_path=None,
                    config_path=None,
                    existing_output_dir=legacy_dir,
                    assumption="old",
                    note="legacy note",
                ),
                AuditCase(
                    case_id="strict",
                    group="g",
                    role="strict",
                    label="strict",
                    data_path=None,
                    config_path=None,
                    existing_output_dir=strict_dir,
                    assumption="strict",
                    note="strict note",
                ),
            ]

            rows = collect_results(cases, tmp_path / "out", rerun=False)
            summary = evaluate_audit(rows)
            report_path = tmp_path / "audit.md"
            write_markdown(rows, summary, report_path)

            self.assertEqual(rows[0].total_return_pct, 11623.01)
            self.assertEqual(rows[1].total_return_pct, -30.08)
            self.assertIn("不得把旧策略直接并入当前模拟盘", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
