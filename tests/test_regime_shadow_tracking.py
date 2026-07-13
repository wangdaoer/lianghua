import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

try:
    from update_regime_shadow_tracking import run_tracking
except ModuleNotFoundError:
    run_tracking = None


class RegimeShadowTrackingTest(unittest.TestCase):
    def test_valid_comparison_writes_collecting_summary(self):
        self.assertIsNotNone(run_tracking, "update_regime_shadow_tracking.py must define run_tracking")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison_dir = root / "comparison"
            ledger = root / "regime_shadow_tracking.csv"
            summary = root / "regime_shadow_tracking_summary.json"
            report = root / "regime_shadow_tracking_report.md"
            self._write_comparison_snapshot(
                comparison_dir,
                asof_date="2026-07-13",
                baseline_points=[
                    {"date": "2026-07-10", "equity": 100.0, "turnover": 0.20, "cost": 0.0010, "gross_exposure": 0.60},
                    {"date": "2026-07-13", "equity": 101.0, "turnover": 0.21, "cost": 0.0012, "gross_exposure": 0.61, "gross_return": 0.0112},
                ],
                dynamic_points=[
                    {
                        "date": "2026-07-10",
                        "equity": 100.0,
                        "turnover": 0.24,
                        "cost": 0.0011,
                        "gross_exposure": 0.66,
                        "risk_regime": "base",
                        "target_leverage": 0.60,
                    },
                    {
                        "date": "2026-07-13",
                        "equity": 101.2,
                        "turnover": 0.25,
                        "cost": 0.0013,
                        "gross_exposure": 0.68,
                        "gross_return": 0.0133,
                        "risk_regime": "strong",
                        "target_leverage": 0.75,
                    },
                ],
            )

            result = run_tracking(
                argparse.Namespace(
                    comparison_dir=str(comparison_dir),
                    ledger=str(ledger),
                    summary=str(summary),
                    report=str(report),
                    target_days=20,
                )
            )

            rows = self._read_csv_rows(ledger)
            payload = json.loads(summary.read_text(encoding="utf-8"))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["asof_date"], "2026-07-13")
            self.assertEqual(rows[0]["observation_valid"], "true")
            self.assertEqual(payload["status"], "collecting")
            self.assertEqual(payload["valid_observation_count"], 1)
            self.assertEqual(payload["invalid_observation_count"], 0)
            self.assertEqual(payload["remaining_days"], 19)
            self.assertFalse(payload["automatic_promotion"])
            self.assertEqual(result["status"], "collecting")

    def test_rerunning_same_date_replaces_row_without_duplication(self):
        self.assertIsNotNone(run_tracking, "update_regime_shadow_tracking.py must define run_tracking")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison_dir = root / "comparison"
            ledger = root / "regime_shadow_tracking.csv"
            summary = root / "regime_shadow_tracking_summary.json"
            report = root / "regime_shadow_tracking_report.md"
            self._write_comparison_snapshot(
                comparison_dir,
                asof_date="2026-07-13",
                baseline_points=[
                    {"date": "2026-07-10", "equity": 100.0, "turnover": 0.20, "cost": 0.0010, "gross_exposure": 0.60},
                    {"date": "2026-07-13", "equity": 101.0, "turnover": 0.21, "cost": 0.0012, "gross_exposure": 0.61, "gross_return": 0.0112},
                ],
                dynamic_points=[
                    {"date": "2026-07-10", "equity": 100.0, "turnover": 0.24, "cost": 0.0011, "gross_exposure": 0.66},
                    {"date": "2026-07-13", "equity": 101.2, "turnover": 0.25, "cost": 0.0013, "gross_exposure": 0.68, "gross_return": 0.0133},
                ],
            )

            run_tracking(
                argparse.Namespace(
                    comparison_dir=str(comparison_dir),
                    ledger=str(ledger),
                    summary=str(summary),
                    report=str(report),
                    target_days=20,
                )
            )

            self._write_comparison_snapshot(
                comparison_dir,
                asof_date="2026-07-13",
                baseline_points=[
                    {"date": "2026-07-10", "equity": 100.0, "turnover": 0.22, "cost": 0.0010, "gross_exposure": 0.60},
                    {"date": "2026-07-13", "equity": 102.0, "turnover": 0.23, "cost": 0.0010, "gross_exposure": 0.62, "gross_return": 0.0210},
                ],
                dynamic_points=[
                    {"date": "2026-07-10", "equity": 100.0, "turnover": 0.25, "cost": 0.0012, "gross_exposure": 0.67},
                    {"date": "2026-07-13", "equity": 102.4, "turnover": 0.26, "cost": 0.0014, "gross_exposure": 0.69, "gross_return": 0.0254},
                ],
            )

            run_tracking(
                argparse.Namespace(
                    comparison_dir=str(comparison_dir),
                    ledger=str(ledger),
                    summary=str(summary),
                    report=str(report),
                    target_days=20,
                )
            )

            rows = self._read_csv_rows(ledger)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["baseline_equity"], "102.0")
            self.assertEqual(rows[0]["dynamic_equity"], "102.4")

    def test_stale_benchmark_row_is_recorded_but_invalid(self):
        self.assertIsNotNone(run_tracking, "update_regime_shadow_tracking.py must define run_tracking")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison_dir = root / "comparison"
            ledger = root / "regime_shadow_tracking.csv"
            summary = root / "regime_shadow_tracking_summary.json"
            report = root / "regime_shadow_tracking_report.md"
            self._write_comparison_snapshot(
                comparison_dir,
                asof_date="2026-07-13",
                benchmark_last_date="2026-07-10",
                benchmark_fresh=False,
                baseline_points=[
                    {"date": "2026-07-10", "equity": 100.0, "turnover": 0.20, "cost": 0.0010, "gross_exposure": 0.60},
                    {"date": "2026-07-13", "equity": 101.0, "turnover": 0.21, "cost": 0.0012, "gross_exposure": 0.61, "gross_return": 0.0112},
                ],
                dynamic_points=[
                    {"date": "2026-07-10", "equity": 100.0, "turnover": 0.24, "cost": 0.0011, "gross_exposure": 0.66},
                    {"date": "2026-07-13", "equity": 100.9, "turnover": 0.25, "cost": 0.0013, "gross_exposure": 0.68, "gross_return": 0.0103},
                ],
            )

            run_tracking(
                argparse.Namespace(
                    comparison_dir=str(comparison_dir),
                    ledger=str(ledger),
                    summary=str(summary),
                    report=str(report),
                    target_days=20,
                )
            )

            rows = self._read_csv_rows(ledger)
            payload = json.loads(summary.read_text(encoding="utf-8"))

            self.assertEqual(rows[0]["benchmark_fresh"], "false")
            self.assertEqual(rows[0]["observation_valid"], "false")
            self.assertEqual(payload["valid_observation_count"], 0)
            self.assertEqual(payload["invalid_observation_count"], 1)
            self.assertEqual(payload["remaining_days"], 20)

    def test_twenty_valid_observations_require_manual_review_only(self):
        self.assertIsNotNone(run_tracking, "update_regime_shadow_tracking.py must define run_tracking")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison_dir = root / "comparison"
            ledger = root / "regime_shadow_tracking.csv"
            summary = root / "regime_shadow_tracking_summary.json"
            report = root / "regime_shadow_tracking_report.md"
            dates = [f"2026-07-{day:02d}" for day in range(1, 21)]
            baseline_equity = 100.0
            dynamic_equity = 100.0

            for index, asof_date in enumerate(dates, start=1):
                next_baseline = baseline_equity * 1.01
                next_dynamic = dynamic_equity * 1.012
                self._write_comparison_snapshot(
                    comparison_dir,
                    asof_date=asof_date,
                    baseline_points=[
                        {"date": f"2026-06-{30 if index == 1 else index - 1:02d}", "equity": baseline_equity, "turnover": 0.20, "cost": 0.0010, "gross_exposure": 0.60},
                        {"date": asof_date, "equity": next_baseline, "turnover": 0.20, "cost": 0.0010, "gross_exposure": 0.60, "gross_return": 0.0110},
                    ],
                    dynamic_points=[
                        {"date": f"2026-06-{30 if index == 1 else index - 1:02d}", "equity": dynamic_equity, "turnover": 0.24, "cost": 0.0010, "gross_exposure": 0.64, "risk_regime": "base", "target_leverage": 0.60},
                        {"date": asof_date, "equity": next_dynamic, "turnover": 0.24, "cost": 0.0010, "gross_exposure": 0.64, "gross_return": 0.0130, "risk_regime": "strong", "target_leverage": 0.75},
                    ],
                )
                baseline_equity = next_baseline
                dynamic_equity = next_dynamic

                result = run_tracking(
                    argparse.Namespace(
                        comparison_dir=str(comparison_dir),
                        ledger=str(ledger),
                        summary=str(summary),
                        report=str(report),
                        target_days=20,
                    )
                )

            payload = json.loads(summary.read_text(encoding="utf-8"))

            self.assertEqual(payload["valid_observation_count"], 20)
            self.assertEqual(payload["invalid_observation_count"], 0)
            self.assertEqual(payload["status"], "manual_review_ready")
            self.assertFalse(payload["automatic_promotion"])
            self.assertEqual(result["status"], "manual_review_ready")

    def test_input_failure_keeps_existing_outputs_unchanged(self):
        self.assertIsNotNone(run_tracking, "update_regime_shadow_tracking.py must define run_tracking")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            comparison_dir = root / "comparison"
            ledger = root / "regime_shadow_tracking.csv"
            summary = root / "regime_shadow_tracking_summary.json"
            report = root / "regime_shadow_tracking_report.md"
            ledger.write_text("sentinel-ledger\n", encoding="utf-8")
            summary.write_text('{"sentinel": true}\n', encoding="utf-8")
            report.write_text("sentinel-report\n", encoding="utf-8")
            original = {
                "ledger": ledger.read_text(encoding="utf-8"),
                "summary": summary.read_text(encoding="utf-8"),
                "report": report.read_text(encoding="utf-8"),
            }
            self._write_comparison_snapshot(
                comparison_dir,
                asof_date="2026-07-13",
                baseline_points=[
                    {"date": "2026-07-10", "equity": 100.0, "turnover": 0.20, "cost": 0.0010, "gross_exposure": 0.60},
                    {"date": "2026-07-12", "equity": 101.0, "turnover": 0.21, "cost": 0.0012, "gross_exposure": 0.61, "gross_return": 0.0112},
                ],
                dynamic_points=[
                    {"date": "2026-07-10", "equity": 100.0, "turnover": 0.24, "cost": 0.0011, "gross_exposure": 0.66},
                    {"date": "2026-07-13", "equity": 101.2, "turnover": 0.25, "cost": 0.0013, "gross_exposure": 0.68, "gross_return": 0.0133},
                ],
            )

            with self.assertRaisesRegex(ValueError, "asof"):
                run_tracking(
                    argparse.Namespace(
                        comparison_dir=str(comparison_dir),
                        ledger=str(ledger),
                        summary=str(summary),
                        report=str(report),
                        target_days=20,
                    )
                )

            self.assertEqual(ledger.read_text(encoding="utf-8"), original["ledger"])
            self.assertEqual(summary.read_text(encoding="utf-8"), original["summary"])
            self.assertEqual(report.read_text(encoding="utf-8"), original["report"])

    def _write_comparison_snapshot(
        self,
        comparison_dir: Path,
        *,
        asof_date: str,
        baseline_points: list[dict[str, object]],
        dynamic_points: list[dict[str, object]],
        benchmark_last_date: str | None = None,
        benchmark_fresh: bool = True,
    ) -> None:
        (comparison_dir / "baseline").mkdir(parents=True, exist_ok=True)
        (comparison_dir / "dynamic").mkdir(parents=True, exist_ok=True)
        comparison_payload = {
            "asof": asof_date,
            "asof_date": asof_date,
            "decision": "experimental_only",
            "benchmark_last_date": benchmark_last_date or asof_date,
            "benchmark_fresh": benchmark_fresh,
            "latest_dynamic_state": {
                "risk_regime": str(dynamic_points[-1].get("risk_regime", "strong")),
                "target_leverage": float(dynamic_points[-1].get("target_leverage", 0.75)),
            },
            "delta": {
                "total_return": 0.01,
                "max_drawdown": 0.0,
            },
        }
        (comparison_dir / "comparison.json").write_text(
            json.dumps(comparison_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._write_csv(comparison_dir / "baseline" / "equity_curve.csv", baseline_points)
        self._write_csv(comparison_dir / "dynamic" / "equity_curve.csv", dynamic_points)

    @staticmethod
    def _read_csv_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        columns = sorted({column for row in rows for column in row})
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
