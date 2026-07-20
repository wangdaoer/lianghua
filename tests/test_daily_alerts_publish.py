from pathlib import Path
from tempfile import TemporaryDirectory
import time

from quant_etf_lab.daily_alerts import write_daily_alerts


def test_daily_alerts_can_be_computed_without_publishing_files() -> None:
    with TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "alerts"
        result = write_daily_alerts(
            {"as_of_date": "2026-07-20"},
            output_dir,
            publish_artifacts=False,
        )

        assert result.payload["alerts_report_path"] == str(output_dir / "alerts.md")
        assert not output_dir.exists()


def test_daily_alerts_do_not_rewrite_semantically_unchanged_artifacts() -> None:
    with TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "alerts"
        first = write_daily_alerts({"as_of_date": "2026-07-20"}, output_dir)
        paths = [first.json_path, first.report_path, first.latest_report_path]
        mtimes = {path: path.stat().st_mtime_ns for path in paths}

        time.sleep(1.05)
        second = write_daily_alerts({"as_of_date": "2026-07-20"}, output_dir)

        assert second.payload["generated_at"] == first.payload["generated_at"]
        assert {path: path.stat().st_mtime_ns for path in paths} == mtimes
