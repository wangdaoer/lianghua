from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from quant_etf_lab.paper_account import _publish_stock_target_review_decision_template


def test_decision_template_excel_rebuilds_only_when_source_changes() -> None:
    template = pd.DataFrame(columns=["code", "manual_status_to_fill"])
    payload = {"generated_at": "2026-07-20T18:00:00", "latest_date": "2026-07-20"}

    with TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        paths = {
            "csv_path": output_dir / "template.csv",
            "json_path": output_dir / "template.json",
            "report_path": output_dir / "template.md",
            "xlsx_path": output_dir / "template.xlsx",
        }

        def fake_excel(_template: pd.DataFrame, _payload: dict, path: Path) -> Path:
            Path(path).write_bytes(b"workbook")
            return Path(path)

        with patch(
            "quant_etf_lab.paper_account.write_stock_target_review_decision_template_xlsx",
            side_effect=fake_excel,
        ) as writer:
            assert _publish_stock_target_review_decision_template(
                template, payload, output_dir=output_dir, **paths
            )
            assert not _publish_stock_target_review_decision_template(
                template,
                {**payload, "generated_at": "2026-07-20T18:01:00"},
                output_dir=output_dir,
                **paths,
            )
            assert _publish_stock_target_review_decision_template(
                template,
                {**payload, "latest_date": "2026-07-21"},
                output_dir=output_dir,
                **paths,
            )

        assert writer.call_count == 2
