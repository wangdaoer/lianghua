import tempfile
import unittest
from pathlib import Path

import pandas as pd

from update_model_panel_from_daily_data import (
    DEFAULT_BASE_PANEL,
    drop_all_zero_placeholders,
    resolve_base_panel,
    select_daily_files,
)


class UpdateModelPanelFromDailyDataTest(unittest.TestCase):
    def test_drop_all_zero_placeholders_keeps_real_rows(self):
        frame = pd.DataFrame(
            {
                "symbol": ["000001", "000002"],
                "open": [0.0, 10.0],
                "high": [0.0, 10.5],
                "low": [0.0, 9.8],
                "close": [0.0, 10.2],
                "volume": [0.0, 100.0],
                "amount": [0.0, 1020.0],
            }
        )

        cleaned = drop_all_zero_placeholders(frame)

        self.assertEqual(cleaned["symbol"].tolist(), ["000002"])

    def test_drop_all_zero_placeholders_requires_complete_zero_signature(self):
        frame = pd.DataFrame(
            {
                "symbol": ["000001"],
                "open": [0.0],
                "high": [0.0],
                "low": [0.0],
                "close": [0.0],
                "volume": [0.0],
                "amount": [1.0],
            }
        )

        cleaned = drop_all_zero_placeholders(frame)

        self.assertEqual(cleaned["symbol"].tolist(), ["000001"])

    def test_resolve_base_panel_uses_latest_dated_panel_for_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "data_panel_history_main_chinext_20220101_20260710.csv"
            latest = root / "data_panel_history_main_chinext_20220101_20260714.csv"
            rolling_output = root / "data_panel_history_main_chinext_20220101_latest.csv"
            older.touch()
            latest.touch()
            rolling_output.touch()
            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(root)
                resolved = resolve_base_panel(DEFAULT_BASE_PANEL)
            finally:
                os.chdir(old_cwd)

        self.assertEqual(resolved, latest)

    def test_select_daily_files_accepts_future_year_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            daily_dir = Path(tmp)
            old_file = daily_dir / "ths_hs_a_share_2026-12-31.csv"
            target_csv = daily_dir / "ths_hs_a_share_2027-01-04.csv"
            target_xls = daily_dir / "ths_hs_a_share_2027-01-04.xls"
            ignored = daily_dir / "not_market_data_2027-01-04.csv"
            for path in [old_file, target_csv, target_xls, ignored]:
                path.write_text("placeholder", encoding="utf-8")

            selected = select_daily_files(daily_dir, "2027-01-01", None)

        self.assertEqual(selected, [target_csv])


if __name__ == "__main__":
    unittest.main()
