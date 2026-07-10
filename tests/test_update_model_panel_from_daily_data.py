import tempfile
import unittest
from pathlib import Path

from update_model_panel_from_daily_data import select_daily_files


class UpdateModelPanelFromDailyDataTest(unittest.TestCase):
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
