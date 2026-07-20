import tempfile
import unittest
from pathlib import Path

import pandas as pd

from update_model_panel_from_daily_data import (
    DEFAULT_BASE_PANEL,
    drop_all_zero_placeholders,
    load_daily_panel,
    read_daily_csv,
    read_daily_xls,
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

    def test_select_daily_files_ignores_backup_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            daily_dir = Path(tmp)
            canonical = daily_dir / "ths_hs_a_share_2026-07-02.xls"
            backup = daily_dir / "ths_hs_a_share_2026-07-02.prev-162821.xls"
            canonical.write_text("canonical", encoding="utf-8")
            backup.write_text("backup", encoding="utf-8")

            selected = select_daily_files(daily_dir, "2026-07-02", "2026-07-02")

        self.assertEqual(selected, [canonical])

    def test_read_daily_xls_preserves_main_net_inflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ths_hs_a_share_2026-07-14.xls"
            raw = pd.DataFrame(
                {
                    "代码": ["000001", "000002"],
                    "现价": [10.0, 20.0],
                    "开盘": [9.8, 19.8],
                    "最高": [10.2, 20.2],
                    "最低": [9.7, 19.7],
                    "换手": [1.0, 2.0],
                    "总市值": ["10亿", "20亿"],
                    "大单净额": ["1.2万", "-8000"],
                    "主力净量": ["0.77", "-0.40"],
                }
            )
            raw.to_csv(path, sep="\t", index=False, encoding="gb18030")

            result, summary = read_daily_xls(
                path,
                "2026-07-14",
                ("000",),
            )

        self.assertEqual(result["main_net_inflow"].tolist(), [12000.0, -8000.0])
        self.assertEqual(result["main_net_volume_ratio"].tolist(), [0.0077, -0.004])
        self.assertEqual(summary.money_flow_source, "大单净额")
        self.assertEqual(summary.money_flow_ratio_source, "主力净量")

    def test_read_daily_csv_keeps_normalized_ratio_in_decimal_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ths_hs_a_share_2026-07-14.csv"
            pd.DataFrame(
                {
                    "trade_date": ["2026-07-14"],
                    "security_code": ["000001"],
                    "open": [10.0],
                    "high": [10.5],
                    "low": [9.8],
                    "close": [10.2],
                    "volume": [1_000_000.0],
                    "amount": [10_200_000.0],
                    "main_net_volume_ratio": [0.0077],
                }
            ).to_csv(path, index=False)

            result, summary = read_daily_csv(path, "2026-07-14", ("000",))

        self.assertEqual(result["main_net_volume_ratio"].tolist(), [0.0077])
        self.assertEqual(summary.money_flow_ratio_source, "main_net_volume_ratio")

    def test_load_daily_panel_merges_xls_flow_onto_preferred_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            daily_dir = Path(tmp)
            csv_path = daily_dir / "ths_hs_a_share_2026-07-14.csv"
            xls_path = daily_dir / "ths_hs_a_share_2026-07-14.xls"
            pd.DataFrame(
                {
                    "trade_date": ["2026-07-14"],
                    "security_code": ["000001"],
                    "open": [10.0],
                    "high": [10.5],
                    "low": [9.8],
                    "close": [10.2],
                    "volume": [1_000_000.0],
                    "amount": [10_200_000.0],
                    "turnover_rate": [1.0],
                    "market_cap": [1_000_000_000.0],
                }
            ).to_csv(csv_path, index=False)
            pd.DataFrame(
                {
                    "代码": ["000001"],
                    "现价": [99.0],
                    "开盘": [99.0],
                    "最高": [99.0],
                    "最低": [99.0],
                    "换手": [1.0],
                    "总市值": ["10亿"],
                    "大单净额": ["2.5万"],
                    "主力净量": ["0.55"],
                }
            ).to_csv(xls_path, sep="\t", index=False, encoding="gb18030")

            result, summaries = load_daily_panel(
                daily_dir,
                "2026-07-14",
                "2026-07-14",
                ("000",),
            )

        self.assertEqual(result.iloc[0]["close"], 10.2)
        self.assertEqual(result.iloc[0]["main_net_inflow"], 25000.0)
        self.assertAlmostEqual(result.iloc[0]["main_net_volume_ratio"], 0.0055)
        self.assertEqual(summaries[0].money_flow_source, "xls:大单净额")
        self.assertEqual(summaries[0].money_flow_ratio_source, "xls:主力净量")


if __name__ == "__main__":
    unittest.main()
