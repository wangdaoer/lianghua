import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from main_net_volume_shadow import (
    SOURCE_COLUMN,
    build_latest_shadow_table,
    build_metadata,
    compute_main_net_volume_features,
    write_outputs,
)


def make_panel(periods: int = 7) -> pd.DataFrame:
    dates = pd.date_range("2026-07-01", periods=periods, freq="B")
    rows = []
    for symbol, offset in (("000001", 0.0), ("000002", 0.01)):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": 10.0 + index + offset,
                    SOURCE_COLUMN: 0.001 * (index + 1) + offset,
                }
            )
    return pd.DataFrame(rows)


class MainNetVolumeShadowTest(unittest.TestCase):
    def test_rolling_features_require_full_history_and_do_not_fill_missing(self):
        panel = make_panel(6)
        missing_date = sorted(panel["date"].unique())[2]
        panel.loc[
            panel["symbol"].eq("000002") & panel["date"].eq(missing_date),
            SOURCE_COLUMN,
        ] = np.nan

        features = compute_main_net_volume_features(panel)
        first = features[features["symbol"].eq("000001")].reset_index(drop=True)
        second = features[features["symbol"].eq("000002")].reset_index(drop=True)

        self.assertTrue(pd.isna(first.loc[3, "main_net_volume_ratio_5d"]))
        self.assertAlmostEqual(first.loc[4, "main_net_volume_ratio_5d"], 0.003)
        self.assertTrue(pd.isna(second.loc[4, "main_net_volume_ratio_5d"]))
        self.assertEqual(second.loc[4, "flow_observation_count"], 4)

    def test_future_changes_do_not_change_previous_shadow_features(self):
        panel = make_panel(7)
        baseline = compute_main_net_volume_features(panel)
        changed = panel.copy()
        last_date = changed["date"].max()
        changed.loc[changed["date"].eq(last_date), SOURCE_COLUMN] = 0.50
        changed.loc[changed["date"].eq(last_date), "close"] = 1000.0
        evaluated = compute_main_net_volume_features(changed)
        compare_date = sorted(panel["date"].unique())[-2]
        columns = [
            SOURCE_COLUMN,
            "main_net_volume_ratio_5d",
            "positive_ratio_5d",
            "flow_price_divergence_5d",
        ]
        left = baseline[baseline["date"].eq(compare_date)].sort_values("symbol")[columns]
        right = evaluated[evaluated["date"].eq(compare_date)].sort_values("symbol")[columns]

        pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True))

    def test_latest_table_marks_pattern_overlap_without_affecting_selection(self):
        features = compute_main_net_volume_features(make_panel(6))
        early = pd.DataFrame(
            {
                "symbol": ["000001"],
                "pattern_type": ["隐性吸筹观察"],
                "pattern_score": [1.2],
                "hidden_accumulation_trade_watch": [True],
            }
        )

        latest = build_latest_shadow_table(
            features,
            asof_date="2026-07-08",
            name_map={"000001": "平安银行"},
            early_watchlist=early,
        )
        row = latest.set_index("symbol").loc["000001"]

        self.assertEqual(row["stock_name"], "平安银行")
        self.assertTrue(bool(row["early_pattern_match"]))
        self.assertTrue(bool(row["hidden_accumulation_match"]))
        self.assertEqual(row["shadow_status"], "5日影子可观察")
        self.assertFalse(bool(row["selection_effect"]))

    def test_metadata_stays_in_warmup_before_five_source_sessions(self):
        panel = make_panel(2)
        features = compute_main_net_volume_features(panel)
        latest = build_latest_shadow_table(features, asof_date="2026-07-02")

        metadata = build_metadata(
            features,
            latest,
            requested_asof_date="2026-07-02",
            source_column_available=True,
            minimum_history=5,
        )

        self.assertEqual(metadata["status"], "warmup")
        self.assertEqual(metadata["available_source_sessions"], 2)
        self.assertEqual(metadata["eligible_5d_rows"], 0)
        self.assertFalse(metadata["selection_effect"])

    def test_write_outputs_creates_chinese_table_metadata_report_and_chart(self):
        features = compute_main_net_volume_features(make_panel(6))
        latest = build_latest_shadow_table(features, asof_date="2026-07-08")
        metadata = build_metadata(
            features,
            latest,
            requested_asof_date="2026-07-08",
            source_column_available=True,
            minimum_history=5,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "main_net_volume_shadow_20260708.csv"

            paths = write_outputs(latest, metadata, output, top_n=4)

            for path in paths.values():
                self.assertTrue(path.exists(), path)
            payload = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "research_ready")
            chinese = pd.read_csv(paths["chinese_csv"])
            self.assertIn("主力净量", chinese.columns)


if __name__ == "__main__":
    unittest.main()
