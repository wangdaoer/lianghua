import json
import math
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from czsc_structure_shadow import (
    VALIDATED_CZSC_VERSION,
    CzscRuntime,
    analyze_symbol_history,
    build_metadata,
    build_structure_shadow,
    load_candidate_union,
    load_czsc_runtime,
    load_panel_prices,
    write_outputs,
)


def synthetic_history(periods: int = 260) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=periods)
    base = np.arange(periods, dtype=float)
    close = 10.0 + base * 0.006 + np.sin(base / 4.0) * 0.9
    previous = np.r_[close[0], close[:-1]]
    open_price = previous * (1.0 + np.sin(base / 7.0) * 0.002)
    high = np.maximum(open_price, close) * 1.012
    low = np.minimum(open_price, close) * 0.988
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1_000_000 + base * 100,
            "amount": (1_000_000 + base * 100) * close,
        }
    )


class CzscStructureShadowTest(unittest.TestCase):
    def test_candidate_union_keeps_source_decisions_and_pattern_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model_20260715.csv"
            early = root / "early_20260715.csv"
            pd.DataFrame(
                {
                    "symbol": ["1", "000002"],
                    "stock_name": ["平安银行", ""],
                    "personal_rank": [2, 3],
                    "personal_selected": [True, False],
                    "trend_state": ["strong", "weak"],
                }
            ).to_csv(model, index=False)
            pd.DataFrame(
                {
                    "symbol": ["000001", "000003"],
                    "stock_name": ["", "测试三"],
                    "pattern_type": ["强势回调", "隐性吸筹"],
                    "pattern_score": [1.2, 0.8],
                }
            ).to_csv(early, index=False)

            result = load_candidate_union([model, early]).set_index("symbol")

            self.assertEqual(set(result.index), {"000001", "000002", "000003"})
            self.assertEqual(result.loc["000001", "stock_name"], "平安银行")
            self.assertEqual(result.loc["000001", "early_pattern_type"], "强势回调")
            self.assertTrue(bool(result.loc["000001", "model_selected"]))
            self.assertEqual(result.loc["000001", "model_rank"], 2)
            self.assertIn(model.name, result.loc["000001", "candidate_sources"])
            self.assertIn(early.name, result.loc["000001", "candidate_sources"])

    def test_panel_loader_excludes_future_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel.csv"
            frame = synthetic_history(5)
            future = frame.iloc[[-1]].copy()
            future["date"] = pd.Timestamp("2027-01-01")
            future["close"] = 999.0
            pd.concat([frame, future], ignore_index=True).to_csv(path, index=False)

            loaded = load_panel_prices(path, ["000001"], asof_date=frame.iloc[-2]["date"])

            self.assertEqual(loaded["date"].max(), frame.iloc[-2]["date"].normalize())
            self.assertNotIn(999.0, loaded["close"].tolist())

    def test_runtime_rejects_unvalidated_version(self):
        module = types.ModuleType("czsc")
        module.__version__ = "1.0.0-rc.8"
        signals = types.ModuleType("czsc.signals")

        runtime = load_czsc_runtime(
            lambda name: module if name == "czsc" else signals,
        )

        self.assertFalse(runtime.available)
        self.assertEqual(runtime.version, "1.0.0-rc.8")
        self.assertIn("unsupported_czsc_version", runtime.reason)

    def test_runtime_unavailable_still_writes_auditable_candidate_rows(self):
        candidates = pd.DataFrame(
            {
                "symbol": ["000001"],
                "stock_name": ["平安银行"],
                "early_pattern_type": ["强势回调"],
            }
        )
        runtime = CzscRuntime(False, None, "czsc_not_installed")

        table = build_structure_shadow(
            candidates,
            pd.DataFrame(),
            runtime,
            asof_date="2026-07-15",
        )

        self.assertEqual(table.loc[0, "analysis_status"], "runtime_unavailable")
        self.assertTrue(bool(table.loc[0, "research_only"]))
        self.assertFalse(bool(table.loc[0, "selection_effect"]))
        self.assertEqual(table.loc[0, "signal_known_at"], "asof_close")
        self.assertEqual(table.loc[0, "earliest_action"], "next_trading_session_open")
        self.assertFalse(bool(table.loc[0, "endpoint_backfill"]))
        metadata = build_metadata(
            table,
            runtime,
            asof_date="2026-07-15",
            panel_path=Path("panel.csv"),
            candidate_paths=[Path("candidates.csv")],
            min_bars=100,
            max_abs_daily_return=0.22,
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_outputs(
                table,
                metadata,
                Path(tmp) / "czsc_structure_shadow_20260715.csv",
            )
            self.assertTrue(paths["report"].exists())
            self.assertEqual(metadata["status"], "unavailable")

    def test_installed_validated_runtime_analyzes_completed_history(self):
        runtime = load_czsc_runtime()
        if not runtime.available:
            self.skipTest(runtime.reason or "validated CZSC is unavailable")

        result = analyze_symbol_history(
            synthetic_history(),
            runtime,
            symbol="000001",
            asof_date="2026-12-31",
            min_bars=100,
        )

        self.assertEqual(runtime.version, VALIDATED_CZSC_VERSION)
        self.assertEqual(result["analysis_status"], "analyzed")
        self.assertGreater(result["completed_bi_count"], 3)
        self.assertIn(result["zone_location"], {"中枢上方", "中枢内部", "中枢下方", "无有效三笔中枢"})
        self.assertFalse(result["production_feature_eligible"])
        self.assertFalse(result["endpoint_backfill"])
        self.assertEqual(result["history_end_date"], synthetic_history().iloc[-1]["date"].strftime("%Y-%m-%d"))

    def test_price_jump_reset_can_make_history_ineligible(self):
        runtime = load_czsc_runtime()
        history = synthetic_history(150)
        history.loc[120:, ["open", "high", "low", "close"]] *= 2.0

        result = analyze_symbol_history(
            history,
            runtime,
            symbol="000001",
            asof_date="2026-12-31",
            min_bars=100,
            max_abs_daily_return=0.22,
        )

        if runtime.available:
            self.assertEqual(result["analysis_status"], "insufficient_history")
            self.assertEqual(result["history_bars"], 30)
            self.assertIsNotNone(result["history_truncated_at"])

    def test_outputs_include_chinese_headers_and_timing_metadata(self):
        table = pd.DataFrame(
            [
                {
                    "date": "2026-07-15",
                    "symbol": "000001",
                    "stock_name": "平安银行",
                    "analysis_status": "analyzed",
                    "pattern_confluence": True,
                    "third_buy_zone_consistent": False,
                    "second_buy_flag": True,
                    "second_sell_flag": False,
                    "overlap_support_flag": False,
                    "overlap_pressure_flag": False,
                    "zone_location": "中枢上方",
                    "last_bi_direction": "向下",
                    "risk_flags": "",
                    "research_only": True,
                    "selection_effect": False,
                    "portfolio_weight_effect": False,
                }
            ]
        )
        runtime = CzscRuntime(True, VALIDATED_CZSC_VERSION, None)
        metadata = build_metadata(
            table,
            runtime,
            asof_date="2026-07-15",
            panel_path=Path("panel.csv"),
            candidate_paths=[Path("candidates.csv")],
            min_bars=100,
            max_abs_daily_return=0.22,
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_outputs(
                table,
                metadata,
                Path(tmp) / "czsc_structure_shadow_20260715.csv",
                top_n=10,
            )

            for path in paths.values():
                self.assertTrue(path.exists(), path)
            chinese = pd.read_csv(paths["chinese_csv"])
            self.assertIn("股票代码", chinese.columns)
            payload = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            self.assertFalse(payload["historical_endpoint_backfill"])
            self.assertFalse(payload["selection_effect"])


if __name__ == "__main__":
    unittest.main()
