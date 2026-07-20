from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from quant_etf_lab.paper_account import _stock_price_history


def test_stock_price_history_reuses_single_run_cache() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        stock_dir = root / "data" / "processed" / "stocks"
        stock_dir.mkdir(parents=True)
        pd.DataFrame(
            [{"date": "2026-07-20", "close": 12.3, "high": 12.5, "low": 12.0}]
        ).to_csv(stock_dir / "000001.csv", index=False)
        cache: dict = {}
        original = pd.read_csv

        with patch("quant_etf_lab.paper_account.pd.read_csv", side_effect=original) as reader:
            first, _ = _stock_price_history(root, "000001", cache)
            second, _ = _stock_price_history(root, "000001", cache)

        assert reader.call_count == 1
        assert first is second


def test_stock_price_history_reuses_persistent_numeric_cache() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        stock_dir = root / "data" / "processed" / "stocks"
        stock_dir.mkdir(parents=True)
        pd.DataFrame(
            [{"date": "2026-07-20", "close": 12.3, "high": 12.5, "low": 12.0}]
        ).to_csv(stock_dir / "000001.csv", index=False)
        persistent = root / "cache"

        first, _ = _stock_price_history(root, "000001", {}, persistent)
        with patch("quant_etf_lab.paper_account.pd.read_csv", side_effect=AssertionError("unexpected CSV read")):
            second, _ = _stock_price_history(root, "000001", {}, persistent)

        pd.testing.assert_frame_equal(first, second)
