from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from quant_etf_lab.paper_account import (
    _cached_stock_target_review_outcome_analysis,
    build_stock_target_review_outcome_analysis,
)


def test_outcome_analysis_cache_reuses_unchanged_history() -> None:
    history = pd.DataFrame(
        [
            {
                "date": "2026-07-20",
                "code": "000001",
                "return_1d": 0.02,
                "outcome_status_1d": "available",
            }
        ]
    )
    with TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "cache"
        with patch(
            "quant_etf_lab.paper_account.build_stock_target_review_outcome_analysis",
            wraps=build_stock_target_review_outcome_analysis,
        ) as builder:
            first = _cached_stock_target_review_outcome_analysis(
                history,
                {},
                min_evaluable=30,
                min_group_evaluable=5,
                cache_dir=cache_dir,
            )
            second = _cached_stock_target_review_outcome_analysis(
                history,
                {},
                min_evaluable=30,
                min_group_evaluable=5,
                cache_dir=cache_dir,
            )

        assert builder.call_count == 1
        assert first[1]["analysis_status"] == second[1]["analysis_status"]
