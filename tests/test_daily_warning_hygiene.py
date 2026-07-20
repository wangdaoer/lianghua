from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd

from quant_etf_lab.daily_pipeline import HISTORY_COLUMNS, _append_history
from quant_etf_lab.market_cap import _numeric_series


def test_market_cap_numeric_series_does_not_emit_future_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        result = _numeric_series(pd.Series([np.nan, np.nan, np.nan]))

    assert result.isna().all()
    assert not [warning for warning in caught if issubclass(warning.category, FutureWarning)]


def test_append_history_does_not_emit_concat_future_warning(tmp_path) -> None:
    history_path = tmp_path / "daily_pipeline_history.csv"
    old_row = {column: None for column in HISTORY_COLUMNS}
    old_row.update(
        {
            "generated_at": "2026-06-18T16:10:00",
            "as_of_date": "2026-06-18",
            "dashboard_posture": "core_base_watch_allocator_gate",
        }
    )
    pd.DataFrame([old_row], columns=HISTORY_COLUMNS).to_csv(history_path, index=False)
    snapshot = {
        "generated_at": "2026-06-19T16:10:00",
        "as_of_date": "2026-06-19",
        "dashboard_posture": "defensive_review_only",
        "action_posture": "review_allocator_satellite_budget",
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        _append_history(snapshot, history_path, date(2026, 6, 19))

    assert snapshot["history_status"] == "appended"
    assert not [warning for warning in caught if issubclass(warning.category, FutureWarning)]
