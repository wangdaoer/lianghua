from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from quant_etf_lab.pipeline_history import run_pipeline_history_review


def test_pipeline_history_review_uses_supplied_frame_without_reading_csv() -> None:
    history = pd.DataFrame(
        [
            {
                "generated_at": "2026-07-20T18:00:00",
                "as_of_date": "2026-07-20",
                "data_freshness_status": "fresh_enough",
                "market_cache_status": "fresh_enough",
                "allocator_input_status": "fresh_enough",
                "paper_freshness_status": "fresh_enough",
                "sentiment_freshness_status": "fresh_enough",
                "trigger_freshness_status": "fresh_enough",
                "dashboard_posture": "core_base_watch_allocator_gate",
                "paper_final_equity": 1_000_000.0,
                "paper_total_return": 0.0,
                "paper_current_drawdown": 0.0,
                "paper_sharpe": 1.0,
            }
        ]
    )

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        with patch("quant_etf_lab.pipeline_history.pd.read_csv") as read_csv:
            result = run_pipeline_history_review(
                project_root=root,
                history_file=root / "does_not_need_to_exist.csv",
                output_dir=root / "review",
                as_of_date="2026-07-20",
                history_frame=history,
            )

        read_csv.assert_not_called()
        assert result.snapshot["history_status"] == "ok"
        assert result.snapshot["history_row_count"] == 1
        assert result.snapshot["latest_as_of_date"] == "2026-07-20"
