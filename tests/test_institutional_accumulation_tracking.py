from pathlib import Path

import pandas as pd

from institutional_accumulation_shadow import AccumulationConfig
from track_institutional_accumulation_shadow import (
    load_shadow_watchlists,
    summarize_tracking,
    write_outputs,
)


def test_loader_filters_non_tracking_rows_and_uses_filename_date(tmp_path: Path):
    path = tmp_path / "institutional_accumulation_shadow_20260720.csv"
    pd.DataFrame(
        [
            {
                "symbol": "000001",
                "institutional_accumulation_score": 80,
                "tracking_eligible": True,
                "signal_active": True,
            },
            {
                "symbol": "000002",
                "institutional_accumulation_score": 40,
                "tracking_eligible": False,
                "signal_active": False,
            },
        ]
    ).to_csv(path, index=False, encoding="utf-8-sig")

    table = load_shadow_watchlists([path])

    assert table["symbol"].tolist() == ["000001"]
    assert table["watch_date"].tolist() == ["2026-07-20"]


def test_gate_stays_provisional_before_preregistered_sample_minimum():
    table = pd.DataFrame(
        {
            "symbol": ["000001", "000002"],
            "watch_date": ["2026-07-20", "2026-07-21"],
            "signal_active": [True, True],
            "institutional_accumulation_level": ["建仓观察", "强建仓迹象"],
            "forward_return_1d": [0.01, -0.01],
            "forward_return_3d": [0.03, 0.01],
            "forward_return_5d": [0.05, 0.02],
            "forward_return_10d": [pd.NA, pd.NA],
        }
    )

    summary_table, summary = summarize_tracking(
        table,
        config=AccumulationConfig(minimum_completed_samples=80, minimum_bucket_samples=20),
        analysis_end_date="2026-07-24",
    )

    assert summary_table.loc[summary_table["horizon"].eq(5), "completed_samples"].iloc[0] == 2
    assert summary["gate_evaluation_allowed"] is False
    assert summary["research_gate_passed"] is None
    assert summary["promotion_allowed"] is False
    assert summary["schema_version"] == 2
    assert len(summary["scorer_implementation_sha256"]) == 64


def test_pre_registration_active_rows_cannot_enter_validation_gate():
    table = pd.DataFrame(
        {
            "symbol": ["000001"],
            "watch_date": ["2026-07-17"],
            "signal_active": [True],
            "institutional_accumulation_level": ["强建仓迹象"],
            "forward_return_5d": [0.20],
        }
    )

    _summary_table, summary = summarize_tracking(
        table,
        config=AccumulationConfig(minimum_completed_samples=1, minimum_bucket_samples=1),
        analysis_end_date="2026-07-24",
    )

    assert summary["active_signal_rows"] == 0
    assert summary["primary_completed_samples"] == 0
    assert summary["gate_evaluation_allowed"] is False


def test_tracking_writer_creates_chinese_summary_and_report(tmp_path: Path):
    table = pd.DataFrame([{"symbol": "000001", "tracking_status": "pending_entry"}])
    summary_table = pd.DataFrame(
        [{"horizon": 5, "completed_samples": 0, "mean_return": pd.NA, "median_return": pd.NA, "hit_rate": pd.NA}]
    )
    summary = {
        "status": "provisional",
        "analysis_end_date": "2026-07-20",
        "tracked_rows": 1,
        "active_signal_rows": 0,
        "primary_completed_samples": 0,
        "minimum_completed_samples": 80,
        "gate_evaluation_allowed": False,
    }

    output = tmp_path / "institutional_accumulation_tracking_20260720.csv"
    paths = write_outputs(table, summary_table, summary, output)

    assert all(path.exists() for path in paths.values())
    assert "自动晋级: 关闭" in paths["report"].read_text(encoding="utf-8")
