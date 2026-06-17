from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_etf_lab.chip_reversal_pit_audit import audit_chip_reversal_events, run_chip_reversal_pit_audit
from quant_etf_lab.cli import build_parser


def _write_prices(path: Path, code: str) -> None:
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "code": [code] * 10,
            "name": [code] * 10,
            "open": [10.0, 10.4, 10.8, 10.6, 9.2, 8.7, 8.7, 10.7, 11.0, 11.2],
            "high": [10.2, 10.6, 11.2, 10.9, 9.4, 8.8, 10.7, 11.0, 11.2, 11.4],
            "low": [9.8, 10.2, 10.6, 10.4, 8.8, 8.4, 8.6, 10.5, 10.8, 11.0],
            "close": [10.0, 10.5, 11.0, 10.8, 9.0, 8.6, 10.5, 10.9, 11.0, 11.2],
            "volume": [1_000_000] * 10,
            "amount": [100_000_000, 120_000_000, 150_000_000, 130_000_000, 170_000_000, 180_000_000, 160_000_000, 190_000_000, 140_000_000, 150_000_000],
        }
    ).to_csv(path, index=False)


def test_audit_chip_reversal_events_recomputes_signal_from_truncated_history() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "code": ["000001"] * 10,
            "name": ["Main"] * 10,
            "open": [10.0, 10.4, 10.8, 10.6, 9.2, 8.7, 8.7, 10.7, 11.0, 11.2],
            "high": [10.2, 10.6, 11.2, 10.9, 9.4, 8.8, 10.7, 11.0, 11.2, 11.4],
            "low": [9.8, 10.2, 10.6, 10.4, 8.8, 8.4, 8.6, 10.5, 10.8, 11.0],
            "close": [10.0, 10.5, 11.0, 10.8, 9.0, 8.6, 10.5, 10.9, 11.0, 11.2],
            "volume": [1_000_000] * 10,
            "amount": [100_000_000, 120_000_000, 150_000_000, 130_000_000, 170_000_000, 180_000_000, 160_000_000, 190_000_000, 140_000_000, 150_000_000],
        }
    )

    audit = audit_chip_reversal_events(
        prices,
        horizons=[1, 2],
        drawdown_window=3,
        cost_window=3,
        min_drawdown_pct=10.0,
        min_score=0.05,
        start_date="2026-01-06",
        end_date="2026-01-06",
    )

    assert audit["status"] == "pass"
    assert audit["audited_event_count"] == 1
    assert audit["pit_mismatch_count"] == 0
    assert audit["future_label_column_count"] >= 4
    assert "next_open" in audit["future_label_columns"]
    assert "return_1d" in audit["future_label_columns"]
    assert audit["selector_column_count"] > 0
    assert "chip_reversal_score" in audit["selector_columns"]


def test_run_chip_reversal_pit_audit_writes_research_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    _write_prices(data_dir / "000001.csv", "000001")
    _write_prices(data_dir / "300001.csv", "300001")

    result = run_chip_reversal_pit_audit(
        project_root=tmp_path,
        data_dir=data_dir,
        output_dir=tmp_path / "pit_audit",
        horizons=[1, 2],
        drawdown_window=3,
        cost_window=3,
        min_drawdown_pct=10.0,
        min_score=0.05,
        score_bucket="deep",
        start_date="2026-01-06",
        end_date="2026-01-06",
        max_events=10,
    )

    assert result.snapshot["status"] == "pass"
    assert result.snapshot["broker_action"] == "none"
    assert result.snapshot["research_only"] is True
    assert result.snapshot["pit_mismatch_count"] == 0
    assert result.audit_path.exists()
    assert result.snapshot_path.exists()
    assert result.report_path.exists()
    assert {"code", "date", "pit_status", "future_label_columns"}.issubset(result.audit.columns)
    assert set(result.audit["score_bucket"]) == {"deep"}
    assert result.snapshot["score_bucket_filter"] == "deep"
    assert "next_open" in result.report_path.read_text(encoding="utf-8")


def test_chip_reversal_pit_audit_cli_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["chip-reversal-pit-audit"])

    assert args.command == "chip-reversal-pit-audit"
    assert args.data_dir == "data/processed/stocks"
    assert args.output_dir == "outputs/research/chip_reversal_pit_audit_latest"
    assert args.horizons == "1,2"
    assert args.max_events == 2000
