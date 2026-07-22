from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from track_dynamic_breadth_overlay import LEDGER_FIELDS, run_overlay


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "dynamic_breadth_overlay_preregistration.json"


def _write_equity_curve(path: Path, dates: pd.DatetimeIndex, daily_returns: list[float], *, initial: float = 100.0) -> None:
    assert len(dates) == len(daily_returns) + 1
    equity = [initial]
    for daily_return in daily_returns:
        equity.append(equity[-1] * (1.0 + daily_return))
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "equity": equity}).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def _run(
    tmp_path: Path,
    *,
    incumbent_path: Path,
    challenger_path: Path,
    ledger_path: Path | None = None,
    summary_path: Path | None = None,
    report_path: Path | None = None,
) -> dict[str, object]:
    ledger = ledger_path or (tmp_path / "dynamic_breadth_overlay_ledger.csv")
    summary = summary_path or (tmp_path / "dynamic_breadth_overlay_summary.json")
    report = report_path or (tmp_path / "dynamic_breadth_overlay_report.md")
    return run_overlay(
        argparse.Namespace(
            incumbent_equity=str(incumbent_path),
            challenger_equity=str(challenger_path),
            ledger=str(ledger),
            summary=str(summary),
            report=str(report),
            config=str(CONFIG),
        )
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_unmatured_forward_window_excludes_background_history(tmp_path: Path) -> None:
    dates = pd.bdate_range("2026-07-20", periods=4)
    incumbent_path = tmp_path / "incumbent.csv"
    challenger_path = tmp_path / "challenger.csv"
    _write_equity_curve(incumbent_path, dates, [0.01, 0.01, 0.01])
    _write_equity_curve(challenger_path, dates, [0.50, -0.02, -0.01])

    result = _run(tmp_path, incumbent_path=incumbent_path, challenger_path=challenger_path)

    ledger_rows = _read_csv_rows(tmp_path / "dynamic_breadth_overlay_ledger.csv")
    summary = json.loads((tmp_path / "dynamic_breadth_overlay_summary.json").read_text(encoding="utf-8"))
    report = (tmp_path / "dynamic_breadth_overlay_report.md").read_text(encoding="utf-8")

    assert [row["date"] for row in ledger_rows] == ["2026-07-22", "2026-07-23"]
    assert summary["valid_observation_count"] == 2
    assert summary["gate_evaluation_allowed"] is False
    assert summary["provisional_only"] is True
    assert summary["research_only"] is True
    assert summary["promotion_allowed"] is False
    assert summary["automatic_model_change"] is False
    assert summary["cumulative_excess_return"] < 0.0
    assert summary["background"]["common_background_days_before_start"] == 2
    assert "Dates before 2026-07-22 are background only" in report
    assert result["status"] == "collecting"


def test_ledger_append_is_idempotent_by_registration_and_date(tmp_path: Path) -> None:
    dates = pd.bdate_range("2026-07-21", periods=3)
    incumbent_path = tmp_path / "incumbent.csv"
    challenger_path = tmp_path / "challenger.csv"
    ledger_path = tmp_path / "dynamic_breadth_overlay_ledger.csv"
    _write_equity_curve(incumbent_path, dates, [0.01, 0.01])
    _write_equity_curve(challenger_path, dates, [0.02, 0.02])

    with ledger_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "registration_id": "another_registration",
                "date": "2026-07-22",
                "recorded_at": "2026-07-22T09:30:00",
                "source_latest_date": "2026-07-22",
                "observation_day_index": "1",
                "incumbent_id": "other",
                "challenger_id": "other",
                "incumbent_equity": "1.0",
                "challenger_equity": "1.0",
                "incumbent_daily_return": "0.0",
                "challenger_daily_return": "0.0",
                "excess_return": "0.0",
                "incumbent_observation_nav": "1.0",
                "challenger_observation_nav": "1.0",
                "incumbent_observation_drawdown": "0.0",
                "challenger_observation_drawdown": "0.0",
            }
        )

    _run(tmp_path, incumbent_path=incumbent_path, challenger_path=challenger_path, ledger_path=ledger_path)
    _run(tmp_path, incumbent_path=incumbent_path, challenger_path=challenger_path, ledger_path=ledger_path)

    ledger_rows = _read_csv_rows(ledger_path)
    keys = {(row["registration_id"], row["date"]) for row in ledger_rows}

    assert len(ledger_rows) == 3
    assert len(keys) == 3
    assert ("another_registration", "2026-07-22") in keys
    assert ("dynamic_breadth_overlay_20260721", "2026-07-22") in keys
    assert ("dynamic_breadth_overlay_20260721", "2026-07-23") in keys


def test_input_latest_dates_must_match(tmp_path: Path) -> None:
    incumbent_path = tmp_path / "incumbent.csv"
    challenger_path = tmp_path / "challenger.csv"
    _write_equity_curve(incumbent_path, pd.bdate_range("2026-07-21", periods=2), [0.01])
    _write_equity_curve(challenger_path, pd.bdate_range("2026-07-21", periods=3), [0.01, 0.01])

    with pytest.raises(ValueError, match="latest dates must match exactly"):
        _run(tmp_path, incumbent_path=incumbent_path, challenger_path=challenger_path)


def test_input_duplicate_dates_are_rejected(tmp_path: Path) -> None:
    incumbent_path = tmp_path / "incumbent.csv"
    challenger_path = tmp_path / "challenger.csv"
    pd.DataFrame(
        {
            "date": ["2026-07-21", "2026-07-22", "2026-07-22"],
            "equity": [100.0, 101.0, 102.0],
        }
    ).to_csv(incumbent_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "date": ["2026-07-21", "2026-07-22", "2026-07-23"],
            "equity": [100.0, 101.0, 102.0],
        }
    ).to_csv(challenger_path, index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="duplicate dates"):
        _run(tmp_path, incumbent_path=incumbent_path, challenger_path=challenger_path)


def test_maturity_gate_opens_at_sixty_valid_observation_days(tmp_path: Path) -> None:
    dates = pd.bdate_range("2026-07-21", periods=61)
    incumbent_path = tmp_path / "incumbent.csv"
    challenger_path = tmp_path / "challenger.csv"
    incumbent_returns = [0.0010] * 60
    challenger_returns = [0.0012] * 40 + [0.0008] * 20
    _write_equity_curve(incumbent_path, dates, incumbent_returns)
    _write_equity_curve(challenger_path, dates, challenger_returns)

    result = _run(tmp_path, incumbent_path=incumbent_path, challenger_path=challenger_path)
    summary = json.loads((tmp_path / "dynamic_breadth_overlay_summary.json").read_text(encoding="utf-8"))

    assert summary["valid_observation_count"] == 60
    assert summary["gate_evaluation_allowed"] is True
    assert summary["provisional_only"] is False
    assert summary["status"] == "manual_review_ready"
    assert summary["manual_review_gates"]["cumulative_excess_return_positive"] is True
    assert summary["manual_review_gates"]["positive_excess_day_ratio_met"] is True
    assert summary["manual_review_gates"]["challenger_drawdown_within_tolerance"] is True
    assert summary["manual_review_gates"]["all_gates_pass"] is True
    assert summary["promotion_allowed"] is False
    assert summary["automatic_model_change"] is False
    assert result["status"] == "manual_review_ready"
