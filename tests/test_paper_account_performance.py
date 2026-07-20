from pathlib import Path

import pandas as pd

import quant_etf_lab.paper_account as paper_account


def test_review_date_text_fast_path_preserves_supported_formats() -> None:
    assert paper_account._review_date_text("2026-07-20") == "2026-07-20"
    assert paper_account._review_date_text("2026-07-20 16:10:00") == "2026-07-20"
    assert paper_account._review_date_text("20260720") == "2026-07-20"


def test_outcome_history_loads_each_price_source_once_per_sync(tmp_path: Path, monkeypatch) -> None:
    price_path = tmp_path / "000001.csv"
    pd.DataFrame(
        [
            {"date": "2026-07-01", "close": 10.0, "high": 10.2, "low": 9.8},
            {"date": "2026-07-02", "close": 10.5, "high": 10.7, "low": 10.1},
            {"date": "2026-07-03", "close": 11.0, "high": 11.2, "low": 10.6},
        ]
    ).to_csv(price_path, index=False)
    history_path = tmp_path / "history.csv"
    rows = []
    for review_date in ("2026-06-30", "2026-07-01"):
        rows.append(
            {
                "date": review_date,
                "layer": "core",
                "code": "000001",
                "price_source": str(price_path),
                "outcome_status": "pending",
                "outcome_status_1d": "pending",
                "outcome_status_5d": "pending",
                "outcome_status_10d": "pending",
                "outcome_status_20d": "pending",
            }
        )
    pd.DataFrame(rows).to_csv(history_path, index=False)

    original = paper_account._load_stock_review_outcome_price_source
    calls: list[Path] = []

    def counted(path: Path) -> pd.DataFrame:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(paper_account, "_load_stock_review_outcome_price_source", counted)
    paper_account.sync_stock_target_review_outcomes_history(
        pd.DataFrame(),
        history_path,
        tmp_path / "history_snapshot.csv",
    )

    assert calls == [price_path]

    calls.clear()
    _, payload = paper_account.sync_stock_target_review_outcomes_history(
        pd.DataFrame(),
        history_path,
        tmp_path / "history_snapshot.csv",
    )

    assert calls == []
    assert payload["history_price_source_count"] == 0

    with price_path.open("a", encoding="utf-8") as handle:
        handle.write("2026-07-04,11.5,11.7,11.1\n")
    calls.clear()
    _, payload = paper_account.sync_stock_target_review_outcomes_history(
        pd.DataFrame(),
        history_path,
        tmp_path / "history_snapshot.csv",
    )

    assert calls == [price_path.resolve()]
    assert payload["history_price_source_count"] == 1
