from pathlib import Path

import pandas as pd

from panel_io import iter_panel, panel_columns, read_panel, write_panel_atomic


def _sample_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-07-17", "2026-07-20"],
            "symbol": ["000001", "600000"],
            "open": [10.0, 20.0],
            "high": [10.5, 20.5],
            "low": [9.8, 19.8],
            "close": [10.2, 20.2],
            "volume": [100.0, 200.0],
            "amount": [1020.0, 4040.0],
        }
    )


def test_csv_and_parquet_read_contracts_match(tmp_path: Path) -> None:
    panel = _sample_panel()
    csv_path = tmp_path / "panel.csv"
    parquet_path = tmp_path / "panel.parquet"
    panel.to_csv(csv_path, index=False)
    write_panel_atomic(panel, parquet_path)

    csv = read_panel(csv_path, dtype={"symbol": str}, parse_dates=["date"])
    parquet = read_panel(parquet_path, dtype={"symbol": str}, parse_dates=["date"])

    pd.testing.assert_frame_equal(csv, parquet, check_dtype=False)
    assert panel_columns(csv_path) == panel_columns(parquet_path)


def test_parquet_iteration_respects_columns_and_batch_size(tmp_path: Path) -> None:
    path = tmp_path / "panel.parquet"
    write_panel_atomic(_sample_panel(), path)

    chunks = list(
        iter_panel(
            path,
            columns=["date", "symbol", "close"],
            dtype={"symbol": str},
            chunksize=1,
        )
    )

    assert len(chunks) == 2
    assert chunks[0].columns.tolist() == ["date", "symbol", "close"]
    assert pd.concat(chunks, ignore_index=True)["symbol"].tolist() == ["000001", "600000"]


def test_csv_and_parquet_filters_match(tmp_path: Path) -> None:
    panel = _sample_panel()
    csv_path = tmp_path / "panel.csv"
    parquet_path = tmp_path / "panel.parquet"
    panel.to_csv(csv_path, index=False)
    write_panel_atomic(panel, parquet_path)

    expected = panel.iloc[[1]].reset_index(drop=True)
    for path in (csv_path, parquet_path):
        actual = read_panel(
            path,
            dtype={"symbol": str},
            filters=[("date", ">=", "2026-07-20")],
        )
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected)
