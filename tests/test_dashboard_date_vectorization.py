import pandas as pd

from quant_etf_lab.dashboard import _parse_date_series


def test_parse_date_series_supports_compact_iso_and_invalid_values() -> None:
    parsed = _parse_date_series(pd.Series(["20260720", "2026-07-19", "", None, "bad"]))

    assert parsed.iloc[0] == pd.Timestamp("2026-07-20")
    assert parsed.iloc[1] == pd.Timestamp("2026-07-19")
    assert parsed.iloc[2:].isna().all()
