from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_etf_lab.ths_export_source import normalize_ths_export


def test_normalize_ths_export_maps_total_hands_and_total_amount(tmp_path: Path) -> None:
    raw = tmp_path / "ths_export.csv"
    export_root = tmp_path / "ths_exports"
    pd.DataFrame(
        [
            {
                "\u4ee3\u7801": "SZ300534",
                "\u540d\u79f0": "sample",
                "\u5f00\u76d8": "9.80",
                "\u6700\u9ad8": "12.04",
                "\u6700\u4f4e": "9.53",
                "\u73b0\u4ef7": "12.04",
                "\u6628\u6536": "10.03",
                "\u6da8\u5e45": "20.04",
                "\u603b\u624b": "112954175",
                "\u603b\u91d1\u989d": "1246240700",
                "\u6362\u624b": "37.40",
            }
        ]
    ).to_csv(raw, index=False, encoding="utf-8-sig")

    result = normalize_ths_export(
        raw,
        trade_date="2026-07-20",
        export_root=export_root,
        min_row_count=1,
    )

    frame = pd.read_csv(result.normalized_path, dtype={"security_code": str})
    assert frame.loc[0, "volume"] == 112_954_175
    assert frame.loc[0, "turnover"] == 1_246_240_700
    assert frame.loc[0, "amount"] == 1_246_240_700

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["mapped_columns"]["volume"] == "\u603b\u624b"
    assert manifest["mapped_columns"]["turnover"] == "\u603b\u91d1\u989d"
    assert manifest["field_coverage"]["volume"]["positive_ratio"] == 1.0
    assert manifest["field_coverage"]["turnover"]["positive_ratio"] == 1.0


def test_large_order_net_amount_is_not_treated_as_turnover(tmp_path: Path) -> None:
    raw = tmp_path / "ths_export.csv"
    export_root = tmp_path / "ths_exports"
    pd.DataFrame(
        [
            {
                "\u4ee3\u7801": "SH600000",
                "\u540d\u79f0": "sample",
                "\u73b0\u4ef7": "10.20",
                "\u6628\u6536": "10.00",
                "\u5927\u5355\u51c0\u989d": "12340000",
            }
        ]
    ).to_csv(raw, index=False, encoding="utf-8-sig")

    result = normalize_ths_export(
        raw,
        trade_date="2026-07-17",
        export_root=export_root,
        min_row_count=1,
    )

    frame = pd.read_csv(result.normalized_path, dtype={"security_code": str})
    assert pd.isna(frame.loc[0, "turnover"])
    assert pd.isna(frame.loc[0, "amount"])

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["mapped_columns"]["turnover"] is None
