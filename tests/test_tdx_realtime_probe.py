import csv

from tdx_realtime_probe import (
    extract_probe_json,
    is_china_intraday_window,
    load_watchlist_symbols,
    normalize_symbol,
    resolve_probe_codes,
    summarize_probe,
)


def test_normalize_symbol_accepts_common_market_formats():
    assert normalize_symbol("sz000001") == "000001"
    assert normalize_symbol("600519.SH") == "600519"
    assert normalize_symbol("  2472 ") == "002472"
    assert normalize_symbol("") is None


def test_load_watchlist_symbols_prefers_chinese_code_header(tmp_path):
    path = tmp_path / "watchlist.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["股票代码", "股票名称"])
        writer.writeheader()
        writer.writerows(
            [
                {"股票代码": "605389", "股票名称": "长龄液压"},
                {"股票代码": "301397", "股票名称": "溯联股份"},
                {"股票代码": "605389", "股票名称": "重复"},
            ]
        )

    assert load_watchlist_symbols(path, top_n=5) == ["605389", "301397"]


def test_resolve_probe_codes_adds_benchmark_without_duplicates(tmp_path):
    path = tmp_path / "watchlist.csv"
    path.write_text("股票代码\n605389\n510300\n", encoding="utf-8-sig")

    codes, used = resolve_probe_codes(
        explicit_codes=None,
        watchlist=path,
        top_n=10,
        extra_codes=["510300", "000001"],
    )

    assert used == path
    assert codes == ["605389", "510300", "000001"]


def test_extract_probe_json_ignores_colored_log_noise():
    raw = "\x1b[36mconnected\x1b[0m\n{\"ok\":true,\"quotes\":[],\"elapsed_ms\":12}\n"

    assert extract_probe_json(raw) == {"ok": True, "quotes": [], "elapsed_ms": 12}


def test_summarize_probe_marks_weekend_snapshot_pending():
    sample_rows = [
        {
            "sample_id": 1,
            "ok": True,
            "elapsed_ms": 480,
            "quote_count": 1,
            "error": "",
        }
    ]
    quote_rows = [
        {
            "sample_id": 1,
            "code": "605389",
            "latest": 78.45,
            "server_time": "15297156",
            "buy1_price": 78.45,
            "sell1_price": 78.47,
        }
    ]

    summary = summarize_probe(
        asof_date="2026-07-18",
        codes=["605389", "510300"],
        sample_rows=sample_rows,
        quote_rows=quote_rows,
        watchlist=None,
        generated_at="2026-07-18T16:30:00",
    )

    assert summary["success_rate"] == 1.0
    assert summary["missing_codes"] == ["510300"]
    if not is_china_intraday_window():
        assert summary["assessment"] == "snapshot_ok_intraday_pending"
