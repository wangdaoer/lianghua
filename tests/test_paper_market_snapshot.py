from quant_etf_lab.market_data_source import MarketSnapshotLoadResult
from quant_etf_lab.paper_account import _market_snapshot_close_lookup


def test_market_snapshot_close_lookup_requires_exact_trade_date() -> None:
    snapshot = MarketSnapshotLoadResult(
        rows=[{"security_code": "000001", "close_price": 12.34}],
        source_kind="daily_hub",
        source_path=None,
        trade_date="2026-07-20",
        fetch_status=None,
    )

    assert _market_snapshot_close_lookup(snapshot, "2026-07-20")["000001"][:2] == (12.34, "2026-07-20")
    assert _market_snapshot_close_lookup(snapshot, "2026-07-19") == {}
