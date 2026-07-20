from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from quant_etf_lab.market_data_source import MarketSnapshotLoadResult
from quant_etf_lab.momentum_focus import build_momentum_focus_candidates, run_momentum_focus


def test_momentum_focus_reuses_candidates_when_inputs_are_unchanged() -> None:
    snapshot = MarketSnapshotLoadResult(
        rows=[
            {
                "security_code": "000001",
                "security_name": "平安银行",
                "market": "szse",
                "trade_date": "2026-07-20",
                "close_price": 12.0,
                "change_pct": 6.0,
                "turnover": 1_000_000,
                "volume": 100_000,
            }
        ],
        source_kind="test",
        source_path=None,
        trade_date="2026-07-20",
        fetch_status=None,
    )

    with TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "momentum"
        with patch(
            "quant_etf_lab.momentum_focus.build_momentum_focus_candidates",
            wraps=build_momentum_focus_candidates,
        ) as builder:
            first = run_momentum_focus(output_dir=output, as_of_date="2026-07-20", market_snapshot=snapshot)
            second = run_momentum_focus(output_dir=output, as_of_date="2026-07-20", market_snapshot=snapshot)

        assert builder.call_count == 1
        assert second.snapshot["cache_hit"] is True
        assert first.candidates.equals(second.candidates)
