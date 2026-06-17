from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "refresh_after_close_from_sina.py"


def load_refresh_module():
    spec = importlib.util.spec_from_file_location("refresh_after_close_from_sina", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_sina_response_extracts_daily_quote() -> None:
    module = load_refresh_module()
    text = (
        'var hq_str_sz000001="平安银行,11.210,11.240,11.060,11.210,10.980,'
        '11.050,11.060,154130495,1711561286.570,327000,11.050,'
        '135500,11.040,271900,11.030,200200,11.020,323100,11.010,'
        '32399,11.060,116000,11.070,470100,11.080,171300,11.090,'
        '311100,11.100,2026-06-15,15:00:00,00";'
    )

    quotes = module.parse_sina_response(text)

    quote = quotes["000001"]
    assert quote.code == "000001"
    assert quote.name == "平安银行"
    assert quote.date == "2026-06-15"
    assert quote.open == 11.21
    assert quote.high == 11.21
    assert quote.low == 10.98
    assert quote.close == 11.06
    assert quote.volume == 154130495.0
    assert quote.amount == 1711561286.57


def test_append_quote_to_csv_is_idempotent(tmp_path: Path) -> None:
    module = load_refresh_module()
    quote = module.SinaQuote(
        code="000001",
        name="平安银行",
        date="2026-06-15",
        time="15:00:00",
        open=11.21,
        high=11.21,
        low=10.98,
        close=11.06,
        volume=154130495.0,
        amount=1711561286.57,
        symbol="sz000001",
    )
    path = tmp_path / "000001.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-12",
                "code": "000001",
                "name": "平安银行",
                "open": 11.0,
                "high": 11.25,
                "low": 10.88,
                "close": 11.24,
                "volume": 203235546.0,
                "amount": 2263042931.0,
            }
        ]
    ).to_csv(path, index=False, encoding="utf-8")

    first = module.append_quote_to_csv(path, quote, target_date="2026-06-15")
    second = module.append_quote_to_csv(path, quote, target_date="2026-06-15")

    frame = pd.read_csv(path)
    assert first["status"] == "appended"
    assert second["status"] == "already_present"
    assert len(frame) == 2
    assert frame.iloc[-1]["date"] == "2026-06-15"
    assert frame.iloc[-1]["close"] == 11.06
