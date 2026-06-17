from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_etf_lab.cli import build_parser
from quant_etf_lab.market_timing_state import build_market_timing_state, run_market_timing_state


FAKE_MARKET_DATA_UTILS = """
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class FetchStatus:
    run_id: str | None
    source: str | None
    market_filter: str | None
    row_count: int
    status: str
    message: str | None
    run_time: str | None

def ensure_latest_fetch_ok(db_path=None, project_root=None):
    Path(project_root, "status_checked.txt").write_text("checked", encoding="utf-8")

def get_latest_fetch_status(db_path=None, project_root=None):
    return FetchStatus("run-1", "fake", "all", 3, "ok", None, "2026-06-15T15:30:00")

def get_latest_trade_date(db_path=None, project_root=None):
    return "2026-06-15"

def load_snapshot_rows(trade_date=None, market="all", db_path=None, project_root=None, fallback_to_csv=True):
    return []
"""


def _write_fake_ingest(root: Path) -> None:
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "market_data_utils.py").write_text(FAKE_MARKET_DATA_UTILS, encoding="utf-8")


def _write_history(path: Path, code: str, closes: list[float], amount: float) -> None:
    dates = pd.date_range("2026-04-01", periods=len(closes), freq="D")
    pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "code": code,
            "name": code,
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": [1_000_000.0] * len(closes),
            "amount": [amount] * len(closes),
        }
    ).to_csv(path, index=False)


def _write_daily_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"market": "sse", "trade_date": "2026-06-15", "security_code": "000001", "security_name": "StrongA"},
            {"market": "szse", "trade_date": "2026-06-15", "security_code": "300001", "security_name": "StrongB"},
            {"market": "sse", "trade_date": "2026-06-15", "security_code": "600001", "security_name": "WeakC"},
        ]
    ).to_csv(path, index=False)


def test_market_timing_state_cli_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["market-timing-state"])

    assert args.command == "market-timing-state"
    assert args.data_dir == "data/processed/stocks"
    assert args.output_dir == "outputs/research/market_timing_state_latest"
    assert args.data_cache_dir == "D:/codex/daily-market-data"
    assert args.ingest_project_dir == "D:/codex/2026-06-15-exchange-data-ingest"
    assert args.lookback_days == 90
    assert args.max_symbols is None
    assert args.require_success is True


def test_build_market_timing_state_keeps_unknown_unavailable_dimensions() -> None:
    latest_rows = pd.DataFrame(
        [
            {
                "date": "2026-06-15",
                "code": "000001",
                "name": "A",
                "close": 13.0,
                "amount": 300.0,
                "daily_return": 0.02,
                "above_ma20": True,
                "above_ma60": True,
            },
            {
                "date": "2026-06-15",
                "code": "300001",
                "name": "B",
                "close": 22.0,
                "amount": 200.0,
                "daily_return": 0.01,
                "above_ma20": True,
                "above_ma60": True,
            },
            {
                "date": "2026-06-15",
                "code": "600001",
                "name": "C",
                "close": 8.0,
                "amount": 100.0,
                "daily_return": -0.01,
                "above_ma20": False,
                "above_ma60": False,
            },
        ]
    )

    state, components = build_market_timing_state(latest_rows)

    assert state["status"] == "ok"
    assert state["market_regime"] == "risk_on"
    assert state["position_reference"] == 0.65
    assert state["above_ma20_ratio"] == 2 / 3
    assert state["up_turnover_share"] == 500 / 600
    assert state["valuation_status"] == "unknown"
    assert state["funding_status"] == "unknown"
    assert state["sentiment_status"] == "unknown"
    assert state["fundamentals_status"] == "unknown"
    assert set(components["dimension"]) >= {"technical", "breadth", "valuation", "funding", "sentiment", "fundamentals"}
    assert components.loc[components["dimension"] == "valuation", "status"].iloc[0] == "unknown"


def test_build_market_timing_state_reports_component_statuses_independently() -> None:
    latest_rows = pd.DataFrame(
        [
            {
                "date": "2026-06-15",
                "code": "000001",
                "name": "A",
                "close": 13.0,
                "amount": 300.0,
                "daily_return": 0.02,
                "above_ma20": False,
                "above_ma60": False,
            },
            {
                "date": "2026-06-15",
                "code": "300001",
                "name": "B",
                "close": 22.0,
                "amount": 200.0,
                "daily_return": 0.01,
                "above_ma20": False,
                "above_ma60": False,
            },
            {
                "date": "2026-06-15",
                "code": "600001",
                "name": "C",
                "close": 8.0,
                "amount": 100.0,
                "daily_return": -0.01,
                "above_ma20": True,
                "above_ma60": True,
            },
        ]
    )

    state, components = build_market_timing_state(latest_rows)

    by_dimension = components.set_index("dimension")
    assert state["market_regime"] == "risk_off"
    assert by_dimension.loc["technical", "status"] == "risk_off"
    assert by_dimension.loc["breadth", "status"] == "risk_on"
    assert by_dimension.loc["turnover_structure", "status"] == "risk_on"


def test_run_market_timing_state_writes_research_outputs_from_local_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "stocks"
    data_dir.mkdir()
    _write_history(data_dir / "000001.csv", "000001", [10 + idx * 0.05 for idx in range(80)], 300_000_000)
    _write_history(data_dir / "300001.csv", "300001", [20 + idx * 0.08 for idx in range(80)], 200_000_000)
    _write_history(data_dir / "600001.csv", "600001", [14 - idx * 0.03 for idx in range(80)], 100_000_000)
    daily = tmp_path / "daily-market-data"
    ingest = tmp_path / "exchange-ingest"
    _write_fake_ingest(ingest)
    _write_daily_snapshot(daily / "snapshots" / "2026-06-15_market_snapshot.csv")

    result = run_market_timing_state(
        project_root=tmp_path,
        data_dir=data_dir,
        output_dir=tmp_path / "market_timing_state",
        daily_data_dir=daily,
        ingest_project_dir=ingest,
        trade_date="2026-06-15",
    )

    assert result.snapshot["status"] == "ok"
    assert result.snapshot["research_only"] is True
    assert result.snapshot["broker_action"] == "none"
    assert result.snapshot["allocator_action"] == "none"
    assert result.snapshot["market_source_kind"] == "daily_market_data_csv"
    assert result.snapshot["fetch_status"] == "ok"
    assert result.snapshot["market_regime"] == "risk_on"
    assert result.snapshot["position_reference"] == 0.65
    assert result.state_path.exists()
    assert result.components_path.exists()
    assert result.report_path.exists()
    assert (ingest / "status_checked.txt").exists()
    payload = json.loads(result.snapshot_path.read_text(encoding="utf-8"))
    assert payload["promotion_note"].startswith("Observation-only")
