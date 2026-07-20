from __future__ import annotations

import json
from pathlib import Path

from quant_etf_lab.uzi_ablation import build_uzi_sidecar_ablation_snapshot, run_uzi_sidecar_ablation


def _write_uzi_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": "2026-06-20T12:00:00",
        "signals": [
            {
                "ticker": "300870.SZ",
                "code": "300870",
                "integration_recommendation": "risk_dampen_candidate",
                "valuation_overheat": True,
                "bearish_ratio": 0.60,
                "bottleneck_consensus": 3.6,
                "current_price": 10.0,
                "position_effect": "none",
                "broker_action": "none",
            }
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_price_cache(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "date,code,name,open,high,low,close,volume,amount",
                "2026-06-18,300870,x,10,10,9.8,10,1,1",
                "2026-06-19,300870,x,9.9,10.1,9.4,9.5,1,1",
                "2026-06-22,300870,x,9.5,9.8,9.0,9.2,1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_stale_duplicate_price_cache(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "date,code,name,open,high,low,close,volume,amount",
                "2026-06-18,300870,x,10,10.5,9.8,10,100,1000",
                "2026-06-19,300870,x,10,10.5,9.8,10,100,1000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_uzi_sidecar_ablation_uses_price_cache_when_outcome_missing(tmp_path: Path) -> None:
    uzi_path = tmp_path / "uzi.json"
    price_dir = tmp_path / "prices"
    _write_uzi_snapshot(uzi_path)
    _write_price_cache(price_dir / "300870.csv")

    snapshot = build_uzi_sidecar_ablation_snapshot(
        uzi_snapshot_path=uzi_path,
        stock_price_dir=price_dir,
        outcome_history_path=tmp_path / "missing.csv",
        horizons=[1, 2],
        generated_at="2026-06-20T12:30:00",
    )

    row = snapshot["rows"][0]
    assert row["risk_flag"] is True
    assert row["price_cache_status"] == "ok"
    assert row["entry_date"] == "2026-06-18"
    assert row["return_1d"] == -0.05
    assert row["max_drawdown_1d"] == -0.06
    assert row["return_2d"] == -0.08
    assert row["max_drawdown_2d"] == -0.1
    assert row["ablation_status"] == "loss_reduction_candidate"
    assert snapshot["position_effect"] == "none"
    assert snapshot["broker_action"] == "none"


def test_uzi_sidecar_ablation_ignores_stale_duplicate_forward_rows(tmp_path: Path) -> None:
    uzi_path = tmp_path / "uzi.json"
    price_dir = tmp_path / "prices"
    _write_uzi_snapshot(uzi_path)
    _write_stale_duplicate_price_cache(price_dir / "300870.csv")

    snapshot = build_uzi_sidecar_ablation_snapshot(
        uzi_snapshot_path=uzi_path,
        stock_price_dir=price_dir,
        outcome_history_path=tmp_path / "missing.csv",
        horizons=[1],
        generated_at="2026-06-20T12:30:00",
    )

    row = snapshot["rows"][0]
    assert row["price_cache_status"] == "ok"
    assert row["entry_date"] == "2026-06-18"
    assert row["latest_available_date"] == "2026-06-18"
    assert row["available_forward_days"] == 0
    assert row["return_1d"] is None
    assert row["status_1d"] == "pending"
    assert row["ablation_status"] == "pending_future_data"


def test_uzi_sidecar_ablation_prefers_existing_outcome_history(tmp_path: Path) -> None:
    uzi_path = tmp_path / "uzi.json"
    price_dir = tmp_path / "prices"
    outcome_path = tmp_path / "outcomes.csv"
    _write_uzi_snapshot(uzi_path)
    _write_price_cache(price_dir / "300870.csv")
    outcome_path.write_text(
        "\n".join(
            [
                "date,code,name,entry_date,entry_close,return_1d,outcome_status_1d,future_date_1d",
                "2026-06-18,300870,x,2026-06-18,10,0.12,available,2026-06-19",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    snapshot = build_uzi_sidecar_ablation_snapshot(
        uzi_snapshot_path=uzi_path,
        stock_price_dir=price_dir,
        outcome_history_path=outcome_path,
        horizons=[1],
    )

    row = snapshot["rows"][0]
    assert row["price_cache_status"] == "outcome_history"
    assert row["return_1d"] == 0.12
    assert row["status_1d"] == "available"
    assert row["ablation_status"] == "opportunity_cost_candidate"
    assert snapshot["summary"]["missing_price_cache_count"] == 0


def test_run_uzi_sidecar_ablation_writes_outputs(tmp_path: Path) -> None:
    uzi_path = tmp_path / "uzi.json"
    price_dir = tmp_path / "prices"
    output_dir = tmp_path / "out"
    _write_uzi_snapshot(uzi_path)
    _write_price_cache(price_dir / "300870.csv")

    result = run_uzi_sidecar_ablation(
        output_dir=output_dir,
        uzi_snapshot_path=uzi_path,
        stock_price_dir=price_dir,
        outcome_history_path=tmp_path / "missing.csv",
        horizons=[1],
    )

    assert result.snapshot_path.exists()
    assert result.summary_path.exists()
    assert result.report_path.exists()
    payload = json.loads(result.snapshot_path.read_text(encoding="utf-8-sig"))
    assert payload["rows"][0]["ticker"] == "300870.SZ"
    assert "position_effect" in result.summary_path.read_text(encoding="utf-8-sig")
    assert "broker_order_generation" in result.report_path.read_text(encoding="utf-8-sig")
