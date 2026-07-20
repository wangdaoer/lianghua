from __future__ import annotations

import json
from pathlib import Path

from quant_etf_lab.uzi_auxiliary import (
    build_uzi_auxiliary_snapshot,
    extract_uzi_auxiliary_signal,
    normalize_uzi_ticker,
    run_uzi_auxiliary_export,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_sample_cache(root: Path, ticker: str = "300870.SZ") -> None:
    cache = root / ticker
    _write_json(
        cache / "dimensions.json",
        {
            "ticker": "300870",
            "fundamental_score": 62.3,
            "dimensions": {
                "1_financials": {"label": "ROE 9.9% 路 revenue +17.5%"},
                "2_kline": {"label": "Stage 2 uptrend"},
                "10_valuation": {"label": "PE 318.95 路 hot"},
            },
        },
    )
    _write_json(
        cache / "panel.json",
        {
            "ticker": "300870",
            "panel_consensus": 31.2,
            "signal_distribution": {"bullish": 9, "neutral": 11, "bearish": 23, "skip": 23},
            "school_scores": {
                "D": {"consensus": 100.0},
                "G": {"consensus": 44.8},
                "I": {"consensus": 3.6},
            },
        },
    )
    _write_json(
        cache / "raw_data.json",
        {
            "dimensions": {
                "10_valuation": {
                    "data": {
                        "dcf_simple": {"intrinsic_value_total": 4_306_146_505.42},
                        "dcf_sensitivity": {"current_price": 380.06, "values": [[31.68, 35.79], [22.39, 25.14]]},
                    }
                },
                "20_valuation_models": {
                    "data": {
                        "summary": {
                            "dcf_intrinsic": 44.71,
                            "dcf_safety_margin_pct": -88.2,
                            "dcf_verdict": "overvalued",
                        }
                    }
                },
            }
        },
    )
    _write_json(cache / "_data_gaps.json", {"coverage_pct": 83.0, "critical_missing": True, "tasks": []})


def test_normalize_uzi_ticker_infers_a_share_market() -> None:
    assert normalize_uzi_ticker("301165") == "301165.SZ"
    assert normalize_uzi_ticker("688629") == "688629.SH"
    assert normalize_uzi_ticker("300870.SZ") == "300870.SZ"


def test_extract_uzi_auxiliary_signal_keeps_research_only_boundaries(tmp_path: Path) -> None:
    _write_sample_cache(tmp_path)

    signal = extract_uzi_auxiliary_signal(tmp_path, "300870")

    assert signal["ticker"] == "300870.SZ"
    assert signal["fundamental_score"] == 62.3
    assert signal["bearish_ratio"] == 0.5349
    assert signal["technical_consensus"] == 100.0
    assert signal["bottleneck_consensus"] == 3.6
    assert signal["valuation_overheat"] is True
    assert signal["integration_recommendation"] == "data_gap_review"
    assert signal["position_effect"] == "none"
    assert signal["broker_action"] == "none"


def test_build_uzi_auxiliary_snapshot_blocks_direct_trading_use(tmp_path: Path) -> None:
    _write_sample_cache(tmp_path)

    snapshot = build_uzi_auxiliary_snapshot(cache_root=tmp_path, tickers=["300870"], generated_at="2026-06-20T12:00:00")

    assert snapshot["integration_status"] == "research_auxiliary_only"
    assert snapshot["position_effect"] == "none"
    assert snapshot["broker_action"] == "none"
    assert "paper_account_target_weights" in snapshot["prohibited_integrations"]
    assert "live_preflight_orders" in snapshot["prohibited_integrations"]
    assert snapshot["promotion_gate"]["requires_oos_ablation"] is True


def test_run_uzi_auxiliary_export_writes_snapshot_csv_and_report(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    output_dir = tmp_path / "out"
    _write_sample_cache(cache_root)

    result = run_uzi_auxiliary_export(output_dir=output_dir, cache_root=cache_root, tickers=["300870"])

    assert result.snapshot_path.exists()
    assert result.signals_path.exists()
    assert result.report_path.exists()
    payload = json.loads(result.snapshot_path.read_text(encoding="utf-8-sig"))
    assert payload["signals"][0]["ticker"] == "300870.SZ"
    assert "position_effect" in result.signals_path.read_text(encoding="utf-8-sig")
    assert "live_preflight_orders" in result.report_path.read_text(encoding="utf-8-sig")
