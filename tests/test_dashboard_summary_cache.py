from pathlib import Path

import quant_etf_lab.dashboard as dashboard


def test_dashboard_reuses_static_summaries_but_refreshes_pipeline_history(tmp_path: Path, monkeypatch) -> None:
    calls = {"market": 0, "history": 0}

    def market_summary(*args, **kwargs):
        calls["market"] += 1
        return {
            "market_cache_status": "fresh_enough",
            "market_cache_latest_date": "2026-07-20",
            "market_cache_source_kind": "test",
        }

    def history_summary(*args, **kwargs):
        calls["history"] += 1
        return {
            "pipeline_history_status": "ok",
            "pipeline_history_health_state": "ok",
            "pipeline_history_alert_count": calls["history"],
        }

    monkeypatch.setattr(dashboard, "_summarize_market_cache", market_summary)
    monkeypatch.setattr(dashboard, "_summarize_pipeline_history", history_summary)
    cache: dict[str, dict] = {}

    first = dashboard.run_daily_dashboard(
        project_root=tmp_path,
        output_dir=tmp_path / "dashboard",
        as_of_date="2026-07-20",
        summary_cache=cache,
    )
    second = dashboard.run_daily_dashboard(
        project_root=tmp_path,
        output_dir=tmp_path / "dashboard",
        as_of_date="2026-07-20",
        summary_cache=cache,
    )

    assert calls == {"market": 1, "history": 2}
    assert first.snapshot["pipeline_history_alert_count"] == 1
    assert second.snapshot["pipeline_history_alert_count"] == 2
