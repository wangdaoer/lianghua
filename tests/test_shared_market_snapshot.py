from pathlib import Path

import quant_etf_lab.dashboard as dashboard
import quant_etf_lab.momentum_focus as momentum_focus
from quant_etf_lab.market_data_source import MarketSnapshotLoadResult


def _snapshot() -> MarketSnapshotLoadResult:
    return MarketSnapshotLoadResult(
        rows=[
            {
                "market": "szse",
                "trade_date": "2026-07-20",
                "security_code": "300001",
                "security_name": "共享行情",
                "change_ratio": 10.0,
                "close_price": 12.0,
                "turnover": 200_000_000,
                "volume": 10_000_000,
            }
        ],
        source_kind="test_shared_snapshot",
        source_path=Path("shared.csv"),
        trade_date="2026-07-20",
        fetch_status=None,
    )


def test_dashboard_uses_shared_market_snapshot_without_reloading(tmp_path: Path, monkeypatch) -> None:
    daily_dir = tmp_path / "daily-market-data"
    (daily_dir / "snapshots").mkdir(parents=True)
    monkeypatch.setattr(
        dashboard,
        "load_market_snapshot_rows",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected market reload")),
    )

    generated = iter(["2026-07-20T16:00:00", "2026-07-20T16:01:00"])
    monkeypatch.setattr(
        dashboard,
        "_summarize_daily_check",
        lambda *args, **kwargs: {"daily_check_generated_at": next(generated)},
    )
    result = dashboard.run_daily_dashboard(
        project_root=tmp_path,
        output_dir=tmp_path / "dashboard",
        data_cache_dir=daily_dir,
        as_of_date="2026-07-20",
        market_snapshot=_snapshot(),
    )
    first_mtimes = {
        result.snapshot_path: result.snapshot_path.stat().st_mtime_ns,
        result.report_path: result.report_path.stat().st_mtime_ns,
    }
    dashboard.run_daily_dashboard(
        project_root=tmp_path,
        output_dir=tmp_path / "dashboard",
        data_cache_dir=daily_dir,
        as_of_date="2026-07-20",
        market_snapshot=_snapshot(),
    )

    assert result.snapshot["market_cache_source_kind"] == "test_shared_snapshot"
    assert result.snapshot["market_cache_snapshot_row_count"] == 1
    assert {path: path.stat().st_mtime_ns for path in first_mtimes} == first_mtimes


def test_momentum_focus_uses_shared_market_snapshot_and_skips_unchanged_writes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        momentum_focus,
        "load_market_snapshot_rows",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("unexpected market reload")),
    )
    output = tmp_path / "momentum"
    first = momentum_focus.run_momentum_focus(
        project_root=tmp_path,
        output_dir=output,
        as_of_date="2026-07-20",
        name_map_path=None,
        market_snapshot=_snapshot(),
    )
    mtimes = {
        path: path.stat().st_mtime_ns
        for path in (first.candidates_path, first.snapshot_path, first.report_path)
    }

    second = momentum_focus.run_momentum_focus(
        project_root=tmp_path,
        output_dir=output,
        as_of_date="2026-07-20",
        name_map_path=None,
        market_snapshot=_snapshot(),
    )

    assert len(second.candidates) == 1
    assert {path: path.stat().st_mtime_ns for path in mtimes} == mtimes
