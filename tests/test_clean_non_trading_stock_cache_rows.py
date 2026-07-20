from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "clean_non_trading_stock_cache_rows.py"


def load_clean_module():
    spec = importlib.util.spec_from_file_location("clean_non_trading_stock_cache_rows", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_cache(path: Path, dates: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "date": date,
            "code": path.stem,
            "name": "Sample",
            "open": 10.0,
            "high": 10.5,
            "low": 9.8,
            "close": 10.2,
            "volume": 1000.0,
            "amount": 10200.0,
        }
        for date in dates
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def _write_meta(path: Path, code: str, end_date: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "code": code,
                "name": "Sample",
                "source": "akshare+daily_market_snapshot",
                "end_date": end_date,
                "latest_snapshot_source": "daily_market_snapshot",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_clean_non_trading_rows_dry_run_reports_without_writing(tmp_path: Path) -> None:
    module = load_clean_module()
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    for root in [data_dir / "processed" / "stocks", data_dir / "raw" / "stocks"]:
        _write_cache(root / "000001.csv", ["2026-06-18", "2026-06-19"])
    _write_meta(data_dir / "meta" / "stocks" / "000001.json", "000001", "2026-06-19")

    summary = module.clean_cache(
        data_dir=data_dir,
        output_dir=output_dir,
        target_date="2026-06-19",
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["removed_rows"] == 2
    assert summary["changed_csv_files"] == 2
    assert summary["updated_meta_files"] == 1
    assert not (output_dir / "backups").exists()
    for cache_path in [
        data_dir / "processed" / "stocks" / "000001.csv",
        data_dir / "raw" / "stocks" / "000001.csv",
    ]:
        frame = pd.read_csv(cache_path, dtype={"date": str})
        assert list(frame["date"]) == ["2026-06-18", "2026-06-19"]
    meta = json.loads((data_dir / "meta" / "stocks" / "000001.json").read_text(encoding="utf-8-sig"))
    assert meta["end_date"] == "2026-06-19"


def test_clean_non_trading_rows_removes_rows_backs_up_and_updates_meta(tmp_path: Path) -> None:
    module = load_clean_module()
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    for root in [data_dir / "processed" / "stocks", data_dir / "raw" / "stocks"]:
        _write_cache(root / "000001.csv", ["2026-06-18", "2026-06-19"])
        _write_cache(root / "000002.csv", ["2026-06-18"])
    _write_meta(data_dir / "meta" / "stocks" / "000001.json", "000001", "2026-06-19")
    _write_meta(data_dir / "meta" / "stocks" / "000002.json", "000002", "2026-06-18")

    summary = module.clean_cache(
        data_dir=data_dir,
        output_dir=output_dir,
        target_date="2026-06-19",
        dry_run=False,
    )

    assert summary["dry_run"] is False
    assert summary["removed_rows"] == 2
    assert summary["changed_csv_files"] == 2
    assert summary["updated_meta_files"] == 1
    assert summary["unchanged_csv_files"] == 2
    assert (output_dir / "non_trading_row_cleanup_audit.csv").exists()
    assert (output_dir / "non_trading_row_cleanup_summary.json").exists()
    assert (output_dir / "backups" / "processed" / "stocks" / "000001.csv").exists()
    assert (output_dir / "backups" / "raw" / "stocks" / "000001.csv").exists()
    assert (output_dir / "backups" / "meta" / "stocks" / "000001.json").exists()

    for cache_path in [
        data_dir / "processed" / "stocks" / "000001.csv",
        data_dir / "raw" / "stocks" / "000001.csv",
    ]:
        frame = pd.read_csv(cache_path, dtype={"date": str})
        assert list(frame["date"]) == ["2026-06-18"]

    meta = json.loads((data_dir / "meta" / "stocks" / "000001.json").read_text(encoding="utf-8-sig"))
    assert meta["end_date"] == "2026-06-18"
    assert meta["non_trading_cleanup_date"] == "2026-06-19"


def test_script_can_be_invoked_directly_for_help() -> None:
    result = subprocess.run(
        ["python", str(SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Remove non-trading date rows" in result.stdout
