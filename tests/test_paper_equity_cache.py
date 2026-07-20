import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from quant_etf_lab.paper_account import _window_equity_fingerprint


def test_window_equity_fingerprint_changes_with_source_curve_content() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        allocator = root / "allocator"
        allocator.mkdir()
        config = root / "config.yaml"
        summary = allocator / "portfolio_walk_forward_summary.csv"
        params = root / "params.json"
        core = root / "core.csv"
        satellite = root / "satellite.csv"
        benchmark = root / "benchmark.csv"
        source = root / "selected_source.csv"
        config.write_text("name: test", encoding="utf-8")
        summary.write_text("window\n1", encoding="utf-8")
        params.write_text(json.dumps({"source_path": str(source)}), encoding="utf-8")
        for path in (core, satellite, benchmark, source):
            path.write_text("date,equity\n2026-07-20,1", encoding="utf-8")

        base = SimpleNamespace(
            project_root=root,
            core=SimpleNamespace(path=core),
            satellite=SimpleNamespace(path=satellite),
            benchmark_path=benchmark,
        )
        windows = pd.DataFrame({"selected_params_resolved": [str(params)]})
        with patch("quant_etf_lab.paper_account._load_allocator_windows", return_value=windows):
            first, _ = _window_equity_fingerprint(base, allocator, config)
            source.write_text("date,equity\n2026-07-20,2", encoding="utf-8")
            second, _ = _window_equity_fingerprint(base, allocator, config)

        assert first != second
