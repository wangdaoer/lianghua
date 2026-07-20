from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from quant_etf_lab.source_selection_stability import (
    build_source_selection_stability,
    run_source_selection_stability_review,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_run_dir(
    root: Path,
    name: str,
    equities: list[float],
    windows: list[tuple[str, str, float]],
) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(equities), freq="D"),
            "stitched_equity": equities,
        }
    ).to_csv(run_dir / "oos_equity_stitched.csv", index=False)
    pd.DataFrame(
        {
            "window": [item[0] for item in windows],
            "selected_source": [item[1] for item in windows],
            "test_total_return": [item[2] for item in windows],
            "test_max_drawdown": [-0.02 for _ in windows],
            "test_sharpe": [1.0 for _ in windows],
        }
    ).to_csv(run_dir / "portfolio_walk_forward_summary.csv", index=False)
    return run_dir


def test_build_source_selection_stability_summarizes_curves_and_window_contribution(tmp_path: Path) -> None:
    strong = _write_run_dir(
        tmp_path,
        "strong",
        [100.0, 110.0, 121.0],
        [("pf_01", "quality", 0.10), ("pf_02", "highgain", 0.10)],
    )
    weak = _write_run_dir(
        tmp_path,
        "weak",
        [100.0, 95.0, 105.0],
        [("pf_01", "quality", -0.05), ("pf_02", "reversal", 0.1052631579)],
    )

    metrics, windows, sources, summary = build_source_selection_stability(
        {"strong": strong, "weak": weak},
        baseline_label="weak",
    )

    assert summary["best_total_return_label"] == "strong"
    strong_metric = metrics.set_index("label").loc["strong"]
    assert abs(float(strong_metric["total_return"]) - 0.21) < 1e-9
    assert abs(float(windows[windows["label"].eq("strong")]["growth_contribution"].sum()) - 0.21) < 1e-9
    assert float(sources[sources["selected_source"].eq("quality")]["selected_count"].sum()) == 2.0
    assert "excess_return_vs_baseline" in metrics.columns


def test_run_source_selection_stability_review_writes_outputs(tmp_path: Path) -> None:
    strong = _write_run_dir(
        tmp_path,
        "strong",
        [100.0, 110.0, 121.0],
        [("pf_01", "quality", 0.10), ("pf_02", "highgain", 0.10)],
    )
    weak = _write_run_dir(
        tmp_path,
        "weak",
        [100.0, 95.0, 105.0],
        [("pf_01", "quality", -0.05), ("pf_02", "reversal", 0.1052631579)],
    )

    result = run_source_selection_stability_review(
        {"strong": strong, "weak": weak},
        output_dir=tmp_path / "review",
        baseline_label="weak",
        name="unit review",
    )

    assert result.report_path.exists()
    assert (result.output_dir / "source_selection_curve_metrics.csv").exists()
    assert (result.output_dir / "source_selection_window_contribution.csv").exists()
    assert (result.output_dir / "source_selection_source_usage.csv").exists()
    assert result.summary["best_total_return_label"] == "strong"


def test_build_source_selection_stability_script_parses_runs() -> None:
    script_path = ROOT / "scripts" / "build_source_selection_stability_review.py"
    spec = importlib.util.spec_from_file_location("build_source_selection_stability_review", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(
        [
            "--run",
            "candidate=outputs/portfolio_source_selection/candidate",
            "--run",
            "baseline=outputs/portfolio_source_selection/baseline",
            "--baseline-label",
            "baseline",
            "--output-dir",
            "outputs/research/source_selection_stability_test",
        ]
    )

    assert args.run == [
        "candidate=outputs/portfolio_source_selection/candidate",
        "baseline=outputs/portfolio_source_selection/baseline",
    ]
    assert args.baseline_label == "baseline"
