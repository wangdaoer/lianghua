from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_etf_lab.source_selection_stability import run_source_selection_stability_review  # noqa: E402


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--run must use LABEL=DIR format: {value}")
    label, path_text = value.split("=", 1)
    label = label.strip()
    path_text = path_text.strip()
    if not label:
        raise argparse.ArgumentTypeError("--run label cannot be empty")
    if not path_text:
        raise argparse.ArgumentTypeError("--run directory cannot be empty")
    return label, Path(path_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a stability review for source-selection walk-forward runs.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run directory in LABEL=DIR format. Repeat for multiple runs.",
    )
    parser.add_argument("--baseline-label", default=None, help="Optional label used for excess-return comparison.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "research" / f"source_selection_stability_{datetime.now():%Y%m%d_%H%M%S}"),
        help="Output directory for CSV, JSON, and Markdown report.",
    )
    parser.add_argument("--name", default="Source-selection stability review", help="Report title.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parsed = [_parse_run(item) for item in args.run]
    if not parsed:
        raise SystemExit("At least one --run LABEL=DIR is required.")
    run_dirs = dict(parsed)
    result = run_source_selection_stability_review(
        run_dirs,
        output_dir=args.output_dir,
        baseline_label=args.baseline_label,
        name=args.name,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "metrics_path": str(result.metrics_path),
                "window_path": str(result.window_path),
                "source_path": str(result.source_path),
                "summary_path": str(result.summary_path),
                "report_path": str(result.report_path),
                "best_total_return_label": result.summary.get("best_total_return_label"),
                "broker_action": result.summary.get("broker_action"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
