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

from quant_etf_lab.historical_oos_review import (  # noqa: E402
    DEFAULT_CURVE_SPECS,
    DEFAULT_END,
    DEFAULT_START,
    CurveSpec,
    run_historical_oos_review,
)


def _parse_curve(value: str) -> CurveSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--curve must use label=path or label=path|category|equity_column")
    label, raw = value.split("=", 1)
    parts = raw.split("|")
    path = Path(parts[0])
    category = parts[1] if len(parts) >= 2 and parts[1] else "custom"
    equity_column = parts[2] if len(parts) >= 3 and parts[2] else "stitched_equity"
    return CurveSpec(label=label.strip(), path=path, category=category, equity_column=equity_column)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a 2018-2025 historical OOS review from stitched equity curves.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root used to resolve relative curve paths.")
    parser.add_argument("--start", default=DEFAULT_START, help="Review start date, e.g. 2018-01-01.")
    parser.add_argument("--end", default=DEFAULT_END, help="Review end date, e.g. 2025-12-31.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "research" / f"historical_2018_2025_oos_review_{datetime.now():%Y%m%d_%H%M%S}"),
        help="Directory for summary CSV, Chinese CSV, JSON snapshot, and Markdown report.",
    )
    parser.add_argument(
        "--curve",
        action="append",
        type=_parse_curve,
        help="Optional curve override: label=path or label=path|category|equity_column. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = tuple(args.curve) if args.curve else DEFAULT_CURVE_SPECS
    result = run_historical_oos_review(
        project_root=Path(args.project_root),
        output_dir=Path(args.output_dir),
        curve_specs=specs,
        start=args.start,
        end=args.end,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "summary_path": str(result.summary_path),
                "chinese_summary_path": str(result.chinese_summary_path),
                "snapshot_path": str(result.snapshot_path),
                "report_path": str(result.report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
