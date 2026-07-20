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

from quant_etf_lab.chip_reversal_stability import (  # noqa: E402
    DEFAULT_EVENTS_PATH,
    run_chip_reversal_stability_review,
)


def _parse_horizons(value: str) -> list[int]:
    horizons = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not horizons:
        raise argparse.ArgumentTypeError("At least one horizon is required.")
    return horizons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a year-by-year stability review for chip-reversal events.")
    parser.add_argument(
        "--events-path",
        default=str(PROJECT_ROOT / DEFAULT_EVENTS_PATH),
        help="Path to chip_reversal_events.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "research" / f"chip_reversal_stability_{datetime.now():%Y%m%d_%H%M%S}"),
        help="Output directory for annual CSV, summary CSV, JSON snapshot, and Markdown report.",
    )
    parser.add_argument("--horizons", type=_parse_horizons, default=[1, 2, 5], help="Comma-separated horizons, e.g. 1,2,5.")
    parser.add_argument("--min-events", type=int, default=30, help="Minimum events per year/group.")
    parser.add_argument("--min-years", type=int, default=2, help="Minimum years required for stable_positive.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_chip_reversal_stability_review(
        events_path=args.events_path,
        output_dir=args.output_dir,
        horizons=args.horizons,
        min_events=args.min_events,
        min_years=args.min_years,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "annual_path": str(result.annual_path),
                "summary_path": str(result.summary_path),
                "snapshot_path": str(result.snapshot_path),
                "report_path": str(result.report_path),
                "stable_positive_group_count": result.snapshot.get("stable_positive_group_count"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
