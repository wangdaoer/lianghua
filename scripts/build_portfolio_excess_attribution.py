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

from quant_etf_lab.portfolio_excess_attribution import (  # noqa: E402
    DEFAULT_EQUITY_PATH,
    run_portfolio_excess_attribution_review,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an excess-return attribution review for a portfolio curve.")
    parser.add_argument(
        "--equity-path",
        default=str(PROJECT_ROOT / DEFAULT_EQUITY_PATH),
        help="Path to portfolio_equity.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "research" / f"portfolio_excess_attribution_{datetime.now():%Y%m%d_%H%M%S}"),
        help="Output directory for Chinese CSV files, JSON summary, and HTML report.",
    )
    parser.add_argument("--name", default="portfolio", help="Review name displayed in the report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_portfolio_excess_attribution_review(
        equity_path=args.equity_path,
        output_dir=args.output_dir,
        name=args.name,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "annual_path": str(result.annual_path),
                "regime_path": str(result.regime_path),
                "daily_path": str(result.daily_path),
                "summary_path": str(result.summary_path),
                "report_path": str(result.report_path),
                "worst_annual_excess_year": result.summary.get("worst_annual_excess_year"),
                "broker_action": result.summary.get("broker_action"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
