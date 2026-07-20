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

from quant_etf_lab.portfolio_satellite_failure_review import (  # noqa: E402
    DEFAULT_EQUITY_PATH,
    run_portfolio_satellite_failure_review,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a risk-on satellite failure review for a portfolio curve.")
    parser.add_argument(
        "--equity-path",
        default=str(PROJECT_ROOT / DEFAULT_EQUITY_PATH),
        help="Path to portfolio_equity.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "research" / f"portfolio_satellite_failure_{datetime.now():%Y%m%d_%H%M%S}"),
        help="Output directory for Chinese CSV files, JSON summary, and HTML report.",
    )
    parser.add_argument("--name", default="portfolio", help="Review name displayed in the report.")
    parser.add_argument("--regime", default="risk_on", help="Portfolio regime to analyze, default: risk_on.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_portfolio_satellite_failure_review(
        equity_path=args.equity_path,
        output_dir=args.output_dir,
        name=args.name,
        regime=args.regime,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "monthly_path": str(result.monthly_path),
                "clusters_path": str(result.clusters_path),
                "reasons_path": str(result.reasons_path),
                "summary_path": str(result.summary_path),
                "report_path": str(result.report_path),
                "worst_month": result.summary.get("worst_month"),
                "worst_reason": result.summary.get("worst_reason"),
                "broker_action": result.summary.get("broker_action"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
