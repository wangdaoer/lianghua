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

from quant_etf_lab.portfolio_stability import DEFAULT_EQUITY_PATH, run_portfolio_stability_review  # noqa: E402


def _parse_months(value: str) -> list[int]:
    months = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not months:
        raise argparse.ArgumentTypeError("At least one rolling month window is required.")
    return months


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a portfolio stability review against the core baseline.")
    parser.add_argument(
        "--equity-path",
        default=str(PROJECT_ROOT / DEFAULT_EQUITY_PATH),
        help="Path to portfolio_equity.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "research" / f"portfolio_stability_{datetime.now():%Y%m%d_%H%M%S}"),
        help="Output directory for Chinese CSV files, JSON summary, and HTML report.",
    )
    parser.add_argument("--name", default="portfolio", help="Review name displayed in the report.")
    parser.add_argument("--portfolio-col", default="portfolio_equity", help="Portfolio equity column.")
    parser.add_argument("--baseline-col", default="core_equity", help="Baseline equity column.")
    parser.add_argument("--rolling-months", type=_parse_months, default=[3, 6, 12], help="Comma-separated windows, e.g. 3,6,12.")
    parser.add_argument("--min-year-trading-days", type=int, default=2, help="Minimum rows required for an annual row.")
    parser.add_argument("--min-window-trading-days", type=int, default=40, help="Minimum rows required for a rolling window row.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_portfolio_stability_review(
        equity_path=args.equity_path,
        output_dir=args.output_dir,
        name=args.name,
        portfolio_col=args.portfolio_col,
        baseline_col=args.baseline_col,
        rolling_months=args.rolling_months,
        min_year_trading_days=args.min_year_trading_days,
        min_window_trading_days=args.min_window_trading_days,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "annual_path": str(result.annual_path),
                "rolling_path": str(result.rolling_path),
                "summary_path": str(result.summary_path),
                "report_path": str(result.report_path),
                "evidence_status": result.summary.get("evidence_status"),
                "broker_action": result.summary.get("broker_action"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
