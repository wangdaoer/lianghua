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

from quant_etf_lab.portfolio_rolling_weakness_review import (  # noqa: E402
    DEFAULT_EQUITY_PATH,
    run_portfolio_rolling_weakness_review,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a rolling-window weakness review for a portfolio curve.")
    parser.add_argument(
        "--equity-path",
        default=str(PROJECT_ROOT / DEFAULT_EQUITY_PATH),
        help="Path to portfolio_equity.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "research" / f"portfolio_rolling_weakness_{datetime.now():%Y%m%d_%H%M%S}"),
        help="Output directory for Chinese CSV files, JSON summary, and HTML report.",
    )
    parser.add_argument("--name", default="portfolio", help="Review name displayed in the report.")
    parser.add_argument("--window-months", type=int, default=3, help="Rolling calendar-month window length.")
    parser.add_argument("--min-window-rows", type=int, default=40, help="Minimum rows required for each rolling window.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_portfolio_rolling_weakness_review(
        equity_path=args.equity_path,
        output_dir=args.output_dir,
        name=args.name,
        window_months=args.window_months,
        min_window_rows=args.min_window_rows,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "windows_path": str(result.windows_path),
                "weak_windows_path": str(result.weak_windows_path),
                "reasons_path": str(result.reasons_path),
                "months_path": str(result.months_path),
                "summary_path": str(result.summary_path),
                "report_path": str(result.report_path),
                "underperform_window_ratio": result.summary.get("underperform_window_ratio"),
                "worst_window": result.summary.get("worst_window"),
                "worst_reason": result.summary.get("worst_reason"),
                "worst_month": result.summary.get("worst_month"),
                "broker_action": result.summary.get("broker_action"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
