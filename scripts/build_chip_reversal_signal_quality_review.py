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

from quant_etf_lab.chip_reversal_signal_quality_review import (  # noqa: E402
    DEFAULT_EVENTS_PATH,
    DEFAULT_TRADES_PATH,
    run_chip_reversal_signal_quality_review,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a chip-reversal satellite signal-quality review.")
    parser.add_argument(
        "--trades-path",
        default=str(PROJECT_ROOT / DEFAULT_TRADES_PATH),
        help="Path to selected_events.csv or another completed trade ledger.",
    )
    parser.add_argument(
        "--events-path",
        default=str(PROJECT_ROOT / DEFAULT_EVENTS_PATH),
        help="Path to the raw chip-reversal event file with signal features.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "research" / f"chip_reversal_signal_quality_{datetime.now():%Y%m%d_%H%M%S}"),
        help="Output directory for Chinese CSV files, JSON summary, and HTML report.",
    )
    parser.add_argument("--name", default="chip_reversal_signal_quality", help="Review name displayed in the report.")
    parser.add_argument("--min-bucket-trades", type=int, default=30, help="Minimum trades required to keep a feature bucket.")
    parser.add_argument("--worst-trade-count", type=int, default=50, help="Number of losing trades to export for review.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    events_path = args.events_path if args.events_path else None
    result = run_chip_reversal_signal_quality_review(
        trades_path=args.trades_path,
        events_path=events_path,
        output_dir=args.output_dir,
        name=args.name,
        min_bucket_trades=args.min_bucket_trades,
        worst_trade_count=args.worst_trade_count,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "rank_path": str(result.rank_path),
                "feature_bins_path": str(result.feature_bins_path),
                "worst_trades_path": str(result.worst_trades_path),
                "summary_path": str(result.summary_path),
                "report_path": str(result.report_path),
                "completed_trade_count": result.summary.get("completed_trade_count"),
                "win_rate": result.summary.get("win_rate"),
                "avg_trade_return": result.summary.get("avg_trade_return"),
                "worst_feature_bucket": result.summary.get("worst_feature_bucket"),
                "extreme_signal_gain_count": result.summary.get("extreme_signal_gain_count"),
                "extreme_signal_gain_ratio": result.summary.get("extreme_signal_gain_ratio"),
                "extreme_signal_gain_avg_trade_return": result.summary.get("extreme_signal_gain_avg_trade_return"),
                "broker_action": result.summary.get("broker_action"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
