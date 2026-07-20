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

from quant_etf_lab.chip_reversal_event_portfolio import (  # noqa: E402
    DEFAULT_EVENTS_PATH,
    DEFAULT_ROUND_TRIP_COST,
    run_chip_reversal_event_portfolio,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a research-only event portfolio curve from chip-reversal events.")
    parser.add_argument(
        "--events-path",
        default=str(PROJECT_ROOT / DEFAULT_EVENTS_PATH),
        help="Path to chip_reversal_events.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "research" / f"chip_reversal_event_portfolio_{datetime.now():%Y%m%d_%H%M%S}"),
        help="Directory for equity curve, selected events, metrics JSON, and report.",
    )
    parser.add_argument("--horizon", type=int, default=5, help="Holding horizon label to use, e.g. 1, 2, or 5.")
    parser.add_argument("--max-positions", type=int, default=10, help="Maximum events selected per signal date.")
    parser.add_argument("--score-bucket", choices=["all", "base", "strong", "deep"], default="deep")
    parser.add_argument("--board-scope", choices=["main_chinext", "all"], default="main_chinext")
    parser.add_argument("--round-trip-cost", type=float, default=DEFAULT_ROUND_TRIP_COST)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--execution-model", choices=["proxy", "true-position"], default="proxy")
    parser.add_argument("--price-data-dir", default=str(PROJECT_ROOT / "data" / "processed" / "stocks"))
    parser.add_argument("--max-total-positions", type=int, default=None)
    parser.add_argument("--per-position-weight", type=float, default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-cost-gap-ma20", type=float, default=None, help="Keep events with cost_gap_ma20 <= this value.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_chip_reversal_event_portfolio(
        events_path=args.events_path,
        output_dir=args.output_dir,
        horizon=args.horizon,
        max_positions=args.max_positions,
        score_bucket=args.score_bucket,
        board_scope=args.board_scope,
        round_trip_cost=args.round_trip_cost,
        exposure=args.exposure,
        initial_cash=args.initial_cash,
        start_date=args.start_date,
        end_date=args.end_date,
        execution_model=args.execution_model,
        price_data_dir=args.price_data_dir,
        max_total_positions=args.max_total_positions,
        per_position_weight=args.per_position_weight,
        max_cost_gap_ma20=args.max_cost_gap_ma20,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "curve_path": str(result.curve_path),
                "trades_path": str(result.trades_path),
                "metrics_path": str(result.metrics_path),
                "snapshot_path": str(result.snapshot_path),
                "report_path": str(result.report_path),
                "total_return": result.metrics.get("total_return"),
                "max_drawdown": result.metrics.get("max_drawdown"),
                "sharpe": result.metrics.get("sharpe"),
                "cost_gap_ma20_max": result.metrics.get("cost_gap_ma20_max"),
                "signal_quality_filter_removed_count": result.metrics.get("signal_quality_filter_removed_count"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
