from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_etf_lab.uzi_ablation import (  # noqa: E402
    DEFAULT_HORIZONS,
    DEFAULT_OUTCOME_HISTORY_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STOCK_PRICE_DIR,
    DEFAULT_UZI_SNAPSHOT_PATH,
    run_uzi_sidecar_ablation,
)


def _parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
    if not horizons or any(item <= 0 for item in horizons):
        raise argparse.ArgumentTypeError("--horizons must be a comma-separated list of positive integers.")
    return horizons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run research-only UZI sidecar ablation against local outcome/price data.")
    parser.add_argument("--uzi-snapshot", default=str(DEFAULT_UZI_SNAPSHOT_PATH), help="Path to uzi_auxiliary_snapshot.json.")
    parser.add_argument("--stock-price-dir", default=str(DEFAULT_STOCK_PRICE_DIR), help="Directory containing processed stock CSV files.")
    parser.add_argument(
        "--outcome-history",
        default=str(DEFAULT_OUTCOME_HISTORY_PATH),
        help="Existing stock target review outcome history CSV. Used before price-cache fallback.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for ablation CSV/JSON/report.")
    parser.add_argument("--horizons", type=_parse_horizons, default=",".join(str(item) for item in DEFAULT_HORIZONS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_uzi_sidecar_ablation(
        output_dir=Path(args.output_dir),
        uzi_snapshot_path=Path(args.uzi_snapshot),
        stock_price_dir=Path(args.stock_price_dir),
        outcome_history_path=Path(args.outcome_history),
        horizons=args.horizons,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "snapshot_path": str(result.snapshot_path),
                "summary_path": str(result.summary_path),
                "report_path": str(result.report_path),
                "sample_count": result.snapshot["summary"]["sample_count"],
                "risk_flag_count": result.snapshot["summary"]["risk_flag_count"],
                "missing_price_cache_count": result.snapshot["summary"]["missing_price_cache_count"],
                "position_effect": result.snapshot["position_effect"],
                "broker_action": result.snapshot["broker_action"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
