from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_etf_lab.uzi_auxiliary import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TICKERS,
    DEFAULT_UZI_CACHE_ROOT,
    run_uzi_auxiliary_export,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export UZI-Skill sidecar cache into research-only auxiliary signals.")
    parser.add_argument(
        "--cache-root",
        default=str(DEFAULT_UZI_CACHE_ROOT),
        help="UZI-Skill .cache directory containing per-ticker raw_data/dimensions/panel JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for uzi_auxiliary_snapshot.json, uzi_auxiliary_signals.csv, and report markdown.",
    )
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated tickers or six-digit A-share codes, e.g. 301165.SZ,300870.SZ,688629.SH.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tickers = [item.strip() for item in str(args.tickers).split(",") if item.strip()]
    result = run_uzi_auxiliary_export(
        output_dir=Path(args.output_dir),
        cache_root=Path(args.cache_root),
        tickers=tickers,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "snapshot_path": str(result.snapshot_path),
                "signals_path": str(result.signals_path),
                "report_path": str(result.report_path),
                "signal_count": len(result.snapshot["signals"]),
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
