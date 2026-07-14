from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from research_database import ResearchDatabase
from tdx_day_source import infer_tdx_archive_specs, iter_tdx_day_archive_member_frames


DATE_TOKEN = re.compile(r"(?<!\d)(20\d{6})(?!\d)")


def extract_date_tokens(path: Path) -> list[str]:
    tokens = []
    for token in DATE_TOKEN.findall(path.stem):
        if not pd.isna(pd.to_datetime(token, format="%Y%m%d", errors="coerce")):
            tokens.append(token)
    return tokens


def discover_latest_panel(root: Path = Path(".")) -> Path | None:
    candidates = list(root.glob("data_panel_history_main_chinext_*.csv"))
    dated = [(max(tokens), path) for path in candidates if (tokens := extract_date_tokens(path))]
    return max(dated, default=(None, None), key=lambda item: item[0])[1]


def select_observation_files(output_dir: Path, asof_date: str | None = None) -> list[Path]:
    candidates = sorted(output_dir.glob("*.csv"))
    dated = [(path, tokens) for path in candidates if (tokens := extract_date_tokens(path))]
    if not dated:
        return []
    selected_date = asof_date or max(max(tokens) for _, tokens in dated)
    if pd.isna(pd.to_datetime(selected_date, format="%Y%m%d", errors="coerce")):
        raise ValueError(f"Invalid as-of date: {selected_date!r}; expected YYYYMMDD")
    return [path for path, tokens in dated if selected_date in tokens]


def read_table(path: Path) -> pd.DataFrame:
    # THS exports can be tab-delimited text despite an .xls suffix.
    if path.suffix.lower() == ".xls":
        return pd.read_csv(path, sep="\t", dtype=str, encoding="gb18030")
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/update the local A-share research SQLite database.")
    parser.add_argument("--db", default="data/research.sqlite3")
    parser.add_argument("--panel", help="Panel CSV. Defaults to the latest dated panel in the current directory.")
    parser.add_argument("--skip-panel", action="store_true")
    parser.add_argument("--daily-dir", default="D:/codex/daily-market-data/ths_exports/normalized")
    parser.add_argument("--output-dir", default="outputs/high_return_v2")
    parser.add_argument("--asof-date", help="Observation date in YYYYMMDD. Defaults to the latest dated output.")
    parser.add_argument("--tdx-root", default="D:/数据源")
    parser.add_argument("--skip-tdx", action="store_true")
    parser.add_argument("--tdx-symbols-from-panel", action="store_true")
    args = parser.parse_args()

    db = ResearchDatabase(args.db)
    panel = Path(args.panel) if args.panel else discover_latest_panel()
    if panel is None:
        panel = Path("__panel_not_found__")
        if not args.skip_panel:
            print("prices panel: no dated panel found; skipped", flush=True)
    if panel.exists() and not args.skip_panel:
        frame = pd.read_csv(panel, low_memory=False)
        price_cols = {"symbol", "date", "open", "high", "low", "close", "volume", "amount"}
        if price_cols.issubset(frame.columns):
            print(f"prices panel: {db.import_prices(frame, str(panel))}")

    daily_dir = Path(args.daily_dir)
    for path in sorted(daily_dir.glob("ths_hs_a_share_*.xls")) + sorted(daily_dir.glob("ths_hs_a_share_*.csv")):
        frame = read_table(path)
        renamed = {column.lower(): column for column in frame.columns}
        mapping = {renamed[key]: key for key in ("symbol", "date", "open", "high", "low", "close", "volume", "amount") if key in renamed}
        mapping.update({
            "代码": "symbol", "    名称": "stock_name", "现价": "close",
            "开盘": "open", "最高": "high", "最低": "low",
            "总成交量": "volume", "成交量": "volume", "成交额": "amount",
        })
        frame = frame.rename(columns=mapping)
        if "date" not in frame.columns:
            token = next((part for part in path.stem.split("_") if len(part) == 8 and part.isdigit()), None)
            if token:
                frame["date"] = pd.to_datetime(token, format="%Y%m%d").strftime("%Y-%m-%d")
        if {"symbol", "date", "close"}.issubset(frame.columns):
            for column in ("open", "high", "low", "volume", "amount"):
                if column not in frame.columns:
                    frame[column] = None
            print(f"prices {path.name}: {db.import_prices(frame, str(path))}")

    output_dir = Path(args.output_dir)
    for path in select_observation_files(output_dir, args.asof_date):
        frame = pd.read_csv(path, low_memory=False)
        if "date" not in frame.columns:
            token = next((part for part in path.stem.split("_") if len(part) == 8 and part.isdigit()), None)
            if token:
                frame["date"] = pd.to_datetime(token, format="%Y%m%d").strftime("%Y-%m-%d")
        if "date" in frame.columns:
            print(f"observations {path.name}: {db.import_observations(frame, path.stem, str(path))}")

    tdx_symbol_filter = None
    if args.tdx_symbols_from_panel and panel.exists():
        symbol_frame = pd.read_csv(panel, usecols=["symbol"], low_memory=False)
        tdx_symbol_filter = set(symbol_frame["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).dropna())
        print(f"tdx symbol filter: {len(tdx_symbol_filter)} symbols", flush=True)

    if not args.skip_tdx:
        tdx_root = Path(args.tdx_root)
        if tdx_root.exists():
            for spec in infer_tdx_archive_specs(tdx_root):
                archive_key = str(spec.path)
                result = db.import_tdx_member_frames(
                    archive_key,
                    iter_tdx_day_archive_member_frames(spec, symbol_filter=tdx_symbol_filter),
                )
                print(
                    f"tdx {spec.path.name}: read={result['read_rows']} inserted={result['inserted_rows']} "
                    f"imported_files={result['imported_files']} skipped_files={result['skipped_files']}",
                    flush=True,
                )

    print(db.query("SELECT COUNT(*) AS rows, MIN(date) AS first_date, MAX(date) AS last_date FROM daily_prices").to_dict("records")[0])
    print(db.query("SELECT COUNT(*) AS rows, MIN(date) AS first_date, MAX(date) AS last_date FROM tdx_daily_prices").to_dict("records")[0])
    print(db.query("SELECT kind, COUNT(*) AS rows FROM observations GROUP BY kind ORDER BY rows DESC").to_dict("records"))


if __name__ == "__main__":
    main()
