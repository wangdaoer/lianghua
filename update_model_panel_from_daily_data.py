"""Merge the historical model panel with normalized daily market exports."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BASE_PANEL = Path("data_panel_history_main_chinext_20220101_20260626.csv")
DEFAULT_DAILY_DIR = Path(r"D:\codex\daily-market-data\ths_exports\normalized")
DEFAULT_OUTPUT = Path("data_panel_history_main_chinext_20220101_latest.csv")
DEFAULT_PREFIXES = ("000", "001", "002", "003", "300", "301", "600", "601", "603", "605")


@dataclass(frozen=True)
class DailySummary:
    date: str
    file: str
    rows: int
    symbols: int
    amount_source: str
    raw_positive_ratio: float


def clean_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6)[-6:] if digits else ""


def allowed_code(code: str, prefixes: tuple[str, ...]) -> bool:
    return bool(code) and any(code.startswith(prefix) for prefix in prefixes)


def parse_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text in {"--", "-", "nan", "None"}:
        return np.nan
    multiplier = 1.0
    if text.endswith("%"):
        text = text[:-1]
    if text.endswith("亿"):
        multiplier = 100_000_000.0
        text = text[:-1]
    elif text.endswith("万"):
        multiplier = 10_000.0
        text = text[:-1]
    text = text.replace(",", "").replace("↑", "").replace("↓", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return np.nan
    return float(match.group(0)) * multiplier


def _numeric(series: pd.Series) -> pd.Series:
    return series.map(parse_number).astype(float)


def _estimate_amount(market_cap: pd.Series, turnover_rate: pd.Series) -> pd.Series:
    amount = pd.to_numeric(market_cap, errors="coerce") * pd.to_numeric(turnover_rate, errors="coerce") / 100.0
    return amount.where(amount.gt(0))


def _finalize(frame: pd.DataFrame, prefixes: tuple[str, ...]) -> pd.DataFrame:
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].map(clean_code)
    frame = frame[frame["symbol"].map(lambda x: allowed_code(x, prefixes))]
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["amount"] = frame["amount"].where(frame["amount"].gt(0))
    frame["volume"] = frame["volume"].where(frame["volume"].gt(0), frame["amount"] / frame["close"])
    frame = frame.dropna(subset=["date", "symbol", "open", "high", "low", "close", "volume", "amount"])
    frame = frame[
        frame["open"].gt(0)
        & frame["high"].gt(0)
        & frame["low"].gt(0)
        & frame["close"].gt(0)
        & frame["volume"].gt(0)
        & frame["amount"].gt(0)
    ]
    return frame[["date", "symbol", "open", "high", "low", "close", "volume", "amount"]]


def read_daily_csv(path: Path, trade_date: str, prefixes: tuple[str, ...]) -> tuple[pd.DataFrame, DailySummary]:
    raw = pd.read_csv(path)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw.get("trade_date", trade_date), errors="coerce"),
            "symbol": raw.get("security_code", raw.get("raw_security_code")),
            "open": pd.to_numeric(raw.get("open", raw.get("open_price")), errors="coerce"),
            "high": pd.to_numeric(raw.get("high", raw.get("high_price")), errors="coerce"),
            "low": pd.to_numeric(raw.get("low", raw.get("low_price")), errors="coerce"),
            "close": pd.to_numeric(raw.get("close", raw.get("close_price")), errors="coerce"),
            "volume": pd.to_numeric(raw.get("volume"), errors="coerce"),
            "amount": pd.to_numeric(raw.get("amount", raw.get("turnover")), errors="coerce"),
            "turnover_rate": pd.to_numeric(raw.get("turnover_rate"), errors="coerce"),
            "market_cap": pd.to_numeric(raw.get("market_cap"), errors="coerce"),
        }
    )
    frame["date"] = frame["date"].fillna(pd.Timestamp(trade_date))
    tradable = frame["close"].gt(0)
    raw_amount = frame["amount"]
    raw_positive_ratio = float(raw_amount[tradable].gt(0).mean()) if tradable.any() else 0.0
    raw_volume_ratio = float(frame["volume"][tradable].gt(0).mean()) if tradable.any() else 0.0
    raw_amount_usable = (
        raw_positive_ratio >= 0.90
        and raw_volume_ratio >= 0.80
        and float(raw_amount[raw_amount.gt(0)].median() or 0.0) >= 1_000_000.0
    )
    amount_source = "raw_amount"
    if not raw_amount_usable:
        frame["amount"] = _estimate_amount(frame["market_cap"], frame["turnover_rate"])
        frame["volume"] = frame["amount"] / frame["close"]
        amount_source = "market_cap_x_turnover_rate"

    out = _finalize(frame, prefixes)
    summary = DailySummary(
        date=trade_date,
        file=str(path),
        rows=int(len(out)),
        symbols=int(out["symbol"].nunique()),
        amount_source=amount_source,
        raw_positive_ratio=raw_positive_ratio,
    )
    return out, summary


def read_daily_xls(path: Path, trade_date: str, prefixes: tuple[str, ...]) -> tuple[pd.DataFrame, DailySummary]:
    raw = pd.read_csv(path, sep="\t", encoding="gb18030")
    raw.columns = [str(c).strip() for c in raw.columns]
    code_col = "代码"
    close_col = "现价"
    frame = pd.DataFrame(
        {
            "date": pd.Timestamp(trade_date),
            "symbol": raw.get(code_col),
            "open": _numeric(raw.get("开盘", raw.get(close_col))),
            "high": _numeric(raw.get("最高", raw.get(close_col))),
            "low": _numeric(raw.get("最低", raw.get(close_col))),
            "close": _numeric(raw.get(close_col)),
            "volume": _numeric(raw.get("总手", pd.Series(np.nan, index=raw.index))),
            "amount": _numeric(raw.get("总金额", pd.Series(np.nan, index=raw.index))),
            "turnover_rate": _numeric(raw.get("换手", pd.Series(np.nan, index=raw.index))),
            "market_cap": _numeric(raw.get("总市值", pd.Series(np.nan, index=raw.index))),
        }
    )
    tradable = frame["close"].gt(0)
    raw_positive_ratio = float(frame["amount"][tradable].gt(0).mean()) if tradable.any() else 0.0
    amount_source = "raw_amount"
    if raw_positive_ratio < 0.90:
        frame["amount"] = _estimate_amount(frame["market_cap"], frame["turnover_rate"])
        amount_source = "market_cap_x_turnover_rate"
    frame["volume"] = frame["volume"].where(frame["volume"].gt(0), frame["amount"] / frame["close"])

    out = _finalize(frame, prefixes)
    summary = DailySummary(
        date=trade_date,
        file=str(path),
        rows=int(len(out)),
        symbols=int(out["symbol"].nunique()),
        amount_source=amount_source,
        raw_positive_ratio=raw_positive_ratio,
    )
    return out, summary


def date_from_name(path: Path) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if not match:
        raise ValueError(f"Cannot infer trade date from filename: {path}")
    return match.group(1)


def select_daily_files(daily_dir: Path, start_date: str, end_date: str | None) -> list[Path]:
    candidates = sorted(daily_dir.glob("ths_hs_a_share_*.csv")) + sorted(
        daily_dir.glob("ths_hs_a_share_*.xls")
    )
    by_date: dict[str, list[Path]] = {}
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else None
    for path in candidates:
        trade_date = pd.Timestamp(date_from_name(path))
        if trade_date < start or (end is not None and trade_date > end):
            continue
        by_date.setdefault(trade_date.strftime("%Y-%m-%d"), []).append(path)

    selected = []
    for trade_date, files in sorted(by_date.items()):
        files = sorted(files, key=lambda p: 0 if p.suffix.lower() == ".csv" else 1)
        selected.append(files[0])
    return selected


def load_daily_panel(
    daily_dir: Path,
    start_date: str,
    end_date: str | None,
    prefixes: tuple[str, ...],
) -> tuple[pd.DataFrame, list[DailySummary]]:
    frames = []
    summaries = []
    for path in select_daily_files(daily_dir, start_date, end_date):
        trade_date = date_from_name(path)
        if path.suffix.lower() == ".csv":
            frame, summary = read_daily_csv(path, trade_date, prefixes)
        else:
            frame, summary = read_daily_xls(path, trade_date, prefixes)
        if not frame.empty:
            frames.append(frame)
        summaries.append(summary)
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "volume", "amount"]), summaries
    return pd.concat(frames, ignore_index=True), summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update the model panel with daily normalized exports.")
    parser.add_argument("--base-panel", default=str(DEFAULT_BASE_PANEL))
    parser.add_argument("--daily-dir", default=str(DEFAULT_DAILY_DIR))
    parser.add_argument("--daily-start", default="2026-06-22")
    parser.add_argument("--daily-end", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--prefixes", default=",".join(DEFAULT_PREFIXES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_panel)
    daily_dir = Path(args.daily_dir)
    output = Path(args.output)
    prefixes = tuple(x.strip() for x in args.prefixes.split(",") if x.strip())

    base = pd.read_csv(base_path, parse_dates=["date"])
    base["symbol"] = base["symbol"].map(clean_code)
    base_cut = base[base["date"] < pd.Timestamp(args.daily_start)].copy()
    daily, summaries = load_daily_panel(daily_dir, args.daily_start, args.daily_end, prefixes)
    if daily.empty:
        raise ValueError(f"No daily rows loaded from {daily_dir}")

    out = pd.concat([base_cut, daily], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["symbol"] = out["symbol"].map(clean_code)
    out = out.dropna(subset=["date", "symbol"])
    out = out.drop_duplicates(["date", "symbol"], keep="last")
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8")

    print(
        f"Updated panel rows={len(out)} days={out['date'].nunique()} "
        f"symbols={out['symbol'].nunique()} latest={out['date'].max()} path={output}"
    )
    for summary in summaries:
        print(
            f"{summary.date} rows={summary.rows} symbols={summary.symbols} "
            f"amount_source={summary.amount_source} raw_positive_ratio={summary.raw_positive_ratio:.3f} "
            f"file={Path(summary.file).name}"
        )


if __name__ == "__main__":
    main()
