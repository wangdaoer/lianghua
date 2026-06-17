"""A-share market-cap snapshot cache helpers."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_STOCK_MARKET_CAP_PATH = Path("data/processed/stock_market_cap_yi.csv")
MARKET_CAP_SOURCE = "akshare.stock_zh_a_spot_em"
MARKET_CAP_COLUMNS = [
    "code",
    "name",
    "market_cap_yi",
    "market_cap_yuan",
    "float_market_cap_yi",
    "float_market_cap_yuan",
    "latest_price",
    "pct_change",
    "snapshot_date",
    "updated_at",
    "source",
]


def _find_column(frame: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    if required:
        raise ValueError(f"Missing any of columns: {candidates}")
    return None


def _code_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\D", "", regex=True).str[-6:].str.zfill(6)


def _numeric_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.replace(",", "", regex=False)
    text = text.replace({"": np.nan, "-": np.nan, "--": np.nan, "None": np.nan, "nan": np.nan})
    return pd.to_numeric(text, errors="coerce")


def _optional_numeric(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    column = _find_column(frame, candidates, required=False)
    if column is None:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return _numeric_series(frame[column])


def _market_cap_yuan(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    yi_column = _find_column(
        frame,
        ["market_cap_yi", "total_market_cap_yi", "总市值_亿元", "总市值亿元"],
        required=False,
    )
    if yi_column is not None:
        market_cap_yi = _numeric_series(frame[yi_column])
        return market_cap_yi * 100_000_000.0, market_cap_yi

    yuan_column = _find_column(
        frame,
        ["market_cap_yuan", "total_market_cap_yuan", "总市值", "总市值-元", "总市值元", "总市值(元)"],
        required=True,
    )
    market_cap_yuan = _numeric_series(frame[yuan_column])
    return market_cap_yuan, market_cap_yuan / 100_000_000.0


def _float_market_cap_yuan(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    yi_column = _find_column(
        frame,
        ["float_market_cap_yi", "circulating_market_cap_yi", "流通市值_亿元", "流通市值亿元"],
        required=False,
    )
    if yi_column is not None:
        float_cap_yi = _numeric_series(frame[yi_column])
        return float_cap_yi * 100_000_000.0, float_cap_yi

    yuan_column = _find_column(
        frame,
        ["float_market_cap_yuan", "circulating_market_cap_yuan", "流通市值", "流通市值-元", "流通市值元", "流通市值(元)"],
        required=False,
    )
    if yuan_column is None:
        empty = pd.Series(np.nan, index=frame.index, dtype="float64")
        return empty, empty
    float_cap_yuan = _numeric_series(frame[yuan_column])
    return float_cap_yuan, float_cap_yuan / 100_000_000.0


def normalize_stock_market_cap_frame(
    frame: pd.DataFrame,
    snapshot_date: str | None = None,
    updated_at: str | None = None,
    source: str = MARKET_CAP_SOURCE,
) -> pd.DataFrame:
    """Normalize A-share market-cap snapshots to Yi Yuan units."""
    if frame.empty:
        return pd.DataFrame(columns=MARKET_CAP_COLUMNS)

    code_column = _find_column(frame, ["code", "代码", "证券代码", "A股代码"])
    name_column = _find_column(frame, ["name", "名称", "证券简称", "A股简称"], required=False)
    market_cap_yuan, market_cap_yi = _market_cap_yuan(frame)
    float_market_cap_yuan, float_market_cap_yi = _float_market_cap_yuan(frame)

    now = updated_at or datetime.now().isoformat(timespec="seconds")
    snapshot = snapshot_date or datetime.now().strftime("%Y-%m-%d")
    normalized = pd.DataFrame(
        {
            "code": _code_series(frame[code_column]),
            "name": frame[name_column].astype(str) if name_column is not None else "",
            "market_cap_yi": market_cap_yi,
            "market_cap_yuan": market_cap_yuan,
            "float_market_cap_yi": float_market_cap_yi,
            "float_market_cap_yuan": float_market_cap_yuan,
            "latest_price": _optional_numeric(frame, ["latest_price", "最新价", "最新", "close"]),
            "pct_change": _optional_numeric(frame, ["pct_change", "涨跌幅"]),
            "snapshot_date": frame["snapshot_date"].astype(str) if "snapshot_date" in frame.columns else snapshot,
            "updated_at": frame["updated_at"].astype(str) if "updated_at" in frame.columns else now,
            "source": frame["source"].astype(str) if "source" in frame.columns else source,
        }
    )
    normalized = normalized.dropna(subset=["code"])
    normalized = normalized[normalized["code"].str.len() == 6]
    normalized = normalized.drop_duplicates(subset=["code"], keep="last").sort_values("code").reset_index(drop=True)
    return normalized[MARKET_CAP_COLUMNS]


def fetch_stock_market_cap_snapshot(
    retry_count: int = 3,
    pause_seconds: float = 1.0,
) -> pd.DataFrame:
    """Fetch the full A-share spot snapshot and normalize total market cap."""
    import akshare as ak

    attempts = max(int(retry_count), 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw = ak.stock_zh_a_spot_em()
            return normalize_stock_market_cap_frame(raw, source=MARKET_CAP_SOURCE)
        except Exception as exc:  # External quote APIs can fail or throttle.
            last_error = exc
            if attempt < attempts and pause_seconds > 0:
                time.sleep(pause_seconds * attempt)
    raise RuntimeError(f"Failed to fetch stock market-cap snapshot after {attempts} attempts: {last_error}") from last_error


def _market_prefix(code: str) -> str:
    if code.startswith(("6", "9")):
        return "sh"
    if code.startswith(("0", "2", "3")):
        return "sz"
    if code.startswith(("4", "8")):
        return "bj"
    return "sz"


def _format_yyyymmdd(value: str | datetime | None) -> str:
    if value is None:
        return datetime.now().strftime("%Y%m%d")
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value).replace("-", "")
    return pd.Timestamp(parsed).strftime("%Y%m%d")


def load_stock_market_cap_symbols(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Stock symbol file not found: {resolved}")
    frame = pd.read_csv(resolved, dtype={"code": str})
    code_column = _find_column(frame, ["code", "代码", "证券代码", "A股代码"])
    name_column = _find_column(frame, ["name", "名称", "证券简称", "A股简称"], required=False)
    symbols = pd.DataFrame(
        {
            "code": _code_series(frame[code_column]),
            "name": frame[name_column].astype(str) if name_column is not None else "",
        }
    )
    return symbols.drop_duplicates(subset=["code"]).sort_values("code").reset_index(drop=True)


def fetch_stock_market_cap_for_symbols(
    symbols: pd.DataFrame,
    as_of_date: str | datetime | None = None,
    lookback_days: int = 10,
    retry_count: int = 3,
    pause_seconds: float = 0.5,
) -> pd.DataFrame:
    """Fetch market cap for a target symbol list using daily close * outstanding shares."""
    import akshare as ak

    if symbols.empty:
        return pd.DataFrame(columns=MARKET_CAP_COLUMNS)

    end = _format_yyyymmdd(as_of_date)
    end_ts = pd.to_datetime(end, format="%Y%m%d", errors="coerce")
    start_ts = (pd.Timestamp(end_ts) - timedelta(days=max(int(lookback_days), 1) * 2)) if pd.notna(end_ts) else datetime.now() - timedelta(days=30)
    start = pd.Timestamp(start_ts).strftime("%Y%m%d")
    attempts = max(int(retry_count), 1)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in symbols.to_dict(orient="records"):
        code = _code_series(pd.Series([item.get("code")])).iloc[0]
        name = str(item.get("name") or "")
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                raw = ak.stock_zh_a_daily(
                    symbol=f"{_market_prefix(code)}{code}",
                    start_date=start,
                    end_date=end,
                    adjust="",
                )
                if raw.empty:
                    raise ValueError("empty daily frame")
                data = raw.copy()
                data["date"] = pd.to_datetime(data["date"], errors="coerce")
                data["close"] = pd.to_numeric(data["close"], errors="coerce")
                data["outstanding_share"] = pd.to_numeric(data["outstanding_share"], errors="coerce")
                data = data.dropna(subset=["date", "close", "outstanding_share"]).sort_values("date")
                if data.empty:
                    raise ValueError("daily frame missing close/outstanding_share")
                latest = data.iloc[-1]
                market_cap_yuan = float(latest["close"]) * float(latest["outstanding_share"])
                rows.append(
                    {
                        "code": code,
                        "name": name,
                        "market_cap_yi": market_cap_yuan / 100_000_000.0,
                        "market_cap_yuan": market_cap_yuan,
                        "float_market_cap_yi": np.nan,
                        "float_market_cap_yuan": np.nan,
                        "latest_price": float(latest["close"]),
                        "pct_change": np.nan,
                        "snapshot_date": pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "source": "akshare.stock_zh_a_daily.outstanding_share",
                    }
                )
                break
            except Exception as exc:  # External data APIs can intermittently fail.
                last_error = exc
                if attempt < attempts and pause_seconds > 0:
                    time.sleep(pause_seconds * attempt)
        else:
            failures.append(f"{code}: {last_error}")
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if not rows:
        raise RuntimeError(f"Failed to fetch market cap for all symbols: {'; '.join(failures)}")
    frame = pd.DataFrame(rows)
    return normalize_stock_market_cap_frame(frame, source="akshare.stock_zh_a_daily.outstanding_share")


def load_stock_market_cap_cache(path: str | Path) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Stock market-cap cache not found: {resolved}")
    frame = pd.read_csv(resolved, dtype={"code": str})
    return normalize_stock_market_cap_frame(frame, source="local_market_cap_cache")


def update_stock_market_cap_cache(
    output_path: str | Path = DEFAULT_STOCK_MARKET_CAP_PATH,
    retry_count: int = 3,
    pause_seconds: float = 1.0,
    symbols: pd.DataFrame | None = None,
    as_of_date: str | datetime | None = None,
    lookback_days: int = 10,
) -> pd.DataFrame:
    if symbols is None:
        frame = fetch_stock_market_cap_snapshot(retry_count=retry_count, pause_seconds=pause_seconds)
    else:
        frame = fetch_stock_market_cap_for_symbols(
            symbols,
            as_of_date=as_of_date,
            lookback_days=lookback_days,
            retry_count=retry_count,
            pause_seconds=pause_seconds,
        )
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(resolved, index=False, encoding="utf-8")
    return frame
