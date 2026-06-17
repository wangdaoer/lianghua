"""Market data fetching, normalization, and cache access."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import ETFSpec, LabConfig, UniverseSourceConfig


COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change_amount",
    "换手率": "turnover_rate",
    "date": "date",
    "open": "open",
    "close": "close",
    "high": "high",
    "low": "low",
    "volume": "volume",
    "amount": "amount",
}

CORE_COLUMNS = ["date", "code", "name", "open", "high", "low", "close", "volume", "amount"]


class AkShareDataError(RuntimeError):
    """Raised when an AKShare request or response cannot be used."""


class UniverseSourceDataError(RuntimeError):
    """Raised when a universe source or fallback cannot be used."""


CACHE_LOAD_EXCEPTIONS = (
    FileNotFoundError,
    OSError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
    ValueError,
)
UNIVERSE_FALLBACK_EXCEPTIONS = (UniverseSourceDataError,) + CACHE_LOAD_EXCEPTIONS


def today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


def parse_market_date(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    if len(text) == 8 and text.isdigit():
        return pd.to_datetime(text, format="%Y%m%d")
    return pd.to_datetime(text)


def filter_history_by_date(
    frame: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    start = parse_market_date(start_date)
    end = parse_market_date(end_date)
    if start is not None:
        data = data[data["date"] >= start]
    if end is not None:
        data = data[data["date"] <= end]
    return data.sort_values("date").reset_index(drop=True)


def _asset_subdir(asset_type: str) -> str:
    asset_type = asset_type.lower()
    if asset_type == "etf":
        return ""
    if asset_type.endswith("s"):
        return asset_type
    return f"{asset_type}s"


def cache_paths(config: LabConfig, etf: ETFSpec) -> dict[str, Path]:
    base = config.project.data_dir
    subdir = _asset_subdir(etf.asset_type)
    raw_dir = base / "raw" / subdir if subdir else base / "raw"
    processed_dir = base / "processed" / subdir if subdir else base / "processed"
    meta_dir = base / "meta" / subdir if subdir else base / "meta"
    return {
        "raw": raw_dir / f"{etf.code}.csv",
        "processed": processed_dir / f"{etf.code}.csv",
        "meta": meta_dir / f"{etf.code}.json",
    }


def normalize_akshare_frame(frame: pd.DataFrame, etf: ETFSpec) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"No data returned for {etf.code}.")
    renamed = frame.rename(columns={col: COLUMN_MAP.get(str(col), str(col)) for col in frame.columns})
    missing = {"date", "open", "high", "low", "close"} - set(renamed.columns)
    if missing:
        raise ValueError(f"{etf.code} data missing required columns: {sorted(missing)}")

    normalized = renamed.copy()
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["code"] = etf.code
    normalized["name"] = etf.name
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        else:
            normalized[column] = 0.0

    optional = [col for col in ["amplitude", "pct_change", "change_amount", "turnover_rate"] if col in normalized]
    normalized = normalized[CORE_COLUMNS + optional]
    normalized = normalized.dropna(subset=["date", "open", "high", "low", "close"])
    normalized = normalized[(normalized[["open", "high", "low", "close"]] > 0).all(axis=1)]
    normalized = normalized.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return normalized


def _error_summary(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def _normalize_fetched_history(frame: pd.DataFrame, etf: ETFSpec, source: str) -> pd.DataFrame:
    try:
        normalized = normalize_akshare_frame(frame, etf)
    except ValueError as exc:
        raise AkShareDataError(f"{etf.code} {source} returned unusable data: {_error_summary(exc)}") from exc
    normalized.attrs["source"] = source
    return normalized


def fetch_akshare_history(
    etf: ETFSpec,
    start_date: str,
    end_date: str | None,
    period: str,
    adjust: str,
) -> pd.DataFrame:
    import akshare as ak

    if etf.asset_type == "stock":
        primary_error: BaseException | None = None
        try:
            raw = ak.stock_zh_a_hist(
                symbol=etf.code,
                period=period,
                start_date=start_date,
                end_date=end_date or today_yyyymmdd(),
                adjust=adjust,
            )
            return _normalize_fetched_history(raw, etf, "akshare.stock_zh_a_hist")
        except AkShareDataError as exc:
            primary_error = exc
        except Exception as exc:
            primary_error = exc
            if period != "daily":
                raise AkShareDataError(
                    f"{etf.code} akshare.stock_zh_a_hist failed: {_error_summary(exc)}"
                ) from exc
        if period != "daily":
            raise AkShareDataError(
                f"{etf.code} akshare.stock_zh_a_hist failed: {_error_summary(primary_error)}"
            ) from primary_error
        try:
            market_prefix = "sh" if etf.code.startswith("6") else "sz"
            raw = ak.stock_zh_a_daily(
                symbol=f"{market_prefix}{etf.code}",
                start_date=start_date,
                end_date=end_date or today_yyyymmdd(),
                adjust=adjust,
            )
            return _normalize_fetched_history(raw, etf, "akshare.stock_zh_a_daily")
        except AkShareDataError as fallback_error:
            raise AkShareDataError(
                f"{etf.code} AKShare stock history failed; "
                f"akshare.stock_zh_a_hist: {_error_summary(primary_error)}; "
                f"akshare.stock_zh_a_daily: {_error_summary(fallback_error)}"
            ) from fallback_error
        except Exception as fallback_error:
            raise AkShareDataError(
                f"{etf.code} AKShare stock history failed; "
                f"akshare.stock_zh_a_hist: {_error_summary(primary_error)}; "
                f"akshare.stock_zh_a_daily: {_error_summary(fallback_error)}"
            ) from fallback_error
    elif etf.asset_type == "etf":
        try:
            raw = ak.fund_etf_hist_em(
                symbol=etf.code,
                period=period,
                start_date=start_date,
                end_date=end_date or today_yyyymmdd(),
                adjust=adjust,
            )
        except Exception as exc:
            raise AkShareDataError(f"{etf.code} akshare.fund_etf_hist_em failed: {_error_summary(exc)}") from exc
    else:
        raise ValueError(f"Unsupported asset_type for {etf.code}: {etf.asset_type}")
    return _normalize_fetched_history(raw, etf, history_source(etf))


def fetch_akshare_history_with_retries(
    etf: ETFSpec,
    start_date: str,
    end_date: str | None,
    period: str,
    adjust: str,
    retry_count: int = 3,
    pause_seconds: float = 0.5,
) -> pd.DataFrame:
    last_error: Exception | None = None
    attempts = max(retry_count, 1)
    for attempt in range(1, attempts + 1):
        try:
            return fetch_akshare_history(etf, start_date, end_date, period, adjust)
        except AkShareDataError as exc:  # External data APIs can intermittently fail.
            last_error = exc
            if attempt < attempts:
                time.sleep(pause_seconds * attempt)
    raise AkShareDataError(f"Failed to fetch {etf.code} after {attempts} attempts: {last_error}") from last_error


def history_source(etf: ETFSpec) -> str:
    if etf.asset_type == "stock":
        return "akshare.stock_zh_a_hist"
    if etf.asset_type == "etf":
        return "akshare.fund_etf_hist_em"
    return "unknown"


def _failure_record(etf: ETFSpec, stage: str, exc: BaseException, **extra: object) -> dict[str, str]:
    record = {
        "code": etf.code,
        "name": etf.name,
        "asset_type": etf.asset_type,
        "stage": stage,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    for key, value in extra.items():
        record[key] = "" if value is None else str(value)
    return record


def _universe_failure_record(
    source: UniverseSourceConfig,
    stage: str,
    exc: BaseException,
    **extra: object,
) -> dict[str, str]:
    record = {
        "source_type": source.type,
        "symbol": source.symbol,
        "asset_type": source.asset_type,
        "limit": "" if source.limit is None else str(source.limit),
        "stage": stage,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    for key, value in extra.items():
        record[key] = "" if value is None else str(value)
    return record


def _write_universe_failures(config: LabConfig, records: list[dict[str, str]]) -> None:
    if not records:
        return
    failure_dir = config.project.data_dir / "meta"
    failure_dir.mkdir(parents=True, exist_ok=True)
    failure_path = failure_dir / "universe_source_failures.csv"
    pd.DataFrame(records).to_csv(failure_path, index=False, encoding="utf-8-sig")


def save_history(config: LabConfig, etf: ETFSpec, frame: pd.DataFrame, source: str) -> None:
    paths = cache_paths(config, etf)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths["processed"], index=False, encoding="utf-8")
    frame.to_csv(paths["raw"], index=False, encoding="utf-8")
    meta = {
        "code": etf.code,
        "name": etf.name,
        "asset_type": etf.asset_type,
        "source": source,
        "rows": int(len(frame)),
        "start_date": frame["date"].min().strftime("%Y-%m-%d") if not frame.empty else None,
        "end_date": frame["date"].max().strftime("%Y-%m-%d") if not frame.empty else None,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    paths["meta"].write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cached_history(config: LabConfig, etf: ETFSpec) -> pd.DataFrame:
    processed = cache_paths(config, etf)["processed"]
    if not processed.exists():
        raise FileNotFoundError(f"Missing cached data for {etf.code}: {processed}")
    frame = pd.read_csv(processed, parse_dates=["date"])
    return normalize_akshare_frame(frame, etf)


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"Missing any of columns: {candidates}")


def normalize_constituents_frame(frame: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    code_column = _find_column(frame, ["成分券代码", "品种代码", "stock_code", "代码", "证券代码"])
    name_column = _find_column(frame, ["成分券名称", "品种名称", "stock_name", "名称", "证券简称"])
    normalized = pd.DataFrame(
        {
            "code": (
                frame[code_column]
                .astype(str)
                .str.replace(r"\D", "", regex=True)
                .str[-6:]
                .str.zfill(6)
            ),
            "name": frame[name_column].astype(str),
            "asset_type": asset_type,
        }
    )
    for optional in ["日期", "指数代码", "指数名称", "纳入日期"]:
        if optional in frame.columns:
            normalized[optional] = frame[optional]
    normalized = normalized.dropna(subset=["code"])
    normalized = normalized.drop_duplicates(subset=["code"]).reset_index(drop=True)
    return normalized


def _normalize_code(value: object) -> str:
    return str(value).strip().replace(".0", "").zfill(6)[-6:]


def is_a_share_main_board_code(code: object) -> bool:
    normalized = _normalize_code(code)
    return normalized.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def is_a_share_chinext_code(code: object) -> bool:
    normalized = _normalize_code(code)
    return normalized.startswith(("300", "301"))


def is_a_share_main_or_chinext_code(code: object) -> bool:
    return is_a_share_main_board_code(code) or is_a_share_chinext_code(code)


def _coalesce_columns(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    available = [column for column in candidates if column in frame.columns]
    if not available:
        raise ValueError(f"Missing any of columns: {candidates}")
    return frame[available].bfill(axis=1).iloc[:, 0]


def normalize_a_share_main_board_frame(
    frame: pd.DataFrame,
    asset_type: str = "stock",
    include_chinext: bool = False,
) -> pd.DataFrame:
    codes = _coalesce_columns(frame, ["code", "证券代码", "A股代码", "stock_code", "代码", "证券代码"])
    names = _coalesce_columns(frame, ["name", "证券简称", "A股简称", "stock_name", "名称", "证券简称"])
    normalized = pd.DataFrame(
        {
            "code": codes.map(_normalize_code),
            "name": names.astype(str),
            "asset_type": asset_type,
        }
    )
    if "板块" in frame.columns:
        normalized["board"] = frame["板块"].astype(str)
    if "上市日期" in frame.columns:
        normalized["list_date"] = frame["上市日期"]
    if "A股上市日期" in frame.columns:
        normalized["list_date"] = frame["A股上市日期"]

    if include_chinext:
        normalized = normalized[normalized["code"].map(is_a_share_main_or_chinext_code)]
    else:
        normalized = normalized[normalized["code"].map(is_a_share_main_board_code)]
    if "board" in normalized.columns:
        allowed_board = normalized["board"].str.contains("主板", na=True)
        if include_chinext:
            allowed_board = allowed_board | normalized["board"].str.contains("创业板", na=False)
        normalized = normalized[
            allowed_board
            | normalized["code"].str.startswith(("600", "601", "603", "605"))
        ]
    normalized = normalized.dropna(subset=["code", "name"])
    normalized = normalized.drop_duplicates(subset=["code"]).sort_values("code").reset_index(drop=True)
    return normalized[[column for column in ["code", "name", "asset_type", "board", "list_date"] if column in normalized.columns]]


def universe_cache_path(config: LabConfig, source: UniverseSourceConfig) -> Path:
    return config.project.data_dir / "universe" / f"{source.type}_{source.symbol}.csv"


def fetch_universe_source(source: UniverseSourceConfig) -> pd.DataFrame:
    import akshare as ak

    if source.type == "csindex":
        try:
            raw = ak.index_stock_cons_csindex(symbol=source.symbol)
        except Exception as exc:
            raise UniverseSourceDataError(
                f"{source.type}:{source.symbol} akshare.index_stock_cons_csindex failed: {_error_summary(exc)}"
            ) from exc
    elif source.type == "sina":
        try:
            raw = ak.index_stock_cons(symbol=source.symbol)
        except Exception as exc:
            raise UniverseSourceDataError(
                f"{source.type}:{source.symbol} akshare.index_stock_cons failed: {_error_summary(exc)}"
            ) from exc
    elif source.type in {"ashare_main_board", "a_share_main_board", "ashare_main_chinext", "a_share_main_chinext"}:
        try:
            raw = pd.concat(
                [
                    ak.stock_info_sh_name_code(),
                    ak.stock_info_sz_name_code(),
                ],
                ignore_index=True,
                sort=False,
            )
            normalized = normalize_a_share_main_board_frame(
                raw,
                source.asset_type,
                include_chinext=source.type in {"ashare_main_chinext", "a_share_main_chinext"},
            )
        except Exception as exc:
            raise UniverseSourceDataError(
                f"{source.type}:{source.symbol} A-share universe source failed: {_error_summary(exc)}"
            ) from exc
        if source.limit is not None:
            normalized = normalized.head(source.limit)
        return normalized
    else:
        raise ValueError(f"Unsupported universe source: {source.type}")
    try:
        normalized = normalize_constituents_frame(raw, source.asset_type)
    except ValueError as exc:
        raise UniverseSourceDataError(
            f"{source.type}:{source.symbol} returned unusable constituents: {_error_summary(exc)}"
        ) from exc
    if source.limit is not None:
        normalized = normalized.head(source.limit)
    return normalized


def save_universe(config: LabConfig, source: UniverseSourceConfig, frame: pd.DataFrame) -> None:
    path = universe_cache_path(config, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def load_cached_universe(config: LabConfig, source: UniverseSourceConfig) -> pd.DataFrame:
    path = universe_cache_path(config, source)
    if not path.exists():
        raise FileNotFoundError(f"Missing cached universe: {path}")
    frame = pd.read_csv(path, dtype={"code": str})
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    return frame


def load_cached_main_board_universe(config: LabConfig, asset_type: str = "stock") -> pd.DataFrame:
    source = UniverseSourceConfig(type="ashare_main_board", symbol="all", asset_type=asset_type)
    return load_cached_universe(config, source)


def fetch_chinext_universe(asset_type: str = "stock") -> pd.DataFrame:
    import akshare as ak

    try:
        raw = ak.stock_info_sz_name_code()
        normalized = normalize_a_share_main_board_frame(raw, asset_type, include_chinext=True)
    except Exception as exc:
        raise UniverseSourceDataError(f"chinext universe fallback failed: {_error_summary(exc)}") from exc
    return normalized[normalized["code"].map(is_a_share_chinext_code)].reset_index(drop=True)


def build_main_chinext_universe_from_fallbacks(config: LabConfig, source: UniverseSourceConfig) -> pd.DataFrame:
    main = load_cached_main_board_universe(config, source.asset_type)
    chinext = fetch_chinext_universe(source.asset_type)
    combined = pd.concat([main, chinext], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(subset=["code"]).sort_values("code").reset_index(drop=True)
    if source.limit is not None:
        combined = combined.head(source.limit)
    return combined


def load_universe_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")
    frame = pd.read_csv(path, dtype={"code": str})
    required = {"code", "name"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Universe file {path} missing columns: {sorted(missing)}")
    normalized = frame.copy()
    normalized["code"] = (
        normalized["code"]
        .astype(str)
        .str.replace(r"\D", "", regex=True)
        .str[-6:]
        .str.zfill(6)
    )
    normalized["name"] = normalized["name"].astype(str)
    if "asset_type" not in normalized.columns:
        normalized["asset_type"] = "stock"
    normalized["asset_type"] = normalized["asset_type"].fillna("stock").astype(str).str.lower()
    normalized = normalized.dropna(subset=["code", "name"])
    normalized = normalized[normalized["code"].str.len() == 6]
    normalized = normalized.drop_duplicates(subset=["code"]).reset_index(drop=True)
    return normalized[["code", "name", "asset_type"]]


def resolve_universe(config: LabConfig) -> tuple[ETFSpec, ...]:
    instruments = list(config.universe)
    if config.universe_file is not None:
        file_frame = load_universe_file(config.universe_file)
        instruments.extend(
            ETFSpec(code=str(row.code).zfill(6), name=str(row.name), asset_type=str(row.asset_type).lower())
            for row in file_frame.itertuples(index=False)
        )
    if config.universe_source is not None:
        source = config.universe_source
        failure_records: list[dict[str, str]] = []
        try:
            source_frame = fetch_universe_source(source)
            save_universe(config, source, source_frame)
        except UniverseSourceDataError as source_exc:
            failure_records.append(_universe_failure_record(source, "source_fetch", source_exc))
            if source.type in {"ashare_main_chinext", "a_share_main_chinext"}:
                try:
                    source_frame = build_main_chinext_universe_from_fallbacks(config, source)
                    save_universe(config, source, source_frame)
                    _write_universe_failures(config, failure_records)
                except UNIVERSE_FALLBACK_EXCEPTIONS as fallback_exc:
                    failure_records.append(
                        _universe_failure_record(source, "main_chinext_fallback", fallback_exc)
                    )
                    try:
                        source_frame = load_cached_universe(config, source)
                        _write_universe_failures(config, failure_records)
                    except CACHE_LOAD_EXCEPTIONS as cache_exc:
                        failure_records.append(_universe_failure_record(source, "cache_load", cache_exc))
                        _write_universe_failures(config, failure_records)
                        raise UniverseSourceDataError(
                            f"{source.type}:{source.symbol} universe unavailable; "
                            f"source_fetch: {_error_summary(source_exc)}; "
                            f"main_chinext_fallback: {_error_summary(fallback_exc)}; "
                            f"cache_load: {_error_summary(cache_exc)}"
                        ) from cache_exc
            else:
                try:
                    source_frame = load_cached_universe(config, source)
                    _write_universe_failures(config, failure_records)
                except CACHE_LOAD_EXCEPTIONS as cache_exc:
                    failure_records.append(_universe_failure_record(source, "cache_load", cache_exc))
                    _write_universe_failures(config, failure_records)
                    raise UniverseSourceDataError(
                        f"{source.type}:{source.symbol} universe unavailable; "
                        f"source_fetch: {_error_summary(source_exc)}; "
                        f"cache_load: {_error_summary(cache_exc)}"
                    ) from cache_exc
        instruments.extend(
            ETFSpec(code=str(row.code).zfill(6), name=str(row.name), asset_type=str(row.asset_type).lower())
            for row in source_frame.itertuples(index=False)
        )
    deduped: dict[str, ETFSpec] = {}
    for instrument in instruments:
        deduped.setdefault(instrument.code, instrument)
    return tuple(deduped.values())


def update_data(
    config: LabConfig,
    start_date: str | None = None,
    end_date: str | None = None,
    skip_existing: bool = False,
    retry_count: int = 3,
    pause_seconds: float = 0.5,
    continue_on_error: bool = False,
) -> list[Path]:
    written: list[Path] = []
    failures: list[dict[str, str]] = []
    start = start_date or config.data.start_date
    end = end_date if end_date is not None else config.data.end_date
    for etf in resolve_universe(config):
        paths = cache_paths(config, etf)
        if skip_existing and paths["processed"].exists():
            written.append(paths["processed"])
            continue
        try:
            frame = fetch_akshare_history_with_retries(
                etf,
                start,
                end,
                config.data.period,
                config.data.adjust,
                retry_count=retry_count,
                pause_seconds=pause_seconds,
            )
        except AkShareDataError as exc:
            if not continue_on_error:
                raise
            failures.append(_failure_record(etf, "fetch", exc))
            continue
        save_history(config, etf, frame, source=str(frame.attrs.get("source", history_source(etf))))
        written.append(paths["processed"])
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    if failures:
        failure_dir = config.project.data_dir / "meta"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_path = failure_dir / "update_failures.csv"
        pd.DataFrame(failures).to_csv(failure_path, index=False, encoding="utf-8-sig")
    return written


def load_universe_history(
    config: LabConfig,
    allow_fetch: bool = True,
    skip_missing: bool = False,
) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []
    for etf in resolve_universe(config):
        try:
            frame = load_cached_history(config, etf)
        except CACHE_LOAD_EXCEPTIONS as exc:
            if not allow_fetch:
                if skip_missing:
                    failures.append(_failure_record(etf, "cache_load", exc))
                    continue
                raise
            try:
                frame = fetch_akshare_history_with_retries(
                    etf,
                    config.data.start_date,
                    config.data.end_date,
                    config.data.period,
                    config.data.adjust,
                )
                save_history(config, etf, frame, source=str(frame.attrs.get("source", history_source(etf))))
            except AkShareDataError as fetch_exc:
                if not skip_missing:
                    raise
                failures.append(
                    _failure_record(
                        etf,
                        "fetch_after_cache_load",
                        fetch_exc,
                        cache_error_type=exc.__class__.__name__,
                        cache_error=str(exc),
                    )
                )
                continue
        frame = filter_history_by_date(frame, config.data.start_date, config.data.end_date)
        if not frame.empty:
            histories[etf.code] = frame
    if failures:
        failure_dir = config.project.data_dir / "meta"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_path = failure_dir / "load_failures.csv"
        pd.DataFrame(failures).to_csv(failure_path, index=False, encoding="utf-8-sig")
    return histories
