"""Market sentiment reference reports from local A-share cache."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ETFSpec, LabConfig, load_config
from .data import load_cached_universe, load_universe_history, resolve_universe


_CONFIG_SAMPLE_EXCEPTIONS = (
    FileNotFoundError,
    OSError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
    ValueError,
)


@dataclass(frozen=True)
class MarketSentimentResult:
    output_dir: Path
    timeseries_path: Path
    latest_json_path: Path
    report_path: Path
    latest: dict[str, Any]
    history_count: int


def _limit_threshold(code: str, name: str) -> float:
    text = str(name).upper()
    if "ST" in text:
        return 4.8
    normalized = str(code).zfill(6)
    if normalized.startswith(("300", "301", "688", "689")):
        return 19.8
    return 9.8


def _load_sampled_config(config: LabConfig, max_symbols: int | None) -> LabConfig:
    if max_symbols is None or max_symbols <= 0:
        return config

    if config.universe_source is not None:
        try:
            source_frame = load_cached_universe(config, config.universe_source)
            instruments = tuple(
                ETFSpec(code=str(row.code).zfill(6), name=str(row.name), asset_type=str(row.asset_type).lower())
                for row in source_frame.head(max_symbols).itertuples(index=False)
            )
            return _replace_universe(config, instruments)
        except _CONFIG_SAMPLE_EXCEPTIONS:
            pass
    return _replace_universe(config, resolve_universe(config)[:max_symbols])


def _replace_universe(config: LabConfig, instruments: tuple[ETFSpec, ...]) -> LabConfig:
    return config.__class__(
        project_root=config.project_root,
        project=config.project,
        data=config.data,
        universe=instruments,
        strategy=config.strategy,
        costs=config.costs,
        universe_file=None,
        universe_source=None,
        risk=config.risk,
    )


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling = series.rolling(window, min_periods=max(20, min(window, 60)))
    mean = rolling.mean()
    std = rolling.std().replace(0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_market_sentiment_frame(histories: dict[str, pd.DataFrame], window: int = 120) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for code, history in histories.items():
        if history.empty:
            continue
        frame = history.sort_values("date").copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["code"] = str(code).zfill(6)
        if "name" not in frame:
            frame["name"] = frame["code"]
        frame["daily_return"] = pd.to_numeric(frame["close"], errors="coerce").pct_change() * 100
        threshold = _limit_threshold(frame["code"].iloc[0], str(frame["name"].iloc[0]))
        frame["limit_up"] = frame["daily_return"] >= threshold
        frame["limit_down"] = frame["daily_return"] <= -threshold
        frame["prior_limit_up"] = frame["limit_up"].shift(1).fillna(False).astype(bool)
        frame["prior_limit_down"] = frame["limit_down"].shift(1).fillna(False).astype(bool)
        records.append(
            frame[
                [
                    "date",
                    "code",
                    "daily_return",
                    "limit_up",
                    "limit_down",
                    "prior_limit_up",
                    "prior_limit_down",
                ]
            ]
        )

    if not records:
        return pd.DataFrame()

    panel = pd.concat(records, ignore_index=True)
    panel = panel.dropna(subset=["date", "daily_return"])
    if panel.empty:
        return pd.DataFrame()

    grouped = panel.groupby("date", sort=True)
    rows: list[dict[str, Any]] = []
    for date, group in grouped:
        valid = group["daily_return"].notna()
        coverage = int(valid.sum())
        if coverage == 0:
            continue
        prior_up = group[group["prior_limit_up"] & valid]
        prior_down = group[group["prior_limit_down"] & valid]
        rows.append(
            {
                "date": pd.Timestamp(date),
                "coverage_count": coverage,
                "market_return": float(group.loc[valid, "daily_return"].mean()),
                "advance_ratio": float((group.loc[valid, "daily_return"] > 0).mean()),
                "limit_up_count": int(group.loc[valid, "limit_up"].sum()),
                "limit_down_count": int(group.loc[valid, "limit_down"].sum()),
                "limit_up_ratio": float(group.loc[valid, "limit_up"].mean()),
                "limit_down_ratio": float(group.loc[valid, "limit_down"].mean()),
                "net_limit_ratio": float(group.loc[valid, "limit_up"].mean() - group.loc[valid, "limit_down"].mean()),
                "prior_limit_up_count": int(len(prior_up)),
                "prior_limit_up_premium": float(prior_up["daily_return"].mean()) if len(prior_up) else np.nan,
                "prior_limit_down_count": int(len(prior_down)),
                "prior_limit_down_recovery": float(prior_down["daily_return"].mean()) if len(prior_down) else np.nan,
            }
        )

    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if result.empty:
        return result

    result["prior_limit_up_premium"] = result["prior_limit_up_premium"].fillna(0.0)
    result["prior_limit_down_recovery"] = result["prior_limit_down_recovery"].fillna(0.0)
    result["sentiment_score"] = (
        0.35 * _rolling_zscore(result["net_limit_ratio"], window)
        + 0.30 * _rolling_zscore(result["prior_limit_up_premium"], window)
        + 0.20 * _rolling_zscore(result["market_return"], window)
        + 0.15 * _rolling_zscore(result["advance_ratio"], window)
    )
    result["sentiment_state"] = result["sentiment_score"].map(classify_sentiment_state)
    result["reference_exposure"] = result["sentiment_state"].map(reference_exposure)
    return result


def classify_sentiment_state(score: float) -> str:
    value = float(score)
    if value >= 0.75:
        return "hot"
    if value >= 0.25:
        return "warm"
    if value > -0.25:
        return "neutral"
    if value > -0.75:
        return "weak"
    return "cold"


def reference_exposure(state: str) -> float:
    return {
        "hot": 1.0,
        "warm": 1.0,
        "neutral": 0.8,
        "weak": 0.6,
        "cold": 0.3,
    }.get(str(state), 0.8)


def run_market_sentiment_reference(
    config_path: Path,
    output_dir: Path,
    max_symbols: int | None = None,
    skip_missing: bool = True,
    window: int = 120,
) -> MarketSentimentResult:
    config = _load_sampled_config(load_config(config_path), max_symbols)
    histories = load_universe_history(config, allow_fetch=False, skip_missing=skip_missing)
    sentiment = build_market_sentiment_frame(histories, window=window)
    if sentiment.empty:
        raise ValueError("No local market sentiment data could be computed.")

    output_dir.mkdir(parents=True, exist_ok=True)
    timeseries_path = output_dir / "market_sentiment_timeseries.csv"
    latest_json_path = output_dir / "latest_market_sentiment.json"
    report_path = output_dir / "latest_market_sentiment.md"

    sentiment.to_csv(timeseries_path, index=False, encoding="utf-8-sig")
    latest = sentiment.iloc[-1].to_dict()
    latest["date"] = pd.Timestamp(latest["date"]).strftime("%Y-%m-%d")
    latest["generated_at"] = datetime.now().isoformat(timespec="seconds")
    latest_json_path.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(sentiment, latest, len(histories)), encoding="utf-8")
    return MarketSentimentResult(
        output_dir=output_dir,
        timeseries_path=timeseries_path,
        latest_json_path=latest_json_path,
        report_path=report_path,
        latest=latest,
        history_count=len(histories),
    )


def _pct(value: object) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return ""


def _ratio(value: object) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return ""


def _render_report(sentiment: pd.DataFrame, latest: dict[str, Any], history_count: int) -> str:
    recent = sentiment.tail(10).copy()
    recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
    rows = [
        "| date | state | score | ref_exposure | up | down | net | up_premium | market_return | coverage |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in recent.to_dict("records"):
        rows.append(
            "| {date} | {state} | {score:.3f} | {exposure} | {up} | {down} | {net} | {premium} | {market} | {coverage} |".format(
                date=row["date"],
                state=row["sentiment_state"],
                score=float(row["sentiment_score"]),
                exposure=_ratio(row["reference_exposure"]),
                up=_ratio(row["limit_up_ratio"]),
                down=_ratio(row["limit_down_ratio"]),
                net=_ratio(row["net_limit_ratio"]),
                premium=_pct(row["prior_limit_up_premium"]),
                market=_pct(row["market_return"]),
                coverage=int(row["coverage_count"]),
            )
        )

    return f"""# Market Sentiment State

Generated at: `{latest["generated_at"]}`

Latest local market date: `{latest["date"]}`

This report is a market-state reference only. It does not connect to brokers, place orders, or change any backtest strategy.

## Latest State

| Metric | Value |
| --- | ---: |
| Sentiment state | {latest["sentiment_state"]} |
| Sentiment score | {float(latest["sentiment_score"]):.3f} |
| Reference exposure | {_ratio(latest["reference_exposure"])} |
| Coverage stocks | {int(latest["coverage_count"]):,} |
| Loaded histories | {history_count:,} |
| Market average return | {_pct(latest["market_return"])} |
| Advancing ratio | {_ratio(latest["advance_ratio"])} |
| Limit-up ratio | {_ratio(latest["limit_up_ratio"])} |
| Limit-down ratio | {_ratio(latest["limit_down_ratio"])} |
| Net limit ratio | {_ratio(latest["net_limit_ratio"])} |
| Prior limit-up count | {int(latest["prior_limit_up_count"])} |
| Prior limit-up premium | {_pct(latest["prior_limit_up_premium"])} |
| Prior limit-down count | {int(latest["prior_limit_down_count"])} |
| Prior limit-down recovery | {_pct(latest["prior_limit_down_recovery"])} |

## Recent States

{chr(10).join(rows)}

## Notes

- `prior_limit_up_premium` uses stocks that hit limit-up on the previous trading day and measures their already-known return on the latest trading day.
- This avoids using same-day unknown next-day returns from the original sentiment-timing reproduction.
- `reference_exposure` is an observation label for review, not a trading instruction.
"""
