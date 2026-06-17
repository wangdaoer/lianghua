from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_etf_lab.backtest import run_backtest
from quant_etf_lab.config import load_config
from quant_etf_lab.data import load_universe_history
from quant_etf_lab.portfolio import (
    CurveConfig,
    PortfolioCandidate,
    PortfolioResult,
    PortfolioSourceCandidate,
    SatelliteFilterConfig,
    _source_candidate_config,
    _stitched_metrics as portfolio_stitched_metrics,
    _write_portfolio_walk_forward_report,
    load_portfolio_source_selection_config,
    run_portfolio_combine,
)
from quant_etf_lab.report import write_report
from quant_etf_lab.strategies import build_signal_frames
from quant_etf_lab.walk_forward import (
    ParameterCandidate,
    _candidate_config,
    _load_backtest_result,
    _metrics_row,
    _signal_warmup_start,
    _slice_histories,
    _stitched_metrics as stock_stitched_metrics,
    _trim_signal_frames,
    _write_walk_forward_report,
)


DEFAULT_STABLE_CONFIG = PROJECT_ROOT / "configs" / "ashare_main_chinext_multifactor_stable.yaml"
DEFAULT_STABLE_WF_DIR = PROJECT_ROOT / "outputs" / "walk_forward" / "main_chinext_stable_v2_full_20260613_011322"
DEFAULT_SATELLITE_CONFIG = PROJECT_ROOT / "configs" / "ashare_main_chinext_multifactor_satellite_quality.yaml"
DEFAULT_SATELLITE_WF_DIR = PROJECT_ROOT / "outputs" / "walk_forward" / "main_chinext_satellite_quality_v2_full"
DEFAULT_PORTFOLIO_CONFIG = PROJECT_ROOT / "configs" / "portfolio_core_source_selection_quality_reversal_v1.yaml"
DEFAULT_PORTFOLIO_WF_DIR = (
    PROJECT_ROOT / "outputs" / "portfolio_source_selection" / "main_chinext_portfolio_source_selection_validation6_v1"
)


def _date_text(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid date value: {value}")
    return pd.Timestamp(parsed).strftime("%Y%m%d")


def _display_date(value: Any) -> str:
    return pd.Timestamp(pd.to_datetime(_date_text(value), format="%Y%m%d")).strftime("%Y-%m-%d")


def _latest_summary_row(summary: pd.DataFrame) -> pd.Series:
    if summary.empty or "test_start" not in summary or "test_end" not in summary:
        raise ValueError("Summary must contain test_start and test_end columns.")
    data = summary.copy()
    data["_test_start"] = data["test_start"].map(_date_text)
    data["_test_end"] = data["test_end"].map(_date_text)
    data["_sort"] = pd.to_datetime(data["_test_start"], format="%Y%m%d")
    return data.sort_values("_sort").iloc[-1].drop(labels=["_test_start", "_test_end", "_sort"], errors="ignore")


def _window_prefix(window_name: str) -> str:
    parts = str(window_name).split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse window name: {window_name}")
    return "_".join(parts[:2])


def _with_new_window_end(window_name: str, test_start: Any, test_end: Any) -> str:
    return f"{_window_prefix(window_name)}_{_date_text(test_start)}_{_date_text(test_end)}"


def _resolve_path(project_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def stock_candidate_from_payload(name: str, payload: dict[str, Any]) -> ParameterCandidate:
    return ParameterCandidate(
        name=name,
        factor_weights={str(key): float(value) for key, value in dict(payload.get("factor_weights", {})).items()},
        rebalance_interval=int(payload.get("rebalance_interval", 20)),
        max_positions=int(payload.get("max_positions", 10)),
        risk_overrides=dict(payload.get("risk_overrides", {})),
    )


def portfolio_source_candidate_from_payload(
    name: str,
    payload: dict[str, Any],
    project_root: Path,
) -> PortfolioSourceCandidate:
    allocation = dict(payload.get("allocation", {}))
    filter_raw = dict(allocation.get("satellite_filter", {}))
    satellite_filter = SatelliteFilterConfig(
        enabled=bool(filter_raw.get("enabled", False)),
        ma_window=int(filter_raw.get("ma_window", 60)),
        momentum_window=int(filter_raw.get("momentum_window", 20)),
        min_momentum=float(filter_raw.get("min_momentum", 0.0)),
        max_drawdown=float(filter_raw.get("max_drawdown", 0.15)),
        reduced_drawdown=float(filter_raw.get("reduced_drawdown", 0.08)),
        reduced_scale=float(filter_raw.get("reduced_scale", 0.5)),
        default_scale=float(filter_raw.get("default_scale", 0.0)),
        require_above_ma=bool(filter_raw.get("require_above_ma", True)),
        reallocate_to=str(filter_raw.get("reallocate_to", "core")),
    )
    allocation_name = name.split("__", 1)[1] if "__" in name else name
    allocation_candidate = PortfolioCandidate(
        name=allocation_name,
        weights=dict(allocation.get("weights", {})),
        satellite_filter=satellite_filter,
        regime_overrides=dict(allocation.get("regime_overrides", {})),
    )
    source_path = payload.get("source_path")
    satellite = None
    if source_path:
        satellite = CurveConfig(name="satellite", path=_resolve_path(project_root, source_path), equity_column="stitched_equity")
    return PortfolioSourceCandidate(
        name=name,
        source_name=str(payload.get("source_name", "core_only")),
        satellite=satellite,
        allocation=allocation_candidate,
    )


def replace_last_window_row(summary: pd.DataFrame, new_row: dict[str, Any]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame([new_row])
    data = summary.copy()
    data["_sort"] = pd.to_datetime(data["test_start"].map(_date_text), format="%Y%m%d")
    last_index = data["_sort"].idxmax()
    without_last = data.drop(index=last_index).drop(columns=["_sort"])
    new_frame = pd.DataFrame([new_row])
    for column in without_last.columns:
        if column not in new_frame:
            new_frame[column] = pd.NA
    for column in new_frame.columns:
        if column not in without_last:
            without_last[column] = pd.NA
    return pd.concat([without_last[summary.columns], new_frame[summary.columns]], ignore_index=True)


def _replace_stitched_tail(
    stitched_path: Path,
    new_equity: pd.DataFrame,
    equity_column: str,
    window_name: str,
    initial_cash: float,
    test_start: str,
) -> pd.DataFrame:
    old = pd.read_csv(stitched_path, parse_dates=["date"]) if stitched_path.exists() else pd.DataFrame()
    start_ts = pd.to_datetime(_date_text(test_start), format="%Y%m%d")
    prior = old[old["date"] < start_ts].copy() if not old.empty else pd.DataFrame(columns=["date", "window", "stitched_equity"])
    base_equity = float(prior["stitched_equity"].iloc[-1]) if not prior.empty else float(initial_cash)
    tail = new_equity[["date", equity_column]].copy()
    tail["date"] = pd.to_datetime(tail["date"])
    tail = tail[tail["date"] >= start_ts].dropna(subset=[equity_column]).sort_values("date")
    if tail.empty:
        raise ValueError(f"No new equity rows on or after {_display_date(test_start)}.")
    first = float(tail[equity_column].iloc[0])
    if first <= 0:
        raise ValueError("New equity first value must be positive.")
    tail["stitched_equity"] = pd.to_numeric(tail[equity_column], errors="coerce") / first * base_equity
    tail["window"] = window_name
    combined = pd.concat([prior[["date", "window", "stitched_equity"]], tail[["date", "window", "stitched_equity"]]], ignore_index=True)
    return combined.sort_values("date").reset_index(drop=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _params_path(wf_dir: Path, row: pd.Series) -> Path:
    raw = row.get("selected_params_path")
    if isinstance(raw, str) and raw:
        path = Path(raw)
        if path.exists():
            return path
    return wf_dir / f"{row['window']}_selected_params.json"


def extend_stock_walk_forward(
    config_path: Path,
    wf_dir: Path,
    grid: str,
    objective: str,
    train_years: int,
    test_months: int,
    step_months: int,
    max_train_drawdown: float,
    skip_missing: bool = True,
) -> dict[str, Any]:
    config = load_config(config_path)
    histories = load_universe_history(config, allow_fetch=False, skip_missing=skip_missing)
    if not histories:
        raise ValueError(f"No histories loaded for {config_path}.")
    latest_end = _date_text(max(pd.to_datetime(frame["date"]).max() for frame in histories.values()))
    summary_path = wf_dir / "walk_forward_summary.csv"
    summary = pd.read_csv(summary_path)
    last = _latest_summary_row(summary)
    old_end = _date_text(last["test_end"])
    if latest_end <= old_end:
        return {"status": "up_to_date", "wf_dir": str(wf_dir), "latest_end": latest_end, "old_end": old_end}

    selected_name = str(last["selected_candidate"])
    old_params_path = _params_path(wf_dir, last)
    payload = _read_json(old_params_path)
    candidate = stock_candidate_from_payload(selected_name, payload)
    test_start = _date_text(last["test_start"])
    new_window = _with_new_window_end(str(last["window"]), test_start, latest_end)
    new_params_path = wf_dir / f"{new_window}_selected_params.json"
    shutil.copyfile(old_params_path, new_params_path)

    test_config = _candidate_config(config, candidate, test_start, latest_end)
    test_histories = _slice_histories(histories, test_start, latest_end)
    warmup_start = _signal_warmup_start(config, test_start)
    signal_histories = _slice_histories(histories, warmup_start, latest_end)
    signal_frames = _trim_signal_frames(build_signal_frames(signal_histories, test_config.strategy), test_start, latest_end)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_result = run_backtest(
        test_config,
        histories=test_histories,
        signal_frames=signal_frames,
        run_id=f"walk_forward_extend_{stamp}_{new_window}_{selected_name}_test",
        write_outputs=True,
    )
    write_report(test_result.run_dir)

    new_row = dict(last.to_dict())
    new_row.update(
        {
            "window": new_window,
            "test_end": latest_end,
            "selected_params_path": str(new_params_path),
            "test_run_dir": str(test_result.run_dir),
            **_metrics_row("test", test_result.metrics),
        }
    )
    train_return = pd.to_numeric(pd.Series([new_row.get("train_total_return")]), errors="coerce").iloc[0]
    train_sharpe = pd.to_numeric(pd.Series([new_row.get("train_sharpe")]), errors="coerce").iloc[0]
    if pd.notna(train_return):
        new_row["train_test_return_gap"] = float(train_return) - float(test_result.metrics.get("total_return", 0.0) or 0.0)
    if pd.notna(train_sharpe):
        new_row["train_test_sharpe_gap"] = float(train_sharpe) - float(test_result.metrics.get("sharpe", 0.0) or 0.0)

    updated_summary = replace_last_window_row(summary, new_row)
    candidate_rows = pd.read_csv(wf_dir / "candidate_results.csv") if (wf_dir / "candidate_results.csv").exists() else pd.DataFrame()
    stitched = _replace_stitched_tail(
        wf_dir / "oos_equity_stitched.csv",
        test_result.equity,
        "equity",
        f"{new_window}_{selected_name}_test",
        config.project.initial_cash,
        test_start,
    )
    stitched.to_csv(wf_dir / "oos_equity_stitched.csv", index=False, encoding="utf-8")
    aggregate = stock_stitched_metrics(stitched, config.project.initial_cash)
    _write_walk_forward_report(
        wf_dir,
        updated_summary.to_dict("records"),
        candidate_rows.to_dict("records"),
        aggregate,
        grid,
        objective,
        train_years,
        test_months,
        step_months,
        max_train_drawdown,
        selection_validation_months=0,
    )
    audit = {
        "status": "extended",
        "kind": "stock_walk_forward",
        "wf_dir": str(wf_dir),
        "config": str(config_path),
        "old_window": str(last["window"]),
        "new_window": new_window,
        "old_test_end": old_end,
        "new_test_end": latest_end,
        "selected_candidate": selected_name,
        "old_params_path": str(old_params_path),
        "new_params_path": str(new_params_path),
        "new_test_run_dir": str(test_result.run_dir),
        "stitched_path": str(wf_dir / "oos_equity_stitched.csv"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Extended with prior selected parameters; no retraining was performed.",
    }
    (wf_dir / "latest_extension_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def _load_portfolio_result(run_id: str, run_dir: Path) -> PortfolioResult:
    equity = pd.read_csv(run_dir / "portfolio_equity.csv", parse_dates=["date"])
    allocation = pd.read_csv(run_dir / "allocation_events.csv", parse_dates=["date"]) if (run_dir / "allocation_events.csv").exists() else pd.DataFrame()
    monthly = pd.read_csv(run_dir / "monthly_returns.csv", parse_dates=["date"]) if (run_dir / "monthly_returns.csv").exists() else pd.DataFrame()
    metrics = _read_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
    return PortfolioResult(run_id=run_id, run_dir=run_dir, equity=equity, allocation_events=allocation, monthly_returns=monthly, metrics=metrics)


def extend_portfolio_source_selection(
    config_path: Path,
    wf_dir: Path,
    train_months: int,
    test_months: int,
    step_months: int,
    grid: str,
    max_train_drawdown: float,
) -> dict[str, Any]:
    config, _ = load_portfolio_source_selection_config(config_path)
    summary_path = wf_dir / "portfolio_walk_forward_summary.csv"
    summary = pd.read_csv(summary_path)
    last = _latest_summary_row(summary)
    old_end = _date_text(last["test_end"])

    core_latest = pd.to_datetime(pd.read_csv(config.core.path, usecols=["date"])["date"]).max()
    selected_payload = _read_json(_params_path(wf_dir, last))
    candidate = portfolio_source_candidate_from_payload(str(last["selected_candidate"]), selected_payload, config.project_root)
    if candidate.satellite is None:
        latest_end = _date_text(core_latest)
    else:
        sat_latest = pd.to_datetime(pd.read_csv(candidate.satellite.path, usecols=["date"])["date"]).max()
        latest_end = _date_text(min(core_latest, sat_latest))
    if latest_end <= old_end:
        return {"status": "up_to_date", "wf_dir": str(wf_dir), "latest_end": latest_end, "old_end": old_end}

    test_start = _date_text(last["test_start"])
    new_window = _with_new_window_end(str(last["window"]), test_start, latest_end)
    new_params_path = wf_dir / f"{new_window}_selected_params.json"
    old_params_path = _params_path(wf_dir, last)
    shutil.copyfile(old_params_path, new_params_path)

    test_config = _source_candidate_config(config, candidate, test_start, latest_end)
    test_result = run_portfolio_combine(test_config, run_id=f"{new_window}_{candidate.name}_test", write_outputs=True)

    new_row = dict(last.to_dict())
    new_row.update(
        {
            "window": new_window,
            "test_end": latest_end,
            "selected_params_path": str(new_params_path),
            **{f"test_{key}": value for key, value in test_result.metrics.items()},
        }
    )
    updated_summary = replace_last_window_row(summary, new_row)
    candidate_rows = (
        pd.read_csv(wf_dir / "portfolio_candidate_results.csv")
        if (wf_dir / "portfolio_candidate_results.csv").exists()
        else pd.DataFrame()
    )
    stitched = _replace_stitched_tail(
        wf_dir / "oos_equity_stitched.csv",
        test_result.equity,
        "portfolio_equity",
        f"{new_window}_{candidate.name}_test",
        config.initial_cash,
        test_start,
    )
    stitched.to_csv(wf_dir / "oos_equity_stitched.csv", index=False, encoding="utf-8")
    aggregate = portfolio_stitched_metrics(stitched, config.initial_cash)
    _write_portfolio_walk_forward_report(
        wf_dir,
        updated_summary.to_dict("records"),
        candidate_rows.to_dict("records"),
        aggregate,
        f"source-selection/{grid}",
        "balanced",
        0.0,
        train_months,
        test_months,
        step_months,
        max_train_drawdown,
    )
    audit = {
        "status": "extended",
        "kind": "portfolio_source_selection",
        "wf_dir": str(wf_dir),
        "config": str(config_path),
        "old_window": str(last["window"]),
        "new_window": new_window,
        "old_test_end": old_end,
        "new_test_end": latest_end,
        "selected_candidate": candidate.name,
        "selected_source": candidate.source_name,
        "old_params_path": str(old_params_path),
        "new_params_path": str(new_params_path),
        "new_test_run_dir": str(test_result.run_dir),
        "stitched_path": str(wf_dir / "oos_equity_stitched.csv"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": "Extended with prior selected allocation/source; no source re-selection was performed.",
    }
    (wf_dir / "latest_extension_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def run_default_all() -> list[dict[str, Any]]:
    results = [
        extend_stock_walk_forward(
            DEFAULT_STABLE_CONFIG,
            DEFAULT_STABLE_WF_DIR,
            grid="stable_v2",
            objective="stable",
            train_years=4,
            test_months=6,
            step_months=6,
            max_train_drawdown=0.20,
        ),
        extend_stock_walk_forward(
            DEFAULT_SATELLITE_CONFIG,
            DEFAULT_SATELLITE_WF_DIR,
            grid="satellite_v2",
            objective="satellite",
            train_years=4,
            test_months=6,
            step_months=6,
            max_train_drawdown=0.35,
        ),
    ]
    results.append(
        extend_portfolio_source_selection(
            DEFAULT_PORTFOLIO_CONFIG,
            DEFAULT_PORTFOLIO_WF_DIR,
            train_months=24,
            test_months=6,
            step_months=6,
            grid="activation",
            max_train_drawdown=0.16,
        )
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Extend latest derived walk-forward curves with prior selected parameters.")
    parser.add_argument(
        "target",
        choices=["all", "stable", "satellite", "portfolio"],
        nargs="?",
        default="all",
        help="Derived curve family to extend.",
    )
    args = parser.parse_args()

    if args.target == "all":
        results = run_default_all()
    elif args.target == "stable":
        results = [
            extend_stock_walk_forward(
                DEFAULT_STABLE_CONFIG,
                DEFAULT_STABLE_WF_DIR,
                grid="stable_v2",
                objective="stable",
                train_years=4,
                test_months=6,
                step_months=6,
                max_train_drawdown=0.20,
            )
        ]
    elif args.target == "satellite":
        results = [
            extend_stock_walk_forward(
                DEFAULT_SATELLITE_CONFIG,
                DEFAULT_SATELLITE_WF_DIR,
                grid="satellite_v2",
                objective="satellite",
                train_years=4,
                test_months=6,
                step_months=6,
                max_train_drawdown=0.35,
            )
        ]
    else:
        results = [
            extend_portfolio_source_selection(
                DEFAULT_PORTFOLIO_CONFIG,
                DEFAULT_PORTFOLIO_WF_DIR,
                train_months=24,
                test_months=6,
                step_months=6,
                grid="activation",
                max_train_drawdown=0.16,
            )
        ]

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
