from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
import yaml
from pandas.errors import EmptyDataError

from run_backtest import load_prices, pivot_prices
from run_strong_pullback_satellite import run_satellite_walk_forward
from strong_pullback_evolution import (
    EvolutionConfig,
    assess_test_result,
    build_group_candidates,
    calculate_segment_metrics,
    choose_group_winner,
)
from train_next_open_rank_model import clean_matrix, load_market_exposure


REQUIRED_EVOLUTION_COLUMNS = {
    "date", "symbol", "open", "high", "low", "close", "volume", "amount",
}
FILTER_KEYS = (
    "min_close", "min_avg_amount_20d", "min_pullback_5d", "max_pullback_5d",
    "min_prior_return_20", "min_prior_return_60", "min_return_20d",
    "min_return_60d", "min_distance_ma60", "max_intraday_return",
)
TRIAL_METRIC_KEYS = (
    "total_return", "annualized_return", "max_drawdown", "sharpe_like", "avg_turnover",
    "avg_gross_exposure", "trade_days", "rolling_window_count", "negative_window_rate",
    "worst_rolling_return",
)
TRIAL_ARTIFACTS = {
    "equity_curve.csv": ("date", "equity", "gross_return", "cost", "turnover", "gross_exposure"),
    "rolling_feature_weights.csv": ("date",),
    "trade_audit.csv": ("signal_date",),
    "selected_candidates.csv": ("signal_date",),
}


@dataclass(frozen=True)
class PriceBundle:
    close: pd.DataFrame
    open_px: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    amount: pd.DataFrame
    market_exposure: pd.Series


@dataclass(frozen=True)
class StrategyRun:
    equity: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    candidates: pd.DataFrame


def stable_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RunEvidence:
    data_path: str
    data_size: int
    data_mtime_ns: int
    config_path: str
    config_hash: str
    git_commit: str

    @property
    def fingerprint(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class EvolutionOutcome:
    run_id: str
    run_dir: Path
    champion_id: str
    champion_params: dict[str, object]
    test_status: str
    test_reason: str


def build_run_evidence(
    data_path: Path,
    config_path: Path,
    config: EvolutionConfig,
    git_commit: str,
) -> RunEvidence:
    stat = data_path.resolve().stat()
    return RunEvidence(
        data_path=str(data_path.resolve()),
        data_size=int(stat.st_size),
        data_mtime_ns=int(stat.st_mtime_ns),
        config_path=str(config_path.resolve()),
        config_hash=stable_hash({
            "strategy": config.strategy,
            "periods": asdict(config.periods),
            "baseline": config.baseline,
            "search_groups": [asdict(group) for group in config.search_groups],
            "selection": asdict(config.selection),
        }),
        git_commit=git_commit,
    )


def resolved_config_dict(config: EvolutionConfig) -> dict[str, object]:
    return {
        "strategy": config.strategy,
        "periods": {
            key: value.strftime("%Y-%m-%d") if value is not None else None
            for key, value in asdict(config.periods).items()
        },
        "baseline": dict(config.baseline),
        "search_groups": [
            {
                "id": group.group_id,
                "hypothesis_cn": group.hypothesis_cn,
                "candidates": [
                    {"id": candidate.candidate_id, "overrides": dict(candidate.overrides)}
                    for candidate in group.candidates
                ],
            }
            for group in config.search_groups
        ],
        "selection": asdict(config.selection),
    }


def can_resume_trial(
    trial_state: Mapping[str, object],
    evidence_fingerprint: str,
    params_hash: str,
    trial_id: str,
) -> bool:
    return (
        trial_state.get("status") == "completed"
        and trial_state.get("trial_id") == trial_id
        and trial_state.get("evidence_fingerprint") == evidence_fingerprint
        and trial_state.get("params_hash") == params_hash
    )


def _bundle_has_holdout_dates(bundle: PriceBundle, validation_end: pd.Timestamp) -> bool:
    members = {
        "close": bundle.close,
        "open_px": bundle.open_px,
        "high": bundle.high,
        "low": bundle.low,
        "amount": bundle.amount,
        "market_exposure": bundle.market_exposure,
    }
    return any(
        not member.index.empty
        and pd.Timestamp(member.index.max()) > validation_end
        for member in members.values()
    )


def _has_complete_cached_metrics(metrics: object) -> bool:
    if not isinstance(metrics, Mapping):
        return False
    for segment in ("train", "validation"):
        values = metrics.get(segment)
        if not isinstance(values, Mapping):
            return False
        for key in TRIAL_METRIC_KEYS:
            value = values.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not math.isfinite(float(value)):
                return False
    return True


def _has_complete_trial_artifacts(trial_dir: Path, run: StrategyRun) -> bool:
    frames = {
        "equity_curve.csv": run.equity,
        "rolling_feature_weights.csv": run.weights,
        "trade_audit.csv": run.trades,
        "selected_candidates.csv": run.candidates,
    }
    for filename, required_columns in TRIAL_ARTIFACTS.items():
        frame = frames[filename]
        if not (trial_dir / filename).exists():
            return False
        if not frame.empty and not set(required_columns).issubset(frame.columns):
            return False
    return not run.equity.empty


def validate_input_schema(path: Path) -> None:
    columns = set(pd.read_csv(path, nrows=0).columns)
    missing = REQUIRED_EVOLUTION_COLUMNS - columns
    if missing:
        raise ValueError(f"Missing evolution input columns: {sorted(missing)}")


def load_price_bundle(
    data_path: Path,
    end_date: pd.Timestamp,
    benchmark_path: Path | None,
    params: Mapping[str, object],
) -> PriceBundle:
    validate_input_schema(data_path)
    raw = load_prices(data_path, None, end_date.strftime("%Y-%m-%d"))
    max_abs_return = float(params["max_abs_daily_return"])
    close = clean_matrix(pivot_prices(raw, "close"), max_abs_return)
    open_px = clean_matrix(pivot_prices(raw, "open").reindex_like(close), max_abs_return)
    high = clean_matrix(pivot_prices(raw, "high").reindex_like(close), max_abs_return)
    low = clean_matrix(pivot_prices(raw, "low").reindex_like(close), max_abs_return)
    amount = pivot_prices(raw, "amount").reindex_like(close)
    market_exposure = load_market_exposure(
        str(benchmark_path) if benchmark_path else None,
        close.index,
        ma_window=int(params["market_ma_window"]),
        risk_off_drawdown_20d=float(params["market_risk_off_drawdown_20d"]),
        below_ma_exposure=float(params["market_below_ma_exposure"]),
        crash_exposure=float(params["market_crash_exposure"]),
    )
    if close.index.max() > end_date:
        raise AssertionError("Price bundle extends beyond requested end date")
    return PriceBundle(close, open_px, high, low, amount, market_exposure)


def execute_strategy_trial(bundle: PriceBundle, params: Mapping[str, object]) -> StrategyRun:
    filter_kwargs = {key: float(params[key]) for key in FILTER_KEYS}
    equity, weights, trades, candidates = run_satellite_walk_forward(
        close=bundle.close,
        open_px=bundle.open_px,
        high=bundle.high,
        low=bundle.low,
        amount=bundle.amount,
        train_days=int(params["train_days"]),
        retrain_frequency=int(params["retrain_frequency"]),
        top_n=int(params["top_n"]),
        rebalance_frequency=int(params["rebalance_frequency"]),
        max_position_weight=float(params["max_position_weight"]),
        leverage=float(params["leverage"]),
        min_score=None if params["min_score"] is None else float(params["min_score"]),
        commission_bps=float(params["commission_bps"]),
        impact_bps=float(params["impact_bps"]),
        max_buy_open_gap=float(params["max_buy_open_gap"]),
        limit_buffer=float(params["limit_buffer"]),
        market_exposure=bundle.market_exposure,
        initial_capital=float(params["initial_capital"]),
        filter_kwargs=filter_kwargs,
        basket_guard_return_20d_min=params["basket_guard_return_20d_min"],
        basket_guard_distance_ma60_min=params["basket_guard_distance_ma60_min"],
        basket_guard_scale=float(params["basket_guard_scale"]),
        rebound_exit_return=params["rebound_exit_return"],
        rebound_exit_scale=float(params["rebound_exit_scale"]),
        rebound_exit_market_exposure_max=params["rebound_exit_market_exposure_max"],
        rebound_exit_market_exposure_min=params["rebound_exit_market_exposure_min"],
    )
    if equity.empty:
        raise ValueError("Trial generated no equity rows")
    return StrategyRun(equity, weights, trades, candidates)


def write_trial_artifacts(
    trial_dir: Path,
    run: StrategyRun,
    metrics: Mapping[str, object],
    trial_state: Mapping[str, object],
) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    run.equity.to_csv(trial_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
    run.weights.to_csv(trial_dir / "rolling_feature_weights.csv", index=False, encoding="utf-8-sig")
    run.trades.to_csv(trial_dir / "trade_audit.csv", index=False, encoding="utf-8-sig")
    run.candidates.to_csv(trial_dir / "selected_candidates.csv", index=False, encoding="utf-8-sig")
    (trial_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (trial_dir / "trial_state.json").write_text(json.dumps(trial_state, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_csv_or_empty(path: Path, **kwargs: object) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except EmptyDataError:
        return pd.DataFrame()


def load_trial_artifacts(trial_dir: Path) -> StrategyRun:
    return StrategyRun(
        equity=_read_csv_or_empty(trial_dir / "equity_curve.csv", parse_dates=["date"]),
        weights=_read_csv_or_empty(trial_dir / "rolling_feature_weights.csv"),
        trades=_read_csv_or_empty(trial_dir / "trade_audit.csv"),
        candidates=_read_csv_or_empty(trial_dir / "selected_candidates.csv", dtype={"symbol": "string"}),
    )


def write_chinese_summary(
    run_dir: Path,
    asof_date: pd.Timestamp,
    champion_id: str,
    round_rows: list[dict[str, object]],
    final_metrics: Mapping[str, Mapping[str, float]],
    test_status: str,
    test_reason: str,
) -> Path:
    comparison = pd.DataFrame([
        {
            "版本": name,
            "测试期总收益": values["total_return"],
            "测试期最大回撤": values["max_drawdown"],
            "测试期Sharpe": values["sharpe_like"],
        }
        for name, values in final_metrics.items()
    ])
    lines = [
        f"# 强势回调策略自进化报告 {asof_date:%Y-%m-%d}", "",
        f"- 研究冠军：`{champion_id}`",
        f"- 测试状态：`{test_status}`",
        f"- 判定原因：{test_reason}", "",
        "## 每轮决定", "",
        pd.DataFrame(round_rows).to_markdown(index=False) if round_rows else "本次没有搜索组。", "",
        "## 保留测试期比较", "", comparison.to_markdown(index=False, floatfmt=".4f"), "",
        "## 风险提示", "",
        "该结果仅用于研究和人工复核，不连接券商、不自动下单，也不会自动覆盖现有策略配置。",
    ]
    path = run_dir / f"evolution_summary_{asof_date:%Y%m%d}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_evolution(
    config: EvolutionConfig,
    data_path: Path,
    config_path: Path,
    benchmark_path: Path | None,
    asof_date: pd.Timestamp,
    output_root: Path,
    run_id: str,
    resume: bool,
    bundle_loader: Callable = load_price_bundle,
    trial_executor: Callable = execute_strategy_trial,
    git_commit: str = "unknown",
) -> EvolutionOutcome:
    run_dir = output_root / run_id
    evidence = build_run_evidence(data_path, config_path, config, git_commit)
    manifest_path = run_dir / "manifest.json"
    if resume:
        if not manifest_path.exists():
            raise ValueError(f"Cannot resume missing run: {run_id}")
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("evidence_fingerprint") != evidence.fingerprint:
            raise ValueError("Resume evidence does not match the existing run")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "run_id": run_id,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence": asdict(evidence),
        "evidence_fingerprint": evidence.fingerprint,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyyaml": yaml.__version__,
        },
        "benchmark": str(benchmark_path.resolve()) if benchmark_path else None,
        "asof_date": asof_date.strftime("%Y-%m-%d"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        (run_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(resolved_config_dict(config), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        research_bundle = bundle_loader(
            data_path, config.periods.validation_end, benchmark_path, config.baseline
        )
        if _bundle_has_holdout_dates(research_bundle, config.periods.validation_end):
            raise AssertionError("Research bundle contains holdout dates")

        trial_rows: list[dict[str, object]] = []
        round_rows: list[dict[str, object]] = []
        trial_metrics: dict[str, dict[str, dict[str, float]]] = {}
        trial_params: dict[str, dict[str, object]] = {"baseline": dict(config.baseline)}

        def cached_trial(trial_id: str, params_hash: str) -> StrategyRun | None:
            trial_dir = run_dir / "trials" / trial_id
            state_path = trial_dir / "trial_state.json"
            if not resume or not state_path.exists():
                return None
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if not isinstance(state, Mapping) or not can_resume_trial(
                    state, evidence.fingerprint, params_hash, trial_id
                ):
                    return None
                metrics = json.loads((trial_dir / "metrics.json").read_text(encoding="utf-8"))
                if not _has_complete_cached_metrics(metrics):
                    return None
                run = load_trial_artifacts(trial_dir)
                if not _has_complete_trial_artifacts(trial_dir, run):
                    return None
                trial_metrics[trial_id] = {
                    "train": dict(metrics["train"]),
                    "validation": dict(metrics["validation"]),
                }
                return run
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return None

        def execute_one(trial_id: str, params: dict[str, object]) -> StrategyRun:
            trial_dir = run_dir / "trials" / trial_id
            params_hash = stable_hash(params)
            cached = cached_trial(trial_id, params_hash)
            if cached is not None:
                return cached
            run = trial_executor(research_bundle, params)
            train_metrics = calculate_segment_metrics(
                run.equity, config.periods.research_start, config.periods.train_end,
                config.selection.rolling_window_days,
            )
            validation_metrics = calculate_segment_metrics(
                run.equity, config.periods.validation_start, config.periods.validation_end,
                config.selection.rolling_window_days,
            )
            trial_metrics[trial_id] = {"train": train_metrics, "validation": validation_metrics}
            write_trial_artifacts(
                trial_dir,
                run,
                trial_metrics[trial_id],
                {
                    "status": "completed",
                    "trial_id": trial_id,
                    "params_hash": params_hash,
                    "evidence_fingerprint": evidence.fingerprint,
                },
            )
            return run

        try:
            execute_one("baseline", trial_params["baseline"])
        except Exception as exc:
            raise RuntimeError(f"Baseline trial failed: {exc}") from exc
        trial_rows.append({
            "group_id": "baseline",
            "trial_id": "baseline",
            "parent_id": "",
            "status": "incumbent",
            "reason_cn": "初始基准",
            **{
                f"validation_{key}": value
                for key, value in trial_metrics["baseline"]["validation"].items()
            },
        })

        incumbent_id = "baseline"
        incumbent_params = dict(config.baseline)
        incumbent_metrics = trial_metrics["baseline"]["validation"]
        for group in config.search_groups:
            candidate_pairs: list[tuple[str, Mapping[str, float]]] = []
            for candidate_id, params in build_group_candidates(incumbent_params, group):
                trial_params[candidate_id] = params
                try:
                    execute_one(candidate_id, params)
                    candidate_pairs.append((candidate_id, trial_metrics[candidate_id]["validation"]))
                except Exception as exc:
                    trial_rows.append({
                        "group_id": group.group_id,
                        "trial_id": candidate_id,
                        "status": "trial_error",
                        "reason_cn": str(exc),
                    })
            if not candidate_pairs:
                raise RuntimeError(f"All candidates failed in group {group.group_id}")
            winner_id, decisions = choose_group_winner(
                incumbent_id, incumbent_metrics, tuple(candidate_pairs), config.selection
            )
            for decision in decisions:
                trial_rows.append({
                    "group_id": group.group_id,
                    "trial_id": decision.candidate_id,
                    "parent_id": incumbent_id,
                    "status": "eligible" if decision.promotion.eligible else "rejected",
                    "reason_cn": "通过全部门槛" if decision.promotion.eligible else "；".join(
                        decision.promotion.reasons
                    ),
                    **{f"validation_{key}": value for key, value in decision.metrics.items()},
                    "turnover_ratio": decision.promotion.turnover_ratio,
                    "robust_score": decision.promotion.robust_score,
                })
            promoted = winner_id != incumbent_id
            round_rows.append({
                "group_id": group.group_id,
                "hypothesis_cn": group.hypothesis_cn,
                "parent_id": incumbent_id,
                "winner_id": winner_id,
                "decision": "保留" if promoted else "回滚",
                "reason_cn": "验证期收益优先且全部门槛通过" if promoted else "没有候选通过全部门槛",
            })
            if promoted:
                incumbent_id = winner_id
                incumbent_params = dict(trial_params[winner_id])
                incumbent_metrics = trial_metrics[winner_id]["validation"]

        champion_id = incumbent_id
        champion_params = incumbent_params
        test_end = min(
            asof_date,
            config.periods.test_end if config.periods.test_end is not None else asof_date,
        )
        full_bundle = bundle_loader(data_path, test_end, benchmark_path, config.baseline)
        final_runs = {
            "baseline": trial_executor(full_bundle, config.baseline),
            "champion": trial_executor(full_bundle, champion_params),
        }
        final_metrics = {
            name: calculate_segment_metrics(
                run.equity, config.periods.test_start, test_end, config.selection.rolling_window_days
            )
            for name, run in final_runs.items()
        }
        final_params = {"baseline": dict(config.baseline), "champion": dict(champion_params)}
        for name, run in final_runs.items():
            write_trial_artifacts(
                run_dir / "final" / name,
                run,
                {"test": final_metrics[name]},
                {
                    "status": "completed",
                    "trial_id": f"final_{name}",
                    "params_hash": stable_hash(final_params[name]),
                    "evidence_fingerprint": evidence.fingerprint,
                },
            )
        test_status, test_reason = assess_test_result(
            final_metrics["baseline"], final_metrics["champion"], config.selection
        )

        pd.DataFrame(trial_rows).to_csv(run_dir / "trials.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(round_rows).to_csv(run_dir / "rounds.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([
            {"version": name, **metrics} for name, metrics in final_metrics.items()
        ]).to_csv(run_dir / "test_comparison.csv", index=False, encoding="utf-8-sig")
        (run_dir / "champion_candidate.yaml").write_text(
            yaml.safe_dump(champion_params, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        summary_path = write_chinese_summary(
            run_dir, asof_date, champion_id, round_rows, final_metrics, test_status, test_reason
        )

        manifest.update({
            "status": "success",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "champion_id": champion_id,
            "test_status": test_status,
            "summary": str(summary_path),
        })
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (output_root / "latest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (output_root / "evolution_registry.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        return EvolutionOutcome(run_id, run_dir, champion_id, champion_params, test_status, test_reason)
    except Exception as exc:
        manifest.update({
            "status": "failed",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        })
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
