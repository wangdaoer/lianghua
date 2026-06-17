"""Research diagnostics for completed backtest equity curves."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class DiagnosticsResult:
    run_dir: Path
    output_dir: Path
    returns: pd.DataFrame
    summary: pd.DataFrame
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    pca_components: pd.DataFrame
    metrics: dict[str, Any]
    report_path: Path


@dataclass(frozen=True)
class BatchDiagnosticsResult:
    output_dir: Path
    comparison: pd.DataFrame
    failures: pd.DataFrame
    report_path: Path


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    return float(numerator / denominator)


def _autocorr(series: pd.Series, lag: int = 1) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) <= lag:
        return 0.0
    current = clean.to_numpy(dtype=float)[lag:]
    previous = clean.to_numpy(dtype=float)[:-lag]
    if current.std() == 0 or previous.std() == 0:
        return 0.0
    return float(np.corrcoef(current, previous)[0, 1])


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    return float((equity / running_max - 1.0).min())


def _return_stats(series: pd.Series, prefix: str) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            f"{prefix}_observations": 0.0,
            f"{prefix}_daily_mean": 0.0,
            f"{prefix}_annual_return": 0.0,
            f"{prefix}_annual_volatility": 0.0,
            f"{prefix}_skew": 0.0,
            f"{prefix}_excess_kurtosis": 0.0,
            f"{prefix}_historical_var_5": 0.0,
            f"{prefix}_historical_cvar_5": 0.0,
            f"{prefix}_downside_deviation": 0.0,
            f"{prefix}_positive_day_ratio": 0.0,
            f"{prefix}_lag1_autocorr": 0.0,
            f"{prefix}_max_drawdown": 0.0,
        }

    daily_mean = float(clean.mean())
    daily_std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    centered = clean - daily_mean
    skew = _safe_ratio(float((centered**3).mean()), float(clean.std(ddof=0) ** 3)) if len(clean) > 1 else 0.0
    kurt = _safe_ratio(float((centered**4).mean()), float(clean.std(ddof=0) ** 4)) - 3.0 if len(clean) > 1 else 0.0
    var_5 = float(clean.quantile(0.05))
    tail = clean[clean <= var_5]
    downside = clean.clip(upper=0.0)
    return {
        f"{prefix}_observations": float(len(clean)),
        f"{prefix}_daily_mean": daily_mean,
        f"{prefix}_annual_return": float((1.0 + daily_mean) ** TRADING_DAYS - 1.0),
        f"{prefix}_annual_volatility": float(daily_std * np.sqrt(TRADING_DAYS)),
        f"{prefix}_skew": float(skew) if np.isfinite(skew) else 0.0,
        f"{prefix}_excess_kurtosis": float(kurt) if np.isfinite(kurt) else 0.0,
        f"{prefix}_historical_var_5": var_5,
        f"{prefix}_historical_cvar_5": float(tail.mean()) if not tail.empty else var_5,
        f"{prefix}_downside_deviation": float(np.sqrt((downside**2).mean()) * np.sqrt(TRADING_DAYS)),
        f"{prefix}_positive_day_ratio": float((clean > 0).mean()),
        f"{prefix}_lag1_autocorr": _autocorr(clean, lag=1),
        f"{prefix}_max_drawdown": _max_drawdown_from_returns(clean),
    }


def _newey_west_covariance(x: np.ndarray, residuals: np.ndarray, lags: int) -> np.ndarray:
    xtx_inv = np.linalg.pinv(x.T @ x)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for t in range(len(residuals)):
        row = x[t : t + 1].T
        meat += float(residuals[t] ** 2) * (row @ row.T)

    for lag in range(1, max(lags, 0) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        for t in range(lag, len(residuals)):
            current = x[t : t + 1].T
            previous = x[t - lag : t - lag + 1].T
            cross = float(residuals[t] * residuals[t - lag])
            meat += weight * cross * (current @ previous.T + previous @ current.T)
    cov = xtx_inv @ meat @ xtx_inv
    return np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)


def _market_model(strategy_returns: pd.Series, benchmark_returns: pd.Series, hac_lags: int | None) -> dict[str, float]:
    data = pd.DataFrame({"strategy": strategy_returns, "benchmark": benchmark_returns}).dropna()
    if len(data) < 3:
        return {
            "ols_alpha_daily": 0.0,
            "ols_alpha_annual": 0.0,
            "ols_beta": 0.0,
            "ols_r_squared": 0.0,
            "ols_residual_volatility_annual": 0.0,
            "ols_alpha_t_hac": 0.0,
            "ols_beta_t_hac": 0.0,
            "ols_alpha_p_hac": 1.0,
            "ols_beta_p_hac": 1.0,
            "ols_hac_lags": 0.0,
            "ols_residual_lag1_autocorr": 0.0,
            "ols_durbin_watson": 0.0,
        }

    y = data["strategy"].to_numpy(dtype=float)
    x_raw = data["benchmark"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(data), dtype=float), x_raw])
    coef = np.linalg.pinv(x.T @ x) @ x.T @ y
    fitted = x @ coef
    residuals = y - fitted
    tss = float(((y - y.mean()) ** 2).sum())
    rss = float((residuals**2).sum())
    r_squared = 1.0 - _safe_ratio(rss, tss) if tss else 0.0
    lag_count = hac_lags
    if lag_count is None:
        lag_count = int(max(1, round(4 * (len(data) / 100) ** (2 / 9))))
    lag_count = min(max(int(lag_count), 0), max(len(data) - 2, 0))
    cov = _newey_west_covariance(x, residuals, lag_count)
    standard_errors = np.sqrt(np.maximum(np.diag(cov), 0.0))
    alpha_t = _safe_ratio(float(coef[0]), float(standard_errors[0]))
    beta_t = _safe_ratio(float(coef[1]), float(standard_errors[1]))
    diff = np.diff(residuals)
    durbin_watson = _safe_ratio(float((diff**2).sum()), rss)
    alpha_annual = (1.0 + float(coef[0])) ** TRADING_DAYS - 1.0 if coef[0] > -1 else -1.0
    return {
        "ols_alpha_daily": float(coef[0]),
        "ols_alpha_annual": float(alpha_annual),
        "ols_beta": float(coef[1]),
        "ols_r_squared": float(r_squared),
        "ols_residual_volatility_annual": float(pd.Series(residuals).std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "ols_alpha_t_hac": float(alpha_t),
        "ols_beta_t_hac": float(beta_t),
        "ols_alpha_p_hac": float(2.0 * (1.0 - _normal_cdf(abs(alpha_t)))),
        "ols_beta_p_hac": float(2.0 * (1.0 - _normal_cdf(abs(beta_t)))),
        "ols_hac_lags": float(lag_count),
        "ols_residual_lag1_autocorr": _autocorr(pd.Series(residuals), lag=1),
        "ols_durbin_watson": float(durbin_watson),
    }


def _pca(covariance: pd.DataFrame) -> pd.DataFrame:
    matrix = covariance.to_numpy(dtype=float)
    if matrix.size == 0:
        return pd.DataFrame(columns=["component", "eigenvalue", "explained_variance", "strategy_loading", "benchmark_loading"])
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    total = float(values.sum())
    rows = []
    for idx, eigenvalue in enumerate(values):
        explained = float(eigenvalue / total) if total else 0.0
        vector = vectors[:, idx]
        rows.append(
            {
                "component": idx + 1,
                "eigenvalue": float(eigenvalue),
                "explained_variance": explained,
                "strategy_loading": float(vector[0]) if len(vector) > 0 else 0.0,
                "benchmark_loading": float(vector[1]) if len(vector) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def load_return_frame(
    run_dir: Path,
    equity_column: str = "equity",
    benchmark_column: str = "benchmark_equity",
) -> pd.DataFrame:
    equity_path = run_dir / "equity_curve.csv"
    benchmark_path = run_dir / "benchmark.csv"
    if not equity_path.exists():
        raise FileNotFoundError(f"Missing equity curve: {equity_path}")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Missing benchmark curve: {benchmark_path}")

    equity = pd.read_csv(equity_path)
    benchmark = pd.read_csv(benchmark_path)
    missing_equity = {"date", equity_column} - set(equity.columns)
    missing_benchmark = {"date", benchmark_column} - set(benchmark.columns)
    if missing_equity:
        raise ValueError(f"Equity curve missing columns: {sorted(missing_equity)}")
    if missing_benchmark:
        raise ValueError(f"Benchmark curve missing columns: {sorted(missing_benchmark)}")

    equity = equity[["date", equity_column]].rename(columns={equity_column: "strategy_equity"})
    benchmark = benchmark[["date", benchmark_column]].rename(columns={benchmark_column: "benchmark_equity"})
    equity["date"] = pd.to_datetime(equity["date"])
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    merged = equity.merge(benchmark, on="date", how="inner").sort_values("date")
    merged["strategy_return"] = pd.to_numeric(merged["strategy_equity"], errors="coerce").pct_change()
    merged["benchmark_return"] = pd.to_numeric(merged["benchmark_equity"], errors="coerce").pct_change()
    merged["active_return"] = merged["strategy_return"] - merged["benchmark_return"]
    return merged.dropna(subset=["strategy_return", "benchmark_return"]).reset_index(drop=True)


def build_diagnostics(
    returns: pd.DataFrame,
    hac_lags: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {"strategy_return", "benchmark_return", "active_return"}
    missing = required - set(returns.columns)
    if missing:
        raise ValueError(f"Return frame missing columns: {sorted(missing)}")

    columns = ["strategy_return", "benchmark_return"]
    covariance = returns[columns].cov()
    correlation = returns[columns].corr()
    covariance.index.name = "series"
    correlation.index.name = "series"
    pca_components = _pca(covariance)

    metrics: dict[str, Any] = {}
    metrics.update(_return_stats(returns["strategy_return"], "strategy"))
    metrics.update(_return_stats(returns["benchmark_return"], "benchmark"))
    metrics.update(_return_stats(returns["active_return"], "active"))
    metrics.update(_market_model(returns["strategy_return"], returns["benchmark_return"], hac_lags=hac_lags))

    active_std = float(returns["active_return"].std(ddof=1)) if len(returns) > 1 else 0.0
    metrics["tracking_error_annual"] = float(active_std * np.sqrt(TRADING_DAYS))
    metrics["information_ratio"] = _safe_ratio(float(returns["active_return"].mean() * TRADING_DAYS), metrics["tracking_error_annual"])
    cov_matrix = covariance.to_numpy(dtype=float)
    metrics["covariance_condition_number"] = float(np.linalg.cond(cov_matrix)) if cov_matrix.size else 0.0
    if not pca_components.empty:
        metrics["pca_first_component_explained"] = float(pca_components.iloc[0]["explained_variance"])
    else:
        metrics["pca_first_component_explained"] = 0.0

    summary = pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in sorted(metrics.items())
            if _as_float(value) is not None
        ]
    )
    return summary, covariance, correlation, pca_components, metrics


def _pct(value: Any) -> str:
    number = _as_float(value)
    return "n/a" if number is None else f"{number * 100:.2f}%"


def _num(value: Any, digits: int = 3) -> str:
    number = _as_float(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _write_report(result: DiagnosticsResult) -> Path:
    metrics = result.metrics
    body = f"""# Research Diagnostics

Run directory: `{result.run_dir}`

These diagnostics add linear algebra, statistics, and econometrics checks to the backtest result. They are for research review only and do not place trades, connect to a broker, or provide investment advice.

## Statistical Profile

| Metric | Strategy | Benchmark | Active |
| --- | ---: | ---: | ---: |
| Observations | {_num(metrics.get("strategy_observations"), 0)} | {_num(metrics.get("benchmark_observations"), 0)} | {_num(metrics.get("active_observations"), 0)} |
| Annual return from mean daily return | {_pct(metrics.get("strategy_annual_return"))} | {_pct(metrics.get("benchmark_annual_return"))} | {_pct(metrics.get("active_annual_return"))} |
| Annual volatility | {_pct(metrics.get("strategy_annual_volatility"))} | {_pct(metrics.get("benchmark_annual_volatility"))} | {_pct(metrics.get("active_annual_volatility"))} |
| Historical VaR 5% | {_pct(metrics.get("strategy_historical_var_5"))} | {_pct(metrics.get("benchmark_historical_var_5"))} | {_pct(metrics.get("active_historical_var_5"))} |
| Historical CVaR 5% | {_pct(metrics.get("strategy_historical_cvar_5"))} | {_pct(metrics.get("benchmark_historical_cvar_5"))} | {_pct(metrics.get("active_historical_cvar_5"))} |
| Skew | {_num(metrics.get("strategy_skew"))} | {_num(metrics.get("benchmark_skew"))} | {_num(metrics.get("active_skew"))} |
| Excess kurtosis | {_num(metrics.get("strategy_excess_kurtosis"))} | {_num(metrics.get("benchmark_excess_kurtosis"))} | {_num(metrics.get("active_excess_kurtosis"))} |
| Lag-1 autocorrelation | {_num(metrics.get("strategy_lag1_autocorr"))} | {_num(metrics.get("benchmark_lag1_autocorr"))} | {_num(metrics.get("active_lag1_autocorr"))} |

## Econometric Market Model

OLS model: `strategy_return = alpha + beta * benchmark_return + residual`, with Newey-West/HAC t-statistics.

| Metric | Value |
| --- | ---: |
| Daily alpha | {_pct(metrics.get("ols_alpha_daily"))} |
| Annualized alpha | {_pct(metrics.get("ols_alpha_annual"))} |
| Beta | {_num(metrics.get("ols_beta"))} |
| R-squared | {_pct(metrics.get("ols_r_squared"))} |
| Alpha HAC t-stat | {_num(metrics.get("ols_alpha_t_hac"))} |
| Alpha HAC p-value | {_num(metrics.get("ols_alpha_p_hac"))} |
| Beta HAC t-stat | {_num(metrics.get("ols_beta_t_hac"))} |
| Beta HAC p-value | {_num(metrics.get("ols_beta_p_hac"))} |
| HAC lags | {_num(metrics.get("ols_hac_lags"), 0)} |
| Residual annual volatility | {_pct(metrics.get("ols_residual_volatility_annual"))} |
| Residual lag-1 autocorrelation | {_num(metrics.get("ols_residual_lag1_autocorr"))} |
| Durbin-Watson | {_num(metrics.get("ols_durbin_watson"))} |

## Linear Algebra Diagnostics

| Metric | Value |
| --- | ---: |
| Strategy/benchmark correlation | {_num(result.correlation.loc["strategy_return", "benchmark_return"]) if "strategy_return" in result.correlation.index and "benchmark_return" in result.correlation.columns else "n/a"} |
| Covariance condition number | {_num(metrics.get("covariance_condition_number"))} |
| First PCA component explained variance | {_pct(metrics.get("pca_first_component_explained"))} |
| Tracking error annualized | {_pct(metrics.get("tracking_error_annual"))} |
| Information ratio | {_num(metrics.get("information_ratio"))} |

## Files

- `return_diagnostics.csv`: aligned daily strategy, benchmark, and active returns.
- `diagnostic_summary.csv`: scalar statistics and model diagnostics.
- `covariance_matrix.csv`: return covariance matrix.
- `correlation_matrix.csv`: return correlation matrix.
- `pca_components.csv`: covariance eigenvalues, explained variance, and loadings.
- `diagnostic_metrics.json`: machine-readable metrics.
"""
    report_path = result.output_dir / "diagnostics_report.md"
    report_path.write_text(body, encoding="utf-8")
    return report_path


def run_diagnostics(
    run_dir: Path,
    equity_column: str = "equity",
    benchmark_column: str = "benchmark_equity",
    hac_lags: int | None = None,
    output_dir: Path | None = None,
) -> DiagnosticsResult:
    run_dir = run_dir.resolve()
    output = output_dir.resolve() if output_dir is not None else run_dir / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)

    returns = load_return_frame(run_dir, equity_column=equity_column, benchmark_column=benchmark_column)
    summary, covariance, correlation, pca_components, metrics = build_diagnostics(returns, hac_lags=hac_lags)

    returns.to_csv(output / "return_diagnostics.csv", index=False, encoding="utf-8")
    summary.to_csv(output / "diagnostic_summary.csv", index=False, encoding="utf-8")
    covariance.to_csv(output / "covariance_matrix.csv", encoding="utf-8")
    correlation.to_csv(output / "correlation_matrix.csv", encoding="utf-8")
    pca_components.to_csv(output / "pca_components.csv", index=False, encoding="utf-8")
    (output / "diagnostic_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    placeholder = output / "diagnostics_report.md"
    result = DiagnosticsResult(
        run_dir=run_dir,
        output_dir=output,
        returns=returns,
        summary=summary,
        covariance=covariance,
        correlation=correlation,
        pca_components=pca_components,
        metrics=metrics,
        report_path=placeholder,
    )
    report_path = _write_report(result)
    return DiagnosticsResult(
        run_dir=result.run_dir,
        output_dir=result.output_dir,
        returns=result.returns,
        summary=result.summary,
        covariance=result.covariance,
        correlation=result.correlation,
        pca_components=result.pca_components,
        metrics=result.metrics,
        report_path=report_path,
    )


def _batch_report(output_dir: Path, comparison: pd.DataFrame, failures: pd.DataFrame) -> Path:
    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        if frame.empty:
            return "No rows."
        rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for item in frame[columns].head(30).itertuples(index=False):
            values = []
            for value in item:
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join(rows)

    columns = [
        "run_id",
        "strategy_annual_return",
        "strategy_annual_volatility",
        "strategy_max_drawdown",
        "ols_alpha_annual",
        "ols_beta",
        "ols_r_squared",
        "information_ratio",
        "tracking_error_annual",
    ]
    available = [column for column in columns if column in comparison.columns]
    body = f"""# Batch Diagnostics

This report compares linear algebra, statistics, and econometrics diagnostics across multiple backtest runs. It is research analysis only and does not place trades or provide investment advice.

## Comparison

{table(comparison, available)}

## Failures

{table(failures, list(failures.columns)) if not failures.empty else "No failures."}

## Files

- `diagnostic_compare.csv`: combined selected diagnostics by run.
- `diagnostic_failures.csv`: runs that could not be diagnosed.
- Each successful run also has its own diagnostics folder under this batch directory.
"""
    report_path = output_dir / "batch_diagnostics_report.md"
    report_path.write_text(body, encoding="utf-8")
    return report_path


def run_batch_diagnostics(
    run_dirs: list[Path],
    output_dir: Path,
    equity_column: str = "equity",
    benchmark_column: str = "benchmark_equity",
    hac_lags: int | None = None,
) -> BatchDiagnosticsResult:
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    selected_metrics = [
        "strategy_observations",
        "strategy_annual_return",
        "strategy_annual_volatility",
        "strategy_historical_var_5",
        "strategy_historical_cvar_5",
        "strategy_lag1_autocorr",
        "strategy_max_drawdown",
        "benchmark_annual_return",
        "benchmark_annual_volatility",
        "active_annual_return",
        "active_annual_volatility",
        "ols_alpha_annual",
        "ols_beta",
        "ols_r_squared",
        "ols_alpha_t_hac",
        "ols_alpha_p_hac",
        "ols_beta_t_hac",
        "ols_beta_p_hac",
        "tracking_error_annual",
        "information_ratio",
        "covariance_condition_number",
        "pca_first_component_explained",
    ]
    for run_dir in run_dirs:
        try:
            result = run_diagnostics(
                run_dir,
                equity_column=equity_column,
                benchmark_column=benchmark_column,
                hac_lags=hac_lags,
                output_dir=output / run_dir.name,
            )
        except Exception as exc:  # noqa: BLE001 - batch mode should keep auditing other runs.
            failures.append({"run_id": run_dir.name, "run_dir": str(run_dir), "error": str(exc)})
            continue
        row: dict[str, Any] = {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "diagnostics_dir": str(result.output_dir),
            "report_path": str(result.report_path),
        }
        row.update({key: result.metrics.get(key) for key in selected_metrics})
        rows.append(row)

    comparison = pd.DataFrame(rows)
    failure_frame = pd.DataFrame(failures)
    comparison.to_csv(output / "diagnostic_compare.csv", index=False, encoding="utf-8")
    failure_frame.to_csv(output / "diagnostic_failures.csv", index=False, encoding="utf-8")
    report_path = _batch_report(output, comparison, failure_frame)
    return BatchDiagnosticsResult(
        output_dir=output,
        comparison=comparison,
        failures=failure_frame,
        report_path=report_path,
    )
