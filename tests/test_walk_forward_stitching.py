from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_etf_lab.backtest import BacktestResult
from quant_etf_lab.walk_forward import _stitch_oos_equity


def _backtest_result(run_id: str, dates: list[str], equity: list[float]) -> BacktestResult:
    return BacktestResult(
        run_id=run_id,
        run_dir=Path("."),
        equity=pd.DataFrame({"date": pd.to_datetime(dates), "equity": equity}),
        trades=pd.DataFrame(),
        benchmark=pd.DataFrame(),
        monthly_returns=pd.DataFrame(),
        metrics={},
        risk_curve=pd.DataFrame(),
        risk_events=pd.DataFrame(),
        cooldown_events=pd.DataFrame(),
    )


def test_stitch_oos_equity_keeps_each_window_first_day_return() -> None:
    results = [
        _backtest_result("w1", ["2026-01-02", "2026-01-03"], [105.0, 110.0]),
        _backtest_result("w2", ["2026-07-02", "2026-07-03"], [99.0, 120.0]),
    ]

    stitched = _stitch_oos_equity(results, initial_cash=100.0)

    assert stitched["stitched_equity"].iloc[0] == 105.0
    assert stitched["stitched_equity"].iloc[-1] == 132.0
