from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from monitor_factor_decay import (
    classify_overall_status,
    compute_daily_factor_competition,
    evaluate_preregistered_replacements,
    incremental_recompute_plan,
    load_incremental_checkpoint,
    run_factor_decay_monitor,
    summarize_factor_competition,
    summarize_factor_decay,
    update_monitor_history,
    update_factor_replacement_preregistration,
)


def _competition_reference(
    factor_panel: pd.DataFrame,
    breadth_state: pd.DataFrame,
    factor_directions: dict[str, float],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    from analyze_factor_selection_alpha import attach_forward_open_returns

    enriched = attach_forward_open_returns(factor_panel, horizons).merge(
        breadth_state.reset_index()[["date", "breadth_gap", "breadth_healthy"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    enriched = enriched.loc[
        enriched["score_eligible"].fillna(False)
        & enriched["breadth_healthy"].fillna(False)
    ]
    rows = []
    for date, group in enriched.groupby("date", sort=True):
        for factor, direction in factor_directions.items():
            signal = pd.to_numeric(group[factor], errors="coerce") * direction
            signal_rank = signal.rank(method="average", pct=True)
            for horizon in horizons:
                forward = pd.to_numeric(
                    group[f"forward_open_return_{horizon}d"], errors="coerce"
                )
                valid = signal.notna() & forward.notna()
                values = pd.DataFrame(
                    {"signal": signal.loc[valid], "forward": forward.loc[valid]}
                ).dropna()
                rank_ic = np.nan
                if (
                    len(values) >= 20
                    and values["signal"].nunique() >= 2
                    and values["forward"].nunique() >= 2
                ):
                    rank_ic = values["signal"].rank().corr(values["forward"].rank())
                top = valid & signal_rank.ge(0.8)
                bottom = valid & signal_rank.le(0.2)
                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "factor": factor,
                        "direction": direction,
                        "horizon": horizon,
                        "observations": int(valid.sum()),
                        "rank_ic": rank_ic,
                        "top_bottom_spread": (
                            float(forward.loc[top].mean() - forward.loc[bottom].mean())
                            if top.any() and bottom.any()
                            else np.nan
                        ),
                        "breadth_gap": float(group["breadth_gap"].iloc[0]),
                    }
                )
    return pd.DataFrame(rows)


def test_vectorized_factor_competition_matches_reference_algorithm():
    rng = np.random.default_rng(20260720)
    dates = pd.bdate_range("2026-01-05", periods=32)
    symbols = [f"{index:06d}" for index in range(30)]
    index = pd.MultiIndex.from_product([symbols, dates], names=["symbol", "date"])
    panel = index.to_frame(index=False)
    panel["open"] = 10.0 * np.exp(
        panel.groupby("symbol", sort=False).cumcount().mul(0.002)
        + rng.normal(0.0, 0.03, len(panel))
    )
    panel["liquidity_stability_20"] = rng.uniform(0.0, 1.0, len(panel))
    panel["momentum_20"] = rng.normal(0.0, 0.2, len(panel))
    panel["score_eligible"] = True
    breadth = pd.DataFrame(
        {
            "breadth_gap": np.linspace(0.01, 0.05, len(dates)),
            "breadth_healthy": True,
        },
        index=dates,
    )
    breadth.index.name = "date"
    directions = {"liquidity_stability_20": 1.0, "momentum_20": -1.0}
    horizons = (5, 10)

    expected = _competition_reference(panel, breadth, directions, horizons)
    actual = compute_daily_factor_competition(
        panel,
        breadth,
        factor_directions=directions,
        horizons=horizons,
    )

    assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)


def test_incremental_checkpoint_matches_full_current_run(tmp_path: Path):
    rng = np.random.default_rng(41)
    dates = pd.bdate_range("2025-11-03", periods=130)
    symbols = [f"{index:06d}" for index in range(30)]
    index = pd.MultiIndex.from_product([symbols, dates], names=["symbol", "date"])
    panel = index.to_frame(index=False)
    sequence = panel.groupby("symbol", sort=False).cumcount()
    panel["close"] = 10.0 * np.exp(
        sequence.mul(0.001) + rng.normal(0.0, 0.02, len(panel))
    )
    panel["open"] = panel["close"] * np.exp(rng.normal(0.0, 0.005, len(panel)))
    panel["high"] = panel[["open", "close"]].max(axis=1) * 1.01
    panel["low"] = panel[["open", "close"]].min(axis=1) * 0.99
    panel["volume"] = rng.uniform(1_000_000, 5_000_000, len(panel))
    panel["amount"] = panel["close"] * panel["volume"]
    previous_asof = dates[-4]
    current_asof = dates[-1]
    previous_panel = panel.loc[panel["date"].le(previous_asof)].copy()
    previous_source = tmp_path / "panel_previous.csv"
    current_source = tmp_path / "panel_current.csv"
    previous_panel.to_csv(previous_source, index=False)
    panel.to_csv(current_source, index=False)
    incremental_dir = tmp_path / "incremental"
    full_dir = tmp_path / "full"

    run_factor_decay_monitor(
        previous_panel,
        asof_date=previous_asof,
        output_dir=incremental_dir,
        selection_start=dates[65],
        selection_end=dates[95],
        source_panel=previous_source,
    )
    checkpoint = load_incremental_checkpoint(
        output_dir=incremental_dir,
        asof_date=current_asof,
        base_panel=previous_source,
    )
    assert checkpoint is not None
    previous_daily, context = checkpoint
    plan = incremental_recompute_plan(panel[["date", "symbol"]], previous_asof)
    recompute_start = pd.Timestamp(plan["recompute_start"])
    breadth_start = pd.Timestamp(plan["breadth_start"])
    tail = panel.loc[panel["date"].ge(pd.Timestamp(plan["raw_start"]))].copy()
    incremental = run_factor_decay_monitor(
        tail,
        asof_date=current_asof,
        output_dir=incremental_dir,
        selection_start=dates[65],
        selection_end=dates[95],
        previous_competition_daily=previous_daily,
        recompute_start=recompute_start,
        breadth_start=breadth_start,
        checkpoint_context=context,
        source_panel=current_source,
    )
    full = run_factor_decay_monitor(
        panel,
        asof_date=current_asof,
        output_dir=full_dir,
        selection_start=dates[65],
        selection_end=dates[95],
        source_panel=current_source,
    )

    assert incremental["calculation_mode"] == "incremental"
    assert full["calculation_mode"] == "full"
    token = current_asof.strftime("%Y%m%d")
    for name in (
        f"factor_replacement_daily_ic_{token}.csv",
        f"factor_decay_monitor_{token}.csv",
        f"factor_replacement_competition_{token}.csv",
        f"liquidity_stability_observation_{token}.csv",
        f"factor_replacement_tracking_{token}.csv",
    ):
        incremental_frame = pd.read_csv(incremental_dir / name)
        full_frame = pd.read_csv(full_dir / name)
        sort_columns = [
            column
            for column in ("date", "factor", "horizon", "symbol")
            if column in incremental_frame
        ]
        if sort_columns:
            incremental_frame = incremental_frame.sort_values(sort_columns).reset_index(drop=True)
            full_frame = full_frame.sort_values(sort_columns).reset_index(drop=True)
        assert_frame_equal(
            incremental_frame,
            full_frame,
            check_exact=False,
            rtol=1e-11,
            atol=1e-11,
        )


def _daily_rows() -> pd.DataFrame:
    rows = []
    for horizon in (5, 10, 20):
        for date in pd.date_range("2023-01-01", periods=12, freq="D"):
            rows.append(
                {
                    "date": date,
                    "factor": "liquidity_stability_20",
                    "horizon": horizon,
                    "rank_ic": 0.10,
                    "top_bottom_spread": 0.01,
                }
            )
        for date in pd.date_range("2026-01-01", periods=12, freq="D"):
            rows.append(
                {
                    "date": date,
                    "factor": "liquidity_stability_20",
                    "horizon": horizon,
                    "rank_ic": -0.05 if horizon != 20 else 0.02,
                    "top_bottom_spread": -0.01,
                }
            )
    return pd.DataFrame(rows)


def test_summary_detects_multi_horizon_direction_reversal():
    summary = summarize_factor_decay(
        _daily_rows(),
        selection_start=pd.Timestamp("2023-01-01"),
        selection_end=pd.Timestamp("2025-12-31"),
        asof_date=pd.Timestamp("2026-02-01"),
        recent_signal_days=12,
    )

    assert summary.set_index("horizon").loc[5, "status"] == "direction_reversal"
    assert summary.set_index("horizon").loc[20, "status"] == "weakened"
    assert classify_overall_status(summary) == "direction_reversal"


def test_history_update_is_idempotent_for_same_date(tmp_path: Path):
    history_path = tmp_path / "factor_decay_monitor_history.csv"
    summary = summarize_factor_decay(
        _daily_rows(),
        selection_start=pd.Timestamp("2023-01-01"),
        selection_end=pd.Timestamp("2025-12-31"),
        asof_date=pd.Timestamp("2026-02-01"),
        recent_signal_days=12,
    )

    first = update_monitor_history(
        history_path, pd.Timestamp("2026-02-01"), summary, "direction_reversal"
    )
    second = update_monitor_history(
        history_path, pd.Timestamp("2026-02-01"), summary, "direction_reversal"
    )

    assert len(first) == 3
    assert len(second) == 3
    assert history_path.exists()


def _competition_rows() -> pd.DataFrame:
    rows = []
    for factor, reference_ic, recent_ic in (
        ("liquidity_stability_20", 0.10, -0.02),
        ("momentum_60", 0.05, 0.08),
    ):
        for horizon in (5, 10, 20):
            for date in pd.date_range("2023-01-01", periods=12, freq="D"):
                rows.append(
                    {
                        "date": date,
                        "factor": factor,
                        "direction": 1.0,
                        "horizon": horizon,
                        "observations": 100,
                        "rank_ic": reference_ic,
                        "top_bottom_spread": reference_ic / 10.0,
                    }
                )
            for date in pd.date_range("2026-01-01", periods=12, freq="D"):
                rows.append(
                    {
                        "date": date,
                        "factor": factor,
                        "direction": 1.0,
                        "horizon": horizon,
                        "observations": 100,
                        "rank_ic": recent_ic,
                        "top_bottom_spread": recent_ic / 10.0,
                    }
                )
    return pd.DataFrame(rows)


def test_factor_competition_shortlists_supported_improvement():
    result = summarize_factor_competition(
        _competition_rows(),
        selection_start=pd.Timestamp("2023-01-01"),
        selection_end=pd.Timestamp("2025-12-31"),
        asof_date=pd.Timestamp("2026-02-01"),
        recent_signal_days=12,
        factor_directions={
            "liquidity_stability_20": 1.0,
            "momentum_60": 1.0,
        },
    ).set_index("factor")

    assert result.loc["liquidity_stability_20", "candidate_status"] == "incumbent"
    assert result.loc["momentum_60", "support_status"] == "supported"
    assert result.loc["momentum_60", "candidate_status"] == "shortlist"
    assert result.loc["momentum_60", "recent_ic_advantage_vs_incumbent"] > 0.0


def test_factor_replacement_preregistration_is_frozen(tmp_path: Path):
    competition = summarize_factor_competition(
        _competition_rows(),
        selection_start=pd.Timestamp("2023-01-01"),
        selection_end=pd.Timestamp("2025-12-31"),
        asof_date=pd.Timestamp("2026-02-01"),
        recent_signal_days=12,
        factor_directions={
            "liquidity_stability_20": 1.0,
            "momentum_60": 1.0,
        },
    )
    path = tmp_path / "factor_replacement_preregistration.json"

    first = update_factor_replacement_preregistration(
        path,
        competition,
        asof_date=pd.Timestamp("2026-02-01"),
        incumbent_status="direction_reversal",
    )
    changed = competition.copy()
    changed.loc[changed["factor"].eq("momentum_60"), "candidate_status"] = "not_shortlisted"
    second = update_factor_replacement_preregistration(
        path,
        changed,
        asof_date=pd.Timestamp("2026-03-01"),
        incumbent_status="direction_reversal",
    )

    assert first is not None
    assert second is not None
    assert first == second
    assert first["candidate_factors"] == ["momentum_60"]
    assert first["validation_start_date"] == "2026-02-02"
    assert first["automatic_model_change"] is False


def test_preregistered_tracking_uses_only_future_mature_samples():
    preregistration = {
        "validation_start_date": "2026-07-15",
        "incumbent_factor": "liquidity_stability_20",
        "candidate_factors": ["momentum_60"],
        "horizons": [5, 10],
        "target_mature_signal_days": 2,
        "gates": {
            "min_mean_rank_ic": 0.01,
            "min_positive_ic_ratio": 0.55,
            "min_rank_ic_advantage_vs_incumbent": 0.0,
        },
    }
    rows = []
    for factor, rank_ic in (
        ("liquidity_stability_20", 0.02),
        ("momentum_60", 0.08),
    ):
        for horizon in (5, 10):
            rows.append(
                {
                    "date": pd.Timestamp("2026-07-14"),
                    "factor": factor,
                    "horizon": horizon,
                    "rank_ic": -1.0,
                }
            )
            for date in pd.to_datetime(["2026-07-15", "2026-07-16"]):
                rows.append(
                    {
                        "date": date,
                        "factor": factor,
                        "horizon": horizon,
                        "rank_ic": rank_ic,
                    }
                )

    tracking_rows, result = evaluate_preregistered_replacements(
        pd.DataFrame(rows), preregistration
    )

    candidate = next(
        row for row in result["factors"] if row["factor"] == "momentum_60"
    )
    assert result["status"] == "ready_for_research_review"
    assert candidate["minimum_mature_signal_days"] == 2
    assert candidate["mean_rank_ic"] == 0.08
    assert candidate["decision"] == "ready_for_research_review"
    assert len(tracking_rows) == 4
