import numpy as np
import pandas as pd
import pytest

from legacy_observation_factors import (
    OBSERVATION_SCORE_INPUTS,
    audit_static_universe,
    compute_observation_factors,
    validate_panel,
)


def make_panel(periods=80):
    dates = pd.date_range("2026-01-01", periods=periods, freq="D")
    rows = []
    for symbol, offset in [("000001", 0.0), ("300001", 2.0)]:
        for idx, date in enumerate(dates):
            close = 10.0 + offset + idx * 0.10
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "close": close,
                    "open": close * 0.999,
                    "amount": 10_000_000.0 + idx * 1_000.0,
                }
            )
    return pd.DataFrame(rows)


def make_extreme_future_rows(base):
    last_date = pd.Timestamp(base["date"].max())
    rows = []
    for symbol in sorted(base["symbol"].unique()):
        rows.append(
            {
                "date": last_date + pd.Timedelta(days=1),
                "symbol": symbol,
                "close": 1_000_000.0,
                "open": 1_000_000.0,
                "amount": 1_000_000_000.0,
            }
        )
    return pd.DataFrame(rows)


def make_monotonic_panel(periods=25):
    dates = pd.date_range("2026-01-01", periods=periods, freq="D")
    close = np.arange(10.0, 10.0 + periods)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "close": close,
            "open": close,
            "amount": 10_000_000.0,
        }
    )


def make_return_panel(returns):
    dates = pd.to_datetime(["2026-01-01", "2026-02-01"])
    rows = []
    for symbol, full_period_return in returns.items():
        rows.extend(
            [
                {"date": dates[0], "symbol": symbol, "close": 100.0},
                {
                    "date": dates[1],
                    "symbol": symbol,
                    "close": 100.0 * (1.0 + full_period_return),
                },
            ]
        )
    return pd.DataFrame(rows)


def test_future_rows_cannot_change_existing_observation_factors():
    base = make_panel(periods=80)
    before, _ = compute_observation_factors(base)
    changed = pd.concat([base, make_extreme_future_rows(base)], ignore_index=True)
    after, _ = compute_observation_factors(changed)
    cutoff = base["date"].max()
    cols = [
        "momentum_20",
        "momentum_60",
        "breakout_distance_20",
        "trend_acceleration",
        "observation_score",
    ]
    pd.testing.assert_frame_equal(
        before.loc[before.date.eq(cutoff), cols].reset_index(drop=True),
        after.loc[after.date.eq(cutoff), cols].reset_index(drop=True),
    )


def test_breakout_uses_prior_twenty_day_high():
    panel = make_monotonic_panel(periods=25)
    factors, _ = compute_observation_factors(panel)
    row = factors.iloc[-1]
    prior_high = panel.sort_values("date")["close"].iloc[-21:-1].max()
    assert row["breakout_distance_20"] == pytest.approx(
        row["close"] / prior_high - 1.0
    )


def test_missing_amount_and_money_flow_are_unavailable_not_fabricated():
    factors, meta = compute_observation_factors(
        make_panel(periods=80).drop(columns=["amount"])
    )

    assert factors["liquidity_20"].isna().all()
    assert factors["liquidity_stability_20"].isna().all()
    assert factors["flow_persistence"].isna().all()
    assert meta["factor_availability"]["liquidity_20"] is False
    assert meta["factor_availability"]["liquidity_stability_20"] is False
    assert meta["factor_availability"]["flow_persistence"] is False


def test_point_in_time_money_flow_uses_trailing_positive_day_share():
    panel = make_monotonic_panel(periods=25)
    panel["net_money_flow"] = np.where(np.arange(len(panel)) % 2 == 0, 1.0, -1.0)

    factors, meta = compute_observation_factors(panel)

    expected = (panel["net_money_flow"].iloc[-20:] > 0.0).mean()
    assert factors.iloc[-1]["flow_persistence"] == pytest.approx(expected)
    assert meta["factor_availability"]["flow_persistence"] is True
    assert meta["money_flow_column"] == "net_money_flow"


def test_factor_metadata_separates_historical_from_current_flow_availability():
    panel = make_monotonic_panel(periods=25)
    panel["net_money_flow"] = 1.0
    panel.loc[panel.index[-1], "net_money_flow"] = np.nan

    factors, meta = compute_observation_factors(panel)

    assert meta["factor_availability"]["flow_persistence"] is True
    assert meta["factor_current_availability"]["flow_persistence"] is False
    assert meta["factor_current_complete"]["flow_persistence"] is False
    assert meta["factor_current_coverage"]["flow_persistence"] == 0.0
    assert meta["factor_latest_dates"]["flow_persistence"] == "2026-01-24"
    assert meta["factor_input_latest_dates"]["net_money_flow"] == "2026-01-24"
    assert meta["money_flow_current_available"] is False
    assert meta["money_flow_current_coverage"] == 0.0
    assert factors.iloc[-1]["flow_persistence"] != factors.iloc[-1]["flow_persistence"]


def test_money_flow_source_without_full_window_is_not_factor_available():
    panel = make_monotonic_panel(periods=10)
    panel["net_money_flow"] = 1.0

    factors, meta = compute_observation_factors(panel)

    assert meta["money_flow_column"] == "net_money_flow"
    assert meta["factor_input_latest_dates"]["net_money_flow"] == "2026-01-10"
    assert meta["factor_availability"]["flow_persistence"] is False
    assert meta["factor_latest_dates"]["flow_persistence"] is None
    assert meta["factor_current_availability"]["flow_persistence"] is False
    assert factors["flow_persistence"].isna().all()


def test_output_uses_explicit_next_observed_trading_session_semantics():
    panel = make_panel(periods=80)
    panel["signal_lag_days"] = 1
    panel["earliest_action_date"] = panel["date"] + pd.Timedelta(days=1)

    factors, meta = compute_observation_factors(panel)

    assert factors["research_only"].eq(True).all()
    assert factors["trade_instruction"].eq(False).all()
    assert factors["signal_lag_sessions"].eq(1).all()
    assert "signal_lag_days" not in factors
    friday = factors.loc[factors["date"].eq(pd.Timestamp("2026-01-02"))].iloc[0]
    assert friday["date"].day_name() == "Friday"
    assert "earliest_action_date" not in friday.index
    assert not any("action_date" in column for column in factors.columns)
    assert meta["signal_lag_sessions"] == 1
    assert (
        meta["signal_timing"]
        == "close_t_actionable_no_earlier_than_next_observed_trading_session"
    )


def test_score_rejects_future_execution_limit_and_static_winner_inputs():
    panel = make_panel(periods=80)
    baseline, _ = compute_observation_factors(panel)
    contaminated = panel.assign(
        next_day_open=np.linspace(1.0, 1_000_000.0, len(panel)),
        opening_gap_risk=np.tile([False, True], len(panel) // 2),
        forward_return_5d=np.linspace(-10.0, 10.0, len(panel)),
        limit_up_outcome=np.tile([True, False], len(panel) // 2),
        limit_down_outcome=np.tile([False, True], len(panel) // 2),
        lag_0_membership=np.tile([True, False], len(panel) // 2),
        static_winner=np.tile([False, True], len(panel) // 2),
    )

    scored, _ = compute_observation_factors(contaminated)

    forbidden = {
        "next_day_open",
        "opening_gap_risk",
        "forward_return_5d",
        "limit_up_outcome",
        "limit_down_outcome",
        "lag_0_membership",
        "static_winner",
    }
    assert forbidden.isdisjoint(OBSERVATION_SCORE_INPUTS)
    pd.testing.assert_series_equal(
        baseline["observation_score"], scored["observation_score"]
    )


def test_released_factor_output_drops_unapproved_upstream_columns():
    panel = make_panel(periods=80).assign(
        forward_return_5d=99.0,
        trade_action="BUY",
        arbitrary_vendor_label="winner",
    )

    factors, _ = compute_observation_factors(panel)

    assert {
        "forward_return_5d",
        "trade_action",
        "arbitrary_vendor_label",
    }.isdisjoint(factors.columns)
    assert {"date", "symbol", "close", "observation_score"}.issubset(
        factors.columns
    )


def test_future_open_changes_evaluation_risk_without_changing_observation_score():
    panel = make_panel(periods=80)
    baseline, _ = compute_observation_factors(panel)
    changed = panel.copy()
    symbol_rows = changed.index[changed["symbol"].eq("000001")]
    observation_index = symbol_rows[-2]
    next_index = symbol_rows[-1]
    changed.loc[next_index, "open"] = changed.loc[observation_index, "close"] * 1.10

    evaluated, _ = compute_observation_factors(changed)
    observation_date = changed.loc[observation_index, "date"]
    baseline_row = baseline.loc[
        baseline["symbol"].eq("000001") & baseline["date"].eq(observation_date)
    ].iloc[0]
    evaluated_row = evaluated.loc[
        evaluated["symbol"].eq("000001") & evaluated["date"].eq(observation_date)
    ].iloc[0]

    assert bool(baseline_row["opening_gap_risk"]) is False
    assert bool(evaluated_row["opening_gap_risk"]) is True
    pd.testing.assert_series_equal(
        baseline["observation_score"], evaluated["observation_score"]
    )
    assert "opening_gap_risk" not in OBSERVATION_SCORE_INPUTS


def test_point_in_time_limit_rate_drives_limit_risk_annotations():
    panel = make_panel(periods=80)
    panel["limit_rate"] = 0.05
    for symbol, multiplier in (("000001", 1.05), ("300001", 0.95)):
        rows = panel.index[panel["symbol"].eq(symbol)]
        panel.loc[rows[-1], "close"] = panel.loc[rows[-2], "close"] * multiplier

    factors, meta = compute_observation_factors(panel)
    latest = factors.groupby("symbol", sort=False).tail(1).set_index("symbol")

    assert bool(latest.loc["000001", "limit_up_risk"]) is True
    assert bool(latest.loc["000001", "limit_down_risk"]) is False
    assert bool(latest.loc["300001", "limit_up_risk"]) is False
    assert bool(latest.loc["300001", "limit_down_risk"]) is True
    assert meta["limit_status_coverage"] == "point_in_time_explicit"


def test_board_default_limit_annotations_are_labeled_partial():
    _, meta = compute_observation_factors(make_panel(periods=80))

    assert meta["limit_status_coverage"] == "partial_board_default"
    assert "ST 5%" in meta["unsupported_limit_status"]
    assert "exceptional limits" in meta["unsupported_limit_status"]


def test_validate_panel_rejects_duplicate_date_symbol_rows():
    panel = make_panel(periods=5)
    duplicate = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate date-symbol"):
        validate_panel(duplicate)


@pytest.mark.parametrize("missing", ["date", "symbol", "close"])
def test_validate_panel_rejects_missing_required_columns(missing):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_panel(make_panel(periods=5).drop(columns=[missing]))


def test_validate_panel_sorts_stably_and_normalizes_dates():
    panel = make_panel(periods=5).sample(frac=1.0, random_state=17)
    panel["date"] = panel["date"].astype(str)

    validated = validate_panel(panel)

    expected_pairs = sorted(zip(panel["symbol"], pd.to_datetime(panel["date"])))
    assert list(zip(validated["symbol"], validated["date"])) == expected_pairs
    assert isinstance(validated["date"].dtype, pd.DatetimeTZDtype) is False


def test_validate_panel_normalizes_numeric_stock_identifiers():
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4, freq="D"),
            "symbol": ["000001", "000100", 1, 100.0],
            "close": [10.0, 11.0, 12.0, 13.0],
        }
    )

    validated = validate_panel(panel)

    assert set(validated["symbol"]) == {"000001", "000100"}


def test_validate_panel_rejects_malformed_stock_identifier():
    panel = make_panel(periods=5)
    panel.loc[0, "symbol"] = "not-a-symbol"

    with pytest.raises(ValueError, match="stock identifier"):
        validate_panel(panel)


def test_static_winner_audit_flags_lookthrough_and_reports_overlap():
    broad = make_return_panel(
        {"000001": 0.00, "000002": 0.10, "000003": 0.20, "000004": 0.40}
    )
    selected = broad.loc[broad["symbol"].eq("000004")].copy()

    audit = audit_static_universe(selected, broad)

    assert audit["selected_full_period_return_median"] == pytest.approx(0.40)
    assert audit["broad_full_period_return_median"] == pytest.approx(0.15)
    assert audit["selected_median_percentile"] == pytest.approx(1.0)
    assert audit["selected_return_percentiles"] == pytest.approx([1.0])
    assert audit["selected_winner_overlap_symbols"] == ["000004"]
    assert audit["selected_winner_overlap_count"] == 1
    assert audit["lookthrough_suspect"] is True


def test_static_universe_audit_ignores_terminal_zero_price_placeholders():
    broad = make_return_panel(
        {"000001": 0.00, "000002": 0.10, "000003": 0.20, "000004": 0.40}
    )
    broad = pd.concat(
        [
            broad,
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-01",
                        "symbol": "000001",
                        "close": 0.0,
                        "open": 0.0,
                        "high": 0.0,
                        "low": 0.0,
                        "volume": 0.0,
                        "amount": 0.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    selected = broad.loc[broad["symbol"].eq("000004")].copy()

    audit = audit_static_universe(selected, broad)

    assert audit["broad_full_period_return_median"] == pytest.approx(0.15)
    assert audit["selected_median_percentile"] == pytest.approx(1.0)


def test_static_universe_audit_rejects_interior_zero_price_row():
    broad = make_return_panel(
        {"000001": 0.00, "000002": 0.10, "000003": 0.20, "000004": 0.40}
    )
    for column in ("open", "high", "low"):
        broad[column] = broad["close"]
    broad["volume"] = 1_000.0
    broad["amount"] = broad["close"] * broad["volume"]
    first_row = broad.index[broad["symbol"].eq("000001")][0]
    broad.loc[first_row, ["open", "high", "low", "close", "volume", "amount"]] = 0.0
    selected = broad.loc[broad["symbol"].eq("000004")].copy()

    with pytest.raises(ValueError, match="terminal all-zero"):
        audit_static_universe(selected, broad)


def test_static_universe_audit_rejects_zero_close_with_nonzero_companion():
    broad = make_return_panel(
        {"000001": 0.00, "000002": 0.10, "000003": 0.20, "000004": 0.40}
    )
    broad = pd.concat(
        [
            broad,
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-01",
                        "symbol": "000001",
                        "close": 0.0,
                        "open": 1.0,
                        "high": 0.0,
                        "low": 0.0,
                        "volume": 0.0,
                        "amount": 0.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    selected = broad.loc[broad["symbol"].eq("000004")].copy()

    with pytest.raises(ValueError, match="terminal all-zero"):
        audit_static_universe(selected, broad)


@pytest.mark.parametrize("invalid_close", [np.nan, np.inf, -np.inf, -1.0])
def test_static_universe_audit_rejects_non_placeholder_invalid_broad_closes(
    invalid_close,
):
    broad = make_return_panel(
        {"000001": 0.00, "000002": 0.10, "000003": 0.20, "000004": 0.40}
    )
    selected = broad.loc[broad["symbol"].eq("000004")].copy()
    broad.loc[broad["symbol"].eq("000001"), "close"] = invalid_close

    with pytest.raises(ValueError):
        audit_static_universe(selected, broad)


def test_static_universe_audit_does_not_flag_low_percentile_selection():
    broad = make_return_panel(
        {"000001": 0.00, "000002": 0.10, "000003": 0.20, "000004": 0.40}
    )
    selected = broad.loc[broad["symbol"].eq("000001")].copy()

    audit = audit_static_universe(selected, broad)

    assert audit["selected_median_percentile"] == pytest.approx(0.25)
    assert audit["selected_winner_overlap_count"] == 0
    assert audit["lookthrough_suspect"] is False
