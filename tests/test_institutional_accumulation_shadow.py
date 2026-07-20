import json
from pathlib import Path

import numpy as np
import pandas as pd

from institutional_accumulation_shadow import (
    AccumulationConfig,
    build_latest_watchlist,
    build_metadata,
    compute_accumulation_features,
    load_panel,
    write_outputs,
)


def make_panel(*, with_flow: bool = True) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=100, freq="B")
    rows = []
    for symbol, flow in (("000001", 0.004), ("000002", -0.002)):
        close = 10.0 + np.arange(len(dates)) * (0.012 if symbol == "000001" else 0.005)
        amount = np.full(len(dates), 100_000_000.0)
        if symbol == "000001":
            amount[-5:] = 140_000_000.0
        for index, date in enumerate(dates):
            flow_available = with_flow and index >= len(dates) - 5
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close[index] * 0.998,
                    "high": close[index] * 1.015,
                    "low": close[index] * 0.985,
                    "close": close[index],
                    "amount": amount[index],
                    "main_net_volume_ratio": flow if flow_available else np.nan,
                    "main_net_inflow": amount[index] * flow if flow_available else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_composite_signal_requires_price_volume_and_flow_confirmation():
    config = AccumulationConfig()
    features = compute_accumulation_features(make_panel(), config)
    table = build_latest_watchlist(
        features,
        asof_date="2026-05-21",
        name_map={"000001": "测试股份", "000002": "对照股份"},
    )

    row = table.set_index("symbol").loc["000001"]
    assert bool(row["price_setup"])
    assert bool(row["volume_setup"])
    assert bool(row["flow_evidence_available"])
    assert bool(row["flow_confirmed"])
    assert bool(row["signal_active"])
    assert row["institutional_accumulation_level"] in {"建仓观察", "强建仓迹象"}
    assert row["institutional_accumulation_score"] >= config.watch_score


def test_missing_flow_is_explicit_warmup_not_active_signal():
    config = AccumulationConfig()
    features = compute_accumulation_features(make_panel(with_flow=False), config)
    table = build_latest_watchlist(features, asof_date="2026-05-21")

    row = table.set_index("symbol").loc["000001"]
    assert bool(row["tracking_eligible"])
    assert not bool(row["flow_evidence_available"])
    assert not bool(row["signal_active"])
    assert row["institutional_accumulation_level"] == "资金流预热观察"
    assert "暂不确认机构属性" in row["accumulation_reason"]


def test_st_name_is_excluded_from_shadow_watchlist():
    features = compute_accumulation_features(make_panel(with_flow=False))
    table = build_latest_watchlist(
        features,
        asof_date="2026-05-21",
        name_map={"000001": "*ST测试"},
    )

    assert "000001" not in table["symbol"].tolist()


def test_already_overheated_daily_move_is_excluded():
    panel = make_panel()
    target = panel["symbol"].eq("000001")
    last_date = panel.loc[target, "date"].max()
    last = target & panel["date"].eq(last_date)
    previous_close = panel.loc[target & panel["date"].lt(last_date), "close"].iloc[-1]
    panel.loc[last, ["open", "high", "low", "close"]] = [
        previous_close * 1.08,
        previous_close * 1.12,
        previous_close * 1.07,
        previous_close * 1.10,
    ]

    features = compute_accumulation_features(panel)
    latest = features[features["date"].eq(last_date)].set_index("symbol")

    assert latest.loc["000001", "return_1d"] > 0.07
    assert not bool(latest.loc["000001", "price_setup"])
    assert not bool(latest.loc["000001", "signal_active"])


def test_future_row_change_does_not_change_prior_features():
    panel = make_panel()
    baseline = compute_accumulation_features(panel)
    changed = panel.copy()
    last_date = changed["date"].max()
    changed.loc[changed["date"].eq(last_date), "close"] = 999.0
    changed.loc[changed["date"].eq(last_date), "main_net_volume_ratio"] = 0.99
    evaluated = compute_accumulation_features(changed)
    compare_date = sorted(panel["date"].unique())[-2]
    columns = [
        "symbol",
        "return_20d",
        "amount_ratio_5d",
        "main_net_volume_ratio_5d",
        "institutional_accumulation_score",
    ]
    left = baseline[baseline["date"].eq(compare_date)][columns].sort_values("symbol").reset_index(drop=True)
    right = evaluated[evaluated["date"].eq(compare_date)][columns].sort_values("symbol").reset_index(drop=True)

    pd.testing.assert_frame_equal(left, right)


def test_metadata_and_outputs_preserve_research_boundary(tmp_path: Path):
    config = AccumulationConfig()
    features = compute_accumulation_features(make_panel())
    table = build_latest_watchlist(features, asof_date="2026-05-21")
    metadata = build_metadata(
        features,
        table,
        requested_asof_date="2026-05-21",
        availability={"main_net_volume_ratio": True, "main_net_inflow": True},
        config=config,
    )
    output = tmp_path / "institutional_accumulation_shadow_20260521.csv"

    paths = write_outputs(table, metadata, output)

    assert metadata["status"] == "research_ready"
    assert metadata["promotion_allowed"] is False
    assert metadata["selection_effect"] is False
    assert all(path.exists() for path in paths.values())
    saved = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert saved["registration_id"] == "institutional_accumulation_shadow_v1"
    assert saved["schema_version"] == 3
    assert len(saved["implementation_sha256"]) == 64
    chinese = pd.read_csv(paths["chinese_csv"])
    assert "疑似建仓评分" in chinese.columns


def test_latest_source_availability_does_not_use_historical_non_null_values(tmp_path: Path):
    panel = make_panel().copy()
    latest_date = panel["date"].max()
    panel.loc[panel["date"].eq(latest_date), "main_net_inflow"] = np.nan
    path = tmp_path / "panel.csv"
    panel.to_csv(path, index=False)

    loaded, availability = load_panel(path, asof_date=latest_date)
    features = compute_accumulation_features(loaded)
    table = build_latest_watchlist(features, asof_date=str(latest_date.date()))
    metadata = build_metadata(
        features,
        table,
        requested_asof_date=str(latest_date.date()),
        availability=availability,
        config=AccumulationConfig(),
    )

    assert availability == {
        "main_net_volume_ratio": True,
        "main_net_inflow": False,
    }
    assert metadata["source_columns_available"] == availability
    assert metadata["source_columns_historical_available"]["main_net_inflow"] is True
    assert metadata["source_column_latest_dates"]["main_net_inflow"] == (
        latest_date - pd.offsets.BDay(1)
    ).strftime("%Y-%m-%d")
    assert metadata["source_column_latest_dates"]["main_net_volume_ratio"] == latest_date.strftime(
        "%Y-%m-%d"
    )
