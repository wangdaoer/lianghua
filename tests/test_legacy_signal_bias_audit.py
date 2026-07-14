from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

import run_legacy_signal_bias_audit as audit_module

from run_legacy_signal_bias_audit import (
    AuditCase,
    build_ablation_cases,
    build_audit_verdict,
    load_raw_yaml,
    main,
    run_ablation_case,
)


BASE_CONFIG = Path("configs/high_risk_strategy_high_return_v2_dynamic_universe.yaml")


def test_git_state_marks_non_repository_snapshot_unavailable(tmp_path):
    revision, dirty = audit_module._git_state(tmp_path)

    assert revision == "unavailable"
    assert dirty is True


def test_load_raw_yaml_rejects_non_mapping(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("- list item\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected mapping config"):
        load_raw_yaml(path)


def test_ablation_cases_change_only_the_named_assumption():
    base = load_raw_yaml(BASE_CONFIG)
    original = deepcopy(base)

    cases = build_ablation_cases(base)
    by_id = {case.case_id: case for case in cases}

    assert [case.case_id for case in cases] == [
        "dynamic_same_close",
        "dynamic_lag1_close",
        "dynamic_lag1_next_open",
        "dynamic_lag1_next_open_limits",
        "dynamic_lag1_next_open_limits_gap3",
        "dynamic_lag1_next_open_limits_gap3_cost2x",
    ]
    assert by_id["dynamic_same_close"].changed_assumption == "legacy_bias_control"
    assert by_id["dynamic_same_close"].config["universe"]["selection_lag_days"] == 0
    assert by_id["dynamic_lag1_close"].config["universe"]["selection_lag_days"] == 1
    assert by_id["dynamic_lag1_next_open"].config["execution"]["model"] == "next_open"
    assert (
        by_id["dynamic_lag1_next_open_limits"].config["execution"][
            "block_limit_up_buys"
        ]
        is True
    )
    assert (
        by_id["dynamic_lag1_next_open_limits_gap3"].config["execution"][
            "max_buy_open_gap"
        ]
        == 0.03
    )
    assert (
        by_id["dynamic_lag1_next_open_limits_gap3_cost2x"].config["cost"][
            "commission_bps"
        ]
        == base["cost"]["commission_bps"] * 2
    )
    assert (
        by_id["dynamic_lag1_next_open_limits_gap3_cost2x"].config["cost"][
            "impact_bps"
        ]
        == base["cost"]["impact_bps"] * 2
    )

    for case in cases:
        assert case.config["signal"] == base["signal"]
        assert case.config["portfolio"] == base["portfolio"]
        assert case.config["risk"] == base["risk"]
    assert all(
        case.config["universe"]["selection_lag_days"] == 1
        for case in cases[1:]
    )

    expected_changes = [
        {"universe.selection_lag_days"},
        {"execution.model"},
        {
            "execution.block_limit_up_buys",
            "execution.block_limit_down_sells",
        },
        {"execution.max_buy_open_gap"},
        {"cost.commission_bps", "cost.impact_bps"},
    ]
    for previous, current, expected in zip(cases, cases[1:], expected_changes):
        assert _changed_paths(previous.config, current.config) == expected
    assert base == original


def _changed_paths(left, right, prefix=""):
    paths = set()
    for key in set(left) | set(right):
        path = f"{prefix}.{key}" if prefix else key
        if key not in left or key not in right:
            paths.add(path)
        elif isinstance(left[key], dict) and isinstance(right[key], dict):
            paths.update(_changed_paths(left[key], right[key], path))
        elif left[key] != right[key]:
            paths.add(path)
    return paths


def _make_panel(periods=130):
    dates = pd.date_range("2025-01-02", periods=periods, freq="B")
    rows = []
    for symbol, daily_return in (("000001", 0.002), ("000002", 0.001)):
        close = 10.0 * np.power(1.0 + daily_return, np.arange(periods))
        for date, value in zip(dates, close):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": value * 1.001,
                    "high": value * 1.01,
                    "low": value * 0.99,
                    "close": value,
                    "volume": 1_000_000.0,
                    "amount": value * 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def _run_skip_audit(tmp_path, selected, label):
    broad = selected.copy()
    extra = selected[selected["symbol"].eq("000001")].copy()
    extra["symbol"] = "000003"
    extra["close"] *= 0.8
    extra["open"] *= 0.8
    broad = pd.concat([broad, extra], ignore_index=True)

    broad_path = tmp_path / f"broad_{label}.csv"
    selected_path = tmp_path / f"selected_{label}.csv"
    output_dir = tmp_path / f"audit_{label}"
    broad.to_csv(broad_path, index=False)
    selected.to_csv(selected_path, index=False)
    main(
        [
            "--broad-data",
            str(broad_path),
            "--selected-data",
            str(selected_path),
            "--base-config",
            str(BASE_CONFIG),
            "--output-dir",
            str(output_dir),
            "--skip-backtests",
        ]
    )
    return output_dir


def test_unmarked_money_flow_cannot_change_clean_observation_score(tmp_path):
    without_flow = _make_panel(85)
    with_unmarked_flow = without_flow.copy()
    with_unmarked_flow["net_money_flow"] = np.where(
        with_unmarked_flow["symbol"].eq("000001"), -1.0, 1.0
    )

    without_dir = _run_skip_audit(tmp_path, without_flow, "without_flow")
    unmarked_dir = _run_skip_audit(
        tmp_path, with_unmarked_flow, "unmarked_flow"
    )

    without_history = pd.read_csv(
        without_dir / "observation_factor_history.csv"
    )
    unmarked_history = pd.read_csv(
        unmarked_dir / "observation_factor_history.csv"
    )
    pd.testing.assert_series_equal(
        unmarked_history["observation_score"],
        without_history["observation_score"],
        check_names=False,
    )
    assert unmarked_history["flow_persistence"].isna().all()

    dictionary = json.loads(
        (unmarked_dir / "observation_factor_dictionary.json").read_text(
            encoding="utf-8"
        )
    )
    assert dictionary["money_flow_provenance"] == "unverified"
    assert dictionary["factors"]["flow_persistence"]["available"] is False
    assert (
        dictionary["factors"]["flow_persistence"]["clean_factor_claim"]
        is False
    )
    audit = json.loads(
        (unmarked_dir / "legacy_signal_bias_audit.json").read_text(
            encoding="utf-8"
        )
    )
    selected_metadata = audit["inputs"]["selected_data"]
    assert selected_metadata["money_flow_provenance"] == "unverified"
    assert selected_metadata["clean_money_flow_evidence"] is False


def test_explicit_raw_point_in_time_money_flow_is_retained(tmp_path):
    selected = _make_panel(85)
    selected["net_money_flow"] = np.where(
        selected["symbol"].eq("000001"), -1.0, 1.0
    )
    selected["money_flow_provenance"] = "raw_point_in_time"

    output_dir = _run_skip_audit(tmp_path, selected, "raw_flow")

    history = pd.read_csv(output_dir / "observation_factor_history.csv")
    assert history["flow_persistence"].notna().any()
    dictionary = json.loads(
        (output_dir / "observation_factor_dictionary.json").read_text(
            encoding="utf-8"
        )
    )
    assert dictionary["money_flow_provenance"] == "raw_point_in_time"
    assert dictionary["factors"]["flow_persistence"]["available"] is True
    assert dictionary["factors"]["flow_persistence"]["clean_factor_claim"] is True
    audit = json.loads(
        (output_dir / "legacy_signal_bias_audit.json").read_text(encoding="utf-8")
    )
    assert audit["inputs"]["selected_data"]["clean_money_flow_evidence"] is True


@pytest.mark.parametrize(
    ("provenance", "expected_status"),
    [
        ("derived_vendor_field", "derived"),
        ("substituted_from_summary", "substituted"),
        ("vendor_adjusted", "unverified"),
        ("mixed", "mixed"),
    ],
)
def test_non_raw_money_flow_provenance_is_neutralized(
    tmp_path, provenance, expected_status
):
    selected = _make_panel(85)
    selected["net_money_flow"] = np.where(
        selected["symbol"].eq("000001"), -1.0, 1.0
    )
    if provenance == "mixed":
        selected["money_flow_provenance"] = np.where(
            selected.index % 2 == 0,
            "raw_point_in_time",
            "derived_vendor_field",
        )
    else:
        selected["money_flow_provenance"] = provenance

    output_dir = _run_skip_audit(tmp_path, selected, expected_status)

    history = pd.read_csv(output_dir / "observation_factor_history.csv")
    assert history["flow_persistence"].isna().all()
    dictionary = json.loads(
        (output_dir / "observation_factor_dictionary.json").read_text(
            encoding="utf-8"
        )
    )
    assert dictionary["money_flow_provenance"] == expected_status
    assert dictionary["factors"]["flow_persistence"]["available"] is False
    assert (
        dictionary["factors"]["flow_persistence"]["clean_factor_claim"]
        is False
    )


def test_per_column_provenance_must_cover_every_recognized_flow_column(tmp_path):
    selected = _make_panel(85)
    selected["net_money_flow"] = 1.0
    selected["money_flow"] = -1.0
    selected["net_money_flow_provenance"] = "raw_point_in_time"

    output_dir = _run_skip_audit(tmp_path, selected, "partial_flow_provenance")

    history = pd.read_csv(output_dir / "observation_factor_history.csv")
    assert history["flow_persistence"].isna().all()
    dictionary = json.loads(
        (output_dir / "observation_factor_dictionary.json").read_text(
            encoding="utf-8"
        )
    )
    assert dictionary["money_flow_provenance"] == "unverified"
    assert dictionary["factors"]["flow_persistence"]["available"] is False


def test_raw_flow_provenance_without_usable_values_is_not_clean_evidence(
    tmp_path,
):
    selected = _make_panel(85)
    selected["net_money_flow"] = np.nan
    selected["money_flow_provenance"] = "raw_point_in_time"

    output_dir = _run_skip_audit(tmp_path, selected, "empty_raw_flow")

    audit = json.loads(
        (output_dir / "legacy_signal_bias_audit.json").read_text(encoding="utf-8")
    )
    assert audit["inputs"]["selected_data"]["clean_money_flow_evidence"] is False
    dictionary = json.loads(
        (output_dir / "observation_factor_dictionary.json").read_text(
            encoding="utf-8"
        )
    )
    assert dictionary["factors"]["flow_persistence"]["available"] is False
    assert (
        dictionary["factors"]["flow_persistence"]["clean_factor_claim"]
        is False
    )


@pytest.mark.parametrize("invalid_amount", ["all_null", "non_numeric", "zero"])
def test_unusable_present_amount_is_not_clean_liquidity_evidence(
    tmp_path, invalid_amount
):
    selected = _make_panel(85)
    if invalid_amount == "all_null":
        selected["amount"] = np.nan
    elif invalid_amount == "non_numeric":
        selected["amount"] = "not-a-number"
    else:
        selected["amount"] = 0.0
    selected["amount_source"] = "raw_point_in_time"

    output_dir = _run_skip_audit(tmp_path, selected, invalid_amount)

    audit = json.loads(
        (output_dir / "legacy_signal_bias_audit.json").read_text(encoding="utf-8")
    )
    selected_metadata = audit["inputs"]["selected_data"]
    assert selected_metadata["amount_provenance"] == "unavailable"
    assert selected_metadata["clean_liquidity_evidence"] is False

    dictionary = json.loads(
        (output_dir / "observation_factor_dictionary.json").read_text(
            encoding="utf-8"
        )
    )
    assert dictionary["amount_provenance"] == "unavailable"
    assert dictionary["factors"]["liquidity_20"]["available"] is False
    assert dictionary["factors"]["liquidity_20"]["clean_factor_claim"] is False
    assert dictionary["factors"]["flow_persistence"]["source"] == "unavailable"


def test_run_ablation_case_returns_release_safe_metrics_and_fingerprints():
    base = load_raw_yaml(BASE_CONFIG)
    case = build_ablation_cases(base)[0]

    row = run_ablation_case(case, _make_panel())

    assert row["case_id"] == "dynamic_same_close"
    assert row["legacy_bias_control"] is True
    assert row["selection_lag_days"] == 0
    assert row["research_only"] is True
    assert row["trade_instruction"] is False
    assert row["delta_vs_previous"] is None
    assert set(
        [
            "total_return",
            "annualized_return",
            "max_drawdown",
            "turnover",
            "gross_exposure",
            "blocked_limit_up_buys",
            "blocked_limit_down_sells",
            "blocked_open_gap_buys",
            "blocked_orders_total",
            "final_equity",
        ]
    ).issubset(row)
    assert row["source_identity"] == "broad_data"
    assert row["source_symbol_count"] == 2
    for name in ("signal_fingerprint", "portfolio_fingerprint", "risk_fingerprint"):
        assert len(row[name]) == 64


def test_build_audit_verdict_separates_legacy_control_from_clean_reference():
    rows = pd.DataFrame(
        [
            {"case_id": "dynamic_same_close", "total_return": 0.50},
            {"case_id": "dynamic_lag1_close", "total_return": 0.40},
            {"case_id": "dynamic_lag1_next_open", "total_return": 0.30},
            {"case_id": "dynamic_lag1_next_open_limits", "total_return": 0.25},
            {"case_id": "dynamic_lag1_next_open_limits_gap3", "total_return": 0.20},
            {
                "case_id": "dynamic_lag1_next_open_limits_gap3_cost2x",
                "total_return": 0.18,
            },
        ]
    )

    verdict = build_audit_verdict(rows, {"lookthrough_suspect": True})

    assert verdict["research_only"] is True
    assert verdict["trade_instruction"] is False
    assert verdict["legacy_bias_control_case"] == "dynamic_same_close"
    assert verdict["clean_reference_case"] == "dynamic_lag1_next_open_limits_gap3_cost2x"
    assert verdict["universe_lag_return_delta"] == 0.10
    assert verdict["same_close_return_delta"] == 0.10
    assert verdict["legacy_bias_risk"] == "confirmed"


@pytest.mark.parametrize("invalid_sequence", ["missing", "duplicate", "misordered"])
def test_build_audit_verdict_requires_exact_unique_ordered_case_ids(
    invalid_sequence,
):
    rows = pd.DataFrame(
        [
            {"case_id": "dynamic_same_close", "total_return": 0.50},
            {"case_id": "dynamic_lag1_close", "total_return": 0.40},
            {"case_id": "dynamic_lag1_next_open", "total_return": 0.30},
            {"case_id": "dynamic_lag1_next_open_limits", "total_return": 0.25},
            {"case_id": "dynamic_lag1_next_open_limits_gap3", "total_return": 0.20},
            {
                "case_id": "dynamic_lag1_next_open_limits_gap3_cost2x",
                "total_return": 0.18,
            },
        ]
    )
    if invalid_sequence == "missing":
        rows = rows[rows["case_id"] != "dynamic_lag1_next_open_limits"]
    elif invalid_sequence == "duplicate":
        rows.loc[4, "case_id"] = "dynamic_lag1_next_open_limits"
    else:
        rows = rows.iloc[[0, 1, 2, 4, 3, 5]].reset_index(drop=True)

    verdict = build_audit_verdict(rows, {"lookthrough_suspect": False})

    assert verdict["backtests_evaluated"] is False
    assert verdict["universe_lag_return_delta"] is None
    assert verdict["same_close_return_delta"] is None


def test_failed_staging_write_preserves_previous_output(tmp_path, monkeypatch):
    selected = _make_panel(85)
    broad = selected.copy()
    broad_path = tmp_path / "broad.csv"
    selected_path = tmp_path / "selected.csv"
    output_dir = tmp_path / "audit"
    output_dir.mkdir()
    durable_csv = output_dir / "legacy_signal_bias_audit.csv"
    durable_csv.write_text("previous durable output\n", encoding="utf-8")
    broad.to_csv(broad_path, index=False)
    selected.to_csv(selected_path, index=False)

    original_write_json = audit_module._write_json

    def fail_during_staging(path, payload):
        if path.name == "legacy_signal_bias_audit.json":
            raise RuntimeError("injected staging failure")
        return original_write_json(path, payload)

    monkeypatch.setattr(audit_module, "_write_json", fail_during_staging)

    with pytest.raises(RuntimeError, match="injected staging failure"):
        main(
            [
                "--broad-data",
                str(broad_path),
                "--selected-data",
                str(selected_path),
                "--base-config",
                str(BASE_CONFIG),
                "--output-dir",
                str(output_dir),
                "--skip-backtests",
            ]
        )

    assert durable_csv.read_text(encoding="utf-8") == "previous durable output\n"
    assert list(tmp_path.glob(".audit.staging-*")) == []


def test_cli_streams_input_hashes_without_path_read_bytes(tmp_path, monkeypatch):
    selected = _make_panel(85)
    broad = selected.copy()
    broad_path = tmp_path / "broad.csv"
    selected_path = tmp_path / "selected.csv"
    output_dir = tmp_path / "audit"
    broad.to_csv(broad_path, index=False)
    selected.to_csv(selected_path, index=False)

    def reject_read_bytes(_path):
        raise AssertionError("Path.read_bytes must not be used for input hashing")

    monkeypatch.setattr(Path, "read_bytes", reject_read_bytes)

    main(
        [
            "--broad-data",
            str(broad_path),
            "--selected-data",
            str(selected_path),
            "--base-config",
            str(BASE_CONFIG),
            "--output-dir",
            str(output_dir),
            "--skip-backtests",
        ]
    )

    assert (output_dir / "legacy_signal_bias_audit_manifest.json").is_file()


def test_input_change_before_publication_fails_without_replacing_manifest(
    tmp_path, monkeypatch
):
    selected = _make_panel(85)
    broad = selected.copy()
    broad_path = tmp_path / "broad.csv"
    selected_path = tmp_path / "selected.csv"
    output_dir = tmp_path / "audit"
    output_dir.mkdir()
    old_manifest = output_dir / "legacy_signal_bias_audit_manifest.json"
    old_manifest.write_text('{"run_id":"previous"}\n', encoding="utf-8")
    broad.to_csv(broad_path, index=False)
    selected.to_csv(selected_path, index=False)

    original = audit_module.compute_observation_factors

    def mutate_input_after_read(data):
        result = original(data)
        broad_path.write_text("changed during run\n", encoding="utf-8")
        return result

    monkeypatch.setattr(
        audit_module, "compute_observation_factors", mutate_input_after_read
    )

    with pytest.raises(RuntimeError, match="input changed during audit"):
        main(
            [
                "--broad-data",
                str(broad_path),
                "--selected-data",
                str(selected_path),
                "--base-config",
                str(BASE_CONFIG),
                "--output-dir",
                str(output_dir),
                "--skip-backtests",
            ]
        )

    assert old_manifest.read_text(encoding="utf-8") == '{"run_id":"previous"}\n'


def test_publish_keeps_output_directory_present_and_recovers_stale_lock(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "audit"
    output_dir.mkdir()
    (output_dir / "legacy_signal_bias_audit.csv").write_text(
        "previous\n", encoding="utf-8"
    )
    lock_path = tmp_path / ".audit.publish.lock"
    lock_path.write_text('{"pid":999999999}', encoding="utf-8")
    observed = []
    original_replace = audit_module.os.replace

    def observing_replace(source, destination):
        observed.append(output_dir.exists())
        return original_replace(source, destination)

    monkeypatch.setattr(audit_module.os, "replace", observing_replace)

    audit_module._publish_atomically(
        output_dir,
        lambda staging: (staging / "legacy_signal_bias_audit.csv").write_text(
            "new\n", encoding="utf-8"
        ),
        run_id="test-run",
    )

    assert observed and all(observed)
    assert output_dir.is_dir()
    assert not lock_path.exists()
    assert (output_dir / "legacy_signal_bias_audit.csv").read_text(
        encoding="utf-8"
    ) == "new\n"
    assert (tmp_path / ".audit.versions" / "test-run").is_dir()
    assert (tmp_path / ".audit.CURRENT.json").is_file()


def test_publish_repairs_mixed_compatibility_files_before_building(tmp_path):
    output_dir = tmp_path / "audit"
    output_dir.mkdir()
    versions_dir = tmp_path / ".audit.versions"
    version_dir = versions_dir / "current-run"
    version_dir.mkdir(parents=True)

    current_files = {
        "legacy_signal_bias_audit.csv": "current audit\n",
        "observation_factors_latest.csv": "current factors\n",
    }
    for name, content in current_files.items():
        (version_dir / name).write_text(content, encoding="utf-8")
    manifest = {
        "run_id": "current-run",
        "artifacts": {
            name: audit_module._stream_file_identity(version_dir / name)
            for name in current_files
        },
    }
    manifest_path = version_dir / "legacy_signal_bias_audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    (tmp_path / ".audit.CURRENT.json").write_text(
        '{"run_id":"current-run"}\n', encoding="utf-8"
    )

    (output_dir / "legacy_signal_bias_audit.csv").write_text(
        "current audit\n", encoding="utf-8"
    )
    (output_dir / "observation_factors_latest.csv").write_text(
        "interrupted old factors\n", encoding="utf-8"
    )
    (output_dir / "legacy_signal_bias_audit_manifest.json").write_text(
        '{"run_id":"previous-run"}\n', encoding="utf-8"
    )

    def interrupted_new_build(_staging):
        raise RuntimeError("stop after startup recovery")

    with pytest.raises(RuntimeError, match="stop after startup recovery"):
        audit_module._publish_atomically(
            output_dir,
            interrupted_new_build,
            run_id="new-run",
        )

    for name, content in current_files.items():
        assert (output_dir / name).read_text(encoding="utf-8") == content
    assert (
        output_dir / "legacy_signal_bias_audit_manifest.json"
    ).read_bytes() == manifest_path.read_bytes()


@pytest.mark.parametrize("pointer_text", [None, "not json\n", '{"run_id":"../bad"}\n'])
def test_publish_leaves_live_output_untouched_when_current_pointer_is_invalid(
    tmp_path, pointer_text
):
    output_dir = tmp_path / "audit"
    output_dir.mkdir()
    live_file = output_dir / "legacy_signal_bias_audit.csv"
    live_file.write_text("known good output\n", encoding="utf-8")
    if pointer_text is not None:
        (tmp_path / ".audit.CURRENT.json").write_text(
            pointer_text, encoding="utf-8"
        )

    def interrupted_new_build(_staging):
        raise RuntimeError("stop after safe recovery check")

    with pytest.raises(RuntimeError, match="stop after safe recovery check"):
        audit_module._publish_atomically(
            output_dir,
            interrupted_new_build,
            run_id="new-run",
        )

    assert live_file.read_text(encoding="utf-8") == "known good output\n"


def test_cli_quarantines_symbol_tail_only_when_explicitly_enabled(tmp_path):
    selected = _make_panel(85)
    extra = selected[selected["symbol"].eq("000001")].copy()
    extra["symbol"] = "000003"
    zero_index = extra.index[70]
    zero_date = pd.Timestamp(extra.loc[zero_index, "date"])
    extra.loc[
        zero_index,
        ["open", "high", "low", "close", "volume", "amount"],
    ] = 0.0
    broad = pd.concat([selected, extra], ignore_index=True)
    broad_path = tmp_path / "broad.csv"
    selected_path = tmp_path / "selected.csv"
    strict_output = tmp_path / "strict_audit"
    quarantine_output = tmp_path / "quarantine_audit"
    broad.to_csv(broad_path, index=False)
    selected.to_csv(selected_path, index=False)

    base_args = [
        "--broad-data",
        str(broad_path),
        "--selected-data",
        str(selected_path),
        "--base-config",
        str(BASE_CONFIG),
    ]
    with pytest.raises(ValueError, match="terminal all-zero"):
        main([*base_args, "--output-dir", str(strict_output), "--skip-backtests"])

    main(
        [
            *base_args,
            "--output-dir",
            str(quarantine_output),
            "--quarantine-interior-zero-tails",
        ]
    )

    history = pd.read_csv(quarantine_output / "observation_factor_history.csv")
    history["symbol"] = history["symbol"].astype(str).str.zfill(6)
    symbol_tail = history.loc[history["symbol"].eq("000003")]
    assert pd.to_datetime(symbol_tail["date"]).max() < zero_date

    payload = json.loads(
        (quarantine_output / "legacy_signal_bias_audit.json").read_text(
            encoding="utf-8"
        )
    )
    quality = payload["data_quality"]
    assert quality["policy"] == "quarantine_symbol_tail_after_interior_zero"
    assert quality["quarantined_symbol_count"] == 1
    assert quality["quarantined_symbols"] == ["000003"]
    assert quality["interior_zero_rows"] == 1
    assert quality["quarantined_tail_rows"] == 15
    assert payload["factor_snapshot"]["stale_symbols"] == ["000003"]
    assert all(
        row["total_return"] is not None for row in payload["ablation_results"]
    )

    manifest = json.loads(
        (quarantine_output / "legacy_signal_bias_audit_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["data_quality"] == quality
    report = (quarantine_output / "legacy_signal_bias_audit.md").read_text(
        encoding="utf-8"
    )
    assert "quarantine_symbol_tail_after_interior_zero" in report


def test_cli_writes_safe_artifacts_without_claiming_substituted_liquidity(tmp_path):
    broad = _make_panel(85)
    extra = broad[broad["symbol"].eq("000001")].copy()
    extra["symbol"] = "000003"
    extra["close"] *= 0.8
    extra["open"] *= 0.8
    broad = pd.concat([broad, extra], ignore_index=True)
    selected = broad[broad["symbol"].isin(["000001", "000002"])].copy()

    for frame in (broad, selected):
        frame["amount_substituted"] = True
        frame["amount"] = frame["volume"] * frame["close"]

    broad_path = tmp_path / "broad.csv"
    selected_path = tmp_path / "selected.csv"
    output_dir = tmp_path / "audit"
    broad.to_csv(broad_path, index=False)
    selected.to_csv(selected_path, index=False)

    main(
        [
            "--broad-data",
            str(broad_path),
            "--selected-data",
            str(selected_path),
            "--base-config",
            str(BASE_CONFIG),
            "--output-dir",
            str(output_dir),
            "--skip-backtests",
        ]
    )

    expected = {
        "legacy_signal_bias_audit.md",
        "legacy_signal_bias_audit.csv",
        "legacy_signal_bias_audit.json",
        "legacy_signal_bias_audit_manifest.json",
        "observation_factors_latest.csv",
        "observation_factor_history.csv",
        "observation_factor_dictionary.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected

    audit = pd.read_json(output_dir / "legacy_signal_bias_audit.json", typ="series")
    assert audit["research_only"] is True
    assert audit["trade_instruction"] is False
    assert audit["reusable_factors"] == [
        "momentum_20",
        "momentum_60",
        "breakout_distance_20",
        "trend_acceleration",
    ]

    manifest = json.loads(
        (output_dir / "legacy_signal_bias_audit_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    run_id = manifest["run_id"]
    manifest_id = manifest["manifest_id"]
    assert run_id == manifest_id
    assert re.fullmatch(r"legacy-signal-bias-audit-[0-9TZ-]+-[0-9a-f]{12}", run_id)
    expected_revision, expected_dirty = audit_module._git_state(
        Path(__file__).resolve().parents[1]
    )
    assert manifest["code_revision"] == expected_revision
    assert manifest["code_dirty"] is expected_dirty
    if expected_revision != "unavailable":
        assert re.fullmatch(r"[0-9a-f]{40}", manifest["code_revision"])
    assert len(manifest["base_config_fingerprint"]) == 64
    assert manifest["source_identity"]["factor_history"] == "broad_data"
    assert manifest["source_identity"]["dynamic_ablations"] == "broad_data"
    assert len(manifest["cases"]) == 6
    required_parameters = {
        "selection_lag_days",
        "execution_model",
        "block_limit_up_buys",
        "block_limit_down_sells",
        "limit_buffer",
        "max_buy_open_gap",
        "commission_bps",
        "impact_bps",
        "leverage",
    }
    for case in manifest["cases"]:
        assert required_parameters.issubset(case["parameters"])
        assert case["source_identity"] == "broad_data"
        assert case["allowed_changed_paths"]
        assert len(case["full_config_fingerprint"]) == 64
        assert isinstance(case["config"], dict)

    for input_id, path in (
        ("broad_data", broad_path),
        ("selected_data", selected_path),
        ("base_config", BASE_CONFIG),
    ):
        metadata = manifest["inputs"][input_id]
        assert metadata["size_bytes"] == path.stat().st_size
        assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    artifact_ids = {
        "legacy_signal_bias_audit.csv": pd.read_csv(
            output_dir / "legacy_signal_bias_audit.csv"
        ),
        "observation_factor_history.csv": pd.read_csv(
            output_dir / "observation_factor_history.csv"
        ),
        "observation_factors_latest.csv": pd.read_csv(
            output_dir / "observation_factors_latest.csv"
        ),
    }
    for frame in artifact_ids.values():
        assert frame["run_id"].eq(run_id).all()
        assert frame["manifest_id"].eq(manifest_id).all()
    audit_payload = json.loads(
        (output_dir / "legacy_signal_bias_audit.json").read_text(encoding="utf-8")
    )
    dictionary_payload = json.loads(
        (output_dir / "observation_factor_dictionary.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit_payload["run_id"] == run_id
    assert audit_payload["manifest_id"] == manifest_id
    assert dictionary_payload["run_id"] == run_id
    assert dictionary_payload["manifest_id"] == manifest_id
    markdown = (output_dir / "legacy_signal_bias_audit.md").read_text(
        encoding="utf-8"
    )
    assert f"run_id: {run_id}" in markdown
    assert f"manifest_id: {manifest_id}" in markdown

    for name, metadata in manifest["artifacts"].items():
        artifact_path = output_dir / name
        assert metadata["size_bytes"] == artifact_path.stat().st_size
        assert metadata["sha256"] == hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()

    audit_rows = pd.read_csv(output_dir / "legacy_signal_bias_audit.csv")
    assert audit_rows["research_only"].eq(True).all()
    assert audit_rows["trade_instruction"].eq(False).all()
    assert audit_rows["case_role"].tolist() == [
        "legacy_bias_control",
        "intermediate_bias_control",
        "execution_transition",
        "partial_strict",
        "partial_strict",
        "partial_strict",
    ]
    assert audit_rows.loc[1:, "selection_lag_days"].eq(1).all()
    assert audit_rows["limit_status_coverage"].eq("partial_board_default").all()
    assert audit_rows["strict_execution_supported"].eq(False).all()

    history = pd.read_csv(output_dir / "observation_factor_history.csv")
    latest = pd.read_csv(output_dir / "observation_factors_latest.csv")
    for factors in (history, latest):
        assert factors["signal_lag_sessions"].eq(1).all()
        assert factors["research_only"].eq(True).all()
        assert factors["trade_instruction"].eq(False).all()
        assert factors["liquidity_20"].isna().all()
        assert factors["liquidity_stability_20"].isna().all()

    dictionary = pd.read_json(
        output_dir / "observation_factor_dictionary.json", typ="series"
    )
    assert dictionary["research_only"] is True
    assert dictionary["trade_instruction"] is False
    assert dictionary["signal_lag_sessions"] == 1
    assert dictionary["factors"]["liquidity_20"]["available"] is False
    assert dictionary["factors"]["liquidity_20"]["clean_factor_claim"] is False
    for name in ("forward_return", "opening_gap"):
        assert dictionary["factors"][name]["evaluation_only"] is True
        assert dictionary["factors"][name]["clean_factor_claim"] is False

    semantic_fields = {
        "momentum_20",
        "momentum_60",
        "breakout_distance_20",
        "trend_acceleration",
        "volatility_20",
        "liquidity_20",
        "liquidity_stability_20",
        "flow_persistence",
        "capacity_risk",
        "limit_up_risk",
        "limit_down_risk",
        "opening_gap_risk",
        "history_eligible",
        "liquidity_eligible",
        "score_eligible",
        "observation_score",
    }
    assert semantic_fields.issubset(dictionary["fields"])
    required_semantics = {
        "direction",
        "timing",
        "source",
        "available",
        "score_participation",
        "clean_factor_claim",
    }
    for name in semantic_fields:
        assert required_semantics.issubset(dictionary["fields"][name])
    assert dictionary["fields"]["opening_gap_risk"]["evaluation_only"] is True
    assert dictionary["fields"]["opening_gap_risk"]["score_participation"] is False
    assert dictionary["limit_status_coverage"] == "partial_board_default"

    report = (output_dir / "legacy_signal_bias_audit.md").read_text(
        encoding="utf-8"
    )
    assert "research_only: true" in report
    assert "trade_instruction: false" in report
    for heading in ("## 已验证证据", "## 解释", "## 不可用字段"):
        assert heading in report
    for role_line in (
        "`dynamic_same_close`: `legacy_bias_control`",
        "`dynamic_lag1_close`: `intermediate_bias_control`",
        "`dynamic_lag1_next_open`: `execution_transition`",
        "`dynamic_lag1_next_open_limits`: `partial_strict`",
    ):
        assert role_line in report
    for phrase in ("动态滚动动量", "前高突破", "趋势加速度", "流动性"):
        assert phrase in report
    assert "ST 5%" in report
    assert "exceptional limits" in report
    assert "broad_data" in report
    assert "selected_data" in report
    for phrase in (
        "静态全周期赢家成员",
        "同收盘价成交",
        "无限制涨跌停成交",
        "杠杆作为阿尔法",
    ):
        assert phrase in report
    released_text = "\n".join(
        path.read_text(encoding="utf-8") for path in output_dir.iterdir()
    ).lower()
    for instruction in (
        "建议买入",
        "建议卖出",
        "should buy",
        "should sell",
        "must buy",
        "must sell",
    ):
        assert instruction not in released_text


def test_cli_uses_broad_factor_panel_and_global_latest_session(tmp_path):
    selected = _make_panel(85)
    extra = selected[selected["symbol"].eq("000001")].copy()
    extra["symbol"] = "000003"
    extra = extra.iloc[:-1].copy()
    extra_close = 8.0 * np.power(1.01, np.arange(len(extra)))
    extra["close"] = extra_close
    extra["open"] = extra_close * 1.001
    extra["high"] = extra_close * 1.01
    extra["low"] = extra_close * 0.99
    extra["amount"] = extra_close * extra["volume"]
    broad = pd.concat([selected, extra], ignore_index=True)

    broad_path = tmp_path / "broad.csv"
    selected_path = tmp_path / "selected.csv"
    output_dir = tmp_path / "audit"
    broad.to_csv(broad_path, index=False)
    selected.to_csv(selected_path, index=False)

    main(
        [
            "--broad-data",
            str(broad_path),
            "--selected-data",
            str(selected_path),
            "--base-config",
            str(BASE_CONFIG),
            "--output-dir",
            str(output_dir),
            "--skip-backtests",
        ]
    )

    history = pd.read_csv(output_dir / "observation_factor_history.csv")
    latest = pd.read_csv(output_dir / "observation_factors_latest.csv")
    assert set(history["symbol"].astype(str).str.zfill(6)) == {
        "000001",
        "000002",
        "000003",
    }
    assert latest["date"].nunique() == 1
    assert latest["date"].iloc[0] == broad["date"].max().date().isoformat()
    assert set(latest["symbol"].astype(str).str.zfill(6)) == {"000001", "000002"}

    extra_last_date = extra["date"].max().date().isoformat()
    ranked = history.loc[history["date"].eq(extra_last_date)].copy()
    ranked["symbol"] = ranked["symbol"].astype(str).str.zfill(6)
    scores = ranked.set_index("symbol")["observation_score"]
    assert scores["000003"] > scores["000001"]

    payload = json.loads(
        (output_dir / "legacy_signal_bias_audit.json").read_text(encoding="utf-8")
    )
    assert payload["source_identity"] == {
        "factor_history": "broad_data",
        "dynamic_ablations": "broad_data",
        "static_audit_selected": "selected_data",
        "static_audit_comparison": "broad_data",
    }
    assert payload["factor_snapshot"]["stale_symbol_count"] == 1
    assert payload["factor_snapshot"]["stale_symbols"] == ["000003"]


def test_cli_runs_full_ablation_with_deltas_and_clean_amount_factors(tmp_path):
    broad = _make_panel(130)
    extra = broad[broad["symbol"].eq("000001")].copy()
    extra["symbol"] = "000003"
    extra["close"] *= 0.75
    extra["open"] *= 0.75
    extra["amount"] *= 1.07
    broad = pd.concat([broad, extra], ignore_index=True)
    broad["amount"] *= 1.03
    selected = broad[broad["symbol"].isin(["000001", "000002"])].copy()

    broad_path = tmp_path / "broad.csv"
    selected_path = tmp_path / "selected.csv"
    output_dir = tmp_path / "audit"
    broad.to_csv(broad_path, index=False)
    selected.to_csv(selected_path, index=False)

    main(
        [
            "--broad-data",
            str(broad_path),
            "--selected-data",
            str(selected_path),
            "--base-config",
            str(BASE_CONFIG),
            "--output-dir",
            str(output_dir),
        ]
    )

    rows = pd.read_csv(output_dir / "legacy_signal_bias_audit.csv")
    assert len(rows) == 6
    assert rows["total_return"].notna().all()
    assert pd.isna(rows.loc[0, "delta_vs_previous"])
    expected_deltas = rows["total_return"].diff().round(12)
    pd.testing.assert_series_equal(
        rows["delta_vs_previous"].round(12),
        expected_deltas,
        check_names=False,
    )
    for name in ("signal_fingerprint", "portfolio_fingerprint", "risk_fingerprint"):
        assert rows[name].nunique() == 1
    assert rows["source_identity"].eq("broad_data").all()
    assert rows["source_symbol_count"].eq(3).all()

    history = pd.read_csv(output_dir / "observation_factor_history.csv")
    assert set(history["symbol"].astype(str).str.zfill(6)) == {
        "000001",
        "000002",
        "000003",
    }

    payload = json.loads(
        (output_dir / "legacy_signal_bias_audit.json").read_text(encoding="utf-8")
    )
    assert payload["reusable_factors"] == [
        "momentum_20",
        "momentum_60",
        "breakout_distance_20",
        "trend_acceleration",
        "liquidity_20",
        "liquidity_stability_20",
    ]
    dictionary = json.loads(
        (output_dir / "observation_factor_dictionary.json").read_text(
            encoding="utf-8"
        )
    )
    assert dictionary["amount_provenance"] == "raw_point_in_time"
    assert dictionary["factors"]["liquidity_20"]["clean_factor_claim"] is True
