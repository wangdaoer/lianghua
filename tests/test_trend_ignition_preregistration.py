from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from validate_trend_ignition_preregistration import (
    load_preregistration,
    validate_preregistration,
    validate_unseen_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "trend_ignition_shortlist_preregistered.yaml"


def test_checked_in_preregistration_matches_frozen_model():
    report = validate_preregistration(CONFIG, repo_root=ROOT)

    assert report["ok"] is True
    assert report["training_end_date"] == "2025-06-12"
    assert report["first_eligible_signal_date"] == "2025-06-13"


def test_preregistration_rejects_tampered_frozen_model(tmp_path: Path):
    config_dir = tmp_path / "configs"
    model_dir = config_dir / "models"
    model_dir.mkdir(parents=True)
    temp_config = config_dir / CONFIG.name
    temp_model = model_dir / "trend_ignition_shortlist_20260714_v1.json"
    shutil.copyfile(CONFIG, temp_config)
    shutil.copyfile(ROOT / "configs" / "models" / temp_model.name, temp_model)
    temp_model.write_text(temp_model.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_preregistration(temp_config, repo_root=tmp_path)


def test_preregistration_rejects_unknown_configuration_keys(tmp_path: Path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["research_contract"]["unregistered_parameter"] = 1
    config_path = tmp_path / "registration.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown=.*unregistered_parameter"):
        validate_preregistration(config_path, repo_root=ROOT)


def test_unseen_evidence_must_match_locked_dates_hash_and_gates():
    registration = load_preregistration(CONFIG)
    evidence = {
        "registration_id": registration["registration_id"],
        "model_sha256": registration["source"]["model_sha256"],
        "signal_start_date": "2025-06-13",
        "completed_samples": 500,
        "periods": [
            {
                "period": "unseen_1",
                "bucket_counts": {"high": 100, "middle": 100, "low": 100},
                "fixed_bucket_spread": 0.02,
                "score_label_corr": 0.03,
            }
        ],
    }

    report = validate_unseen_evidence(evidence, registration)

    assert report["ok"] is True
    assert report["completed_samples"] == 500


def test_unseen_evidence_rejects_training_period_dates():
    registration = load_preregistration(CONFIG)
    evidence = {
        "registration_id": registration["registration_id"],
        "model_sha256": registration["source"]["model_sha256"],
        "signal_start_date": "2025-06-12",
        "completed_samples": 500,
        "periods": [],
    }

    with pytest.raises(ValueError, match="strictly unseen"):
        validate_unseen_evidence(evidence, registration)
