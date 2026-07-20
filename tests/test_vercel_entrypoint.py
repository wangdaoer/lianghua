from __future__ import annotations

import tomllib
from pathlib import Path

from app import build_response, handler


ROOT = Path(__file__).resolve().parents[1]


def test_vercel_entrypoint_is_explicitly_configured() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["tool"]["vercel"]["entrypoint"] == "app:handler"
    assert handler.__name__ == "handler"


def test_health_response_is_read_only() -> None:
    status, payload = build_response("/health")

    assert status == 200
    assert payload["status"] == "ok"
    assert payload["mode"] == "research_only"
    assert payload["broker_action"] == "none"


def test_unknown_path_returns_not_found() -> None:
    status, payload = build_response("/missing")

    assert status == 404
    assert payload["status"] == "not_found"
