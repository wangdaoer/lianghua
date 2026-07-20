import json
from pathlib import Path

from quant_etf_lab.artifact_io import publish_json_if_semantically_changed, write_text_if_changed


def test_write_text_if_changed_preserves_unchanged_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    assert write_text_if_changed(path, "same") is True
    first_mtime = path.stat().st_mtime_ns

    assert write_text_if_changed(path, "same") is False
    assert path.stat().st_mtime_ns == first_mtime


def test_publish_json_ignores_generated_at_only(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    first, changed = publish_json_if_semantically_changed(
        path,
        {"generated_at": "2026-07-20T16:00:00", "status": "ok"},
    )
    first_mtime = path.stat().st_mtime_ns
    second, changed_again = publish_json_if_semantically_changed(
        path,
        {"generated_at": "2026-07-20T16:01:00", "status": "ok"},
    )

    assert changed is True
    assert changed_again is False
    assert second == first
    assert path.stat().st_mtime_ns == first_mtime
    assert json.loads(path.read_text(encoding="utf-8"))["generated_at"] == "2026-07-20T16:00:00"
