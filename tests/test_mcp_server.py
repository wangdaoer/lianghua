from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from quant_etf_lab import mcp_server


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_list_strategy_configs_returns_yaml_metadata(tmp_path: Path) -> None:
    write_text(tmp_path / "configs" / "alpha.yaml", "name: alpha\n")
    write_text(tmp_path / "configs" / "nested" / "beta.yml", "name: beta\n")
    write_text(tmp_path / "data" / "raw" / "ignored.yaml", "name: ignored\n")

    result = mcp_server.list_strategy_configs(project_root=tmp_path)

    paths = [item["path"] for item in result["configs"]]
    assert paths == ["configs/alpha.yaml", "configs/nested/beta.yml"]
    assert result["count"] == 2


def test_read_strategy_config_parses_yaml_and_rejects_escape(tmp_path: Path) -> None:
    write_text(
        tmp_path / "configs" / "alpha.yaml",
        "name: alpha\nrisk:\n  max_drawdown: 0.08\n",
    )
    write_text(tmp_path / "README.md", "outside configs\n")

    result = mcp_server.read_strategy_config("configs/alpha.yaml", project_root=tmp_path)

    assert result["path"] == "configs/alpha.yaml"
    assert result["parsed"]["risk"]["max_drawdown"] == 0.08
    assert "name: alpha" in result["text"]

    with pytest.raises(ValueError, match="configs"):
        mcp_server.read_strategy_config("../README.md", project_root=tmp_path)


def test_get_latest_run_status_reads_bounded_status_files(tmp_path: Path) -> None:
    json_path = write_text(
        tmp_path / "tmp_daily_run_status_json.txt",
        json.dumps({"status": "ok", "run_id": "run_1"}, ensure_ascii=False),
    )
    text_path = write_text(tmp_path / "outputs" / "logs" / "daily_status.txt", "latest status ok\n")
    os.utime(json_path, (100, 100))
    os.utime(text_path, (200, 200))

    result = mcp_server.get_latest_run_status(project_root=tmp_path)

    paths = [item["path"] for item in result["status_files"]]
    assert paths[:2] == ["outputs/logs/daily_status.txt", "tmp_daily_run_status_json.txt"]
    assert result["status_files"][1]["json"]["run_id"] == "run_1"
    assert "latest status ok" in result["status_files"][0]["preview"]


def test_search_project_outputs_skips_data_and_large_files(tmp_path: Path) -> None:
    write_text(tmp_path / "README.md", "alpha signal overview\n")
    write_text(tmp_path / "outputs" / "logs" / "run.txt", "daily alpha signal passed\n")
    write_text(tmp_path / "data" / "raw" / "prices.txt", "alpha should not be searched\n")
    write_text(tmp_path / "outputs" / "logs" / "large.txt", "alpha\n" * 100_000)

    result = mcp_server.search_project_outputs("alpha", project_root=tmp_path)

    paths = {match["path"] for match in result["matches"]}
    assert paths == {"README.md", "outputs/logs/run.txt"}
    assert result["skipped_large_files"] == 1


def test_registered_tool_names_are_read_only() -> None:
    assert set(mcp_server.READ_ONLY_TOOL_NAMES) == {
        "get_project_status",
        "list_strategy_configs",
        "read_strategy_config",
        "get_latest_run_status",
        "search_project_outputs",
    }
    assert not any(name.startswith(("run_", "write_", "delete_", "update_")) for name in mcp_server.READ_ONLY_TOOL_NAMES)


def test_create_mcp_server_registers_expected_tool_names() -> None:
    server = mcp_server.create_mcp_server()
    tools = server._tool_manager.list_tools()

    assert {tool.name for tool in tools} == set(mcp_server.READ_ONLY_TOOL_NAMES)
