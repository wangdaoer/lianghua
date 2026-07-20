"""Read-only MCP server for the local quant research workspace."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT_ENV = "QUANT_MCP_PROJECT_ROOT"
MAX_CONFIG_BYTES = 512 * 1024
MAX_PREVIEW_BYTES = 32 * 1024
MAX_SEARCH_FILE_BYTES = 256 * 1024
MAX_SEARCHED_FILES = 2_000
READ_ONLY_TOOL_NAMES = (
    "get_project_status",
    "list_strategy_configs",
    "read_strategy_config",
    "get_latest_run_status",
    "search_project_outputs",
)
PROTECTED_PATHS = ("data/", "outputs/", ".codex/", "tmp*/", "tmp_*")
TEXT_SUFFIXES = {".csv", ".json", ".log", ".md", ".ps1", ".txt", ".yaml", ".yml"}
SKIP_DIR_NAMES = {".codex", ".git", ".pytest_cache", "__pycache__", "data"}

SERVER_INSTRUCTIONS = (
    "quant-mcp is a read-only MCP server for D:\\codex\\閲忓寲. "
    "Use it to inspect Git state, strategy configs, latest run-status files, "
    "and bounded text snippets from outputs. Never treat it as permission to "
    "trade, write files, delete files, modify configs, or read whole data/output trees."
)


def default_project_root() -> Path:
    """Return the workspace root used by this server."""
    env_root = os.environ.get(PROJECT_ROOT_ENV)
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _project_root(project_root: Path | str | None) -> Path:
    return Path(project_root).expanduser().resolve() if project_root is not None else default_project_root()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _run_git(root: Path, args: list[str]) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, "", str(exc)
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _safe_config_path(root: Path, config_path: str) -> Path:
    candidate = Path(config_path)
    if candidate.is_absolute():
        raise ValueError("Strategy config path must be relative to configs/.")
    if not candidate.parts or candidate.parts[0] != "configs":
        candidate = Path("configs") / candidate
    target = (root / candidate).resolve()
    configs_dir = (root / "configs").resolve()
    try:
        target.relative_to(configs_dir)
    except ValueError as exc:
        raise ValueError("Strategy config path must stay under configs/.") from exc
    if target.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Strategy config path must be a .yaml or .yml file.")
    if not target.is_file():
        raise FileNotFoundError(f"Strategy config not found: {candidate.as_posix()}")
    return target


def _read_text_preview(path: Path, max_bytes: int = MAX_PREVIEW_BYTES) -> str:
    data = path.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="replace")


def get_project_status(project_root: Path | str | None = None) -> dict[str, Any]:
    """Return Git and workspace status without modifying the project."""
    root = _project_root(project_root)
    branch_code, branch, branch_err = _run_git(root, ["rev-parse", "--abbrev-ref", "HEAD"])
    head_code, head, head_err = _run_git(root, ["rev-parse", "HEAD"])
    remote_code, remote, remote_err = _run_git(root, ["config", "--get", "remote.origin.url"])
    status_code, status, status_err = _run_git(root, ["status", "--short"])
    changes = [line for line in status.splitlines() if line.strip()]
    return {
        "project_root": str(root),
        "read_only": True,
        "protected_paths": list(PROTECTED_PATHS),
        "tools": list(READ_ONLY_TOOL_NAMES),
        "git": {
            "available": branch_code == 0 and head_code == 0,
            "branch": branch if branch_code == 0 else None,
            "head": head if head_code == 0 else None,
            "remote_origin": remote if remote_code == 0 else None,
            "clean": status_code == 0 and not changes,
            "change_count": len(changes) if status_code == 0 else None,
            "changes": changes[:200] if status_code == 0 else [],
            "errors": [err for err in (branch_err, head_err, remote_err, status_err) if err],
        },
    }


def list_strategy_configs(
    pattern: str = "*.y*ml",
    limit: int = 200,
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    """List strategy YAML files under configs/ with small metadata."""
    root = _project_root(project_root)
    configs_dir = root / "configs"
    safe_limit = max(1, min(int(limit), 500))
    if not configs_dir.is_dir():
        return {"project_root": str(root), "count": 0, "configs": []}
    configs: list[dict[str, Any]] = []
    for path in sorted(configs_dir.rglob(pattern)):
        if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
            continue
        stat = path.stat()
        configs.append(
            {
                "path": _relative(path, root),
                "name": path.stem,
                "size_bytes": stat.st_size,
                "modified_time": _iso_mtime(path),
            }
        )
        if len(configs) >= safe_limit:
            break
    return {"project_root": str(root), "count": len(configs), "configs": configs}


def read_strategy_config(config_path: str, project_root: Path | str | None = None) -> dict[str, Any]:
    """Read and parse a YAML strategy config under configs/."""
    root = _project_root(project_root)
    target = _safe_config_path(root, config_path)
    size = target.stat().st_size
    if size > MAX_CONFIG_BYTES:
        raise ValueError(f"Strategy config is too large to return safely: {size} bytes.")
    text = target.read_text(encoding="utf-8-sig")
    parsed = yaml.safe_load(text) or {}
    return {
        "project_root": str(root),
        "path": _relative(target, root),
        "size_bytes": size,
        "modified_time": _iso_mtime(target),
        "parsed": parsed,
        "text": text,
    }


def _status_candidates(root: Path) -> list[Path]:
    candidates: dict[Path, None] = {}
    for pattern in ("tmp_*status*.txt", "tmp_*status*.json", "tmp_*run*.txt", "tmp_status_payload.json"):
        for path in root.glob(pattern):
            if path.is_file():
                candidates[path.resolve()] = None
    for directory in (root / "outputs" / "logs", root / "outputs" / "research"):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if any(token in name for token in ("status", "summary", "run")) and path.suffix.lower() in TEXT_SUFFIXES:
                candidates[path.resolve()] = None
    return sorted(candidates.keys(), key=lambda item: item.stat().st_mtime, reverse=True)


def get_latest_run_status(
    limit: int = 10,
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return bounded previews from recent run/status artifacts."""
    root = _project_root(project_root)
    safe_limit = max(1, min(int(limit), 25))
    status_files: list[dict[str, Any]] = []
    skipped_large_files = 0
    candidates = _status_candidates(root)
    for path in candidates:
        size = path.stat().st_size
        if size > MAX_PREVIEW_BYTES:
            skipped_large_files += 1
            continue
        preview = _read_text_preview(path)
        item: dict[str, Any] = {
            "path": _relative(path, root),
            "size_bytes": size,
            "modified_time": _iso_mtime(path),
            "preview": preview,
        }
        try:
            item["json"] = json.loads(preview)
        except json.JSONDecodeError:
            pass
        status_files.append(item)
        if len(status_files) >= safe_limit:
            break
    return {
        "project_root": str(root),
        "candidate_count": len(candidates),
        "skipped_large_files": skipped_large_files,
        "status_files": status_files,
    }


def _searchable_files(root: Path) -> list[Path]:
    files: list[Path] = []
    readme = root / "README.md"
    if readme.is_file():
        files.append(readme)
    for directory in (root / "docs", root / "outputs"):
        if not directory.is_dir():
            continue
        for current_root, dirs, names in os.walk(directory):
            dirs[:] = [name for name in dirs if name not in SKIP_DIR_NAMES]
            for name in names:
                path = Path(current_root) / name
                if path.suffix.lower() in TEXT_SUFFIXES:
                    files.append(path)
                if len(files) >= MAX_SEARCHED_FILES:
                    return files
    for path in root.glob("tmp_*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return files[:MAX_SEARCHED_FILES]


def search_project_outputs(
    query: str,
    max_results: int = 20,
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    """Search bounded text files in README, docs, outputs, and root tmp status files."""
    needle = query.strip()
    if not needle:
        raise ValueError("query must not be empty")
    root = _project_root(project_root)
    safe_limit = max(1, min(int(max_results), 100))
    needle_lower = needle.lower()
    matches: list[dict[str, Any]] = []
    skipped_large_files = 0
    searched_files = 0
    for path in _searchable_files(root):
        if any(part in SKIP_DIR_NAMES for part in path.relative_to(root).parts):
            continue
        size = path.stat().st_size
        if size > MAX_SEARCH_FILE_BYTES:
            skipped_large_files += 1
            continue
        searched_files += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if needle_lower not in line.lower():
                continue
            matches.append(
                {
                    "path": _relative(path, root),
                    "line": line_number,
                    "modified_time": _iso_mtime(path),
                    "snippet": line.strip()[:240],
                }
            )
            if len(matches) >= safe_limit:
                return {
                    "project_root": str(root),
                    "query": needle,
                    "searched_files": searched_files,
                    "skipped_large_files": skipped_large_files,
                    "matches": matches,
                }
            break
    return {
        "project_root": str(root),
        "query": needle,
        "searched_files": searched_files,
        "skipped_large_files": skipped_large_files,
        "matches": matches,
    }


def create_mcp_server() -> Any:
    """Create the FastMCP server with read-only tools registered."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError("Install the MCP Python SDK first: python -m pip install 'mcp>=1.27,<2'") from exc

    try:
        server = FastMCP("quant-mcp", instructions=SERVER_INSTRUCTIONS)
    except TypeError:
        server = FastMCP("quant-mcp")

    @server.tool(name="get_project_status")
    def get_project_status_tool() -> dict[str, Any]:
        """Inspect Git state and read-only guardrails for the quant workspace."""
        return get_project_status()

    @server.tool(name="list_strategy_configs")
    def list_strategy_configs_tool(pattern: str = "*.y*ml", limit: int = 200) -> dict[str, Any]:
        """List YAML strategy configs under configs/."""
        return list_strategy_configs(pattern=pattern, limit=limit)

    @server.tool(name="read_strategy_config")
    def read_strategy_config_tool(config_path: str) -> dict[str, Any]:
        """Read one YAML strategy config from configs/."""
        return read_strategy_config(config_path)

    @server.tool(name="get_latest_run_status")
    def get_latest_run_status_tool(limit: int = 10) -> dict[str, Any]:
        """Read bounded previews from latest run/status files."""
        return get_latest_run_status(limit=limit)

    @server.tool(name="search_project_outputs")
    def search_project_outputs_tool(query: str, max_results: int = 20) -> dict[str, Any]:
        """Search bounded text snippets in README, docs, outputs, and root tmp files."""
        return search_project_outputs(query=query, max_results=max_results)

    return server


def main() -> int:
    """Run the MCP server over STDIO."""
    create_mcp_server().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
