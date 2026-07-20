"""Write reproducible daily workflow run cards."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


VOLATILE_RECORD_KEYS = {"recorded_at"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_daily_run_card(record: dict[str, Any], generated_at: str | None = None) -> dict[str, Any]:
    stable_record = {key: value for key, value in record.items() if key not in VOLATILE_RECORD_KEYS}
    artifact_map = stable_record.get("artifacts") if isinstance(stable_record.get("artifacts"), dict) else {}
    artifact_map = {
        key: path
        for key, path in artifact_map.items()
        if not str(key).startswith("daily_run_card_")
    }
    stable_record["artifacts"] = artifact_map
    artifacts = [
        _artifact_entry(key, Path(str(path)))
        for key, path in sorted(artifact_map.items())
        if path not in (None, "")
    ]
    commands = list(stable_record.get("commands") or [])
    return {
        "schema_version": 2,
        "generated_at": generated_at or datetime.now().isoformat(timespec="seconds"),
        "asof_date": stable_record.get("asof_date"),
        "run_status": stable_record.get("run_status", "unknown"),
        "failure": stable_record.get("failure"),
        "run_type": stable_record.get("run_type"),
        "data_source": stable_record.get("data_source"),
        "data_source_exists": stable_record.get("data_source_exists"),
        "config_hash": stable_json_hash(
            {
                "asof_date": stable_record.get("asof_date"),
                "run_type": stable_record.get("run_type"),
                "data_source": stable_record.get("data_source"),
                "steps": stable_record.get("steps"),
                "commands": commands,
                "argv": stable_record.get("argv"),
            }
        ),
        "record_hash": stable_json_hash(stable_record),
        "command_count": len(commands),
        "commands": commands,
        "execution": stable_record.get("execution") or {},
        "verification": stable_record.get("verification") or {},
        "fallback_fetch_status": stable_record.get("fallback_fetch_status"),
        "top10": stable_record.get("top10") or [],
        "artifacts": artifacts,
        "warnings": _warnings(stable_record, artifacts),
    }


def write_daily_run_card(output_dir: Path, record: dict[str, Any], generated_at: str | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    asof_date = str(record.get("asof_date") or "unknown")
    token = asof_date.replace("-", "")
    card = build_daily_run_card(record, generated_at=generated_at)
    json_path = output_dir / f"daily_run_card_{token}.json"
    markdown_path = output_dir / f"daily_run_card_{token}.md"
    json_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(card), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _artifact_entry(key: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    is_file = path.is_file() if exists else False
    return {
        "key": key,
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if is_file else None,
        "sha256": file_sha256(path) if is_file else None,
    }


def _warnings(record: dict[str, Any], artifacts: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    if not record.get("data_source_exists", True):
        warnings.append("data_source_missing")
    if record.get("run_status") == "failed":
        warnings.append("run_failed")
        warnings.append("artifacts_may_include_same_day_pre_failure_outputs")
    for item in artifacts:
        if not item["exists"]:
            warnings.append(f"artifact_missing:{item['key']}")
    verification = record.get("verification") or {}
    if verification.get("missing_stock_names"):
        warnings.append("stock_names_missing")
    benchmark_status = str(verification.get("benchmark_refresh_status") or "")
    if "degraded" in benchmark_status.lower():
        warnings.append(f"benchmark_refresh:{benchmark_status}")
    tests = verification.get("tests")
    if tests and _test_status_has_warning(str(tests)):
        warnings.append(f"tests:{tests}")
    return warnings


def _test_status_has_warning(status: str) -> bool:
    normalized = status.strip().lower()
    if normalized in {"passed", "not_run_by_pipeline"}:
        return False
    failed = re.search(r"\b(\d+)\s+failed\b", normalized)
    if failed and int(failed.group(1)) > 0:
        return True
    return re.search(r"\b\d+\s+passed\b", normalized) is None


def _markdown(card: dict[str, Any]) -> str:
    lines = [
        f"# Daily Run Card {card.get('asof_date')}",
        "",
        "Research workflow reproducibility record. This file records inputs, commands, checks, and artifact hashes.",
        "",
        "## Summary",
        f"- Generated at: {card.get('generated_at')}",
        f"- Run status: {card.get('run_status')}",
        f"- Run type: {card.get('run_type')}",
        f"- Data source: {card.get('data_source')}",
        f"- Config hash: {card.get('config_hash')}",
        f"- Record hash: {card.get('record_hash')}",
        f"- Commands: {card.get('command_count')}",
        "",
        "## Failure",
        f"- {card.get('failure') or 'none'}",
        "",
        "## Verification",
    ]
    verification = card.get("verification") or {}
    if verification:
        for key, value in verification.items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")
    execution = card.get("execution") or {}
    lines.extend(
        [
            "",
            "## Execution",
            f"- Max parallel steps: {execution.get('max_parallel_steps')}",
            f"- Wall duration seconds: {execution.get('wall_duration_seconds')}",
            f"- Summed step duration seconds: {execution.get('summed_step_duration_seconds')}",
            f"- Cache hits: {execution.get('cache_hits')}",
            "",
            "| step | status | duration_seconds | cache_hit |",
            "|---|---|---:|---:|",
        ]
    )
    step_executions = execution.get("steps") or []
    if step_executions:
        for item in step_executions:
            lines.append(
                f"| {item.get('name')} | {item.get('status')} | "
                f"{item.get('duration_seconds')} | {item.get('cache_hit')} |"
            )
    else:
        lines.append("| none | not_run | 0 | False |")
    lines.extend(["", "## Artifacts", "| key | exists | size_bytes | sha256 | path |", "|---|---:|---:|---|---|"])
    for item in card.get("artifacts") or []:
        lines.append(
            f"| {item['key']} | {item['exists']} | {item['size_bytes']} | {item['sha256'] or ''} | {item['path']} |"
        )
    warnings = card.get("warnings") or []
    lines.extend(["", "## Warnings"])
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    lines.extend(["", "## Commands"])
    commands = card.get("commands") or []
    lines.extend([f"{idx}. `{command}`" for idx, command in enumerate(commands, start=1)] if commands else ["- none"])
    return "\n".join(lines) + "\n"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
