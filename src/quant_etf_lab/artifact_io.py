"""Atomic artifact writes with semantic change detection."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


def write_bytes_if_changed(path: str | Path, content: bytes) -> bool:
    """Atomically write bytes only when the target content changed."""

    resolved = Path(path)
    if resolved.exists() and resolved.read_bytes() == content:
        return False
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def write_text_if_changed(path: str | Path, content: str, *, encoding: str = "utf-8") -> bool:
    return write_bytes_if_changed(path, content.encode(encoding))


def _normalize_semantic(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_semantic(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_semantic(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_semantic(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _without_fields(payload: dict[str, Any], ignored_fields: Iterable[str]) -> dict[str, Any]:
    ignored = set(ignored_fields)
    return _normalize_semantic({key: value for key, value in payload.items() if key not in ignored})


def publish_json_if_semantically_changed(
    path: str | Path,
    payload: dict[str, Any],
    *,
    ignored_fields: Iterable[str] = ("generated_at",),
) -> tuple[dict[str, Any], bool]:
    """Publish JSON unless only volatile top-level fields changed."""

    resolved = Path(path)
    if resolved.exists():
        try:
            existing = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and _without_fields(existing, ignored_fields) == _without_fields(
            payload,
            ignored_fields,
        ):
            return existing, False
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    return payload, write_text_if_changed(resolved, text, encoding="utf-8")
