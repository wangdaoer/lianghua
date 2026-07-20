"""Compatibility helpers for text file reads."""

from __future__ import annotations

from pathlib import Path


def read_text(path: str | Path, *, encoding: str = "utf-8", errors: str | None = None) -> str:
    resolved = Path(path)
    if encoding != "utf-8" or errors is not None:
        if errors is None:
            return resolved.read_text(encoding=encoding)
        return resolved.read_text(encoding=encoding, errors=errors)
    for candidate in ("utf-8-sig", "utf-8"):
        try:
            return resolved.read_text(encoding=candidate)
        except UnicodeDecodeError:
            continue
    return resolved.read_text(encoding="utf-8")
