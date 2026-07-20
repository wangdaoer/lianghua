"""Lightweight process locks for long-running local research commands."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from ._compat import read_text


class ProcessLockError(RuntimeError):
    """Raised when an equivalent command is already running."""


@dataclass(frozen=True)
class LockPayload:
    key: str
    pid: int
    command: str
    created_at: str


@dataclass(frozen=True)
class ProcessLockStatus:
    path: Path
    payload: LockPayload
    active: bool


def lock_path(lock_dir: Path, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return lock_dir / f"{digest}.lock.json"


def _tasklist_contains_pid(output: str, pid: int) -> bool:
    expected = str(pid)
    for line in output.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = next(csv.reader([text]))
        except (csv.Error, StopIteration):
            continue
        if len(row) >= 2 and row[1].strip() == expected:
            return True
    return False


def _is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return _tasklist_contains_pid(result.stdout, pid)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_payload(path: Path) -> LockPayload | None:
    try:
        raw = json.loads(read_text(path))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return LockPayload(
            key=str(raw.get("key", "")),
            pid=int(raw.get("pid", 0)),
            command=str(raw.get("command", "")),
            created_at=str(raw.get("created_at", "")),
        )
    except (TypeError, ValueError):
        return None


def _write_payload(path: Path, payload: LockPayload) -> None:
    path.write_text(json.dumps(payload.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_process_lock(lock_dir: Path, key: str, pid: int, command: str) -> Path:
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_path(lock_dir, key)
    payload = LockPayload(
        key=key,
        pid=int(pid),
        command=command,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    _write_payload(path, payload)
    return path


def list_process_locks(lock_dir: Path) -> list[ProcessLockStatus]:
    if not lock_dir.exists():
        return []
    statuses: list[ProcessLockStatus] = []
    for path in sorted(lock_dir.glob("*.lock.json")):
        payload = _read_payload(path)
        if payload is None:
            continue
        statuses.append(ProcessLockStatus(path=path, payload=payload, active=_is_pid_running(payload.pid)))
    return statuses


@contextmanager
def process_lock(lock_dir: Path, key: str, command: str) -> Iterator[Path]:
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_path(lock_dir, key)
    existing = _read_payload(path) if path.exists() else None
    if existing is not None and _is_pid_running(existing.pid):
        raise ProcessLockError(
            f"Equivalent command is already running with PID {existing.pid}. "
            f"Lock file: {path}. Command: {existing.command}"
        )
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass

    payload = LockPayload(
        key=key,
        pid=os.getpid(),
        command=command,
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload.__dict__, handle, ensure_ascii=False, indent=2)
    except FileExistsError as exc:
        current = _read_payload(path)
        if current is not None and _is_pid_running(current.pid):
            raise ProcessLockError(
                f"Equivalent command is already running with PID {current.pid}. "
                f"Lock file: {path}. Command: {current.command}"
            ) from exc
        raise ProcessLockError(f"Could not create process lock: {path}") from exc

    try:
        yield path
    finally:
        current = _read_payload(path)
        if current is not None and current.pid == os.getpid():
            try:
                path.unlink()
            except OSError:
                pass
