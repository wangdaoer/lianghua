"""Model-building audit helpers for speed, redundancy, and workflow hygiene."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_WALK_FORWARD_RESOLUTION_PATH = Path("outputs/research/walk_forward_run_resolutions.csv")
_RESOLVED_WALK_FORWARD_STATUSES = {"resolved", "superseded", "abandoned", "duplicate", "replaced", "finalized"}


@dataclass(frozen=True)
class ModelBuildAuditResult:
    output_dir: Path
    report_path: Path
    snapshot_path: Path
    config_redundancy_path: Path
    config_map_path: Path
    walk_forward_runs_path: Path
    walk_forward_actions_path: Path
    snapshot: dict[str, Any]


def _stable_json(value: Any) -> str:
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True, default=str)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()[:16]


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except UnicodeDecodeError:
        raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _extends_list(raw: dict[str, Any]) -> list[str]:
    value = raw.get("extends", raw.get("base_config"))
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _scan_config_inheritance(config_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    paths = sorted(config_dir.glob("*.yaml"))
    base_dir = config_dir / "base"
    if base_dir.exists():
        paths.extend(sorted(base_dir.glob("*.yaml")))
    for path in paths:
        raw = _read_yaml(path)
        project_raw = raw.get("project", {})
        strategy_raw = raw.get("strategy", {})
        extends = _extends_list(raw)
        role = "base" if path.parent.name.lower() == "base" else "root"
        sections = sorted(key for key in raw if key not in {"extends", "base_config", "project", "notes"})
        if raw.get("universe"):
            universe_kind = "explicit"
        elif raw.get("universe_file"):
            universe_kind = "file"
        elif raw.get("universe_source"):
            universe_kind = "source"
        else:
            universe_kind = ""
        rows.append(
            {
                "file": str(path),
                "role": role,
                "name": (
                    str(project_raw.get("name", path.stem))
                    if isinstance(project_raw, dict)
                    else path.stem
                ),
                "has_extends": bool(extends),
                "extends_count": len(extends),
                "extends": "; ".join(extends),
                "sections": "; ".join(sections),
                "strategy_name": (
                    str(strategy_raw.get("name", ""))
                    if isinstance(strategy_raw, dict)
                    else ""
                ),
                "universe_kind": universe_kind,
            }
        )
    return pd.DataFrame(rows).sort_values(["role", "file"]).reset_index(drop=True)


def _scan_config_redundancy(config_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(config_dir.glob("*.yaml")):
        raw = _read_yaml(path)
        for section in ("strategy", "risk", "costs", "universe_source"):
            payload = raw.get(section, {})
            rows.append(
                {
                    "file": str(path),
                    "name": str(raw.get("project", {}).get("name", path.stem)) if isinstance(raw.get("project"), dict) else path.stem,
                    "section": section,
                    "fingerprint": _fingerprint(payload),
                    "is_empty": not bool(payload),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    group_sizes = frame.groupby(["section", "fingerprint"])["file"].transform("count")
    frame["duplicate_group_size"] = group_sizes
    frame["is_duplicate"] = group_sizes > 1
    return frame.sort_values(["section", "fingerprint", "file"]).reset_index(drop=True)


def _markdown_metric(summary_path: Path, label: str) -> str | None:
    if not summary_path.exists():
        return None
    pattern = re.compile(rf"^\|\s*{re.escape(label)}\s*\|\s*(.*?)\s*\|")
    for line in summary_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    return None


def _mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _scan_walk_forward_runs(walk_forward_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not walk_forward_dir.exists():
        return pd.DataFrame(rows)
    for run_dir in sorted(path for path in walk_forward_dir.iterdir() if path.is_dir()):
        summary_path = run_dir / "walk_forward_summary.csv"
        candidate_path = run_dir / "candidate_results.csv"
        stitched_path = run_dir / "oos_equity_stitched.csv"
        md_path = run_dir / "summary.md"
        selected_params = sorted(run_dir.glob("*_selected_params.json"))
        summary_rows = 0
        candidate_rows = 0
        if summary_path.exists():
            try:
                summary_rows = len(pd.read_csv(summary_path))
            except pd.errors.EmptyDataError:
                summary_rows = 0
        if candidate_path.exists():
            try:
                candidate_rows = len(pd.read_csv(candidate_path))
            except pd.errors.EmptyDataError:
                candidate_rows = 0
        if summary_path.exists() and stitched_path.exists() and summary_rows == len(selected_params):
            status = "complete"
        elif selected_params and not summary_path.exists():
            status = "partial_window_outputs"
        elif summary_path.exists() and summary_rows != len(selected_params):
            status = "summary_selected_mismatch"
        else:
            status = "missing_outputs"
        rows.append(
            {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "status": status,
                "selected_params_count": len(selected_params),
                "summary_windows": summary_rows,
                "candidate_rows": candidate_rows,
                "has_stitched_equity": stitched_path.exists(),
                "last_modified": _mtime(run_dir),
                "oos_total_return": _markdown_metric(md_path, "Stitched OOS total return"),
                "oos_max_drawdown": _markdown_metric(md_path, "Stitched OOS max drawdown"),
                "oos_sharpe": _markdown_metric(md_path, "Stitched OOS Sharpe"),
                "candidate_grid": _markdown_metric(md_path, "Candidate grid"),
            }
        )
    return pd.DataFrame(rows).sort_values("last_modified", ascending=False).reset_index(drop=True)


_WALK_FORWARD_RESOLUTION_COLUMNS = [
    "run_name",
    "resolution_status",
    "replacement_run",
    "resolved_at",
    "resolved_by",
    "resolution_note",
]


def _read_walk_forward_resolutions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=_WALK_FORWARD_RESOLUTION_COLUMNS)
    try:
        data = pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=_WALK_FORWARD_RESOLUTION_COLUMNS)
    for column in _WALK_FORWARD_RESOLUTION_COLUMNS:
        if column not in data:
            data[column] = ""
    data["run_name"] = data["run_name"].astype(str).str.strip()
    data["resolution_status"] = data["resolution_status"].astype(str).str.strip().str.lower()
    data = data[data["run_name"] != ""].copy()
    return data[_WALK_FORWARD_RESOLUTION_COLUMNS].drop_duplicates("run_name", keep="last").reset_index(drop=True)


def _apply_walk_forward_resolutions(walk_frame: pd.DataFrame, resolutions: pd.DataFrame) -> pd.DataFrame:
    result = walk_frame.copy()
    for column in _WALK_FORWARD_RESOLUTION_COLUMNS:
        if column not in result:
            result[column] = ""
    if result.empty:
        result["is_resolved"] = pd.Series(dtype=bool)
        return result
    if not resolutions.empty:
        result = result.drop(columns=[column for column in _WALK_FORWARD_RESOLUTION_COLUMNS if column in result and column != "run_name"])
        result = result.merge(resolutions, on="run_name", how="left")
    for column in _WALK_FORWARD_RESOLUTION_COLUMNS:
        if column not in result:
            result[column] = ""
    result[_WALK_FORWARD_RESOLUTION_COLUMNS] = result[_WALK_FORWARD_RESOLUTION_COLUMNS].fillna("")
    result["resolution_status"] = result["resolution_status"].astype(str).str.strip().str.lower()
    result["is_resolved"] = result["resolution_status"].isin(_RESOLVED_WALK_FORWARD_STATUSES)
    return result


_WALK_FORWARD_ACTION_COLUMNS = [
    "run_dir",
    "run_name",
    "status",
    "recommended_action",
    "priority",
    "requires_manual_confirmation",
    "evidence_summary",
    "next_step",
]


def _walk_forward_action(row: pd.Series) -> dict[str, Any]:
    status = str(row.get("status", ""))
    selected_params_count = int(row.get("selected_params_count") or 0)
    summary_windows = int(row.get("summary_windows") or 0)
    candidate_rows = int(row.get("candidate_rows") or 0)
    has_stitched = bool(row.get("has_stitched_equity"))
    evidence = (
        f"selected_params={selected_params_count}; "
        f"summary_windows={summary_windows}; "
        f"candidate_rows={candidate_rows}; "
        f"has_stitched_equity={has_stitched}"
    )
    if status == "partial_window_outputs":
        action = "resume_or_finalize"
        priority = "high"
        requires_manual_confirmation = False
        next_step = (
            "Rerun the matching walk-forward command with --resume and the same run-id prefix, "
            "or explicitly mark the directory as abandoned after reviewing the checkpoint files."
        )
    elif status == "summary_selected_mismatch":
        action = "inspect_then_resume"
        priority = "high"
        requires_manual_confirmation = False
        next_step = (
            "Inspect walk_forward_summary.csv versus *_selected_params.json, then rerun with "
            "--resume if the mismatch is from an interrupted run."
        )
    elif status == "missing_outputs":
        action = "archive_candidate_after_review"
        priority = "medium"
        requires_manual_confirmation = True
        next_step = (
            "Review the directory manually; if it has no useful checkpoints or reports, move it "
            "to an archive location outside active walk-forward results."
        )
    else:
        action = "inspect"
        priority = "medium"
        requires_manual_confirmation = True
        next_step = "Review this unexpected status before using the run as model evidence."
    return {
        "run_dir": row.get("run_dir"),
        "run_name": row.get("run_name"),
        "status": status,
        "recommended_action": action,
        "priority": priority,
        "requires_manual_confirmation": requires_manual_confirmation,
        "evidence_summary": evidence,
        "next_step": next_step,
    }


def _build_walk_forward_actions(walk_frame: pd.DataFrame) -> pd.DataFrame:
    if walk_frame.empty:
        return pd.DataFrame(columns=_WALK_FORWARD_ACTION_COLUMNS)
    unresolved = ~walk_frame.get("is_resolved", pd.Series(False, index=walk_frame.index)).astype(bool)
    actions = [
        _walk_forward_action(row)
        for _, row in walk_frame[(walk_frame["status"] != "complete") & unresolved].iterrows()
    ]
    if not actions:
        return pd.DataFrame(columns=_WALK_FORWARD_ACTION_COLUMNS)
    return pd.DataFrame(actions, columns=_WALK_FORWARD_ACTION_COLUMNS)


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _scan_locks(lock_dir: Path) -> list[dict[str, Any]]:
    if not lock_dir.exists():
        return []
    locks: list[dict[str, Any]] = []
    current_pid = os.getpid()
    for path in sorted(lock_dir.glob("*.lock.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            locks.append({"file": str(path), "status": "invalid_json"})
            continue
        pid = int(raw.get("pid") or 0)
        is_current_process = pid == current_pid
        locks.append(
            {
                "file": str(path),
                "pid": pid,
                "pid_active": _pid_exists(pid),
                "is_current_process": is_current_process,
                "created_at": raw.get("created_at"),
                "command": raw.get("command"),
            }
        )
    return locks


def _duplicate_summary(config_frame: pd.DataFrame) -> pd.DataFrame:
    if config_frame.empty:
        return pd.DataFrame(columns=["section", "fingerprint", "duplicate_count", "files"])
    duplicates = config_frame[config_frame["is_duplicate"] & ~config_frame["is_empty"]].copy()
    if duplicates.empty:
        return pd.DataFrame(columns=["section", "fingerprint", "duplicate_count", "files"])
    grouped = (
        duplicates.groupby(["section", "fingerprint"])
        .agg(duplicate_count=("file", "count"), files=("file", lambda values: "; ".join(sorted(values))))
        .reset_index()
        .sort_values(["duplicate_count", "section"], ascending=[False, True])
    )
    return grouped


def _write_report(
    output_dir: Path,
    config_frame: pd.DataFrame,
    config_map: pd.DataFrame,
    duplicate_groups: pd.DataFrame,
    walk_frame: pd.DataFrame,
    walk_actions: pd.DataFrame,
    locks: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> Path:
    report_path = output_dir / "model_build_audit.md"
    partial_runs = walk_frame[walk_frame["status"] != "complete"] if not walk_frame.empty else pd.DataFrame()
    duplicate_preview = duplicate_groups.head(10).to_dict("records") if not duplicate_groups.empty else []
    partial_preview = partial_runs.head(10).to_dict("records") if not partial_runs.empty else []
    action_preview = walk_actions.head(10).to_dict("records") if not walk_actions.empty else []

    duplicate_lines = ["| section | duplicate_count | files |", "| --- | ---: | --- |"]
    if duplicate_preview:
        for row in duplicate_preview:
            duplicate_lines.append(
                f"| {row['section']} | {row['duplicate_count']} | {row['files']} |"
            )
    else:
        duplicate_lines.append("| none | 0 |  |")

    partial_lines = [
        "| run_name | status | resolution | replacement | selected_params | summary_windows |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    if partial_preview:
        for row in partial_preview:
            partial_lines.append(
                f"| {row['run_name']} | {row['status']} | {row.get('resolution_status', '')} | "
                f"{row.get('replacement_run', '')} | {row['selected_params_count']} | {row['summary_windows']} |"
            )
    else:
        partial_lines.append("| none | complete |  |  | 0 | 0 |")

    action_lines = ["| run_name | action | priority | manual_confirm |", "| --- | --- | --- | --- |"]
    if action_preview:
        for row in action_preview:
            action_lines.append(
                f"| {row['run_name']} | {row['recommended_action']} | {row['priority']} | {row['requires_manual_confirmation']} |"
            )
    else:
        action_lines.append("| none | keep | low | False |")

    body = f"""# Model Build Audit

Generated at: `{snapshot['generated_at']}`

This audit checks local research workflow hygiene only. It does not connect to brokers, place orders, or provide investment advice.

## Summary

| Check | Value |
| --- | ---: |
| Config files scanned | {snapshot['config_files']} |
| Root configs with extends | {snapshot['root_configs_with_extends']} |
| Root configs without extends | {snapshot['root_configs_without_extends']} |
| Base config files | {snapshot['base_config_files']} |
| Duplicate config section groups | {snapshot['duplicate_config_groups']} |
| Walk-forward runs scanned | {snapshot['walk_forward_runs']} |
| Incomplete walk-forward runs | {snapshot['incomplete_walk_forward_runs']} |
| Resolved incomplete walk-forward runs | {snapshot['resolved_walk_forward_runs']} |
| Unresolved incomplete walk-forward runs | {snapshot['unresolved_incomplete_walk_forward_runs']} |
| Walk-forward action items | {snapshot['walk_forward_action_items']} |
| Walk-forward resume candidates | {snapshot['walk_forward_resume_candidates']} |
| Walk-forward archive-review candidates | {snapshot['walk_forward_archive_review_candidates']} |
| Lock files | {snapshot['lock_files']} |
| Active lock files | {snapshot['active_lock_files']} |

## Speed Controls

- Use `walk-forward --resume` with the same `--run-id-prefix` for long full-universe runs.
- Run sample configs before full-universe configs when changing a factor family or grid.
- Keep `factor-lab --no-save-panel` for broad runs unless the panel itself is needed.
- Check candidate-window scale before launching full runs: `windows x candidate rows` is the main cost driver.
- Treat partial `*_selected_params.json` windows as checkpoint evidence, not as final model evidence.

## Duplicate Config Section Groups

{chr(10).join(duplicate_lines)}

## Config Inheritance Map

- Root runnable configs: {snapshot['root_config_files']}
- Root configs using `extends`: {snapshot['root_configs_with_extends']}
- Root configs still standalone: {snapshot['root_configs_without_extends']}
- Base/fragment configs: {snapshot['base_config_files']}
- Inheritance edges: {snapshot['config_inheritance_edges']}

## Incomplete Or Suspicious Walk-Forward Runs

{chr(10).join(partial_lines)}

## Walk-Forward Run Actions

{chr(10).join(action_lines)}

## Output Files

- `config_redundancy.csv`: one row per config section fingerprint.
- `config_duplicate_groups.csv`: duplicate config sections grouped by fingerprint.
- `config_inheritance_map.csv`: root/base config map and direct `extends` links.
- `walk_forward_runs.csv`: walk-forward completeness and headline metrics.
- `walk_forward_run_actions.csv`: non-destructive action list for incomplete or suspicious walk-forward runs.
- Resolution registry: `{snapshot['walk_forward_resolution_path']}`.
- `model_build_audit_snapshot.json`: machine-readable summary.
"""
    report_path.write_text(body, encoding="utf-8")
    return report_path


def run_model_build_audit(
    project_root: Path,
    config_dir: Path | None = None,
    walk_forward_dir: Path | None = None,
    walk_forward_resolution_path: Path | None = None,
    lock_dir: Path | None = None,
    output_dir: Path | None = None,
) -> ModelBuildAuditResult:
    config_dir = config_dir or (project_root / "configs")
    walk_forward_dir = walk_forward_dir or (project_root / "outputs" / "walk_forward")
    walk_forward_resolution_path = walk_forward_resolution_path or (project_root / DEFAULT_WALK_FORWARD_RESOLUTION_PATH)
    if not walk_forward_resolution_path.is_absolute():
        walk_forward_resolution_path = project_root / walk_forward_resolution_path
    lock_dir = lock_dir or (project_root / "outputs" / "locks")
    output_dir = output_dir or (project_root / "outputs" / "research" / "model_build_audit_latest")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_map = _scan_config_inheritance(config_dir)
    config_frame = _scan_config_redundancy(config_dir)
    duplicate_groups = _duplicate_summary(config_frame)
    walk_frame = _scan_walk_forward_runs(walk_forward_dir)
    walk_resolutions = _read_walk_forward_resolutions(walk_forward_resolution_path)
    walk_frame = _apply_walk_forward_resolutions(walk_frame, walk_resolutions)
    walk_actions = _build_walk_forward_actions(walk_frame)
    locks = _scan_locks(lock_dir)

    config_map_path = output_dir / "config_inheritance_map.csv"
    config_path = output_dir / "config_redundancy.csv"
    duplicates_path = output_dir / "config_duplicate_groups.csv"
    walk_path = output_dir / "walk_forward_runs.csv"
    walk_actions_path = output_dir / "walk_forward_run_actions.csv"
    config_map.to_csv(config_map_path, index=False, encoding="utf-8")
    config_frame.to_csv(config_path, index=False, encoding="utf-8")
    duplicate_groups.to_csv(duplicates_path, index=False, encoding="utf-8")
    walk_frame.to_csv(walk_path, index=False, encoding="utf-8")
    walk_actions.to_csv(walk_actions_path, index=False, encoding="utf-8")

    incomplete_count = int((walk_frame["status"] != "complete").sum()) if not walk_frame.empty else 0
    resolved_incomplete_count = (
        int(((walk_frame["status"] != "complete") & walk_frame["is_resolved"]).sum())
        if not walk_frame.empty and "is_resolved" in walk_frame
        else 0
    )
    unresolved_incomplete_count = incomplete_count - resolved_incomplete_count
    active_lock_count = sum(1 for item in locks if item.get("pid_active") and not item.get("is_current_process"))
    root_config_count = int((config_map["role"] == "root").sum()) if not config_map.empty else 0
    base_config_count = int((config_map["role"] == "base").sum()) if not config_map.empty else 0
    root_extends_count = (
        int(((config_map["role"] == "root") & config_map["has_extends"]).sum())
        if not config_map.empty
        else 0
    )
    inheritance_edges = int(config_map["extends_count"].sum()) if not config_map.empty else 0
    walk_forward_resume_candidates = (
        int(walk_actions["recommended_action"].isin({"resume_or_finalize", "inspect_then_resume"}).sum())
        if not walk_actions.empty
        else 0
    )
    walk_forward_archive_review_candidates = (
        int((walk_actions["recommended_action"] == "archive_candidate_after_review").sum())
        if not walk_actions.empty
        else 0
    )
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "config_dir": str(config_dir),
        "walk_forward_dir": str(walk_forward_dir),
        "walk_forward_resolution_path": str(walk_forward_resolution_path),
        "lock_dir": str(lock_dir),
        "config_files": int(config_frame["file"].nunique()) if not config_frame.empty else 0,
        "root_config_files": root_config_count,
        "base_config_files": base_config_count,
        "root_configs_with_extends": root_extends_count,
        "root_configs_without_extends": root_config_count - root_extends_count,
        "config_inheritance_edges": inheritance_edges,
        "duplicate_config_groups": int(len(duplicate_groups)),
        "walk_forward_runs": int(len(walk_frame)),
        "incomplete_walk_forward_runs": incomplete_count,
        "resolved_walk_forward_runs": resolved_incomplete_count,
        "unresolved_incomplete_walk_forward_runs": unresolved_incomplete_count,
        "walk_forward_action_items": int(len(walk_actions)),
        "walk_forward_resume_candidates": walk_forward_resume_candidates,
        "walk_forward_archive_review_candidates": walk_forward_archive_review_candidates,
        "lock_files": int(len(locks)),
        "active_lock_files": int(active_lock_count),
        "locks": locks,
    }
    snapshot_path = output_dir / "model_build_audit_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(
        output_dir,
        config_frame,
        config_map,
        duplicate_groups,
        walk_frame,
        walk_actions,
        locks,
        snapshot,
    )
    return ModelBuildAuditResult(
        output_dir=output_dir,
        report_path=report_path,
        snapshot_path=snapshot_path,
        config_redundancy_path=config_path,
        config_map_path=config_map_path,
        walk_forward_runs_path=walk_path,
        walk_forward_actions_path=walk_actions_path,
        snapshot=snapshot,
    )
