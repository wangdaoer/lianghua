from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_etf_lab.ths_export_source import (  # noqa: E402
    DEFAULT_THS_EXPORT_ROOT,
    normalize_ths_export,
    normalize_trade_date,
    ths_export_paths,
    write_ths_export_status,
)


def _today_trade_date() -> str:
    return date.today().strftime("%Y-%m-%d")


def _is_weekend(trade_date: str) -> bool:
    parsed = datetime.strptime(trade_date, "%Y-%m-%d").date()
    return parsed.weekday() >= 5


def _load_automation_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing THS automation config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"THS automation config must be a JSON object: {path}")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"THS automation config requires a non-empty steps array: {path}")
    return payload


def _powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _replace_tokens(value: Any, token_values: dict[str, str]) -> str:
    text = str(value)
    for key, replacement in token_values.items():
        text = text.replace("{" + key + "}", replacement)
    return text


def _run_gui_automation(
    raw_target: Path,
    automation_config: Path,
    app_path: str | None = None,
    timeout_seconds: int = 300,
) -> None:
    config = _load_automation_config(automation_config)
    token_values = {"raw_file": str(raw_target), "raw_dir": str(raw_target.parent)}
    resolved_app_path = app_path or str(config.get("app_path") or "")
    window_title = str(config.get("window_title") or "THS")
    pre_delay = float(config.get("pre_export_delay_seconds") or 3.0)
    wait_for_file_seconds = int(config.get("wait_for_file_seconds") or timeout_seconds)

    lines = [
        "$ErrorActionPreference = 'Stop'",
        "Add-Type @'",
        "using System;",
        "using System.Runtime.InteropServices;",
        "public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }",
        "public class NativeMouse {",
        "  [DllImport(\"user32.dll\")] public static extern IntPtr GetForegroundWindow();",
        "  [DllImport(\"user32.dll\")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);",
        "  [DllImport(\"user32.dll\")] public static extern bool SetCursorPos(int X, int Y);",
        "  [DllImport(\"user32.dll\")] public static extern void mouse_event(uint flags, uint dx, uint dy, uint data, UIntPtr extraInfo);",
        "}",
        "'@",
        "function Invoke-WindowClick {",
        "  param([int]$X, [int]$Y, [string]$Button = 'left')",
        "  $hwnd = [NativeMouse]::GetForegroundWindow()",
        "  $rect = New-Object RECT",
        "  if (-not [NativeMouse]::GetWindowRect($hwnd, [ref]$rect)) { throw 'Cannot get foreground window rect.' }",
        "  [NativeMouse]::SetCursorPos($rect.Left + $X, $rect.Top + $Y) | Out-Null",
        "  Start-Sleep -Milliseconds 120",
        "  if ($Button -eq 'right') {",
        "    [NativeMouse]::mouse_event(0x0008, 0, 0, 0, [UIntPtr]::Zero)",
        "    [NativeMouse]::mouse_event(0x0010, 0, 0, 0, [UIntPtr]::Zero)",
        "  } else {",
        "    [NativeMouse]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)",
        "    [NativeMouse]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)",
        "  }",
        "}",
        "$ws = New-Object -ComObject WScript.Shell",
    ]
    if resolved_app_path:
        lines.append(f"Start-Process -FilePath {_powershell_literal(resolved_app_path)} | Out-Null")
        lines.append(f"Start-Sleep -Seconds {max(1, int(pre_delay))}")
    lines.extend(
        [
            f"$activated = $ws.AppActivate({_powershell_literal(window_title)})",
            "if (-not $activated) { throw 'Tonghuashun window activation failed.' }",
        ]
    )

    for step in config["steps"]:
        if not isinstance(step, dict):
            raise ValueError("Each THS automation step must be a JSON object.")
        if "sleep_seconds" in step:
            lines.append(f"Start-Sleep -Milliseconds {int(float(step['sleep_seconds']) * 1000)}")
        if "set_clipboard" in step:
            text = _replace_tokens(step["set_clipboard"], token_values)
            lines.append(f"Set-Clipboard -Value {_powershell_literal(text)}")
        if "send_keys" in step:
            keys = _replace_tokens(step["send_keys"], token_values)
            lines.append(f"$ws.SendKeys({_powershell_literal(keys)})")
        if "click" in step:
            click = step["click"]
            if not isinstance(click, dict):
                raise ValueError("click step must be an object with x/y/button fields.")
            x = int(click.get("x", 0))
            y = int(click.get("y", 0))
            button = str(click.get("button", "left")).lower()
            if button not in {"left", "right"}:
                raise ValueError("click.button must be left or right.")
            lines.append(f"Invoke-WindowClick -X {x} -Y {y} -Button {_powershell_literal(button)}")
        if "app_activate" in step:
            title = _replace_tokens(step["app_activate"], token_values)
            lines.append(f"$null = $ws.AppActivate({_powershell_literal(title)})")

    lines.extend(
        [
            f"$target = {_powershell_literal(str(raw_target))}",
            f"$deadline = (Get-Date).AddSeconds({wait_for_file_seconds})",
            "while ((Get-Date) -lt $deadline) {",
            "  if (Test-Path -LiteralPath $target) { exit 0 }",
            "  Start-Sleep -Seconds 1",
            "}",
            "throw \"THS export did not create expected file: $target\"",
        ]
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "\n".join(lines)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_seconds + 30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "THS GUI automation failed.").strip())


def _copy_raw_file(source: Path, target: Path, force: bool) -> None:
    if target.exists() and not force:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    shutil.copy2(source, target)


def _raw_target_with_suffix(default_raw_target: Path, suffix: str) -> Path:
    normalized_suffix = suffix.lower().strip()
    if not normalized_suffix:
        return default_raw_target
    if not normalized_suffix.startswith("."):
        normalized_suffix = "." + normalized_suffix
    if normalized_suffix not in {".xlsx", ".xls", ".csv", ".txt"}:
        raise ValueError("raw extension must be one of: .xlsx, .xls, .csv, .txt")
    return default_raw_target.with_suffix(normalized_suffix)


def _existing_raw_export(default_raw_target: Path) -> Path | None:
    for suffix in (default_raw_target.suffix, ".xls", ".csv", ".txt", ".xlsx"):
        candidate = default_raw_target.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export/normalize Tonghuashun HS A-share daily data.")
    parser.add_argument("--trade-date", default=None, help="Trade date YYYY-MM-DD or YYYYMMDD. Default: today.")
    parser.add_argument("--export-root", default=str(DEFAULT_THS_EXPORT_ROOT), help="THS export root directory.")
    parser.add_argument("--raw-file", default=None, help="Existing raw THS export file to ingest.")
    parser.add_argument("--automation-config", default=None, help="JSON SendKeys automation config. Default: <export-root>/automation.json.")
    parser.add_argument("--app-path", default=None, help="Optional Tonghuashun executable path for GUI automation.")
    parser.add_argument("--raw-extension", default=".xls", help="Raw export extension for GUI runs. Default: .xls.")
    parser.add_argument("--min-row-count", type=int, default=5000, help="Minimum normalized rows required for success.")
    parser.add_argument("--gui", action="store_true", help="Run calibrated GUI automation when the raw file is missing.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing raw/normalized files.")
    parser.add_argument("--allow-weekend", action="store_true", help="Allow running on Saturday/Sunday.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trade_date = normalize_trade_date(args.trade_date or _today_trade_date())
    export_root = Path(args.export_root)
    default_raw_target, normalized_path, _ = ths_export_paths(trade_date, export_root)
    raw_target = _raw_target_with_suffix(default_raw_target, args.raw_extension)

    if _is_weekend(trade_date) and not args.allow_weekend:
        write_ths_export_status(
            trade_date=trade_date,
            status="skipped_non_trading_weekend",
            message="Weekend date skipped; no THS export attempted.",
            export_root=export_root,
            raw_path=raw_target,
            normalized_path=normalized_path,
        )
        print(f"Skipped weekend date: {trade_date}")
        return 0

    try:
        export_root.mkdir(parents=True, exist_ok=True)
        (export_root / "raw").mkdir(parents=True, exist_ok=True)
        (export_root / "normalized").mkdir(parents=True, exist_ok=True)
        (export_root / "manifests").mkdir(parents=True, exist_ok=True)

        if args.raw_file:
            source_raw = Path(args.raw_file)
            raw_target = _raw_target_with_suffix(default_raw_target, source_raw.suffix)
            _copy_raw_file(source_raw, raw_target, force=args.force)
        elif args.force or not (existing_raw := _existing_raw_export(default_raw_target)):
            if not args.gui:
                raise FileNotFoundError(
                    f"Missing raw THS export and --gui was not set: {raw_target}"
                )
            automation_config = Path(args.automation_config) if args.automation_config else export_root / "automation.json"
            _run_gui_automation(
                raw_target=raw_target,
                automation_config=automation_config,
                app_path=args.app_path,
            )
        else:
            raw_target = existing_raw

        result = normalize_ths_export(
            raw_path=raw_target,
            trade_date=trade_date,
            export_root=export_root,
            min_row_count=args.min_row_count,
        )
        print(f"THS export normalized: {result.normalized_path} rows={result.row_count}")
        return 0
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        write_ths_export_status(
            trade_date=trade_date,
            status="failed",
            message=str(exc),
            export_root=export_root,
            raw_path=raw_target,
            normalized_path=normalized_path,
            extra={"error_type": type(exc).__name__, "failed_at": datetime.now().isoformat(timespec="seconds")},
        )
        print(f"THS export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
