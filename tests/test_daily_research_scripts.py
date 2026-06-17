import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_research_refresh_with_observation_wires_refresh_then_observation() -> None:
    script = ROOT / "scripts" / "run_daily_research_refresh_with_observation.ps1"
    text = script.read_text(encoding="utf-8")

    assert "run_daily_research_refresh_logged.ps1" in text
    assert "allocator-observation" in text
    assert "daily_pipeline_" in text
    assert "DateStamp" in text
    assert "daily_pipeline_snapshot.json" in text
    assert "allocator_observation_$DateStamp" in text
    assert "Rollback quality-v2" in text
    assert "--observation-date" in text


def test_research_refresh_with_observation_writes_daily_run_status_summary() -> None:
    script = ROOT / "scripts" / "run_daily_research_refresh_with_observation.ps1"
    text = script.read_text(encoding="utf-8")

    assert "function Invoke-DailyRunStatusSummary" in text
    assert "python -m quant_etf_lab daily-run-status" in text
    assert 'Invoke-DailyRunStatusSummary -Reason "refresh_failed"' in text
    assert 'Invoke-DailyRunStatusSummary -Reason "observation_skipped"' in text
    assert 'Invoke-DailyRunStatusSummary -Reason "after_observation"' in text
    assert "daily_run_status_${Stamp}.status.json" in text


def test_research_refresh_with_observation_writes_wrapper_status_summary() -> None:
    script = ROOT / "scripts" / "run_daily_research_refresh_with_observation.ps1"
    text = script.read_text(encoding="utf-8")

    assert "function Write-DailyResearchWrapperStatus" in text
    assert "daily_research_refresh_with_observation_${WrapperStamp}.status.json" in text
    assert "refresh_exit_code" in text
    assert "observation_exit_code" in text
    assert "daily_run_status_exit_code" in text
    assert "pipeline_preflight_smoke_exit_code" in text
    assert "pipeline_preflight_smoke_status_path" in text
    assert "final_stage" in text
    assert "wrapper_status_path" in text


def test_research_refresh_with_observation_supports_daily_pipeline_preflight_smoke() -> None:
    script = ROOT / "scripts" / "run_daily_research_refresh_with_observation.ps1"
    text = script.read_text(encoding="utf-8")

    assert "Invoke-DailyPipelinePreflightSmoke" in text
    assert "--run-daily-pipeline-preflight-smoke" in text
    assert "pipeline_preflight_smoke_${Stamp}" in text


def test_research_refresh_with_observation_supports_auto_retry_paper_account() -> None:
    script = ROOT / "scripts" / "run_daily_research_refresh_with_observation.ps1"
    text = script.read_text(encoding="utf-8")

    assert "Resolve-LatestPipelineSnapshot" in text
    assert "Pipeline-BlockedForPaperAccount" in text
    assert "Invoke-DailyPipelineRefresh" in text
    assert "--retry-paper-account" in text
    assert "retry_paper_account_failed" in text


def test_research_refresh_with_observation_accepts_empty_refresh_args() -> None:
    script = ROOT / "scripts" / "run_daily_research_refresh_with_observation.ps1"
    text = script.read_text(encoding="utf-8")

    assert "[string[]]$RefreshArgs = @()" in text


def test_research_refresh_with_observation_bat_invokes_wrapper() -> None:
    script = ROOT / "scripts" / "run_daily_research_refresh_with_observation.bat"
    text = script.read_text(encoding="utf-8")

    assert "run_daily_research_refresh_with_observation.ps1" in text


def test_daily_research_refresh_uses_existing_market_cap_cache_when_update_fails(tmp_path: Path) -> None:
    if os.name != "nt":
        return

    cache_path = ROOT / "data" / "processed" / "stock_market_cap_yi.csv"
    created_cache = False
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("code,name,market_cap_yi,snapshot_date\n000001,fixture,1,2026-06-12\n", encoding="utf-8")
        created_cache = True

    fake_python = tmp_path / "python.cmd"
    fake_log = tmp_path / "python_calls.log"
    fake_python.write_text(
        "\n".join(
            [
                "@echo off",
                "echo %*>>\"%FAKE_PYTHON_LOG%\"",
                "echo %* | findstr /C:\"data update-market-cap\" >nul",
                "if %errorlevel%==0 exit /b 1",
                "echo %* | findstr /C:\"daily-pipeline\" >nul",
                "if %errorlevel%==0 exit /b 0",
                "exit /b 0",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env.get('PATH', '')}"
    env["FAKE_PYTHON_LOG"] = str(fake_log)

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ROOT / "scripts" / "run_daily_research_refresh.ps1"),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        if created_cache:
            cache_path.unlink(missing_ok=True)

    combined = result.stdout + result.stderr
    calls = fake_log.read_text(encoding="utf-8")
    assert result.returncode == 0, combined
    assert "Market-cap update failed; continuing with existing cache:" in combined
    assert "data update-market-cap" in calls
    assert "daily-pipeline" in calls


def test_daily_research_refresh_logged_allows_market_cap_stderr_when_cache_exists(tmp_path: Path) -> None:
    if os.name != "nt":
        return

    cache_path = ROOT / "data" / "processed" / "stock_market_cap_yi.csv"
    created_cache = False
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("code,name,market_cap_yi,snapshot_date\n000001,fixture,1,2026-06-12\n", encoding="utf-8")
        created_cache = True

    logs_dir = ROOT / "outputs" / "logs"
    before_logs = {path.name for path in logs_dir.glob("daily_research_refresh_*.json")}
    before_logs.update(path.name for path in logs_dir.glob("daily_research_refresh_*.log"))

    fake_python = tmp_path / "python.cmd"
    fake_log = tmp_path / "python_calls.log"
    fake_python.write_text(
        "\n".join(
            [
                "@echo off",
                "echo %*>>\"%FAKE_PYTHON_LOG%\"",
                "echo %* | findstr /C:\"data update-market-cap\" >nul",
                "if %errorlevel%==0 (",
                "  echo simulated market-cap outage 1>&2",
                "  exit /b 1",
                ")",
                "echo %* | findstr /C:\"daily-pipeline\" >nul",
                "if %errorlevel%==0 exit /b 0",
                "exit /b 0",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env.get('PATH', '')}"
    env["FAKE_PYTHON_LOG"] = str(fake_log)

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(ROOT / "scripts" / "run_daily_research_refresh_logged.ps1"),
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        new_statuses = sorted(
            (logs_dir / name)
            for name in ({path.name for path in logs_dir.glob("daily_research_refresh_*.status.json")} - before_logs)
        )
        assert new_statuses
        status = new_statuses[-1].read_text(encoding="utf-8")
    finally:
        after_logs = {path.name for path in logs_dir.glob("daily_research_refresh_*.json")}
        after_logs.update(path.name for path in logs_dir.glob("daily_research_refresh_*.log"))
        for name in after_logs - before_logs:
            (logs_dir / name).unlink(missing_ok=True)
        if created_cache:
            cache_path.unlink(missing_ok=True)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert '"exit_code":  0' in status
    assert "daily-pipeline" in fake_log.read_text(encoding="utf-8")


def test_research_refresh_with_observation_preflight_smoke_bat_invokes_wrapper_with_flag() -> None:
    script = ROOT / "scripts" / "run_daily_research_refresh_with_observation_preflight_smoke.bat"
    text = script.read_text(encoding="utf-8")

    assert "run_daily_research_refresh_with_observation.ps1" in text
    assert "--run-daily-pipeline-preflight-smoke" in text


def test_check_daily_research_status_scripts_invoke_cli_status_command() -> None:
    ps1_text = (ROOT / "scripts" / "check_daily_research_status.ps1").read_text(encoding="utf-8")
    bat_text = (ROOT / "scripts" / "check_daily_research_status.bat").read_text(encoding="utf-8")

    assert "python -m quant_etf_lab daily-run-status" in ps1_text
    assert "check_daily_research_status.ps1" in bat_text


def test_check_daily_research_preflight_task_status_scripts_exist_and_match() -> None:
    ps1_text = (ROOT / "scripts" / "check_daily_research_preflight_task_status.ps1").read_text(encoding="utf-8")
    bat_text = (ROOT / "scripts" / "check_daily_research_preflight_task_status.bat").read_text(encoding="utf-8")

    assert "schtasks /query /TN" in ps1_text
    assert "daily_research_refresh_with_observation" in ps1_text
    assert "No daily_research_refresh_with_observation status files found" in ps1_text
    assert "check_daily_research_preflight_task_status.ps1" in bat_text


def test_check_daily_research_preflight_task_status_handles_no_status_files() -> None:
    if os.name != "nt":
        return

    import tempfile

    with tempfile.TemporaryDirectory(prefix="check-preflight-status-empty-") as temp_root:
        temp_root_path = Path(temp_root)
        ps1 = ROOT / "scripts" / "check_daily_research_preflight_task_status.ps1"
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ps1),
            "-ProjectRoot",
            str(temp_root_path),
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        combined = result.stdout + result.stderr

        assert result.returncode == 0
        assert "Task not found:" in combined or "Status:" in combined
        assert "No daily_research_refresh_with_observation status files found in:" in combined


def test_research_refresh_with_observation_uses_scheduler_level_lock() -> None:
    script = ROOT / "scripts" / "run_daily_research_refresh_with_observation.ps1"
    text = script.read_text(encoding="utf-8")

    assert "outputs\\locks" in text
    assert "daily_research_refresh_with_observation.lock" in text
    assert "[System.IO.FileShare]::None" in text
    assert "Another daily research refresh with observation is already running" in text
    assert "$LockStream.Dispose()" in text
    assert "Remove-Item -LiteralPath $LockPath" in text


def test_research_refresh_with_observation_lock_exits_75_when_already_locked() -> None:
    if os.name != "nt":
        return

    script = ROOT / "scripts" / "run_daily_research_refresh_with_observation.ps1"
    lock_dir = ROOT / "outputs" / "locks"
    lock_path = lock_dir / "daily_research_refresh_with_observation.lock"
    logs_dir = ROOT / "outputs" / "logs"
    before_logs = {path.name for path in logs_dir.glob("daily_research_refresh_with_observation_*.status.json")}
    command = f"""
$LockDir = '{lock_dir}'
New-Item -ItemType Directory -Force -Path $LockDir | Out-Null
$LockPath = '{lock_path}'
$Script = '{script}'
$Stream = $null
try {{
  $Stream = [System.IO.File]::Open($LockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
}} catch {{
  Write-Host "LOCK_OPEN_FAILED"
}}
try {{
  & powershell -NoProfile -ExecutionPolicy Bypass -File $Script --skip-observation --skip-market-cap-update 2>&1 | ForEach-Object {{ $_.ToString() }}
  $Code = $LASTEXITCODE
  Write-Host "CHILD_EXIT_CODE=$Code"
  if ($Code -ne 75) {{ exit 1 }}
}} finally {{
  if ($Stream) {{
    $Stream.Dispose()
    Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
  }}
}}
"""
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    after_logs = {path.name for path in logs_dir.glob("daily_research_refresh_with_observation_*.status.json")}
    for name in after_logs - before_logs:
        (logs_dir / name).unlink(missing_ok=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Another daily research refresh with observation is already running" in result.stdout + result.stderr
    assert "CHILD_EXIT_CODE=75" in result.stdout


def test_daily_pipeline_preflight_smoke_bat_invokes_smoke_script() -> None:
    text = (ROOT / "scripts" / "run_daily_pipeline_preflight_smoke.bat").read_text(encoding="utf-8")
    assert "run_daily_pipeline_preflight_smoke.ps1" in text


def test_daily_pipeline_preflight_smoke_script_runs_key_cases() -> None:
    text = (ROOT / "scripts" / "run_daily_pipeline_preflight_smoke.ps1").read_text(encoding="utf-8")
    assert "daily_pipeline_preflight_smoke_result.json" in text
    assert "preflight pass path" in text
    assert "blocked preflight case" in text
    assert "blocked preflight but non-failing path" in text
    assert "--as-of-date" in text


def test_daily_pipeline_preflight_smoke_caps_as_of_date_to_allocator_curve() -> None:
    text = (ROOT / "scripts" / "run_daily_pipeline_preflight_smoke.ps1").read_text(encoding="utf-8")

    assert "function Resolve-SmokeAsOfDate" in text
    assert "data\\raw\\stocks\\000001.csv" in text
    assert "outputs\\portfolio_source_selection\\main_chinext_portfolio_source_selection_validation6_v1\\oos_equity_stitched.csv" in text
    assert "Sort-Object | Select-Object -First 1" in text


def test_preflight_smoke_and_status_script_invokes_smoke_then_status_check() -> None:
    text = (ROOT / "scripts" / "run_daily_research_preflight_smoke_and_status.bat").read_text(encoding="utf-8")

    assert "run_daily_research_preflight_smoke_and_status.ps1" in text


def test_preflight_smoke_and_status_ps1_invokes_smoke_then_status_check() -> None:
    text = (ROOT / "scripts" / "run_daily_research_preflight_smoke_and_status.ps1").read_text(encoding="utf-8")

    assert "run_daily_research_refresh_with_observation_preflight_smoke.bat" in text
    assert "check_daily_research_preflight_task_status.bat" in text
    assert "-TaskName $TaskName" in text


def test_preflight_ops_scripts_exist() -> None:
    bat_text = (ROOT / "scripts" / "run_daily_research_preflight_ops.bat").read_text(encoding="utf-8")
    ps1_text = (ROOT / "scripts" / "run_daily_research_preflight_ops.ps1").read_text(encoding="utf-8")

    assert "run_daily_research_preflight_ops.ps1" in bat_text
    assert "check_daily_research_preflight_task_status.bat" in ps1_text
    assert "run_daily_research_preflight_smoke_and_status.bat" in ps1_text
    assert "check_daily_research_status.bat" in ps1_text
    assert "run_latest_dashboard.bat" in ps1_text
    assert "NoDashboard" in ps1_text


def test_preflight_ops_script_mentions_no_dashboard_path() -> None:
    ps1_text = (ROOT / "scripts" / "run_daily_research_preflight_ops.ps1").read_text(encoding="utf-8")
    assert "if ($NoDashboard)" in ps1_text
    assert "Dashboard step skipped because -NoDashboard was set." in ps1_text
    assert "--live-preflight-fail-on-blocked" in ps1_text
    assert "$FailOnBlocked" in ps1_text
    assert "QuantETF Daily Pipeline (Preflight Ops, FailFast)" in ps1_text
    assert "-TaskName" in ps1_text


def test_preflight_task_install_scripts_exist_and_invoked() -> None:
    bat_text = (ROOT / "scripts" / "install_daily_research_preflight_task.bat").read_text(encoding="utf-8")
    ps1_text = (ROOT / "scripts" / "install_daily_research_preflight_task.ps1").read_text(encoding="utf-8")

    assert "install_daily_research_preflight_task.ps1" in bat_text
    assert "schtasks" in ps1_text
    assert "QuantETF Daily Pipeline (Preflight Smoke)" in ps1_text
    assert "run_daily_research_refresh_with_observation_preflight_smoke.bat" in ps1_text
    assert "/SC" in ps1_text
    assert "/D" in ps1_text
    assert "MON,TUE,WED,THU,FRI" in ps1_text
    assert "/ST" in ps1_text
    assert "/RU" in ps1_text
    assert "/RL" in ps1_text
    assert "HIGHEST" in ps1_text
    assert "WithStatusCheck" in ps1_text


def test_preflight_task_install_with_status_flag_uses_and_status_script() -> None:
    ps1_text = (ROOT / "scripts" / "install_daily_research_preflight_task.ps1").read_text(encoding="utf-8")
    assert "run_daily_research_preflight_smoke_and_status.bat" in ps1_text
    assert "run_daily_research_refresh_with_observation_preflight_smoke.bat" in ps1_text


def test_preflight_task_install_with_ops_uses_ops_script() -> None:
    ps1_text = (ROOT / "scripts" / "install_daily_research_preflight_task.ps1").read_text(encoding="utf-8")
    assert "run_daily_research_preflight_ops.bat" in ps1_text
    assert "Including full ops flow:" in ps1_text
    assert "-WithOps" in ps1_text


def test_preflight_task_install_dry_run() -> None:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "install_daily_research_preflight_task.ps1"),
            "-DryRun",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert "skip execution" in combined


def test_preflight_task_install_dry_run_with_status_check() -> None:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "install_daily_research_preflight_task.ps1"),
            "-DryRun",
            "-WithStatusCheck",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert "Including immediate status check: True" in combined
    assert "run_daily_research_preflight_smoke_and_status.bat" in combined


def test_preflight_task_install_dry_run_with_ops() -> None:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "install_daily_research_preflight_task.ps1"),
            "-DryRun",
            "-WithOps",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert "Including full ops flow: True" in combined
    assert "run_daily_research_preflight_ops.bat" in combined


def test_preflight_task_install_dry_run_with_status_check_and_fail_on_blocked() -> None:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "install_daily_research_preflight_task.ps1"),
            "-DryRun",
            "-WithStatusCheck",
            "-FailOnBlocked",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert "FailOnBlocked: True" in combined
    assert 'run_daily_research_preflight_smoke_and_status.bat" -FailOnBlocked' in combined


def test_preflight_task_install_dry_run_with_ops_and_fail_on_blocked() -> None:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "install_daily_research_preflight_task.ps1"),
            "-DryRun",
            "-WithOps",
            "-FailOnBlocked",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0
    assert "FailOnBlocked: True" in combined
    assert 'run_daily_research_preflight_ops.bat" -FailOnBlocked' in combined
