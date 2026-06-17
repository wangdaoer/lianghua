param(
    [switch]$NoDashboard,
    [switch]$FailOnBlocked,
    [string]$TaskName = "",
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments = @()
)

$ScriptRoot = $PSScriptRoot
$ProjectRoot = Resolve-Path (Join-Path $ScriptRoot "..")
Set-Location $ProjectRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$StatusTaskName = if ($TaskName) {
    $TaskName
} elseif ($FailOnBlocked) {
    "QuantETF Daily Pipeline (Preflight Ops, FailFast)"
} else {
    "QuantETF Daily Pipeline (Preflight Smoke)"
}

Write-Host "== Preflight ops step 1/3: task status check =="
& "$ScriptRoot\check_daily_research_preflight_task_status.bat" -TaskName $StatusTaskName
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Task status check returned non-zero, but continue to preflight smoke run."
}

Write-Host "== Preflight ops step 2/3: smoke + immediate status =="
$SmokeArguments = @("-TaskName", $StatusTaskName)
if ($FailOnBlocked) {
    $SmokeArguments += "--live-preflight-fail-on-blocked"
}
$SmokeArguments += $Arguments
& "$ScriptRoot\run_daily_research_preflight_smoke_and_status.bat" @SmokeArguments
$SmokeExit = $LASTEXITCODE
if ($SmokeExit -ne 0) {
    Write-Host "Preflight smoke run exited with $SmokeExit."
    if ($SmokeExit -eq 75) {
        Write-Host "Hint: another preflight run is already in progress (lock held)."
    } else {
        Write-Host "Hint: inspect latest wrapper status under outputs\logs\daily_research_refresh_with_observation_*.status.json."
    }
    exit $SmokeExit
}

Write-Host "== Preflight ops step 3/3: refresh status + optional dashboard =="
& "$ScriptRoot\check_daily_research_status.bat"
$StatusExit = $LASTEXITCODE
if ($StatusExit -ne 0) {
    Write-Host "Daily status check exited with $StatusExit."
    exit $StatusExit
}

if ($NoDashboard) {
    Write-Host "Dashboard step skipped because -NoDashboard was set."
    exit 0
}

& "$ScriptRoot\run_latest_dashboard.bat"
$DashboardExit = $LASTEXITCODE
if ($DashboardExit -ne 0) {
    Write-Host "Dashboard refresh exited with $DashboardExit."
    exit $DashboardExit
}

Write-Host "Preflight ops completed successfully."
exit 0
