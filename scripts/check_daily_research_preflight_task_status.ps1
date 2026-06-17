param(
    [string]$TaskName = "QuantETF Daily Pipeline (Preflight Smoke)",
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$LogDir = Join-Path $ProjectRoot "outputs\logs"
$StatusPrefix = "daily_research_refresh_with_observation"
$Latest = Get-ChildItem -LiteralPath $LogDir -Filter "${StatusPrefix}_*.status.json" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

Write-Host "== Scheduled Task =="
$TaskQuery = schtasks /query /TN $TaskName /FO LIST /V 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Task not found: $TaskName"
} else {
    $TaskQuery | Select-String -Pattern "Status:|Last Run Time:|Last Result:|Next Run Time:|Task To Run:|Scheduled Task State:" | ForEach-Object {
        Write-Host $_
    }
}

if (-not $Latest) {
    Write-Host "No daily_research_refresh_with_observation status files found in: $LogDir"
    exit 0
}

$LatestStatus = Get-Content -Path $Latest.FullName -Raw | ConvertFrom-Json
Write-Host ""
Write-Host "== Latest Wrapper Run =="
Write-Host "File: $($Latest.FullName)"
Write-Host "Started: $($LatestStatus.started_at)"
Write-Host "Finished: $($LatestStatus.finished_at)"
Write-Host "Exit: $($LatestStatus.exit_code)"
Write-Host "Final Stage: $($LatestStatus.final_stage)"
Write-Host "Refresh Exit: $($LatestStatus.refresh_exit_code)"
Write-Host "Preflight Exit: $($LatestStatus.pipeline_preflight_smoke_exit_code)"
Write-Host "Observation Exit: $($LatestStatus.observation_exit_code)"
Write-Host "Daily Run-Status Exit: $($LatestStatus.daily_run_status_exit_code)"
if ($LatestStatus.pipeline_preflight_smoke_status_path) {
    Write-Host "Preflight Smoke Status: $($LatestStatus.pipeline_preflight_smoke_status_path)"
}
if ($LatestStatus.observation_status_path) {
    Write-Host "Observation Status: $($LatestStatus.observation_status_path)"
}
if ($LatestStatus.daily_run_status_status_path) {
    Write-Host "Daily Run-Status Snapshot: $($LatestStatus.daily_run_status_snapshot_path)"
}
