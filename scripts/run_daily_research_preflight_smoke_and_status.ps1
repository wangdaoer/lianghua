param(
    [string]$TaskName = "QuantETF Daily Pipeline (Preflight Smoke)",
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments = @()
)

$ScriptRoot = $PSScriptRoot

& "$ScriptRoot\run_daily_research_refresh_with_observation_preflight_smoke.bat" @Arguments
$SmokeExit = $LASTEXITCODE
& "$ScriptRoot\check_daily_research_preflight_task_status.bat" -TaskName $TaskName
if ($SmokeExit -ne 0) {
    Write-Host "Preflight smoke run exited with $SmokeExit."
    if ($SmokeExit -eq 75) {
        Write-Host "Hint: lock conflict (another refresh is running)."
    }
    exit $SmokeExit
}
exit $SmokeExit
