$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$LogDir = Join-Path $ProjectRoot "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutLog = Join-Path $LogDir "daily_research_refresh_${Stamp}.out.log"
$ErrLog = Join-Path $LogDir "daily_research_refresh_${Stamp}.err.log"
$StatusPath = Join-Path $LogDir "daily_research_refresh_${Stamp}.status.json"
$RefreshScript = Join-Path $PSScriptRoot "run_daily_research_refresh.ps1"

Set-Location $ProjectRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$StartedAt = Get-Date
$CommandArgs = @("-ExecutionPolicy", "Bypass", "-File", $RefreshScript) + $args

try {
    & powershell @CommandArgs 1> $OutLog 2> $ErrLog
    $ExitCode = $LASTEXITCODE
} catch {
    $_ | Out-String | Set-Content -Path $ErrLog -Encoding UTF8
    $ExitCode = 1
}

$FinishedAt = Get-Date
[pscustomobject]@{
    started_at = $StartedAt.ToString("s")
    finished_at = $FinishedAt.ToString("s")
    exit_code = $ExitCode
    project_root = [string]$ProjectRoot
    stdout_log = $OutLog
    stderr_log = $ErrLog
    command = "powershell " + ($CommandArgs -join " ")
} | ConvertTo-Json -Depth 3 | Set-Content -Path $StatusPath -Encoding UTF8

exit $ExitCode
