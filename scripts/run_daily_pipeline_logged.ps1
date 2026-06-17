$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$LogDir = Join-Path $ProjectRoot "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutLog = Join-Path $LogDir "daily_pipeline_${Stamp}.out.log"
$ErrLog = Join-Path $LogDir "daily_pipeline_${Stamp}.err.log"
$StatusPath = Join-Path $LogDir "daily_pipeline_${Stamp}.status.json"

Set-Location $ProjectRoot
$StartedAt = Get-Date
$PythonArgs = @("-m", "quant_etf_lab", "daily-pipeline") + $args

try {
    & python @PythonArgs 1> $OutLog 2> $ErrLog
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
    command = "python " + ($PythonArgs -join " ")
} | ConvertTo-Json -Depth 3 | Set-Content -Path $StatusPath -Encoding UTF8

exit $ExitCode
