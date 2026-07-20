param(
    [string]$TradeDate = "",
    [string]$ExportRoot = "D:\codex\daily-market-data\ths_exports",
    [string]$RawFile = "",
    [string]$AutomationConfig = "",
    [string]$AppPath = "",
    [string]$RawExtension = ".xls",
    [int]$MinRowCount = 5000,
    [switch]$Force,
    [switch]$NoGui,
    [switch]$AllowWeekend,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$LogDir = Join-Path $ProjectRoot "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutLog = Join-Path $LogDir "ths_hs_a_share_export_${Stamp}.out.log"
$ErrLog = Join-Path $LogDir "ths_hs_a_share_export_${Stamp}.err.log"
$StatusPath = Join-Path $LogDir "ths_hs_a_share_export_${Stamp}.status.json"
$ScriptPath = Join-Path $ProjectRoot "scripts\ths_hs_a_share_export.py"

$ArgsList = @(
    $ScriptPath,
    "--export-root", $ExportRoot,
    "--raw-extension", $RawExtension,
    "--min-row-count", [string]$MinRowCount
)
if ($TradeDate) {
    $ArgsList += @("--trade-date", $TradeDate)
}
if ($RawFile) {
    $ArgsList += @("--raw-file", $RawFile)
}
if ($AutomationConfig) {
    $ArgsList += @("--automation-config", $AutomationConfig)
}
if ($AppPath) {
    $ArgsList += @("--app-path", $AppPath)
}
if ($Force) {
    $ArgsList += "--force"
}
if ($AllowWeekend) {
    $ArgsList += "--allow-weekend"
}
if (-not $NoGui) {
    $ArgsList += "--gui"
}

$CommandText = "python " + (($ArgsList | ForEach-Object { if ($_ -match "\s") { '"' + $_ + '"' } else { $_ } }) -join " ")
Write-Host "THS HS A-share export command: $CommandText"
Write-Host "Export root: $ExportRoot"
Write-Host "Logs: $OutLog / $ErrLog"

if ($DryRun) {
    Write-Host "[DryRun] skip execution"
    [pscustomobject]@{
        started_at = (Get-Date).ToString("s")
        finished_at = (Get-Date).ToString("s")
        exit_code = 0
        dry_run = $true
        project_root = [string]$ProjectRoot
        export_root = $ExportRoot
        command = $CommandText
    } | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8
    exit 0
}

$StartedAt = Get-Date
$ExitCode = 1
try {
    Push-Location $ProjectRoot
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python @ArgsList 1> $OutLog 2> $ErrLog
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
} catch {
    $_ | Out-String | Set-Content -Path $ErrLog -Encoding UTF8
    $ExitCode = 1
} finally {
    Pop-Location
}

$FinishedAt = Get-Date
[pscustomobject]@{
    started_at = $StartedAt.ToString("s")
    finished_at = $FinishedAt.ToString("s")
    exit_code = $ExitCode
    project_root = [string]$ProjectRoot
    export_root = $ExportRoot
    trade_date = $TradeDate
    raw_file = $RawFile
    automation_config = $AutomationConfig
    raw_extension = $RawExtension
    gui_enabled = -not $NoGui
    stdout_log = $OutLog
    stderr_log = $ErrLog
    command = $CommandText
} | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath -Encoding UTF8

Write-Host "Status JSON: $StatusPath"
exit $ExitCode
