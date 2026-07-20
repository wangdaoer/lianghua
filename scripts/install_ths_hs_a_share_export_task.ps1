param(
    [string]$TaskName = "Codex_THS_HS_A_Share_Daily_Export",
    [string]$StartTime = "15:45",
    [string]$RunAsUser = "$env:USERDOMAIN\$env:USERNAME",
    [string]$RunLevel = "HIGHEST",
    [string]$ExportRoot = "D:\codex\daily-market-data\ths_exports",
    [switch]$NoGui,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$TaskAction = Join-Path $ProjectRoot "scripts\run_ths_hs_a_share_export.bat"
$TaskCommand = "`"$TaskAction`" -ExportRoot `"$ExportRoot`""
if ($NoGui) {
    $TaskCommand += " -NoGui"
}

Write-Host "Installing scheduled task: $TaskName"
Write-Host "Run: $TaskCommand"
Write-Host "Time: $StartTime"
Write-Host "Days: MON,TUE,WED,THU,FRI"
Write-Host "User: $RunAsUser"
Write-Host "Run-level: $RunLevel"

function Invoke-SchtasksCreate {
    param([string]$RunLevelValue)

    $args = @(
        "/create",
        "/F",
        "/TN", $TaskName,
        "/TR", $TaskCommand,
        "/SC", "WEEKLY",
        "/D", "MON,TUE,WED,THU,FRI",
        "/ST", $StartTime,
        "/RU", $RunAsUser,
        "/RL", $RunLevelValue
    )

    Write-Host "Command: schtasks $($args -join ' ')"

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $Output = & schtasks @args 2>&1
        $Code = $LASTEXITCODE
        if ($Code -ne 0) {
            foreach ($Line in $Output) {
                Write-Host $Line
            }
        }
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    return @{
        Code = $Code
        Output = ($Output | Out-String)
        RunLevel = $RunLevelValue
    }
}

if ($DryRun) {
    Write-Host "[DryRun] skip execution"
    exit 0
}

$Result = Invoke-SchtasksCreate -RunLevelValue $RunLevel
if (($Result.Code -ne 0) -and ($RunLevel -eq "HIGHEST") -and ($Result.Output -match "Access is denied")) {
    Write-Host "Access denied with run-level HIGHEST. Retrying with LIMITED."
    $Result = Invoke-SchtasksCreate -RunLevelValue "LIMITED"
}

if ($Result.Code -ne 0) {
    exit $Result.Code
}

Write-Host "Scheduled task '$TaskName' created with run level '$($Result.RunLevel)'."
