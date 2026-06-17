param(
    [string]$TaskName = "QuantETF Daily Pipeline (Preflight Smoke)",
    [string]$StartTime = "16:10",
    [string]$RunAsUser = "$env:USERDOMAIN\$env:USERNAME",
    [string]$RunLevel = "HIGHEST",
    [switch]$WithStatusCheck = $false,
    [switch]$WithOps = $false,
    [switch]$FailOnBlocked = $false,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$TaskName = if ($WithOps -and $TaskName -eq "QuantETF Daily Pipeline (Preflight Smoke)") {
    "QuantETF Daily Pipeline (Preflight Smoke + Ops)"
} elseif ($WithStatusCheck -and $TaskName -eq "QuantETF Daily Pipeline (Preflight Smoke)") {
    "QuantETF Daily Pipeline (Preflight Smoke + Status)"
} else {
    $TaskName
}
$TaskScript = if ($WithOps) {
    if ($WithStatusCheck) {
        Write-Host "Warning: -WithStatusCheck is redundant when -WithOps is used; it already includes status check."
    }
    "run_daily_research_preflight_ops.bat"
} elseif ($WithStatusCheck) {
    "run_daily_research_preflight_smoke_and_status.bat"
} else {
    "run_daily_research_refresh_with_observation_preflight_smoke.bat"
}
$TaskAction = Join-Path $ProjectRoot "scripts\$TaskScript"

$TaskCommand = "`"$TaskAction`""
if ($FailOnBlocked) {
    $TaskCommand += " -FailOnBlocked"
}

Write-Host "Installing scheduled task: $TaskName"
Write-Host "Run: $TaskCommand"
if ($FailOnBlocked) {
    Write-Host "FailOnBlocked: $True"
}
if ($WithStatusCheck) {
    Write-Host "Including immediate status check: $true"
}
if ($WithOps) {
    Write-Host "Including full ops flow: $true"
}
Write-Host "Time: $StartTime"
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

    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & schtasks @args 2>&1
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            foreach ($line in $output) {
                Write-Host $line
            }
        }
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
    }
    return @{
        Code = $code
        Output = ($output | Out-String)
        RunLevel = $RunLevelValue
    }
}

if ($DryRun) {
    Write-Host "[DryRun] skip execution"
    exit 0
}

$result = Invoke-SchtasksCreate -RunLevelValue $RunLevel
if (($result.Code -ne 0) -and ($RunLevel -eq "HIGHEST") -and ($result.Output -match "Access is denied")) {
    Write-Host "Access denied with run-level HIGHEST. Retrying with LIMITED."
    $result = Invoke-SchtasksCreate -RunLevelValue "LIMITED"
}

$Code = $result.Code
if ($Code -ne 0) {
    exit $Code
}

Write-Host "Scheduled task '$TaskName' created with run level '$($result.RunLevel)'."
