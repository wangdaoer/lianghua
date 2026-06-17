$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "========== $Message =========="
}

function Assert-Value {
    param(
        [Parameter(Mandatory = $true)] [object]$Actual,
        [Parameter(Mandatory = $true)] [object]$Expected,
        [Parameter(Mandatory = $true)] [string]$Message
    )
    if ($Actual -ne $Expected) {
        throw "ASSERTION FAILED: $Message | actual=$Actual expected=$Expected"
    }
}

function Assert-Contains {
    param(
        [Parameter(Mandatory = $true)] [string]$Text,
        [Parameter(Mandatory = $true)] [string]$Needle,
        [Parameter(Mandatory = $true)] [string]$Message
    )
    if ($Text -notlike "*$Needle*") {
        throw "ASSERTION FAILED: $Message | missing token '$Needle' in '$Text'"
    }
}

function Assert-File {
    param(
        [Parameter(Mandatory = $true)] [string]$Path,
        [Parameter(Mandatory = $true)] [string]$Message
    )
    if (-not (Test-Path $Path)) {
        throw "ASSERTION FAILED: $Message | file not found: $Path"
    }
}

function Load-Snapshot {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        throw "Missing snapshot file: $Path"
    }
    return Get-Content -Path $Path -Raw | ConvertFrom-Json
}

function Run-Preflight {
    param(
        [string]$HoldingsFile,
        [string]$TargetsFile,
        [string]$OutputDir,
        [switch]$ExpectFailOnBlocked
    )

    $cmdArgs = @(
        "-m", "quant_etf_lab", "live-shadow-preflight",
        "--holdings-file", $HoldingsFile,
        "--targets-file", $TargetsFile,
        "--cash", "100000",
        "--output-dir", $OutputDir
    )
    if ($ExpectFailOnBlocked.IsPresent) {
        $cmdArgs += "--fail-on-blocked"
    }

    $commandOutput = & python @cmdArgs 2>&1
    if ($commandOutput) {
        $commandOutput | ForEach-Object { Write-Host $_ }
    }
    return $LASTEXITCODE
}

$SmokeRoot = Join-Path $ProjectRoot ("tmp_live_shadow_smoke_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
if (Test-Path $SmokeRoot) {
    Remove-Item -Recurse -Force $SmokeRoot
}
New-Item -ItemType Directory -Force -Path $SmokeRoot | Out-Null

$HoldingsPath = Join-Path $SmokeRoot "live_holdings.csv"
$BadHoldingsPath = Join-Path $SmokeRoot "bad_holdings.csv"
$TargetsPath = Join-Path $SmokeRoot "stock_targets.csv"

@"
code,quantity,current_price
000001,0,10.0
"@ | Set-Content -Path $HoldingsPath -Encoding UTF8

@"
code,current_price
000001,10.0
"@ | Set-Content -Path $BadHoldingsPath -Encoding UTF8

@"
code,name,target_weight,target_value,target_price,layer,target_action,target_explanation
000001,TEST_TARGET_A,0.2,10000,10.0,core,target_hold,Live-shadow smoke target
"@ | Set-Content -Path $TargetsPath -Encoding UTF8

Write-Host "project_root = $ProjectRoot"
Write-Host "smoke_root = $SmokeRoot"
Write-Host "holdings = $HoldingsPath"
Write-Host "bad_holdings = $BadHoldingsPath"
Write-Host "targets = $TargetsPath"

Write-Header "Case 1: preflight passes with complete inputs"
$okOutput = Join-Path $SmokeRoot "preflight_ok"
$okExit = Run-Preflight -HoldingsFile $HoldingsPath -TargetsFile $TargetsPath -OutputDir $okOutput
Assert-Value -Actual $okExit -Expected 0 -Message "live-shadow-preflight should pass on valid inputs"
$okSnapshot = Load-Snapshot -Path (Join-Path $okOutput "live_shadow_preflight_snapshot.json")
Assert-Value -Actual $okSnapshot.status -Expected "passed" -Message "preflight snapshot status"
Assert-Value -Actual $okSnapshot.research_only -Expected $true -Message "preflight research-only flag"
Assert-Value -Actual $okSnapshot.trade_plan_status -Expected "manual_review_only" -Message "trade plan mode"
if ([double]$okSnapshot.target_gross_weight -le 0.0) {
    throw "ASSERTION FAILED: target_gross_weight should be positive for a non-empty target in ok case"
}
if ([int]$okSnapshot.order_count -lt 1) {
    throw "ASSERTION FAILED: order_count should be positive in the ok case"
}
Assert-File -Path (Join-Path $okOutput "live_shadow_preflight_report.md") -Message "preflight report exists (ok case)"

Write-Header "Case 2: preflight blocks on missing holding columns"
$blockedOutput = Join-Path $SmokeRoot "preflight_blocked"
$blockedExit = Run-Preflight -HoldingsFile $BadHoldingsPath -TargetsFile $TargetsPath -OutputDir $blockedOutput -ExpectFailOnBlocked
Assert-Value -Actual $blockedExit -Expected 7 -Message "live-shadow-preflight should exit 7 when fail-on-blocked"
$blockedSnapshot = Load-Snapshot -Path (Join-Path $blockedOutput "live_shadow_preflight_snapshot.json")
Assert-Value -Actual $blockedSnapshot.status -Expected "blocked" -Message "preflight blocked status"
Assert-Contains -Text $blockedSnapshot.error -Needle "missing required columns: quantity" -Message "preflight error message"
Assert-Value -Actual $blockedSnapshot.order_count -Expected 0 -Message "blocked case should keep order_count=0"
Assert-File -Path (Join-Path $blockedOutput "live_shadow_preflight_report.md") -Message "preflight report exists (blocked case)"

$SmokeResult = [ordered]@{
    generated_at = (Get-Date).ToString("s")
    smoke_root = $SmokeRoot
    ok_exit_code = $okExit
    blocked_exit_code = $blockedExit
    ok_snapshot_status = $okSnapshot.status
    blocked_snapshot_status = $blockedSnapshot.status
    ok_snapshot_path = (Join-Path $okOutput "live_shadow_preflight_snapshot.json")
    blocked_snapshot_path = (Join-Path $blockedOutput "live_shadow_preflight_snapshot.json")
    ok_report_path = (Join-Path $okOutput "live_shadow_preflight_report.md")
    blocked_report_path = (Join-Path $blockedOutput "live_shadow_preflight_report.md")
}
$ResultPath = Join-Path $SmokeRoot "live_shadow_preflight_smoke_result.json"
$SmokeResult | ConvertTo-Json -Depth 20 | Set-Content -Path $ResultPath -Encoding UTF8

Write-Header "Smoke result"
Write-Host "Smoke artifacts written:"
Write-Host "  $ResultPath"
Write-Host "  $($SmokeResult.ok_snapshot_path)"
Write-Host "  $($SmokeResult.blocked_snapshot_path)"
Write-Host "  $($SmokeResult.ok_report_path)"
Write-Host "  $($SmokeResult.blocked_report_path)"
Write-Host ""
Write-Host "live-shadow-preflight smoke passed."
