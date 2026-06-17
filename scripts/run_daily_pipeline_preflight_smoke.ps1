$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

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

function Get-LatestCsvDate {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $null
    }
    $LastDataLine = Get-Content -Path $Path -Tail 1
    if ($LastDataLine -match "^(\d{4}-\d{2}-\d{2}),") {
        return $Matches[1]
    }
    return $null
}

function Resolve-SmokeAsOfDate {
    param([string]$ProjectRoot)

    $CandidateDates = @()
    $RawProbe = Join-Path $ProjectRoot "data\raw\stocks\000001.csv"
    $AllocatorCurve = Join-Path $ProjectRoot "outputs\portfolio_source_selection\main_chinext_portfolio_source_selection_validation6_v1\oos_equity_stitched.csv"

    $RawDate = Get-LatestCsvDate -Path $RawProbe
    if ($RawDate) {
        $CandidateDates += $RawDate
    }
    $AllocatorDate = Get-LatestCsvDate -Path $AllocatorCurve
    if ($AllocatorDate) {
        $CandidateDates += $AllocatorDate
    }
    if ($CandidateDates.Count -eq 0) {
        return "2026-06-12"
    }
    return ($CandidateDates | Sort-Object | Select-Object -First 1)
}

function Resolve-SnapshotPath {
    param(
        [Parameter(Mandatory = $true)] [string]$RunRoot,
        [Parameter(Mandatory = $false)] [string]$ProjectRoot,
        [Parameter(Mandatory = $false)] [string]$PathFromSnapshot,
        [Parameter(Mandatory = $true)] [string[]]$FallbackRelativePaths
    )
    if ($PathFromSnapshot -and (Test-Path $PathFromSnapshot)) {
        return $PathFromSnapshot
    }

    foreach ($fallbackPath in $FallbackRelativePaths) {
        if ([System.IO.Path]::IsPathRooted($fallbackPath)) {
            if (Test-Path $fallbackPath) {
                return $fallbackPath
            }
            continue
        }

        $candidateFromRunRoot = Join-Path $RunRoot $fallbackPath
        if (Test-Path $candidateFromRunRoot) {
            return $candidateFromRunRoot
        }

        if ($ProjectRoot) {
            $candidateFromProjectRoot = Join-Path $ProjectRoot $fallbackPath
            if (Test-Path $candidateFromProjectRoot) {
                return $candidateFromProjectRoot
            }
        }
    }

    $firstFallback = $FallbackRelativePaths[0]
    if ([System.IO.Path]::IsPathRooted($firstFallback)) {
        return $firstFallback
    }
    return Join-Path $RunRoot $firstFallback
}

function New-RunEnv {
    param(
        [string]$Root,
        [string]$CaseName,
        [string[]]$ExtraArgs
    )

    $RunRoot = Join-Path $Root $CaseName
    New-Item -ItemType Directory -Force -Path (Join-Path $RunRoot "pipeline") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $RunRoot "daily") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $RunRoot "paper") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $RunRoot "dashboard") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $RunRoot "sat_risk") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $RunRoot "history") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $RunRoot "history_review") | Out-Null

    $cmdArgs = @(
        "-m", "quant_etf_lab", "daily-pipeline",
        "--as-of-date", $SmokeAsOfDate,
        "--no-date-stamp",
        "--output-dir", (Join-Path $RunRoot "pipeline"),
        "--daily-check-output-dir", (Join-Path $RunRoot "daily"),
        "--paper-account-output-dir", (Join-Path $RunRoot "paper"),
        "--dashboard-output-dir", (Join-Path $RunRoot "dashboard"),
        "--history-file", (Join-Path $RunRoot (Join-Path "history" "history.csv")),
        "--history-review-output-dir", (Join-Path $RunRoot (Join-Path "history_review" "review")),
        "--satellite-risk-budget-output-dir", (Join-Path $RunRoot "sat_risk"),
        "--skip-history-review"
    )
    $cmdArgs += $ExtraArgs

    $commandOutput = & python @cmdArgs 2>&1
    if ($commandOutput) {
        $commandOutput | ForEach-Object { Write-Host $_ }
    }

    $SnapshotPath = Join-Path $RunRoot (Join-Path "pipeline" "daily_pipeline_snapshot.json")
    Assert-Value -Actual $LASTEXITCODE -Expected 0 -Message "daily-pipeline should complete in $CaseName smoke case"
    Assert-File -Path $SnapshotPath -Message "daily-pipeline snapshot exists for $CaseName"

    return [ordered]@{
        run_root = $RunRoot
        snapshot = Load-Snapshot -Path $SnapshotPath
    }
}

$SmokeAsOfDate = Resolve-SmokeAsOfDate -ProjectRoot $ProjectRoot

$SmokeRoot = Join-Path $ProjectRoot ("tmp_daily_pipeline_preflight_smoke_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
if (Test-Path $SmokeRoot) {
    Remove-Item -Recurse -Force $SmokeRoot
}
New-Item -ItemType Directory -Force -Path $SmokeRoot | Out-Null

Write-Host "project_root = $ProjectRoot"
Write-Host "smoke_root = $SmokeRoot"
Write-Host "smoke_as_of_date = $SmokeAsOfDate"

Write-Header "Step 1: build paper targets (no preflight)"
$caseBase = New-RunEnv -Root $SmokeRoot -CaseName "base" -ExtraArgs @()
$baseSnapshot = $caseBase.snapshot
Assert-Value -Actual $baseSnapshot.live_shadow_status -Expected "disabled" -Message "baseline should not run live-shadow"
$basePipelineSnapshotPath = Resolve-SnapshotPath -RunRoot $caseBase.run_root -PathFromSnapshot $baseSnapshot.pipeline_snapshot_path -FallbackRelativePaths @("pipeline\\daily_pipeline_snapshot.json")

$stockTargetsPathFromSnapshot = $baseSnapshot.paper_stock_targets_path
$stockTargetsPathFallback = Join-Path $caseBase.run_root (Join-Path "paper" "stock_targets.csv")
if (Test-Path $stockTargetsPathFromSnapshot) {
    $stockTargetsPath = $stockTargetsPathFromSnapshot
} else {
    Write-Host "INFO: snapshot paper_stock_targets_path not accessible: $stockTargetsPathFromSnapshot"
    Write-Host "INFO: fallback to run-local paper stock targets path: $stockTargetsPathFallback"
    $stockTargetsPath = $stockTargetsPathFallback
}
if (-not (Test-Path $stockTargetsPath)) {
    throw "ASSERTION FAILED: paper stock targets file not found (snapshot=$stockTargetsPathFromSnapshot, fallback=$stockTargetsPathFallback)"
}
$stockRows = Import-Csv -Path $stockTargetsPath
if (-not $stockRows -or $stockRows.Count -eq 0) {
    throw "ASSERTION FAILED: baseline stock_targets.csv is empty"
}

$passCode = $stockRows[0].code
$passPrice = $stockRows[0].close_price
if (-not $passPrice) {
    $passPrice = 10.0
}

$passHoldings = Join-Path $SmokeRoot "holdings_pass.csv"
"code,quantity,current_price`n$passCode,100,$passPrice" | Set-Content -Path $passHoldings -Encoding UTF8

$badHoldings = Join-Path $SmokeRoot "holdings_blocked.csv"
"code,current_price`n$passCode,$passPrice" | Set-Content -Path $badHoldings -Encoding UTF8

Write-Header "Step 2: preflight pass path"
$casePass = New-RunEnv -Root $SmokeRoot -CaseName "pass" -ExtraArgs @(
    "--live-shadow-preflight",
    "--live-shadow-holdings-file", $passHoldings,
    "--live-shadow-cash", "100000"
)
$passSnapshot = $casePass.snapshot
Assert-Value -Actual $passSnapshot.live_shadow_preflight_decision -Expected "passed" -Message "preflight decision in pass case"
Assert-Value -Actual $passSnapshot.live_shadow_status -Expected "completed" -Message "live shadow completed in pass case"
if (-not $passSnapshot.pipeline_next_step_stage) {
    throw "ASSERTION FAILED: pipeline_next_step_stage missing in pass case"
}
if ($passSnapshot.pipeline_next_step_stage -eq "review_live_shadow_preflight") {
    throw "ASSERTION FAILED: pass case should not require review_live_shadow_preflight"
}
$passPipelineSnapshotPath = Resolve-SnapshotPath -RunRoot $casePass.run_root -PathFromSnapshot $passSnapshot.pipeline_snapshot_path -FallbackRelativePaths @("pipeline\\daily_pipeline_snapshot.json")
$passPreflightSnapshotPath = Resolve-SnapshotPath -RunRoot $casePass.run_root -ProjectRoot $ProjectRoot -PathFromSnapshot $passSnapshot.live_shadow_preflight_snapshot_path -FallbackRelativePaths @(
    "pipeline\\live_shadow_preflight_snapshot.json",
    "outputs\\research\\live_shadow_preflight_latest\\live_shadow_preflight_snapshot.json"
)
$passPreflightReportPath = Resolve-SnapshotPath -RunRoot $casePass.run_root -ProjectRoot $ProjectRoot -PathFromSnapshot $passSnapshot.live_shadow_preflight_report_path -FallbackRelativePaths @(
    "pipeline\\live_shadow_preflight_report.md",
    "outputs\\research\\live_shadow_preflight_latest\\live_shadow_preflight_report.md"
)
Assert-File -Path $passPipelineSnapshotPath -Message "pipeline snapshot path in pass case"
Assert-File -Path $passPreflightSnapshotPath -Message "live-shadow preflight snapshot exists in pass case"
Assert-File -Path $passPreflightReportPath -Message "live-shadow preflight report exists in pass case"

Write-Header "Step 3: blocked preflight case (default fail-on-blocked)"
$caseBlocked = New-RunEnv -Root $SmokeRoot -CaseName "blocked" -ExtraArgs @(
    "--live-shadow-preflight",
    "--live-shadow-holdings-file", $badHoldings,
    "--live-shadow-cash", "100000"
)
$blockedSnapshot = $caseBlocked.snapshot
Assert-Value -Actual $blockedSnapshot.live_shadow_preflight_decision -Expected "blocked" -Message "preflight decision in blocked case"
Assert-Value -Actual $blockedSnapshot.live_shadow_status -Expected "blocked_by_preflight" -Message "live-shadow status when default fail-on-blocked"
Assert-Value -Actual $blockedSnapshot.pipeline_next_step_stage -Expected "review_live_shadow_preflight" -Message "blocked path should review live-shadow preflight"
Assert-Value -Actual $blockedSnapshot.pipeline_blocker -Expected "live_shadow_preflight_blocked" -Message "pipeline blocker in blocked case"
Assert-Value -Actual $blockedSnapshot.live_shadow_order_count -Expected 0 -Message "order_count should be zero in blocked_by_preflight"
$blockedPipelineSnapshotPath = Resolve-SnapshotPath -RunRoot $caseBlocked.run_root -PathFromSnapshot $blockedSnapshot.pipeline_snapshot_path -FallbackRelativePaths @("pipeline\\daily_pipeline_snapshot.json")
$blockedPreflightReportPath = Resolve-SnapshotPath -RunRoot $caseBlocked.run_root -ProjectRoot $ProjectRoot -PathFromSnapshot $blockedSnapshot.live_shadow_preflight_report_path -FallbackRelativePaths @(
    "pipeline\\live_shadow_preflight_report.md",
    "outputs\\research\\live_shadow_preflight_latest\\live_shadow_preflight_report.md"
)
$blockedPreflightSnapshotPath = Resolve-SnapshotPath -RunRoot $caseBlocked.run_root -ProjectRoot $ProjectRoot -PathFromSnapshot $blockedSnapshot.live_shadow_preflight_snapshot_path -FallbackRelativePaths @(
    "pipeline\\live_shadow_preflight_snapshot.json",
    "outputs\\research\\live_shadow_preflight_latest\\live_shadow_preflight_snapshot.json"
)
$blockedPreflightError = ""
if ($blockedSnapshot.live_shadow_preflight_error) {
    $blockedPreflightError = $blockedSnapshot.live_shadow_preflight_error
} elseif ($blockedPreflightReportPath -and (Test-Path $blockedPreflightReportPath)) {
    $blockedPreflightError = Get-Content -Path $blockedPreflightReportPath -Raw
}
if ($blockedPreflightError -notlike "*missing required columns: quantity*") {
    throw "ASSERTION FAILED: blocked preflight error text | expected substring 'missing required columns: quantity' not found in: $blockedPreflightError"
}
Assert-File -Path $blockedPipelineSnapshotPath -Message "pipeline snapshot path in blocked case"
Assert-File -Path $blockedPreflightReportPath -Message "preflight report exists in blocked case"
Assert-File -Path $blockedPreflightSnapshotPath -Message "preflight snapshot exists in blocked case"

Write-Header "Step 4: blocked preflight but non-failing path"
$caseAllow = New-RunEnv -Root $SmokeRoot -CaseName "allow" -ExtraArgs @(
    "--live-shadow-preflight",
    "--no-live-shadow-preflight-fail-on-blocked",
    "--live-shadow-holdings-file", $badHoldings,
    "--live-shadow-cash", "100000"
)
$allowSnapshot = $caseAllow.snapshot
Assert-Value -Actual $allowSnapshot.live_shadow_preflight_decision -Expected "blocked" -Message "preflight decision should still be blocked"
Assert-Value -Actual $allowSnapshot.live_shadow_status -Expected "failed" -Message "live-shadow status when non-failing preflight"
Assert-Value -Actual $allowSnapshot.pipeline_next_step_stage -Expected "review_live_shadow_preflight" -Message "non-failing blocked path should still enter review stage"
Assert-Value -Actual $allowSnapshot.pipeline_blocker -Expected "live_shadow_preflight_blocked" -Message "pipeline blocker in non-failing blocked case"
$allowPipelineSnapshotPath = Resolve-SnapshotPath -RunRoot $caseAllow.run_root -PathFromSnapshot $allowSnapshot.pipeline_snapshot_path -FallbackRelativePaths @("pipeline\\daily_pipeline_snapshot.json")
$allowPreflightSnapshotPath = Resolve-SnapshotPath -RunRoot $caseAllow.run_root -ProjectRoot $ProjectRoot -PathFromSnapshot $allowSnapshot.live_shadow_preflight_snapshot_path -FallbackRelativePaths @(
    "pipeline\\live_shadow_preflight_snapshot.json",
    "outputs\\research\\live_shadow_preflight_latest\\live_shadow_preflight_snapshot.json"
)
Assert-File -Path $allowPipelineSnapshotPath -Message "pipeline snapshot path in allow case"
Assert-File -Path $allowPreflightSnapshotPath -Message "preflight snapshot exists in allow case"

$SmokeResult = [ordered]@{
    generated_at = (Get-Date).ToString("s")
    smoke_root = $SmokeRoot
    base_snapshot_path = $basePipelineSnapshotPath
    pass_snapshot_path = $passPipelineSnapshotPath
    blocked_snapshot_path = $blockedPipelineSnapshotPath
    allow_snapshot_path = $allowPipelineSnapshotPath
    pass_preflight_status = $passSnapshot.live_shadow_preflight_decision
    blocked_preflight_status = $blockedSnapshot.live_shadow_preflight_decision
    allow_preflight_status = $allowSnapshot.live_shadow_preflight_decision
    pass_live_shadow_status = $passSnapshot.live_shadow_status
    blocked_live_shadow_status = $blockedSnapshot.live_shadow_status
    allow_live_shadow_status = $allowSnapshot.live_shadow_status
}
$ResultPath = Join-Path $SmokeRoot "daily_pipeline_preflight_smoke_result.json"
$SmokeResult | ConvertTo-Json -Depth 20 | Set-Content -Path $ResultPath -Encoding UTF8

Write-Header "Smoke result"
Write-Host "Smoke artifacts written:"
Write-Host "  $ResultPath"
Write-Host "  $($SmokeResult.base_snapshot_path)"
Write-Host "  $($SmokeResult.pass_snapshot_path)"
Write-Host "  $($SmokeResult.blocked_snapshot_path)"
Write-Host "  $($SmokeResult.allow_snapshot_path)"
Write-Host ""
Write-Host "daily-pipeline preflight smoke passed."
