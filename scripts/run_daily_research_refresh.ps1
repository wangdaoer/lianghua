$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$LogDir = Join-Path $ProjectRoot "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$SkipMarketCapUpdate = $false
$SkipMarketDataPreflight = $false
$SkipTriggerSync = $false
$StockTrackingMaxMarketCapYi = 0
$ExtraArgs = @()
for ($Index = 0; $Index -lt $args.Length; $Index++) {
    $Arg = $args[$Index]
    if ($Arg -eq "--skip-market-cap-update") {
        $SkipMarketCapUpdate = $true
        continue
    }
    if ($Arg -eq "--skip-market-data-preflight") {
        $SkipMarketDataPreflight = $true
        continue
    }
    if ($Arg -eq "--skip-trigger-sync") {
        $SkipTriggerSync = $true
        continue
    }
    if ($Arg -eq "--stock-tracking-max-market-cap-yi") {
        $NextIndex = $Index + 1
        if ($NextIndex -ge $args.Length) {
            throw "--stock-tracking-max-market-cap-yi requires a numeric value."
        }
        $StockTrackingMaxMarketCapYi = $args[$NextIndex]
        $Index = $NextIndex
        continue
    }
    $ExtraArgs += $Arg
}

$MarketCapPath = "data/processed/stock_market_cap_yi.csv"
$MarketDataPreflightPath = Join-Path $PSScriptRoot "check_daily_market_data_preflight.py"
$MarketDataPreflightMinRowCount = 1
$MarketDataPreflightStatusPath = $null
$ExpectedTriggerTradeDate = $null
$TriggerMonitorSourceRoot = "D:\codex\2026-06-18-investment-strategy\outputs"
$PromotionReviewDir = "outputs/research/allocator_promotion_stock_only_highyield_group_pos8margin10_fixed_strict_20260627"

if (-not $SkipMarketDataPreflight) {
    $MarketDataPreflightStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $MarketDataPreflightOutLog = Join-Path $LogDir "daily_market_data_preflight_${MarketDataPreflightStamp}.out.log"
    $MarketDataPreflightErrLog = Join-Path $LogDir "daily_market_data_preflight_${MarketDataPreflightStamp}.err.log"
    $MarketDataPreflightStatusPath = Join-Path $LogDir "daily_market_data_preflight_${MarketDataPreflightStamp}.status.json"
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python $MarketDataPreflightPath --output $MarketDataPreflightStatusPath --min-row-count $MarketDataPreflightMinRowCount 1> $MarketDataPreflightOutLog 2> $MarketDataPreflightErrLog
        $MarketDataPreflightExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($MarketDataPreflightExitCode -ne 0) {
        if (Test-Path -LiteralPath $MarketDataPreflightStatusPath) {
            [Console]::Error.WriteLine((Get-Content -Path $MarketDataPreflightStatusPath -Raw))
        }
        if (Test-Path -LiteralPath $MarketDataPreflightErrLog) {
            [Console]::Error.WriteLine((Get-Content -Path $MarketDataPreflightErrLog -Raw))
        }
        exit $MarketDataPreflightExitCode
    }
    if (Test-Path -LiteralPath $MarketDataPreflightStatusPath) {
        try {
            $MarketDataPreflightStatus = Get-Content -LiteralPath $MarketDataPreflightStatusPath -Raw | ConvertFrom-Json
            $ExpectedTriggerTradeDate = [string]$MarketDataPreflightStatus.trade_date
        } catch {
            Write-Warning "Unable to parse market-data preflight status for trigger freshness audit: $MarketDataPreflightStatusPath"
        }
    }
}

if (-not $SkipTriggerSync) {
    $TriggerSyncStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $TriggerSyncOutLog = Join-Path $LogDir "trigger_sync_${TriggerSyncStamp}.out.log"
    $TriggerSyncErrLog = Join-Path $LogDir "trigger_sync_${TriggerSyncStamp}.err.log"
    $TriggerSyncArgs = @("-m", "quant_etf_lab", "trigger-sync")
    if (Test-Path -LiteralPath $TriggerMonitorSourceRoot) {
        $TriggerReportFilter = "thsdk_strategy_monitor_*.md"
        if ($ExpectedTriggerTradeDate) {
            $TriggerReportFilter = "thsdk_strategy_monitor_${ExpectedTriggerTradeDate}_*.md"
        }
        $LatestExternalTriggerReport = Get-ChildItem -LiteralPath $TriggerMonitorSourceRoot -Filter $TriggerReportFilter -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($LatestExternalTriggerReport) {
            $TriggerSyncArgs += @("--report-path", [string]$LatestExternalTriggerReport.FullName)
            $ExternalTriggerJournalPath = Join-Path $TriggerMonitorSourceRoot "signal_journal\signal_journal.csv"
            if (Test-Path -LiteralPath $ExternalTriggerJournalPath) {
                $TriggerSyncArgs += @("--journal-path", $ExternalTriggerJournalPath)
            }
        }
    }
    if ($ExpectedTriggerTradeDate) {
        $TriggerSyncArgs += @("--expected-trade-date", $ExpectedTriggerTradeDate)
    }
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python @TriggerSyncArgs 1> $TriggerSyncOutLog 2> $TriggerSyncErrLog
        $TriggerSyncExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($TriggerSyncExitCode -ne 0) {
        Write-Warning "Trigger monitor sync failed; continuing with dashboard freshness gates active (exit code $TriggerSyncExitCode, stderr $TriggerSyncErrLog)"
    } else {
        $TriggerSyncSnapshotPath = "D:\codex\outputs\trigger_reports\latest_trigger_sync_snapshot.json"
        if (Test-Path -LiteralPath $TriggerSyncSnapshotPath) {
            try {
                $TriggerSyncSnapshot = Get-Content -LiteralPath $TriggerSyncSnapshotPath -Raw | ConvertFrom-Json
                if ($TriggerSyncSnapshot.trade_date_matches_expected -eq $false) {
                    Write-Warning "Trigger monitor stale: expected trade date $($TriggerSyncSnapshot.expected_trade_date), synced trade date $($TriggerSyncSnapshot.trade_date), source $($TriggerSyncSnapshot.source_report_path)"
                }
            } catch {
                Write-Warning "Unable to parse trigger sync snapshot: $TriggerSyncSnapshotPath"
            }
        }
    }
}

if (-not $SkipMarketCapUpdate) {
    $MarketCapStamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $MarketCapOutLog = Join-Path $LogDir "market_cap_update_${MarketCapStamp}.out.log"
    $MarketCapErrLog = Join-Path $LogDir "market_cap_update_${MarketCapStamp}.err.log"
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & python -m quant_etf_lab data update-market-cap --output $MarketCapPath 1> $MarketCapOutLog 2> $MarketCapErrLog
        $MarketCapExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($MarketCapExitCode -ne 0) {
        $ExistingMarketCapCache = Get-Item -LiteralPath $MarketCapPath -ErrorAction SilentlyContinue
        if ($ExistingMarketCapCache -and $ExistingMarketCapCache.Length -gt 0) {
            Write-Warning "Market-cap update failed; continuing with existing cache: $MarketCapPath (exit code $MarketCapExitCode, stderr $MarketCapErrLog)"
        } else {
            if (Test-Path -LiteralPath $MarketCapErrLog) {
                [Console]::Error.WriteLine((Get-Content -Path $MarketCapErrLog -Raw))
            }
            exit $MarketCapExitCode
        }
    }
}

$WeakSentimentTrialDateStamp = Get-Date -Format "yyyyMMdd"
if ($ExpectedTriggerTradeDate) {
    $WeakSentimentTrialDateStamp = ([string]$ExpectedTriggerTradeDate) -replace "-", ""
}
$WeakSentimentTrialDir = "outputs/research/weak_sentiment_opportunity_trial_$WeakSentimentTrialDateStamp"

$PipelineArgs = @(
    "-m", "quant_etf_lab", "daily-pipeline",
    "--output-dir", "outputs/research/daily_pipeline_latest",
    "--no-date-stamp",
    "--config", "configs/portfolio_core_source_selection_quality_reversal_highyield_dd50_highgain_dd40_stock_only.yaml",
    "--allocator-dir", "outputs/portfolio_source_selection/main_chinext_source_selection_highyield_group_dd40_stock_only_cap30_pos8margin10_fixed_v1",
    "--promotion-review-dir", $PromotionReviewDir,
    "--stock-market-cap-path", $MarketCapPath,
    "--stock-tracking-max-market-cap-yi", $StockTrackingMaxMarketCapYi,
    "--stock-review-outcomes-history-path", "outputs/research/stock_target_review_outcomes_history.csv",
    "--stock-review-warning-only-after-close",
    "--publish-fast-decision",
    "--publish-stock-outcome-gate",
    "--stock-outcome-gate-output-dir", "outputs/research/stock_outcome_gate_latest",
    "--stock-outcome-gate-primary-horizon", "5d",
    "--stock-outcome-gate-relaxed-caution-multiplier", "0.5",
    "--stock-outcome-gate-relaxed-prefer-trial-weight", "0.20",
    "--stock-outcome-gate-weak-output-dir", $WeakSentimentTrialDir,
    "--stock-outcome-gate-weak-single-weight-cap", "0.05",
    "--stock-outcome-gate-weak-total-weight-cap", "0.20"
) + $ExtraArgs

& python @PipelineArgs
$PipelineExitCode = $LASTEXITCODE
if ($PipelineExitCode -ne 0) {
    exit $PipelineExitCode
}
exit 0
