$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$LogDir = Join-Path $ProjectRoot "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$SkipMarketCapUpdate = $false
$ExtraArgs = @()
foreach ($Arg in $args) {
    if ($Arg -eq "--skip-market-cap-update") {
        $SkipMarketCapUpdate = $true
    } else {
        $ExtraArgs += $Arg
    }
}

$MarketCapPath = "data/processed/stock_market_cap_yi.csv"
$PromotionReviewDir = "outputs/research/allocator_promotion_with_execution_cost_20260614"

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

$PipelineArgs = @(
    "-m", "quant_etf_lab", "daily-pipeline",
    "--config", "configs/portfolio_core_satellite_quality_v2_guarded.yaml",
    "--allocator-dir", "outputs/portfolio_source_selection/main_chinext_portfolio_source_selection_validation6_v1",
    "--promotion-review-dir", $PromotionReviewDir,
    "--stock-market-cap-path", $MarketCapPath,
    "--stock-tracking-max-market-cap-yi", "1500",
    "--stock-review-outcomes-history-path", "outputs/research/stock_target_review_outcomes_history.csv",
    "--stock-review-warning-only-after-close"
) + $ExtraArgs

& python @PipelineArgs
exit $LASTEXITCODE
