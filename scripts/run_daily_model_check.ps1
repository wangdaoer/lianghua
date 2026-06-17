$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$PromotionReviewDir = "outputs/research/allocator_promotion_with_execution_cost_20260614"

python -m quant_etf_lab daily-check --date-stamp --promotion-review-dir $PromotionReviewDir @args
exit $LASTEXITCODE
