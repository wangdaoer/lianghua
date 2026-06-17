$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

$config = "configs/ashare_main_chinext_multifactor_stable.yaml"
$cacheDir = "outputs/cache_runs/main_chinext_stable"
$runId = "main_chinext_stable_full_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss")

Write-Host "Project: $projectRoot"
Write-Host "Config: $config"
Write-Host "Run id: $runId"

python scripts/cache_market_data.py `
  --config $config `
  --output-dir $cacheDir `
  --audit-only

python -m quant_etf_lab walk-forward `
  --config $config `
  --preset main-chinext-stable `
  --skip-missing `
  --run-id-prefix $runId
