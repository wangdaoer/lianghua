$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

python -m quant_etf_lab daily-pipeline @args
exit $LASTEXITCODE
