@echo off
setlocal

cd /d "%~dp0\.."
python -m quant_etf_lab daily-check --date-stamp --promotion-review-dir outputs\research\allocator_promotion_with_execution_cost_20260614 %*
exit /b %ERRORLEVEL%
