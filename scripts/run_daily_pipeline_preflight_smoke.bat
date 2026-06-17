@echo off
setlocal
cd /d "%~dp0.."
powershell -ExecutionPolicy Bypass -File scripts\\run_daily_pipeline_preflight_smoke.ps1 %*
exit /b %ERRORLEVEL%
