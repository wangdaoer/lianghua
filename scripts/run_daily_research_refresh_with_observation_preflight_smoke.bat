@echo off
setlocal
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_daily_research_refresh_with_observation.ps1" --run-daily-pipeline-preflight-smoke %*
exit /b %ERRORLEVEL%
