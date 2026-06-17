@echo off
setlocal
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_daily_research_preflight_ops.ps1" %*
exit /b %ERRORLEVEL%
