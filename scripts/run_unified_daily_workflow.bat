@echo off
setlocal
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_unified_daily_workflow.ps1" %*
exit /b %ERRORLEVEL%
