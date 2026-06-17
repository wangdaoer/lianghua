@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_daily_research_preflight_task.ps1" %*
exit /b %ERRORLEVEL%
