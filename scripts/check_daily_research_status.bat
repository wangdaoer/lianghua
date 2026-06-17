@echo off
setlocal
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_daily_research_status.ps1" %*
exit /b %ERRORLEVEL%
