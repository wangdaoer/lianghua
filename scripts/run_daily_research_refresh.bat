@echo off
setlocal
cd /d "%~dp0.."
powershell -ExecutionPolicy Bypass -File "%~dp0run_daily_research_refresh.ps1" %*
exit /b %ERRORLEVEL%
