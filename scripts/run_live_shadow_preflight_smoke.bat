@echo off
setlocal
cd /d "%~dp0.."
powershell -ExecutionPolicy Bypass -File scripts\run_live_shadow_preflight_smoke.ps1 %*
exit /b %ERRORLEVEL%
