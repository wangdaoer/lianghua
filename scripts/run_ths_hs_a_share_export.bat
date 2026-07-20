@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_ths_hs_a_share_export.ps1" %*
exit /b %ERRORLEVEL%
