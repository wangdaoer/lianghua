@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_ths_hs_a_share_export_task.ps1" %*
exit /b %ERRORLEVEL%
