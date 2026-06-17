@echo off
setlocal
cd /d "%~dp0.."
python -m quant_etf_lab daily-pipeline %*
exit /b %ERRORLEVEL%
