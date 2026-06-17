@echo off
setlocal

cd /d "%~dp0\.."
python -m quant_etf_lab dashboard %*
exit /b %ERRORLEVEL%
