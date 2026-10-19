@echo off
setlocal
cd /d "%~dp0"
python -u run_extended.py
set "RUN_RESULT=%ERRORLEVEL%"
echo.
echo Validation exit code: %RUN_RESULT%
pause
exit /b %RUN_RESULT%
