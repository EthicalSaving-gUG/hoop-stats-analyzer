@echo off
echo ===================================================
echo Hoop Stats Analyzer - Runner
echo ===================================================

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please run install.bat first to set up the environment.
    pause
    exit /b
)

echo Starting the backend analysis pipeline...
venv\Scripts\python.exe main.py

echo.
echo Executed finished.
pause
