@echo off
echo ===================================================
echo Hoop Stats Analyzer - GUI Runner
echo ===================================================

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please run install.bat first to set up the environment.
    pause
    exit /b
)

echo Starting the backend API server (port 8000)...
start "Hoop Stats API Backend" cmd /k "venv\Scripts\python.exe server.py"

echo Starting the web frontend...
if exist "frontend-v6\" (
    cd frontend-v6
    
    if not exist "node_modules\" (
        echo Installing frontend dependencies...
        call npm install
    )
    
    echo Launching React Web GUI...
    call npm run dev
) else (
    echo [ERROR] Frontend directory "frontend-v6" not found!
    pause
)
