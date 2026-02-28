@echo off
echo ===================================================
echo Hoop Stats Analyzer - Setup Script
echo ===================================================

echo [1/4] Checking Python installation...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python 3.9+ first.
    pause
    exit /b
)

echo [2/4] Setting up Python virtual environment...
if not exist "venv\" (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo [3/4] Installing Python dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt

echo [4/4] Installing Node.js dependencies for the web server...
npm --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Node.js is not installed. You may skip this if you don't need the web UI.
) ELSE (
    npm install
)

echo ===================================================
echo Setup Complete!
echo.
echo To run the analysis pipeline:
echo   venv\Scripts\python main.py
echo.
echo To run the web dashboard:
echo   node server.js    (or npm start)
echo ===================================================
pause
