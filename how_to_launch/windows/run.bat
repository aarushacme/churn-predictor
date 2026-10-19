@echo off
title Telecom Customer Churn Predictor
:: Navigate to project root directory
cd /d "%~dp0..\.." 2>nul || cd /d "%~dp0.." 2>nul || cd /d "%~dp0"

echo ====================================================
echo Starting Telecom Customer Churn Predictor...
echo Project Folder: %CD%
echo ====================================================
echo.

:: Detect Python executable
where python >nul 2>&1
if %errorlevel% equ 0 (
    set PY_CMD=python
) else (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        set PY_CMD=py
    ) else (
        echo [ERROR] Python not found in system PATH.
        echo Please install Python 3.10+ and add it to your system PATH.
        pause
        exit /b 1
    )
)

echo Step 1: Installing required packages...
%PY_CMD% -m pip install -r requirements.txt
echo.
echo Step 2: Launching Streamlit Frontend...
echo Application will open in your browser at http://localhost:8501
%PY_CMD% -m streamlit run frontend/app.py
pause
