@echo off
title Traffic Density Estimator
echo ==================================================
echo    Traffic Density Estimator - Starting Up...
echo ==================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment (.venv)...
    python -m venv .venv
) else (
    echo [1/3] Virtual environment found.
)

echo [2/3] Installing dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo [3/3] Launching Streamlit Dashboard...
".venv\Scripts\python.exe" -m streamlit run app.py
pause
