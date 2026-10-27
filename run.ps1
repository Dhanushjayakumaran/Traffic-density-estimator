# ==============================================================================
# Traffic Density Estimator - One-Click Launcher for Windows PowerShell
# ==============================================================================

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   Traffic Density Estimator - Starting Up...     " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Ensure we are in the script's directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

# 2. Check virtual environment
$venvPython = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (-Not (Test-Path $venvPython)) {
    Write-Host "[1/3] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "[1/3] Virtual environment found (.venv)" -ForegroundColor Green
}

# 3. Check and install dependencies
Write-Host "[2/3] Checking dependencies from requirements.txt..." -ForegroundColor Yellow
& $venvPython -m pip install -r requirements.txt

# 4. Launch Streamlit app
Write-Host "[3/3] Launching Streamlit Dashboard..." -ForegroundColor Green
Write-Host "Opening in your browser at http://localhost:8501" -ForegroundColor Cyan
& $venvPython -m streamlit run app.py
