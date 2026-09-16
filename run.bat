@echo off
REM Dive - one-command start on Windows.
REM
REM   run.bat          launch the app
REM   run.bat check    run the validation harness and tests
REM   run.bat data     regenerate the synthetic cohort

cd /d "%~dp0"
set VENV=.venv
set PY=%VENV%\Scripts\python.exe

where python >nul 2>nul
if errorlevel 1 (
  echo Python 3 is required but was not found.
  echo Install it from https://www.python.org/downloads/ and tick "Add Python to PATH".
  exit /b 1
)

if not exist "%VENV%" (
  echo First run - setting up. This takes a few minutes and only happens once.
  python -m venv %VENV%
  "%PY%" -m pip install --quiet --upgrade pip
  echo Installing dependencies...
  "%PY%" -m pip install --quiet -r requirements.txt
  echo Done.
)

if not exist "data\synthetic\dive_synthetic_cohort.xlsx" (
  echo Generating the synthetic cohort...
  "%PY%" -m dive.generate_synthetic_data
)

if "%1"=="check" (
  "%PY%" -m dive.validation
  "%PY%" -m pytest tests/ -q
  exit /b 0
)
if "%1"=="data" (
  "%PY%" -m dive.generate_synthetic_data
  exit /b 0
)

echo.
echo Starting Dive. Your browser should open at http://localhost:8501
echo Press Ctrl+C here to stop it.
echo.
"%PY%" -m streamlit run app/app.py
