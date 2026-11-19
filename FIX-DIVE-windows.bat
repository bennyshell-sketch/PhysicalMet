@echo off
REM Dive - repair and diagnose.
REM
REM Double-click this when Dive will not start.
REM It rebuilds everything from scratch and writes a full report to
REM dive-log.txt, then tries to start the app.
REM
REM Safe to run any number of times. It only deletes the .venv folder, which
REM is rebuilt automatically. Your data and settings are untouched.

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"
set "LOG=%~dp0dive-log.txt"
set "LAUNCHER="

echo.
echo   Dive - repair
echo   -------------
echo.
echo   This rebuilds everything and writes a report to:
echo     %LOG%
echo.
echo   It takes a few minutes. Leave this window open.
echo.

REM ------------------------------------------------------------ find python
py -3 --version >nul 2>&1
if not errorlevel 1 (
  set "LAUNCHER=py -3"
) else (
  python --version >nul 2>&1
  if not errorlevel 1 set "LAUNCHER=python"
)

if "!LAUNCHER!"=="" (
  echo   PROBLEM: Python is not installed, or was installed without
  echo            "Add python.exe to PATH" ticked.
  echo.
  echo   FIX: go to https://www.python.org/downloads/
  echo        download, run the installer, and TICK
  echo        "Add python.exe to PATH" on the first screen.
  echo        Then run this file again.
  echo.
  pause
  exit /b 1
)

echo   Found Python. Rebuilding...
echo.

REM ------------------------------------------------------- rebuild the venv
if exist "%VENV%" rmdir /s /q "%VENV%"

> "%LOG%" 2>&1 (
  echo === Dive repair report ===
  echo.
  echo --- folder ---
  echo %CD%
  echo.
  echo --- files present ---
  dir /b
  echo.
  echo --- python version ---
  !LAUNCHER! --version
  echo.
  echo --- creating environment ---
  !LAUNCHER! -m venv "%VENV%"
  echo.
  echo --- upgrading pip ---
  "%PY%" -m pip install --upgrade pip
  echo.
  echo --- installing dependencies ---
  "%PY%" -m pip install -r requirements.txt
  echo.
  echo --- installed packages ---
  "%PY%" -m pip list
  echo.
  echo --- import check ---
  "%PY%" -c "import streamlit, pandas, numpy, scipy, statsmodels, sklearn, plotly, yaml, openpyxl; print('core imports OK')"
  echo.
  echo --- generating data ---
  "%PY%" -m dive.generate_synthetic_data
  echo.
  echo --- loading dataset ---
  "%PY%" -c "from dive.data_pipeline import load_dataset; d=load_dataset(); print('participants:', d.report.n_participants)"
  echo.
  echo --- port check ---
  netstat -ano ^| findstr :8501
)

echo.
echo   Rebuild finished. Checking the result...
echo.

REM ------------------------------------------------------- did it work?
"%PY%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
  echo   PROBLEM: the dependencies did not install.
  echo.
  echo   Usually this is no internet, or a work firewall blocking
  echo   downloads. Try again on a home network.
  echo.
  echo   The details are in:
  echo     %LOG%
  echo.
  start notepad "%LOG%"
  pause
  exit /b 1
)

"%PY%" -c "from dive.data_pipeline import load_dataset; load_dataset()" >nul 2>&1
if errorlevel 1 (
  echo   PROBLEM: Dive installed, but could not load its data.
  echo.
  echo   The details are in:
  echo     %LOG%
  echo.
  start notepad "%LOG%"
  pause
  exit /b 1
)

echo   Everything checks out. Starting Dive.
echo.
echo   Open your browser at:  http://localhost:8501
echo.
echo   Leave this window open. Press Ctrl+C here to stop.
echo.

REM Started outside the log redirect so you can see it working.
"%PY%" -m streamlit run app/app.py --server.port 8501
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
  echo   Dive has stopped.
  pause
  exit /b 0
)

echo   ------------------------------------------------------------
echo   Dive failed to start. The error is in the lines just above.
echo.
echo   A full report was also saved to:
echo     %LOG%
echo.
echo   Opening it now - send that file on and this can be fixed.
echo   ------------------------------------------------------------
echo.
start notepad "%LOG%"
pause
exit /b 1
