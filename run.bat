@echo off
REM Dive - Windows launcher.
REM
REM   Double-click this file to start Dive.
REM   run.bat check   runs the validation harness and the test suite
REM   run.bat data    regenerates the synthetic cohort
REM
REM The window stays open on any error so you can read what went wrong.

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"
set "LAUNCHER="

echo.
echo   Dive
echo   ----
echo.

REM ---------------------------------------------------------------- Python
REM "where python" is not a reliable test on Windows: the OS ships an App
REM Execution Alias that matches, does nothing, and opens the Microsoft Store.
REM Prefer the py launcher, then fall back to python, and in both cases confirm
REM it actually reports a version.

py -3 --version >nul 2>&1
if not errorlevel 1 (
  set "LAUNCHER=py -3"
) else (
  python --version >nul 2>&1
  if not errorlevel 1 set "LAUNCHER=python"
)

if "!LAUNCHER!"=="" (
  echo   Python was not found.
  echo.
  echo   Install it from:  https://www.python.org/downloads/
  echo.
  echo   IMPORTANT: on the first screen of the installer, tick
  echo   "Add python.exe to PATH" before clicking Install.
  echo   That one checkbox is the most common reason this fails.
  echo.
  echo   Then close this window, open the folder again, and
  echo   double-click run.bat.
  echo.
  pause
  exit /b 1
)

for /f "tokens=2" %%v in ('!LAUNCHER! --version 2^>^&1') do set "PYVER=%%v"
for /f "tokens=1,2 delims=." %%a in ("!PYVER!") do (
  set "MAJOR=%%a"
  set "MINOR=%%b"
)

if !MAJOR! LSS 3 goto :oldpython
if !MAJOR! EQU 3 if !MINOR! LSS 10 goto :oldpython
goto :pythonok

:oldpython
echo   Python !PYVER! was found, but Dive needs 3.10 or newer.
echo.
echo   Install a current version from https://www.python.org/downloads/
echo   and tick "Add python.exe to PATH" during setup.
echo.
pause
exit /b 1

:pythonok
echo   Using Python !PYVER!
echo.

REM ---------------------------------------------------------------- setup
if not exist "%VENV%" (
  echo   First run - setting up. This takes a few minutes, once only.
  echo   Leave this window open.
  echo.
  !LAUNCHER! -m venv "%VENV%"
  if errorlevel 1 (
    echo.
    echo   Could not create the environment.
    echo   If this folder is in OneDrive or Program Files, move it somewhere
    echo   simple like C:\Dive and try again.
    echo.
    pause
    exit /b 1
  )

  echo   Installing dependencies...
  "%PY%" -m pip install --quiet --upgrade pip
  "%PY%" -m pip install --quiet -r requirements.txt
  if errorlevel 1 (
    echo.
    echo   Installing dependencies failed. The usual causes are no internet
    echo   connection, or a firewall blocking pip.
    echo.
    pause
    exit /b 1
  )
  echo   Setup complete.
  echo.
)

if not exist "data\synthetic\dive_synthetic_cohort.xlsx" (
  echo   Generating the synthetic cohort...
  "%PY%" -m dive.generate_synthetic_data
  echo.
)

REM ---------------------------------------------------------------- run
if /i "%~1"=="check" (
  "%PY%" -m dive.validation
  echo.
  "%PY%" -m pytest tests/ -q
  echo.
  pause
  exit /b 0
)

if /i "%~1"=="data" (
  "%PY%" -m dive.generate_synthetic_data
  echo.
  pause
  exit /b 0
)

echo   Starting Dive.
echo   Your browser will open at http://localhost:8501
echo.
echo   Leave this window open while you use it.
echo   Press Ctrl+C here to stop.
echo.
"%PY%" -m streamlit run app/app.py

echo.
echo   Dive has stopped.
pause
