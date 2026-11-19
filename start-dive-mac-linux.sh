#!/usr/bin/env bash
# Dive - one-command start.
#
#   ./start-dive-mac-linux.sh            launch the app
#   ./start-dive-mac-linux.sh check      run the validation harness and tests
#   ./start-dive-mac-linux.sh data       regenerate the synthetic cohort
#
# Creates a local virtual environment on first run and installs into it.
# Nothing is installed system-wide, and nothing is sent anywhere.

set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required but was not found."
  echo "Install it from https://www.python.org/downloads/ and run this again."
  exit 1
fi

VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
MAJOR=${VERSION%%.*}; MINOR=${VERSION##*.}
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]; }; then
  echo "Python 3.10 or newer is required; found $VERSION."
  exit 1
fi

if [ ! -d "$VENV" ]; then
  echo "First run - setting up. This takes a few minutes and only happens once."
  python3 -m venv "$VENV"
  "$PY" -m pip install --quiet --upgrade pip
  echo "Installing dependencies..."
  "$PY" -m pip install --quiet -r requirements.txt
  echo "Done."
fi

if [ ! -f "data/synthetic/dive_synthetic_cohort.xlsx" ]; then
  echo "Generating the synthetic cohort..."
  "$PY" -m dive.generate_synthetic_data
fi

case "${1:-app}" in
  app)
    echo
    echo "Starting Dive. Your browser should open at http://localhost:8501"
    echo "Press Ctrl+C here to stop it."
    echo
    exec "$PY" -m streamlit run app/app.py
    ;;
  check)
    echo "Running the validation harness..."
    "$PY" -m dive.validation
    echo
    echo "Running the test suite..."
    "$PY" -m pytest tests/ -q
    ;;
  data)
    "$PY" -m dive.generate_synthetic_data
    ;;
  *)
    echo "Usage: ./start-dive-mac-linux.sh [app|check|data]"
    exit 1
    ;;
esac
