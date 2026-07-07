#!/bin/bash
# ==========================================================
#  ML Threshold Selection - double-click launcher (macOS)
#  No coding required. On first run this installs the required
#  libraries (needs internet), then starts the app.
#  If double-clicking is blocked by Gatekeeper: right-click the
#  file -> Open -> Open, once.
# ==========================================================
cd "$(dirname "$0")" || exit 1

PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python was not found on this Mac."
  echo "Please install Python 3.8 or newer from https://www.python.org/downloads/"
  echo "(the python.org installer includes Tkinter). Then double-click this file again."
  read -r -p "Press Return to close..."
  exit 1
fi

if ! "$PY" -c "import tkinter" >/dev/null 2>&1; then
  echo "Your Python installation is missing Tkinter (the graphical toolkit)."
  echo "Install Python from python.org (it includes Tkinter),"
  echo "or with Homebrew run:  brew install python-tk"
  read -r -p "Press Return to close..."
  exit 1
fi

echo "Checking required libraries (quick after the first run)..."
if ! "$PY" -m pip install -q -r requirements.txt; then
  echo "Could not install the required libraries. Check your internet connection and try again."
  read -r -p "Press Return to close..."
  exit 1
fi

echo "Launching ML Threshold Selection..."
"$PY" main.py
