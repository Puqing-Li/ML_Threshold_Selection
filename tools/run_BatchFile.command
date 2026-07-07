#!/bin/bash
# ==========================================================
#  Data preparation, step 1 (double-click, macOS)
#  Raw Avizo Label-Analysis CSV  ->  cleaned spreadsheets:
#    total<Sample>.xlsx     (input for the ML app)
#    Quantity_<Sample>.xlsx (optional volume-filtered copy)
#  A file-picker window will open to select your files.
#  If blocked by Gatekeeper: right-click -> Open -> Open, once.
# ==========================================================
cd "$(dirname "$0")" || exit 1

PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python was not found. Please see QUICKSTART.md in the repository root."
  read -r -p "Press Return to close..."
  exit 1
fi

"$PY" -m pip install -q pandas openpyxl
"$PY" BatchFile.py
