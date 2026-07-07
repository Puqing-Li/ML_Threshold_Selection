#!/bin/bash
# ==========================================================
#  Data preparation, step 2 (double-click, macOS)
#  Cleaned spreadsheet  ->  TomoFab input (TT_total<Sample>.xls)
#  Use this only for the training samples that go to TomoFab.
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

"$PY" -m pip install -q pandas openpyxl xlrd
"$PY" To_tomofab.py
