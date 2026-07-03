@echo off
REM ==========================================================
REM  Data preparation, step 1 (double-click, no coding needed)
REM  Raw Avizo Label-Analysis CSV  ->  cleaned spreadsheets:
REM    total^<Sample^>.xlsx     (input for the ML app)
REM    Quantity_^<Sample^>.xlsx (optional volume-filtered copy)
REM  A file-picker window will open to select your files.
REM ==========================================================
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Please see QUICKSTART.md in the
    echo repository root for a step-by-step installation guide.
    pause
    exit /b 1
)

python -m pip install -q pandas openpyxl
python BatchFile.py
if errorlevel 1 pause
