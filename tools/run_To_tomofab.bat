@echo off
REM ==========================================================
REM  Data preparation, step 2 (double-click, no coding needed)
REM  Cleaned spreadsheet -> TomoFab input (TT_^<Sample^>.xls)
REM  Use this only for the training samples that go to TomoFab.
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

python -m pip install -q pandas openpyxl xlrd
python To_tomofab.py
if errorlevel 1 pause
