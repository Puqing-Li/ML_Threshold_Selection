@echo off
REM ==========================================================
REM  ML Threshold Selection - double-click launcher (Windows)
REM  No coding required: this window will guide you.
REM ==========================================================
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on this computer.
    echo.
    echo Please install Python 3.8 or newer from:
    echo     https://www.python.org/downloads/
    echo IMPORTANT: during installation, tick "Add python.exe to PATH".
    echo Then double-click this file again.
    echo.
    pause
    exit /b 1
)

python -c "import tkinter" >nul 2>nul
if errorlevel 1 (
    echo Your Python installation is missing Tkinter [the graphical toolkit].
    echo Install Python from python.org [it includes Tkinter],
    echo or in Anaconda run:  conda install tk
    echo.
    pause
    exit /b 1
)

echo Checking required libraries [quick after the first run]...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo Could not install the required libraries.
    echo Check your internet connection and try again.
    echo.
    pause
    exit /b 1
)

echo Launching ML Threshold Selection...
python main.py
if errorlevel 1 pause
