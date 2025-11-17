@echo off
cd /d "%~dp0"

REM Se non esiste il venv, lo crea e installa i pacchetti
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
) else (
    call .venv\Scripts\activate.bat
)

echo Installing dependencies...
python.exe -m pip install --upgrade pip
python.exe -m pip install -r requirements.txt

echo Starting Magnethon Application...
streamlit run dashboard_dinamic_4.py
