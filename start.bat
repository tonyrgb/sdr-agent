@echo off
cd /d "%~dp0"

:: Check for .env
if not exist ".env" (
    echo [SDR Agent] .env not found. Copying .env.example to .env...
    copy .env.example .env
    echo [SDR Agent] Please edit .env and add your API keys, then run this script again.
    pause
    exit /b 1
)

:: Install / upgrade dependencies
echo [SDR Agent] Installing dependencies...
python -m pip install -r requirements.txt --quiet

:: Start the server
echo [SDR Agent] Starting server at http://127.0.0.1:8000
echo [SDR Agent] Press Ctrl+C to stop.
echo.
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

pause
