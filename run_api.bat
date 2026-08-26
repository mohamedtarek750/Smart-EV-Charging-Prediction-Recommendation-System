@echo off
REM Launches the FastAPI service (docs at http://127.0.0.1:8000/docs).
cd /d "%~dp0"
python -m uvicorn app.api:app --reload
pause
