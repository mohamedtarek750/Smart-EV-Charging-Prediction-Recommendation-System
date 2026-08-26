@echo off
REM Launches the Streamlit dashboard through the SAME interpreter that trained the
REM models.  Using the bare `streamlit` command can pick a different Python
REM installation, which fails to unpickle artifacts/models/*.joblib.
cd /d "%~dp0"
python -m streamlit run app/dashboard.py
pause
