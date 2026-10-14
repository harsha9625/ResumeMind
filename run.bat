@echo off
title Smart ATS Checker: Resume-to-Job Matching System
echo ============================================================
echo Starting Smart ATS Checker Web Application...
echo Deep Learning and NLP Academic Project
echo ============================================================

REM Activate virtual environment if available
if exist "..\.venv\Scripts\activate.bat" (
    call "..\.venv\Scripts\activate.bat"
) else if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)

streamlit run app.py
pause
