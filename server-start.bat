@echo off
title EMB-CAR ID Card Generator - Server

REM Change to the folder this .bat file lives in, regardless of where it's
REM double-clicked from -- makes this work even if the project folder moves.
cd /d "%~dp0"

REM Activate the project's virtual environment
call venv\Scripts\activate.bat

echo.
echo Starting ID Card Generator server...
echo Once running, other PCs on the office network can access it at:
echo     http://192.168.2.215:8501
echo (Run "ipconfig" in a separate window to find YOUR-PC-IP)
echo.
echo Keep this window open while the server is in use.
echo Close this window to stop the server.
echo.

streamlit run app.py --server.address 0.0.0.0

REM If Streamlit exits/crashes, keep the window open so the error is visible
REM instead of the window vanishing before you can read what went wrong.
pause