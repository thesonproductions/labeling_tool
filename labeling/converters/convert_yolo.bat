@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."

echo ==========================================================
echo   CHUYEN DOI DU LIEU YOLO -^> LABEL STUDIO JSON
echo ==========================================================

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" labeling\converters\yolo_converter.py %*
) else (
    python labeling\converters\yolo_converter.py %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [LOI] Qua trinh chuyen doi gap su co. Vui long kiem tra thong bao tren.
    pause
)