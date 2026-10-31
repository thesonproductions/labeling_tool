@echo off
chcp 65001 >nul
echo =====================================================================
echo   KHỞI ĐỘNG LABEL STUDIO SAM ML BACKEND (MOBILESAM / PORT 9090)
echo =====================================================================

set PYTHONUTF8=1
set MODEL_DIR=%~dp0labeling\sam_backend

if exist "%~dp0labeling\sam_backend\_sam_backend.py" (
    cd "%~dp0labeling\sam_backend"
    python _sam_backend.py --port 9090 --host 0.0.0.0
) else (
    python _sam_backend.py --port 9090 --host 0.0.0.0
)

pause
