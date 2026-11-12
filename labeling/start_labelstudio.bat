@echo off
chcp 65001 >nul
title KHOI DONG LABEL STUDIO (PORT 19090)

:: 1. Thiet lap bien moi truong tranh chan mang noi bo SSRF va MO KHOA DOC FILE CUC BO
set LABEL_STUDIO_DISABLE_SSRF_PROTECTION=true
set USE_DEFAULT_BANNED_SUBNETS=false
set ML_BLOCK_LOCAL_IP=false
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=%~d0\

:: 2. Chuyen thang vao thu muc chua file bat nay
cd /d "%~dp0"

:: 2.5. Tu dong kich hoat ban va Dual-Mode YOLO Export (kem anh + ten file sach) tren moi may tinh
python converters\patch_sdk_converter.py

:: 3. Chay Label Studio tro vao thu muc data ngay ben canh file bat
label-studio start --port 19090 --data-dir ./data

pause
