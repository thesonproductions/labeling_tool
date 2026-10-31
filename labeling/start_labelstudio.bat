@echo off
chcp 65001 >nul
title KHOI DONG LABEL STUDIO (PORT 19090)

:: 1. Thiet lap bien moi truong tranh chan mang noi bo SSRF
set LABEL_STUDIO_DISABLE_SSRF_PROTECTION=true
set USE_DEFAULT_BANNED_SUBNETS=false
set ML_BLOCK_LOCAL_IP=false

:: 2. Chuyen thang vao thu muc chua file bat nay
cd /d "%~dp0"

:: 3. Chay Label Studio tro vao thu muc data ngay ben canh file bat
label-studio start --port 19090 --data-dir ./data

pause