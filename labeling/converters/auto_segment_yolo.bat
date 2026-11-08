@echo off
chcp 65001 >nul
set PYTHONUTF8=1
title TỰ ĐỘNG BỌC VIỀN POLYGON BẰNG MOBILESAM TỪ THƯ MỤC YOLO

echo ===============================================================================
echo   CÔNG CỤ TỰ ĐỘNG BỌC VIỀN ĐA GIÁC (MOBILESAM OFFLINE BATCH SEGMENTATION)
echo ===============================================================================
echo.
echo [1] Chạy chuyển đổi trực tiếp thư mục YOLO sang JSON đã có sẵn 100%% Polygon.
echo [2] Không cần khởi động server Label Studio hay ML Backend.
echo [3] File JSON sinh ra chỉ cần Import vào Label Studio là có sẵn viền ôm sát!
echo.
echo -------------------------------------------------------------------------------

cd /d "%~dp0"
python yolo_to_sam_polygons.py

pause
