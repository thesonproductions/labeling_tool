#!/bin/bash

# 1. Kích hoạt môi trường Conda (nếu có) hoặc .venv (nếu có)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DIR/.." && pwd)"
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
    conda activate labelstudio 2>/dev/null
elif [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# 2. Nhảy vào thư mục chứa script
cd "$DIR"

# 2.5. Tự động kích hoạt bản vá Dual-Mode YOLO Export (kèm ảnh + tên file sạch) trên Linux/macOS
if command -v python3 &>/dev/null; then
    python3 "$DIR/converters/patch_sdk_converter.py"
else
    python "$DIR/converters/patch_sdk_converter.py"
fi

# 3. Mở khóa chặn mạng nội bộ (SSRF) cho ML Backend
export LABEL_STUDIO_DISABLE_SSRF_PROTECTION=true
export USE_DEFAULT_BANNED_SUBNETS=false
export ML_BLOCK_LOCAL_IP=false

# 4. MỞ KHÓA ĐỌC FILE CỤC BỘ (Tương ứng với file .bat)
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
# Trên Linux/Mac, thư mục gốc là "/" để cho phép đọc ảnh ở mọi nơi trên máy
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="/"

# 5. Chạy Label Studio
exec label-studio start \
  --port 19090 \
  --no-browser \
  --data-dir "$DIR/data"
