#!/usr/bin/env bash
set -e

# Tự động nhảy về thư mục gốc của dự án (labeling_tool)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Ưu tiên dùng Python trong môi trường ảo (.venv) nếu có
if [ -x ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "=========================================================="
echo "  CHUYỂN ĐỔI DỮ LIỆU YOLO -> LABEL STUDIO JSON            "
echo "=========================================================="

# Truyền toàn bộ tham số (nếu có) hoặc mở hộp thoại chọn thư mục nếu để trống
"$PYTHON_CMD" labeling/converters/yolo_converter.py "$@"
