#!/bin/bash
# Khởi động MobileSAM ML Backend trên Linux/macOS
export PYTHONUTF8=1

# Tự động nhảy thẳng vào thư mục chứa file script này
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DIR/../.." && pwd)"
cd "$DIR"

# Kích hoạt Conda (nếu có) hoặc .venv (nếu có)
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
    conda activate labelstudio 2>/dev/null
fi

if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON_CMD="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

"$PYTHON_CMD" _sam_backend.py --port 9090 --host 0.0.0.0
