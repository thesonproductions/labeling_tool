#!/bin/bash
# Khởi động MobileSAM ML Backend trên Linux/macOS
export PYTHONUTF8=1

# Tự động nhảy thẳng vào thư mục chứa file script này
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Kích hoạt Conda nếu có dùng môi trường labelstudio
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate labelstudio 2>/dev/null

python _sam_backend.py --port 9090 --host 0.0.0.0
