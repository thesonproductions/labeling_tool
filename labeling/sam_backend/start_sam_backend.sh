#!/bin/bash
# Khởi động Label Studio SAM ML Backend trên Linux/macOS
export PYTHONUTF8=1
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$DIR/labeling/sam_backend/_sam_backend.py" ]; then
    cd "$DIR/labeling/sam_backend"
    python _sam_backend.py --port 9090 --host 0.0.0.0
else
    python _sam_backend.py --port 9090 --host 0.0.0.0
fi
