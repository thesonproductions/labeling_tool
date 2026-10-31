#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate labelstudio 2>/dev/null

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true
export LABEL_STUDIO_DISABLE_SSRF_PROTECTION=true
export USE_DEFAULT_BANNED_SUBNETS=false
export ML_BLOCK_LOCAL_IP=false

exec label-studio start \
  --port 19090 \
  --no-browser \
  --data-dir "$DIR/data"