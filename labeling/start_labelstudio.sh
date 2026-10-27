#!/bin/bash

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate labelstudio

export LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true

exec label-studio start \
  --port 19090 \
  --no-browser \
  --data-dir /datausers3/kttv/tien/ImageRegression/plant_journal/labeling/data