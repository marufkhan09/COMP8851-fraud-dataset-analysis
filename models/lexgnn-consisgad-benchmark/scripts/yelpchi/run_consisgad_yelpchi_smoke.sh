#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/comp8851_two_models
source "$ROOT/environment_paths/consisgad_cuda.sh"
export DGLBACKEND=pytorch
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONHASHSEED=2

mkdir -p "$ROOT/logs/consisgad/yelpchi"

"$ROOT/venvs/consisgad/bin/python" \
  "$ROOT/adapters/consisgad/train_yelpchi_unified.py" \
  --config "$ROOT/configs/consisgad/yelpchi_tr40_seed2_smoke.json" \
  2>&1 | tee "$ROOT/logs/consisgad/yelpchi/tr40_seed2_two_epoch.log"
