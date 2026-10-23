#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
ROOT=/workspace/comp8851
KIT=/workspace/EOGFD_TSOCIAL_STAGE4_I2_20261023
PY="$ROOT/env/eogfd-legacy/bin/python"
REPO="$ROOT/repos/EOGFD"
LOG="$ROOT/logs/EOGFD_57231045_TSocial_STAGE4_FROZEN_FINALS.log"
test "$(hostname)" = a92ee6fa0813 || { echo 'STOP: not I2 57231045';exit 2; }
test -x "$PY" && test -f "$KIT/stage4_tsocial.py"
test -f "$ROOT/bootstrap/enable_eogfd_cuda11.sh"
test -f /workspace/EOGFD_TSOCIAL_STAGE3_V3_I2_20261023/stage3_tsocial_v3.py
mkdir -p "$ROOT/logs" "$ROOT/locks"
# Block concurrent Stage3 and Stage4 access to the same GPU & result tree.
exec 8>"$ROOT/locks/eogfd_tsocial_stage3.lock"
flock -n 8 || { echo 'STOP: Stage3 is still running. Do not interrupt.';exit 2; }
exec 9>"$ROOT/locks/eogfd_tsocial_stage4.lock"
flock -n 9 || { echo 'STOP: Stage4 is already running';exit 2; }
source "$ROOT/bootstrap/enable_eogfd_cuda11.sh"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 DGLBACKEND=pytorch PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
case "${1:-}" in
  --audit-only|--all) test "$#" = 1 || exit 2 ;;
  --one) test "$#" = 3 || exit 2 ;;
  *) echo 'Usage: bash vast_i2_stage4.sh --audit-only | --one TR40 2 | --all';exit 2;;
esac
cd "$KIT"
sha256sum -c KIT_SHA256SUMS.txt
"$PY" -c 'import torch,dgl; from models.eogfd import EOGFDHeteroModel; assert torch.cuda.is_available(); print("I2_TSOCIAL_STAGE4_PINNED_AUTHOR_IMPORT=PASS",flush=True)'
"$PY" -u "$KIT/stage4_tsocial.py" "$@" 2>&1 | tee -a "$LOG"
echo 'I2_TSOCIAL_STAGE4_LAUNCHER_EXIT=PASS'
