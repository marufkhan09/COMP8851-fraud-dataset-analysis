#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
ROOT=/workspace/comp8851
KIT=/workspace/EOGFD_TFINANCE_STAGE4_I1_20261023
PY="$ROOT/env/eogfd-legacy/bin/python"
REPO="$ROOT/repos/EOGFD"
LOG="$ROOT/logs/EOGFD_57212396_TFinance_STAGE4.log"
test -x "$PY"; test -f "$KIT/stage4_tfinance.py"
test -f "$ROOT/bootstrap/enable_eogfd_cuda11.sh"
mkdir -p "$ROOT/logs" "$ROOT/locks"
exec 9>"$ROOT/locks/eogfd_tfinance_stage4.lock"
flock -n 9 || { echo 'STOP: Stage4 already running'; exit 2; }
source "$ROOT/bootstrap/enable_eogfd_cuda11.sh"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export DGLBACKEND=pytorch CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
"$PY" -c 'from models.eogfd import EOGFDHeteroModel; print("I1_STAGE4_AUTHOR_MODEL_IMPORT=PASS")'
case "${1:-}" in
  --audit-only|--all) ;;
  *) echo 'Usage: bash vast_i1_stage4.sh --audit-only|--all'; exit 2;;
esac
"$PY" -u "$KIT/stage4_tfinance.py" "$1" 2>&1 | tee -a "$LOG"
echo 'I1_TFINANCE_STAGE4_LAUNCHER_EXIT=PASS'
