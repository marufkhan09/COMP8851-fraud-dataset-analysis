#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
ROOT=/workspace/comp8851
BASE=/workspace/EOGFD_TSOCIAL_STAGE3_V3_I2_20261023
PY="$ROOT/env/eogfd-legacy/bin/python"
REPO="$ROOT/repos/EOGFD"
LOG="$ROOT/logs/EOGFD_57231045_TSocial_STAGE3_V3_TR40.log"
test "$(hostname)" = a92ee6fa0813 || { echo STOP_WRONG_I2_HOST; exit 2; }
test -x "$PY"; test -f "$BASE/stage3_tsocial_v3.py"
test -f "$ROOT/bootstrap/enable_eogfd_cuda11.sh"
test -f /workspace/EOGFD_TSOCIAL_I2_MEMDIAG_20261023/tsocial_select_diagnostic.py
test -f /workspace/EOGFD_TSOCIAL_STAGE3_I2_20261023/stage3_tsocial.py
echo "a34673e909da2c421ba3e200cfa7320a8c811cb2e78be3c428ac22f41b41ddd7  /workspace/EOGFD_TSOCIAL_STAGE3_I2_20261023/stage3_tsocial.py" | sha256sum -c -
mkdir -p "$ROOT/logs" "$ROOT/locks"
exec 9>"$ROOT/locks/eogfd_tsocial_stage3.lock"
flock -n 9 || { echo 'STOP: existing Stage 3 process owns lock';exit 2; }
source "$ROOT/bootstrap/enable_eogfd_cuda11.sh"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 DGLBACKEND=pytorch PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
"$PY" -c 'from models.eogfd import EOGFDHeteroModel; print("I2_STAGE3_V3_AUTHOR_MODEL_IMPORT=PASS")'
[ "$#" -eq 1 ] || { echo 'Usage: bash vast_i2_stage3_v2.sh --trial8|--all';exit 2; }
case "$1" in --trial8) ARGS=(--trial 8);; --all) ARGS=(--all);; *) echo STOP_BAD_STAGE3_OPTION;exit 2;; esac
"$PY" -u "$BASE/stage3_tsocial_v3.py" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
echo 'I2_TSOCIAL_STAGE3_V3_LAUNCHER_EXIT=PASS'
