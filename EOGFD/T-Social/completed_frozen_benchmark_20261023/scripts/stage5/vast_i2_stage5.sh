#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
ROOT=/workspace/comp8851
KIT=/workspace/EOGFD_TSOCIAL_STAGE5_I2_20261023
PY="$ROOT/env/eogfd-legacy/bin/python"
REPO="$ROOT/repos/EOGFD"
LOG="$ROOT/logs/EOGFD_57231045_TSocial_STAGE5_HELDOUT_TEST_ONCE.log"
test "$(hostname)" = a92ee6fa0813 || { echo 'STOP: this is not I2';exit 2; }
test -x "$PY" && test -f "$KIT/stage5_tsocial.py"
test -f "$ROOT/bootstrap/enable_eogfd_cuda11.sh"
mkdir -p "$ROOT/logs" "$ROOT/locks"
exec 7>"$ROOT/locks/eogfd_tsocial_stage3.lock"
flock -n 7 || { echo 'STOP: Stage3 process active';exit 2; }
exec 8>"$ROOT/locks/eogfd_tsocial_stage4.lock"
flock -n 8 || { echo 'STOP: Stage4 process active';exit 2; }
exec 9>"$ROOT/locks/eogfd_tsocial_stage5.lock"
flock -n 9 || { echo 'STOP: Stage5 process active';exit 2; }
source "$ROOT/bootstrap/enable_eogfd_cuda11.sh"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 DGLBACKEND=pytorch PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
case "${1:-}" in
 --selftest|--audit-only|--all) test "$#" -eq 1 ;;
 *) echo 'Usage: --selftest | --audit-only | --all';exit 2;;
esac
(cd "$KIT" && sha256sum -c KIT_SHA256SUMS.txt)
"$PY" -u "$KIT/stage5_tsocial.py" "$@" 2>&1 | tee -a "$LOG"
echo "I2_TSOCIAL_STAGE5_${1#--}_LAUNCHER_EXIT=PASS"
