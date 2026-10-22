#!/usr/bin/env bash
# Run only in Vast I1 JupyterLab terminal, after uploading this deployment zip to /workspace.
set -Eeuo pipefail
cd /workspace
ROOT=/workspace/comp8851
BASE=/workspace/EOGFD_TFINANCE_STAGE3_I1_20261023
PY="$ROOT/env/eogfd-legacy/bin/python"
LOG="$ROOT/logs/EOGFD_57212396_TFinance_STAGE3_TR40.log"
mkdir -p "$ROOT/logs" "$ROOT/locks"
[ -x "$PY" ] && [ -f "$BASE/stage3_tfinance.py" ] && [ -f "$ROOT/bootstrap/enable_eogfd_cuda11.sh" ]
# Exactly one worker can use the I1 EOGFD T-Finance trial tree at a time.
exec 9>"$ROOT/locks/eogfd_tfinance_stage3.lock"
flock -n 9 || { echo 'STOP: T-Finance tuning is already running. Inspect existing process.'; exit 2; }
source "$ROOT/bootstrap/enable_eogfd_cuda11.sh"
export PYTHONPATH="$ROOT/repos/EOGFD${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c 'from models.eogfd import EOGFDHeteroModel; print("I1_EOGFD_AUTHOR_MODEL_IMPORT=PASS")'
export CUDA_VISIBLE_DEVICES=0 DGLBACKEND=pytorch PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
[ "$#" -eq 1 ] || { echo 'Usage: bash vast_i1_one_command.sh --trial8|--all'; exit 2; }
case "$1" in --trial8) ARGS=(--trial 8);; --all) ARGS=(--all);; *) echo 'STOP: choose --trial8 or --all';exit 2;; esac
set -o pipefail
"$PY" -u "$BASE/stage3_tfinance.py" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
echo 'EOGFD_I1_TFINANCE_TUNING_LAUNCHER_EXIT=PASS'
