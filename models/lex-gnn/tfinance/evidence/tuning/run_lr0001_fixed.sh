#!/usr/bin/env bash
set -uo pipefail

BASE=/workspace/comp8851_two_models
EVIDENCE="$BASE/evidence/tfinance/lexgnn_tuning"

source "$BASE/environment_paths/lexgnn_cuda.sh"

echo "LEX-GNN T-FINANCE CORRECTED AUPRC TUNING START"
date -u

timeout --signal=TERM --kill-after=30s 240m \
  "$BASE/venvs/lexgnn/bin/python" -u \
  "$BASE/adapters/tfinance_staging/train_tfinance_lexgnn_auprc_fixed.py" \
  --config "$BASE/configs/tfinance_tuning/lexgnn_tfinance_tr40_seed2_lr0001_fixed.json" \
  > "$EVIDENCE/lr0001_fixed.log" 2>&1

RC=$?
printf '%s\n' "$RC" > "$EVIDENCE/lr0001_fixed_exit_status.txt"

echo "EXIT CODE: $RC"
tail -n 25 "$EVIDENCE/lr0001_fixed.log"
date -u
