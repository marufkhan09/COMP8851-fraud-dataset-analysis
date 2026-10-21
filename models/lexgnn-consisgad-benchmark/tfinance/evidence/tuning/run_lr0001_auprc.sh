#!/usr/bin/env bash
set -uo pipefail

ROOT=/workspace/comp8851_two_models
EVIDENCE="$ROOT/evidence/tfinance/lexgnn_tuning"

source "$ROOT/environment_paths/lexgnn_cuda.sh"

echo "LEX-GNN T-FINANCE LR=0.001 AUPRC TUNING START"
date -u

timeout --signal=TERM --kill-after=30s 240m \
  "$ROOT/venvs/lexgnn/bin/python" -u \
  "$ROOT/adapters/tfinance_staging/train_tfinance_lexgnn_auprc.py" \
  --config "$ROOT/configs/tfinance_tuning/lexgnn_tfinance_tr40_seed2_lr0001.json" \
  > "$EVIDENCE/lr0001_auprc.log" 2>&1

RC=$?
printf '%s\n' "$RC" > "$EVIDENCE/lr0001_auprc_exit_status.txt"

echo "EXIT CODE: $RC"
tail -n 25 "$EVIDENCE/lr0001_auprc.log"
date -u
