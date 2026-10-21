#!/usr/bin/env bash
set -uo pipefail

ROOT=/workspace/comp8851_two_models
EVIDENCE="$ROOT/evidence/tfinance/lexgnn_tuning"

source "$ROOT/environment_paths/lexgnn_cuda.sh"

PYTHON="$ROOT/venvs/lexgnn/bin/python"
ADAPTER="$ROOT/adapters/tfinance_staging/train_tfinance_lexgnn.py"
CONFIG="$ROOT/configs/tfinance_tuning/lexgnn_tfinance_tr40_seed2_baseline100.json"

LOG="$EVIDENCE/baseline100.log"
STATUS="$EVIDENCE/baseline100_exit_status.txt"

echo "===== LEX-GNN BASELINE STARTED ====="
date -u

timeout --signal=TERM --kill-after=30s 240m \
  "$PYTHON" -u "$ADAPTER" \
  --config "$CONFIG" \
  > "$LOG" 2>&1

RC=$?
printf '%s\n' "$RC" > "$STATUS"

echo "===== EXIT CODE: $RC ====="
tail -n 35 "$LOG"
date -u
