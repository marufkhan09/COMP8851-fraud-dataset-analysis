#!/usr/bin/env bash
set -uo pipefail

ROOT=/workspace/comp8851_two_models
PYTHON="$ROOT/venvs/lexgnn/bin/python"
ADAPTER="$ROOT/adapters/lexgnn/train_yelpchi_unified.py"
CONFIG_DIR="$ROOT/configs/lexgnn/tuning_yelpchi_tr40_seed2"
RESULT_ROOT="$ROOT/runs/lexgnn/yelpchi/tuning_tr40_seed2"
LOG_DIR="$ROOT/logs/lexgnn/yelpchi/tuning_tr40_seed2"
EVIDENCE="$ROOT/evidence/tuning/lexgnn/yelpchi"

source "$ROOT/environment_paths/lexgnn_cuda.sh"

export DGLBACKEND=pytorch
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONHASHSEED=2

TRIALS=(
  trial_01_author_baseline
  trial_02_lr_0p001
  trial_03_lr_0p01
  trial_04_beta_0p25
  trial_05_beta_1p0
)

TOTAL=${#TRIALS[@]}
COMPLETED=()
SKIPPED=()
FAILED=()
START_TIME=$(date +%s)

progress_bar() {
  CURRENT=$1
  TOTAL_COUNT=$2
  WIDTH=30
  FILLED=$((CURRENT * WIDTH / TOTAL_COUNT))
  EMPTY=$((WIDTH - FILLED))
  PERCENT=$((CURRENT * 100 / TOTAL_COUNT))

  printf "["
  printf "%${FILLED}s" "" | tr " " "#"
  printf "%${EMPTY}s" "" | tr " " "-"
  printf "] %d/%d (%3d%%)" \
    "$CURRENT" "$TOTAL_COUNT" "$PERCENT"
}

for INDEX in "${!TRIALS[@]}"
do
  TRIAL=${TRIALS[$INDEX]}
  CURRENT=$((INDEX + 1))
  CONFIG="$CONFIG_DIR/${TRIAL}.json"
  TRIAL_RESULTS="$RESULT_ROOT/$TRIAL"
  LOG="$LOG_DIR/${TRIAL}.log"

  echo ""
  echo "------------------------------------------------------------"
  printf "OVERALL PROGRESS "
  progress_bar "$CURRENT" "$TOTAL"
  echo ""
  echo "Trial: $TRIAL"
  echo "------------------------------------------------------------"

  mkdir -p "$TRIAL_RESULTS"

  EXISTING=$(
    find "$TRIAL_RESULTS" \
      -type f \
      -name summary.json \
      -print |
    sort |
    tail -n 1
  )

  if [ -n "$EXISTING" ]; then
    echo "Status: SKIPPED — completed result already exists."
    echo "Summary: $EXISTING"
    SKIPPED+=("$TRIAL")
    continue
  fi

  "$PYTHON" "$ADAPTER" \
    --config "$CONFIG" \
    2>&1 |
  tee "$LOG"

  STATUS=${PIPESTATUS[0]}

  COMPLETED_SUMMARY=$(
    find "$TRIAL_RESULTS" \
      -type f \
      -name summary.json \
      -print |
    sort |
    tail -n 1
  )

  if [ "$STATUS" -eq 0 ] &&
     [ -n "$COMPLETED_SUMMARY" ]; then
    COMPLETED+=("$TRIAL")
    echo "Status: COMPLETED"
    echo "Summary: $COMPLETED_SUMMARY"
  else
    FAILED+=("$TRIAL")
    echo "Status: FAILED"
    echo "Log: $LOG"
    break
  fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

COMPLETED_TEXT="${COMPLETED[*]}"
SKIPPED_TEXT="${SKIPPED[*]}"
FAILED_TEXT="${FAILED[*]}"

COMPLETED_TEXT="$COMPLETED_TEXT" \
SKIPPED_TEXT="$SKIPPED_TEXT" \
FAILED_TEXT="$FAILED_TEXT" \
ELAPSED="$ELAPSED" \
EVIDENCE="$EVIDENCE" \
python3 - <<'PY_STATUS'
import json
import os
from pathlib import Path

def words(variable):
    return [
        item
        for item in os.environ.get(variable, "").split()
        if item
    ]

status = {
    "completed": words("COMPLETED_TEXT"),
    "skipped_as_already_complete": words("SKIPPED_TEXT"),
    "failed": words("FAILED_TEXT"),
    "elapsed_seconds": int(os.environ["ELAPSED"]),
    "test_evaluated": False,
}

output = (
    Path(os.environ["EVIDENCE"]) /
    "tuning_batch_status.json"
)

output.write_text(
    json.dumps(status, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print(json.dumps(status, indent=2))
PY_STATUS

echo ""
echo "============================================================"
echo "LEX-GNN YELPCHI TUNING STATUS"
echo "============================================================"
echo "Completed: ${COMPLETED[*]}"
echo "Skipped: ${SKIPPED[*]}"
echo "Failed: ${FAILED[*]}"
echo "Elapsed seconds: $ELAPSED"

PROCESSED=$((${#COMPLETED[@]} + ${#SKIPPED[@]}))

if [ "${#FAILED[@]}" -eq 0 ] &&
   [ "$PROCESSED" -eq "$TOTAL" ]; then
  echo "LEX-GNN YELPCHI TUNING: COMPLETE"
  echo "Test dataset evaluated: NO"
  exit 0
fi

echo "LEX-GNN YELPCHI TUNING: INCOMPLETE"
exit 1
