#!/usr/bin/env bash
set -uo pipefail

ROOT=/workspace/comp8851_two_models
LEX_PYTHON="$ROOT/venvs/lexgnn/bin/python"
CONSIS_PYTHON="$ROOT/venvs/consisgad/bin/python"
LEX_ADAPTER="$ROOT/adapters/lexgnn/train_amazon_unified.py"
CONSIS_ADAPTER="$ROOT/adapters/consisgad/train_amazon_unified.py"
LOG_DIR="$ROOT/logs/final_amazon"
EVIDENCE="$ROOT/evidence/final_runs/amazon"
STATUS_FILE="$EVIDENCE/amazon_final_batch_status.json"

mkdir -p "$LOG_DIR" "$EVIDENCE"

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export PYTHONHASHSEED=2

progress_bar() {
  CURRENT=$1
  TOTAL=$2
  WIDTH=30
  FILLED=$((CURRENT * WIDTH / TOTAL))
  EMPTY=$((WIDTH - FILLED))
  PERCENT=$((CURRENT * 100 / TOTAL))

  printf "["
  printf "%${FILLED}s" "" | tr " " "#"
  printf "%${EMPTY}s" "" | tr " " "-"
  printf "] %d/%d (%3d%%)" \
    "$CURRENT" "$TOTAL" "$PERCENT"
}

duration_text() {
  TOTAL_SECONDS=$1
  HOURS=$((TOTAL_SECONDS / 3600))
  MINUTES=$(((TOTAL_SECONDS % 3600) / 60))
  SECONDS=$((TOTAL_SECONDS % 60))

  printf "%02dh:%02dm:%02ds" \
    "$HOURS" "$MINUTES" "$SECONDS"
}

write_status() {
  COMPLETED_TEXT=$(IFS=,; echo "${COMPLETED[*]}")
  SKIPPED_TEXT=$(IFS=,; echo "${SKIPPED[*]}")
  FAILED_TEXT=$(IFS=,; echo "${FAILED[*]}")

  COMPLETED_TEXT="$COMPLETED_TEXT" \
  SKIPPED_TEXT="$SKIPPED_TEXT" \
  FAILED_TEXT="$FAILED_TEXT" \
  CURRENT_INDEX="$CURRENT_INDEX" \
  TOTAL_RUNS="$TOTAL_RUNS" \
  BATCH_START="$BATCH_START" \
  STATUS_FILE="$STATUS_FILE" \
  python3 - <<'PY_STATUS'
import json
import os
import time
from pathlib import Path

def parse(name):
    text = os.environ.get(name, "")
    return [value for value in text.split(",") if value]

data = {
    "completed": parse("COMPLETED_TEXT"),
    "skipped_as_already_complete": parse("SKIPPED_TEXT"),
    "failed": parse("FAILED_TEXT"),
    "processed_count": int(os.environ["CURRENT_INDEX"]),
    "total_planned_runs": int(os.environ["TOTAL_RUNS"]),
    "elapsed_seconds": (
        int(time.time()) - int(os.environ["BATCH_START"])
    ),
    "execution_policy": "sequential_single_gpu",
}

Path(os.environ["STATUS_FILE"]).write_text(
    json.dumps(data, indent=2) + "\n",
    encoding="utf-8",
)
PY_STATUS
}

validate_completed_run() {
  RESULT_DIR=$1
  MODEL=$2

  RESULT_DIR="$RESULT_DIR" \
  MODEL="$MODEL" \
  python3 - <<'PY_VALIDATE'
import json
import os
from pathlib import Path

run = Path(os.environ["RESULT_DIR"])
model = os.environ["MODEL"]

summary_path = run / "summary.json"
metrics_path = run / "test_metrics.json"
protocol_path = run / "protocol_identity.json"

assert summary_path.exists()
assert metrics_path.exists()
assert protocol_path.exists()

summary = json.loads(
    summary_path.read_text(encoding="utf-8")
)
metrics = json.loads(
    metrics_path.read_text(encoding="utf-8")
)
protocol = json.loads(
    protocol_path.read_text(encoding="utf-8")
)

assert summary["status"] == "PASS"
assert summary["model"] == model
assert summary["dataset"] == "Amazon"
assert summary["test_evaluated"] is True
assert isinstance(summary["test_metrics"], dict)
assert float(summary["test_inference_seconds"]) > 0

assert protocol["architecture_changed"] is False
assert int(
    protocol["test_evaluations_during_training"]
) == 0
assert (
    protocol["test_evaluation_after_training"]
    is True
)

required_metrics = [
    "auroc",
    "auprc",
    "macro_f1",
    "fraud_f1",
    "fraud_precision",
    "fraud_recall",
    "g_mean",
    "accuracy",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
]

for metric in required_metrics:
    assert metric in metrics, (
        f"Missing final test metric: {metric}"
    )

print("Final artifact verification: PASS")
print(f"Test AUROC: {metrics['auroc']:.6f}")
print(f"Test AUPRC: {metrics['auprc']:.6f}")
print(f"Test Macro-F1: {metrics['macro_f1']:.6f}")
print(f"Test Fraud-F1: {metrics['fraud_f1']:.6f}")
print(f"Test G-Mean: {metrics['g_mean']:.6f}")
PY_VALIDATE
}

run_one() {
  MODEL=$1
  RATIO=$2
  SEED=$3
  INDEX=$4

  if [ "$MODEL" = "LEX-GNN" ]; then
    MODEL_DIR="lexgnn"
    PYTHON="$LEX_PYTHON"
    ADAPTER="$LEX_ADAPTER"
    CUDA_ENV="$ROOT/environment_paths/lexgnn_cuda.sh"
    CONFIG="$ROOT/configs/lexgnn/final_amazon/lexgnn_amazon_tr${RATIO}_seed${SEED}.json"
  else
    MODEL_DIR="consisgad"
    PYTHON="$CONSIS_PYTHON"
    ADAPTER="$CONSIS_ADAPTER"
    CUDA_ENV="$ROOT/environment_paths/consisgad_cuda.sh"
    CONFIG="$ROOT/configs/consisgad/final_amazon/consisgad_amazon_tr${RATIO}_seed${SEED}.json"
  fi

  RUN_NAME="${MODEL_DIR}_amazon_tr${RATIO}_seed${SEED}"
  LOG="$LOG_DIR/${RUN_NAME}.log"

  RESULTS_ROOT=$(
    CONFIG="$CONFIG" "$PYTHON" - <<'PY_CONFIG'
import json
import os
from pathlib import Path

config = json.loads(
    Path(os.environ["CONFIG"]).read_text(
        encoding="utf-8"
    )
)

print(config["results_root"])
PY_CONFIG
  )

  EXISTING_SUMMARY=$(
    find "$RESULTS_ROOT" \
      -type f \
      -name summary.json \
      2>/dev/null |
    sort |
    tail -n 1
  )

  echo ""
  echo "============================================================"
  printf "OVERALL FINAL-RUN PROGRESS "
  progress_bar "$INDEX" "$TOTAL_RUNS"
  echo ""
  echo "Model: $MODEL"
  echo "Dataset: Amazon"
  echo "Training ratio: TR${RATIO}"
  echo "Training seed: $SEED"
  echo "Split seed: 2"
  echo "Configuration: $CONFIG"
  echo "Log: $LOG"
  echo "============================================================"

  if [ -n "$EXISTING_SUMMARY" ]; then
    EXISTING_DIR=$(dirname "$EXISTING_SUMMARY")

    if validate_completed_run \
      "$EXISTING_DIR" \
      "$MODEL"; then
      echo "Status: SKIPPED — verified final run exists."
      SKIPPED+=("$RUN_NAME")
      return 0
    fi

    echo "ERROR: Existing run is incomplete or invalid."
    return 1
  fi

  GPU_MEMORY=$(
    nvidia-smi \
      --query-gpu=memory.used \
      --format=csv,noheader,nounits |
    head -n 1
  )

  echo "GPU memory before run: ${GPU_MEMORY} MiB"

  if [ "$GPU_MEMORY" -gt 1000 ]; then
    echo "ERROR: Unexpected GPU activity."
    nvidia-smi
    return 1
  fi

  source "$CUDA_ENV"

  RUN_START=$(date +%s)

  "$PYTHON" -u "$ADAPTER" \
    --config "$CONFIG" \
    2>&1 |
    tee "$LOG"

  RUN_STATUS=${PIPESTATUS[0]}
  RUN_END=$(date +%s)
  RUN_SECONDS=$((RUN_END - RUN_START))

  if [ "$RUN_STATUS" -ne 0 ]; then
    echo "ERROR: Model command returned $RUN_STATUS"
    return 1
  fi

  SUMMARY=$(
    find "$RESULTS_ROOT" \
      -type f \
      -name summary.json \
      2>/dev/null |
    sort |
    tail -n 1
  )

  if [ -z "$SUMMARY" ]; then
    echo "ERROR: Completed summary was not created."
    return 1
  fi

  RESULT_DIR=$(dirname "$SUMMARY")

  if ! validate_completed_run \
    "$RESULT_DIR" \
    "$MODEL"; then
    echo "ERROR: Final artifact verification failed."
    return 1
  fi

  COMPLETED+=("$RUN_NAME")

  echo "Run duration: $(duration_text "$RUN_SECONDS")"
  echo "Run status: PASS"

  return 0
}

echo "============================================================"
echo "COMP8851 FINAL AMAZON BATCH"
echo "============================================================"
echo "Execution policy: sequential single GPU"
echo "Models: LEX-GNN and ConsisGAD"
echo "Training ratios: TR40, TR30, TR20, TR10"
echo "Training seeds: 2, 42, 72"
echo "Total runs: 24"
echo "Test evaluation: once after each completed training run"
echo "============================================================"

BATCH_START=$(date +%s)
TOTAL_RUNS=24
CURRENT_INDEX=0
COMPLETED=()
SKIPPED=()
FAILED=()

for MODEL in "LEX-GNN" "ConsisGAD"
do
  for RATIO in 40 30 20 10
  do
    for SEED in 2 42 72
    do
      CURRENT_INDEX=$((CURRENT_INDEX + 1))
      RUN_NAME="$(
        echo "$MODEL" |
        tr '[:upper:]' '[:lower:]' |
        tr -d '-'
      )_amazon_tr${RATIO}_seed${SEED}"

      if ! run_one \
        "$MODEL" \
        "$RATIO" \
        "$SEED" \
        "$CURRENT_INDEX"; then
        FAILED+=("$RUN_NAME")
        write_status

        echo ""
        echo "============================================================"
        echo "FINAL AMAZON BATCH STOPPED"
        echo "============================================================"
        echo "Failed run: $RUN_NAME"
        echo "Completed: ${#COMPLETED[@]}"
        echo "Skipped: ${#SKIPPED[@]}"
        echo "Failed: ${#FAILED[@]}"
        exit 1
      fi

      write_status

      ELAPSED=$((
        $(date +%s) - BATCH_START
      ))

      echo ""
      printf "BATCH CHECKPOINT "
      progress_bar "$CURRENT_INDEX" "$TOTAL_RUNS"
      echo ""
      echo "Elapsed: $(duration_text "$ELAPSED")"
      echo "Completed in this execution: ${#COMPLETED[@]}"
      echo "Previously complete and skipped: ${#SKIPPED[@]}"
    done
  done
done

BATCH_END=$(date +%s)
BATCH_SECONDS=$((BATCH_END - BATCH_START))

write_status

echo ""
echo "============================================================"
echo "FINAL AMAZON BATCH COMPLETE"
echo "============================================================"
echo "Completed now: ${#COMPLETED[@]}"
echo "Skipped: ${#SKIPPED[@]}"
echo "Failed: ${#FAILED[@]}"
echo "Total processed: $CURRENT_INDEX/$TOTAL_RUNS"
echo "Elapsed: $(duration_text "$BATCH_SECONDS")"
echo "FINAL AMAZON RUNS COMPLETE: TRUE"
