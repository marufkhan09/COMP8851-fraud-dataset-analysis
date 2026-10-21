#!/usr/bin/env bash
set -uo pipefail

ROOT=/workspace/comp8851_two_models

LEX_PYTHON="$ROOT/venvs/lexgnn/bin/python"
CONSIS_PYTHON="$ROOT/venvs/consisgad/bin/python"

LEX_ADAPTER="$ROOT/adapters/lexgnn/train_yelpchi_unified.py"
CONSIS_ADAPTER="$ROOT/adapters/consisgad/train_yelpchi_unified.py"

LEX_CONFIG_DIR="$ROOT/configs/lexgnn/final_yelpchi"
CONSIS_CONFIG_DIR="$ROOT/configs/consisgad/final_yelpchi"

LOG_DIR="$ROOT/logs/final_yelpchi"
STATUS_DIR="$ROOT/evidence/final_runs/yelpchi"
STATUS_TSV="$STATUS_DIR/yelpchi_final_batch_status.tsv"
STATUS_JSON="$STATUS_DIR/yelpchi_final_batch_status.json"

mkdir -p "$LOG_DIR" "$STATUS_DIR"

printf "index\tmodel\tratio\tseed\tstatus\texit_code\tsummary\tlog\n" \
  > "$STATUS_TSV"

TASKS=(
  "lexgnn:40:2"
  "consisgad:40:2"
  "lexgnn:40:42"
  "consisgad:40:42"
  "lexgnn:40:72"
  "consisgad:40:72"

  "lexgnn:30:2"
  "consisgad:30:2"
  "lexgnn:30:42"
  "consisgad:30:42"
  "lexgnn:30:72"
  "consisgad:30:72"

  "lexgnn:20:2"
  "consisgad:20:2"
  "lexgnn:20:42"
  "consisgad:20:42"
  "lexgnn:20:72"
  "consisgad:20:72"

  "lexgnn:10:2"
  "consisgad:10:2"
  "lexgnn:10:42"
  "consisgad:10:42"
  "lexgnn:10:72"
  "consisgad:10:72"
)

TOTAL=${#TASKS[@]}
COMPLETED=0
SKIPPED=0
FAILED=0
BATCH_START=$(date +%s)

echo "============================================================"
echo "FINAL YELPCHI BATCH START"
echo "============================================================"
echo "UTC start: $(date -u)"
echo "Runs scheduled: $TOTAL"
echo "Execution mode: sequential, one GPU"
echo "Test policy: once after training"
echo "Estimated duration: approximately 6–8 hours"

for ARRAY_INDEX in "${!TASKS[@]}"
do
  TASK="${TASKS[$ARRAY_INDEX]}"
  RUN_INDEX=$((ARRAY_INDEX + 1))

  IFS=: read -r MODEL RATIO SEED <<< "$TASK"

  FILLED=$((RUN_INDEX * 30 / TOTAL))
  EMPTY=$((30 - FILLED))
  PERCENT=$((RUN_INDEX * 100 / TOTAL))

  BAR_FILLED=$(
    printf "%${FILLED}s" "" |
    tr " " "#"
  )

  BAR_EMPTY=$(
    printf "%${EMPTY}s" "" |
    tr " " "-"
  )

  if [ "$MODEL" = "lexgnn" ]; then
    DISPLAY_MODEL="LEX-GNN"
    PYTHON="$LEX_PYTHON"
    ADAPTER="$LEX_ADAPTER"
    CUDA_ENV="$ROOT/environment_paths/lexgnn_cuda.sh"
    CONFIG="$LEX_CONFIG_DIR/lexgnn_yelpchi_tr${RATIO}_seed${SEED}.json"
  else
    DISPLAY_MODEL="ConsisGAD"
    PYTHON="$CONSIS_PYTHON"
    ADAPTER="$CONSIS_ADAPTER"
    CUDA_ENV="$ROOT/environment_paths/consisgad_cuda.sh"
    CONFIG="$CONSIS_CONFIG_DIR/consisgad_yelpchi_tr${RATIO}_seed${SEED}.json"
  fi

  RUN_ROOT="$ROOT/runs/$MODEL/yelpchi/final/tr${RATIO}_seed${SEED}"
  LOG="$LOG_DIR/${MODEL}_yelpchi_tr${RATIO}_seed${SEED}.log"

  echo ""
  echo "============================================================"
  printf "OVERALL PROGRESS [%s%s] %d/%d (%3d%%)\n" \
    "$BAR_FILLED" \
    "$BAR_EMPTY" \
    "$RUN_INDEX" \
    "$TOTAL" \
    "$PERCENT"
  echo "Model: $DISPLAY_MODEL"
  echo "Dataset: YelpChi"
  echo "Training ratio: TR$RATIO"
  echo "Training seed: $SEED"
  echo "Split seed: 2"
  echo "Configuration: $CONFIG"
  echo "Log: $LOG"
  echo "============================================================"

  if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Configuration is missing."

    printf "%s\t%s\t%s\t%s\tFAILED\t2\t\t%s\n" \
      "$RUN_INDEX" \
      "$DISPLAY_MODEL" \
      "$RATIO" \
      "$SEED" \
      "$LOG" \
      >> "$STATUS_TSV"

    FAILED=$((FAILED + 1))
    continue
  fi

  EXISTING_PASS=$(
    RUN_ROOT="$RUN_ROOT" \
    EXPECTED_MODEL="$DISPLAY_MODEL" \
    "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
expected_model = os.environ["EXPECTED_MODEL"]

valid = []

for summary_path in root.glob("*/summary.json"):
    try:
        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
    except Exception:
        continue

    checks = [
        summary.get("status") == "PASS",
        summary.get("model") == expected_model,
        summary.get("dataset") == "YelpChi",
        summary.get("test_evaluated") is True,
        isinstance(summary.get("test_metrics"), dict),
    ]

    if all(checks):
        valid.append(str(summary_path))

print(sorted(valid)[-1] if valid else "")
PY
  )

  if [ -n "$EXISTING_PASS" ]; then
    echo "Status: SKIPPED — verified completed result exists."
    echo "Summary: $EXISTING_PASS"

    printf "%s\t%s\t%s\t%s\tSKIPPED\t0\t%s\t%s\n" \
      "$RUN_INDEX" \
      "$DISPLAY_MODEL" \
      "$RATIO" \
      "$SEED" \
      "$EXISTING_PASS" \
      "$LOG" \
      >> "$STATUS_TSV"

    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Wait briefly for GPU memory from the previous process to clear.
  GPU_READY=0

  for ATTEMPT in $(seq 1 30)
  do
    GPU_MEMORY=$(
      nvidia-smi \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits |
      head -n 1
    )

    if [ "$GPU_MEMORY" -le 1000 ]; then
      GPU_READY=1
      break
    fi

    echo "Waiting for GPU memory to clear: ${GPU_MEMORY} MiB"
    sleep 1
  done

  if [ "$GPU_READY" -ne 1 ]; then
    echo "ERROR: GPU did not become available."

    printf "%s\t%s\t%s\t%s\tFAILED\t3\t\t%s\n" \
      "$RUN_INDEX" \
      "$DISPLAY_MODEL" \
      "$RATIO" \
      "$SEED" \
      "$LOG" \
      >> "$STATUS_TSV"

    FAILED=$((FAILED + 1))
    continue
  fi

  echo "GPU memory before run: ${GPU_MEMORY} MiB"
  RUN_START=$(date +%s)

  (
    source "$CUDA_ENV"

    export DGLBACKEND=pytorch
    export CUDA_VISIBLE_DEVICES=0
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    export OPENBLAS_NUM_THREADS=8
    export PYTHONHASHSEED="$SEED"

    "$PYTHON" "$ADAPTER" \
      --config "$CONFIG"
  ) 2>&1 |
  tee "$LOG"

  RUN_STATUS=${PIPESTATUS[0]}
  RUN_END=$(date +%s)
  RUN_ELAPSED=$((RUN_END - RUN_START))

  echo "Run elapsed seconds: $RUN_ELAPSED"

  if [ "$RUN_STATUS" -ne 0 ]; then
    echo "Status: FAILED with exit code $RUN_STATUS"

    printf "%s\t%s\t%s\t%s\tFAILED\t%s\t\t%s\n" \
      "$RUN_INDEX" \
      "$DISPLAY_MODEL" \
      "$RATIO" \
      "$SEED" \
      "$RUN_STATUS" \
      "$LOG" \
      >> "$STATUS_TSV"

    FAILED=$((FAILED + 1))
    continue
  fi

  VERIFIED_SUMMARY=$(
    RUN_ROOT="$RUN_ROOT" \
    EXPECTED_MODEL="$DISPLAY_MODEL" \
    "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
expected_model = os.environ["EXPECTED_MODEL"]

valid = []

for summary_path in root.glob("*/summary.json"):
    try:
        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
    except Exception:
        continue

    metrics = summary.get("test_metrics")

    checks = [
        summary.get("status") == "PASS",
        summary.get("model") == expected_model,
        summary.get("dataset") == "YelpChi",
        summary.get("test_evaluated") is True,
        isinstance(metrics, dict),
        isinstance(metrics, dict)
        and "auroc" in metrics
        and "auprc" in metrics
        and "macro_f1" in metrics
        and "fraud_f1" in metrics
        and "g_mean" in metrics,
    ]

    if all(checks):
        valid.append(str(summary_path))

print(sorted(valid)[-1] if valid else "")
PY
  )

  if [ -z "$VERIFIED_SUMMARY" ]; then
    echo "Status: FAILED — valid final summary not found."

    printf "%s\t%s\t%s\t%s\tFAILED\t4\t\t%s\n" \
      "$RUN_INDEX" \
      "$DISPLAY_MODEL" \
      "$RATIO" \
      "$SEED" \
      "$LOG" \
      >> "$STATUS_TSV"

    FAILED=$((FAILED + 1))
    continue
  fi

  echo "Status: COMPLETED"
  echo "Verified summary: $VERIFIED_SUMMARY"

  printf "%s\t%s\t%s\t%s\tCOMPLETED\t0\t%s\t%s\n" \
    "$RUN_INDEX" \
    "$DISPLAY_MODEL" \
    "$RATIO" \
    "$SEED" \
    "$VERIFIED_SUMMARY" \
    "$LOG" \
    >> "$STATUS_TSV"

  COMPLETED=$((COMPLETED + 1))
done

BATCH_END=$(date +%s)
BATCH_ELAPSED=$((BATCH_END - BATCH_START))

STATUS_TSV="$STATUS_TSV" \
STATUS_JSON="$STATUS_JSON" \
BATCH_ELAPSED="$BATCH_ELAPSED" \
python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

tsv = Path(os.environ["STATUS_TSV"])
target = Path(os.environ["STATUS_JSON"])

with tsv.open(
    newline="",
    encoding="utf-8",
) as handle:
    rows = list(
        csv.DictReader(handle, delimiter="\t")
    )

record = {
    "dataset": "YelpChi",
    "phase": "unified final benchmark",
    "scheduled_runs": 24,
    "processed_runs": len(rows),
    "completed_runs": sum(
        row["status"] == "COMPLETED"
        for row in rows
    ),
    "skipped_verified_runs": sum(
        row["status"] == "SKIPPED"
        for row in rows
    ),
    "failed_runs": sum(
        row["status"] == "FAILED"
        for row in rows
    ),
    "elapsed_seconds": int(
        os.environ["BATCH_ELAPSED"]
    ),
    "runs": rows,
}

target.write_text(
    json.dumps(record, indent=2) + "\n",
    encoding="utf-8",
)

print(json.dumps(record, indent=2))
PY

HOURS=$((BATCH_ELAPSED / 3600))
MINUTES=$(((BATCH_ELAPSED % 3600) / 60))
SECONDS=$((BATCH_ELAPSED % 60))

echo ""
echo "============================================================"
echo "FINAL YELPCHI BATCH STATUS"
echo "============================================================"
echo "Completed now: $COMPLETED"
echo "Skipped as already verified: $SKIPPED"
echo "Failed: $FAILED"
echo "Total processed: $((COMPLETED + SKIPPED + FAILED))/$TOTAL"
printf "Elapsed: %02dh:%02dm:%02ds\n" \
  "$HOURS" \
  "$MINUTES" \
  "$SECONDS"

if [ "$FAILED" -ne 0 ]; then
  echo "FINAL YELPCHI RUNS COMPLETE: FALSE"
  exit 1
fi

if [ $((COMPLETED + SKIPPED)) -ne "$TOTAL" ]; then
  echo "FINAL YELPCHI RUNS COMPLETE: FALSE"
  exit 1
fi

echo "FINAL YELPCHI RUNS COMPLETE: TRUE"
