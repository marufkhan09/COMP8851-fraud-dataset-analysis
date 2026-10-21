#!/usr/bin/env bash
set -uo pipefail

ROOT=/workspace/comp8851_two_models

LEX_PYTHON="$ROOT/venvs/lexgnn/bin/python"
CONSIS_PYTHON="$ROOT/venvs/consisgad/bin/python"

LEX_ADAPTER="$ROOT/adapters/lexgnn/train_fdcompcn_unified.py"
CONSIS_ADAPTER="$ROOT/adapters/consisgad/train_fdcompcn_unified.py"

LEX_CONFIG_DIR="$ROOT/configs/lexgnn/final_fdcompcn"
CONSIS_CONFIG_DIR="$ROOT/configs/consisgad/final_fdcompcn"

LOG_DIR="$ROOT/logs/final_fdcompcn"
STATUS_DIR="$ROOT/evidence/final_runs/fdcompcn"
STATUS_FILE="$STATUS_DIR/fdcompcn_final_batch_status.tsv"

mkdir -p "$LOG_DIR" "$STATUS_DIR"

printf \
  "index\tmodel\tratio\tseed\tstatus\texit_code\tsummary\tlog\n" \
  > "$STATUS_FILE"

START_TIME=$(date +%s)

TOTAL=24
CURRENT=0
COMPLETED=0
SKIPPED=0
FAILED=0

find_valid_summary() {
  local result_root=$1
  local expected_model=$2
  local expected_seed=$3

  python3 - \
    "$result_root" \
    "$expected_model" \
    "$expected_seed" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_model = sys.argv[2]
expected_seed = int(sys.argv[3])

if not root.exists():
    raise SystemExit(0)

for path in sorted(root.rglob("summary.json"), reverse=True):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue

    if (
        data.get("status") == "PASS"
        and data.get("model") == expected_model
        and data.get("dataset") == "FDCompCN"
        and int(data.get("seed", -1)) == expected_seed
        and data.get("test_evaluated") is True
        and isinstance(data.get("test_metrics"), dict)
    ):
        print(path)
        break
PY
}

format_elapsed() {
  local seconds=$1
  printf "%02dh:%02dm:%02ds" \
    $((seconds / 3600)) \
    $(((seconds % 3600) / 60)) \
    $((seconds % 60))
}

run_one() {
  local model_key=$1
  local ratio=$2
  local seed=$3

  local model
  local python_bin
  local adapter
  local config
  local run_log
  local env_file

  CURRENT=$((CURRENT + 1))

  if [ "$model_key" = "lexgnn" ]; then
    model="LEX-GNN"
    python_bin="$LEX_PYTHON"
    adapter="$LEX_ADAPTER"
    config="$LEX_CONFIG_DIR/lexgnn_fdcompcn_tr${ratio}_seed${seed}.json"
    run_log="$LOG_DIR/lexgnn_fdcompcn_tr${ratio}_seed${seed}.log"
    env_file="$ROOT/environment_paths/lexgnn_cuda.sh"
  else
    model="ConsisGAD"
    python_bin="$CONSIS_PYTHON"
    adapter="$CONSIS_ADAPTER"
    config="$CONSIS_CONFIG_DIR/consisgad_fdcompcn_tr${ratio}_seed${seed}.json"
    run_log="$LOG_DIR/consisgad_fdcompcn_tr${ratio}_seed${seed}.log"

    if [ -f "$ROOT/environment_paths/consisgad_cuda.sh" ]; then
      env_file="$ROOT/environment_paths/consisgad_cuda.sh"
    else
      env_file="$ROOT/environment_paths/lexgnn_cuda.sh"
    fi
  fi

  local result_root
  result_root=$(
    python3 - "$config" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(
    Path(sys.argv[1]).read_text(encoding="utf-8")
)
print(config["results_root"])
PY
  )

  local filled=$((CURRENT * 30 / TOTAL))
  local empty=$((30 - filled))
  local percent=$((CURRENT * 100 / TOTAL))
  local bar

  bar=$(
    printf "%${filled}s" "" |
    tr " " "#"
    printf "%${empty}s" "" |
    tr " " "-"
  )

  echo ""
  echo "============================================================"
  printf "OVERALL PROGRESS [%s] %d/%d (%3d%%)\n" \
    "$bar" "$CURRENT" "$TOTAL" "$percent"
  echo "Model: $model"
  echo "Dataset: FDCompCN"
  echo "Training ratio: TR${ratio}"
  echo "Training seed: $seed"
  echo "Split seed: 2"
  echo "Configuration: $config"
  echo "Log: $run_log"
  echo "============================================================"

  local existing_summary
  existing_summary=$(
    find_valid_summary \
      "$result_root" \
      "$model" \
      "$seed"
  )

  if [ -n "$existing_summary" ]; then
    echo "Status: SKIPPED — verified completed result exists."
    echo "Summary: $existing_summary"

    printf \
      "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$CURRENT" \
      "$model" \
      "$ratio" \
      "$seed" \
      "SKIPPED_VERIFIED" \
      "0" \
      "$existing_summary" \
      "$run_log" \
      >> "$STATUS_FILE"

    SKIPPED=$((SKIPPED + 1))
    return 0
  fi

  echo "GPU memory before run: $(
    nvidia-smi \
      --query-gpu=memory.used \
      --format=csv,noheader,nounits |
    head -n 1
  ) MiB"

  local run_start
  local run_end
  local run_exit
  local summary

  run_start=$(date +%s)

  set +e

  (
    source "$env_file"

    export CUDA_VISIBLE_DEVICES=0
    export PYTHONHASHSEED="$seed"
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    export OPENBLAS_NUM_THREADS=8

    "$python_bin" \
      "$adapter" \
      --config "$config"
  ) 2>&1 |
  tee "$run_log"

  run_exit=${PIPESTATUS[0]}

  set -e

  run_end=$(date +%s)

  echo "Run elapsed seconds: $((run_end - run_start))"
  echo "Process exit code: $run_exit"

  summary=$(
    find_valid_summary \
      "$result_root" \
      "$model" \
      "$seed"
  )

  if [ "$run_exit" -eq 0 ] && [ -n "$summary" ]; then
    echo "Status: COMPLETED"
    echo "Verified summary: $summary"

    printf \
      "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$CURRENT" \
      "$model" \
      "$ratio" \
      "$seed" \
      "COMPLETED" \
      "$run_exit" \
      "$summary" \
      "$run_log" \
      >> "$STATUS_FILE"

    COMPLETED=$((COMPLETED + 1))
  else
    echo "Status: FAILED"

    printf \
      "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$CURRENT" \
      "$model" \
      "$ratio" \
      "$seed" \
      "FAILED" \
      "$run_exit" \
      "${summary:--}" \
      "$run_log" \
      >> "$STATUS_FILE"

    FAILED=$((FAILED + 1))
  fi
}

set -e

for ratio in 40 30 20 10
do
  for seed in 2 42 72
  do
    run_one lexgnn "$ratio" "$seed"
    run_one consisgad "$ratio" "$seed"
  done
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
PROCESSED=$((COMPLETED + SKIPPED + FAILED))

echo ""
echo "============================================================"
echo "FINAL FDCOMPCN BATCH STATUS"
echo "============================================================"
echo "Completed now: $COMPLETED"
echo "Skipped as already verified: $SKIPPED"
echo "Failed: $FAILED"
echo "Total processed: $PROCESSED/$TOTAL"
echo "Elapsed: $(format_elapsed "$ELAPSED")"

if [ "$PROCESSED" -eq "$TOTAL" ] &&
   [ "$FAILED" -eq 0 ]; then
  echo "FINAL FDCOMPCN RUNS COMPLETE: TRUE"
  exit 0
else
  echo "FINAL FDCOMPCN RUNS COMPLETE: FALSE"
  exit 1
fi
