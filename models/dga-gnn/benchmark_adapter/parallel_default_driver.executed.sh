#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/COMP8851-fraud-dataset-analysis
DGA_ROOT="$ROOT/models/dga-gnn"
CODE="$DGA_ROOT/dga_gnn_upstream/code"
PROCESSED="$DGA_ROOT/dga_gnn_upstream/data/processed"
QUEUE=/workspace/dga_parallel_queue
ADAPTER_SOURCE="$QUEUE/train_benchmark.py"
NOTES_SOURCE="$QUEUE/DGA_BENCHMARK_PROTOCOL.md"
DRIVER_SOURCE="$QUEUE/dga_parallel_default_driver.sh"
ADAPTER_DIR="$DGA_ROOT/benchmark_adapter"
ADAPTER="$ADAPTER_DIR/train_benchmark.py"
DRIVER_EVIDENCE="$DGA_ROOT/evidence/benchmark_parallel_default_seed2"
MASTER_LOG="$DRIVER_EVIDENCE/master.log"
STATUS_FILE="$DRIVER_EVIDENCE/status.txt"
GPU_TRACE="$DRIVER_EVIDENCE/shared_gpu_trace.csv"

CURRENT_STAGE=bootstrap
FINISHED=0
GPU_MONITOR_PID=""

mkdir -p "$DRIVER_EVIDENCE"
exec > >(tee -a "$MASTER_LOG") 2>&1

status() {
  printf '%s | %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" \
    | tee "$STATUS_FILE"
}

cleanup() {
  local rc=$?
  if [ -n "${GPU_MONITOR_PID:-}" ] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
  fi
  if [ "$FINISHED" -ne 1 ]; then
    printf '%s | FAILED at stage %s (exit %s)\n' \
      "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$CURRENT_STAGE" "$rc" \
      | tee "$STATUS_FILE"
  fi
}
trap cleanup EXIT

stage_tree() {
  local directory=$1
  while IFS= read -r -d '' file; do
    git add -- "${file#$ROOT/}"
  done < <(
    find "$directory" -type f \
      ! -path '*/wandb/*' \
      ! -name '*.ckpt' \
      ! -name 'master.log' \
      ! -name 'status.txt' \
      ! -name 'final_summary.txt' \
      -print0
  )
  while IFS= read -r -d '' checkpoint; do
    git add -f -- "${checkpoint#$ROOT/}"
  done < <(find "$directory" -type f -name '*.ckpt' -print0)
}

echo "=== DGA-GNN PROTOCOL-SAFE PARALLEL DEFAULT RUNS ==="
date -u
hostname

CURRENT_STAGE=activate_environment
status "RUNNING: activating dga-gnn environment"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate dga-gnn

export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=2
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export WANDB_SILENT=true
export DGA_EFFICIENCY_REPORTABLE=0

CURRENT_STAGE=preflight
status "RUNNING: repository, source, graph and capacity preflight"
cd "$ROOT"
test -d .git
test -f "$ADAPTER_SOURCE"
test -f "$NOTES_SOURCE"
test -f "$DRIVER_SOURCE"

if ! git diff --cached --quiet; then
  echo "ERROR: pre-existing staged changes detected."
  git --no-pager diff --cached --name-only
  exit 30
fi

LOCAL_HEAD=$(git rev-parse HEAD)
REMOTE_HEAD=$(git rev-parse origin/main)
echo "Local head:  $LOCAL_HEAD"
echo "Remote head: $REMOTE_HEAD"
test "$LOCAL_HEAD" = "$REMOTE_HEAD"
test "$LOCAL_HEAD" = "9e78521012a843859cebbec2ca2909c029981882"

echo 'd86e610867cde2fd16c181497fbce10c80758b8cd054147006a3fa6544902b1a  models/dga-gnn/dga_gnn_upstream/code/train.py' \
  | sha256sum -c -

for graph in \
  tfinance.dgldata \
  yelpchi.dgldata \
  amazon.dgldata \
  elliptic_of_amnet.dgldata; do
  test -s "$PROCESSED/$graph"
done

TOTAL_GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
PEAK_FILES=(
  "$DGA_ROOT/tfinance/evidence/compatibility/native_two_epoch_seed2_attempt_02_cuda_dgl/peak_gpu_memory_mib.txt"
  "$DGA_ROOT/yelpchi/evidence/compatibility/native_two_epoch_seed2/peak_gpu_memory_mib.txt"
  "$DGA_ROOT/amazon/evidence/compatibility/native_two_epoch_seed2/peak_gpu_memory_mib.txt"
  "$DGA_ROOT/elliptic/evidence/compatibility/native_two_epoch_seed2/peak_gpu_memory_mib.txt"
)
PEAK_SUM=0
for peak_file in "${PEAK_FILES[@]}"; do
  test -s "$peak_file"
  peak=$(tr -dc '0-9' < "$peak_file")
  PEAK_SUM=$((PEAK_SUM + peak))
done
REQUIRED_WITH_HEADROOM=$((PEAK_SUM * 3 / 2))
echo "Compatibility peak sum MiB: $PEAK_SUM"
echo "Required with 50% headroom MiB: $REQUIRED_WITH_HEADROOM"
echo "GPU total MiB: $TOTAL_GPU_MIB"
if [ "$REQUIRED_WITH_HEADROOM" -ge "$TOTAL_GPU_MIB" ]; then
  echo "ERROR: aggregate compatibility peaks do not leave enough GPU headroom."
  exit 31
fi

AVAILABLE_GIB=$(df --output=avail -BG /workspace | tail -n 1 | tr -dc '0-9')
echo "Available workspace GiB: $AVAILABLE_GIB"
if [ "$AVAILABLE_GIB" -lt 20 ]; then
  echo "ERROR: fewer than 20 GiB available."
  exit 32
fi

mkdir -p "$ADAPTER_DIR"
cp -p "$ADAPTER_SOURCE" "$ADAPTER"
cp -p "$NOTES_SOURCE" "$ADAPTER_DIR/PROTOCOL.md"
cp -p "$DRIVER_SOURCE" "$ADAPTER_DIR/parallel_default_driver.executed.sh"
ADAPTER="$ADAPTER" python - <<'PY'
import os
from pathlib import Path
path = Path(os.environ["ADAPTER"])
compile(path.read_text(), str(path), "exec")
print("Adapter syntax check: PASS")
PY
sha256sum "$ADAPTER" "$ADAPTER_DIR/PROTOCOL.md" \
  "$ADAPTER_DIR/parallel_default_driver.executed.sh" \
  | tee "$DRIVER_EVIDENCE/adapter_files.sha256"

ADAPTER="$ADAPTER" python - <<'PY' \
  | tee "$DRIVER_EVIDENCE/static_leakage_audit.txt"
import os
from pathlib import Path

source = Path(os.environ["ADAPTER"]).read_text()
start = source.index("    def validation_step")
end = source.index("    def configure_optimizers", start)
validation_section = source[start:end]

for forbidden in ("tst_auc", "tst_aps", "self.tst_idx", "tst_msk"):
    if forbidden in validation_section:
        raise RuntimeError("Forbidden test reference in validation section: " + forbidden)

required = (
    'dstdata["val_msk"]',
    'self.log("val_aps"',
    'monitor="val_aps"',
    'test_labels_used_for_training_or_selection',
)
for token in required:
    if token not in source:
        raise RuntimeError("Required protocol control is missing: " + token)

print("Validation section contains no test-index or test-metric reference.")
print("Validation-mask, validation-AP and final-audit controls are present.")
print("STATIC LEAKAGE AUDIT: PASS")
PY

CURRENT_STAGE=adapter_gate
status "RUNNING: two-epoch protocol-adapter gate on YelpChi"
GATE="$DRIVER_EVIDENCE/yelpchi_two_epoch_gate"
mkdir -p "$GATE/checkpoints" "$GATE/wandb" "$GATE/hydra"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export WANDB_DIR="$GATE/wandb"
export DGA_CHECKPOINT_DIR="$GATE/checkpoints"
export DGA_RESULT_JSON="$GATE/result.json"

set +e
python -u "$ADAPTER" \
  --config-name yelpchi \
  seed=2 max_epochs=2 patience=2 nowandb=True usegpu=True gpuid=0 \
  hydra.run.dir="$GATE/hydra" hydra.output_subdir=.hydra \
  2>&1 | tee "$GATE/train.log"
GATE_STATUS=${PIPESTATUS[0]}
set -e
echo "$GATE_STATUS" > "$GATE/exit_status.txt"
if [ "$GATE_STATUS" -ne 0 ]; then
  exit "$GATE_STATUS"
fi

RESULT="$GATE/result.json" LOG="$GATE/train.log" python - <<'PY' \
  | tee "$GATE/protocol_validation.txt"
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["RESULT"]).read_text())
log = Path(os.environ["LOG"]).read_text()
assert result["status"] == "PASS"
assert result["protocol_safe"] is True
assert result["test_labels_used_for_training_or_selection"] is False
assert result["checkpoint_monitor"] == "val_aps"
assert result["epochs_completed"] == 2
assert all("tst" not in str(epoch) for epoch in result["epoch_history"])
evaluation_position = log.find("Evaluating model in")
test_position = log.find("'final_tst/")
assert evaluation_position >= 0
assert test_position > evaluation_position
print("Two validation epochs contain no test metric.")
print("Test metrics appear only after best-checkpoint loading.")
print("PROTOCOL ADAPTER GATE: PASS")
PY

CURRENT_STAGE=parallel_accuracy_runs
status "RUNNING: four parallel protocol-safe default seed-2 accuracy runs"

declare -a DATASETS=(tfinance yelpchi amazon elliptic)
declare -A CONFIGS=(
  [tfinance]=tfinance
  [yelpchi]=yelpchi
  [amazon]=amazon
  [elliptic]=elliptic_of_amnet
)
declare -A PIDS

nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv -l 2 > "$GPU_TRACE" &
GPU_MONITOR_PID=$!

run_dataset() {
  local dataset=$1
  local config_name=${CONFIGS[$dataset]}
  local run_dir="$DGA_ROOT/$dataset/evidence/benchmark/protocol_safe_default_seed2_parallel_accuracy"

  mkdir -p "$run_dir/checkpoints" "$run_dir/wandb" "$run_dir/hydra"
  (
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    export OPENBLAS_NUM_THREADS=8
    export WANDB_DIR="$run_dir/wandb"
    export DGA_CHECKPOINT_DIR="$run_dir/checkpoints"
    export DGA_RESULT_JSON="$run_dir/result.json"
    export DGA_EFFICIENCY_REPORTABLE=0

    date -u > "$run_dir/start_utc.txt"
    python "$ADAPTER" \
      --config-name "$config_name" \
      --cfg job \
      seed=2 nowandb=True usegpu=True gpuid=0 \
      > "$run_dir/resolved_config.yaml"
    sha256sum "$run_dir/resolved_config.yaml" > "$run_dir/resolved_config.sha256"

    set +e
    python -u "$ADAPTER" \
      --config-name "$config_name" \
      seed=2 nowandb=True usegpu=True gpuid=0 \
      hydra.run.dir="$run_dir/hydra" hydra.output_subdir=.hydra \
      2>&1 | tee "$run_dir/train.log"
    run_status=${PIPESTATUS[0]}
    set -e
    echo "$run_status" > "$run_dir/exit_status.txt"
    date -u > "$run_dir/end_utc.txt"
    if [ "$run_status" -ne 0 ]; then
      exit "$run_status"
    fi

    RESULT="$run_dir/result.json" LOG="$run_dir/train.log" python - <<'PY' \
      > "$run_dir/protocol_validation.txt"
import json
import os
from pathlib import Path

result = json.loads(Path(os.environ["RESULT"]).read_text())
log = Path(os.environ["LOG"]).read_text()
assert result["status"] == "PASS"
assert result["protocol_safe"] is True
assert result["efficiency_reportable"] is False
assert result["checkpoint_monitor"] == "val_aps"
assert all("tst" not in str(epoch) for epoch in result["epoch_history"])
assert log.find("'final_tst/") > log.find("Evaluating model in") >= 0
print("PROTOCOL VALIDATION: PASS")
PY

    find "$run_dir" -type f ! -path '*/wandb/*' -printf '%p | %s bytes\n' \
      | sort > "$run_dir/generated_files.txt"
  )
}

for dataset in "${DATASETS[@]}"; do
  run_dataset "$dataset" &
  PIDS[$dataset]=$!
  echo "Started $dataset with driver PID ${PIDS[$dataset]}"
  sleep 10
done

FAILED=0
for dataset in "${DATASETS[@]}"; do
  if wait "${PIDS[$dataset]}"; then
    echo "$dataset: PASS"
  else
    echo "$dataset: FAIL"
    FAILED=1
  fi
done

kill "$GPU_MONITOR_PID" 2>/dev/null || true
wait "$GPU_MONITOR_PID" 2>/dev/null || true
GPU_MONITOR_PID=""

if [ "$FAILED" -ne 0 ]; then
  echo "At least one parallel dataset run failed. No automatic commit will be made."
  exit 40
fi

CURRENT_STAGE=aggregate_results
status "RUNNING: aggregating parallel accuracy results"
ROOT="$ROOT" python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
base = root / "models/dga-gnn"
datasets = ["tfinance", "yelpchi", "amazon", "elliptic"]
rows = []
for dataset in datasets:
    path = base / dataset / "evidence/benchmark/protocol_safe_default_seed2_parallel_accuracy/result.json"
    result = json.loads(path.read_text())
    metrics = result["final_metrics"]
    rows.append({
        "dataset": dataset,
        "seed": result["seed"],
        "epochs_completed": result["epochs_completed"],
        "best_epoch": result["best_epoch"],
        "best_validation_ap": result["best_validation_ap"],
        "test_ap": metrics["final_tst/aps"],
        "test_roc_auc": metrics["final_tst/auc"],
        "test_macro_f1": metrics["final_tst/mf1"],
        "test_fraud_f1": metrics["final_tst/fraud_f1"],
        "efficiency_reportable": result["efficiency_reportable"],
    })

output_dir = base / "evidence/benchmark_parallel_default_seed2"
(output_dir / "accuracy_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
with (output_dir / "accuracy_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
PY

CURRENT_STAGE=commit_results
status "RUNNING: staging, committing and pushing adapter and parallel accuracy evidence"
cd "$ROOT"
stage_tree "$ADAPTER_DIR"
stage_tree "$DRIVER_EVIDENCE"
for dataset in "${DATASETS[@]}"; do
  stage_tree "$DGA_ROOT/$dataset/evidence/benchmark/protocol_safe_default_seed2_parallel_accuracy"
done

if git --no-pager diff --cached --name-only \
  | grep -Ev '^models/dga-gnn/(benchmark_adapter/|evidence/benchmark_parallel_default_seed2/|(tfinance|yelpchi|amazon|elliptic)/evidence/benchmark/protocol_safe_default_seed2_parallel_accuracy/)' \
  | grep -q .; then
  echo "ERROR: staging contains an unexpected path."
  git --no-pager diff --cached --name-only
  exit 41
fi

git commit -m "Add DGA-GNN protocol-safe parallel default runs"
BRANCH=$(git branch --show-current)
git push origin "$BRANCH"
git fetch origin "$BRANCH"
test "$(git rev-parse HEAD)" = "$(git rev-parse "origin/$BRANCH")"

CURRENT_STAGE=complete
{
  echo "DGA-GNN protocol-safe parallel default runs: COMPLETE"
  date -u
  echo "Commit: $(git rev-parse HEAD)"
  echo "Summary: models/dga-gnn/evidence/benchmark_parallel_default_seed2/accuracy_summary.csv"
  echo "Efficiency figures from these concurrent runs are NOT reportable."
} | tee "$DRIVER_EVIDENCE/final_summary.txt"

FINISHED=1
status "COMPLETE: four protocol-safe default seed-2 accuracy runs committed and pushed"
