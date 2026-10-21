#!/usr/bin/env bash
set -euo pipefail

source /workspace/activate_comp8851_elliptic.sh

WORK=/workspace/comp8851_elliptic_work
BATCH="$WORK/elliptic_reconstructed_v1"

export PYTHONPATH="$WORK/adapters"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

echo "===== WAITING FOR ALL 12 LEX-GNN RUNS ====="

while true; do
    count=$(find "$BATCH/logs" -maxdepth 1 \
        -name 'lexgnn_*.PASS' -type f | wc -l)

    echo "$(date -u) | LEX-GNN verified: $count/12"

    if [ "$count" -eq 12 ]; then
        for ratio in 40 30 20 10; do
            for seed in 2 42 72; do
                test -s "$BATCH/logs/lexgnn_tr${ratio}_seed${seed}.PASS"
            done
        done
        break
    fi

    if ! tmux has-session -t elliptic_lex_batch 2>/dev/null; then
        echo "ERROR: LEX-GNN batch stopped before all 12 runs passed"
        exit 1
    fi

    sleep 30
done

echo "===== ALL LEX-GNN RUNS VERIFIED ====="
echo "===== STARTING CONSISGAD QUEUE ====="

for ratio in 40 30 20 10; do
    for seed in 2 42 72; do

        id="consisgad_tr${ratio}_seed${seed}"
        config="$BATCH/configs/${id}.json"
        log="$BATCH/logs/${id}.log"
        marker="$BATCH/logs/${id}.PASS"

        if [ -f "$marker" ]; then
            echo "SKIP COMPLETED: $id"
            continue
        fi

        echo "$(date -u) | START: $id"

        if ! python -u \
            "$WORK/adapters/consisgad/train_elliptic_compat.py" \
            --config "$config" > "$log" 2>&1; then

            echo "ERROR: Training failed for $id"
            tail -n 50 "$log"
            exit 1
        fi

        BATCH="$BATCH" ID="$id" RATIO="$ratio" SEED="$seed" \
        python - <<'PY'
import json
import os
from pathlib import Path

batch = Path(os.environ["BATCH"])
run_id = os.environ["ID"]
ratio = os.environ["RATIO"]
seed = int(os.environ["SEED"])

expected_mode = (
    f"reconstructed_v1_tr{ratio}_seed{seed}_validation_only"
)

matches = []
for path in (batch / "results/consisgad").glob("*/summary.json"):
    record = json.loads(path.read_text())
    if (
        record.get("model") == "ConsisGAD"
        and record.get("dataset") == "Elliptic"
        and record.get("run_mode") == expected_mode
        and record.get("seed") == seed
        and record.get("status") == "PASS"
        and record.get("epochs_completed") == 100
        and record.get("test_evaluated") is False
    ):
        matches.append(path)

if len(matches) != 1:
    raise RuntimeError(
        f"Expected one verified result for {run_id}; "
        f"found {len(matches)}"
    )

marker = batch / "logs" / f"{run_id}.PASS"
marker.write_text(str(matches[0]) + "\n")

print(f"VERIFIED: {run_id}", flush=True)
PY

    done
done

echo "PASS: ALL 12 CONSISGAD RUNS VERIFIED"
