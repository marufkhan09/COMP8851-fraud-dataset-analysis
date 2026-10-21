#!/usr/bin/env bash
set -euo pipefail
source /workspace/activate_comp8851_elliptic.sh

BATCH=/workspace/comp8851_elliptic_work/elliptic_reconstructed_v1
OUT="$BATCH/final_evaluation"

export PYTHONPATH="/workspace/comp8851_elliptic_work/adapters"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

for model in lexgnn consisgad; do
  for ratio in 40 30 20 10; do
    for seed in 2 42 72; do
      name="${model}_tr${ratio}_seed${seed}"

      if [[ -f "$OUT/${name}_test.json" ]]; then
        echo "SKIP EXISTING: $name"
        continue
      fi

      echo "===== EVALUATING $name ====="

      python -u "$OUT/evaluate_saved.py" \
        --model "$model" \
        --ratio "$ratio" \
        --seed "$seed" \
        > "$OUT/${name}.log" 2>&1 || {
          echo "FAILED: $name"
          tail -n 45 "$OUT/${name}.log"
          exit 1
        }

      echo "PASS: $name"
    done
  done
done

echo "PASS: ALL 24 ELLIPTIC TEST EVALUATIONS COMPLETED"
