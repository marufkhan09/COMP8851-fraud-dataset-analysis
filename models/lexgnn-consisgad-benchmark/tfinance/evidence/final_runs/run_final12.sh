#!/usr/bin/env bash
set -uo pipefail

BASE=/workspace/comp8851_two_models
EVIDENCE="$BASE/evidence/tfinance/final_lexgnn"
PYTHON="$BASE/venvs/lexgnn/bin/python"
ADAPTER="$BASE/adapters/tfinance_staging/train_tfinance_lexgnn_auprc_fixed.py"

source "$BASE/environment_paths/lexgnn_cuda.sh"

echo "===== LEX-GNN T-FINANCE FINAL BATCH START ====="
date -u

success=0
failed=0

for ratio in 10 20 30 40; do
    for seed in 2 42 72; do
        name="tr${ratio}_seed${seed}"
        config="$BASE/configs/tfinance_final/lexgnn_tfinance_${name}_final.json"
        log="$EVIDENCE/${name}.log"
        status="$EVIDENCE/${name}_exit_status.txt"

        echo ""
        echo "===== START $name ====="
        date -u

        if [ -f "$status" ]; then
            previous=$(cat "$status")
            if [ "$previous" = "0" ]; then
                echo "SKIP: Previously successful $name"
                success=$((success + 1))
                continue
            fi
            echo "SKIP: Existing non-success status for $name ($previous); manual review required"
            failed=$((failed + 1))
            continue
        fi

        if [ ! -f "$config" ]; then
            echo "MISSING CONFIG: $config"
            printf '%s\n' "98" > "$status"
            failed=$((failed + 1))
            continue
        fi

        timeout --signal=TERM --kill-after=30s 240m \
            "$PYTHON" -u "$ADAPTER" \
            --config "$config" \
            > "$log" 2>&1

        rc=$?
        printf '%s\n' "$rc" > "$status"

        if [ "$rc" -eq 0 ]; then
            echo "SUCCESS: $name"
            success=$((success + 1))
        else
            echo "FAILED: $name (exit code $rc)"
            failed=$((failed + 1))
            tail -n 15 "$log"
        fi

        date -u
    done
done

echo ""
echo "===== FINAL BATCH SUMMARY ====="
echo "SUCCESS: $success"
echo "FAILED OR SKIPPED FOR REVIEW: $failed"
date -u

if [ "$failed" -gt 0 ]; then
    exit 1
fi
