#!/usr/bin/env bash
set -uo pipefail

ROOT=/workspace/comp8851_two_models
EVIDENCE="$ROOT/evidence/tfinance/final_consisgad"

source "$ROOT/environment_paths/lexgnn_cuda.sh"

export WANDB_MODE=disabled
export WANDB_DISABLED=true

PYTHON="$ROOT/venvs/lexgnn/bin/python"
ADAPTER="$ROOT/adapters/tfinance_staging/train_tfinance_consisgad.py"

COMPLETED=0

for RATIO in 40 30 20 10; do
    for SEED in 2 42 72; do

        NAME="tr${RATIO}_seed${SEED}"
        CONFIG="$ROOT/configs/tfinance_final/consisgad_tfinance_${NAME}_final.json"
        LOG="$EVIDENCE/${NAME}.log"
        STATUS="$EVIDENCE/${NAME}_exit_status.txt"

        if [ -e "$LOG" ] || [ -e "$STATUS" ]; then
            echo "ERROR: Existing evidence for $NAME. Stopping."
            exit 3
        fi

        echo ""
        echo "========================================"
        echo "STARTING: $NAME"
        date -u
        echo "========================================"

        timeout --signal=TERM --kill-after=30s 45m \
            "$PYTHON" -u "$ADAPTER" \
            --config "$CONFIG" \
            > "$LOG" 2>&1

        RC=$?
        printf '%s\n' "$RC" > "$STATUS"

        echo "FINISHED: $NAME | EXIT CODE: $RC"
        tail -n 16 "$LOG"

        if [ "$RC" -ne 0 ]; then
            echo "ERROR: $NAME failed. Stopping queue."
            exit "$RC"
        fi

        # Verify completion and test evaluation before advancing.
        "$PYTHON" - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

c = json.loads(Path(sys.argv[1]).read_text())
result_root = Path(c["results_root"])
summaries = list(result_root.glob("*/summary.json"))

assert len(summaries) == 1, (
    f"Expected one result summary, found {len(summaries)}"
)

result_dir = summaries[0].parent
s = json.loads(summaries[0].read_text())

assert s["status"] == "PASS"
assert s["test_evaluated"] is True
assert s["test_metrics"] is not None
assert s["epochs_completed"] == 100

for filename in (
    "checkpoint.pt",
    "protocol_identity.json",
    "split_summary.json",
    "test_metrics.json",
    "validation_metrics.csv",
):
    assert (result_dir / filename).is_file(), filename

print("PASS: Final result and test metrics verified")
PY

        if [ "$?" -ne 0 ]; then
            echo "ERROR: Result verification failed for $NAME"
            exit 4
        fi

        COMPLETED=$((COMPLETED + 1))
        echo "VERIFIED COMPLETED RUNS: $COMPLETED / 12"

    done
done

echo ""
echo "SUCCESS: ALL 12 CONSISGAD T-FINANCE FINAL RUNS VERIFIED"
date -u
