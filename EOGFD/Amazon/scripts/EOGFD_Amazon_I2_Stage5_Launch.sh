#!/usr/bin/env bash
set -euo pipefail
source /workspace/comp8851/bootstrap/enable_rgtan_cuda11.sh
BASE=/workspace/comp8851
PREFIX=EOGFD_Amazon_I2
LOG="$BASE/logs/eogfd/${PREFIX}_Stage4_Master.log"
PY="$BASE/env/rgtan-legacy/bin/python"
SCRIPT="$BASE/eogfd/pipeline/${PREFIX}_Stage5_TestOnce.py"
test "$(hostname)" = "f24d1bb2672d"
grep -Fq "${PREFIX}_STAGE4_12_FROZEN_FINAL_TRAININGS_COMPLETE=PASS TEST_SCORED=NO" "$LOG"
# Hold the Stage4 lock through Stage5 to block competing re-runs and modifications.
exec 9>"$BASE/logs/eogfd/${PREFIX}_Stage4.lock"
flock -n 9 || { echo "${PREFIX}_STAGE5_STOP_STAGE4_PROCESS_STILL_RUNNING"; exit 3; }
exec flock -n "$BASE/logs/eogfd/${PREFIX}_Stage5.lock" "$PY" -u "$SCRIPT"
