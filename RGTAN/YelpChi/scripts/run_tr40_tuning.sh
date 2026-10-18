#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/comp8851
SCRIPT="$ROOT/scripts/rgtan_yelpchi_v1/rgtan_yelpchi_tuning_v1.py"
PY="$ROOT/envs/rgtan/bin/python"
WRAP="$ROOT/diagnostics/rgtan/rgtan_python_cuda_i3.sh"
STATUS="$ROOT/diagnostics/rgtan/yelpchi_tr40_tuning.status"

export DGLBACKEND=pytorch
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/scripts/rgtan_yelpchi_v1:$ROOT/repos/antifraud${PYTHONPATH:+:$PYTHONPATH}"

exec 9>"$ROOT/diagnostics/rgtan/yelpchi_tr40_tuning.lock"

if ! flock -n 9; then
    echo "TUNING_ALREADY_LOCKED"
    exit 2
fi

trap 'rc=$?; echo "TUNING_EXIT_CODE=$rc"; printf "%s\n" "$rc" > "$STATUS"' EXIT

echo "TUNING_STARTED_UTC=$(date -u +%FT%TZ)"
echo "TUNING_MODE=12_TRIALS_TR40"
echo "MAX_EPOCHS=100 PATIENCE=20 BATCH=32"
echo "TEST_EVALUATION=DISABLED"

bash "$WRAP" "$PY" -u "$SCRIPT" tune-all

echo "I3_RGTAN_YELPCHI_TUNING_FINISHED"
