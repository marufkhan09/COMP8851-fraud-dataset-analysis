#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/comp8851
RUNNER=/workspace/rgtan_amazon_finals_v2.py
BUILDER=/workspace/rgtan_amazon_crossfit_lower_v2.py
WRAPPER="$ROOT/diagnostics/rgtan/rgtan_python_cuda_i2.sh"
LOCK="$ROOT/diagnostics/rgtan/amazon_finals_v2.lock"
LOGDIR="$ROOT/logs/rgtan"
MODE="${1:-}"
case "$MODE" in
  preflight|build-crossfit|pilot|finals|status) ;;
  *) echo 'Usage: bash rgtan_amazon_finals_launch_v2.sh {preflight|build-crossfit|pilot|finals|status}' >&2; exit 2;;
esac
cd /workspace
sha256sum -c rgtan_amazon_finals_v2.py.sha256 rgtan_amazon_crossfit_lower_v2.py.sha256
"$ROOT/envs/rgtan/bin/python" -m py_compile "$RUNNER" "$BUILDER"
test -x "$WRAPPER" || { echo "STOP: CUDA Python wrapper missing: $WRAPPER" >&2; exit 1; }
test -f /workspace/rgtan_amazon_tr40_tuning_v1.py || { echo 'STOP: pinned tuning runner missing' >&2; exit 1; }
sha256sum -c rgtan_amazon_tr40_tuning_v1.py.sha256
test -f "$ROOT/results/rgtan/amazon_tr40_tuning_v1/winner.json" || { echo 'STOP: missing frozen winner' >&2; exit 1; }
mkdir -p "$LOGDIR" "$(dirname "$LOCK")" "$ROOT/results/rgtan/amazon_finals_v2"
export PYTHONPATH="$ROOT/repos/antifraud${PYTHONPATH:+:$PYTHONPATH}"
export DGLBACKEND=pytorch PYTHONUNBUFFERED=1
if [ "$MODE" = status ]; then
  "$WRAPPER" -u "$RUNNER" status
  pgrep -af '[r]gtan_amazon_finals_v2.py' || true
  exit 0
fi
# Shared lock across builder and both training modes.
if ! flock -n "$LOCK" true; then
  echo 'STOP: RGTAN finals package is already running' >&2
  exit 1
fi
if [ "$MODE" = preflight ]; then
  flock -n "$LOCK" "$WRAPPER" -u "$RUNNER" preflight
  exit 0
fi
if [ "$MODE" = build-crossfit ]; then
  ( flock -x 9
    "$WRAPPER" -u "$RUNNER" preflight
    for ratio in TR30 TR20 TR10; do
      output="$ROOT/assets/rgtan/amazon/amazon_${ratio,,}_crossfit.npz"
      manifest="$ROOT/assets/rgtan/amazon/amazon_${ratio,,}_crossfit_manifest.json"
      if [ -e "$output" ] || [ -e "$manifest" ]; then
        if [ -f "$output" ] && [ -f "$manifest" ]; then
          echo "CROSSFIT_EXISTING_${ratio}=VERIFYING_NO_OVERWRITE"
          "$WRAPPER" -u "$RUNNER" preflight
          continue
        fi
        echo "STOP: $ratio artifact/manifest pair is incomplete; manual audit required" >&2
        exit 1
      fi
      "$WRAPPER" -u "$BUILDER" --ratio "$ratio"
    done
    "$WRAPPER" -u "$RUNNER" preflight
  ) 9>"$LOCK"
  echo 'RGTAN_THREE_LOWER_CROSSFIT_ARTIFACTS=BUILT'
  exit 0
fi
LOG="$LOGDIR/amazon_finals_v2_${MODE}_$(date -u +%Y%m%d_%H%M%S).log"
nohup flock -n "$LOCK" "$WRAPPER" -u "$RUNNER" "$MODE" >"$LOG" 2>&1 </dev/null &
PID=$!
sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
  echo 'STOP: launch exited early' >&2
  tail -n 65 "$LOG"
  exit 1
fi
printf 'RUN_MODE=%s\nPROCESS_PID=%s\nLOG=%s\nMONITOR=tail -n 30 %s\n' "$MODE" "$PID" "$LOG" "$LOG"
echo 'I2 remains allocated; verify the log before claiming training progress.'
