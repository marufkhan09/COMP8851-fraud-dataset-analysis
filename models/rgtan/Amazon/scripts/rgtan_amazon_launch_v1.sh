#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/comp8851
RUNNER=/workspace/rgtan_amazon_tr40_tuning_v1.py
PY="$ROOT/diagnostics/rgtan/rgtan_python_cuda_i2.sh"
OUT="$ROOT/results/rgtan/amazon_tr40_tuning_v1"
LOGDIR="$ROOT/logs/rgtan"
LOCK="$ROOT/diagnostics/rgtan/amazon_tr40_tuning_v1.lock"
MODE="${1:-}"
case "$MODE" in
  tune-one|tune-all|status) ;;
  *) echo 'Usage: bash rgtan_amazon_launch_v1.sh {tune-one|tune-all|status}'; exit 2 ;;
esac
if [ "$MODE" != status ]; then
  cd /workspace
  sha256sum -c rgtan_amazon_tr40_tuning_v1.py.sha256
  "$ROOT/envs/rgtan/bin/python" -m py_compile "$RUNNER"
fi
test -x "$PY" || { echo "STOP: Missing CUDA launcher $PY"; exit 1; }
test -f "$RUNNER" || { echo "STOP: Missing runner $RUNNER"; exit 1; }
mkdir -p "$LOGDIR" "$OUT" "$(dirname "$LOCK")"
export PYTHONPATH="$ROOT/repos/antifraud${PYTHONPATH:+:$PYTHONPATH}"
export DGLBACKEND=pytorch PYTHONUNBUFFERED=1
if [ "$MODE" = status ]; then
  "$PY" -u "$RUNNER" status
  echo '========== RUNNING PROCESS =========='
  pgrep -af '[r]gtan_amazon_tr40_tuning_v1.py' || true
  echo '========== LATEST LOG =========='
  newest="$(ls -1t "$LOGDIR"/amazon_tr40_tune-all_*.log "$LOGDIR"/amazon_tr40_tune-one_*.log 2>/dev/null | head -1 || true)"
  if [ -n "$newest" ]; then tail -n 20 "$newest"; fi
  exit 0
fi
if ! flock -n "$LOCK" true; then
  echo 'STOP: RGTAN Amazon tuning is already running'
  exit 1
fi
LOG="$LOGDIR/amazon_tr40_${MODE}_$(date -u +%Y%m%d_%H%M%S).log"
nohup flock -n "$LOCK" "$PY" -u "$RUNNER" "$MODE" >"$LOG" 2>&1 < /dev/null &
PID=$!
sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
  echo 'STOP: Launch failed; latest log follows:'
  tail -n 60 "$LOG"
  exit 1
fi
echo "RUN_MODE=$MODE"
echo "PROCESS_PID=$PID"
echo "LOG=$LOG"
echo "MONITOR=tail -n 30 $LOG"
echo 'Leave Vast I2 running. Training is in the background.'
