#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
ROOT=/workspace/comp8851
PY="$ROOT/envs/sawgad/bin/python"
RUNNER=/workspace/sawgad_elliptic_full_v1.py
ADAPTER=/workspace/sawgad_elliptic_adapter_v1.py
LOGDIR="$ROOT/logs/sawgad"
mkdir -p "$LOGDIR"
test -x "$PY"
test -f "$RUNNER"
test -f "$ADAPTER"
sha256sum -c sawgad_elliptic_adapter_v1.py.sha256
sha256sum -c sawgad_elliptic_full_v1.py.sha256
MODE="${1:-status}"
case "$MODE" in
  preflight)
    "$PY" -u "$ADAPTER" 2>&1 | tee -a "$LOGDIR/elliptic_temporal_preflight_v1.log"
    ;;
  capacity)
    "$PY" -u "$RUNNER" --phase capacity 2>&1 | tee -a "$LOGDIR/elliptic_full_capacity_v1.log"
    ;;
  tune-one|tune-all|finals)
    if [[ "$MODE" == tune* ]]; then
      test -f "$ROOT/diagnostics/sawgad/elliptic_full_batch_capacity_v1.json" || {
        echo 'STOP: Run the capacity check before tuning'; exit 1;
      }
    fi
    PHASE=tune; EXTRA=()
    if [[ "$MODE" == tune-one ]]; then EXTRA=(--max-trials 1); fi
    if [[ "$MODE" == finals ]]; then PHASE=finals; fi
    STAMP="$(date -u +%Y%m%d_%H%M%S)"
    LOG="$LOGDIR/elliptic_${MODE}_${STAMP}.log"
    nohup "$PY" -u "$RUNNER" --phase "$PHASE" "${EXTRA[@]}" >"$LOG" 2>&1 </dev/null &
    PID=$!
    echo "RUN_MODE=$MODE"
    echo "PROCESS_PID=$PID"
    echo "LOG=$LOG"
    echo "MONITOR=tail -n 30 $LOG"
    echo 'Keep Vast I1 running while training runs.'
    ;;
  status)
    "$PY" -u "$RUNNER" --phase status
    pgrep -af '[p]ython.*sawgad_elliptic_full_v1.py' || true
    ;;
  *) echo "Usage: bash $0 {preflight|capacity|tune-one|tune-all|finals|status}"; exit 2;;
esac
