#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
ROOT=/workspace/comp8851
PY="$ROOT/envs/sawgad/bin/python"
ADAPTER=/workspace/sawgad_tfinance_canonical_adapter_v2.py
RUNNER=/workspace/sawgad_tfinance_canonical_full_v2.py
LOGDIR="$ROOT/logs/sawgad"
DIAGDIR="$ROOT/diagnostics/sawgad"
mkdir -p "$LOGDIR" "$DIAGDIR"
test -x "$PY"
test -f "$ADAPTER" && test -f "$RUNNER"
sha256sum -c sawgad_tfinance_canonical_adapter_v2.py.sha256
sha256sum -c sawgad_tfinance_canonical_full_v2.py.sha256
MODE="${1:-status}"
case "$MODE" in
  preflight)
    "$PY" -u "$ADAPTER" 2>&1 | tee "$LOGDIR/tfinance_canonical_preflight_v2.log"
    ;;
  capacity)
    test -s "$DIAGDIR/tfinance_canonical_adapter_v2.json" || { echo 'STOP: Run preflight first'; exit 1; }
    "$PY" -u "$RUNNER" --phase capacity 2>&1 | tee "$LOGDIR/tfinance_canonical_capacity_v2.log"
    ;;
  tune-one|tune-all|finals)
    test -s "$DIAGDIR/tfinance_canonical_capacity_v2.json" || { echo 'STOP: Capacity test not passed'; exit 1; }
    phase=tune
    extra=()
    if [[ "$MODE" == tune-one ]]; then extra=(--max-trials 1); fi
    if [[ "$MODE" == finals ]]; then phase=finals; fi
    log="$LOGDIR/tfinance_canonical_${MODE}_$(date -u +%Y%m%d_%H%M%S).log"
    nohup "$PY" -u "$RUNNER" --phase "$phase" "${extra[@]}" >"$log" 2>&1 </dev/null &
    pid=$!
    echo "RUN_MODE=$MODE"
    echo "PROCESS_PID=$pid"
    echo "LOG=$log"
    echo "MONITOR=tail -n 30 $log"
    echo 'Keep I1 running; verify log before claiming trial progress.'
    ;;
  status)
    "$PY" -u "$RUNNER" --phase status
    pgrep -af '[p]ython.*sawgad_tfinance_canonical_full_v2.py' || true
    ;;
  *)
    echo 'Usage: bash sawgad_tfinance_canonical_launch_v2.sh {preflight|capacity|tune-one|tune-all|finals|status}' >&2
    exit 2
    ;;
esac
