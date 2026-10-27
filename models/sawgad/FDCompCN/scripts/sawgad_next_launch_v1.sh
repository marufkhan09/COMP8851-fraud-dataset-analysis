#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
PY=/workspace/comp8851/envs/sawgad/bin/python
SCRIPT=/workspace/sawgad_fdcompcn_full_v1.py
DIAG=/workspace/comp8851/diagnostics/sawgad
LOGDIR=/workspace/comp8851/logs/sawgad
mkdir -p "$LOGDIR" "$DIAG"
test -x "$PY"
test -f "$SCRIPT"
sha256sum -c sawgad_fdcompcn_adapter_v1.py.sha256
sha256sum -c sawgad_fdcompcn_full_v1.py.sha256
sha256sum -c sawgad_elliptic_preflight_v1.py.sha256
MODE="${1:-status}"
case "$MODE" in
  preflight)
    echo '========== FDCOMPCN ADAPTER =========='
    "$PY" -u /workspace/sawgad_fdcompcn_adapter_v1.py
    echo '========== ELLIPTIC TIME / LABEL ALIGNMENT =========='
    "$PY" -u /workspace/sawgad_elliptic_preflight_v1.py
    echo 'SAWGAD_TWO_DATASET_PREFLIGHT=PASS'
    ;;
  capacity)
    test -f "$DIAG/fdcompcn_adapter_v1.json" || { echo 'STOP: run preflight first'; exit 1; }
    "$PY" -u "$SCRIPT" --phase capacity 2>&1 | tee "$LOGDIR/fdcompcn_full_batch_capacity_v1.log"
    ;;
  tune-one|tune-all|finals)
    if [[ "$MODE" != finals ]]; then
      test -f "$DIAG/fdcompcn_full_batch_capacity_v1.json" || { echo 'STOP: run capacity first'; exit 1; }
    fi
    phase=tune
    args=()
    case "$MODE" in
      tune-one) args=(--max-trials 1) ;;
      tune-all) args=(--max-trials 12) ;;
      finals) phase=finals ;;
    esac
    log="$LOGDIR/fdcompcn_${MODE}_$(date -u +%Y%m%d_%H%M%S).log"
    nohup "$PY" -u "$SCRIPT" --phase "$phase" "${args[@]}" >"$log" 2>&1 </dev/null &
    pid=$!
    echo "RUN_MODE=$MODE"
    echo "PROCESS_PID=$pid"
    echo "LOG=$log"
    echo "MONITOR=tail -n 30 $log"
    ;;
  status)
    "$PY" "$SCRIPT" --phase status
    pgrep -af '[s]awgad_fdcompcn_full_v1.py' || true
    ;;
  *) echo 'Use preflight|capacity|tune-one|tune-all|finals|status' >&2; exit 2 ;;
esac
