#!/usr/bin/env bash
# Unattended AMNet queue for the COMP8851 unified benchmark.
#
#  1. waits until every GDN process on this instance has exited
#  2. for each dataset, smallest first:
#       build the AMNet view (if not built yet)
#       2-epoch smoke test (TR40, seed 2)
#       only if the smoke passes: all 12 runs (TR40/30/20/10 x seeds 2/42/72), one at a time
#     a failure at any step is logged, that dataset is skipped, and the queue moves on
#  3. writes a one-line status per dataset to queue_status.txt
#
# Start it inside tmux:   bash /workspace/amnet_vast/queue_amnet.sh
# Watch it:               tail -f /workspace/amnet_vast/logs/queue_log.txt
set -u

VAST=/workspace/amnet_vast
AMNET_PY=$VAST/envs/amnet/bin/python
DGL_PY=/venv/gdn-adapt/bin/python          # python of an env with DGL (the gdn-adapt env); set before starting
DATASETS="comp elliptic amazon yelp tsocial tfinance"

mkdir -p $VAST/logs
LOG=$VAST/logs/queue_log.txt
STATUS=$VAST/queue_status.txt
source $VAST/env.sh

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if [ ! -x "$DGL_PY" ]; then
  log "DGL_PY is not set to a valid python ($DGL_PY). Edit queue_amnet.sh first. Stopping."
  exit 1
fi

echo "--- queue run started $(date '+%F %T') ---" >> "$STATUS"
log "=== AMNet queue started; waiting for GDN processes to exit ==="
while pgrep -f "run_gdn_unified.py|batch_run_gdn.py" > /dev/null; do
  sleep 120
done
log "No GDN processes running. GPU state:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$LOG"

for ds in $DATASETS; do
  view=$VAST/views/${ds}_view.npz
  if [ ! -f "$view" ]; then
    case $ds in
      comp|tfinance|tsocial) PY=$DGL_PY ;;
      *) PY=$AMNET_PY ;;
    esac
    log "$ds: building view"
    if ! $PY $VAST/build_amnet_views.py --dataset $ds > $VAST/logs/build_${ds}.log 2>&1; then
      log "$ds: BUILD FAILED, skipping (see logs/build_${ds}.log)"
      echo "$ds BUILD_FAILED" >> "$STATUS"
      continue
    fi
    log "$ds: view built"
  fi

  log "$ds: smoke test (TR40, seed 2, 2 epochs)"
  if ! $AMNET_PY $VAST/run_amnet_unified.py --dataset $ds --ratio 40 --seed 2 --track smoke --max_epoch 2 \
       > $VAST/logs/smoke_${ds}.log 2>&1; then
    log "$ds: SMOKE FAILED, skipping (see logs/smoke_${ds}.log)"
    echo "$ds SMOKE_FAILED" >> "$STATUS"
    continue
  fi
  log "$ds: smoke PASSED, starting 12 runs"

  if $AMNET_PY $VAST/batch_run_amnet.py --dataset $ds > $VAST/logs/batch_${ds}.out 2>&1; then
    log "$ds: all 12 runs DONE"
    echo "$ds DONE" >> "$STATUS"
  else
    log "$ds: batch finished with failures (see logs/batch_log_${ds}.txt)"
    echo "$ds PARTIAL" >> "$STATUS"
  fi
done

log "=== AMNet queue finished ==="
cat "$STATUS" | tee -a "$LOG"
