#!/usr/bin/env bash
# CARE-GNN x T-Finance  --  fast budget
#
#   bash scripts/train_tfinance_caregnn.sh
#
# 39,357 nodes / 42.4M edges. The smallest of the two remaining datasets and
# the one most likely to finish quickly.
#
# Budget: 3 tuning trials at 15 epochs, then 3 final seeds at 100 epochs with
# patience 20. The tuning search is deliberately small to save time; the final
# runs keep the full protocol budget so the reported numbers are comparable
# with the eight cells already completed.
#
# Every one of these settings is written into each run's run_config.yml, so the
# deliverable reports the budget that actually produced the numbers.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=/venv/main/bin/python
[ -x "$PY" ] || PY=python3

echo "=============================================================="
echo " CARE-GNN x T-Finance"
echo "=============================================================="

"$PY" scripts/run_benchmark.py \
  --models CARE-GNN \
  --datasets tfinance \
  --ratios TR40 \
  --seeds 2 42 72 \
  --trials 3 \
  --tune-epochs 15 --tune-patience 8 \
  --epochs 100 --patience 20 \
  --max-minutes 45 \
  --results-root results \
  --no-archive --no-report

echo
echo "Done. Check the result with:"
echo "  $PY scripts/build_deliverable.py --results results --out /tmp/check --platform 'Vast.ai'"
