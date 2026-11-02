#!/usr/bin/env bash
# GHRN x T-Finance  --  fast budget
#
#   bash scripts/train_tfinance_ghrn.sh
#
# 39,357 nodes / 42.4M edges, and GHRN trains in two stages (an unrefined
# baseline, then a retrain on the pruned graph), so it does roughly twice the
# epochs of a single-stage model for the same --epochs value.
#
# Observed on this hardware: GHRN epochs on the far larger T-Social ran at
# about 0.94s, so T-Finance should be comfortably quick.
#
# Budget: 3 tuning trials at 15 epochs, then 3 final seeds at the full
# protocol budget of 100 epochs / patience 20.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=/venv/main/bin/python
[ -x "$PY" ] || PY=python3

echo "=============================================================="
echo " GHRN x T-Finance"
echo "=============================================================="

"$PY" scripts/run_benchmark.py \
  --models GHRN \
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
echo "Done."
