#!/usr/bin/env bash
# GHRN x T-Social  --  fast budget
#
#   bash scripts/train_tsocial_ghrn.sh
#
# 5,781,065 nodes / 146.2M edges: the largest cell in the benchmark. GHRN
# handles it because its beta-wavelet backbone works on sparse tensors rather
# than materialising per-node adjacency structures.
#
# The cost here is NOT the training. Epochs run at roughly 0.94s; what is slow
# is loading and building the graph once per run invocation. Fewer invocations
# therefore matter far more than fewer epochs, which is why the tuning budget
# is cut to 2 trials rather than the epoch count being reduced.
#
# Needs ~34 GB of GPU memory. Do not run this concurrently with another large
# job on the same GPU.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=/venv/main/bin/python
[ -x "$PY" ] || PY=python3

echo "=============================================================="
echo " GHRN x T-Social   (largest cell; expect a long first load)"
echo "=============================================================="
df -h . | tail -1
echo

"$PY" scripts/run_benchmark.py \
  --models GHRN \
  --datasets tsocial \
  --ratios TR40 \
  --seeds 2 42 72 \
  --trials 2 \
  --tune-epochs 12 --tune-patience 6 \
  --epochs 100 --patience 20 \
  --max-minutes 180 \
  --results-root results \
  --no-archive --no-report

echo
echo "Done."
