#!/usr/bin/env bash
# CARE-GNN x T-Social  --  the hard one
#
#   bash scripts/train_tsocial_caregnn.sh
#
# This pair is refused by default. CARE-GNN materialises a Python adjacency
# set per node per relation, and T-Social has 5.78M nodes and 146.2M edges, so
# the guard at 2M nodes exists to stop the run exhausting memory rather than
# failing cleanly.
#
# --allow-large overrides that guard. It is worth attempting here, and only
# here, because this host reports ~755 GB of system RAM, which is what the
# guard's error message means by "a machine with enough RAM".
#
# Expect it to be slow and memory-hungry. Two honest outcomes:
#
#   it completes  -> a real result for the 12th cell, recorded with
#                    allow_large: true in every run_config.yml
#   it is killed  -> recorded as FAILED_TECHNICAL with the trace, which is a
#                    stronger deliverable entry than an unattempted cell
#
# Run this LAST, after the other three have finished and their results are
# safely archived. Do not run it alongside another large job.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=/venv/main/bin/python
[ -x "$PY" ] || PY=python3

echo "=============================================================="
echo " CARE-GNN x T-Social   (node guard overridden)"
echo "=============================================================="
free -g | head -2
echo
echo "Starting in 10s. Ctrl-C to abort."
sleep 10

"$PY" scripts/run_benchmark.py \
  --models CARE-GNN \
  --datasets tsocial \
  --ratios TR40 \
  --seeds 2 42 72 \
  --trials 2 \
  --tune-epochs 10 --tune-patience 5 \
  --epochs 100 --patience 20 \
  --max-minutes 240 \
  --allow-large \
  --results-root results \
  --no-archive --no-report

echo
echo "Done. If this was killed, the failure is recorded - that is a valid"
echo "FAILED_TECHNICAL outcome, not a gap. Do not delete the record."
