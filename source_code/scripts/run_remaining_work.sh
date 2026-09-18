#!/usr/bin/env bash
# Everything this benchmark still owes, in priority order.
#
#   bash scripts/run_remaining_work.sh [phase]
#
#   phase 1   GHRN re-run with corrected between-stage selection   ~1 session
#   phase 2   T-Social, the two empty cells                        1-2 sessions
#   phase 3   Extra seeds on the two contested cells               ~1 session
#   phase 4   TR30 / TR20 / TR10 label-scarcity ladder             90 runs
#   all       every phase in order (long)
#
# Run on a GPU host with the datasets already fetched. Nothing here runs on CPU.
# Each phase writes into its own results root so a phase can be inspected, kept
# or discarded without disturbing the published TR40 results in results/.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=/venv/main/bin/python
[ -x "$PY" ] || PY=python3

PHASE="${1:-all}"
SEEDS="2 42 72"
DATASETS_DONE="yelpchi amazon tfinance elliptic fdcompcn"

banner() {
  echo
  echo "=============================================================="
  echo " $1"
  echo "=============================================================="
  date -u +"started %Y-%m-%dT%H:%M:%SZ"
  echo
}

# --------------------------------------------------------------------------
# Phase 1 - the correction, and the cheapest thing on this list
# --------------------------------------------------------------------------
# run_one.py now defaults to --stage-selection validation, which ships whichever
# of stage 1 and stage 2 scored higher on validation AUPRC. The published runs
# shipped stage 2 unconditionally, including on the six runs where stage 1
# validated better. Re-running under the corrected rule is expected to change
# T-Finance most, and possibly the headline comparison.
#
# Configurations are already frozen from the TR40 tuning, so --skip-tuning keeps
# this to 15 final runs and nothing else.
phase1() {
  banner "PHASE 1  GHRN, corrected between-stage selection (15 runs)"
  "$PY" scripts/run_benchmark.py \
    --models GHRN \
    --datasets $DATASETS_DONE \
    --ratios TR40 \
    --seeds $SEEDS \
    --skip-tuning \
    --epochs 100 --patience 20 \
    --max-minutes 90 \
    --results-root results_stagefix \
    --no-archive --no-report
  echo
  echo "Compare against the published run before adopting:"
  echo "  results/            stage 2 shipped unconditionally"
  echo "  results_stagefix/   stage selected on validation AUPRC"
  echo "Adopt only if the corrected runs are complete and clean."
}

# --------------------------------------------------------------------------
# Phase 2 - the two empty cells
# --------------------------------------------------------------------------
# GHRN first. It is the cheaper model per epoch by a wide margin and needs about
# 34 GB of GPU memory on T-Social, so it fits an A6000 (48 GB) but not a T4
# (16 GB). Running it first secures at least one T-Social column.
#
# CARE-GNN second, and only after GHRN's results are safe. It materialises a
# Python adjacency set per node per relation; on 5.78M nodes and 146.2M edges
# that is what exhausted the previous attempt's time cap. --allow-large
# overrides the 2M-node guard and needs a high-RAM host.
phase2() {
  banner "PHASE 2a  GHRN x T-Social  (needs ~34 GB GPU; use the A6000)"
  "$PY" scripts/run_benchmark.py \
    --models GHRN \
    --datasets tsocial \
    --ratios TR40 \
    --seeds $SEEDS \
    --trials 2 \
    --tune-epochs 12 --tune-patience 6 \
    --epochs 100 --patience 20 \
    --max-minutes 180 \
    --results-root results_tsocial \
    --no-archive --no-report

  banner "PHASE 2b  CARE-GNN x T-Social  (node guard overridden; expect slow)"
  echo "If this is killed, the failure is recorded and that is a valid"
  echo "FAILED_TECHNICAL outcome. Do not delete the record."
  echo
  "$PY" scripts/run_benchmark.py \
    --models CARE-GNN \
    --datasets tsocial \
    --ratios TR40 \
    --seeds $SEEDS \
    --trials 2 \
    --tune-epochs 10 --tune-patience 5 \
    --epochs 100 --patience 20 \
    --max-minutes 240 \
    --allow-large \
    --results-root results_tsocial \
    --no-archive --no-report
}

# --------------------------------------------------------------------------
# Phase 3 - settle the two cells three seeds could not call
# --------------------------------------------------------------------------
# Amazon: CARE-GNN's mean is dragged by one collapsed seed, so its spread says
# more about collapse frequency than about the model's ceiling. More seeds
# estimate that frequency properly.
# T-Finance: GHRN's SD exceeds the gap between the two means, so the ranking
# currently flips with the seed draw.
phase3() {
  banner "PHASE 3  Extra seeds on Amazon and T-Finance (24 runs)"
  "$PY" scripts/run_benchmark.py \
    --models CARE-GNN GHRN \
    --datasets amazon tfinance \
    --ratios TR40 \
    --seeds 7 13 99 \
    --skip-tuning \
    --epochs 100 --patience 20 \
    --max-minutes 90 \
    --results-root results_moreseeds \
    --no-archive --no-report
  echo
  echo "Pool with the published seeds only if the protocol is identical."
  echo "Report the collapse rate over all six seeds, not just the mean."
}

# --------------------------------------------------------------------------
# Phase 4 - the label-scarcity ladder (RQ2), currently unaddressed
# --------------------------------------------------------------------------
# Hyperparameters stay frozen from TR40 so the measurement reflects label
# scarcity and not re-tuning effort. Training pools are nested by construction
# (TR10 subset of TR20 subset of TR30 subset of TR40) and validation and test
# node IDs are identical at every ratio, so every point is scored on the same
# nodes. 2 models x 5 datasets x 3 ratios x 3 seeds = 90 runs.
phase4() {
  banner "PHASE 4  TR30 / TR20 / TR10 label ladder (90 runs)"
  for ratio in TR30 TR20 TR10; do
    echo "--- $ratio ---"
    "$PY" scripts/run_benchmark.py \
      --models CARE-GNN GHRN \
      --datasets $DATASETS_DONE \
      --ratios "$ratio" \
      --seeds $SEEDS \
      --skip-tuning \
      --epochs 100 --patience 20 \
      --max-minutes 90 \
      --results-root results_ladder \
      --no-archive --no-report
  done
  echo
  echo "Do NOT re-tune at lower ratios. Reusing the TR40 configuration is what"
  echo "makes the degradation attributable to label scarcity."
}

case "$PHASE" in
  1) phase1 ;;
  2) phase2 ;;
  3) phase3 ;;
  4) phase4 ;;
  all) phase1; phase2; phase3; phase4 ;;
  *) echo "usage: $0 [1|2|3|4|all]" >&2; exit 2 ;;
esac

banner "DONE - phase ${PHASE}"
echo "Rebuild the deliverable tables against the new results:"
echo "  cd ../scripts"
echo "  python consolidate_results.py && python dataset_profile.py"
echo "  python stage_ablation.py && python consistency_audit.py"
echo "  python build_report.py"
