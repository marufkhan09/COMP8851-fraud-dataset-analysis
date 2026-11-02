#!/usr/bin/env bash
# ==============================================================================
# COMP8851 Phase 2: Fully Automated Production Benchmark Pipeline
# ==============================================================================
# Orchestrates the end-to-end execution of SAGAD (WWW 2026), SpaceGNN (ICLR 2025),
# and CRoC (ECAI 2025) across all 6 canonical fraud datasets under protocol vast-v4.4.
#
# Resumable by design: re-run the same command after any interruption and
# already-completed cells are automatically skipped while interrupted runs
# resume from their latest epoch checkpoint (checkpoints/last.pt).
#
# Example usage (inside tmux session `tmux new -s comp8851`):
#   CONCURRENCY=2 bash scripts/run_all.sh
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# Configure Python interpreter and parallel GPU execution flags
PY="${PYTHON:-python3}"
CONC="${CONCURRENCY:-1}"
GPUS="${GPUS:-}"
GPU_ARGS=(--require-cuda --concurrency "$CONC")
[ -n "$GPUS" ] && GPU_ARGS+=(--gpus "$GPUS")

# Force PyTorch backend for DGL and expose project root on PYTHONPATH
export DGLBACKEND=pytorch
export PYTHONPATH=".:${PYTHONPATH:-}"
mkdir -p results

# Helper function to print timestamped stage banners
stage() { echo; echo "=================== $1 ($(date -u +%FT%TZ)) ==================="; }

# Stage 1: Verify CUDA, PyTorch, DGL, and canonical datasets
stage "1/8 setup + environment verification"
[ -f results/.setup_done ] || { bash scripts/vastai_setup.sh && touch results/.setup_done; }

# Stage 2: Execute the 52-test automated pytest verification suite
stage "2/8 automated tests (real data, leak/isolation/exactness checks)"
"$PY" -m pytest tests -q -p no:cacheprovider -W ignore::DeprecationWarning | tee results/pytest_vastai.log

# Stage 3: Execute 2-epoch GPU smoke tests across all 18 model-dataset pairs
stage "3/8 GPU smoke test of all 18 pairs (2 epochs each)"
"$PY" scripts/run_matrix.py smoke "${GPU_ARGS[@]}"
"$PY" scripts/aggregate_results.py --phase smoke
"$PY" scripts/verify_results.py --phase smoke

# Stage 4: Execute full-protocol pilot runs on T-Social and FDCompCN (Seed 2)
stage "4/8 production pilot (seed 2, protocol budget) - T-Social & FDCompCN"
"$PY" scripts/run_matrix.py pilot --datasets tsocial fdcompcn "${GPU_ARGS[@]}"
"$PY" scripts/aggregate_results.py --phase pilot

# Stage 5: Wave 1 hyperparameter tuning (180 trials) + final multi-seed runs (45 runs)
stage "5/8 wave 1: tuning + final runs for the 15 non-T-Social pairs"
# Tuning: <= 12 trials per pair on TR40 seed 2, selected on validation AUPRC (--skip-test enforced)
"$PY" scripts/run_matrix.py tune  --datasets yelpchi amazon tfinance elliptic fdcompcn "${GPU_ARGS[@]}"
# Final evaluation: best tuned configuration evaluated across seeds 2, 42, 72 at TR40
"$PY" scripts/run_matrix.py final --datasets yelpchi amazon tfinance elliptic fdcompcn "${GPU_ARGS[@]}"
"$PY" scripts/aggregate_results.py --phase final

# Stage 6: Wave 2 hyperparameter tuning + final multi-seed runs for T-Social (5.78M nodes, 73.1M edges)
stage "6/8 wave 2: tuning + final runs for T-Social"
"$PY" scripts/run_matrix.py tune  --datasets tsocial "${GPU_ARGS[@]}"
"$PY" scripts/run_matrix.py final --datasets tsocial "${GPU_ARGS[@]}"

# Stage 7: Aggregate all 54 final runs and execute independent metric & checkpoint verification
stage "7/8 aggregation and independent integrity verification"
"$PY" scripts/aggregate_results.py --phase final
"$PY" scripts/verify_results.py --phase final --deep | tee results/integrity_final.log

# Stage 8: Package full and light deliverable archives with SHA-256 manifests
stage "8/8 package results"
bash scripts/package_results.sh
"$PY" scripts/run_matrix.py status
echo "done. Deliverable archives written to results/*.tar.gz."
