#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
[[ "$(hostname)" == f24d1bb2672d ]] || { echo 'STOP: Wrong instance for RGTAN_T-Finance';exit 1; }
source /workspace/comp8851/bootstrap/enable_rgtan_cuda11.sh
[[ "$(git -C /workspace/comp8851/repos/antifraud rev-parse HEAD)" == 65e583f03ac5c5983e49cc35179fe7a65908b9c1 ]] || exit 1
mkdir -p /workspace/comp8851/logs/rgtan
exec 9>/workspace/comp8851/logs/rgtan/RGTAN_T-Finance_Stage3_Launch.lock
flock -n 9 || { echo 'STOP: RGTAN_T-Finance Stage3 is already running';exit 1; }
PY=/workspace/comp8851/env/rgtan-legacy/bin/python
SCRIPT=/workspace/comp8851/pipeline/RGTAN_T-Finance_TR40_Candidate_Pilot_Crossfit_v1.py
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
printf '%s\n' 'RGTAN_T-Finance_I2_Stage3: capped sampler GPU pilot, then exact-risk crossfit. NO tuning.'
"$PY" -u "$SCRIPT" 2>&1 | tee -a /workspace/comp8851/logs/rgtan/RGTAN_T-Finance_TR40_Stage3_CandidatePilot_ExactCrossfit.log
echo 'RGTAN_T-Finance_I2_TR40_PILOT_CROSSFIT=PASS TUNING_STARTED=NO'
