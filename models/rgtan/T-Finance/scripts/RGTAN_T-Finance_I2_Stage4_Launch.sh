#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
[[ "$(hostname)" == f24d1bb2672d ]] || { echo 'STOP: Wrong instance (RGTAN_T-Finance I2 expected)';exit 1; }
source /workspace/comp8851/bootstrap/enable_rgtan_cuda11.sh
[[ "$(git -C /workspace/comp8851/repos/antifraud rev-parse HEAD)" == 65e583f03ac5c5983e49cc35179fe7a65908b9c1 ]] || exit 1
ROOT=/workspace/comp8851
LOGDIR=$ROOT/logs/rgtan
mkdir -p "$LOGDIR"
exec 9>"$LOGDIR/RGTAN_T-Finance_I2_Stage4_Launch.lock"
flock -n 9 || { echo 'STOP: RGTAN T-Finance Stage4 is already running';exit 1; }
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
PY=$ROOT/env/rgtan-legacy/bin/python
SCRIPT=$ROOT/pipeline/RGTAN_T-Finance_TR40_Tune12_Candidate_v1.py
printf '%s\n' 'RGTAN_T-Finance I2 Stage4: train-only OOF real-risk one-epoch training+validation pilot; TEST_SCORING=NO'
"$PY" -u "$SCRIPT" --mode pilot 2>&1 | tee -a "$LOGDIR/RGTAN_T-Finance_I2_Stage4_CandidatePilot.log"
echo 'RGTAN_T-Finance_I2_TR40_ONE_EPOCH_REALRISK_PILOT=PASS'
if [[ "${RGTAN_TFinance_PILOT_ONLY:-0}" == 1 ]];then
  echo 'RGTAN_T-Finance_I2_PILOT_ONLY=PASS; TUNING_NOT_STARTED'; exit 0
fi
printf '%s\n' 'RGTAN_T-Finance I2 Stage4: pilot passed; 12 VAL-ONLY CANDIDATE trials begin. 10x10 neighbor fanout deviation, supervisor approval REQUIRED for benchmark.'
"$PY" -u "$SCRIPT" --mode tune 2>&1 | tee -a "$LOGDIR/RGTAN_T-Finance_I2_Stage4_CandidateTune12.log"
echo 'RGTAN_T-Finance_I2_TR40_CANDIDATE_TUNE12_COMPLETE=PASS; TEST_SCORING=NO; BENCHMARK_APPROVAL_REQUIRED=YES'
