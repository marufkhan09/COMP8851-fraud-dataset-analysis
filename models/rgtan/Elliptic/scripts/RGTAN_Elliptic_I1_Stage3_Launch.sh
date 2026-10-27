#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
[[ "$(hostname)" == 2be00658fac4 ]] || { echo 'STOP: Wrong instance for RGTAN_Elliptic';exit 1; }
source /workspace/comp8851/bootstrap/enable_rgtan_cuda11.sh
[[ "$(git -C /workspace/comp8851/repos/antifraud rev-parse HEAD)" == 65e583f03ac5c5983e49cc35179fe7a65908b9c1 ]] || exit 1
mkdir -p /workspace/comp8851/logs/rgtan
exec 9>/workspace/comp8851/logs/rgtan/RGTAN_Elliptic_Stage3_Launch.lock
flock -n 9 || { echo 'STOP: RGTAN_Elliptic Stage3 is already running';exit 1; }
PY=/workspace/comp8851/env/rgtan-legacy/bin/python
SCRIPT=/workspace/comp8851/pipeline/RGTAN_Elliptic_TR40_Tune12_v1.py
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
printf '%s\n' 'RGTAN_Elliptic_I1_Stage3: real-data one-epoch pilot begins; no test scoring.'
"$PY" -u "$SCRIPT" --mode pilot 2>&1 | tee -a /workspace/comp8851/logs/rgtan/RGTAN_Elliptic_TR40_Stage3_Pilot.log
echo 'RGTAN_Elliptic_TR40_ONE_EPOCH_TRAINING_PILOT=PASS'
if [[ "${RGTAN_Elliptic_PILOT_ONLY:-0}" == 1 ]];then
 echo 'RGTAN_Elliptic_I1_PILOT_ONLY_REQUESTED=PASS; TUNING_NOT_STARTED';exit 0
fi
printf '%s\n' 'RGTAN_Elliptic_I1_Stage3: pilot passed; launching all 12 validation-only trials.'
"$PY" -u "$SCRIPT" --mode tune 2>&1 | tee -a /workspace/comp8851/logs/rgtan/RGTAN_Elliptic_TR40_Stage3_Tune12.log
echo 'RGTAN_Elliptic_I1_TR40_TUNE12_COMPLETE=PASS TEST_SCORED=NO'
