#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
[[ "$(hostname)" == '2be00658fac4' ]] || { echo 'STOP: Wrong machine; this is RGTAN_Elliptic I1 only';exit 1; }
source /workspace/comp8851/bootstrap/enable_rgtan_cuda11.sh
PY=/workspace/comp8851/env/rgtan-legacy/bin/python
ROOT=/workspace/comp8851
PIPE=$ROOT/pipeline
LOGDIR=$ROOT/logs/rgtan
mkdir -p "$LOGDIR"
exec 9>"$LOGDIR/RGTAN_Elliptic_I1_Stage4_Exclusive.lock"
flock -n 9 || { echo 'STOP: RGTAN_Elliptic_I1_Stage4 already running';exit 1; }
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
F=$PIPE/RGTAN_Elliptic_I1_Stage4_Finals_v1.py
C=$PIPE/RGTAN_Elliptic_I1_Stage4_Crossfit_v1.py
"$PY" -m py_compile "$F" "$C"
echo 'RGTAN_Elliptic_I1_STAGE4: Frozen winner audit, 3 missing crossfits, 12 final trainings, 12 one-time held-out evaluations.'
"$PY" -u "$F" --mode freeze 2>&1 | tee -a "$LOGDIR/RGTAN_Elliptic_I1_Stage4_Freeze.log"
for ratio in TR30 TR20 TR10;do
 echo "RGTAN_Elliptic_I1_STAGE4_CROSSFIT_START=$ratio"
 "$PY" -u "$C" --ratio "$ratio" 2>&1 | tee -a "$LOGDIR/RGTAN_Elliptic_I1_Stage4_Crossfit_${ratio}.log"
done
echo 'RGTAN_Elliptic_I1_STAGE4: Starting 12 frozen-configuration final trainings; NO TEST ACCESS.'
"$PY" -u "$F" --mode train 2>&1 | tee -a "$LOGDIR/RGTAN_Elliptic_I1_Stage4_FinalTrainings.log"
if [[ "${RGTAN_ELLIPTIC_I1_TRAIN_ONLY:-0}" == 1 ]];then
 echo 'RGTAN_Elliptic_I1_STAGE4_TRAIN_ONLY=PASS; TEST_NOT_SCORED';exit 0
fi
echo 'RGTAN_Elliptic_I1_STAGE4: All 12 frozen; each held-out final is scored once only.'
"$PY" -u "$F" --mode test 2>&1 | tee -a "$LOGDIR/RGTAN_Elliptic_I1_Stage4_FinalTests.log"
echo 'RGTAN_Elliptic_I1_STAGE4_ALL_FINALS_COMPLETE=PASS'
