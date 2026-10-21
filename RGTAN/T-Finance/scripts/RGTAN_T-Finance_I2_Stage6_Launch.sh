#!/usr/bin/env bash
# RGTAN x T-Finance I2: complete frozen [10,10] method via held-out test ONCE.
# Author full-neighbor configuration was not used. Never report as author-identical.
set -Eeuo pipefail
cd /workspace
[[ "$(hostname)" == 'f24d1bb2672d' ]] || { echo 'STOP: WRONG INSTANCE (I2 ONLY)'; exit 1; }
ROOT=/workspace/comp8851
source "$ROOT/bootstrap/enable_rgtan_cuda11.sh"
[[ "$(git -C "$ROOT/repos/antifraud" rev-parse HEAD)" == '65e583f03ac5c5983e49cc35179fe7a65908b9c1' ]] || { echo 'STOP: WRONG PINNED SOURCE'; exit 1; }
[[ -x "$ROOT/env/rgtan-legacy/bin/python" ]] || { echo 'STOP: Legacy Python missing'; exit 1; }
command -v flock >/dev/null || { echo 'STOP: flock not installed'; exit 1; }
mkdir -p "$ROOT/logs/rgtan"
exec 9>"$ROOT/logs/rgtan/RGTAN_T-Finance_I2_Stage6_Once.lock"
flock -n 9 || { echo 'STOP: Test-only Stage6 already running'; exit 1; }
trap 'echo "RGTAN_T-Finance_I2_STAGE6_ERROR at line $LINENO; inspect master log"' ERR
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/repos/antifraud:${PYTHONPATH:-}"
[[ -f "$ROOT/results/rgtan/tfinance/RGTAN_T-Finance_I2_12CandidateFinals_Fanout10x10_v1/RGTAN_T-Finance_I2_AllRatios_CandidateValidationSummary.json" ]] || { echo 'STOP: Stage5 must finish first'; exit 1; }
echo 'RGTAN_T-Finance I2 Stage6: ONE held-out test per frozen model, ZERO retraining/tuning; sampler=[10,10] methodological deviation transparently retained; TEST_SCORING=YES'
"$ROOT/env/rgtan-legacy/bin/python" -u "$ROOT/pipeline/RGTAN_T-Finance_I2_Stage6_TestOnce_v1.py"
echo 'RGTAN_T-Finance_I2_STAGE6_FINISHED=PASS; ADAPTED_BASELINE=YES; AUTHOR_IDENTICAL=NO'
