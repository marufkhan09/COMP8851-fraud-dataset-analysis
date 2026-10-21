#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
[[ "$(hostname)" == '2be00658fac4' ]] || { echo 'WRONG I1 INSTANCE';exit 1; }
source /workspace/comp8851/bootstrap/enable_rgtan_cuda11.sh
cd /workspace/comp8851/repos/antifraud
[[ "$(git rev-parse HEAD)" == '65e583f03ac5c5983e49cc35179fe7a65908b9c1' ]] || exit 1
mkdir -p /workspace/comp8851/logs/rgtan
PY=/workspace/comp8851/env/rgtan-legacy/bin/python
set -o pipefail
"$PY" -u /workspace/comp8851/pipeline/RGTAN_Elliptic_TR40_Crossfit_v1.py \
  2>&1 | tee /workspace/comp8851/logs/rgtan/RGTAN_Elliptic_TR40_RealCrossfit_v1.log
"$PY" -u /workspace/comp8851/pipeline/RGTAN_Elliptic_RealRisk_GradientPilot_v1.py \
  2>&1 | tee /workspace/comp8851/logs/rgtan/RGTAN_Elliptic_RealRisk_GradientPilot_v1.log
echo 'RGTAN_Elliptic_I1_TR40_CROSSFIT_AND_PILOT=PASS'
