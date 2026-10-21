#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname)" == "2be00658fac4" ]] || { echo "EOGFD_YelpChi_I1_STAGE4_WRONG_HOST_STOP"; exit 8; }
cd /workspace
source /workspace/comp8851/bootstrap/enable_rgtan_cuda11.sh
PY=/workspace/comp8851/env/rgtan-legacy/bin/python
[[ -x "$PY" ]] || { echo "EOGFD_YelpChi_I1_STAGE4_PY_MISSING_STOP"; exit 7; }
# Reject if Stage3 is in progress; retain Stage3 lock for duration to prevent backward tuning relaunch.
exec 9>/workspace/comp8851/logs/eogfd/EOGFD_YelpChi_I1_Stage3.lock
flock -n 9 || { echo "EOGFD_YelpChi_I1_STAGE4_STAGE3_STILL_RUNNING_STOP"; exit 10; }
exec flock -n /workspace/comp8851/logs/eogfd/EOGFD_YelpChi_I1_Stage4.lock "$PY" -u /workspace/comp8851/eogfd/pipeline/EOGFD_YelpChi_I1_Stage4_FrozenFinals12.py
