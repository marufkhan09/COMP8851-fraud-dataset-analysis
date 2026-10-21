#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname)" == "2be00658fac4" ]] || { echo 'EOGFD_YelpChi_I1_WRONG_INSTANCE_STOP'; exit 5; }
cd /workspace
[[ -x /workspace/comp8851/env/rgtan-legacy/bin/python ]] || { echo 'EOGFD_YelpChi_I1_MISSING_PINNED_ENV_STOP'; exit 6; }
mkdir -p /workspace/comp8851/logs/eogfd
[[ ! -e /workspace/comp8851/logs/eogfd/EOGFD_YelpChi_I1_Stage2_FullGraphPilot.json ]] || { echo 'EOGFD_YelpChi_I1_ALREADY_COMPLETE_NO_RERUN'; exit 7; }
source /workspace/comp8851/bootstrap/enable_rgtan_cuda11.sh
exec flock -n /workspace/comp8851/logs/eogfd/EOGFD_YelpChi_I1_Stage2.lock /workspace/comp8851/env/rgtan-legacy/bin/python -u /workspace/comp8851/eogfd/pipeline/EOGFD_YelpChi_I1_Stage2_RealGraphPilot.py
