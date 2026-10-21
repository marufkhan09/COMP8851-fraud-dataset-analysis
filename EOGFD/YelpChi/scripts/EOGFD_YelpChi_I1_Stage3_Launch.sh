#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname)" == "2be00658fac4" ]] || { echo 'EOGFD_YelpChi_I1_WRONG_HOST_STOP'; exit 7; }
cd /workspace
[[ -x /workspace/comp8851/env/rgtan-legacy/bin/python ]] || { echo 'EOGFD_YelpChi_I1_NO_PINNED_PYTHON_STOP'; exit 6; }
[[ -f /workspace/comp8851/eogfd/pipeline/EOGFD_YelpChi_I1_Stage3_ValidationTune12.py ]] || { echo 'EOGFD_YelpChi_I1_STAGE3_SCRIPT_MISSING_STOP'; exit 6; }
source /workspace/comp8851/bootstrap/enable_rgtan_cuda11.sh
mkdir -p /workspace/comp8851/logs/eogfd
# flock guarantees only one running Stage3 per dataset even if a terminal reconnects.
exec flock -n /workspace/comp8851/logs/eogfd/EOGFD_YelpChi_I1_Stage3.lock /workspace/comp8851/env/rgtan-legacy/bin/python -u /workspace/comp8851/eogfd/pipeline/EOGFD_YelpChi_I1_Stage3_ValidationTune12.py
