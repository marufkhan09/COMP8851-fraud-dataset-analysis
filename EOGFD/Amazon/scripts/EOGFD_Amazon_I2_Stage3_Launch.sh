#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname)" == "f24d1bb2672d" ]] || { echo 'EOGFD_Amazon_I2_WRONG_HOST_STOP'; exit 7; }
cd /workspace
[[ -x /workspace/comp8851/env/rgtan-legacy/bin/python ]] || { echo 'EOGFD_Amazon_I2_NO_PINNED_PYTHON_STOP'; exit 6; }
[[ -f /workspace/comp8851/eogfd/pipeline/EOGFD_Amazon_I2_Stage3_ValidationTune12.py ]] || { echo 'EOGFD_Amazon_I2_STAGE3_SCRIPT_MISSING_STOP'; exit 6; }
source /workspace/comp8851/bootstrap/enable_rgtan_cuda11.sh
mkdir -p /workspace/comp8851/logs/eogfd
# flock guarantees only one running Stage3 per dataset even if a terminal reconnects.
exec flock -n /workspace/comp8851/logs/eogfd/EOGFD_Amazon_I2_Stage3.lock /workspace/comp8851/env/rgtan-legacy/bin/python -u /workspace/comp8851/eogfd/pipeline/EOGFD_Amazon_I2_Stage3_ValidationTune12.py
