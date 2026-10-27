#!/usr/bin/env bash
set -euo pipefail
[[ "$(hostname)" == "f24d1bb2672d" ]] || { echo 'EOGFD_Amazon_I2_WRONG_INSTANCE_STOP'; exit 5; }
cd /workspace
[[ -x /workspace/comp8851/env/rgtan-legacy/bin/python ]] || { echo 'EOGFD_Amazon_I2_MISSING_PINNED_ENV_STOP'; exit 6; }
mkdir -p /workspace/comp8851/logs/eogfd
[[ ! -e /workspace/comp8851/logs/eogfd/EOGFD_Amazon_I2_Stage2_FullGraphPilot.json ]] || { echo 'EOGFD_Amazon_I2_ALREADY_COMPLETE_NO_RERUN'; exit 7; }
source /workspace/comp8851/bootstrap/enable_rgtan_cuda11.sh
exec flock -n /workspace/comp8851/logs/eogfd/EOGFD_Amazon_I2_Stage2.lock /workspace/comp8851/env/rgtan-legacy/bin/python -u /workspace/comp8851/eogfd/pipeline/EOGFD_Amazon_I2_Stage2_RealGraphPilot.py
