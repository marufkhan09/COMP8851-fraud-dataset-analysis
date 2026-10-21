#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
[ "$(hostname)" = "f24d1bb2672d" ] || { echo 'WRONG_INSTANCE_STOP'; exit 2; }
source /workspace/comp8851/bootstrap/enable_rgtan_cuda11.sh
cd /workspace/comp8851
/workspace/comp8851/env/rgtan-legacy/bin/python -u /workspace/comp8851/preflight/I2_preflight.py 2>&1 | tee /workspace/comp8851/preflight/I2_execution.log
