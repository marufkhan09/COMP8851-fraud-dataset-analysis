#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
ROOT=/workspace/comp8851
STAGE4=/workspace/EOGFD_NEWI_STAGE4
PY="$ROOT/env/eogfd-legacy/bin/python"
REPO="$ROOT/repos/EOGFD"
BOOT="$ROOT/bootstrap/enable_eogfd_cuda11.sh"
SCRIPT="$STAGE4/stage4_runner.py"

# Explicit acknowledgment: not supervisor-approved benchmark results.
test "${EOGFD_CONDITIONAL_EXECUTION:-}" = '1' || {
  echo 'STOP: Stage4 adapted experiments lack recorded Venus approval. Set EOGFD_CONDITIONAL_EXECUTION=1 only to run conditional exploratory training; this does not imply supervisor approval.'
  exit 2
}
test "$(hostname)" = '60d39211b1fd'
test -x "$PY" && test -f "$SCRIPT" && test -f "$BOOT"
source "$BOOT"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 DGLBACKEND=pytorch PYTHONDONTWRITEBYTECODE=1
mkdir -p "$ROOT/logs/eogfd_newi_57212396" "$ROOT/locks"
exec 9>"$ROOT/locks/eogfd_newi_gpu0.lock"
flock -n 9 || { echo 'STOP: GPU0 EOGFD pipeline already running'; exit 2; }
"$PY" - <<'PY'
import torch,dgl
from models.eogfd import EOGFDHeteroModel
assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1'
assert torch.cuda.is_available() and torch.cuda.device_count()==1
print('EOGFD_STAGE4_PINNED_HETEROMODEL_CUDA_GPU0=PASS',flush=True)
PY
for dataset in FDCompCN Elliptic; do
  echo "=== EOGFD_${dataset}_STAGE4_CONDITIONAL_FROZEN_FINALS_START ==="
  "$PY" -u "$SCRIPT" --dataset "$dataset" 2>&1 | tee -a "$ROOT/logs/eogfd_newi_57212396/${dataset}_Stage4.log"
  echo "EOGFD_${dataset}_STAGE4_SEQUENCE_STEP=PASS"
done
echo 'EOGFD_NEWI_BOTH_CONDITIONAL_STAGE4_12X2_FROZEN_TRAININGS_COMPLETE=PASS TEST_SCORED=NO VENUS_APPROVAL_NOT_RECORDED'
