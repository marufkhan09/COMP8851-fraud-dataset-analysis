#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
ROOT=/workspace/comp8851
source "$ROOT/bootstrap/enable_eogfd_cuda11.sh"
PY="$ROOT/env/eogfd-legacy/bin/python"
SCRIPT=/workspace/EOGFD_NEWI_STAGE3/stage3_runner.py
mkdir -p "$ROOT/logs/eogfd_newi_57212396" "$ROOT/locks"
exec 9>"$ROOT/locks/eogfd_newi_gpu0.lock"
flock -n 9 || { echo 'STOP: GPU0 EOGFD Stage3 already running'; exit 2; }
test "$(hostname)" = '60d39211b1fd'
test -x "$PY" && test -f "$SCRIPT"
"$PY" - <<'PY'
import torch,dgl
assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1'
assert torch.cuda.is_available() and torch.cuda.device_count()==1
print('EOGFD_NEWI_STAGE3_CUDA_DGL_GPU0=PASS',flush=True)
PY
for dataset in FDCompCN Elliptic; do
 echo "=== EOGFD_${dataset}_STAGE3_VALIDATION_ONLY_START ==="
 "$PY" -u "$SCRIPT" --dataset "$dataset" 2>&1 | tee -a "$ROOT/logs/eogfd_newi_57212396/${dataset}_Stage3.log"
 echo "EOGFD_${dataset}_STAGE3_SEQUENCE_STEP=PASS"
done
echo 'EOGFD_NEWI_BOTH_CONDITIONAL_STAGE3_TUNINGS_COMPLETE=PASS TEST_SCORED=NO FINAL_TRAINING_STARTED=NO'
