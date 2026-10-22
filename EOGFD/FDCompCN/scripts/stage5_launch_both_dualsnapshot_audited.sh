#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
ROOT=/workspace/comp8851
REPO="$ROOT/repos/EOGFD"
KIT=/workspace/EOGFD_NEWI_STAGE5
RECOVERY=/workspace/EOGFD_NEWI_STAGE5_RECOVERY
PY="$ROOT/env/eogfd-legacy/bin/python"
BOOT="$ROOT/bootstrap/enable_eogfd_cuda11.sh"
STAGE4_LOG="$ROOT/logs/EOGFD_57212396_Stage4_Master.log"

# Explicit opt-in for irreversible, once-only conditional test scoring.
test "${EOGFD_CONDITIONAL_TEST_ONCE:-}" = '1' || {
  echo 'STOP: To score the held-out test once, set EOGFD_CONDITIONAL_TEST_ONCE=1 explicitly.'
  echo 'This is conditional/unapproved benchmark testing, not evidence of Venus approval.'
  exit 2
}
test "$(hostname)" = '60d39211b1fd'
test -x "$PY" && test -f "$BOOT" && test -f "$RECOVERY/stage5_test_once_dualsnapshot_audited.py"
test "$(git -C "$REPO" rev-parse HEAD)" = a4726f173bc8c923109ddcb2786f4d9c5206ae7f
source "$BOOT"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 DGLBACKEND=pytorch PYTHONDONTWRITEBYTECODE=1

grep -Fqx 'EOGFD_FDCompCN_STAGE4_12_FROZEN_TRAININGS_COMPLETE=PASS TEST_SCORED=NO VENUS_APPROVAL_NOT_RECORDED' "$STAGE4_LOG"
grep -Fqx 'EOGFD_Elliptic_STAGE4_12_FROZEN_TRAININGS_COMPLETE=PASS TEST_SCORED=NO VENUS_APPROVAL_NOT_RECORDED' "$STAGE4_LOG"
grep -Fqx 'EOGFD_NEWI_BOTH_CONDITIONAL_STAGE4_12X2_FROZEN_TRAININGS_COMPLETE=PASS TEST_SCORED=NO VENUS_APPROVAL_NOT_RECORDED' "$STAGE4_LOG"

mkdir -p "$ROOT/locks" "$ROOT/logs/eogfd_newi_57212396"
exec 9>"$ROOT/locks/eogfd_newi_gpu0.lock"
flock -n 9 || { echo 'STOP: EOGFD GPU0 is occupied'; exit 2; }

"$PY" - <<'PY'
import torch,dgl
from models.eogfd import EOGFDHeteroModel
assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1'
assert torch.cuda.is_available() and torch.cuda.device_count()==1
print('EOGFD_NEWI_STAGE5_PINNED_MODEL_CUDA_DGL=PASS',flush=True)
PY

# NO TEST LABEL ACCESS occurs in either audit call.
for dataset in FDCompCN Elliptic; do
    echo "=== EOGFD_${dataset}_STAGE5_ALL12_FROZEN_AUDIT ==="
    "$PY" -u "$RECOVERY/stage5_test_once_dualsnapshot_audited.py" --dataset "$dataset" --phase audit \
      2>&1 | tee -a "$ROOT/logs/eogfd_newi_57212396/${dataset}_Stage5.log"
done

echo 'EOGFD_BOTH_DATASETS_ALL24_PRETEST_FROZEN_AUDITS=PASS TEST_LABELS_NOT_OPENED=YES'

# Opening each held-out label set and scoring is only permitted after all audits pass.
for dataset in FDCompCN Elliptic; do
    echo "=== EOGFD_${dataset}_STAGE5_TEST_SCORING_ONCE ==="
    "$PY" -u "$RECOVERY/stage5_test_once_dualsnapshot_audited.py" --dataset "$dataset" --phase score \
      2>&1 | tee -a "$ROOT/logs/eogfd_newi_57212396/${dataset}_Stage5.log"
done

echo 'EOGFD_NEWI_BOTH_CONDITIONAL_STAGE5_12X2_HELDOUT_TEST_ONCE_COMPLETE=PASS VENUS_APPROVAL_NOT_RECORDED'
