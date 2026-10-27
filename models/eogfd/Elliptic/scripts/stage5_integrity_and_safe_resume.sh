#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
ROOT=/workspace/comp8851
PREV=/workspace/EOGFD_NEWI_STAGE5
RECOVER=/workspace/EOGFD_NEWI_STAGE5_RECOVERY
PY="$ROOT/env/eogfd-legacy/bin/python"
REPO="$ROOT/repos/EOGFD"
LOG="$ROOT/logs/EOGFD_57212396_Stage5_Master.log"
BOOT="$ROOT/bootstrap/enable_eogfd_cuda11.sh"
: "${EOGFD_CONDITIONAL_TEST_ONCE:?STOP: explicit conditional test once consent required}"
test "$EOGFD_CONDITIONAL_TEST_ONCE" = 1
test "$(hostname)" = 60d39211b1fd
test -f "$LOG" && grep -Fq 'FROZEN_CHECKPOINT_VAL_CONFLICT' "$LOG"
test -f "$PREV/stage5_test_once.py"
test -f "$RECOVER/KIT_SHA256_MANIFEST.json"
source "$BOOT"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 DGLBACKEND=pytorch PYTHONDONTWRITEBYTECODE=1
# No hidden unverified modifications to the original Stage5 installation.
"$PY" - <<'PY'
import json,hashlib
from pathlib import Path
orig=Path('/workspace/EOGFD_NEWI_STAGE5')
rec=Path('/workspace/EOGFD_NEWI_STAGE5_RECOVERY')
for folder in (orig,rec):
    mani=json.loads((folder/'KIT_SHA256_MANIFEST.json').read_text())
    for filename,expected in mani.items():
        actual=hashlib.sha256((folder/filename).read_bytes()).hexdigest()
        assert actual==expected, ('STOP: PACKAGE_SCRIPT_MUTATED',str(folder/filename))
assert hashlib.sha256((orig/'stage5_test_once.py').read_bytes()).hexdigest()=="b775575f81d45a610a01e43a003b42854662f9976afd2750a1bf11f0df8e4269"
print('EOGFD_STAGE5_ORIGINAL_AND_RECOVERY_SCRIPT_INTEGRITY=PASS',flush=True)
PY
# First perform all 24 checkpoint/epoch/replay comparisons WITHOUT test labels.
echo '=== READ-ONLY PRETEST FORENSICS (ALL 24 FROZEN RUNS) ==='
"$PY" -u "$RECOVER/stage5_pretest_forensics.py"
echo '=== DUAL-SNAPSHOT AUDIT + ONCE-ONLY SCORING; ABORTS ON ANY MATERIAL DRIFT ==='
# All existing Stage5 safety gates, including exclusive test claims, are preserved.
bash "$RECOVER/stage5_launch_both_dualsnapshot_audited.sh"
