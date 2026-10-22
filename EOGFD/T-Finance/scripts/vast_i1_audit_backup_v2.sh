#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace
ROOT=/workspace/comp8851
BASE=/workspace/EOGFD_TFINANCE_PRETEST_AUDIT_I1_V2_20261023
PY="$ROOT/env/eogfd-legacy/bin/python"
test "$(hostname)" = 60d39211b1fd || { echo STOP_NOT_I1;exit 2; }
test -x "$PY";test -f "$BASE/i1_pretest_audit_backup_v2.py"
source "$ROOT/bootstrap/enable_eogfd_cuda11.sh"
export CUDA_VISIBLE_DEVICES=0 DGLBACKEND=pytorch PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/repos/EOGFD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/locks" "$ROOT/logs"
exec 9>"$ROOT/locks/eogfd_tfinance_pretest_audit.lock"
flock -n 9 || { echo STOP_PRETEST_AUDIT_ALREADY_RUNNING;exit 2; }
"$PY" -u "$BASE/i1_pretest_audit_backup_v2.py" --audit-and-backup 2>&1 | tee -a "$ROOT/logs/EOGFD_57212396_TFinance_PRETEST_AUDIT_V2.log"
