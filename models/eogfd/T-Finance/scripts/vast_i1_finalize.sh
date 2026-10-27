#!/usr/bin/env bash
# I1 57212396: strict audit, once-only test, read-only publication exporter.
set -Eeuo pipefail
ROOT=/workspace/comp8851
KIT=/workspace/EOGFD_TFINANCE_FINALIZE_I1_20261023
PY="$ROOT/env/eogfd-legacy/bin/python"
REPO="$ROOT/repos/EOGFD"
test -x "$PY" && test -f "$REPO/models/eogfd.py"
test "$(hostname)" = '60d39211b1fd'
test "$(git -C "$REPO" rev-parse HEAD)" = 'a4726f173bc8c923109ddcb2786f4d9c5206ae7f'
test -z "$(git -C "$REPO" status --porcelain)"
source "$ROOT/bootstrap/enable_eogfd_cuda11.sh"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0 DGLBACKEND=pytorch PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
mkdir -p "$ROOT/logs"
case "${1:-}" in
 --audit-only)
  "$PY" -u "$KIT/i1_stage5_once.py" --audit-only | tee -a "$ROOT/logs/EOGFD_57212396_TFinance_STAGE5.log"
  ;;
 --score-once-and-export)
  "$PY" -u "$KIT/i1_stage5_once.py" --audit-only | tee -a "$ROOT/logs/EOGFD_57212396_TFinance_STAGE5.log"
  "$PY" -u "$KIT/i1_stage5_once.py" --score-once | tee -a "$ROOT/logs/EOGFD_57212396_TFinance_STAGE5.log"
  "$PY" -u "$KIT/i1_stage6_public_export.py" --export-public | tee -a "$ROOT/logs/EOGFD_57212396_TFinance_STAGE6.log"
  (cd /workspace && sha256sum -c EOGFD_TFINANCE_I1_GITHUB_PUBLICATION_20261023.zip.sha256)
  echo 'I1_TFINANCE_STAGE5_AND_STAGE6_EXPORT=PASS; MAC_GITHUB_PUSH_PENDING'
  ;;
 --export-only)
  "$PY" -u "$KIT/i1_stage6_public_export.py" --export-public | tee -a "$ROOT/logs/EOGFD_57212396_TFinance_STAGE6.log"
  ;;
 *) echo 'Usage: vast_i1_finalize.sh --audit-only | --score-once-and-export | --export-only' >&2; exit 2;;
esac
