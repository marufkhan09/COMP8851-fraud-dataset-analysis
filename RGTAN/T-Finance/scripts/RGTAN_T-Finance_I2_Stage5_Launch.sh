#!/usr/bin/env bash
# One-launch, resumable RGTAN x T-Finance I2 candidate final training.
# NEVER accesses held-out test labels or starts test scoring.
set -Eeuo pipefail
cd /workspace
[[ "$(hostname)" == "f24d1bb2672d" ]] || { echo 'STOP: This script belongs to I2 (57075171) only'; exit 1; }
ROOT=/workspace/comp8851
PY="$ROOT/env/rgtan-legacy/bin/python"
[[ -x "$PY" ]] || { echo 'STOP: Legacy python not found'; exit 1; }
source "$ROOT/bootstrap/enable_rgtan_cuda11.sh"
export PYTHONPATH="$ROOT/repos/antifraud:${PYTHONPATH:-}"
command -v flock >/dev/null || { echo 'STOP: flock command is required'; exit 1; }
exec 9>"$ROOT/pipeline/RGTAN_T-Finance_I2_Stage5_SingleRun.lock"
flock -n 9 || { echo 'STOP: RGTAN_T-Finance_I2 Stage5 already running'; exit 1; }
trap 'echo "RGTAN_T-Finance_I2_STAGE5_ERROR at line $LINENO; inspect master log"' ERR
FINAL="$ROOT/pipeline/RGTAN_T-Finance_I2_Stage5_FrozenCandidateFinals_v1.py"
CROSS="$ROOT/pipeline/RGTAN_T-Finance_I2_Stage5_RatiosCrossfit_v1.py"
[[ -f "$FINAL" && -f "$CROSS" ]] || { echo 'STOP: Stage5 scripts missing'; exit 1; }
echo 'RGTAN_T-Finance_I2_STAGE5 START. FROZEN TR40 TUNING WINNER; ORIGINAL FULL 42M GRAPH; [10,10] CANDIDATE FANOUT; NO TEST SCORING; SUPERVISOR APPROVAL REQUIRED'
echo '--- Stage5a: Verify all twelve finished validation-only tuning trials and freeze winner ---'
"$PY" -u "$FINAL" --mode freeze
echo '--- Stage5b: Build/reuse OOF crossfits TR30/TR20/TR10 in ONE exact 2hop pass ---'
"$PY" -u "$CROSS"
echo '--- Stage5c: Four ratios x three seeds, validation-frozen candidate trainings ---'
"$PY" -u "$FINAL" --mode train
echo 'RGTAN_T-Finance_I2_STAGE5_ALL_12_CANDIDATE_FINALS_COMPLETE=PASS; TEST_SCORED=NO; SUPERVISOR_APPROVAL_REQUIRED=YES'
