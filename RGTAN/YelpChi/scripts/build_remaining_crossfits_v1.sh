#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace/comp8851
export DGLBACKEND=pytorch
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/scripts/rgtan_yelpchi_v1:$ROOT/repos/antifraud${PYTHONPATH:+:$PYTHONPATH}"

PY="$ROOT/envs/rgtan/bin/python"
WRAP="$ROOT/diagnostics/rgtan/rgtan_python_cuda_i3.sh"
CROSS="$ROOT/scripts/rgtan_yelpchi_v1/rgtan_yelpchi_crossfit_v1.py"
ASSETS="$ROOT/assets/rgtan/yelpchi"
LOCK="$ROOT/diagnostics/rgtan/yelpchi_remaining_crossfits_v1.lock"
STATUS="$ROOT/diagnostics/rgtan/yelpchi_remaining_crossfits_v1.exit"

exec 9>"$LOCK"
flock -n 9 || {
  echo "I3_CROSSFIT_ALREADY_LOCKED"
  exit 1
}

trap 'rc=$?; printf "%s\n" "$rc" > "$STATUS"; echo "I3_CROSSFIT_EXIT_CODE=$rc"' EXIT

for RATIO in TR30 TR20 TR10; do
  echo "========== BUILD_START=$RATIO =========="

  bash "$WRAP" "$PY" -u "$CROSS" --ratio "$RATIO"

  echo "========== VERIFY=$RATIO =========="

  "$PY" - "$ASSETS" "$RATIO" <<'VERIFY_ADAPTER'
import hashlib
import json
import sys
from pathlib import Path

assets = Path(sys.argv[1])
ratio = sys.argv[2]

p = assets / f"yelpchi_{ratio.lower()}_crossfit.npz"
m = json.loads(
    (assets / f"yelpchi_{ratio.lower()}_crossfit_manifest.json").read_text()
)

assert p.is_file()
assert m["ratio"] == ratio
assert m["status"] == "RGTAN_YELPCHI_CROSSFIT_PASS"
assert m["source_sha256"] == (
    "fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42"
)
assert m["split_sha256"] == (
    "0ea0af36dfc5a3a1f381ea2e3168377e45826498b6063190185c673e9ec8c22b"
)

assert m["fold_labels_masked_from_their_own_risk_views"] is True
assert m["validation_test_labels_used_in_features"] is False
assert m["normalization_fit_on_oof_training_only"] is True

h = hashlib.sha256()
with p.open("rb") as f:
    for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
        h.update(block)

assert h.hexdigest() == m["artifact_sha256"]

print(
    f"I3_CROSSFIT_VERIFIED={ratio} SHA256={h.hexdigest()}",
    flush=True
)
VERIFY_ADAPTER

  echo "========== BUILD_COMPLETE=$RATIO =========="
done

echo "I3_ALL_FOUR_CROSSFIT_ADAPTERS_READY=PASS"
echo "I3_HELD_OUT_TEST_EVALUATIONS=ZERO"
