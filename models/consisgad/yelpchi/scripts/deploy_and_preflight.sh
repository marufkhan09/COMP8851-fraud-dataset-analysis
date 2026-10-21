#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=/workspace/comp8851_two_models
EVIDENCE="$ROOT/evidence/adapter_fixes/yelpchi_adapter_v1"

echo "============================================================"
echo "DEPLOYING YELPCHI ADAPTER V1"
echo "============================================================"

test -d "$ROOT"
test -x "$ROOT/venvs/lexgnn/bin/python"
test -x "$ROOT/venvs/consisgad/bin/python"
test -f "$ROOT/datasets/canonical/yelpchi/YelpChi.mat"
test -f "$ROOT/shared/splits/yelpchi/yelpchi_tr40_split_seed2.npz"

DATA_SHA=$(sha256sum "$ROOT/datasets/canonical/yelpchi/YelpChi.mat" | awk '{print $1}')
SPLIT_SHA=$(sha256sum "$ROOT/shared/splits/yelpchi/yelpchi_tr40_split_seed2.npz" | awk '{print $1}')

test "$DATA_SHA" = "fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42"
test "$SPLIT_SHA" = "830d884f365eacbb0669f4808ccdb53acc0afdc5b655f6a74c66dc8ec928ef0a"

mkdir -p \
  "$EVIDENCE/before" \
  "$ROOT/adapters/lexgnn" \
  "$ROOT/adapters/consisgad" \
  "$ROOT/configs/lexgnn" \
  "$ROOT/configs/consisgad"

for CURRENT in \
  "$ROOT/adapters/benchmark_common.py" \
  "$ROOT/adapters/lexgnn/train_yelpchi_unified.py" \
  "$ROOT/adapters/consisgad/train_yelpchi_unified.py"
do
  if [ -f "$CURRENT" ]; then
    cp "$CURRENT" "$EVIDENCE/before/$(basename "$CURRENT")"
  fi
done

cp "$PACKAGE_ROOT/adapters/benchmark_common.py" "$ROOT/adapters/benchmark_common.py"
cp "$PACKAGE_ROOT/adapters/lexgnn/train_yelpchi_unified.py" "$ROOT/adapters/lexgnn/"
cp "$PACKAGE_ROOT/adapters/consisgad/train_yelpchi_unified.py" "$ROOT/adapters/consisgad/"
cp "$PACKAGE_ROOT/configs/lexgnn/yelpchi_tr40_seed2_smoke.json" "$ROOT/configs/lexgnn/"
cp "$PACKAGE_ROOT/configs/consisgad/yelpchi_tr40_seed2_smoke.json" "$ROOT/configs/consisgad/"

"$ROOT/venvs/lexgnn/bin/python" -m py_compile \
  "$ROOT/adapters/benchmark_common.py" \
  "$ROOT/adapters/lexgnn/train_yelpchi_unified.py"

"$ROOT/venvs/consisgad/bin/python" -m py_compile \
  "$ROOT/adapters/benchmark_common.py" \
  "$ROOT/adapters/consisgad/train_yelpchi_unified.py"

sha256sum \
  "$ROOT/adapters/benchmark_common.py" \
  "$ROOT/adapters/lexgnn/train_yelpchi_unified.py" \
  "$ROOT/adapters/consisgad/train_yelpchi_unified.py" \
  "$ROOT/configs/lexgnn/yelpchi_tr40_seed2_smoke.json" \
  "$ROOT/configs/consisgad/yelpchi_tr40_seed2_smoke.json" \
  > "$EVIDENCE/deployed_files.sha256"

echo "Canonical YelpChi checksum: PASS"
echo "Frozen TR40 checksum: PASS"
echo "Adapter syntax: PASS"
echo "YELPCHI ADAPTER V1 DEPLOYMENT AND PREFLIGHT: PASS"
echo "Training has NOT started."
