#!/usr/bin/env bash
set -euo pipefail
source /workspace/activate_comp8851_elliptic.sh

WORK=/workspace/comp8851_elliptic_work
BATCH="$WORK/elliptic_reconstructed_v1"
export PYTHONPATH="$WORK/adapters"

for ratio in 40 30 20 10; do
  for seed in 2 42 72; do
    id="lexgnn_tr${ratio}_seed${seed}"
    config="$BATCH/configs/${id}.json"
    log="$BATCH/logs/${id}.log"
    marker="$BATCH/logs/${id}.PASS"

    if [[ -f "$marker" ]]; then
      echo "SKIP VERIFIED: $id"
      continue
    fi

    echo "===== START $id ====="

    if python -u "$WORK/adapters/lexgnn/train_elliptic_compat.py" \
       --config "$config" > "$log" 2>&1; then

      BATCH="$BATCH" ID="$id" python - <<'PY'
import json,os
from pathlib import Path

batch=Path(os.environ["BATCH"])
id=os.environ["ID"]
root=batch/"results"/"lexgnn"
matches=[]
for p in root.glob("*/summary.json"):
    d=json.loads(p.read_text())
    if d.get("run_mode")==f"reconstructed_v1_{id[7:]}_validation_only":
        if d.get("status")=="PASS" and d.get("epochs_completed")==100 and d.get("test_evaluated") is False:
            matches.append(p)
assert len(matches)==1,(id,matches)
(batch/"logs"/f"{id}.PASS").write_text(str(matches[0])+"\n")
print("VERIFIED:",id,flush=True)
PY

    else
      echo "FAILED: $id"
      tail -n 30 "$log"
      exit 1
    fi
  done
done

echo "PASS: ALL 12 LEX-GNN VALIDATION RUNS COMPLETED"
