#!/usr/bin/env bash
# Package results for download. Never deletes anything.
#   results_full_<stamp>.tar.gz   everything incl. checkpoints and tuning trials
#   results_light_<stamp>.tar.gz  tables, summaries, predictions, logs (no checkpoints)
set -euo pipefail
cd "$(dirname "$0")/.."
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
tar -czf "results_full_${STAMP}.tar.gz" --exclude='*.tar.gz' results configs third_party/manifest.json data/canonical/data_status.json
tar -czf "results_light_${STAMP}.tar.gz" --exclude='*.tar.gz' --exclude='*/checkpoints/*' results configs data/canonical/data_status.json
sha256sum "results_full_${STAMP}.tar.gz" "results_light_${STAMP}.tar.gz" | tee "results_${STAMP}.sha256"
mv "results_full_${STAMP}.tar.gz" "results_light_${STAMP}.tar.gz" "results_${STAMP}.sha256" results/ 2>/dev/null || true
echo "packaged: results/results_full_${STAMP}.tar.gz and results/results_light_${STAMP}.tar.gz"
echo "copy off the instance, e.g.:  scp -P <port> root@<host>:$(pwd)/results/results_full_${STAMP}.tar.gz ."
