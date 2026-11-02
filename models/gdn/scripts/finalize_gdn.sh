#!/usr/bin/env bash
# Packs the finished GDN work into the team repo layout, ready to download.
#   /workspace/gdn_export.tar.gz   -> models/gdn/ and results/unified/gdn_results/ (no checkpoints)
#   /workspace/gdn_checkpoints.tar -> every unified run's checkpoints/best_model.pt (kept off GitHub)
# Usage on the instance: bash /workspace/gdn_vast/finalize_gdn.sh
set -euo pipefail
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate gdn
cd /workspace/gdn_vast

EXP=/workspace/gdn_export
rm -rf "$EXP"
mkdir -p "$EXP/models/gdn/scripts" "$EXP/models/gdn/diagnostics" "$EXP/models/gdn/native_reproduction" \
         "$EXP/models/gdn/splits" "$EXP/results/unified/gdn_results"

echo "== summaries"
python summarize_gdn.py > "$EXP/models/gdn/summary_report.txt"
cp full_metrics_summary.csv timing_summary.csv aggregate_summary.csv README.md "$EXP/models/gdn/"

echo "== Elliptic NaN evidence (raw features, then column min-max)"
python diag_nan.py --dataset elliptic --batches 10 > "$EXP/models/gdn/diagnostics/elliptic_raw_features.log" 2>&1 || true
python diag_nan.py --dataset elliptic --batches 18 --transform column_minmax \
    > "$EXP/models/gdn/diagnostics/elliptic_column_minmax.log" 2>&1 || true

echo "== scripts, native log, splits"
cp run_gdn_unified.py run_gdn_unified_v1.py batch_run_gdn.py build_gdn_views.py summarize_gdn.py diag_nan.py \
   finalize_gdn.sh "$EXP/models/gdn/scripts/"
cp native_yelp.log "$EXP/models/gdn/native_reproduction/"
cp splits/*_seed2_nested_splits.npz "$EXP/models/gdn/splits/"
( cd splits && sha256sum *_seed2_nested_splits.npz ) > "$EXP/models/gdn/splits/SHA256SUMS"
( cd /workspace/gdn && git rev-parse HEAD ) > "$EXP/models/gdn/gdn_source_commit.txt"

echo "== per-run evidence bundles (unified track only, no checkpoints)"
for ds in yelp amazon comp tfinance elliptic tsocial; do
    mkdir -p "$EXP/results/unified/gdn_results/$ds"
    rsync -a --exclude checkpoints "results/gdn/$ds/unified" "$EXP/results/unified/gdn_results/$ds/"
done

n_runs=$(find "$EXP/results/unified/gdn_results" -name summary.json | wc -l)
echo "runs exported: $n_runs (expected 72)"
big=$(find "$EXP" -type f -size +95M)
if [ -n "$big" ]; then echo "FILES TOO BIG FOR GITHUB:"; echo "$big"; exit 1; fi

echo "== archives"
tar czf /workspace/gdn_export.tar.gz -C /workspace gdn_export
( cd results && find gdn -path '*/unified/*' -name best_model.pt | sort | tar cf /workspace/gdn_checkpoints.tar -T - )
ls -la /workspace/gdn_export.tar.gz /workspace/gdn_checkpoints.tar
sha256sum /workspace/gdn_export.tar.gz /workspace/gdn_checkpoints.tar
echo "checkpoints archived: $(tar tf /workspace/gdn_checkpoints.tar | wc -l) (expected 72)"
echo "DONE"
