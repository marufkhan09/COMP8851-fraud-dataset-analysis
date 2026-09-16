#!/bin/bash
set -e  # stop on first real error, don't cascade into corrupted next steps

MASTER_LOG="/tmp/gaga_overnight_master.log"
PREPROC_DIR="/workspace/COMP8851-fraud-dataset-analysis/models/gaga/gaga_upstream/pytorch_gaga/preprocessing"
TRAIN_DIR="/workspace/COMP8851-fraud-dataset-analysis/models/gaga/gaga_upstream/pytorch_gaga"

log() { echo "[$(date -u +%H:%M:%S)] $1" | tee -a "$MASTER_LOG"; }

log "=== OVERNIGHT RUN START ==="

# --- ELLIPTIC ---
log "--- Elliptic: split ---"
cd "$PREPROC_DIR"
python dataset_split.py --dataset elliptic --save_dir seq_data --train_size 0.4 --val_size 0.198 2>&1 | tee /tmp/gaga_elliptic_split.log
if ! grep -q "Elapsed time" /tmp/gaga_elliptic_split.log; then
    log "FAILED: Elliptic split did not complete. Check /tmp/gaga_elliptic_split.log"
    exit 1
fi
log "Elliptic split OK"

log "--- Elliptic: sequence generation ---"
python graph2seq_single.py --dataset elliptic --fanouts -1 -1 --save_dir seq_data --train_size 0.4 --val_size 0.198 --add_self_loop --norm_feat 2>&1 | tee /tmp/gaga_elliptic_graph2seq.log
if ! grep -q "Elapsed Time" /tmp/gaga_elliptic_graph2seq.log; then
    log "FAILED: Elliptic graph2seq did not complete. Check /tmp/gaga_elliptic_graph2seq.log"
    exit 1
fi
log "Elliptic sequence generation OK"

log "--- Elliptic: training ---"
cd "$TRAIN_DIR"
echo "=== elliptic_paper.json contents ===" | tee -a "$MASTER_LOG"
cat configs/elliptic_paper.json | tee -a "$MASTER_LOG"
python main_transformer.py --config configs/elliptic_paper.json --gpu 0 --log_dir logs --early_stop 100 2>&1 | tee /tmp/gaga_elliptic_train.log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "FAILED: Elliptic training exited with error. Check /tmp/gaga_elliptic_train.log"
    exit 1
fi
log "Elliptic training OK"

# --- FDCOMPCN ---
log "--- FDCompCN: split ---"
cd "$PREPROC_DIR"
python dataset_split.py --dataset fdcompcn --save_dir seq_data --train_size 0.4 --val_size 0.198 2>&1 | tee /tmp/gaga_fdcompcn_split.log
if ! grep -q "Elapsed time" /tmp/gaga_fdcompcn_split.log; then
    log "FAILED: FDCompCN split did not complete. Check /tmp/gaga_fdcompcn_split.log"
    exit 1
fi
log "FDCompCN split OK"

log "--- FDCompCN: sequence generation ---"
python graph2seq_single.py --dataset fdcompcn --fanouts -1 -1 --save_dir seq_data --train_size 0.4 --val_size 0.198 --add_self_loop --norm_feat 2>&1 | tee /tmp/gaga_fdcompcn_graph2seq.log
if ! grep -q "Elapsed Time" /tmp/gaga_fdcompcn_graph2seq.log; then
    log "FAILED: FDCompCN graph2seq did not complete. Check /tmp/gaga_fdcompcn_graph2seq.log"
    exit 1
fi
log "FDCompCN sequence generation OK"

log "--- FDCompCN: training ---"
cd "$TRAIN_DIR"
echo "=== fdcompcn_paper.json contents ===" | tee -a "$MASTER_LOG"
cat configs/fdcompcn_paper.json | tee -a "$MASTER_LOG"
python main_transformer.py --config configs/fdcompcn_paper.json --gpu 0 --log_dir logs --early_stop 100 2>&1 | tee /tmp/gaga_fdcompcn_train.log
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "FAILED: FDCompCN training exited with error. Check /tmp/gaga_fdcompcn_train.log"
    exit 1
fi
log "FDCompCN training OK"

log "=== OVERNIGHT RUN COMPLETE — ALL STEPS SUCCEEDED ==="
