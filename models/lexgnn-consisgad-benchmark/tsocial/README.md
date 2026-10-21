# LEX-GNN and ConsisGAD on T-Social

Status: 24 final runs per model complete (TR40/30/20/10 x seeds 2/42/72), all PASS.
Raw data: T-Social, sha256 8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0
(5,781,065 nodes, 146,211,016 directed edges, 174,280 fraud). Homogeneous single-relation view.

## Protocol deviations (declared)
- Split is PROVISIONAL: regenerated from the documented convention (40/20/40 stratified base,
  nested TR30/TR20/TR10, split seed 2), NOT the BWGNN/PMP/GAAP frozen file
  (tsocial_seed2_nested_splits.npz, sha256 b7e76835...). Hashes in splits/*.json.
- ConsisGAD: tuning 20 epochs, final 30 epochs (original 100). Validation AUROC plateaued early.
- LEX-GNN: tuning 2 epochs and 3 trials, final 5 epochs with early stopping 3 (original 300/100).
  One epoch costs about 434 s on T-Social.
- Tuning and final runs were executed concurrently on one A6000, so per-run wall times are inflated.
  Use the isolated two-epoch gate runs (evidence/compatibility) for efficiency figures.

## Results (test, mean +- SD over 3 seeds)
See evidence/final_runs/tables/tsocial_results_summary.csv
- ConsisGAD AUPRC TR40 0.551, TR30 0.526, TR20 0.530, TR10 0.537
- LEX-GNN   AUPRC TR40 0.997, TR30 0.994, TR20 0.989, TR10 0.979

## Leakage audit
LEX-GNN reads only y_mask (training labels, others = 2) in its forward pass (model.py:138);
true labels are read only where y_mask != 2 (model_trainer.py:74-75).
Excluded from git: datasets, checkpoints, prediction arrays, split arrays.
