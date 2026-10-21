# LEX-GNN — T-Finance benchmark

## Scope

12 final experiments:
- Training ratios: TR10, TR20, TR30, TR40
- Model initialization seeds: 2, 42, 72
- Persistent dataset split seed: 2
- Canonical dataset: 39,357 nodes, 1,804 fraud nodes
- Canonical dataset SHA-256:
  `b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700`

## Model and selection

- Original LEX-GNN architecture
- Selected learning rate: 0.001
- Maximum epochs: 100
- Early-stopping patience: 20
- Checkpoint selection: validation AUPRC
- Test evaluation performed only for final runs
- No initialization restarts during final runs

The original learning-rate 0.005 baseline was retained as diagnostic
evidence. Its checkpoint selection used AUROC. A subsequent 0.001
trial exposed an AUPRC checkpoint-selection implementation bug; it was
interrupted and replaced with the corrected trial. The selected
corrected tuning run achieved validation AUPRC 0.8301 at epoch 58.

## Results

See:
- `results/tables/lexgnn_tfinance_12_test_results.csv`
- `results/tables/lexgnn_tfinance_3seed_summary.csv`
- `results/final/` for individual experiment evidence.

All 12 final training runs exited successfully and passed the
post-training artifact verification.

## Interpretation

Seed 42 produced zero fraud-class F1 at TR20, TR30 and TR40.
These results are preserved and included in the aggregates. They
indicate initialization sensitivity under the fixed experimental
protocol, rather than grounds for selective replacement.

## Reproduction

The scripts and JSON configuration files in this package document
the original execution environment paths and frozen experimental
settings. Those absolute paths are specific to the original Vast.ai
workspace and must be remapped if reproduced elsewhere.

The original 652 MB canonical DGL dataset is excluded from Git.
The binary checkpoints and prediction arrays are preserved in
the external complete backup archive:

`LEXGNN_TFINANCE_COMPLETE_20261021.tar.gz`

Archive SHA-256:
`69a745a5356f90da3222b0bcf3f4127c62ce255168d31a181c12aaec313c7455`

The archive must be retained separately; this Git directory alone
does not contain every binary required for exact replay.
