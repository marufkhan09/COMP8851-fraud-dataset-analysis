# LEX-GNN and ConsisGAD — Final YelpChi Benchmark

## Completion status

- Final runs: 24/24
- Failed final runs: 0
- Models: LEX-GNN and ConsisGAD
- Training ratios: TR40, TR30, TR20 and TR10
- Training seeds: 2, 42 and 72
- Frozen split seed: 2
- Strict artifact verification: passed
- Test isolation: passed
- Three-seed aggregation: passed
- Publication-figure review: passed

## Main result

LEX-GNN produced stronger mean results across all five reported
performance metrics at TR20, TR30 and TR40.

At TR10, ConsisGAD produced slightly higher AUROC, AUPRC and G-Mean,
while Macro-F1 and fraud-class F1 were similar.

All hyperparameter selection used validation data only. The test set was
evaluated once after final training.

## Complete binary archive

The full locally verified archive includes checkpoints and prediction
arrays:

`lexgnn_consisgad_yelpchi_complete_20261018T122555Z.tar.gz`

SHA256:

`e55b2418c9a6f2e31ad174ff06fae0c7dd7e6e516848bcb424aeb8d3828f6bbc`

Large checkpoint and prediction files are intentionally excluded from
Git because they are preserved in the verified archive.
