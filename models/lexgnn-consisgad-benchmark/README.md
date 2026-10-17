# LEX-GNN and ConsisGAD Unified Benchmark

This directory contains the reproducible COMP8851 benchmark implementation,
configurations and lightweight evidence for LEX-GNN and ConsisGAD.

## Amazon status

- Compatibility tests completed for both models.
- Validation-only hyperparameter tuning completed.
- TR40, TR30, TR20 and TR10 evaluated.
- Training seeds 2, 42 and 72 evaluated.
- All 24 final runs completed successfully.

## YelpChi status

- Canonical DGL YelpChi validated.
- All 45,954 nodes and 32 features retained.
- Relations net_rsr, net_rtr and net_rur preserved.
- Frozen nested splits validated.
- Two-epoch compatibility adapters and configurations prepared.
- Compatibility training had not started at the time of this commit.

## Evaluation protocol

Checkpoint selection uses validation AUROC, with validation AUPRC as the
tuning tie-breaker. Decision thresholds are selected using validation
Macro-F1. Test data are evaluated once after training and are never used for
hyperparameter or checkpoint selection.

Recorded metrics include AUROC, AUPRC, Macro-F1, Fraud-F1, G-Mean, accuracy,
confusion-matrix counts, timing, peak GPU memory and parameter counts.

## Upstream sources

LEX-GNN:
https://github.com/wdhyun/LEX-GNN
Commit: 0642925539c414809c401066b2b06c8e24188dee

ConsisGAD:
https://github.com/Xtra-Computing/ConsisGAD
Commit: 36811c5bc79be49c9740f25a1f260496bb4736af

Raw datasets, environments, upstream clones, checkpoints, prediction arrays,
logs and compressed archives are intentionally excluded from Git.
