# RGTAN × YelpChi — Completed

**Status: 12/12 TR40 validation-only batch-1024 tuning trials and 12/12 final ratio–seed runs completed; independent evidence audit passed.**

This is the lightweight *public research record* extracted from the SHA-256-verified private Mac archive. **It does not contain the original YelpChi.mat, frozen split NPZ, cross-fitted feature NPZs, model `.pt` weights, Python environments or the complete backup.** Those remain private in the Mac archive. The public prediction NPZ files include saved held-out test predictions and their labels, not raw graph or source data.

## Experiment protocol

- Model: RGTAN, pinned upstream [AI4Risk/antifraud commit](https://github.com/AI4Risk/antifraud/tree/65e583f03ac5c5983e49cc35179fe7a65908b9c1) `65e583f03ac5c5983e49cc35179fe7a65908b9c1`.
- Dataset: YelpChi. Original source SHA-256 `fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42`. Original frozen nested Assessment 2 split SHA-256 `0ea0af36dfc5a3a1f381ea2e3168377e45826498b6063190185c673e9ec8c22b`; split seed 2.
- Fixed validation/test partitions; nested training ratios **TR40, TR30, TR20, TR10**; three final seeds **2, 42, 72**.
- Graph: original YelpChi relationship-union graph, symmetric with self-loops; 45,954 nodes, 32 base features and 7,739,912 graph edges. Model has two graph layers, `[4,4]` heads and four risk-attention heads.
- Six cross-fitted risk-statistic contexts for each training fraction: five out-of-fold training contexts and one full-training context. Training labels are masked from their own training risk/label inputs. Validation/test labels were not used to build training risk features or normalize them.
- Validation-only **12-trial TR40 hyperparameter grid** uses batch **1024**. Frozen winner: **trial 4**, validation AUPRC **0.772519980**; `hidden_dim=32`, `drop=[0.3, 0.15]`, `base_lr=0.001`, max `100` epochs and patience `20`. Effective LR is defined in the published scripts.
- Final runs freeze these hyperparameters. Validation AUPRC chooses checkpoints; validation macro-F1 selects thresholds. Every held-out test partition was scored once per final run, after freezing validation choices.
- An earlier two-trial partial batch-32 tuning experiment was preserved in the **private backup only**. It was not combined with the completed batch-1024 grid.

## Held-out test performance

Three-seed mean ± **sample standard deviation** (`ddof=1`) for each metric. AUPRC is the primary fraud-ranking metric. These are held-out test results, not tuning-set results.

| Training ratio | Test AUPRC | Test AUROC | Test macro-F1 |
|---|---:|---:|---:|
| TR40 | 0.781747 ± 0.009146 | 0.934055 ± 0.005351 | 0.828723 ± 0.005194 |
| TR30 | 0.738798 ± 0.007032 | 0.917612 ± 0.004244 | 0.806083 ± 0.001860 |
| TR20 | 0.690425 ± 0.008863 | 0.899553 ± 0.003229 | 0.786421 ± 0.005982 |
| TR10 | 0.614686 ± 0.010843 | 0.872089 ± 0.006827 | 0.747457 ± 0.001479 |

### Individual held-out runs

| Ratio | Seed | Test AUPRC | Test AUROC | Test macro-F1 | Recorded test scorings |
|---|---:|---:|---:|---:|---:|
| TR40 | 2 | 0.775452 | 0.930749 | 0.824580 | 1 |
| TR40 | 42 | 0.792239 | 0.940228 | 0.834550 | 1 |
| TR40 | 72 | 0.777551 | 0.931188 | 0.827038 | 1 |
| TR30 | 2 | 0.731104 | 0.912719 | 0.804282 | 1 |
| TR30 | 42 | 0.740400 | 0.919833 | 0.805969 | 1 |
| TR30 | 72 | 0.744891 | 0.920286 | 0.807997 | 1 |
| TR20 | 2 | 0.680220 | 0.895840 | 0.779759 | 1 |
| TR20 | 42 | 0.694856 | 0.901118 | 0.791331 | 1 |
| TR20 | 72 | 0.696199 | 0.901701 | 0.788174 | 1 |
| TR10 | 2 | 0.603989 | 0.871258 | 0.748022 | 1 |
| TR10 | 42 | 0.625669 | 0.879294 | 0.748571 | 1 |
| TR10 | 72 | 0.614399 | 0.865716 | 0.745779 | 1 |

## Reproducibility and evidence

- `scripts/`: the original graph/cross-fit implementation, validation-only tuner and final-run implementation; source code is published, execution environment and weights are not.
- `risk_features/`: four audited training-only cross-fit manifest JSON files; actual cross-fit arrays remain in private backup.
- `results/tuning/`: twelve tuning JSON records and histories, frozen winner; checkpoint binaries excluded.
- `results/finals/`: twelve final JSON records, training histories, frozen validation selection, exactly-once scoring claims, saved test-prediction NPZ files and aggregate summary; weights excluded.
- `evidence/`: I3 independent integrity audit, environment inventory, logs, publication manifest and source/split provenance. Historical batch-32 log is reference evidence only, not part of selected grid.

### Private archive

`I3_COMP8851_YELPCHI_COMPLETE_RESEARCH_BACKUP_20261018T144631Z.tar.gz` — **389.22 MiB on Mac**; SHA-256 `0d767969c32c3200203bf1b1fd12d81921a2cbe91bc7e442e81817e092060106`. Independently verified on Mac before this public export. The archive is **not** in the GitHub repository. Exact rerunning or evaluating saved weights requires the private archive and a compatible environment.

**Protocol caveat:** Final test-scoring evidence and saved prediction metric recomputation are documented by the I3 audit; the publisher only verifies the saved records and hashes and performs **no model inference**.
