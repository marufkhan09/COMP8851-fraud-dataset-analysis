# RGTAN × T-Social — frozen-split benchmark

## Source, data and protocol

- Original model: [AI4Risk/antifraud](https://github.com/AI4Risk/antifraud/tree/65e583f03ac5c5983e49cc35179fe7a65908b9c1) at pinned commit `65e583f03ac5c5983e49cc35179fe7a65908b9c1`; the original RGTAN model source remains unchanged.
- Canonical T-Social graph: 5,781,065 nodes, 146,211,016 stored directed edges and 10 features.
- Canonical graph SHA-256: `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`. Frozen split SHA-256: `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519`.
- Frozen split seed 2; final training seeds 2, 42, 72; TR40/TR30/TR20/TR10 nested training subsets, fixed validation/test partitions.
- TR40 seed-2 tuning: 12 configurations, selected using validation AUPRC. Highest recorded validation AUPRC: 0.995064 (trial 7); configuration `hidden_dim=64`, `drop=[0.2,0.1]`, `base_lr=0.001`. Independent validation-score verification remains necessary before freezing the winner.
- RGTAN implementation uses five-fold training-only out-of-fold label contexts, sparse one-hop/two-hop risk statistics and training-only scaling. Independent Stage 2 checks covered 16 saved artifacts and 18,777,751,032 candidate sparse two-hop walks.
- Training adaptation: two graph layers, `MultiLayerNeighborSampler([10,10])`, batch 1024; maximum 100 epochs, patience 20. This differs from the original full-neighbour workflow.
- Assessment protocol: validation AUPRC selects checkpoints; validation macro-F1 is used to select classification thresholds before once-only held-out test evaluation. Held-out test scores are not used for tuning or checkpoint selection.
- Scientific limitation: bounded-neighbour sampling is a computational adaptation rather than an author-identical RGTAN reproduction.

## Held-out test results

Three-seed mean ± sample standard deviation (`ddof=1`):

| Training ratio | Test AUPRC | Test AUROC | Test macro-F1 |
|---|---:|---:|---:|
| TR40 | | | |
| TR30 | | | |
| TR20 | | | |
| TR10 | | | |

## Individual final held-out results

| Ratio | Seed | Test AUPRC | Test AUROC | Test macro-F1 |
|---|---:|---:|---:|---:|
| TR40 | 2 | | | |
| TR40 | 42 | | | |
| TR40 | 72 | | | |
| TR30 | 2 | | | |
| TR30 | 42 | | | |
| TR30 | 72 | | | |
| TR20 | 2 | | | |
| TR20 | 42 | | | |
| TR20 | 72 | | | |
| TR10 | 2 | | | |
| TR10 | 42 | | | |
| TR10 | 72 | | | |

## Audit and reproducibility

- `tuning_evidence_20261025/results/tr40_tuning_trials.csv`: 12 TR40 tuning configurations with saved validation AUPRC, training loss, runtime and selected epochs.
- `tuning_evidence_20261025/results/epoch_level_validation.csv`: 402 saved epoch-level training and validation measurements.
- `tuning_evidence_20261025/results/ratio_seed_evaluation_protocol.csv`: four-ratio, three-seed final evaluation design.
- `tuning_evidence_20261025/scripts/`: Stage 2 preprocessing and Stage 3 training scripts from the recorded experiment.
- `tuning_evidence_20261025/evidence/`: independent risk-feature audit, validation review and per-configuration tuning records.
- `tuning_evidence_20261025/PUBLIC_SHA256_MANIFEST.json`: fingerprints of the public evidence files.
- Excluded from Git: raw graph, frozen split binaries, risk-feature matrices, model checkpoints, private predictions and Python/CUDA environments. Private Mac backups preserve exported tuning evidence and checkpoint recovery artifacts.
- Held-out test results for the four ratios require matching final-model prediction scores, labels and validation-selected thresholds; the archived TR40 tuning records do not contain those measurements.
