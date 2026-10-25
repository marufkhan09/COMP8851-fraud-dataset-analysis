# RGTAN × T-Social — Frozen-split benchmark

Model source: https://github.com/AI4Risk/antifraud/tree/65e583f03ac5c5983e49cc35179fe7a65908b9c1

- Original model commit: `65e583f03ac5c5983e49cc35179fe7a65908b9c1`.
- Canonical T-Social graph: 5,781,065 nodes, 146,211,016 stored directed edges, 10 original node features; source SHA-256 `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`.
- Frozen Assessment 2 split SHA-256: `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519`; split seed 2.
- Four training ratios: TR40, TR30, TR20, TR10. Three final training seeds: 2, 42, 72; frozen validation and test membership.
- Training-only, five-fold cross-fitted label/risk features with exact sparse one-hop/two-hop neighbourhood statistics and training-only feature scaling. Independent Stage 2 checks covered 16 saved artifacts and 18,777,751,032 candidate two-hop walks.
- RGTAN two-layer bounded-neighbour adapter: `MultiLayerNeighborSampler([10,10])`, batch 1024, maximum 100 epochs and patience 20. This is an adapted sampling protocol, not the original author's full-neighbour training procedure.
- TR40 12-configuration validation AUPRC tuning grid; highest recorded result is trial 7, validation AUPRC 0.995064, `hidden_dim=64`, `drop=[0.2,0.1]`, `base_lr=0.001`.
- Checkpoint selection uses validation AUPRC. Independent validation-score alignment and label-isolation verification are required before declaring a frozen winner. Test scores are not used for selecting configurations or thresholds.

## Held-out test results

The table schema follows the RGTAN × Amazon benchmark: four training ratios, three final seeds per ratio and the same metric definitions. Only verified held-out measurements belong in these cells. The available T-Social archive provides validation-tuning measurements, not held-out label-score pairs; unmeasured cells are left blank.

### Primary fraud-detection metrics (mean ± population SD; three seeds)

| Ratio | AUPRC | AUROC | Macro-F1 | Fraud F1 |
|---|---:|---:|---:|---:|
| TR40 |  |  |  |  |
| TR30 |  |  |  |  |
| TR20 |  |  |  |  |
| TR10 |  |  |  |  |

Macro-F1 is the average of F1 for the two classes. Fraud F1 is F1 for the fraud class. AUROC measures ranking across false-positive rates; AUPRC is the primary fraud-ranking measure. Neither AUPRC nor AUROC identifies a decision threshold by itself.

### Additional metrics (mean ± population SD; three seeds)

| Ratio | Fraud precision | Fraud recall | Accuracy | Balanced accuracy | MCC | Brier score |
|---|---:|---:|---:|---:|---:|---:|
| TR40 |  |  |  |  |  |  |
| TR30 |  |  |  |  |  |  |
| TR20 |  |  |  |  |  |  |
| TR10 |  |  |  |  |  |  |

Precision, recall and Fraud F1 concern the fraud class. Balanced accuracy averages recall across classes. MCC incorporates all four confusion-matrix outcomes. Lower Brier score indicates better probabilistic calibration. Calculating these metrics requires ground-truth labels and appropriately ordered prediction scores for each evaluated run.

### Individual final-run evidence (held-out test set)

| Ratio | Seed | AUPRC | AUROC | Macro-F1 | Fraud F1 | Decision threshold |
|---|---:|---:|---:|---:|---:|---:|
| TR40 | 2 |  |  |  |  |  |
| TR40 | 42 |  |  |  |  |  |
| TR40 | 72 |  |  |  |  |  |
| TR30 | 2 |  |  |  |  |  |
| TR30 | 42 |  |  |  |  |  |
| TR30 | 72 |  |  |  |  |  |
| TR20 | 2 |  |  |  |  |  |
| TR20 | 42 |  |  |  |  |  |
| TR20 | 72 |  |  |  |  |  |
| TR10 | 2 |  |  |  |  |  |
| TR10 | 42 |  |  |  |  |  |
| TR10 | 72 |  |  |  |  |  |

Each row corresponds to one of the twelve ratio-by-seed combinations defined by the frozen Assessment 2 protocol. A validation-selected threshold and held-out node-level scores would be needed to populate each measured test row.

### Recorded training cost (three seeds per ratio)

| Ratio | Mean training time (hours) | Mean peak allocated GPU memory (GiB) |
|---|---:|---:|
| TR40 |  |  |
| TR30 |  |  |
| TR20 |  |  |
| TR10 |  |  |

This table uses the same final-run cost convention as RGTAN × Amazon. TR40 tuning runtime cannot substitute for a three-seed final-run mean; observed tuning time appears in the separate tuning table below.

### Validation-only model selection

- TR40 tuning: twelve recorded configurations, using best validation AUPRC to select checkpoints.
- Highest observed trial: 7; validation AUPRC 0.9950644969608748.
- Recorded configuration: `hidden_dim=64`, `drop=[0.2,0.1]`, `base_lr=0.001`.
- Tuning design: hidden dimensions 32 and 64; learning rates 0.001, 0.003 and 0.005; paired dropout settings 0.2/0.1 and 0.3/0.15.
- Training adaptation: bounded two-hop neighbour sampling, batch 1024, maximum 100 epochs, patience 20.
- Selection integrity: exceptional validation scores require independent label ordering, risk-context and checkpoint-replay verification. The tuning maximum is not a held-out test result.

### TR40 tuning evidence

The following metrics were recorded during validation-only configuration search. AUPRC, loss and time refer to these training and validation runs, not the held-out test matrix.

| Trial | Hidden dim | Base learning rate | Dropout | Recorded epochs | Best epoch | Validation AUPRC | Loss at best epoch | Recorded runtime (hours) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 01 | 32 | 0.001 | 0.2/0.1 | 43 | 23 | 0.584989 | 0.076348 | 4.99 |
| 02 | 32 | 0.003 | 0.2/0.1 | 22 | 2 | 0.991912 | 0.012406 | 2.55 |
| 03 | 32 | 0.005 | 0.2/0.1 | 26 | 6 | 0.477618 | 0.088160 | 2.97 |
| 04 | 32 | 0.001 | 0.3/0.15 | 39 | 19 | 0.576455 | 0.077145 | 4.44 |
| 05 | 32 | 0.003 | 0.3/0.15 | 37 | 17 | 0.519692 | 0.083761 | 4.26 |
| 06 | 32 | 0.005 | 0.3/0.15 | 31 | 11 | 0.446338 | 0.093600 | 3.57 |
| 07 | 64 | 0.001 | 0.2/0.1 | 36 | 16 | 0.995064 | 0.008944 | 4.23 |
| 08 | 64 | 0.003 | 0.2/0.1 | 33 | 13 | 0.992948 | 0.011529 | 3.81 |
| 09 | 64 | 0.005 | 0.2/0.1 | 32 | 12 | 0.499670 | 0.087329 | 3.80 |
| 10 | 64 | 0.001 | 0.3/0.15 | 28 | 8 | 0.994164 | 0.009982 | 3.26 |
| 11 | 64 | 0.003 | 0.3/0.15 | 28 | 8 | 0.514858 | 0.086862 | 3.28 |
| 12 | 64 | 0.005 | 0.3/0.15 | 47 | 29 | 0.484938 | 0.089085 | 5.59 |

Recorded runtime is the sum of saved per-epoch training and validation seconds, excluding initialization and other external overhead. Four configurations produced validation AUPRC above 0.99; verifying those results independently is essential before scientific performance claims.

Complete lightweight records: [TR40 tuning CSV](tuning_evidence_20261025/results/tr40_tuning_trials.csv), [epoch-level validation CSV](tuning_evidence_20261025/results/epoch_level_validation.csv), [held-out aggregate table schema](tuning_evidence_20261025/results/heldout_test_summary.csv), [individual ratio–seed table schema](tuning_evidence_20261025/results/heldout_test_per_seed.csv), [additional-metric table schema](tuning_evidence_20261025/results/heldout_test_additional_metrics.csv) and [published scripts and audits](tuning_evidence_20261025/README.md).

## Reproducibility and backup

- `tuning_evidence_20261025/scripts/stage2/`: actual sparse crossfit-risk construction and independent Stage 2 audit code.
- `tuning_evidence_20261025/scripts/stage3/`: exact recorded TR40 training and recovery runners, including the computational adaptation.
- `tuning_evidence_20261025/evidence/stage3_tuning/`: per-configuration hyperparameters, validation results, loss/timing metadata and model/source fingerprints.
- `tuning_evidence_20261025/evidence/stage2_audit_summary.json`: checks on training-only risk feature provenance and label isolation.
- `tuning_evidence_20261025/evidence/validation_integrity_review.md`: conditions for independent validation-score confirmation.
- `tuning_evidence_20261025/PUBLIC_SHA256_MANIFEST.json`: hashes for every research-evidence file in the public release.
- Private Mac archives preserve recorded tuning evidence, source code, model checkpoint state, log files, training label contexts and the frozen split; raw graph and full derived risk arrays are not published in this repository.

The public folder provides the common benchmark reporting structure without substituting validation measurements for unseen held-out outcomes. Source data, private split arrays, private node predictions, full model checkpoints and the GPU environment are excluded from this publication.
