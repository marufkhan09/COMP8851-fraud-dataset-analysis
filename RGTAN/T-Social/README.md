# RGTAN × T-Social — Frozen-split assessment benchmark

RGTAN was evaluated on the canonical T-Social graph using the frozen Assessment 2 split and a 12-configuration TR40 validation-only tuning grid.

The highest observed tuning AUPRC is 0.995064 from Trial 7. Several configurations show similarly high validation scores, requiring independent scoring and feature-integrity verification before final winner selection.

The ratio-specific three-seed final training matrix and held-out evaluation are separate stages. Tuning outcomes must not be interpreted as final test results.

The original RGTAN model source is pinned and unchanged. The T-Social training pipeline uses bounded-neighbour sampling for computational feasibility and is not an author-identical full-neighbour reproduction.

## Provenance

- Original model: `AI4Risk/antifraud`, commit `65e583f03ac5c5983e49cc35179fe7a65908b9c1`.
- Canonical T-Social dataset: 5,781,065 nodes, 146,211,016 stored directed edges and 10 node features.
- Dataset SHA-256: `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`.
- Frozen split SHA-256: `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519`.
- Split seed: 2; training ratios: TR40, TR30, TR20 and TR10; final training seeds: 2, 42 and 72.
- Training-only risk features were generated using five-fold out-of-fold label contexts and exact sparse one-hop/two-hop neighbourhood statistics.
- Stage 2 independent verification checked 16 saved artifacts, label masking, scaling and sampled exact two-hop calculations. The processing recorded 18,777,751,032 candidate two-hop walks.
- RGTAN training adapter: two graph layers, `MultiLayerNeighborSampler([10,10])`, batch size 1024, maximum 100 epochs and early-stopping patience 20.
- The 12-configuration tuning grid varies hidden dimension, learning rate and dropout. Checkpoints are selected using full-validation AUPRC without held-out test model selection.

## Validation-only TR40 tuning results

The table presents measured best validation AUPRC, corresponding hyperparameters and observed epoch counts. These values are not held-out test metrics.

| Trial | Hidden dim | Learning rate | Dropout | Recorded epochs | Best epoch | Validation AUPRC | Loss at best epoch | Recorded runtime (hours) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 01 | 32 | 0.001 | 0.2/0.10 | 43 | 23 | 0.584989 | 0.076348 | 4.99 |
| 02 | 32 | 0.003 | 0.2/0.10 | 22 | 2 | 0.991912 | 0.012406 | 2.55 |
| 03 | 32 | 0.005 | 0.2/0.10 | 26 | 6 | 0.477618 | 0.088160 | 2.97 |
| 04 | 32 | 0.001 | 0.3/0.15 | 39 | 19 | 0.576455 | 0.077145 | 4.44 |
| 05 | 32 | 0.003 | 0.3/0.15 | 37 | 17 | 0.519692 | 0.083761 | 4.26 |
| 06 | 32 | 0.005 | 0.3/0.15 | 31 | 11 | 0.446338 | 0.093600 | 3.57 |
| 07 | 64 | 0.001 | 0.2/0.10 | 36 | 16 | 0.995064 | 0.008944 | 4.23 |
| 08 | 64 | 0.003 | 0.2/0.10 | 33 | 13 | 0.992948 | 0.011529 | 3.81 |
| 09 | 64 | 0.005 | 0.2/0.10 | 32 | 12 | 0.499670 | 0.087329 | 3.80 |
| 10 | 64 | 0.001 | 0.3/0.15 | 28 | 8 | 0.994164 | 0.009982 | 3.26 |
| 11 | 64 | 0.003 | 0.3/0.15 | 28 | 8 | 0.514858 | 0.086862 | 3.28 |
| 12 | 64 | 0.005 | 0.3/0.15 | 47 | 29 | 0.484938 | 0.089085 | 5.59 |

The validation results exhibit a substantial separation between configurations. Trials 2, 7, 8 and 10 achieved scores above 0.99, whereas the remaining configurations achieved best recorded scores between 0.446338 and 0.584989. Runtime is the sum of recorded training and validation seconds across observed epochs; setup and external overhead are excluded.

These exceptional results should not be attributed solely to architecture or hyperparameters without independent verification. Saved-score alignment, label isolation, validation metric computation and checkpoint reproducibility require further examination.

## Results across three final seeds

The frozen evaluation protocol specifies three training seeds for each of four training ratios. These final-run measurements are distinct from the TR40 tuning table.

| Ratio | Seeds | Held-out AUPRC | Held-out AUROC | Held-out macro-F1 |
|---|---|---|---|---|
| TR40 | 2, 42, 72 | Not evaluated | Not evaluated | Not evaluated |
| TR30 | 2, 42, 72 | Not evaluated | Not evaluated | Not evaluated |
| TR20 | 2, 42, 72 | Not evaluated | Not evaluated | Not evaluated |
| TR10 | 2, 42, 72 | Not evaluated | Not evaluated | Not evaluated |

## Individual held-out evaluation results

The 12 final ratio/seed combinations use the same frozen split protocol. Once-only held-out measurements and validation-selected thresholds are reported separately from tuning outcomes when independently verified. Validation AUPRC does not supply held-out test AUPRC, AUROC or macro-F1.

## Public evidence and scripts

- `tuning_evidence_20261025/results/tr40_tuning_trials.csv`: all 12 configurations and their exact measured validation values, selected epochs, loss and runtime.
- `tuning_evidence_20261025/results/epoch_level_validation.csv`: epoch-level training loss, validation AUPRC, early stopping and runtime measurements.
- `tuning_evidence_20261025/results/ratio_seed_evaluation_protocol.csv`: frozen ratio and seed specification.
- `tuning_evidence_20261025/evidence/stage2_audit_summary.json`: independent preprocessing and crossfit verification.
- `tuning_evidence_20261025/evidence/validation_integrity_review.md`: exceptional validation-score analysis and methodological limitations.
- `tuning_evidence_20261025/evidence/stage3_tuning/`: public per-configuration validation records and provenance fingerprints.
- `tuning_evidence_20261025/scripts/`: pinned experiment-support scripts and the recorded bounded-neighbour adaptation.
- `tuning_evidence_20261025/PUBLIC_SHA256_MANIFEST.json`: cryptographic fingerprints of publicly released evidence files.

Original source data, frozen split binaries, model checkpoints, risk-feature matrices and node-level predictions are retained outside the public repository.

The evidence supports interpretation of validation tuning, training behaviour and implementation challenges. Broader comparative conclusions require corresponding verified observations from the other model–dataset combinations.
