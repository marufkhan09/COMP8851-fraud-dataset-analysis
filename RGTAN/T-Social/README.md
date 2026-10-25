# RGTAN × T-Social — Frozen-split benchmark

Model source: https://github.com/AI4Risk/antifraud/tree/65e583f03ac5c5983e49cc35179fe7a65908b9c1

## Model, dataset and methodology

- Original RGTAN source commit: `65e583f03ac5c5983e49cc35179fe7a65908b9c1`.
- Canonical T-Social graph: 5,781,065 nodes, 146,211,016 stored directed edges and 10 base node features; source SHA-256 `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`.
- Frozen Assessment 2 split SHA-256: `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519`; split seed 2; fixed validation and test membership.
- Training ratios: TR40, TR30, TR20 and TR10. Model seeds under the unified protocol: 2, 42 and 72.
- Five-fold training-only out-of-fold label contexts and sparse one-hop/two-hop risk statistics, with training-only scaling. Independent Stage 2 checks covered 16 saved artifacts and 18,777,751,032 candidate sparse two-hop walks.
- T-Social sampling adaptation: two RGTAN graph layers with `MultiLayerNeighborSampler([10,10])`, batch size 1024, maximum 100 epochs, early stopping patience 20. This differs from the author's original full-neighbour method.
- TR40 tuning uses 12 configurations varying hidden dimensions 32/64, base learning rates 0.001/0.003/0.005 and dropout pairs [0.2,0.1] / [0.3,0.15]. Validation AUPRC selects the best checkpoint for each configuration.

## Held-out test results

Under the common Assessment 2 protocol, held-out AUPRC, AUROC, macro-F1, fraud-F1, precision, recall, accuracy, balanced accuracy, MCC and Brier score are computed from frozen final-run prediction scores, ground-truth labels and validation-selected classification thresholds. The uploaded I1 research archives contain validation-only TR40 tuning histories and checkpoint scores, rather than final-run held-out predictions. Thus they do not supply held-out numerical outcomes for RGTAN × T-Social. Validation results are reported separately below, not converted into test measurements.

### Primary fraud-detection metrics

AUPRC is the primary ranking metric for imbalanced fraud detection; AUROC measures ranking across false-positive rates. Macro-F1 averages fraud and legitimate-class F1, while fraud-F1 refers only to the positive class. These measures have different definitions and cannot be calculated from AUPRC alone.

### Additional metrics

Fraud precision and recall, overall accuracy, balanced accuracy and Matthews correlation coefficient depend on the prediction labels and the frozen decision threshold. Brier score additionally requires class-probability estimates paired with true labels. None can be reconstructed from aggregate tuning AUPRC values.

### Individual final-run evidence (held-out test set)

The unified final-run design comprises TR40, TR30, TR20 and TR10, each with model seeds 2, 42 and 72. Each ratio–seed result must be derived from that model's own frozen held-out predictions. The TR40 validation-only hyperparameter trials are not replacements for those twelve final runs.

### Recorded training cost

The available recorded timing covers TR40 validation-tuning configurations. It must not be presented as mean three-seed final-run training cost. Individual tuning runtimes are tabulated below; they exclude external setup overhead.

## Validation-only model selection

- Highest recorded TR40 validation AUPRC: 0.9950644969608748 (Trial 07; `hidden_dim=64`, `base_lr=0.001`, `drop=[0.2,0.1]`).
- Other high recorded validation AUPRC measurements: Trial 02 (0.9919115114291359), Trial 08 (0.9929484256358545) and Trial 10 (0.9941635055787835).
- These values describe hyperparameter selection on the full frozen validation set. The unusually high scores require independent prediction-order alignment, risk-context label-isolation and checkpoint-replay verification before scientific performance conclusions.

### TR40 tuning evidence — measured validation metrics

The following table uses original saved experiment histories and trial configuration records. Runtime is the sum of per-epoch training and validation seconds. It is not a test-set metric.

| Trial | Hidden dim | Learning rate | Dropout | Epochs recorded | Best epoch | Validation AUPRC | Training loss at best epoch | Recorded runtime (h) |
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

### Validation tuning and optimisation interpretation

The saved TR40 results span best validation AUPRC 0.446338–0.995064 across the grid. Four trials exceed 0.99 and several have substantially lower validation scores. This separation warrants careful validation of feature construction and score alignment. Learning rate 0.005 corresponds to markedly lower observed peak AUPRC in both hidden-dimension settings, while rapid epoch-to-epoch fluctuations show why saved checkpoint selection and replication matter. This is an association within these recordings, not a controlled causal finding.

## Reproducibility and evidence

- [TR40 trial-level metrics](tuning_evidence_20261025/results/tr40_tuning_trials.csv): full precision AUPRC, training loss, recorded epochs, train and validation time, and checkpoint fingerprints.
- [Epoch-level validation history](tuning_evidence_20261025/results/epoch_level_validation.csv): 402 saved training/validation epoch records.
- [Frozen evaluation design](tuning_evidence_20261025/results/ratio_seed_evaluation_protocol.csv): all 12 ratio/seed specifications; design rows are not measured evaluation outputs.
- [Risk-feature integrity audit](tuning_evidence_20261025/evidence/stage2_audit_summary.json): independent Stage 2 checks of sparse risk-context construction.
- [Validation-integrity review](tuning_evidence_20261025/evidence/validation_integrity_review.md): scoring and provenance questions requiring verification.
- [Per-configuration evidence](tuning_evidence_20261025/evidence/stage3_tuning): 12 source-linked records with observed validation results.
- [Experiment scripts](tuning_evidence_20261025/scripts): actual Stage 2 and Stage 3 source runners.
- [Public SHA-256 manifest](tuning_evidence_20261025/PUBLIC_SHA256_MANIFEST.json): cryptographic file fingerprints.

The published files exclude the canonical raw graph, frozen split binaries, crossfit feature arrays, model checkpoint binaries, complete environment and per-node private predictions. The verified Mac archives preserve the exported validation evidence and recovery state. Final held-out metrics require the corresponding final-run evidence and are not derivable from these saved tuning records.
