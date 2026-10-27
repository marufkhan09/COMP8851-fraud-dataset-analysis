# RGTAN × T-Social — Research evidence

This directory archives the actual TR40 validation-tuning measurements, experimental scripts and the Stage 2 feature-provenance audit. The 12-configuration records document checkpoint selection by validation AUPRC, training loss, epoch histories and runtime. They do not constitute four-ratio held-out test evaluations.

## Results

- `results/tr40_tuning_trials.csv`: measured best validation AUPRC, hyperparameters, selected epoch, training loss and timing for each tuning configuration.
- `results/epoch_level_validation.csv`: 402 saved epoch-level loss/AUPRC and runtime measurements.
- `results/ratio_seed_evaluation_protocol.csv`: frozen ratios TR40/TR30/TR20/TR10 and seeds 2, 42 and 72, as evaluation design only.

## Audit and source

- `evidence/stage2_audit_summary.json`: independent sparse risk-feature audit.
- `evidence/stage3_tuning/`: per-configuration validation measurements and checkpoint hashes.
- `evidence/validation_integrity_review.md`: score-integrity checks and methodological interpretation.
- `scripts/stage2/` and `scripts/stage3/`: original captured source scripts for preprocessing and validation-only tuning.
- `PUBLIC_SHA256_MANIFEST.json`: hashes and byte counts of published research files.

## Metric provenance

Validation AUPRC is a threshold-independent ranking metric and is not interchangeable with held-out test AUPRC. AUROC requires aligned labels and scores. F1, fraud-class precision and recall, accuracy, balanced accuracy and MCC additionally depend on frozen decision thresholds. Brier score requires true labels paired with probabilistic outputs. The published archive provides no final held-out ratio-by-seed label-score records, so no test results are imputed.

Raw graph data, the frozen split, private risk arrays, checkpoints and per-node predictions remain outside this public directory.
