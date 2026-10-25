# RGTAN × T-Social — Benchmark evidence

This public directory mirrors the reporting categories used by RGTAN × Amazon, RGTAN × FDCompCN and the other model–dataset folders. Measured validation-only tuning results are preserved separately from four-ratio held-out and per-seed reporting tables.

## Results

- `results/tr40_tuning_trials.csv`: twelve recorded TR40 configurations with full-precision validation AUPRC, hyperparameters, best epoch, training loss and timing.
- `results/epoch_level_validation.csv`: 402 recorded epoch events with training loss, validation AUPRC, runtime and early-stopping fields.
- `results/ratio_seed_evaluation_protocol.csv`: fixed four training ratios and seeds 2, 42 and 72.
- `results/heldout_test_summary.csv`: standard four-ratio AUPRC, AUROC, macro-F1 and fraud-F1 table columns.
- `results/heldout_test_additional_metrics.csv`: standard precision, recall, accuracy, balanced accuracy, MCC and Brier-score columns.
- `results/heldout_test_per_seed.csv`: standard twelve-row individual evaluation columns, including decision threshold.
- `results/final_run_cost_summary.csv`: standard training-duration and GPU-memory columns.

The held-out and final-run cost tables retain blank metric cells when no verified measurement exists in the archived evidence. Validation-only scores cannot be used to calculate held-out outcomes.

## Model and provenance

- Canonical T-Social graph: 5,781,065 nodes, 146,211,016 directed stored edges, 10 features.
- Split seed: 2; immutable split fingerprint recorded in the root README.
- Original RGTAN author commit: `65e583f03ac5c5983e49cc35179fe7a65908b9c1`.
- Two-layer bounded-neighbour sampler with fanouts [10,10], batch 1024.
- Five-fold training-only crossfit label/risk contexts with exact sparse neighbour features.
- `evidence/stage2_audit_summary.json`: independent risk-feature provenance and masking checks.
- `evidence/stage3_tuning/`: measured per-trial tuning records.
- `evidence/validation_integrity_review.md`: validation-score verification requirements.
- `scripts/stage2/` and `scripts/stage3/`: saved preprocessing, auditing and training source.

## Reproducibility and publication

The release includes lightweight source and evidence files. Binary graph data, full risk arrays, split arrays, checkpoints and node-level predictions are held privately. `PUBLIC_SHA256_MANIFEST.json` lists each published research-evidence file with SHA-256 and byte size.
