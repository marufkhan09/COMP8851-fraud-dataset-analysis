# RGTAN × T-Social — Published research evidence

This folder preserves measured TR40 validation-tuning values, corresponding epoch-level trajectories, computational timings, source scripts and Stage 2 audit provenance. Scores in the tuning CSV are validation AUPRC, not held-out test metrics.

## Results

- `results/tr40_tuning_trials.csv`: twelve configurations, precise AUPRC, selected epoch, optimisation settings, training loss at selection and measured runtime.
- `results/epoch_level_validation.csv`: the full set of recorded epoch-by-epoch training losses, validation AUPRC, runtime and early stopping counters.
- `results/ratio_seed_evaluation_protocol.csv`: the planned four training ratios and three random seeds, distinct from validation tuning.

## Audit and scripts

- `evidence/stage2_audit_summary.json`: independent preprocessing and risk-feature provenance checks.
- `evidence/validation_integrity_review.md`: interpretation limits and the required scoring-consistency audit.
- `evidence/stage3_tuning/trial_XX_validation_metrics.json`: sanitized measured metrics and checkpoint/source fingerprints for each configuration.
- `scripts/stage2/`: training-only sparse risk-feature construction and independent Stage 2 audit.
- `scripts/stage3/`: tuning runner and its bounded-neighbour adaptation.
- `PUBLIC_SHA256_MANIFEST.json`: contents and SHA-256 digests of every released file in this folder.

## Reproducibility scope

The original RGTAN model source is pinned at `65e583f03ac5c5983e49cc35179fe7a65908b9c1`; published scripts describe the T-Social computational adaptation. Binary graph data, frozen split arrays, node-level predictions, risk matrices and model/optimizer checkpoints are kept privately. Stage 3 scoring alignment requires independent verification before methodological conclusions about exceptional scores.
