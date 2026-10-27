# RGTAN × T-Social — Validation integrity and interpretation

## Recorded tuning measurements

The TR40 grid contains twelve configurations, with best validation AUPRC ranging from 0.446338 to 0.995064. Trials 02, 07, 08 and 10 recorded scores above 0.99, whereas the remaining configurations recorded lower best validation values. Trial 07 provides the highest observed value (0.995064). This is an observed tuning result, not a held-out test estimate.

## Architectural and optimisation sensitivity

The sampled two-layer RGTAN network was tuned over hidden dimension, learning rate and dropout. The learning-rate 0.005 configurations recorded validation AUPRC below 0.50 in this experiment. Several other configurations approached 0.99, while the same base learning rate did not consistently produce high AUPRC across dropout and hidden-dimension settings. These relationships motivate controlled sensitivity analysis, not a causal claim based on a single grid.

## Training dynamics and cost

The published epoch timeline includes training loss, validation AUPRC, best-so-far validation AUPRC, training and evaluation seconds, and early-stopping records for every tuning configuration. Strong epoch-to-epoch variation in selected trials should be evaluated through checkpoint replay, sampling determinism and metric consistency. Recorded runtime totals represent sums of per-epoch training and validation times; these are not equivalent to all wall-clock or setup overhead.

## Provenance and independent checks

The Stage 2 independent audit covered 16 saved preprocessing artifacts, five-fold training-only risk contexts and exact sparse neighbour computations. That audit supports preprocessing provenance. It does not independently validate the saved Stage 3 prediction arrays against the frozen validation labels.

Before accepting the numerical winner, recompute average precision from the stored validation scores and exactly ordered frozen validation labels; verify seed-index alignment; check masked target-node label contexts at inference; and compare saved-checkpoint replay with recorded predictions. These checks must be performed without using the held-out test for selection.

## Interpretation boundary

Validation-only tuning results describe relative configurations on TR40. Final training and once-only held-out evaluation are separate stages, and no final ratio-by-seed test AUROC, AUPRC or macro-F1 can be inferred from this export.
