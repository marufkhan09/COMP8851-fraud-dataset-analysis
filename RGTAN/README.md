# RGTAN

COMP8851 Assessment 3 - graph fraud detection benchmark.

Original repository: https://github.com/AI4Risk/antifraud

Pinned commit: `65e583f03ac5c5983e49cc35179fe7a65908b9c1`

Compatibility issue: Risk-feature leakage and native five-fold training must be reconciled with the canonical split.

Experimental protocol:
- Canonical Ass2 source and split verification before training.
- Split seed: 2.
- Training seeds: 2, 42, 72.
- Ratios: TR40, TR30, TR20, TR10.
- TR40 tuning: 12 trials, validation AUPRC only.
- Freeze winning settings for lower ratios.
- Seven metrics, raw seed evidence, timing and memory.
- No test access during tuning or checkpoint selection.
- Model-specific deviations require Venus's approval.

No benchmark runs are claimed as completed.
