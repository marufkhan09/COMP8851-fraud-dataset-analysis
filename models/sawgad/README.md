# SAWGAD

COMP8851 Assessment 3 - graph fraud detection benchmark.

Original repository: https://github.com/yj-zhou/SAWGAD

Pinned commit: `64d6d531a2f03549f0bfbe461092beedd6f70bd2`

Compatibility issue: Weak supervision and native two-stage 100+350 epoch schedule require methodological approval.

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
