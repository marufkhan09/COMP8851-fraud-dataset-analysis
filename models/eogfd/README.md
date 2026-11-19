# EOGFD

COMP8851 Assessment 3 - graph fraud detection benchmark.

Original repository: https://github.com/BERA-wx/EOGFD

Pinned commit: `a4726f173bc8c923109ddcb2786f4d9c5206ae7f`

Compatibility note: the original release has an import mismatch and uses test-set access during training. Dataset READMEs describe the benchmark adaptations.

Experimental protocol:
- Canonical Ass2 source and split verification before training.
- Split seed: 2.
- Training seeds: 2, 42, 72.
- Ratios: TR40, TR30, TR20, TR10.
- TR40 tuning: 12 trials, validation AUPRC only.
- Freeze winning settings for lower ratios.
- Seven metrics, raw seed evidence, timing and memory.
- No test access during tuning or checkpoint selection.
