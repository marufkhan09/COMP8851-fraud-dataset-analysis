# EOGFD

COMP8851 Assessment 3 - graph fraud detection benchmark.

Original repository: https://github.com/BERA-wx/EOGFD

Pinned commit: `a4726f173bc8c923109ddcb2786f4d9c5206ae7f`

Compatibility note: the original release required an import fix and used test-set access during training; the submitted benchmark uses a documented validation-only adaptation.

Experimental protocol:
- Canonical Ass2 source and split verification before training.
- Split seed: 2.
- Training seeds: 2, 42, 72.
- Ratios: TR40, TR30, TR20, TR10.
- TR40 tuning: 12 trials, validation AUPRC only.
- Freeze winning settings for lower ratios.
- Seven metrics, raw seed evidence, timing and memory.
- No test access during tuning or checkpoint selection.
- Supervisor approval for model-specific deviations must be documented when obtained; none is currently recorded for these changes.

## Status

**Complete on all six datasets.** For each dataset: TR40 validation-only tuning 12/12, frozen final training 12/12 (TR40/TR30/TR20/TR10 x seeds 2, 42, 72) and once-only held-out test evaluation 12/12. Per-dataset results, evidence and caveats are in the README inside each dataset folder (Amazon, Elliptic, FDCompCN, T-Finance, T-Social, YelpChi).

Deviation: the scripts use the Assessment 2 frozen nested splits and validation-only checkpoint/threshold selection, whereas upstream `run_eogfd.py` uses its own split and scores the test set each epoch. These runs are therefore a unified-protocol adaptation, not an author-identical reproduction. On T-Social a process-local memory optimisation of the evolutionary step was also used (see `T-Social/completed_frozen_benchmark_20261023/`). Supervisor approval for these deviations is not documented in the repository.
