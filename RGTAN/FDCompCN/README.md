# RGTAN × FDCompCN — Completed

**Status: 12/12 validation-only TR40 tuning trials and 12/12 final runs completed and audited.**

This folder contains the public, reproducible experiment record extracted from the independently SHA-256-verified Mac backup. **Raw FDCompCN data, frozen split arrays, model checkpoints, cross-fitted feature arrays, complete private backup, and Python environment binaries are intentionally not published here.** The private Mac archive retains them.

## Dataset, model, and protocol

- Model: [RGTAN upstream source](https://github.com/AI4Risk/antifraud/tree/65e583f03ac5c5983e49cc35179fe7a65908b9c1), pinned commit `65e583f03ac5c5983e49cc35179fe7a65908b9c1`.
- FDCompCN source graph SHA-256: `e252b9a6b619b28a7bf6d9f5b16aacc232d43207d87ac26cd5160fb0d8a98baa`.
- Frozen Assessment 2 nested split SHA-256: `5cd7084342f4af48e9b82828dcac0e97879314cd5c9c010642b30c89e7c95168`.
- Fixed validation/test partitions, nested TR40 / TR30 / TR20 / TR10 training fractions, seeds `2, 42, 72`.
- TR40 validation-AUPRC tuning: **trial 12**, best validation AUPRC **0.176437479**.
- Frozen hyperparameters: `hidden_dim=64`, `drop=[0.3, 0.15]`, `base_lr=0.005`, batch size `256`; maximum `100` epochs, patience `20`.
- Input architecture: 57 continuous features, RGTAN risk-attention heads `3` to match feature dimensionality; source graph edge labels and original masks are not model inputs.
- Training-only five-fold out-of-fold risk/label contexts for each ratio; no held-out validation/test labels used to construct train-time risk features.
- Validation AUPRC selects checkpoints; validation macro-F1 selects thresholds. Held-out test was scored once per completed ratio–seed run, never used for selecting model, checkpoint, or threshold.

## Held-out test metrics

Three-seed **mean ± population SD** (`ddof=0`) shown below. The original saved `results/finals/complete_summary.json` also provides **sample SD** for AUPRC (`ddof=1`). AUPRC is the primary fraud-detection ranking metric.

| Training ratio | Test AUPRC | Test AUROC | Test macro-F1 |
|---|---:|---:|---:|
| TR40 | 0.139436 ± 0.018783 | 0.554583 ± 0.018501 | 0.515239 ± 0.017325 |
| TR30 | 0.137133 ± 0.010832 | 0.522375 ± 0.021323 | 0.515074 ± 0.009957 |
| TR20 | 0.118220 ± 0.003104 | 0.533278 ± 0.005999 | 0.490118 ± 0.001236 |
| TR10 | 0.113738 ± 0.007232 | 0.518195 ± 0.014119 | 0.496526 ± 0.006722 |

### Individual runs — held-out test

| Ratio | Seed | AUPRC | AUROC | Macro-F1 | Recorded test scoring count |
|---|---:|---:|---:|---:|---:|
| TR40 | 2 | 0.132342 | 0.551641 | 0.509873 | 1 |
| TR40 | 42 | 0.120815 | 0.533538 | 0.497219 | 1 |
| TR40 | 72 | 0.165152 | 0.578570 | 0.538626 | 1 |
| TR30 | 2 | 0.136157 | 0.519680 | 0.515670 | 1 |
| TR30 | 42 | 0.124381 | 0.497712 | 0.502591 | 1 |
| TR30 | 72 | 0.150860 | 0.549732 | 0.526960 | 1 |
| TR20 | 2 | 0.122496 | 0.540745 | 0.488780 | 1 |
| TR20 | 42 | 0.116939 | 0.533033 | 0.491762 | 1 |
| TR20 | 72 | 0.115224 | 0.526056 | 0.489811 | 1 |
| TR10 | 2 | 0.113567 | 0.522940 | 0.493950 | 1 |
| TR10 | 42 | 0.104968 | 0.499026 | 0.489889 | 1 |
| TR10 | 72 | 0.122680 | 0.532620 | 0.505738 | 1 |

## Reproducibility and evidence

- `scripts/` — four ratio-specific cross-fit adapters, TR40 validation-only tuner, and 12-final-run implementation.
- `risk_features/` — four ratio-specific cross-fit metadata manifests; original NPZ feature arrays reside only in the private backup.
- `results/tuning/` — tuning metadata, twelve trial records and validation histories, plus frozen winner; no model `.pt` checkpoints are committed.
- `results/finals/` — twelve final records, frozen validation selections, saved test predictions, aggregate summary and log evidence, excluding `.pt` checkpoints and temporary locks.
- `evidence/` — hashes of published artifacts, source/split/adapter fingerprints and backup provenance.

Full original experiment backup: `I2_COMP8851_COMPLETE_RESEARCH_BACKUP_20261017T205945Z.tar.gz`, **247 MB compressed**, SHA-256 `11ed569d840cffd1a1e764344e7f6fe31632778ea573633f6e478eba4ddfad83`; downloaded to Mac and verified there. **The archive is not stored on GitHub.** Restoring training or independently re-evaluating model weights requires that archive and the corresponding environment, not this lightweight public folder alone.
