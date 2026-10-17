# RGTAN × Amazon — Completed

**Status: 12/12 TR40 validation-only tuning trials and 12/12 final runs completed and audited.**

Model source: https://github.com/AI4Risk/antifraud/tree/65e583f03ac5c5983e49cc35179fe7a65908b9c1

- Original model commit: `65e583f03ac5c5983e49cc35179fe7a65908b9c1`
- Canonical Amazon source SHA-256: `4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63`
- Frozen Assessment 2 split SHA-256: `0fd96816c440b247b8cac768896bc454b1ecf4abebaf5001c1a3e1d900efa4fd`
- TR40 12-trial validation-AUPRC grid winner: trial 1
- Winner parameters: `hidden_dim=32`, `drop=[0.2, 0.1]`, `base_lr=0.001`
- Four training ratios; seeds 2, 42, 72; maximum 100 epochs; patience 20; batch 32
- Five-fold training-only cross-fitted risk features; validation-only tuning
- Test scored once per final seed/ratio combination
- Offline validation of saved predictions and aggregate statistics completed

## Held-out test results

| Training ratio | Mean AUPRC | AUPRC sample SD | Mean AUROC |
|---|---:|---:|---:|
| TR40 | 0.907382 | 0.003919 | 0.977245 |
| TR30 | 0.886073 | 0.007535 | 0.971872 |
| TR20 | 0.876366 | 0.012392 | 0.959548 |
| TR10 | 0.846615 | 0.012518 | 0.943637 |

See `results/complete_summary.json` for aggregate metrics, `results/finals/` for each final record, history and stored predictions, `results/tuning/` for validation-only tuning, and `scripts/` for the exact experiment runner.

## Reproducibility and backup

The full, independently audited backup on I2 contains all 188 original files, including datasets, frozen split, cross-fitted risk-feature arrays, checkpoints, source and logs:

- Archive: `RGTAN_Amazon_I2_COMPLETE_BACKUP.tar.gz`
- Archive size: 53,801,795 bytes
- Archive SHA-256: `a19cd115b1ae2e51c97344f5ad9d94bc6016d28a1528bc82e9f2d37711802d37`
- `evidence/backup_manifest.json` lists every archived file and its SHA-256.
- **The off-host Mac copy has not yet been independently verified.**

Binary training checkpoints, raw benchmark data and large risk-feature arrays are deliberately excluded from GitHub. They remain in the offline archive. This repository includes runnable code, configuration, metadata, validation histories and stored test prediction artifacts.
