# SAWGAD × YelpChi

**Completed:** 12 TR40 tuning trials and 12 final runs.

- Original model: `yj-zhou/SAWGAD` at commit `64d6d531a2f03549f0bfbe461092beedd6f70bd2`
- Frozen split seed: 2; training seeds: 2, 42, 72
- Ratios: TR40, TR30, TR20, TR10
- Hyperparameters selected using TR40 validation AUPRC
- Checkpoint selection: validation AUPRC; threshold selection: validation macro-F1
- Synthetic warmup: 100 epochs; full training: maximum 100 epochs, patience 20
- Full training uses labels from the frozen training partition

## Winning configuration

- Trial: 9
- Validation AUPRC: 0.402039
- Configuration: `{"drop_rate": 0.1, "full_lr": 0.003, "hidden_dim": 128}`

## Test metrics (mean ± population SD, 3 seeds)

| Ratio | AUPRC | AUROC | Macro-F1 | Fraud F1 |
|---|---:|---:|---:|---:|
| TR40 | 0.403578 ± 0.002931 | 0.769499 ± 0.001290 | 0.649903 ± 0.000988 | 0.403305 ± 0.003141 |
| TR30 | 0.397335 ± 0.008987 | 0.764874 ± 0.005199 | 0.645707 ± 0.005020 | 0.405709 ± 0.007507 |
| TR20 | 0.388831 ± 0.009812 | 0.751189 ± 0.013659 | 0.639562 ± 0.006372 | 0.382623 ± 0.015772 |
| TR10 | 0.398989 ± 0.003092 | 0.763327 ± 0.002476 | 0.649112 ± 0.001327 | 0.401420 ± 0.009724 |

## Files

- `results/`: tuning records, final metrics, frozen checkpoints' metadata, and summary
- `scripts/`: execution source and checksum
- `evidence/`: adapter manifest, compressed logs and checkpoint SHA256 hashes

Model weights and original datasets are preserved in the separate full backup.
