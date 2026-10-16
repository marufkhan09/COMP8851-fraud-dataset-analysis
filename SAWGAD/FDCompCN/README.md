# SAWGAD × FDCompCN

**Status:** 12 TR40 tuning trials and 12 final runs completed.

- Original model: `yj-zhou/SAWGAD` at commit `64d6d531a2f03549f0bfbe461092beedd6f70bd2`
- Source: FDCompCN `comp.dgl`, 5,317 nodes and 57 features
- Edge input: union of four source relations, followed by SAWGAD graph cleanup
- Frozen split seed: 2; training seeds: 2, 42, 72
- Ratios: TR40, TR30, TR20, TR10
- Hyperparameters and model checkpoints selected by validation AUPRC
- Classification threshold selected by validation macro-F1
- Training: 100 synthetic warmup epochs, up to 100 full epochs (early stopping patience 20)
- Model training uses labels in the frozen training partition

## Winner

- Trial: 6
- Validation AUPRC: 0.152642
- Configuration: `{"drop_rate": 0.25, "full_lr": 0.003, "hidden_dim": 64}`

## Final test metrics (mean ± population SD across three seeds)

| Ratio | AUPRC | AUROC | Macro-F1 | Fraud F1 |
|---|---:|---:|---:|---:|
| TR40 | 0.124781 ± 0.008445 | 0.542343 ± 0.005265 | 0.472208 ± 0.000000 | 0.000000 ± 0.000000 |
| TR30 | 0.120550 ± 0.004872 | 0.524385 ± 0.016890 | 0.472165 ± 0.000062 | 0.000000 ± 0.000000 |
| TR20 | 0.118592 ± 0.007656 | 0.539177 ± 0.013416 | 0.481991 ± 0.013834 | 0.041344 ± 0.058469 |
| TR10 | 0.117137 ± 0.007418 | 0.537937 ± 0.012065 | 0.481874 ± 0.013670 | 0.049894 ± 0.070561 |

## Files

- `results/`: all tuning, checkpoint-selection, final-run and summary JSON records
- `scripts/`: source code used for the dataset adapter and benchmark runner, with checksums
- `evidence/`: checkpoint hashes, dataset hashes, and compressed training logs

Original dataset and model checkpoints are preserved in the separately verified full backup.
