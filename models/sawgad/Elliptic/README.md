# SAWGAD × Elliptic

**Status:** 12 TR40 tuning trials and 12 final runs completed.

- Model: `yj-zhou/SAWGAD` pinned at commit `64d6d531a2f03549f0bfbe461092beedd6f70bd2`
- Source: Elliptic transaction graph, 203,769 transactions, 165 model features, directed transaction edges
- Temporal evaluation: training graphs restricted to each ratio cutoff; validation through timestep 31; test starts at timestep 32
- Training cutoffs: TR40 through timestep 20; TR30 through timestep 12; TR20 through timestep 7; TR10 through timestep 3
- Training ratios: TR40, TR30, TR20, TR10; seeds: 2, 42, 72
- Model selection: TR40, seed 2, validation AUPRC only; one frozen winner for all 12 final runs
- Threshold selection: validation macro-F1; no test information used during tuning
- Final test evaluation: once per frozen checkpoint, supported by individual test-claim records
- Training: 100 synthetic warm-up epochs, up to 100 full epochs, patience 20

## Frozen winner

- Trial: 12
- Best validation AUPRC: 0.598211
- Hyperparameters: `{"drop_rate": 0.25, "full_lr": 0.003, "hidden_dim": 128}`

## Final test metrics (mean ± population SD; three seeds)

| Ratio | AUPRC | AUROC | Macro-F1 | Fraud F1 |
|---|---:|---:|---:|---:|
| TR40 | 0.408205 ± 0.024812 | 0.863423 ± 0.004023 | 0.707597 ± 0.014712 | 0.478752 ± 0.021882 |
| TR30 | 0.431004 ± 0.017182 | 0.872853 ± 0.002754 | 0.727939 ± 0.006919 | 0.501904 ± 0.012148 |
| TR20 | 0.380429 ± 0.052074 | 0.835913 ± 0.008157 | 0.643650 ± 0.009609 | 0.383530 ± 0.015489 |
| TR10 | 0.256421 ± 0.010523 | 0.769614 ± 0.011194 | 0.604143 ± 0.021050 | 0.319488 ± 0.020064 |

## Reproducibility files

- `results/`: tuning, frozen selection, test-claim and final metrics JSON files
- `scripts/`: dataset adapter, benchmark runner and launcher; SHA-256 checksums
- `evidence/`: source and checkpoint hashes, preflight/capacity reports and compressed training logs
- `status.json`: machine-readable run protocol and provenance

Original transaction CSVs, frozen split archive, and model checkpoints are intentionally excluded from GitHub.
They remain in the independently verified full backup.

### Evaluation and runtime notes

This dataset uses chronological train/validation/test partitions; it must not be treated as a random-split benchmark.
The capacity-check startup line inherited a stale `train_nodes=2126` display from FDCompCN;
the validated capacity JSON records the actual Elliptic TR40 training count of **18,889**.
Runtime differences against RGTAN on Amazon are not a controlled model-speed comparison.
