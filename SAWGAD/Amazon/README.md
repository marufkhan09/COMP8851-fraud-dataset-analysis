# SAWGAD × Amazon — completed benchmark candidate

**Status:** All 12 TR40 tuning trials and 12 final runs complete; supervisor methodological approval pending.

- Source: `yj-zhou/SAWGAD` @ `64d6d531a2f03549f0bfbe461092beedd6f70bd2`
- Frozen split seed: 2; train seeds: 2, 42, 72; ratios: TR40/TR30/TR20/TR10
- Winner selected using TR40 validation AUPRC, before final test evaluations
- Final checkpoint selected using validation AUPRC; threshold selected using validation macro-F1
- Each final run has a test-evaluation claim and final metric record
- 100-epoch native synthetic warm-up, followed by up to 100 full-training epochs with patience 20
- Uses all frozen training labels: **benchmark adaptation, not unchanged author weak-supervision protocol**
- Checkpoints and original `.mat`/`.npz` are preserved in the downloadable full backup, **not** on public GitHub

## Recorded test metrics (mean ± population SD; 3 seeds)

| Ratio | AUPRC | AUROC | Macro-F1 | Fraud F1 |
|---|---:|---:|---:|---:|
| TR40 | 0.817593 ± 0.023880 | 0.944745 ± 0.002395 | 0.859896 ± 0.030043 | 0.742813 ± 0.056281 |
| TR30 | 0.844150 ± 0.006492 | 0.948382 ± 0.003280 | 0.896513 ± 0.005282 | 0.811864 ± 0.009202 |
| TR20 | 0.797099 ± 0.017375 | 0.940997 ± 0.004591 | 0.843888 ± 0.031243 | 0.715273 ± 0.056851 |
| TR10 | 0.837408 ± 0.019726 | 0.948889 ± 0.010944 | 0.880673 ± 0.015642 | 0.781873 ± 0.028683 |

## Repository contents

- `scripts/sawgad_amazon_complete.py`: exact execution script (SHA256 audited)
- `results/`: 12 tuning records, 12 final records, checkpoint-selection records, test-evaluation claims and summary
- `evidence/checkpoint_sha256sums.txt`: hashes of 24 checkpoint files held in local backup
- `evidence/amazon_training.log.gz`: compressed full training log
- `status.json`: machine-readable status and methodological caveat

**Interpretation:** These are completed **benchmark-candidate** experiments, not supervisor-approved results. Do not represent the adaptation as identical to the published SAWGAD protocol.
