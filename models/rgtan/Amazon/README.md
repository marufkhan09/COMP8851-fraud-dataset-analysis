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

All numbers are **test-set** metrics from the 12 stored final-run records, not additional test evaluations. Each mean ± standard deviation uses the three seeds (2, 42, 72) and the **population SD (ddof=0)** convention used by the completed SAWGAD dataset READMEs. The original **sample SD (ddof=1)** for AUPRC remains in `results/complete_summary.json`.

### Primary fraud-detection metrics (mean ± population SD; three seeds)

| Ratio | AUPRC | AUROC | Macro-F1 | Fraud F1 |
|---|---:|---:|---:|---:|
| TR40 | 0.907382 ± 0.003200 | 0.977245 ± 0.001734 | 0.917183 ± 0.003255 | 0.848729 ± 0.005921 |
| TR30 | 0.886073 ± 0.006153 | 0.971872 ± 0.002156 | 0.914990 ± 0.002375 | 0.844595 ± 0.004498 |
| TR20 | 0.876366 ± 0.010118 | 0.959548 ± 0.004893 | 0.914322 ± 0.002004 | 0.843306 ± 0.003609 |
| TR10 | 0.846615 ± 0.010221 | 0.943637 ± 0.016949 | 0.908110 ± 0.001240 | 0.832344 ± 0.002113 |

**Macro-F1** averages F1 across both classes; **Fraud F1** is the positive (fraud) class F1. AUPRC is the primary ranking metric; AUROC complements it.

### Additional metrics (mean ± population SD; three seeds)

| Ratio | Fraud precision | Fraud recall | Accuracy | Balanced accuracy | MCC | Brier score |
|---|---:|---:|---:|---:|---:|---:|
| TR40 | 0.937562 ± 0.012883 | 0.775407 ± 0.008002 | 0.973765 ± 0.001091 | 0.884986 ± 0.003839 | 0.839065 ± 0.006799 | 0.022388 ± 0.001722 |
| TR30 | 0.942623 ± 0.008036 | 0.765244 ± 0.012447 | 0.973283 ± 0.000492 | 0.880171 ± 0.005832 | 0.835628 ± 0.003522 | 0.022740 ± 0.000357 |
| TR20 | 0.946917 ± 0.008819 | 0.760163 ± 0.001437 | 0.973187 ± 0.000722 | 0.877843 ± 0.000818 | 0.834829 ± 0.004517 | 0.023631 ± 0.000882 |
| TR10 | 0.907132 ± 0.016951 | 0.769309 ± 0.010364 | 0.970583 ± 0.000722 | 0.880498 ± 0.004310 | 0.819795 ± 0.003417 | 0.030428 ± 0.006388 |

Precision and recall refer to the fraud class. Higher is better for these metrics and MCC; **lower Brier score is better**.

### Individual final-run evidence (held-out test set)

| Ratio | Seed | AUPRC | AUROC | Macro-F1 | Fraud F1 | Decision threshold |
|---|---:|---:|---:|---:|---:|---:|
| TR40 | 2 | 0.903755 | 0.976042 | 0.912659 | 0.840532 | 0.888017 |
| TR40 | 42 | 0.911539 | 0.979698 | 0.920177 | 0.854305 | 0.757417 |
| TR40 | 72 | 0.906853 | 0.975997 | 0.918714 | 0.851351 | 0.686879 |
| TR30 | 2 | 0.878593 | 0.969069 | 0.912366 | 0.839590 | 0.857474 |
| TR30 | 42 | 0.885962 | 0.972233 | 0.918118 | 0.850498 | 0.644625 |
| TR30 | 72 | 0.893663 | 0.974312 | 0.914488 | 0.843697 | 0.888681 |
| TR20 | 2 | 0.888644 | 0.966184 | 0.916354 | 0.846939 | 0.875346 |
| TR20 | 42 | 0.876590 | 0.954530 | 0.915019 | 0.844595 | 0.751164 |
| TR20 | 72 | 0.863863 | 0.957931 | 0.911595 | 0.838384 | 0.535126 |
| TR10 | 2 | 0.833357 | 0.920999 | 0.907693 | 0.831386 | 0.440628 |
| TR10 | 42 | 0.858229 | 0.961778 | 0.906843 | 0.830372 | 0.630831 |
| TR10 | 72 | 0.848259 | 0.948136 | 0.909794 | 0.835275 | 0.601066 |

Decision thresholds were selected by **validation macro-F1**, not by inspecting test labels or outcomes. For each seed and ratio, `results/finals/<ratio>/seed_<seed>/final.json` also records TP, TN, FP, FN, best epoch, training counts, and checkpoint hashes.

### Recorded training cost (three seeds per ratio)

| Ratio | Mean training time (hours) | Mean peak allocated GPU memory (GiB) |
|---|---:|---:|
| TR40 | 1.659 | 1.246 |
| TR30 | 1.015 | 1.247 |
| TR20 | 0.641 | 1.247 |
| TR10 | 0.534 | 1.250 |

These are averages of recorded per-run wall-clock training time and peak allocated GPU memory, not fresh runtime measurements.

### Validation-only model selection

- TR40 tuning winner: **trial 1**, validation AUPRC **0.922785**.
- Frozen configuration: `hidden_dim=32`, `drop=[0.2, 0.1]`, `base_lr=0.001`.
- Seeds: 2, 42, 72; TR40/TR30/TR20/TR10; maximum 100 epochs; patience 20; batch size 32.
- Five-fold, training-only out-of-fold risk feature construction; no validation or test labels used in risk-feature fitting.
- Best checkpoints selected with validation AUPRC; thresholds selected with validation macro-F1.
- The held-out test partition was scored once per completed run and was not used for model or threshold selection.

Complete records: `results/complete_summary.json`, `results/finals/`, `results/tuning/`, `risk_features/`, and `scripts/`.

## Reproducibility and backup

The full, independently audited backup on I2 contains all 188 original files, including datasets, frozen split, cross-fitted risk-feature arrays, checkpoints, source and logs:

- Archive: `RGTAN_Amazon_I2_COMPLETE_BACKUP.tar.gz`
- Archive size: 53,801,795 bytes
- Archive SHA-256: `a19cd115b1ae2e51c97344f5ad9d94bc6016d28a1528bc82e9f2d37711802d37`
- `evidence/backup_manifest.json` lists every archived file and its SHA-256.
- **The off-host Mac copy has not yet been independently verified.**

Binary training checkpoints, raw benchmark data and large risk-feature arrays are deliberately excluded from GitHub. They remain in the offline archive. This repository includes runnable code, configuration, metadata, validation histories and stored test prediction artifacts.
