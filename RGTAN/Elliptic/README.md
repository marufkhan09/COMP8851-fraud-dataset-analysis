# RGTAN × Elliptic — Completed benchmark

**Status:** 12/12 validation-only TR40 tuning trials, 12/12 frozen final trainings, and 12/12 once-only held-out test scores audited from original I1 files. This directory contains the **public reproducibility record**, not a full GPU instance archive or private training checkpoint backup.

## Reproducibility and chronological isolation

- Model: [AI4Risk/antifraud pinned commit](https://github.com/AI4Risk/antifraud/tree/65e583f03ac5c5983e49cc35179fe7a65908b9c1) `65e583f03ac5c5983e49cc35179fe7a65908b9c1`; RGTAN two graph layers, **full-neighbour** sampling.
- Elliptic original source: 203,769 transactions; 165 model features; three original CSV files and the frozen Assessment 2 split are SHA-256 checked in `evidence/RGTAN_Elliptic_I1_Independent_Publication_Audit.json`.
- Nested TR40/TR30/TR20/TR10 training ratios, corresponding chronological training-prefix cutoff timesteps 20, 12, 7, 3. Validation timesteps 21–31, test 32–49. Graph training excludes future nodes and edges.
- Training-only five-fold crossfit for label/risk features at each ratio; validation/test labels excluded from feature generation. Risk normalizer fitted only on out-of-fold training examples. Four crossfit manifests (not original large arrays) are published.
- TR40 **12-trial validation-only tuning**; chosen trial **11** with validation AUPRC **0.760064**, hidden 64, drop `[0.3,0.15]`, base learning rate 0.003.
- Four training ratios × seeds **2, 42, 72**; final checkpoints selected by validation AUPRC, decision thresholds selected by validation macro-F1; held-out test accessed only after all 12 model selections froze.

## Held-out test performance

Mean ± **sample standard deviation** (`ddof=1`) across three seeds.

| Ratio | Test AUPRC | Test AUROC (mean) | Test macro-F1 (mean) | Fraud F1 (mean) |
|---|---:|---:|---:|---:|
| TR40 | 0.543047 ± 0.172785 | 0.861685 | 0.717674 | 0.498659 |
| TR30 | 0.331537 ± 0.177562 | 0.805732 | 0.608857 | 0.334258 |
| TR20 | 0.343989 ± 0.162923 | 0.790161 | 0.566378 | 0.313783 |
| TR10 | 0.184060 ± 0.034103 | 0.733305 | 0.513239 | 0.252454 |

## Evidence contents

- `scripts/` actual model-specific preprocessing, five-fold crossfitting, validation tuning, final-run and launch scripts from I1.
- `risk_features/` crossfit manifests containing SHA-256 references to the independently checked, non-public risk arrays.
- `results/tuning/` 12 validation-only tuning histories/records and frozen tuning winner.
- `results/finals/` all 12 seed/ratio histories, training freeze records, exactly-once scoring claims, final test metrics and saved per-test-node fraud scores (without label identities), plus aggregate summary.
- `evidence/` publication audit, relevant runtime logs, preflight diagnostics, Python package inventory and SHA-256 publication manifest.

**Scope:** Raw Elliptic CSVs, the private frozen split NPZ, crossfit arrays, checkpoints and the full Python/CUDA environment are intentionally excluded from this public GitHub export. Checkpoint hashes and provenance remain in the published run records. No model inference or held-out test recalculation was performed by the publication script.
