# SAWGAD × T-Finance

**Status:** Completed canonical benchmark candidate — 12 TR40 tuning trials and 12 final runs. Supervisor methodological approval remains separate.

- Model: `yj-zhou/SAWGAD` at pinned commit `64d6d531a2f03549f0bfbe461092beedd6f70bd2`
- Canonical source: original BWGNN authors’ `tfinance` DGL, 39,357 nodes, 10 features, 42,445,086 directed edges without self-loops
- Model graph view: pinned SAWGAD preprocessing removes/re-adds self-loops and simplifies edges (42,484,443 entries)
- Canonical source SHA-256: `b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700` (1,804 fraud labels)
- Original author ZIP SHA-256: `0a0091978425a0e9f1cd0e001240517ecd786344bd580c2bd00ef32ad1c80475`
- Frozen COMP8851 split SHA-256: `de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1`
- Split seed: 2; training seeds: 2, 42, 72; ratios: TR40, TR30, TR20, TR10
- Hyperparameters selected by TR40/seed-2 **validation AUPRC only**
- Final checkpoint selected by validation AUPRC; classification threshold selected by validation macro-F1
- Training: 100 synthetic warm-up epochs, up to 100 full-training epochs (early stopping patience 20)
- The source labels of node 27028 differ from the GADBench-derived file; this benchmark uses the canonical BWGNN label (fraud = 1)
- Original author weights, data and frozen split retained only in the separately verified complete backup

## Frozen winner

- Trial: **8**
- Validation AUPRC: **0.678160**
- Hyperparameters: `{"drop_rate": 0.1, "full_lr": 0.001, "hidden_dim": 128}`

## Test metrics (mean ± population SD across three seeds)

| Ratio | AUPRC | AUROC | Macro-F1 | Fraud F1 |
|---|---:|---:|---:|---:|
| TR40 | 0.639643 ± 0.092589 | 0.892766 ± 0.011586 | 0.721989 ± 0.165288 | 0.460391 ± 0.325590 |
| TR30 | 0.627107 ± 0.107264 | 0.891522 ± 0.015818 | 0.718526 ± 0.162855 | 0.454353 ± 0.321337 |
| TR20 | 0.700311 ± 0.003433 | 0.902611 ± 0.003322 | 0.832014 ± 0.005111 | 0.677530 ± 0.009347 |
| TR10 | 0.695709 ± 0.002650 | 0.902325 ± 0.001081 | 0.830118 ± 0.002804 | 0.674157 ± 0.005152 |

## Per-seed AUPRC (do not omit outliers)

| Ratio | Seed 2 | Seed 42 | Seed 72 |
|---|---:|---:|---:|
| TR40 | 0.703545 | 0.706668 | 0.508715 |
| TR30 | 0.703444 | 0.702463 | 0.475415 |
| TR20 | 0.702614 | 0.702861 | 0.695458 |
| TR10 | 0.699316 | 0.693023 | 0.694789 |

**Variability note:** Seed 72 is substantially weaker for TR40/TR30 than the other two seeds. The original seed results are retained; no result is excluded.

## Public evidence

- `results/`: all 12 tuning records, frozen selection JSON, one-time test claims, all 12 final-result JSONs, and complete summary
- `scripts/`: exact canonical adapter, runner, launcher, and script SHA-256 manifests
- `evidence/`: original log files compressed with gzip, exact checkpoint SHA-256 fingerprints, and dataset/split hashes
- `status.json`: source provenance, frozen winner, completed runs and protocol summary

### Provenance and limitations

Canonical BWGNN source and GADBench source have identical non-self-loop edge multiset and float32 features, but different one-hot/binary label representations, one label (node 27028), and self-loop storage. Tuning on the GADBench-derived source was preserved off-host as separate non-canonical evidence and is **not** mixed into this benchmark.

Final scores are descriptive experiment results; they do not establish supervisor approval or comparability to author-reproduction settings.
