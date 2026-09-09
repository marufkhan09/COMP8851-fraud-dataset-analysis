# BWGNN on T-Finance

This folder contains the verified COMP8851 BWGNN experiment on T-Finance.
It includes the runnable notebook and source, persistent splits, validation-only
tuning evidence, 12 final controlled runs, every recorded epoch duration,
aggregate results, logs, an independent audit and the experiment record.

## Status

- Technical run integrity: **PASS**
- Split integrity and leakage checks: **PASS**
- Validation-only tuning: **PASS**
- Per-epoch timing: **PASS** — 1,135 final-run epoch records
- Final run matrix: **PASS** — TR40/TR30/TR20/TR10 × seeds 2/42/72
- Cross-model comparison readiness: **CONDITIONAL**

## Required dataset-version decision

The executed T-Finance graph has:

- SHA-256: `b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700`
- 39,357 nodes
- 1,804 fraud/anomaly nodes
- 37,553 normal nodes
- 42,445,086 stored directed edge entries
- 10 features

The earlier team dataset audit used a different raw-file SHA-256 and reported
1,803 fraud nodes. Before comparing BWGNN with another model on T-Finance, the
team must freeze one raw-file hash. The recommended action is to adopt the file
used here, update the dataset and heterophily report on that exact file, and
reuse the split files included here for every later T-Finance model. If the team
keeps the earlier file instead, this BWGNN experiment must be rerun.

## Controlled protocol

- Model: homogeneous BWGNN
- Beta-wavelet order: 2, unchanged from the author setting
- Selected hidden dimension: 32
- Tuning criterion: validation AUROC only; test evaluation disabled
- Optimizer: Adam
- Learning rate: 0.01
- Weight decay: 0
- Maximum epochs: 100
- Validation interval: 5 epochs
- Early-stopping patience: 20 epochs
- Split seed: 2
- Training seeds: 2, 42 and 72
- Hardware: one NVIDIA Tesla T4, GPU 0
- Software: Python 3.11.16, PyTorch 2.2.2+cu121, DGL 1.1.3+cu121

## Mean final results across three seeds

| Ratio | AUROC | AUPRC | Macro-F1 | Fraud F1 | Mean epoch time |
|---|---:|---:|---:|---:|---:|
| TR40 | 95.73 ± 0.63% | 80.15 ± 8.89% | 87.59 ± 3.08% | 76.29 ± 5.80% | 0.276 s |
| TR30 | 95.88 ± 0.12% | 84.13 ± 1.52% | 88.80 ± 1.12% | 78.58 ± 2.16% | 0.276 s |
| TR20 | 95.62 ± 0.29% | 82.94 ± 3.72% | 88.36 ± 2.15% | 77.66 ± 4.12% | 0.274 s |
| TR10 | 95.49 ± 0.42% | 82.83 ± 1.81% | 88.03 ± 1.01% | 77.11 ± 1.96% | 0.272 s |

TR30 has the largest mean values in this three-seed run, but the differences
are small and do not establish that removing labels improves the model. No
claim of superiority over another model should be made until the same raw graph,
split IDs, evaluator and experimental conditions are used.

## Folder guide

- `docs/` — verified experiment record and version-decision note
- `notebooks/` — Kaggle Run All notebook
- `source/` — original and instrumented BWGNN workflow code
- `shared/splits/tfinance/` — persistent nested split files and manifest
- `results/` — smoke, author-setting, tuning and all final runs
- `logs/` — terminal logs for all stages
- `reports/` — registries, aggregate tables and independent audit
- `evidence/` — original package manifest and verification evidence

## Re-running

Attach the raw T-Finance DGL graph privately in Kaggle, then import and run:

`notebooks/BWGNN_TFinance_COMP8851_Run_All.ipynb`

The raw graph is intentionally not committed to GitHub. See the notebook and
`source/README_WORKFLOW.md` for the complete execution sequence.

## Sources

- Paper: https://proceedings.mlr.press/v162/tang22b.html
- Official implementation: https://github.com/squareRoot3/Rethinking-Anomaly-Detection

