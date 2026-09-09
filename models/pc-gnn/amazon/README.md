# PC-GNN Amazon experiment evidence

This directory contains the verified Amazon extension of the COMP8851 PC-GNN benchmark.

## Final status

- Dataset: Amazon opinion-fraud graph
- Eligible labelled users: 8,639; the official preprocessing excludes the first 3,305 unlabelled users
- Ratios: TR40, TR30, TR20 and TR10
- Training seeds: 2, 42 and 72
- Persistent split seed: 2
- Selected rho: 0.2, chosen using TR40 validation AUROC only
- Hardware: one NVIDIA Tesla T4 on GPU 0
- Epochs: 51 per run
- Final runs: 12 of 12 verified
- Epoch timing rows: 612
- Source commit: `9d7d7fae491081178b6e12e193fb89cfc330f2be`

## Paths

- `results/aggregate/` — seed-level results, mean/SD results, paired deltas, figures and verification JSON.
- `evidence/raw/final/` — exact final JSON, CSV, configurations and logs; checkpoints are excluded.
- `evidence/raw/tuning/` — validation-only rho-tuning evidence.
- `evidence/raw/author-mode/` — preliminary author-mode evidence; it is not the unified result block.
- `../../../shared/splits/amazon/` — canonical persistent Amazon split family. Do not regenerate per model.
- `docs/PC-GNN_Experiment_Record_YelpChi_Amazon_Final.docx` — consolidated YelpChi and Amazon readable record.

## Reproduction classification

The saved author-mode run is approximate rather than exact: the supplied graph has 25 features and the run used 51 epochs, whereas the paper reports 100 features and 100 epochs. Use the 12-run final block for unified benchmarking.

## Repository rule

Merge this package into the repository root. Do not commit raw Amazon data, checkpoints or ZIP archives. Compare models only when they use the same Amazon split IDs, ratio, seed set and evaluation rule.
