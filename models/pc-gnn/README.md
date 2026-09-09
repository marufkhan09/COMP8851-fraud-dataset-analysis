# PC-GNN YelpChi experiment evidence

This folder contains the verified PC-GNN implementation snapshot, exact experiment evidence and consolidated results for the COMP8851 benchmark.

## Current status

- Model: PC-GNN
- Dataset: YelpChi
- Controlled ratios: TR40, TR30, TR20 and TR10
- Training seeds: 2, 42 and 72
- Split seed: 2
- Selected rho: 0.4, chosen by validation AUROC only
- Hardware: one NVIDIA Tesla T4 on GPU 0
- Final runs: 12 of 12 verified
- Epoch timing rows: 1,200

## Important paths

- `source/` contains the instrumented source used for persistent splits, metrics and timing.
- `notebooks/pcgnn_kaggle_execution.ipynb` contains the Kaggle workflow.
- `results/aggregate/` contains the seed-level table, mean and sample-SD table, matched deltas and verification report.
- `evidence/raw/` contains machine-readable outputs, configurations and logs. Checkpoints are intentionally excluded from Git.
- `../../shared/splits/yelpchi/` contains the canonical nested YelpChi split family.
- `docs/PC-GNN_Experiment_Record_Final.docx` is the readable experiment record for the team and supervisor.

## Source identity

- Paper: https://doi.org/10.1145/3442381.3449989
- Official repository: https://github.com/PonderLY/PC-GNN
- Recorded upstream commit: `9d7d7fae491081178b6e12e193fb89cfc330f2be`

## Controlled final configuration

Adam, learning rate 0.01, weight decay 0.001, batch size 1024, embedding dimension 64, alpha 2, rho 0.4, 100 epochs, validation every five epochs and threshold 0.5. The PC-GNN architecture was not redesigned.

## Run and review order

1. Read the experiment record in `docs/`.
2. Check `results/aggregate/pcgnn_verification_report.json` reports `PASS`.
3. Use the persistent split files in `shared/splits/yelpchi/`; do not regenerate them per model.
4. Use the saved configuration and environment evidence for reproduction.
5. Compare models only on the same dataset, split IDs, ratio, seed set and evaluation rule.
6. Keep test data isolated from hyperparameter, checkpoint and threshold selection.

## Repository size policy

Raw YelpChi data and model checkpoints are excluded because of size and redistribution concerns. Keep the original downloaded result ZIPs outside Git or use an approved large-file store. Their complete SHA-256 hashes are in `results/aggregate/pcgnn_archive_checksums.csv`.
