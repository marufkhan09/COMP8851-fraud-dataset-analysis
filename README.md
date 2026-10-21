# COMP8851 - Benchmarking Graph Neural Networks for Fraud Detection

Major project, COMP8851 (Macquarie University). Supervisor: Dr Venus Haghighi.
Team: Md Maruf Khan, Pathik Ahmed, Ishraq Ahmed ("Raiyan"), Akib Hasan Maruf.

## Purpose

Published fraud-detection GNNs are reported on different datasets, splits and
metrics, so their results cannot be compared directly. This repository holds the
code, configurations, logs and result tables for a controlled comparison of 12
fraud-specific or fraud-adapted GNNs on six public fraud-graph datasets. It is
meant to let a reader see what was run, check it, and regenerate it.

Research questions (integrated in the report, not a separate section):
1. How do the models compare in predictive performance?
2. How do class imbalance and label scarcity affect them?
3. How do they compare in computational efficiency and scalability?
4. How do graph characteristics (heterophily, structure) relate to model behaviour?

## Models and datasets

Models (12): CARE-GNN, PC-GNN, FRAUDRE, BWGNN, GHRN, GAGA, SplitGNN, PMP,
DGA-GNN, GAAP, LEX-GNN, ConsisGAD.

Datasets (6):

| Dataset | Nodes | Fraud nodes | Notes |
|---|---:|---:|---|
| YelpChi | 45,954 | about 14.5% | three review relations |
| Amazon | 11,944 | 821 of 8,639 labelled (9.5%) | opinion-fraud graph; unlabelled users excluded |
| T-Finance | 39,357 | 1,804 | single relation |
| Elliptic | 203,769 | labelled subset only | temporal transaction graph, chronological split |
| FDCompCN | 5,317 | 559 (10.51%) | three company relations |
| T-Social | 5,781,065 | 174,280 (3.01%) | 146,211,016 directed edges |

Per-dataset analysis (class balance, structure, global and fraud-node local
heterophily) is in the dataset notebooks: `yelpchi/`, `amazon/`, `elliptic/`,
`fdcompcn/`, `tfinance/`, `tsocial/`.

## Evaluation protocol

- Nested label-scarcity ladder: training ratios TR40, TR30, TR20 and TR10.
  Validation and test nodes stay fixed, and each smaller training set is a
  subset of the larger one.
- Split seed 2. Training seeds 2, 42 and 72. Results are mean +/- sample SD.
- Primary metric AUPRC. Also AUROC, Macro-F1, fraud-class F1, G-Mean,
  fraud precision and recall, accuracy, and confusion-matrix counts.
- Hyperparameters, checkpoints and decision thresholds are chosen on validation
  nodes only. The test set is evaluated once, after training.
- Per-epoch time, total time, inference time and peak GPU memory are recorded.
  Runs that shared a GPU are marked, and their timings are not used for the
  efficiency comparison.

Deviations from this protocol are not hidden. They are declared in the README of
the model and dataset concerned. Examples: native-protocol runs, provisional or
reconstructed splits (LEX-GNN and ConsisGAD on Elliptic and T-Social), reduced epoch
budgets, and runs that failed or timed out. A failed run
is kept as a result, not replaced.

## Repository layout

```
models/<model>/            one folder per model
    README.md              what the model does, how to run it, results status
    source/                code or patches used (where applicable)
    <dataset>/             one folder per dataset
        README.md          status, configuration, deviations, findings
        configs/ results/ evidence/   configs, result tables, logs
shared/splits/             frozen nested split files used by several models
docs/planning, docs/run-guides, docs/references   plans, guides, paper links
<dataset>/                 dataset analysis notebooks
```

Layout of the model folders is not yet identical for every model. The models with
full per-dataset folders are PC-GNN, DGA-GNN, GAGA, LEX-GNN and ConsisGAD. Where a
model's folder holds only a README and `references/`, its evidence has not been
added to this repository yet.

## Reproducing a result

1. Clone this repository.
2. Get the data. Raw datasets are not stored here (size and licensing).
   YelpChi and Amazon come from `dgl.data.FraudDataset`. T-Finance, Elliptic and
   T-Social come from the shared raw-data archive `fraud_graph_rawdata.7z` used
   by the original repositories. FDCompCN comes from the SplitGNN repository
   (`data/FDCompCN.zip`). Each dataset folder lists the source and a SHA-256 of the
   file used.
3. Create the environment for the model. Versions differ between models (for
   example PC-GNN's original code targets an old PyTorch, GAGA needs DGL
   0.6.1 with CUDA 11.0 libraries), so there is no single requirements file. Each
   model README lists the working setup and its record of installed packages.
4. Run the model's configs. The exact commands are in each model README.
5. Check your output against the result tables in the model's dataset folder and
   the SHA-256 manifests.

Large artefacts (datasets, checkpoints, prediction arrays) are excluded from
Git on purpose. The `.gitignore` also ignores folders named `runs/`, `logs/`,
`datasets/` and `exports/`, so name result folders differently.

## Reports

Assessment 2 report and slides are submitted separately. The final tables and
figures are produced from the result files in this repository.

## Verified LEX-GNN and ConsisGAD outcomes

The following outcomes were recorded in the October 2026 benchmark:

- **FDCompCN:** ConsisGAD completed 12/12 final runs.
  LEX-GNN completed 11/12; TR20 seed72 failed the configured
  initialization-restart gate. No successful test evaluation was
  produced for that run. The failure is documented under
  `models/lex-gnn/fdcompcn/evidence/final_runs/`.
- **Elliptic:** Both models completed 12/12 runs using the
  `elliptic_reconstructed_v1` protocol. These were not the original
  team-frozen splits.
- **T-Social:** Both models completed 12/12 runs using provisional
  splits reconstructed from the documented convention, not the
  original team-frozen split files.

These qualifications must be retained when comparing results or
reporting the number of successful experiments.
