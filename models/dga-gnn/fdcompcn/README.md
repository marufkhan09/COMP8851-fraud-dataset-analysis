# DGA-GNN on FDCompCN

## Status: Provisional (non-native dataset)

FDCompCN is **not natively supported** by DGA-GNN's official repository
(no config, no data-loading path in `data_handle.py`). Unlike YelpChi,
Amazon, T-Finance, and Elliptic — which all have paper-provided configs
and native loaders — this required building an adapter from scratch.

## Source

- Official SplitGNN repository: https://github.com/split-gnn/splitgnn
- Commit: see `evidence/preprocessing/fdcompcn_splitgnn_commit.txt`
- File: `data/FDCompCN.zip` → `comp.dgl` (pre-built DGL heterograph)
- SHA256: see `evidence/preprocessing/fdcompcn_zip.sha256`
- 5,317 nodes (`company`), 559 fraud (10.51%), 57 features
- Three semantic relations preserved: `invest_bc2bc`, `provide_bc2bc`, `sale_bc2bc`
  (the graph also ships a `homo` union relation, which was intentionally
  excluded — DGA-GNN is relation-aware and per the master plan, relation-aware
  models receive the relation-preserving view, not the homogeneous union)

## Split: PROVISIONAL, not the frozen COMP8851 registry

Per the master plan, all final unified runs must use frozen COMP8851 split
IDs from the team's dataset registry (owned by Pathik Ahmed). **That
registry does not yet contain FDCompCN as of this run** (2026-09-16).

To avoid blocking DGA-GNN work, splits were generated locally following the
exact same convention used for every other dataset in this pipeline:
stratified `train_test_split`, `random_state=2`, TR40/30/20/10 train ratios,
remainder split 33%/67% into validation/test. See
`evidence/splits/fdcompcn_split_manifest_PROVISIONAL.json` for exact counts,
per-split SHA256, and full generation parameters.

**This split MUST be replaced with the official frozen COMP8851 split once
Pathik's registry publishes one for FDCompCN**, and any results here
re-validated against it before being treated as final/comparable to other
models' FDCompCN numbers.

## Hyperparameter tuning: k (bin-encoding bins)

DGA-GNN's native configs tier the `k` hyperparameter by dataset size:
- `k=32`: YelpChi (45,954 nodes), T-Social (millions of nodes)
- `k=4`: Amazon (11,944), Elliptic, T-Finance (39,357)

Since FDCompCN has no native config, `k=32` was tried first (naively
mirroring YelpChi, since both are multi-relation). This caused severe
overfitting:

| k | epochs | trn_auc | tst_auc | gap |
|---|---|---|---|---|
| 32 (rejected) | 76 | 0.989 | 0.690 | ~30 pts |
| 4 (final) | 254 | 0.938 | 0.784 | ~15 pts |

FDCompCN (5,317 nodes) is smaller than every dataset in the `k=4` tier, so
`k=4` is the correct bracket. The rejected `k=32` run log is kept at
`evidence/tuning/train_log_k32_overfit_REJECTED.log` as documentation of
this decision — not a result to report.

All other hyperparameters match the team protocol default: `seed=2`,
`bs=256`, `lr=1e-3`, `weight_decay=5e-4`, `patience=25`, matching the
`protocol_safe_default_seed2_parallel_accuracy` convention used for the
other four datasets.

## Final results (TR40, seed 2, k=4)

- Epochs to convergence: 254 (early stopping, patience=25)
- `final_trn/auc`: 0.938, `final_trn/aps`: 0.731
- `final_val/auc`: 0.794, `final_val/aps`: 0.410
- `final_tst/auc`: 0.784, `final_tst/aps`: 0.390
- `final_tst/pre`: 0.442, `final_tst/rec`: 0.411, `final_tst/mf1`: 0.681

Test and validation AUC track closely (0.784 vs 0.794), suggesting the
model generalizes consistently rather than overfitting specifically to the
validation set. The remaining train/test gap (~15 points) is plausible
given FDCompCN's small size (2,126 training nodes) and is comparable in
character — though not magnitude — to what's expected on such a small graph.

## What's still needed before this is "final"

1. Frozen COMP8851 split registry entry for FDCompCN (from Pathik)
2. Re-run once that split exists; compare against this provisional run
3. TR30/TR20/TR10 runs (only TR40 done so far)
4. Multi-seed runs (42, 72) once TR40/seed2 is confirmed stable
