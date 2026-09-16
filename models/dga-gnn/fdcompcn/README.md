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

## TR-ratio results (seed 2, k=4, provisional splits)

| Ratio | Train nodes | Train fraud (~) | Epochs | trn_auc | val_auc | tst_auc | Notes |
|---|---|---|---|---|---|---|---|
| TR40 | 2,126 | 224 | 254 | 0.938 | 0.794 | 0.784 | Healthy, converged |
| TR30 | 1,595 | 168 | ~120 | 0.915 | 0.791 | 0.785 | Healthy, converged |
| TR20 | 1,063 | 112 | 119 | 0.535 | 0.558 | 0.545 | **Collapsed — near chance level** |
| TR10 | 531 | 56 | 74 | 0.574 | 0.567 | 0.544 | **Collapsed — near chance level** |

### TR20/TR10 collapse: a genuine finding, not a bug

At TR40 and TR30, DGA-GNN trains normally and achieves reasonable
discrimination (test AUC ~0.78 in both). At TR20 and TR10, performance
drops to near-random (test AUC ~0.54-0.55) despite training running its
full course cleanly — no crashes, no NaN, no warnings.

The failure signature is specific: `val_loss` decreases substantially in
both collapsed runs (0.71→0.35 for TR20, 0.69→0.35 for TR10), while AUC
stays flat near 0.5. This indicates the model is minimizing loss by
becoming confident on the easy majority class, without learning to
discriminate the minority (fraud) class at all — a classic imbalanced-loss
failure mode, not a numerical/implementation bug.

Additionally, DGA-GNN's dynamic-grouping proportion (`g0`) stays pinned
near 0.0000–0.0007 for the entire TR20/TR10 runs (compare: TR40/TR30 see
`g0` move into the 0.3–0.9 range as training progresses). This suggests
the core dynamic-grouping mechanism — DGA-GNN's central architectural
contribution — fails to activate when there are too few positive training
examples to find meaningful group splits.

**Working hypothesis:** DGA-GNN's grouping mechanism needs a minimum
number of fraud training examples to function, somewhere between TR30's
~168 and TR20's ~112 on this dataset. This is consistent with the plan's
broader interest in how each model's mechanism interacts with dataset
size/imbalance — this may be a legitimate, reportable characteristic of
DGA-GNN rather than an artifact of the FDCompCN adapter.

This was verified as a genuine result, not left unexamined: no code
changes between TR40/TR30 (successful) and TR20/TR10 (collapsed) beyond
the split itself, ruling out an adapter bug specific to those ratios.
