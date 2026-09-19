# DGA-GNN — T-Social

## Status: PASS (native support, no adapter required)

T-Social is one of DGA-GNN's five natively supported datasets — the official
repository ships a ready-made `configs/tsocial.yaml` config, so this required
no adapter, no code changes to `data_handle.py`, and no compatibility gate
beyond the standard preprocessing → training pipeline already used for
YelpChi/Amazon/T-Finance/Elliptic.

## Source identity

- Official repository: https://github.com/AtwoodDuan/DGA-GNN
- Recorded commit: see `evidence/preprocessing/dga_gnn_commit.txt`
- Raw dataset: `tsocial` file from the shared `fraud_graph_rawdata.7z` archive
  (Google Drive, same source used for T-Finance/Elliptic/Amazon/YelpChi).
  Raw file hash: see `evidence/preprocessing/raw_tsocial.sha256`
- Dataset scale (from `data_handle.py`'s own printed stats, cross-verified
  independently against PC-GNN's separate intake of the same raw file):
  5,781,065 nodes, 146,211,016 edges, 174,280 fraud (3.01%), 10 features,
  single relation.

## Configuration used

Config file: `evidence/benchmark/native_tsocial_seed2/resolved_config.yaml`
(copied directly from the repository's own `code/configs/tsocial.yaml`,
unmodified).

| Field | Value |
|---|---|
| model | dga_bin |
| bin_encoding | true |
| n_head | 2 |
| n_hidden | 128 |
| k | 32 |
| p | 0.2 |
| z | 0.2 |
| seed | 321 |
| bs | 10240 |
| lr | 1e-3 |
| weight_decay | 5e-4 |
| max_epochs | 10000 (early-stopped) |
| patience | 10 |

`k=32` is the author-recommended bin-encoding tier for the two largest
datasets in DGA-GNN's own dataset roster (YelpChi and T-Social) — this is
the paper's own setting, not a value we chose.

One launch-time override was necessary: the shipped config has
`nowandb: False`, which routes wandb into online mode and would otherwise
block on an interactive API-key prompt on a fresh, unauthenticated instance.
Launched with `nowandb=True` (wandb offline logging only, no cloud sync,
no account required) — this changes only how metrics are logged locally,
not any training behavior.

## Result

Training converged cleanly and the LR scheduler triggered a reduction
around epoch 86, consistent with genuine convergence rather than an
early stop from stagnation. Best validation checkpoint at epoch 97.

| Split | AUC | APS |
|---|---|---|
| Train | 0.9995 | 0.9920 |
| Validation | 0.9990 | 0.9824 |
| **Test** | **0.9988** | **0.9805** |

(Final numbers taken from the run's own wandb summary at checkpoint
restore, in `evidence/benchmark/native_tsocial_seed2/final_epochs_and_wandb_summary.txt`;
per-epoch progression in the same folder's `train.log`.)

This is DGA-GNN's strongest result across its full dataset roster so far,
consistent with `k=32` bin-encoding being specifically tuned for
large-scale graphs like this one.

## Cross-model note (T-Social feasibility)

T-Social was attempted on all three models assigned to this contributor
(PC-GNN, DGA-GNN, GAGA) on the same hardware in the same session. Only
DGA-GNN completed successfully:

- **PC-GNN**: did not complete. The vanilla `main.py` author-mode pipeline
  loaded the dataset correctly (confirmed via the printed data summary,
  matching the same node/fraud counts above) but never produced a single
  training epoch after 2+ hours, with flat CPU/memory usage the entire
  time — consistent with an adjacency-construction step (`InterAgg`/
  `IntraAgg`) that does not scale to this edge count. Killed after 2h6m
  with zero output as a substantiated non-completion, not a crash.
- **GAGA**: did not complete. Its reference preprocessing
  (`GroupFeatureSequenceLoader.load_batch`) processes nodes one at a time
  in a Python loop; at T-Social's scale this measured 1.44 iterations/
  second, giving a projected completion time of approximately 1,117 hours
  (~46 days). Killed after confirming the rate was stable, not transient.

This is not a claim that PC-GNN or GAGA cannot in principle handle graphs
this large — DGA-GNN's own bin-encoding step is vectorized where these two
reference implementations are not — but it is a real, measured finding
about their reference implementations as shipped, on this hardware,
without further re-engineering.
