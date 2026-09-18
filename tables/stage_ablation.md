# GHRN refinement ablation — stage 1 against stage 2

GHRN trains in two stages: **stage 1** fits the beta-wavelet backbone on the unpruned graph, and its predictions drive the edge pruning; **stage 2** retrains on the refined graph. Comparing the two isolates what the heterophily refinement contributes, independently of any comparison with CARE-GNN.

All values are **best validation AUPRC**, the protocol's selection metric. Nothing here touches the test set.

| Dataset | Seed | Stage 1 (unrefined) | Stage 2 (refined) | Δ | Refinement helped? |
|---|---|---|---|---|---|
| YelpChi | 2 | 0.5050 | 0.5167 | +0.0117 | yes |
| YelpChi | 42 | 0.5175 | 0.5200 | +0.0026 | yes |
| YelpChi | 72 | 0.5141 | 0.5370 | +0.0228 | yes |
| Amazon | 2 | 0.8807 | 0.8545 | -0.0262 | **no** |
| Amazon | 42 | 0.8885 | 0.8399 | -0.0486 | **no** |
| Amazon | 72 | 0.8614 | 0.8268 | -0.0346 | **no** |
| T-Finance | 2 | 0.8782 | 0.4441 | -0.4341 | **no** |
| T-Finance | 42 | 0.8386 | 0.8586 | +0.0200 | yes |
| T-Finance | 72 | 0.8453 | 0.7603 | -0.0850 | **no** |
| Elliptic | 2 | 0.7191 | 0.7173 | -0.0018 | **no** |
| Elliptic | 42 | 0.6857 | 0.7172 | +0.0315 | yes |
| Elliptic | 72 | 0.6918 | 0.7101 | +0.0183 | yes |
| FDCompCN | 2 | 0.1817 | 0.2081 | +0.0264 | yes |
| FDCompCN | 42 | 0.1632 | 0.2058 | +0.0426 | yes |
| FDCompCN | 72 | 0.1738 | 0.2239 | +0.0501 | yes |

## Mean effect per dataset

| Dataset | Stage 1 | Stage 2 | Mean Δ | Seeds helped |
|---|---|---|---|---|
| YelpChi | 0.5122 | 0.5246 | +0.0124 | 3 of 3 |
| Amazon | 0.8769 | 0.8404 | -0.0365 | 0 of 3 |
| T-Finance | 0.8541 | 0.6877 | -0.1664 | 1 of 3 |
| Elliptic | 0.6989 | 0.7149 | +0.0160 | 2 of 3 |
| FDCompCN | 0.1729 | 0.2126 | +0.0397 | 3 of 3 |

Across all 15 GHRN runs, refinement improved validation AUPRC in **9 of 15**.

`selected_stage` in each run's `summary.json` records which stage supplied the final checkpoint.
