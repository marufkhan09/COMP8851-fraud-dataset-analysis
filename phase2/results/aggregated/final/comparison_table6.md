# Test AUPRC at TR40 - previous ten models (report Table 6) and the three new models

| Model | YelpChi | Amazon | T-Finance | Elliptic | FDCompCN | T-Social |
|---|---|---|---|---|---|---|
| CARE-GNN | 0.510 ±0.013 | 0.715 ±0.232 | 0.712 ±0.040 | 0.484 ±0.027 | 0.134 ±0.013 | FAILED |
| FRAUDRE | 0.512 ±0.006 | 0.850 ±0.004 | 0.801 ±0.170 | 0.021 ±0.002 + | 0.156 ±0.028 | 0.925 ±0.070 |
| PC-GNN | 0.524 ±0.011 # | 0.882 ±0.006 | 0.751 ±0.002 | 0.465 ±0.063 | 0.246 ±0.020 | - |
| GAGA | 0.795 * | 0.871 * | 0.845 * | 0.599 * | 0.228 * | - |
| DGA-GNN | 0.922 * | 0.927 * | 0.904 * | 0.813 * | 0.298 ±0.147 + | 0.981 * |
| GAAP | 0.899 ±0.008 | 0.909 ±0.008 | 0.887 ±0.016 | 0.756 ±0.023 | 0.329 ±0.013 | 0.994 ±0.031 |
| BWGNN | 0.537 ±0.018 | 0.881 ±0.013 | 0.705 ±0.134 | 0.444 ±0.037 | 0.158 ±0.004 | 0.714 ±0.059 |
| GHRN | 0.525 ±0.012 | 0.876 ±0.024 | 0.676 ±0.206 | 0.498 ±0.057 | 0.120 ±0.003 | FAILED |
| PMP | 0.758 ±0.003 | 0.908 ±0.004 | 0.901 ±0.010 | 0.458 ±0.073 | 0.123 ±0.003 | 0.919 ±0.052 |
| SplitGNN | 0.496 ±0.005 + | 0.550 ±0.016 + | 0.798 ±0.069 + | 0.577 ±0.028 + | 0.110 ±0.022 + | 0.437 ±0.019 + |
| **SAGAD** | 0.506 ±0.024 | 0.878 ±0.010 | 0.812 ±0.060 | 0.703 ±0.007 | 0.197 ±0.077 | 0.883 ±0.004 |
| **SpaceGNN** | 0.729 ±0.008 | 0.915 ±0.001 | 0.905 ±0.006 | 0.613 ±0.058 | 0.224 ±0.024 | 0.962 ±0.012 |
| **CRoC** | 0.584 ±0.076 | 0.855 ±0.032 | 0.810 ±0.004 | 0.469 ±0.041 | 0.109 ±0.003 | 0.772 ±0.046 |

| Random-ranker AUPRC (fraud prevalence) | 0.1453 | 0.0952 | 0.0458 | 0.0761 | 0.1048 | 0.0301 |

Comparability: the new rows use the frozen registry splits (byte-identical data and TR40 partitions to CARE-GNN/GHRN), seeds 2/42/72, validation-AUPRC checkpointing and the shared evaluator, so they are directly comparable with the unmarked three-seed rows. '*' single native run and '+' non-canonical split (report markers) are not comparable. '#' audit finding: PC-GNN's YelpChi partitions differ from the registry.
