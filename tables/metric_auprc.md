# AUPRC — TR40

Mean ± sample standard deviation over training seeds 2, 42 and 72. A status in place of a number means the cell produced no test evaluation; see `../02_EXPERIMENT_MATRIX.md` for the reason.

| Dataset | CARE-GNN | GHRN | Difference |
|---|---|---|---|
| YelpChi | 0.5104 ± 0.0127 | 0.5248 ± 0.0123 | +0.0144 |
| Amazon | 0.7152 ± 0.2320 | 0.8756 ± 0.0235 | +0.1604 |
| T-Finance | 0.7120 ± 0.0403 | 0.6764 ± 0.2060 | -0.0356 |
| T-Social | _FAILED_ | _FAILED_ | — |
| Elliptic | 0.4839 ± 0.0269 | 0.4978 ± 0.0574 | +0.0139 |
| FDCompCN | 0.1340 ± 0.0131 | 0.1204 ± 0.0032 | -0.0136 |

Difference is GHRN minus CARE-GNN; positive favours GHRN.
