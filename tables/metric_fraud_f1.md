# Fraud F1 — TR40

Mean ± sample standard deviation over training seeds 2, 42 and 72. A status in place of a number means the cell produced no test evaluation; see `../02_EXPERIMENT_MATRIX.md` for the reason.

| Dataset | CARE-GNN | GHRN | Difference |
|---|---|---|---|
| YelpChi | 0.4888 ± 0.0094 | 0.5046 ± 0.0178 | +0.0158 |
| Amazon | 0.6911 ± 0.2384 | 0.8598 ± 0.0098 | +0.1686 |
| T-Finance | 0.5712 ± 0.1697 | 0.6978 ± 0.1025 | +0.1266 |
| T-Social | _FAILED_ | _FAILED_ | — |
| Elliptic | 0.5482 ± 0.0690 | 0.5168 ± 0.0289 | -0.0315 |
| FDCompCN | 0.1368 ± 0.0147 | 0.0964 ± 0.0222 | -0.0404 |

Difference is GHRN minus CARE-GNN; positive favours GHRN.
