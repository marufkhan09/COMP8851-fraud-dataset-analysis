# Fraud recall — TR40

Mean ± sample standard deviation over training seeds 2, 42 and 72. A status in place of a number means the cell produced no test evaluation; see `../02_EXPERIMENT_MATRIX.md` for the reason.

| Dataset | CARE-GNN | GHRN | Difference |
|---|---|---|---|
| YelpChi | 0.4731 ± 0.0236 | 0.4876 ± 0.0433 | +0.0145 |
| Amazon | 0.6099 ± 0.2939 | 0.8136 ± 0.0137 | +0.2036 |
| T-Finance | 0.4212 ± 0.1665 | 0.6778 ± 0.0923 | +0.2566 |
| T-Social | _FAILED_ | _FAILED_ | — |
| Elliptic | 0.6916 ± 0.0392 | 0.7436 ± 0.0185 | +0.0520 |
| FDCompCN | 0.1390 ± 0.0237 | 0.0867 ± 0.0407 | -0.0523 |

Difference is GHRN minus CARE-GNN; positive favours GHRN.
