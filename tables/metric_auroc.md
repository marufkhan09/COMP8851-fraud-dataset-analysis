# AUROC — TR40

Mean ± sample standard deviation over training seeds 2, 42 and 72. A status in place of a number means the cell produced no test evaluation; see `../02_EXPERIMENT_MATRIX.md` for the reason.

| Dataset | CARE-GNN | GHRN | Difference |
|---|---|---|---|
| YelpChi | 0.8374 ± 0.0077 | 0.8384 ± 0.0075 | +0.0011 |
| Amazon | 0.8951 ± 0.1016 | 0.9707 ± 0.0107 | +0.0756 |
| T-Finance | 0.8794 ± 0.0436 | 0.9101 ± 0.0666 | +0.0308 |
| T-Social | _FAILED_ | _FAILED_ | — |
| Elliptic | 0.8811 ± 0.0078 | 0.8911 ± 0.0030 | +0.0100 |
| FDCompCN | 0.5527 ± 0.0245 | 0.5525 ± 0.0038 | -0.0001 |

Difference is GHRN minus CARE-GNN; positive favours GHRN.
