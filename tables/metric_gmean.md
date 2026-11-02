# G-mean — TR40

Mean ± sample standard deviation over training seeds 2, 42 and 72. A status in place of a number means the cell produced no test evaluation; see `../02_EXPERIMENT_MATRIX.md` for the reason.

| Dataset | CARE-GNN | GHRN | Difference |
|---|---|---|---|
| YelpChi | 0.6601 ± 0.0141 | 0.6709 ± 0.0258 | +0.0109 |
| Amazon | 0.7586 ± 0.2072 | 0.8982 ± 0.0071 | +0.1396 |
| T-Finance | 0.6389 ± 0.1373 | 0.8167 ± 0.0586 | +0.1778 |
| T-Social | _FAILED_ | _FAILED_ | — |
| Elliptic | 0.8007 ± 0.0077 | 0.8206 ± 0.0020 | +0.0199 |
| FDCompCN | 0.3519 ± 0.0266 | 0.2765 ± 0.0592 | -0.0754 |

Difference is GHRN minus CARE-GNN; positive favours GHRN.
