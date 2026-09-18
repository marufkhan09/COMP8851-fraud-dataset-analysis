# Fraud precision — TR40

Mean ± sample standard deviation over training seeds 2, 42 and 72. A status in place of a number means the cell produced no test evaluation; see `../02_EXPERIMENT_MATRIX.md` for the reason.

| Dataset | CARE-GNN | GHRN | Difference |
|---|---|---|---|
| YelpChi | 0.5066 ± 0.0162 | 0.5249 ± 0.0123 | +0.0183 |
| Amazon | 0.8893 ± 0.0085 | 0.9120 ± 0.0229 | +0.0226 |
| T-Finance | 0.9671 ± 0.0267 | 0.7228 ± 0.1337 | -0.2443 |
| T-Social | _FAILED_ | _FAILED_ | — |
| Elliptic | 0.4654 ± 0.1147 | 0.3973 ± 0.0381 | -0.0681 |
| FDCompCN | 0.1363 ± 0.0167 | 0.1249 ± 0.0179 | -0.0114 |

Difference is GHRN minus CARE-GNN; positive favours GHRN.
