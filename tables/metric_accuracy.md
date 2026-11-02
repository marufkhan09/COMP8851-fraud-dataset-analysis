# Accuracy — TR40

Mean ± sample standard deviation over training seeds 2, 42 and 72. A status in place of a number means the cell produced no test evaluation; see `../02_EXPERIMENT_MATRIX.md` for the reason.

| Dataset | CARE-GNN | GHRN | Difference |
|---|---|---|---|
| YelpChi | 0.8563 ± 0.0042 | 0.8612 ± 0.0027 | +0.0049 |
| Amazon | 0.9554 ± 0.0241 | 0.9747 ± 0.0020 | +0.0193 |
| T-Finance | 0.9727 ± 0.0069 | 0.9729 ± 0.0096 | +0.0003 |
| T-Social | _FAILED_ | _FAILED_ | — |
| Elliptic | 0.9103 ± 0.0299 | 0.8935 ± 0.0155 | -0.0168 |
| FDCompCN | 0.8167 ± 0.0161 | 0.8363 ± 0.0394 | +0.0196 |

Difference is GHRN minus CARE-GNN; positive favours GHRN.
