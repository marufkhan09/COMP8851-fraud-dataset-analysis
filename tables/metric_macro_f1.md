# Macro-F1 — TR40

Mean ± sample standard deviation over training seeds 2, 42 and 72. A status in place of a number means the cell produced no test evaluation; see `../02_EXPERIMENT_MATRIX.md` for the reason.

| Dataset | CARE-GNN | GHRN | Difference |
|---|---|---|---|
| YelpChi | 0.7026 ± 0.0046 | 0.7119 ± 0.0080 | +0.0093 |
| Amazon | 0.8335 ± 0.1255 | 0.9229 ± 0.0054 | +0.0894 |
| T-Finance | 0.7786 ± 0.0866 | 0.8418 ± 0.0538 | +0.0633 |
| T-Social | _FAILED_ | _FAILED_ | — |
| Elliptic | 0.7491 ± 0.0433 | 0.7284 ± 0.0191 | -0.0207 |
| FDCompCN | 0.5171 ± 0.0080 | 0.5030 ± 0.0060 | -0.0141 |

Difference is GHRN minus CARE-GNN; positive favours GHRN.
