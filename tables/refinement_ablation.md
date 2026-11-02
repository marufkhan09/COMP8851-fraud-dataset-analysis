# GHRN graph refinement — what it actually changed (RQ4)

GHRN prunes the lowest-scoring `del_ratio` fraction of edges, aiming to remove connections between nodes of different classes. Every GHRN run recorded edge heterophily immediately before and after that prune, so the mechanism can be measured rather than assumed. Values are from training seed 2; the diagnostic covers edges with both endpoints in the training split.

| Dataset | del_ratio | Edges before | Removed | Removed % | Heterophily before | Heterophily after | Reduction |
|---|---|---|---|---|---|---|---|
| YelpChi | 0.03 | 7,739,912 | 232,197 | 3.00% | 0.226638 | 0.213358 | +0.013280 |
| Amazon | 0.015 | 8,847,096 | 132,706 | 1.50% | 0.052157 | 0.052519 | -0.000361 |
| T-Finance | 0.015 | 42,484,443 | 637,266 | 1.50% | 0.030429 | 0.029055 | +0.001374 |
| Elliptic | 0.03 | 672,479 | 20,174 | 3.00% | 0.042618 | 0.035254 | +0.007364 |
| FDCompCN | 0.03 | 12,585 | 377 | 3.00% | 0.115789 | 0.116034 | -0.000244 |

A positive reduction means refinement made the training-edge set more homophilic, which is what the method intends. A negative value means it did not.
