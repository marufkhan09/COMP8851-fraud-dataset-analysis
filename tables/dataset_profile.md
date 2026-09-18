# Dataset profile — graph characteristics (RQ4)

Measured on the canonical graph view each run actually trained on and recorded in that run's `summary.json`. CARE-GNN and GHRN report identical values for every dataset, confirming both models were served the same view.

Conventions: **global heterophily** is different-label eligible edges over all eligible labelled edges. **Fraud-local heterophily** is, for each fraud node, the share of its labelled one-hop neighbours that are benign; mean and median are taken over fraud nodes that have at least one labelled neighbour.

| Dataset | Global heterophily | Fraud-local mean | Fraud-local median | Fraud-local SD | Labelled edges | Test nodes | Test fraud % |
|---|---|---|---|---|---|---|---|
| YelpChi | 0.2270 | 0.8053 | 0.8396 | 0.1450 | 7,693,958 | 18,382 | 14.53% |
| Amazon | 0.0512 | 0.8626 | 0.8958 | 0.1160 | 6,598,582 | 3,456 | 9.52% |
| T-Finance | 0.0292 | 0.4562 | 0.3715 | 0.2940 | 42,445,086 | 15,744 | 4.58% |
| T-Social \* | 0.3761 | 0.8260 | 0.9205 | 0.2440 | 146,211,016 | 2,312,426 | 3.02% |
| Elliptic | 0.0463 | 0.4903 | 0.5000 | 0.4785 | 73,248 | 18,625 | 7.61% |
| FDCompCN | 0.1450 | 0.8617 | 1.0000 | 0.3094 | 7,268 | 2,128 | 10.48% |

\* T-Social has no completed run, so its figures come from the frozen dataset manifest rather than from a run summary — the statistics were computed when the canonical view was built. The record is in `../evidence/tsocial/dataset/`. Everything else is measured inside the runs themselves.

T-Social is the **most heterophilic graph in the suite** by a wide margin (global 0.3761 against YelpChi's 0.2270), and its fraud nodes are among the most camouflaged (fraud-local median 0.9205). That makes the absence of a T-Social result a real gap for RQ4 and not merely a missing row: it removes the single most informative point on the heterophily axis.
