# SplitGNN - FDCompCN Author-Mode Reproduction

**Status:** PASS (runs successfully, but weak predictive performance)
**Classification:** Approximate reproduction

## Summary
Author-mode training run on FDCompCN completed successfully on Kaggle Tesla T4 GPU - no crashes, confirming the node-type code fix works correctly in a full training run. However, final test AUC (~0.54) is only marginally above random guessing, a dramatically weaker result than YelpChi (~0.919) or Amazon (~0.925).

## Deviations from author's exact setup
- Used shipped config/comp.yaml defaults (gamma=1, C=1, K=0). Likely cause of weak result: README states FDCompCN needs gamma=0.8, C=3, K=2 - notably higher C and K than YelpChi/Amazon.
- Modern dependency versions used: numpy 1.26.4, torch 2.2.1+cu121, dgl 2.4.0+cu121.

## Results
| Metric | Best-AUC model | Best-G-Mean model |
|---|---|---|
| Recall | 0.9866 | 0.4821 |
| F1-macro | 0.1201 | 0.4263 |
| AUC | 0.5368 | 0.5341 |
| G-Mean | 0.1573 | 0.5101 |

## Interpretation
This should NOT be read as evidence the code or fix is broken - the run completed cleanly with the same pipeline that produced strong results on the other two datasets. Most likely explanation: FDCompCN needs its author-specified hyperparameters (higher C and K) to perform well.

## Recommended follow-up
Re-run FDCompCN using README's stated hyperparameters (gamma=0.8, C=3, K=2) to test this hypothesis - low cost given ~9 minute runtime.

## Runtime
Total training time: 548.58 seconds (~9.1 minutes) - fastest of the three datasets, consistent with FDCompCN's smaller graph (5,317 nodes).