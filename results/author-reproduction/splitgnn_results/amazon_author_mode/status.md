# SplitGNN - Amazon Author-Mode Reproduction

**Status:** PASS
**Classification:** Approximate reproduction

## Summary
Author-mode training run on Amazon completed successfully on Kaggle Tesla T4 GPU. Early stopping triggered: best validation AUC was reached at epoch 179, training continued until patience (200 epochs) was exhausted, stopping around epoch 370-379.

## Deviations from author's exact setup
- Used shipped config/amazon.yaml defaults (gamma=1, C=1, K=0), which differ from the README's stated per-dataset table (gamma=0.4, C=2, K=1 for Amazon).
- Modern dependency versions used instead of author's original environment: numpy 1.26.4, torch 2.2.1+cu121, dgl 2.4.0+cu121.

## Results
| Metric | Best-AUC model | Best-G-Mean model |
|---|---|---|
| Recall | 0.9061 | 0.9000 |
| F1-macro | 0.6875 | 0.7049 |
| AUC | 0.9250 | 0.9217 |
| G-Mean | 0.8559 | 0.8634 |

## Interpretation
AUC of ~0.925 is slightly higher than YelpChi's ~0.919, consistent with Amazon generally being reported as an easier fraud-detection benchmark.

## Runtime
Total training time: 4526.82 seconds (~75.4 minutes). Longer than YelpChi (~26.5 minutes) despite Amazon's smaller graph, driven by the higher early_stop patience (200 vs 100).