# SplitGNN - YelpChi Author-Mode Reproduction

**Status:** PASS
**Classification:** Approximate reproduction

## Summary
Full 1000-epoch author-mode training run completed successfully on Kaggle Tesla T4 GPU. Training ran the complete epoch budget without early stopping triggering (best AUC epoch was 940, within patience of 100 from epoch 1000).

## Deviations from author's exact setup
- Used shipped config/yelp.yaml defaults (gamma=1, C=1, K=0), which differ from the README's stated per-dataset table (gamma=0.6, C=2, K=1 for YelpChi).
- Modern dependency versions used instead of author's original environment: numpy 1.26.4, torch 2.2.1+cu121, dgl 2.4.0+cu121.

## Results
| Metric | Best-AUC model | Best-G-Mean model |
|---|---|---|
| Recall | 0.8621 | 0.8446 |
| F1-macro | 0.7410 | 0.7330 |
| AUC | 0.9189 | 0.9126 |
| G-Mean | 0.8414 | 0.8311 |

## Interpretation
AUC of ~0.92 is consistent with the range of strong published results for YelpChi-style fraud detection benchmarks.

## Runtime
Total training time: 1589.97 seconds (~26.5 minutes) for 1000 epochs on Tesla T4 GPU.