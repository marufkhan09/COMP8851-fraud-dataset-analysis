# Efficiency — TR40

Mean training time per epoch, peak GPU memory and trainable parameter count, averaged over the three training seeds.

> Timings and memory are comparable **only within** one GPU block. The two blocks must not be pooled. Predictive metrics are unaffected by GPU class and remain comparable everywhere.

## NVIDIA RTX A6000

| Model | Dataset | Mean epoch (s) | Peak GPU (MB) | Parameters | Mean epochs |
|---|---|---|---|---|---|
| CARE-GNN | YelpChi | 7.636 | 276.7 | 1,314 | 100 |
| CARE-GNN | Amazon | 1.300 | 115.9 | 2,100 | 46 |
| CARE-GNN | Elliptic | 1.391 | 276.5 | 11,278 | 86 |
| CARE-GNN | FDCompCN | 0.710 | 30.8 | 4,276 | 33 |
| GHRN | YelpChi | 0.020 | 1106.8 | 18,754 | 200 |
| GHRN | Amazon | 0.017 | 1184.2 | 18,306 | 178 |
| GHRN | Elliptic | 0.028 | 725.2 | 27,330 | 161 |
| GHRN | FDCompCN | 0.013 | 48.6 | 73,474 | 114 |

## Tesla T4

| Model | Dataset | Mean epoch (s) | Peak GPU (MB) | Parameters | Mean epochs |
|---|---|---|---|---|---|
| CARE-GNN | T-Finance | 1.741 | 302.3 | 982 | 34 |
| GHRN | T-Finance | 0.364 | 5570.6 | 17,346 | 163 |

