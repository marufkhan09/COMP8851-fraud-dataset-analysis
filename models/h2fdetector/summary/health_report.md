# H2-FDetector unified results: health report

- Run folders: 84
- Completed runs: 72
- Failed runs: 12

## Collapsed runs (no fraud predicted / fraud recall 0)

- comp tr10 seed2: fraud_recall=0.0000, predicted fraud=0, AUPRC=0.1165
- comp tr20 seed72: fraud_recall=0.0000, predicted fraud=0, AUPRC=0.1084

## Weak outlier seeds (AUPRC >0.10 below the other seeds at the same ratio)

- amazon tr10 seed42: AUPRC=0.344 vs other seeds mean 0.715, best_epoch=9
- amazon tr30 seed42: AUPRC=0.600 vs other seeds mean 0.743, best_epoch=6
- amazon tr40 seed42: AUPRC=0.554 vs other seeds mean 0.730, best_epoch=14
- yelp tr10 seed2: AUPRC=0.243 vs other seeds mean 0.428, best_epoch=2
- yelp tr30 seed72: AUPRC=0.238 vs other seeds mean 0.357, best_epoch=3
- yelp tr40 seed42: AUPRC=0.248 vs other seeds mean 0.403, best_epoch=2

## Threshold at the edge of the 0.05-0.95 grid

- comp tr10 seed2: threshold=0.95
- comp tr30 seed72: threshold=0.95
- comp tr40 seed72: threshold=0.95
- tfinance_sub15 tr10 seed42: threshold=0.95
- tfinance_sub15 tr10 seed72: threshold=0.95
- tfinance_sub15 tr30 seed42: threshold=0.95
- tfinance_sub15 tr40 seed2: threshold=0.95
- tfinance_sub15 tr40 seed42: threshold=0.95
- yelp tr10 seed2: threshold=0.95
- yelp tr40 seed42: threshold=0.05

## Checkpoint reload mismatch

- None. Reloaded checkpoints reproduce best val AUPRC.

## Runs with many slow epochs (>10% of epochs >1.5x median)

- comp tr40 seed2: 13/45 epochs >1.5x median (0.1729s), max 0.68s
- tfinance_sub15 tr10 seed2: 8/47 epochs >1.5x median (1.7296s), max 3.26s

## Failed runs

- tfinance tr10 seed2: 0 epochs, last log: train=3935 valid=7872 test=15743  dataset_load=2.35s
- tfinance tr10 seed42: 0 epochs, last log: train=3935 valid=7872 test=15743  dataset_load=1.99s
- tfinance tr10 seed72: 0 epochs, last log: train=3935 valid=7872 test=15743  dataset_load=3.56s
- tfinance tr20 seed2: 0 epochs, last log: train=7870 valid=7872 test=15743  dataset_load=1.91s
- tfinance tr20 seed42: 0 epochs, last log: train=7870 valid=7872 test=15743  dataset_load=1.98s
- tfinance tr20 seed72: 0 epochs, last log: train=7870 valid=7872 test=15743  dataset_load=1.93s
- tfinance tr30 seed2: 0 epochs, last log: train=11807 valid=7872 test=15743  dataset_load=2.22s
- tfinance tr30 seed42: 0 epochs, last log: train=11807 valid=7872 test=15743  dataset_load=2.16s
- tfinance tr30 seed72: 0 epochs, last log: train=11807 valid=7872 test=15743  dataset_load=2.22s
- tfinance tr40 seed2: 0 epochs, last log: train=15742 valid=7872 test=15743  dataset_load=2.17s
- tfinance tr40 seed42: 0 epochs, last log: train=15742 valid=7872 test=15743  dataset_load=2.09s
- tfinance tr40 seed72: 0 epochs, last log: train=15742 valid=7872 test=15743  dataset_load=2.41s

## Mean test metrics over seeds

| dataset | ratio | n | auprc | auc | f1_macro | fraud_precision | fraud_recall | fraud_f1 | g_mean |
|---|---|---|---|---|---|---|---|---|---|
| amazon | TR40 | 3 | 0.672 ± 0.102 | 0.919 ± 0.005 | 0.854 ± 0.041 | 0.693 ± 0.147 | 0.788 ± 0.040 | 0.730 ± 0.074 | 0.874 ± 0.020 |
| amazon | TR30 | 3 | 0.695 ± 0.109 | 0.905 ± 0.033 | 0.865 ± 0.032 | 0.758 ± 0.083 | 0.747 ± 0.090 | 0.749 ± 0.059 | 0.855 ± 0.051 |
| amazon | TR20 | 3 | 0.690 ± 0.020 | 0.900 ± 0.023 | 0.879 ± 0.022 | 0.819 ± 0.016 | 0.735 ± 0.066 | 0.774 ± 0.043 | 0.852 ± 0.039 |
| amazon | TR10 | 3 | 0.591 ± 0.214 | 0.852 ± 0.075 | 0.825 ± 0.094 | 0.722 ± 0.227 | 0.635 ± 0.130 | 0.674 ± 0.173 | 0.786 ± 0.091 |
| comp | TR40 | 3 | 0.131 ± 0.018 | 0.542 ± 0.031 | 0.495 ± 0.049 | 0.131 ± 0.035 | 0.217 ± 0.156 | 0.145 ± 0.046 | 0.387 ± 0.110 |
| comp | TR30 | 3 | 0.116 ± 0.006 | 0.518 ± 0.026 | 0.510 ± 0.013 | 0.128 ± 0.016 | 0.220 ± 0.042 | 0.161 ± 0.023 | 0.424 ± 0.038 |
| comp | TR20 | 3 | 0.115 ± 0.009 | 0.517 ± 0.021 | 0.489 ± 0.019 | 0.128 ± 0.125 | 0.033 ± 0.046 | 0.043 ± 0.054 | 0.132 ± 0.142 |
| comp | TR10 | 3 | 0.129 ± 0.013 | 0.549 ± 0.024 | 0.503 ± 0.028 | 0.112 ± 0.099 | 0.079 ± 0.088 | 0.084 ± 0.080 | 0.212 ± 0.198 |
| elliptic | TR40 | 3 | 0.138 ± 0.012 | 0.558 ± 0.033 | 0.535 ± 0.029 | 0.261 ± 0.067 | 0.104 ± 0.075 | 0.134 ± 0.068 | 0.302 ± 0.108 |
| elliptic | TR30 | 3 | 0.127 ± 0.009 | 0.569 ± 0.026 | 0.528 ± 0.041 | 0.180 ± 0.032 | 0.119 ± 0.097 | 0.129 ± 0.098 | 0.298 ± 0.178 |
| elliptic | TR20 | 3 | 0.159 ± 0.040 | 0.602 ± 0.082 | 0.555 ± 0.031 | 0.200 ± 0.039 | 0.244 ± 0.123 | 0.208 ± 0.071 | 0.451 ± 0.120 |
| elliptic | TR10 | 3 | 0.132 ± 0.012 | 0.553 ± 0.029 | 0.545 ± 0.013 | 0.207 ± 0.011 | 0.136 ± 0.039 | 0.163 ± 0.031 | 0.356 ± 0.048 |
| tfinance_sub15 | TR40 | 3 | 0.846 ± 0.009 | 0.956 ± 0.004 | 0.889 ± 0.012 | 0.893 ± 0.025 | 0.707 ± 0.054 | 0.788 ± 0.024 | 0.839 ± 0.031 |
| tfinance_sub15 | TR30 | 3 | 0.848 ± 0.009 | 0.962 ± 0.003 | 0.884 ± 0.018 | 0.867 ± 0.052 | 0.710 ± 0.081 | 0.777 ± 0.035 | 0.839 ± 0.047 |
| tfinance_sub15 | TR20 | 3 | 0.858 ± 0.014 | 0.968 ± 0.000 | 0.894 ± 0.016 | 0.890 ± 0.037 | 0.722 ± 0.033 | 0.797 ± 0.031 | 0.848 ± 0.020 |
| tfinance_sub15 | TR10 | 3 | 0.856 ± 0.010 | 0.957 ± 0.010 | 0.899 ± 0.012 | 0.825 ± 0.063 | 0.793 ± 0.023 | 0.807 ± 0.023 | 0.887 ± 0.012 |
| tsocial | TR40 | 3 | 0.634 ± 0.021 | 0.936 ± 0.002 | 0.812 ± 0.008 | 0.624 ± 0.022 | 0.647 ± 0.017 | 0.635 ± 0.016 | 0.800 ± 0.011 |
| tsocial | TR30 | 3 | 0.628 ± 0.034 | 0.939 ± 0.004 | 0.814 ± 0.013 | 0.633 ± 0.037 | 0.645 ± 0.013 | 0.639 ± 0.025 | 0.798 ± 0.009 |
| tsocial | TR20 | 3 | 0.627 ± 0.022 | 0.933 ± 0.011 | 0.810 ± 0.010 | 0.627 ± 0.025 | 0.638 ± 0.016 | 0.632 ± 0.020 | 0.794 ± 0.010 |
| tsocial | TR10 | 3 | 0.638 ± 0.006 | 0.938 ± 0.002 | 0.814 ± 0.002 | 0.635 ± 0.030 | 0.645 ± 0.028 | 0.639 ± 0.003 | 0.798 ± 0.017 |
| yelp | TR40 | 3 | 0.352 ± 0.099 | 0.775 ± 0.058 | 0.629 ± 0.070 | 0.347 ± 0.088 | 0.623 ± 0.174 | 0.427 ± 0.041 | 0.677 ± 0.010 |
| yelp | TR30 | 3 | 0.317 ± 0.120 | 0.740 ± 0.076 | 0.605 ± 0.076 | 0.343 ± 0.117 | 0.304 ± 0.157 | 0.317 ± 0.141 | 0.511 ± 0.138 |
| yelp | TR20 | 3 | 0.259 ± 0.007 | 0.719 ± 0.020 | 0.545 ± 0.060 | 0.298 ± 0.069 | 0.392 ± 0.324 | 0.249 ± 0.187 | 0.473 ± 0.296 |
| yelp | TR10 | 3 | 0.367 ± 0.111 | 0.767 ± 0.055 | 0.645 ± 0.069 | 0.388 ± 0.127 | 0.549 ± 0.124 | 0.433 ± 0.056 | 0.661 ± 0.012 |
