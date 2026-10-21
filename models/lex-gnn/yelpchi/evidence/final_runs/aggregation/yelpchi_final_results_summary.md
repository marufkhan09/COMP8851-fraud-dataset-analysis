# YelpChi final three-seed results

Values are mean ± sample standard deviation across training seeds 2, 42 and 72. The split seed remained fixed at 2.

## Performance

| Model | Ratio | AUROC | AUPRC | Macro-F1 | Fraud-F1 | G-Mean |
|---|---:|---:|---:|---:|---:|---:|
| LEX-GNN | TR40 | 0.9615 ± 0.0022 | 0.8483 ± 0.0103 | 0.8641 ± 0.0052 | 0.7670 ± 0.0101 | 0.8525 ± 0.0198 |
| ConsisGAD | TR40 | 0.9074 ± 0.0023 | 0.6451 ± 0.0132 | 0.7674 ± 0.0019 | 0.6077 ± 0.0020 | 0.7741 ± 0.0052 |
| LEX-GNN | TR30 | 0.9434 ± 0.0057 | 0.7758 ± 0.0236 | 0.8261 ± 0.0094 | 0.7030 ± 0.0136 | 0.8176 ± 0.0120 |
| ConsisGAD | TR30 | 0.9087 ± 0.0072 | 0.6445 ± 0.0338 | 0.7679 ± 0.0116 | 0.6093 ± 0.0173 | 0.7781 ± 0.0059 |
| LEX-GNN | TR20 | 0.9228 ± 0.0017 | 0.7099 ± 0.0101 | 0.7919 ± 0.0040 | 0.6468 ± 0.0086 | 0.7918 ± 0.0147 |
| ConsisGAD | TR20 | 0.9062 ± 0.0041 | 0.6360 ± 0.0157 | 0.7623 ± 0.0038 | 0.6007 ± 0.0072 | 0.7755 ± 0.0089 |
| LEX-GNN | TR10 | 0.8903 ± 0.0031 | 0.6037 ± 0.0398 | 0.7565 ± 0.0084 | 0.5871 ± 0.0134 | 0.7513 ± 0.0076 |
| ConsisGAD | TR10 | 0.8992 ± 0.0009 | 0.6107 ± 0.0099 | 0.7535 ± 0.0035 | 0.5848 ± 0.0075 | 0.7597 ± 0.0149 |

## Efficiency

| Model | Ratio | Epochs | Training seconds | Wall seconds | Inference seconds | Peak GPU MB | Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| LEX-GNN | TR40 | 256.67 ± 41.62 | 918.60 ± 196.40 | 1192.89 ± 254.93 | 1.3409 ± 0.0711 | 7581.00 ± 10.09 | 117676 ± 0 |
| ConsisGAD | TR40 | 100.00 ± 0.00 | 802.14 ± 12.04 | 839.67 ± 12.31 | 0.1568 ± 0.0704 | 5844.15 ± 0.62 | 109506 ± 0 |
| LEX-GNN | TR30 | 239.33 ± 21.57 | 680.95 ± 62.76 | 952.68 ± 87.23 | 1.5051 ± 0.0296 | 7587.42 ± 11.50 | 117676 ± 0 |
| ConsisGAD | TR30 | 100.00 ± 0.00 | 843.09 ± 45.69 | 880.94 ± 46.37 | 0.1098 ± 0.0037 | 5843.87 ± 0.25 | 109506 ± 0 |
| LEX-GNN | TR20 | 233.67 ± 22.28 | 417.05 ± 30.02 | 667.40 ± 43.69 | 1.4051 ± 0.0279 | 7581.42 ± 5.81 | 117676 ± 0 |
| ConsisGAD | TR20 | 100.00 ± 0.00 | 954.60 ± 140.52 | 994.16 ± 142.63 | 0.1142 ± 0.0026 | 5844.51 ± 0.43 | 109506 ± 0 |
| LEX-GNN | TR10 | 243.33 ± 26.76 | 244.54 ± 16.44 | 525.85 ± 30.68 | 1.4670 ± 0.0214 | 7573.36 ± 11.29 | 117676 ± 0 |
| ConsisGAD | TR10 | 100.00 ± 0.00 | 968.73 ± 117.45 | 1008.79 ± 118.64 | 0.1217 ± 0.0075 | 5844.14 ± 0.62 | 109506 ± 0 |

## Interpretation safeguards

- Hyperparameters were selected using validation data only.
- Test evaluation occurred once after training for each run.
- Model comparisons use the same dataset, fixed split registry and matched training seeds.
- The 95% intervals are descriptive because each group contains only three seeds.
- No statistical-significance claim is made.
