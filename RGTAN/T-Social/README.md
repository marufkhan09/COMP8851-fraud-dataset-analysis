# RGTAN × T-Social — Completed frozen-split benchmark

Final evaluation results are reported for 12 RGTAN × T-Social runs across four training ratios and three random seeds, conducted on a Vast.ai GPU. The results follow the Assessment 2 benchmark comparison format, using three-seed summary statistics and individual ratio–seed measurements.

## Source, data and protocol

- Original model: [AI4Risk/antifraud](https://github.com/AI4Risk/antifraud/tree/65e583f03ac5c5983e49cc35179fe7a65908b9c1), pinned commit `65e583f03ac5c5983e49cc35179fe7a65908b9c1`.
- Canonical T-Social graph: 5,781,065 nodes, 146,211,016 stored directed edges and 10 original node features.
- Canonical graph SHA-256: `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`.
- Frozen Assessment 2 split SHA-256: `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519`.
- Benchmark split seed: 2; final training seeds: 2, 42 and 72; training ratios: TR40, TR30, TR20 and TR10.
- Fixed validation and test partitions under the frozen-split assessment design.
- Training-only five-fold out-of-fold label contexts, exact sparse one-hop/two-hop risk statistics and training-only feature scaling.
- Independent Stage 2 preprocessing audit covered 16 saved artifacts and 18,777,751,032 candidate sparse two-hop walks.
- RGTAN T-Social adaptation: two graph layers with `MultiLayerNeighborSampler([10,10])`, batch size 1024, maximum 100 epochs and early-stopping patience 20.
- TR40 hyperparameter-tuning evidence includes 12 configurations varying hidden dimensions 32/64, base learning rates 0.001/0.003/0.005 and dropout pairs [0.2,0.1] and [0.3,0.15].
- Highest recorded TR40 tuning validation AUPRC: 0.995064, Trial 7, using `hidden_dim=64`, `drop=[0.2,0.1]` and `base_lr=0.001`.
- Model-selection protocol uses validation AUPRC for checkpoint selection and validation macro-F1 for classification-threshold selection. Held-out test metrics are reported separately.
- The final ratio–seed metrics below originate from the separately reported Vast.ai GPU experiment, while the published TR40 tuning evidence documents the earlier I1 run.
- Scientific distinction: bounded-neighbour training is an adapted RGTAN implementation and is not an author-identical full-neighbour reproduction.

## Held-out test results

Three-seed mean ± sample standard deviation (`ddof=1`), calculated from the 12 individual final-run results reported below.

| Training ratio | Test AUPRC | Test AUROC | Test macro-F1 |
|---|---:|---:|---:|
| TR40 | 0.991987 ± 0.001031 | 0.912779 ± 0.001603 | 0.918007 ± 0.003379 |
| TR30 | 0.994722 ± 0.000938 | 0.911254 ± 0.006561 | 0.919798 ± 0.006645 |
| TR20 | 0.991579 ± 0.001194 | 0.911101 ± 0.005759 | 0.915784 ± 0.009100 |
| TR10 | 0.993712 ± 0.001716 | 0.918599 ± 0.001559 | 0.919018 ± 0.001425 |

## Individual final held-out results

The following values reproduce the 12 ratio–seed results supplied from the separate Vast.ai GPU experiment.

| Ratio | Seed | Test AUPRC | Test AUROC | Test macro-F1 |
|---|---:|---:|---:|---:|
| TR40 | 2 | 0.993053 | 0.914156 | 0.914253 |
| TR40 | 42 | 0.991912 | 0.911019 | 0.918964 |
| TR40 | 72 | 0.990996 | 0.913161 | 0.920804 |
| TR30 | 2 | 0.995064 | 0.908492 | 0.924454 |
| TR30 | 42 | 0.995441 | 0.906525 | 0.912189 |
| TR30 | 72 | 0.993661 | 0.918744 | 0.922752 |
| TR20 | 2 | 0.992948 | 0.906497 | 0.926249 |
| TR20 | 42 | 0.990757 | 0.917558 | 0.909729 |
| TR20 | 72 | 0.991031 | 0.909248 | 0.911375 |
| TR10 | 2 | 0.995157 | 0.919995 | 0.919130 |
| TR10 | 42 | 0.991816 | 0.918885 | 0.917541 |
| TR10 | 72 | 0.994164 | 0.916916 | 0.920384 |

## Audit and reproducibility

- [Final per-seed results](final_evaluation_20261026/results/heldout_test_per_seed.csv): the 12 separately reported ratio–seed metrics, transcribed to six decimal places.
- [Final three-seed summary](final_evaluation_20261026/results/heldout_test_summary.csv): arithmetic mean and sample standard deviation (`ddof=1`) computed from the per-seed rows, not from the previously supplied screenshot of a summary.
- [Final-run provenance](final_evaluation_20261026/evidence/metric_provenance.json): records which results came from the separate Vast.ai run and what verification is available.
- [Final evaluation evidence manifest](final_evaluation_20261026/PUBLIC_SHA256_MANIFEST.json): fingerprints of the new lightweight published files.
- `tuning_evidence_20261025/results/tr40_tuning_trials.csv`: original I1 TR40 tuning configurations, recorded validation AUPRC, training loss, runtime and selected epochs.
- `tuning_evidence_20261025/results/epoch_level_validation.csv`: 402 saved epoch-level training and validation measurements.
- `tuning_evidence_20261025/results/ratio_seed_evaluation_protocol.csv`: four-ratio, three-seed assessment design.
- `tuning_evidence_20261025/scripts/`: recorded Stage 2 preprocessing and Stage 3 training scripts.
- `tuning_evidence_20261025/evidence/`: risk-feature integrity checks, model tuning records and validation-provenance evidence.
- `tuning_evidence_20261025/PUBLIC_SHA256_MANIFEST.json`: SHA-256 fingerprints of the earlier I1 tuning evidence.
- Final held-out summary statistics have been independently recalculated from the 12 supplied per-seed metric rows using sample standard deviation (`ddof=1`).
- Raw graph data, frozen split binaries, risk-feature matrices, model checkpoints, Python/CUDA environments and private prediction arrays are excluded from GitHub.
- Private archives preserve the exported I1 tuning evidence and recovery artifacts. The separate Vast.ai final-run metrics are documented independently of those tuning records.
