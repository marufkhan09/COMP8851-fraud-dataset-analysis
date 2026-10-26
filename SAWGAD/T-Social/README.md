# SAWGAD × T-Social — Completed frozen-split benchmark

Final evaluation results are reported for **12 SAWGAD × T-Social runs** across four training ratios and three training seeds, from a separately reported Vast.ai RTX A6000 48-GB GPU experiment. The original SAWGAD implementation is pinned; the T-Social training procedure uses a bounded-neighbour adaptation rather than full-graph author-identical training.

## Source, data and protocol

- Original source: [yj-zhou/SAWGAD](https://github.com/yj-zhou/SAWGAD/tree/64d6d531a2f03549f0bfbe461092beedd6f70bd2); pinned author commit `64d6d531a2f03549f0bfbe461092beedd6f70bd2`.
- Canonical T-Social graph: **5,781,065 nodes**, **146,211,016 stored directed edges**, 10 node features. Source SHA-256: `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`.
- Frozen split SHA-256: `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519`; split seed 2; four nested training ratios TR40, TR30, TR20, TR10; final seeds 2, 42 and 72; fixed validation and test memberships.
- Archived TR40 validation-tuning winner: trial 09, best full-validation AUPRC 0.167209; hidden dimension 64, learning rate 0.005, dropout 0.0. The tuning and frozen training histories are retained separately in `completed_frozen_training_20261024/`.
- Recorded training adaptation: synthetic-anomaly warm-up, one-hop inbound fanout-10 sampling, one encoder layer, four attention heads; 32 warm-up steps and 16 rounds of 1,024 sampled optimiser updates for each archived final training. Sampled rounds must not be equated with full-dataset epochs.
- Reported final test metrics below originate from the separate Vast.ai GPU table; they are not copied from or calculated from the archived validation-tuning values.
- The 12 per-seed test values are source-reported. Their aggregate arithmetic is independently recomputed, but the originating separate GPU's ordered test-label arrays, prediction scores and evaluation logs were not supplied for an independent metric-level replay.

## Held-out test results

Three-seed **mean ± sample standard deviation (`ddof=1`)** calculated directly from the 12 individual results below.

| Training ratio | Test AUPRC | Test AUROC | Test macro-F1 |
|---|---:|---:|---:|
| TR40 | 0.951182 ± 0.043547 | 0.951352 ± 0.043427 | 0.954111 ± 0.003589 |
| TR30 | 0.959695 ± 0.025889 | 0.959869 ± 0.025794 | 0.959290 ± 0.008292 |
| TR20 | 0.930748 ± 0.040264 | 0.931043 ± 0.040269 | 0.950947 ± 0.007389 |
| TR10 | 0.985270 ± 0.012147 | 0.985404 ± 0.012212 | 0.953822 ± 0.003892 |

## Individual final held-out results

| Ratio | Seed | Test AUPRC | Test AUROC | Test macro-F1 |
|---|---:|---:|---:|---:|
| TR40 | 2 | 0.910770 | 0.911194 | 0.956500 |
| TR40 | 42 | 0.945476 | 0.945424 | 0.955849 |
| TR40 | 72 | 0.997301 | 0.997439 | 0.949983 |
| TR30 | 2 | 0.935728 | 0.935880 | 0.952384 |
| TR30 | 42 | 0.987152 | 0.987152 | 0.968487 |
| TR30 | 72 | 0.956206 | 0.956576 | 0.956999 |
| TR20 | 2 | 0.908091 | 0.908259 | 0.945996 |
| TR20 | 42 | 0.906918 | 0.907331 | 0.947405 |
| TR20 | 72 | 0.977236 | 0.977539 | 0.959441 |
| TR10 | 2 | 0.999279 | 0.999497 | 0.956996 |
| TR10 | 42 | 0.977669 | 0.977959 | 0.954991 |
| TR10 | 72 | 0.978863 | 0.978755 | 0.949479 |

## Audit and reproducibility

- [Final per-seed CSV](final_evaluation_20261027/results/heldout_test_per_seed.csv): all 12 reported test rows.
- [Final summary CSV](final_evaluation_20261027/results/heldout_test_summary.csv): recomputed mean and sample SD (`ddof=1`) for each ratio and metric.
- [Source screenshot](final_evaluation_20261027/evidence/source_test_results.png): immutable image supplied with the separate-GPU evaluation table.
- [Final-run metric provenance](final_evaluation_20261027/evidence/metric_provenance.json) and [release SHA-256 manifest](final_evaluation_20261027/PUBLIC_SHA256_MANIFEST.json) document traceability and verification scope.
- [Earlier validation/tuning evidence](completed_frozen_training_20261024/README.md) retains the original 12 TR40 tuning trials, 12 frozen training histories, source scripts, SHA-256 checks and checkpoint-validation replay evidence, without alteration.
- Test Fraud-F1 and G-Mean are not available in the source table; these require matching thresholded predictions or confusion-matrix/class-recall measurements and cannot be uniquely recovered from AUPRC, AUROC and macro-F1.
- The original graph, private split binaries, model checkpoints and score arrays are not distributed in Git.
