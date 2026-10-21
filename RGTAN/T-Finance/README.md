# RGTAN × T-Finance — Adapted benchmark completed

**Status:** COMPLETE — 12/12 TR40 validation tuning trials, 12/12 frozen final training runs and **12/12 one-time held-out test evaluations**. This is the completed **bounded-neighbour RGTAN adaptation**, not an author-identical full-neighbour baseline. This GitHub record includes public evidence, not checkpoint binaries or the full private dataset.

## Model, dataset and methodology

- Pinned author source: [AI4Risk/antifraud](https://github.com/AI4Risk/antifraud/tree/65e583f03ac5c5983e49cc35179fe7a65908b9c1) commit `65e583f03ac5c5983e49cc35179fe7a65908b9c1`.

- Canonical T-Finance graph: 39,357 nodes, 10 features, 42,445,086 stored directed edges; source SHA-256 `b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700`.

- Frozen Assessment 2 nested split SHA-256 `de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1`; split seed 2; ratios **TR40/TR30/TR20/TR10**, final model seeds **2/42/72**; validation and test membership fixed. Test set contains **15,743 nodes, 722 fraud cases**.

- Train-only five-fold OOF label/risk features, exact two-hop risk statistics, training-only scaling. Neither validation nor held-out test target labels were used to construct training risk statistics.

- TR40 12-trial validation AUPRC grid; frozen winner **trial 7**, configuration `hidden_dim=64, drop=[0.2,0.1], base_lr=0.001`; validation chose checkpoints and macro-F1 thresholds. No test-set model selection or retuning.

- **Important adaptation:** the author full-neighbour two-layer propagation produced impractical dense expansion (an individual seed can exceed 33 million two-hop edges). This run uses `MultiLayerNeighborSampler([10,10])`, batch 1024, for both training and inference. **This is a computationally complete adapted RGTAN experiment, not an exact full-neighbour reproduction.** No supervisor approval for an author-identical comparison is claimed.

## Completed adapted held-out test performance

Mean ± **sample standard deviation** (`ddof=1`) across seeds 2, 42, 72. The values below are **test-set**, unlike the preserved older Stage 5 *validation-only* records.

| Training ratio | Test AUPRC | Test AUROC | Test macro-F1 |
|---|---:|---:|---:|
| TR40 | 0.859164 ± 0.033611 | 0.960336 ± 0.006975 | 0.896537 ± 0.020238 |
| TR30 | 0.880378 ± 0.027470 | 0.964437 ± 0.009835 | 0.911459 ± 0.014398 |
| TR20 | 0.867179 ± 0.033257 | 0.960498 ± 0.005375 | 0.905148 ± 0.014943 |
| TR10 | 0.818031 ± 0.059034 | 0.940285 ± 0.016038 | 0.880780 ± 0.034437 |

### Individual held-out test scores

| Ratio | Seed | Test AUPRC | Test AUROC | Test macro-F1 |
|---|---:|---:|---:|---:|
| TR40 | 2 | 0.857084 | 0.958319 | 0.893671 |
| TR40 | 42 | 0.826641 | 0.954591 | 0.877885 |
| TR40 | 72 | 0.893767 | 0.968097 | 0.918056 |
| TR30 | 2 | 0.848728 | 0.953233 | 0.894933 |
| TR30 | 42 | 0.894367 | 0.968434 | 0.918152 |
| TR30 | 72 | 0.898037 | 0.971644 | 0.921292 |
| TR20 | 2 | 0.884905 | 0.963616 | 0.912659 |
| TR20 | 42 | 0.887818 | 0.963587 | 0.914846 |
| TR20 | 72 | 0.828814 | 0.954292 | 0.887940 |
| TR10 | 2 | 0.878159 | 0.957612 | 0.914783 |
| TR10 | 42 | 0.815777 | 0.937283 | 0.881634 |
| TR10 | 72 | 0.760155 | 0.925961 | 0.845924 |

## Reproducibility and interpretation

- `scripts/` has original I2 preflight, crossfit, tuning, training and Stage 6 frozen-model test-evaluation code.
- `risk_features/` retains four SHA-256-referenced risk-feature manifests; actual input feature arrays remain private.
- `results/tuning/` preserves the 12 validation-only candidate tuning records and frozen winner; `results/finals/` preserves all 12 frozen training records **and newly published held-out test metrics, test score arrays and one-time scoring claims**.
- `evidence/` includes the independent Stage 6 test-metric recomputation audit, Stage 6 execution log, dataset/split provenance and SHA-256 publication manifest.
- The **original Stage 5 candidate audit and its publication file manifest** are historical records of the snapshot *before* Stage 6 test scoring. They are intentionally preserved, including their original “test not scored” assertions; these refer to that earlier stage, not the completed benchmark. The newer Stage 7 manifest covers the incremental final-publication files.
- Checkpoint SHA-256 hashes are documented, but checkpoint weights (`.pt`), source graph, split NPZ, crossfit NPZ and Python environment are **not** committed to this public repo. Keep the private I2 files if future inference/reproduction is needed.

**Interpretation:** The benchmark is computationally complete with an explicit sampling adaptation. Direct comparison with author-identical full-neighbour results requires methodological consideration and, if relevant to the assessment, supervisor sign-off.
