# SAWGAD × T-Social — Frozen validation-only training completed

**Status as recorded on 24 October 2026:** the canonical T-Social source and frozen Assessment 2 splits were verified. All **12/12 TR40 validation-only tuning trials** and **12/12 frozen final training runs** (TR40/TR30/TR20/TR10 × seeds 2, 42, 72) completed on I2. All 24 runs passed independent saved-artifact metadata and SHA-256 integrity checks. An independent checkpoint-based validation replay remains **unresolved**. **No held-out test evaluation has been performed, and no held-out test performance is reported.**

## Provenance and reproducibility

- Original SAWGAD source: `yj-zhou/SAWGAD`, pinned commit `64d6d531a2f03549f0bfbe461092beedd6f70bd2`; original author repository retained separately.
- Canonical T-Social graph: **5,781,065 nodes**, **146,211,016 stored directed edges**, **10 node features**; source SHA-256 `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`.
- Frozen nested split SHA-256 `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519`; split seed **2**; training ratios **TR40/TR30/TR20/TR10**; final model seeds **2, 42, 72**. Validation/test membership is fixed.
- Full frozen validation comprises **1,156,213 nodes**, of which **34,856** have the fraud label. **Validation AUPRC** selects the tuning winner and each run's best saved checkpoint; the held-out test was not scored.
- Architecture: SAWGAD's attention-based graph encoder, synthetic-anomaly objectives, two-phase synthetic warm-up and supervised fraud scoring. The T-Social adapter uses **one-hop bounded inbound sampling (fanout 10)**, a one-layer encoder (4 attention heads; winner hidden dimension 64), and 32 synthetic warm-up updates followed by 8 × 1,024 supervised updates per tuning trial or 16 × 1,024 per final run. These differ from the author full-graph implementation and a full-dataset-epoch training schedule; no author-identical reproduction or methodological approval is claimed.

## Validation-only tuning — available evidence

- **All 12 TR40, seed-2 candidate configurations completed.** Grid: hidden dimensions `{32,64}`, learning rates `{0.001,0.003,0.005}` and dropout rates `{0,0.25}`.
- **Frozen winner: trial 9**, validation AUPRC **0.167209** at round 8; hyperparameters `hidden_dim=64`, `full_lr=0.005`, `drop_rate=0.0`. The same frozen configuration was used for all 12 final runs.

| Trial | Hidden dimension | Learning rate | Dropout | Best validation AUPRC |
|---|---:|---:|---:|---:|
| 01 | 32 | 0.001 | 0.00 | 0.141124 |
| 02 | 32 | 0.003 | 0.00 | 0.156752 |
| 03 | 32 | 0.005 | 0.00 | 0.165589 |
| 04 | 32 | 0.001 | 0.25 | 0.138848 |
| 05 | 32 | 0.003 | 0.25 | 0.143647 |
| 06 | 32 | 0.005 | 0.25 | 0.144503 |
| 07 | 64 | 0.001 | 0.00 | 0.160432 |
| 08 | 64 | 0.003 | 0.00 | 0.164567 |
| **09** | **64** | **0.005** | **0.00** | **0.167209** |
| 10 | 64 | 0.001 | 0.25 | 0.157340 |
| 11 | 64 | 0.003 | 0.25 | 0.157255 |
| 12 | 64 | 0.005 | 0.25 | 0.149725 |

## Frozen final training — full-validation results

The values below are **best full-validation AUPRC**, not held-out test metrics. The mean and **sample standard deviation** (`ddof=1`) use the three independent training seeds, consistent with the T-Social frozen-training archive.

### Three-seed summary

| Training ratio | Validation AUPRC (mean ± sample SD) | Seed 2 | Seed 42 | Seed 72 |
|---|---:|---:|---:|---:|
| TR40 | 0.171703 ± 0.002465 | 0.170170 | 0.174546 | 0.170391 |
| TR30 | 0.171425 ± 0.005535 | 0.173258 | 0.175812 | 0.165206 |
| TR20 | 0.173948 ± 0.005635 | 0.180091 | 0.169018 | 0.172736 |
| TR10 | 0.176150 ± 0.002892 | 0.172999 | 0.176769 | 0.178683 |

### Individual final-run evidence

| Ratio | Seed | Best validation AUPRC | Selected checkpoint round |
|---|---:|---:|---:|
| TR40 | 2 | 0.170170 | 11 |
| TR40 | 42 | 0.174546 | 15 |
| TR40 | 72 | 0.170391 | 11 |
| TR30 | 2 | 0.173258 | 16 |
| TR30 | 42 | 0.175812 | 14 |
| TR30 | 72 | 0.165206 | 14 |
| TR20 | 2 | 0.180091 | 15 |
| TR20 | 42 | 0.169018 | 14 |
| TR20 | 72 | 0.172736 | 14 |
| TR10 | 2 | 0.172999 | 13 |
| TR10 | 42 | 0.176769 | 7 |
| TR10 | 72 | 0.178683 | 9 |

**Descriptive observation:** TR10 has the highest three-seed mean validation AUPRC, whereas the single best run is TR20/seed 2. This is an observed validation pattern, not evidence that lower training fractions generally outperform larger ones; three seeds and the current training design do not establish that conclusion.

## Evidence and verification

- **Integrity passed:** all **12/12 tuning records**, **12/12 final records**, frozen trial-9 selection, public training-script hashes, histories and checkpoint hashes matched the saved records.
- **Checkpoint replay requires review:** independent re-evaluation of the trial-9 checkpoint and all 12 final checkpoints produced absolute validation AUPRC discrepancies of **0.000122–0.001805**; **13/13** exceeded the replay tolerance of `0.00001`. This is not a passing independent prediction-reproduction result.
- Public evidence is retained in `completed_frozen_training_20261024/`: `results/tr40_tuning_trials.csv`, `results/full_validation_per_seed.csv`, `results/full_validation_summary.csv`, `results/validation_replay_differences.csv`, `evidence/stage3_tuning/`, `evidence/stage4_finals/`, `evidence/pretest_validation_replay/`, `scripts/stage3/`, `scripts/stage4/`, and `PUBLIC_SHA256_MANIFEST.json`.
- The original graph, split NPZ, model/optimizer checkpoints, environment files and node-level prediction arrays remain private and are not committed to this repository.

## Planned remaining phases

1. Investigate the validation replay differences and independently reconcile checkpoint evaluation against the recorded scoring procedure, without accessing test labels.
2. Complete the pretest integrity, label-isolation, methodology and private-checkpoint retention checks; determine any decision thresholds from validation alone.
3. Only after those gates, perform a frozen, once-only held-out evaluation for each of the 12 final ratio/seed runs.
4. Independently verify saved held-out scores and metrics, then update this root README with **test**, rather than validation, results.

**Current scope:** 24 completed training runs and validation-only results; **0 held-out test evaluations**. No held-out AUROC, F1, fraud precision/recall or other test metrics are available at this stage.
