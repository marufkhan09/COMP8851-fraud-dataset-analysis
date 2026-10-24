# RGTAN × T-Social — Validation-only TR40 tuning in progress

**Progress:** **6/12 TR40 tuning trials completed**, with **trial 7 in progress** at the most recent I1 check (epoch 31). The frozen 12-run training matrix (TR40/TR30/TR20/TR10 × seeds 2, 42, 72) and once-only held-out test evaluations have **not started**. **No held-out test results are available.** The source RGTAN model is pinned and unchanged; the T-Social experiment uses an explicitly adapted bounded-neighbour training procedure, not an author-identical reproduction.

## Provenance

- Original RGTAN author: `AI4Risk/antifraud`, pinned commit `65e583f03ac5c5983e49cc35179fe7a65908b9c1`; original author model source remains unchanged.
- Canonical T-Social graph: **5,781,065 nodes**, **146,211,016 stored directed edges**, **10 node features**; original source SHA-256 `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`.
- Frozen Assessment 2 split SHA-256 `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519`; split seed **2**, ratios **TR40/TR30/TR20/TR10**, final model seeds **2, 42, 72**; fixed validation and test memberships.
- Five-fold training-only out-of-fold label contexts and exact sparse one-hop/two-hop risk features. Independent Stage 2 checks covered **16 saved artifacts**, label masking, crossfit scaling and sampled exact two-hop computations; **18,777,751,032** candidate binary-CSR two-hop walks were recorded. Validation/test labels were excluded from training risk-feature fitting.
- T-Social computational adaptation: two RGTAN graph layers with `MultiLayerNeighborSampler([10,10])`, batch size **1024**, rather than original full-neighbour propagation. Up to **100 epochs** and early stopping patience **20** per trial; checkpoints selected using validation AUPRC. This is **not** an author-identical experiment; supervisor methodological approval has not been recorded.
- TR40 tuning grid: **12 configurations**, hidden dimensions `{32,64}`, base learning rates `{0.001,0.003,0.005}` and dropout pairs `[0.2,0.1]` and `[0.3,0.15]`. The winning configuration has **not** been independently approved and frozen. No test outcomes have been used for model selection.

## Validation-only TR40 tuning results

The results below are **validation AUPRC**, not held-out test results. Six tuning trials have completed; trial 7 was still running at the most recent I1 terminal check. Trial 2's near-perfect AUPRC and trial 7's provisional near-perfect AUPRC require independent feature-provenance, label-isolation and metric checks before winner selection.

| Trial | Training progress | Epochs reached | Best validation AUPRC | Evidence |
|---|---|---:|---:|---|
| 01 | Complete | 43 | 0.584989 | Checkpoint, saved scores and history hashes passed |
| 02 | Complete | 22 | 0.991912 | Hashes passed; unusually high score requires independent review |
| 03 | Complete | 26 | 0.477618 | Checkpoint, saved scores and history hashes passed |
| 04 | Complete | 39 | 0.576455 | Checkpoint, saved scores and history hashes passed |
| 05 | Complete | 37 | 0.519692 | Checkpoint, saved scores and history hashes passed |
| 06 | Complete | 31 | 0.446338 | Completion verified; hashes not rechecked in the most recent audit |
| 07 | In progress | 31 | 0.995064 | Provisional best at epoch 16; early-stopping stale count 15/20 |
| 08–12 | Pending | — | — | No completed trial results |

## Held-out test results across three final seeds

**Not available.** Tuning remains in progress; the frozen final-model matrix and once-only held-out evaluations have not been completed. The following is a **planned evaluation matrix**, not observed performance.

| Ratio | Planned final seeds | Test AUPRC | Test AUROC | Test macro-F1 |
|---|---|---|---|---|
| TR40 | 2, 42, 72 | Not evaluated | Not evaluated | Not evaluated |
| TR30 | 2, 42, 72 | Not evaluated | Not evaluated | Not evaluated |
| TR20 | 2, 42, 72 | Not evaluated | Not evaluated | Not evaluated |
| TR10 | 2, 42, 72 | Not evaluated | Not evaluated | Not evaluated |

## Individual once-only held-out test results

**Not available:** no final ratio/seed test results or frozen validation-selected classification thresholds have been recorded for publication. Validation-only tuning scores must not be represented as held-out test outcomes.

## Public evidence and scripts

- `README.md` records the most recently supplied I1 tuning progress. `status.json` still describes an earlier stage and is **not updated** by this README-only publication.
- Original pinned model source, canonical graph/split hashes and Stage 2 risk-feature audit are retained in the private I1 experiment; binary graph data, split arrays and model checkpoints are not published here.
- I1 Stage 3 runner: `RGTAN_TSOCIAL_I1_STAGE3_GRID_V2_20261023/stage3_tr40_rgtan_grid_v2.py`; its `--mode all` operation covers tuning and winner freezing, **not** the twelve Stage 4 final trainings or Stage 5 held-out scoring.
- Remaining work: complete all 12 validation-only trials, independently investigate the trial-2/trial-7 scoring anomalies, select and freeze the winner using validation only, prepare ratio-specific training-only risk contexts, train and audit the twelve ratio/seed final models, and evaluate each frozen checkpoint exactly once on the held-out test set.

**Withheld privately:** original graph, frozen split data, training-only risk-feature arrays, model/optimizer checkpoints and node-level predictions. The GitHub folder is a public progress/evidence record, not a complete private training archive.
