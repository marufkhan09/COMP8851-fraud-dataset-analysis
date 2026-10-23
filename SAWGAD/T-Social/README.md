# SAWGAD × T-Social — Frozen-split assessment benchmark (training complete; pretest review pending)

**Training complete:** **12/12 TR40 validation-only tuning trials** and **12/12 frozen final trainings** (TR40/TR30/TR20/TR10 × seeds 2, 42, 72). All 24 completion records, checkpoint fingerprints and validation histories passed the independent **metadata/SHA-256** audit. **Held-out test evaluations: 0/12; no held-out test results are available.** The separate independent checkpoint-validation replay **did not pass** the specified score-agreement tolerance and requires review before held-out evaluation. The original author source is retained at its pinned commit, but this T-Social training adapter is not an author-identical full-graph reproduction.

## Provenance

- Original SAWGAD author: `yj-zhou/SAWGAD` (pinned commit `64d6d531a2f03549f0bfbe461092beedd6f70bd2`); original author source was retained separately.
- Canonical T-Social graph: **5,781,065 nodes**, **146,211,016 stored directed edges**, **10 node features**; original source SHA-256 `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`.
- Frozen Assessment 2 nested split SHA-256 `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519`; split seed **2**, ratios **TR40/TR30/TR20/TR10**, final training seeds **2, 42, 72**, fixed validation and test membership.
- Full frozen validation: **1,156,213 nodes**, including **34,856 fraud nodes**. TR40/seed-2 tuning used **full-validation AUPRC only**; best final-run checkpoints also selected using full-validation AUPRC. Test labels and scores were not used in training or model selection, and the held-out test has not been evaluated.
- Frozen tuning winner **trial 9**, full-validation AUPRC **0.167209**; `hidden_dim=64`, `full_lr=0.005`, `drop_rate=0.0`. The winning configuration was reused in the 12 final trainings.
- Architecture and evaluation distinction: attention-based SAWGAD with synthetic-anomaly warm-up and supervised training, adapted to **one-hop bounded inbound neighbour sampling (fanout 10)** with one encoder layer and four attention heads. The recorded run used 32 warm-up updates, 8 rounds × 1,024 supervised updates for each tuning trial, and 16 rounds × 1,024 supervised updates for each final training. This is **not** the author full-graph procedure or a common full-dataset-epoch schedule; supervisor methodological approval has not been recorded.

## Validation results across three final seeds

These are the **best full-validation AUPRC** values, **not held-out test** results. Each mean ± standard deviation uses **sample SD (`ddof=1`)** across the three training seeds. Test AUROC and test macro-F1 are not reported because they have not been evaluated.

| Ratio | Validation AUPRC mean ± sample SD | Seed 2 | Seed 42 | Seed 72 |
|---|---:|---:|---:|---:|
| TR40 | 0.171703 ± 0.002465 | 0.170170 | 0.174546 | 0.170391 |
| TR30 | 0.171425 ± 0.005535 | 0.173258 | 0.175812 | 0.165206 |
| TR20 | 0.173948 ± 0.005635 | 0.180091 | 0.169018 | 0.172736 |
| TR10 | 0.176150 ± 0.002892 | 0.172999 | 0.176769 | 0.178683 |

Original per-seed validation scores and checkpoint rounds are in `completed_frozen_training_20261024/results/full_validation_per_seed.csv`. Tuning results are in `completed_frozen_training_20261024/results/tr40_tuning_trials.csv`. **These must not be described as held-out test performance.**

## Individual final validation results

All 12 final runs used the **frozen trial-9 hyperparameters**. The selected round is the checkpoint with the highest recorded full-validation AUPRC. This table reports **validation**, not once-only test evaluation, results.

| Ratio | Seed | Validation AUPRC | Best checkpoint round | Test evaluated |
|---|---:|---:|---:|---|
| TR40 | 2 | 0.170170 | 11 | No |
| TR40 | 42 | 0.174546 | 15 | No |
| TR40 | 72 | 0.170391 | 11 | No |
| TR30 | 2 | 0.173258 | 16 | No |
| TR30 | 42 | 0.175812 | 14 | No |
| TR30 | 72 | 0.165206 | 14 | No |
| TR20 | 2 | 0.180091 | 15 | No |
| TR20 | 42 | 0.169018 | 14 | No |
| TR20 | 72 | 0.172736 | 14 | No |
| TR10 | 2 | 0.172999 | 13 | No |
| TR10 | 42 | 0.176769 | 7 | No |
| TR10 | 72 | 0.178683 | 9 | No |

## Held-out test results

**Not available.** None of the 12 frozen models has a recorded once-only held-out test evaluation. Test AUPRC, test AUROC, test macro-F1 and test-selected decision thresholds must not be inferred from the validation-only tables. No test-time checkpoint or threshold selection is permitted.

## Public evidence and scripts

- `completed_frozen_training_20261024/scripts/stage3/sawgad_job.py`: training/validation worker for the recorded experiment.
- `completed_frozen_training_20261024/scripts/stage4/sawgad_pipeline.py`: 12-trial tuning, winner freezing and 12 final ratio/seed training orchestration.
- `completed_frozen_training_20261024/evidence/stage3_tuning/`: all 12 TR40 tuning records and the frozen trial-9 winner.
- `completed_frozen_training_20261024/evidence/stage4_finals/`: all 12 training/validation histories, run specifications, recorded protocol and completion evidence.
- `completed_frozen_training_20261024/results/full_validation_summary.csv` and `results/full_validation_per_seed.csv`: three-seed aggregates and individual validation results; `results/tr40_tuning_trials.csv` stores the 12 tuning outcomes.
- `completed_frozen_training_20261024/evidence/pretest_validation_replay/checkpoint_validation_replay.json`: independent checkpoint replay found **13/13** score differences beyond tolerance `0.00001`; absolute AUPRC differences range **0.000122–0.001805**. **Replay not passed**, despite metadata and hash integrity passing.
- `completed_frozen_training_20261024/PUBLIC_SHA256_MANIFEST.json`: manifest of published sanitized evidence files. `SAWGAD/T-Social/status.json` records this run as training complete, test scoring not performed and replay requiring review.

**Withheld privately:** original graph, frozen split `.npz`, model and optimizer `.pt` checkpoints, node-level predictions and Python environment. Preserve and independently verify those private artifacts before destroying I2. Next steps are to reconcile checkpoint replay, complete the independent pretest/methodology gate, then perform once-only held-out test evaluation and publish verified test metrics; **none of those test metrics is yet available**.
