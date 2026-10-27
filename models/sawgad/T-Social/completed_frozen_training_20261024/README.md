# SAWGAD × T-Social — Completed frozen-configuration training evidence

**Training evidence complete:** 12 TR40 validation-only tuning configurations and 12 ratio/seed final trainings. **Held-out test evaluations: 0.** The original author implementation has been retained at the pinned commit, but the sampled-neighbourhood training adapter and limited update schedule differ from author full-graph training. This is not an author-identical reproduction and should not be represented as a fully approved common-protocol benchmark.

## Provenance

- [SAWGAD pinned original source](https://github.com/yj-zhou/SAWGAD/tree/64d6d531a2f03549f0bfbe461092beedd6f70bd2) (`64d6d531a2f03549f0bfbe461092beedd6f70bd2`).
- T-Social: 5,781,065 nodes, 146,211,016 stored directed edges; source SHA-256 `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`.
- Frozen nested split SHA-256 `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519`; TR40/TR30/TR20/TR10; seeds 2/42/72.
- Winner: tuning trial **09**, validation AUPRC **0.167209**; hidden_dim=64, dropout=0, LR=0.005.
- Model selection was validation-only. No held-out test labels were scored.

## Full-validation results (three seeds)

| Ratio | Mean ± sample SD AUPRC | Seed 2 | Seed 42 | Seed 72 |
|---|---:|---:|---:|---:|
| TR40 | 0.171703 ± 0.002465 | 0.170170 | 0.174546 | 0.170391 |
| TR30 | 0.171425 ± 0.005535 | 0.173258 | 0.175812 | 0.165206 |
| TR20 | 0.173948 ± 0.005635 | 0.180091 | 0.169018 | 0.172736 |
| TR10 | 0.176150 ± 0.002892 | 0.172999 | 0.176769 | 0.178683 |

These are **validation AUPRC**, not held-out test performance. See [per-seed results](results/full_validation_per_seed.csv) and [tuning results](results/tr40_tuning_trials.csv).

## Evidence status

- All 24 saved completion records, checkpoints, source script hashes and validation histories passed the independent **metadata** audit.
- A separate checkpoint-based full-validation replay found discrepancies greater than 1e-5 in all 13 evaluated checkpoints. The scores require further evaluator-reproducibility investigation; see [replay evidence](evidence/pretest_validation_replay/checkpoint_validation_replay.json). This is not a passed independent prediction-score reproduction.
- No held-out test metrics are available; final test evaluation remains gated.

## Public scripts and evidence

- [scripts/stage3](scripts/stage3) and [scripts/stage4](scripts/stage4): pinned split-adapted training pipeline.
- [evidence/stage3_tuning](evidence/stage3_tuning): frozen winner, 12 trial specs, histories and completion reports.
- [evidence/stage4_finals](evidence/stage4_finals): 12 frozen-configuration validation-only training records.
- [results](results): per-seed full-validation AUPRC, three-seed summaries, tuning and replay comparisons.
- [PUBLIC_SHA256_MANIFEST.json](PUBLIC_SHA256_MANIFEST.json): SHA-256 of published files.

**Not published:** dataset, split archive, model checkpoints, optimizer states and node-level predictions. The private archive is retained outside GitHub.
