# EOGFD × T-Social — Completed frozen-split assessment benchmark

**Complete:** 12/12 TR40 validation-only tuning trials, 12/12 frozen final trainings (TR40/TR30/TR20/TR10 × seeds 2, 42, 72), and exactly **12/12 once-only held-out test evaluations**. The independently audited test records were created before publication. **Not an author-identical reproduction:** the original author network source was not modified, but process-local memory optimizations, streamed evaluation and a unified assessment protocol were used; supervisor approval has not been recorded.

## Provenance

- Original EOGFD author: [BERA-wx/EOGFD](https://github.com/BERA-wx/EOGFD/tree/a4726f173bc8c923109ddcb2786f4d9c5206ae7f) (pinned commit `a4726f173bc8c923109ddcb2786f4d9c5206ae7f`).
- Canonical T-Social graph: **5,781,065 nodes**, **146,211,016 stored directed edges**, 10 node features; original source SHA-256 `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`.
- Frozen Assessment 2 split SHA-256 `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519`. Test labels were not used in training, tuning, early stopping or threshold selection.
- Frozen tuning winner **trial 9**, validation AUPRC **0.859637**; lr=0.01, population=10, mutation_rate=0.05, hidden_dim=10, order=8, beta_p=1, lamb=0.0, weight_decay=0.0.
- Best model and threshold were selected on validation **before** held-out test access. Fixed test evaluation RNG seed 100921. The held-out score arrays were subsequently checked using independent stored-score metric recomputation **without new inference**.

## Held-out test results across three final seeds

| Ratio | Test AUPRC mean ± sample SD | Test AUROC mean | Test macro-F1 mean |
|---|---:|---:|---:|
| TR40 | 0.859576 ± 0.005528 | 0.966815 | 0.911035 |
| TR30 | 0.852670 ± 0.017457 | 0.965600 | 0.907367 |
| TR20 | 0.859530 ± 0.008795 | 0.966758 | 0.910545 |
| TR10 | 0.860037 ± 0.002237 | 0.966461 | 0.911692 |


Per-seed held-out metrics are in [results/heldout_test_per_seed.csv](results/heldout_test_per_seed.csv), along with exact per-run validation thresholds, provenance and test-once claims in [evidence](evidence/). These are **test**, not validation, metrics. Do not use the test split to select or modify a model.

## Public evidence and scripts

- [scripts/stage3](scripts/stage3): validation-only tuning and runtime optimization code.
- [scripts/stage4](scripts/stage4): frozen training of all 12 ratio/seed combinations.
- [scripts/stage5](scripts/stage5): write-once held-out evaluation and frozen thresholds.
- [evidence/stage3_tuning](evidence/stage3_tuning), [evidence/stage4_finals](evidence/stage4_finals), [evidence/stage5_test_once](evidence/stage5_test_once): saved public JSON/CSV training and evaluation records.
- [evidence/independent_posttest_audit.json](evidence/independent_posttest_audit.json): independent saved-prediction metric/hash audit.
- [PUBLIC_SHA256_MANIFEST.json](PUBLIC_SHA256_MANIFEST.json): SHA-256 of every published archived evidence/script file, plus corresponding original archive-member hash.

**Withheld privately:** full original dataset, frozen split `.npz`, model and optimizer checkpoint files `.pt`, and per-node predictions `.npy`. The independent Mac backup is preserved with verified 196 original files (whole-archive SHA-256 `3a358b242f1a6e0e4b405ecc7fbbbc808ed3541c5f5c7deb38bc8c690abe16f9`). The backup is **not** uploaded to GitHub. Do not rerun Stage 5: all 12 test-once claims have already been used.
