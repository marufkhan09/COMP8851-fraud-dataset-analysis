# EOGFD × YelpChi — completed frozen-split benchmark

**Status: COMPLETE.** TR40 validation-only tuning **12/12**, frozen final training **12/12**, one-time held-out test evaluation **12/12**. This is the experimentally complete benchmark using the original EOGFD heterograph model with the Assessment 2 fixed nested-split protocol—not an exact reproduction of the authors' original train/test workflow.

## Data, source and experimental protocol

- Model source: [BERA-wx/EOGFD pinned commit](https://github.com/BERA-wx/EOGFD/tree/a4726f173bc8c923109ddcb2786f4d9c5206ae7f) `a4726f173bc8c923109ddcb2786f4d9c5206ae7f`; actual model class `EOGFDHeteroModel`, original relationship-specific propagation and evolutionary theta optimiser.
- Source SHA-256: `fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42`; frozen split SHA-256: `0ea0af36dfc5a3a1f381ea2e3168377e45826498b6063190185c673e9ec8c22b`; split seed **2**.
- Frozen ratios **TR40 / TR30 / TR20 / TR10**, final seeds **2, 42, 72**, fixed validation/test nodes. Held-out test size **18382**, fraud cases **2671**.
- TR40 grid has 12 validation-AUPRC configurations. Selected **trial 8** with validation AUPRC **0.613103**, parameters `{"beta": 0.5, "beta_p": 1, "hidden_dim": 64, "lamb": 0.0, "lr": 0.003, "mutation_rate": 0.1, "order": 3, "population_size": 20}`. Frozen configuration reused for each of the 12 final runs.
- Maximum **100 epochs**, **patience 20**, checkpoints chosen by **validation AUPRC**, decision thresholds by **validation macro-F1**. Test labels and scores were never used for training, hyperparameter search, checkpoint selection or threshold selection.
- During EOGFD evaluation the evolutionary proposal is stochastic; validation and final held-out inference use the same fixed RNG seed **100921**, with separate, one-time test scoring per frozen model.
- **Methodological distinction:** The unmodified upstream `run_eogfd.py` uses its own split and scores the test set each epoch. The published scripts instead enforce Assessment 2 frozen nested splits and prohibit test access until all selections are frozen. No claim of author-identical reproduction or supervisor methodological sign-off is made.

## Held-out test results

Three-seed **mean ± sample standard deviation** (`ddof=1`), not validation-set metrics.

| Training ratio | Test AUPRC | Test AUROC | Test macro-F1 |
|---|---:|---:|---:|
| TR40 | 0.607243 ± 0.006921 | 0.882973 ± 0.001463 | 0.750763 ± 0.003003 |
| TR30 | 0.598122 ± 0.004263 | 0.878804 ± 0.002158 | 0.744224 ± 0.006083 |
| TR20 | 0.585589 ± 0.008998 | 0.869690 ± 0.003099 | 0.739681 ± 0.007418 |
| TR10 | 0.529598 ± 0.002701 | 0.843883 ± 0.002507 | 0.714070 ± 0.004573 |

### Individual final test results

| Ratio | Seed | Test AUPRC | Test AUROC | Test macro-F1 | Fraud F1 |
|---|---:|---:|---:|---:|---:|
| TR40 | 2 | 0.613031 | 0.884079 | 0.754009 | 0.579840 |
| TR40 | 42 | 0.599576 | 0.881314 | 0.748082 | 0.570320 |
| TR40 | 72 | 0.609122 | 0.883525 | 0.750199 | 0.574569 |
| TR30 | 2 | 0.599453 | 0.881276 | 0.749712 | 0.574883 |
| TR30 | 42 | 0.593352 | 0.877297 | 0.737683 | 0.549943 |
| TR30 | 72 | 0.601560 | 0.877840 | 0.745276 | 0.569317 |
| TR20 | 2 | 0.581405 | 0.868308 | 0.744946 | 0.568399 |
| TR20 | 42 | 0.579444 | 0.867522 | 0.731197 | 0.536768 |
| TR20 | 72 | 0.595917 | 0.873239 | 0.742899 | 0.566528 |
| TR10 | 2 | 0.526535 | 0.840995 | 0.713586 | 0.522280 |
| TR10 | 42 | 0.530623 | 0.845488 | 0.709758 | 0.497468 |
| TR10 | 72 | 0.531636 | 0.845167 | 0.718865 | 0.528928 |

## Reproducibility and scope

- `scripts/`: Stage 2 cost pilot, Stage 3 tuning, Stage 4 finals, Stage 5 one-time test evaluation, the actual corrected CUDA bootstrap launchers and configurations.
- `results/tuning/`: 12 original validation-only trial records and histories, frozen selection and cost-gate evidence (no `.pt` model binaries).
- `results/finals/`: 12 frozen training JSON files, histories and validation-score arrays; 12 final test JSON records, one-time test claims and saved test-score arrays; aggregate Stage 4/5 summaries.
- `evidence/`: complete execution logs, failed import diagnostic, recovered CUDA launch and independent offline metric-recomputation audit, with SHA-256 file manifest.
- Original model source remains at the pinned upstream Git commit linked above; it is not republished here.
- **Not uploaded:** `.mat` source graph, frozen `.npz` split, model checkpoints (`.pt`), optimizer-state backups, CUDA environment or any full private research backup. Checkpoint hash fields are retained, but weights require private preservation for model replay.

**Publication export rechecks saved score arrays against frozen test labels without running any GPU inference, training or retuning.**
