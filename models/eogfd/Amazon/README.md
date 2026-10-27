# EOGFD × Amazon — completed frozen-split benchmark

**Status: COMPLETE.** TR40 validation-only tuning **12/12**, frozen final training **12/12**, one-time held-out test evaluation **12/12**. This is the experimentally complete benchmark using the original EOGFD heterograph model with the Assessment 2 fixed nested-split protocol—not an exact reproduction of the authors' original train/test workflow.

## Data, source and experimental protocol

- Model source: [BERA-wx/EOGFD pinned commit](https://github.com/BERA-wx/EOGFD/tree/a4726f173bc8c923109ddcb2786f4d9c5206ae7f) `a4726f173bc8c923109ddcb2786f4d9c5206ae7f`; actual model class `EOGFDHeteroModel`, original relationship-specific propagation and evolutionary theta optimiser.
- Source SHA-256: `4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63`; frozen split SHA-256: `0fd96816c440b247b8cac768896bc454b1ecf4abebaf5001c1a3e1d900efa4fd`; split seed **2**.
- Frozen ratios **TR40 / TR30 / TR20 / TR10**, final seeds **2, 42, 72**, fixed validation/test nodes. Held-out test size **3456**, fraud cases **328**.
- TR40 grid has 12 validation-AUPRC configurations. Selected **trial 7** with validation AUPRC **0.922895**, parameters `{"beta": 0.9, "beta_p": 1, "hidden_dim": 64, "lamb": 0.0, "lr": 0.003, "mutation_rate": 0.05, "order": 4, "population_size": 20}`. Frozen configuration reused for each of the 12 final runs.
- Maximum **100 epochs**, **patience 20**, checkpoints chosen by **validation AUPRC**, decision thresholds by **validation macro-F1**. Test labels and scores were never used for training, hyperparameter search, checkpoint selection or threshold selection.
- During EOGFD evaluation the evolutionary proposal is stochastic; validation and final held-out inference use the same fixed RNG seed **100921**, with separate, one-time test scoring per frozen model.
- **Methodological distinction:** The unmodified upstream `run_eogfd.py` uses its own split and scores the test set each epoch. The published scripts instead enforce Assessment 2 frozen nested splits and prohibit test access until all selections are frozen. No claim of author-identical reproduction or supervisor methodological sign-off is made.

## Held-out test results

Three-seed **mean ± sample standard deviation** (`ddof=1`), not validation-set metrics.

| Training ratio | Test AUPRC | Test AUROC | Test macro-F1 |
|---|---:|---:|---:|
| TR40 | 0.905155 ± 0.002139 | 0.973647 ± 0.001547 | 0.919026 ± 0.004824 |
| TR30 | 0.888092 ± 0.012523 | 0.970534 ± 0.002339 | 0.918206 ± 0.009406 |
| TR20 | 0.890266 ± 0.001469 | 0.966848 ± 0.001405 | 0.918828 ± 0.007010 |
| TR10 | 0.856886 ± 0.003406 | 0.944672 ± 0.002008 | 0.905674 ± 0.002619 |

### Individual final test results

| Ratio | Seed | Test AUPRC | Test AUROC | Test macro-F1 | Fraud F1 |
|---|---:|---:|---:|---:|---:|
| TR40 | 2 | 0.907622 | 0.972128 | 0.924469 | 0.862620 |
| TR40 | 42 | 0.903835 | 0.975220 | 0.915278 | 0.845118 |
| TR40 | 72 | 0.904006 | 0.973592 | 0.917332 | 0.849088 |
| TR30 | 2 | 0.899051 | 0.972218 | 0.923697 | 0.860884 |
| TR30 | 42 | 0.890783 | 0.967863 | 0.907345 | 0.830508 |
| TR30 | 72 | 0.874441 | 0.971521 | 0.923576 | 0.860465 |
| TR20 | 2 | 0.891958 | 0.965801 | 0.923576 | 0.860465 |
| TR20 | 42 | 0.889519 | 0.966299 | 0.910777 | 0.836735 |
| TR20 | 72 | 0.889320 | 0.968444 | 0.922132 | 0.858075 |
| TR10 | 2 | 0.852978 | 0.944533 | 0.904015 | 0.824138 |
| TR10 | 42 | 0.859220 | 0.946746 | 0.904314 | 0.824742 |
| TR10 | 72 | 0.858462 | 0.942736 | 0.908693 | 0.833052 |

## Reproducibility and scope

- `scripts/`: Stage 2 cost pilot, Stage 3 tuning, Stage 4 finals, Stage 5 one-time test evaluation, the actual corrected CUDA bootstrap launchers and configurations.
- `results/tuning/`: 12 original validation-only trial records and histories, frozen selection and cost-gate evidence (no `.pt` model binaries).
- `results/finals/`: 12 frozen training JSON files, histories and validation-score arrays; 12 final test JSON records, one-time test claims and saved test-score arrays; aggregate Stage 4/5 summaries.
- `evidence/`: complete execution logs, failed import diagnostic, recovered CUDA launch and independent offline metric-recomputation audit, with SHA-256 file manifest.
- Original model source remains at the pinned upstream Git commit linked above; it is not republished here.
- **Not uploaded:** `.mat` source graph, frozen `.npz` split, model checkpoints (`.pt`), optimizer-state backups, CUDA environment or any full private research backup. Checkpoint hash fields are retained, but weights require private preservation for model replay.

**Publication export rechecks saved score arrays against frozen test labels without running any GPU inference, training or retuning.**
