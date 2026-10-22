# EOGFD × Elliptic — completed conditional unified benchmark

**Status:** 12/12 TR40 validation-only tuning trials, 12/12 fresh frozen final trainings, 12/12 once-only held-out test evaluations. These are completed experimental outputs, **not an approved author-identical reproduction**.

- Model: [BERA-wx/EOGFD](https://github.com/BERA-wx/EOGFD/tree/a4726f173bc8c923109ddcb2786f4d9c5206ae7f) pinned at `a4726f173bc8c923109ddcb2786f4d9c5206ae7f`; original `EOGFDHeteroModel` implementation unchanged.
- Assessment 2 canonical source SHA-256: `{"elliptic_txs_classes.csv": "93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493", "elliptic_txs_edgelist.csv": "a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e", "elliptic_txs_features.csv": "fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0"}`.
- Frozen split SHA-256: `1e7963e9d09935fb33e0df74cfb786cab826d1b95ae10b9592dbc240b48ffdbc`; split seed 2; training seeds 2, 42, 72.
- Input adapter: single original directed transaction relation with ratio-specific training-prefix cutoffs TR40 ≤20, TR30 ≤12, TR20 ≤7, TR10 ≤3, validation ≤31, test period 32–49; no invented or reversed edges.
- Stage 3 tuning: TR40 seed 2, **12 attempts**, selection by validation AUPRC. Winner trial **10**; validation AUPRC **0.708897**. Frozen configuration: `{"beta_p": 1, "hidden_dim": 64, "lamb": 0.0, "lr": 0.01, "mutation_rate": 0.1, "order": 3, "population_size": 10, "weight_decay": 0.0}`.
- Stage 4: four nested training ratios × three training seeds, maximum 100 epochs, patience 20; checkpoint selection by validation AUPRC, classification threshold selected by validation macro-F1 (fraud-recall/0.5 tie-breaking).
- Stage 5: one exclusive test claim per selected checkpoint; exactly one test inference and saved scores. Independent Stage 6 metric recomputation uses **saved scores only**, not new GPU inference.
- EOGFD evaluation RNG seed 100921 fixed for validation/test. Adam optimizer beta1=.9, beta2=.999, eps=1e-8; training loss includes model-specific class weighting; parameters recorded for each trial.
- **Methodological status:** `UNIFIED_CONDITIONAL_ADAPTED`; Venus's model-specific adaptation sign-off has **not been recorded**. This flag reflects documentation status, not rejection. Dataset eligibility should be reviewed before interpreting as a supervisor-approved benchmark.
- This benchmark's time/memory figures are from RTX A6000 instance 57212396, not the Kaggle T4 described as a planning baseline; treat cross-instance runtime comparisons as non-controlled.

## Once-only held-out test metrics (mean ± sample SD; three seeds)

| Ratio | AUPRC | AUROC | Macro-F1 |
|---|---:|---:|---:|
| TR40 | 0.554900 ± 0.077137 | 0.849932 ± 0.014314 | 0.703586 ± 0.073903 |
| TR30 | 0.523639 ± 0.062707 | 0.838100 ± 0.021868 | 0.773330 ± 0.016490 |
| TR20 | 0.444356 ± 0.077342 | 0.817598 ± 0.017623 | 0.693589 ± 0.032676 |
| TR10 | 0.307366 ± 0.081398 | 0.783826 ± 0.042705 | 0.619153 ± 0.073792 |

### Individual held-out runs

| Ratio | Seed | AUPRC | AUROC | Macro-F1 |
|---|---:|---:|---:|---:|
| TR40 | 2 | 0.569324 | 0.841271 | 0.632792 |
| TR40 | 42 | 0.471569 | 0.842071 | 0.697718 |
| TR40 | 72 | 0.623806 | 0.866454 | 0.780249 |
| TR30 | 2 | 0.561073 | 0.814148 | 0.771168 |
| TR30 | 42 | 0.451246 | 0.843152 | 0.758028 |
| TR30 | 72 | 0.558599 | 0.856999 | 0.790794 |
| TR20 | 2 | 0.393795 | 0.817854 | 0.690151 |
| TR20 | 42 | 0.405884 | 0.799848 | 0.662769 |
| TR20 | 72 | 0.533390 | 0.835091 | 0.727848 |
| TR10 | 2 | 0.374117 | 0.803293 | 0.651042 |
| TR10 | 42 | 0.216686 | 0.734856 | 0.534780 |
| TR10 | 72 | 0.331296 | 0.813329 | 0.671639 |

## Public evidence

- `results/tuning/`: 12 trial histories, timing CSVs, saved validation scores, frozen winning configuration, manifest and capacity gate.
- `results/finals/`: all 12 final validation records, full epoch timing CSVs, validation-score arrays, one-time held-out test claims, test-score arrays, frozen summaries and audit notes.
- `scripts/`: actual adapted runner scripts, corrected Stage 2 pilot and Stage 5 discrepancy recovery. Original EOGFD author's source lives only in linked pinned upstream.
- `evidence/`: execution logs, pinned source/split fingerprints, independent saved-score audit and publication hash manifest.
- **Excluded from Git:** source datasets, frozen split arrays, model `.pt` weights, full training recovery `.pt` files, Python/CUDA environments and private backup.
- A separate private ZIP preserves the 24 final and 24 tuning checkpoints across both datasets, including the frozen split arrays. Keep it outside the public repository.

**Original released `run_eogfd.py` evaluates held-out data each epoch; this external benchmark runner instead froze Assessment 2 split IDs and isolated test access. Not an author-identical reproduction.**
