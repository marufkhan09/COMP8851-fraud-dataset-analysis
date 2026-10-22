# EOGFD × FDCompCN — completed conditional unified benchmark

**Status:** 12/12 TR40 validation-only tuning trials, 12/12 fresh frozen final trainings, 12/12 once-only held-out test evaluations. These are completed experimental outputs, **not an approved author-identical reproduction**.

- Model: [BERA-wx/EOGFD](https://github.com/BERA-wx/EOGFD/tree/a4726f173bc8c923109ddcb2786f4d9c5206ae7f) pinned at `a4726f173bc8c923109ddcb2786f4d9c5206ae7f`; original `EOGFDHeteroModel` implementation unchanged.
- Assessment 2 canonical source SHA-256: `{"comp.dgl": "e252b9a6b619b28a7bf6d9f5b16aacc232d43207d87ac26cd5160fb0d8a98baa"}`.
- Frozen split SHA-256: `5cd7084342f4af48e9b82828dcac0e97879314cd5c9c010642b30c89e7c95168`; split seed 2; training seeds 2, 42, 72.
- Input adapter: original four directed company relations (static transductive unlabeled graph).
- Stage 3 tuning: TR40 seed 2, **12 attempts**, selection by validation AUPRC. Winner trial **1**; validation AUPRC **0.215919**. Frozen configuration: `{"beta_p": 1, "hidden_dim": 64, "lamb": 0.0, "lr": 0.001, "mutation_rate": 0.05, "order": 3, "population_size": 10, "weight_decay": 0.0}`.
- Stage 4: four nested training ratios × three training seeds, maximum 100 epochs, patience 20; checkpoint selection by validation AUPRC, classification threshold selected by validation macro-F1 (fraud-recall/0.5 tie-breaking).
- Stage 5: one exclusive test claim per selected checkpoint; exactly one test inference and saved scores. Independent Stage 6 metric recomputation uses **saved scores only**, not new GPU inference.
- EOGFD evaluation RNG seed 100921 fixed for validation/test. Adam optimizer beta1=.9, beta2=.999, eps=1e-8; training loss includes model-specific class weighting; parameters recorded for each trial.
- **Methodological status:** `UNIFIED_CONDITIONAL_ADAPTED`; Venus's model-specific adaptation sign-off has **not been recorded**. This flag reflects documentation status, not rejection. Dataset eligibility should be reviewed before interpreting as a supervisor-approved benchmark.
- This benchmark's time/memory figures are from RTX A6000 instance 57212396, not the Kaggle T4 described as a planning baseline; treat cross-instance runtime comparisons as non-controlled.

## Once-only held-out test metrics (mean ± sample SD; three seeds)

| Ratio | AUPRC | AUROC | Macro-F1 |
|---|---:|---:|---:|
| TR40 | 0.149602 ± 0.017820 | 0.568062 ± 0.035745 | 0.521723 ± 0.003718 |
| TR30 | 0.142049 ± 0.008232 | 0.547843 ± 0.012934 | 0.514941 ± 0.012162 |
| TR20 | 0.139759 ± 0.009211 | 0.552961 ± 0.014440 | 0.510695 ± 0.013694 |
| TR10 | 0.134169 ± 0.006599 | 0.552444 ± 0.005038 | 0.507217 ± 0.010708 |

### Individual held-out runs

| Ratio | Seed | AUPRC | AUROC | Macro-F1 |
|---|---:|---:|---:|---:|
| TR40 | 2 | 0.169451 | 0.609310 | 0.521983 |
| TR40 | 42 | 0.144373 | 0.546147 | 0.525305 |
| TR40 | 72 | 0.134982 | 0.548729 | 0.517883 |
| TR30 | 2 | 0.141177 | 0.560288 | 0.509466 |
| TR30 | 42 | 0.150681 | 0.548772 | 0.528878 |
| TR30 | 72 | 0.134287 | 0.534470 | 0.506479 |
| TR20 | 2 | 0.136492 | 0.545837 | 0.497310 |
| TR20 | 42 | 0.150158 | 0.569579 | 0.510095 |
| TR20 | 72 | 0.132628 | 0.543468 | 0.524679 |
| TR10 | 2 | 0.131407 | 0.548753 | 0.500151 |
| TR10 | 42 | 0.141700 | 0.558184 | 0.519537 |
| TR10 | 72 | 0.129399 | 0.550395 | 0.501964 |

## Public evidence

- `results/tuning/`: 12 trial histories, timing CSVs, saved validation scores, frozen winning configuration, manifest and capacity gate.
- `results/finals/`: all 12 final validation records, full epoch timing CSVs, validation-score arrays, one-time held-out test claims, test-score arrays, frozen summaries and audit notes.
- `scripts/`: actual adapted runner scripts, corrected Stage 2 pilot and Stage 5 discrepancy recovery. Original EOGFD author's source lives only in linked pinned upstream.
- `evidence/`: execution logs, pinned source/split fingerprints, independent saved-score audit and publication hash manifest.
- **Excluded from Git:** source datasets, frozen split arrays, model `.pt` weights, full training recovery `.pt` files, Python/CUDA environments and private backup.
- A separate private ZIP preserves the 24 final and 24 tuning checkpoints across both datasets, including the frozen split arrays. Keep it outside the public repository.

**Original released `run_eogfd.py` evaluates held-out data each epoch; this external benchmark runner instead froze Assessment 2 split IDs and isolated test access. Not an author-identical reproduction.**
