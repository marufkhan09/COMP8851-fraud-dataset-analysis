# EOGFD × T-Finance — completed frozen-split benchmark

**Status: COMPLETE.** Validation-only TR40 tuning **12/12**, fresh frozen final training **12/12**, once-only held-out test evaluation **12/12**. These are the completed Assessment 2 frozen-split benchmark outputs, not an author-identical workflow reproduction.

## Source, data and protocol

- Original model: [BERA-wx/EOGFD](https://github.com/BERA-wx/EOGFD/tree/a4726f173bc8c923109ddcb2786f4d9c5206ae7f) at pinned commit `a4726f173bc8c923109ddcb2786f4d9c5206ae7f`; original `EOGFDHeteroModel` source remains untouched.
- Canonical T-Finance original graph: 39,357 nodes, 42,445,086 stored directed edges, 10 features, one original relation, two-column author labels converted with `argmax`.
- Canonical graph SHA-256: `b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700`. Frozen split SHA-256: `de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1`.
- Frozen split seed 2; training seeds 2, 42, 72; TR40/TR30/TR20/TR10 nested training subsets, fixed validation/test partitions.
- TR40 seed-2 tuning: 12 planned trials, selected by validation AUPRC. Winner: trial 8 (validation AUPRC 0.883130); configuration `{"beta_p": 1, "hidden_dim": 64, "lamb": 0.0, "lr": 0.003, "mutation_rate": 0.1, "order": 7, "population_size": 20, "weight_decay": 0.0}`.
- Fresh final trainings: up to 100 epochs, patience 20. Best checkpoint from validation AUPRC. Classification threshold from validation macro-F1, tie-break fraud recall then nearest 0.5.
- Test protocol: one exclusive test-evaluation claim per final checkpoint, claims persisted **before** inference; exactly one test inference per checkpoint, no test-based checkpoint or threshold choice. Publication recomputes saved-score metrics offline only.
- Training class weights from each ratio's training labels; Adam beta1 0.9, beta2 0.999, eps 1e-8; evaluation RNG seed 100921.
- Scientific limitation: canonical frozen-split input adapter differs from upstream's original split/test workflow. This is **not** an exact author-protocol reproduction. Supervisor methodological approval is **not recorded**.

## Held-out test results

Three-seed mean ± **sample** standard deviation (`ddof=1`):

| Training ratio | Test AUPRC | Test AUROC | Test macro-F1 |
|---|---:|---:|---:|
| TR40 | 0.891626 ± 0.005091 | 0.967877 ± 0.001584 | 0.917495 ± 0.003925 |
| TR30 | 0.894572 ± 0.006039 | 0.969509 ± 0.002436 | 0.918465 ± 0.002023 |
| TR20 | 0.884958 ± 0.005405 | 0.969315 ± 0.001229 | 0.915674 ± 0.002528 |
| TR10 | 0.868687 ± 0.006001 | 0.957873 ± 0.003207 | 0.911600 ± 0.002028 |

## Individual final held-out results

| Ratio | Seed | Test AUPRC | Test AUROC | Test macro-F1 |
|---|---:|---:|---:|---:|
| TR40 | 2 | 0.893327 | 0.968081 | 0.917069 |
| TR40 | 42 | 0.885902 | 0.966201 | 0.913800 |
| TR40 | 72 | 0.895649 | 0.969350 | 0.921615 |
| TR30 | 2 | 0.900770 | 0.972283 | 0.918923 |
| TR30 | 42 | 0.888706 | 0.967713 | 0.916252 |
| TR30 | 72 | 0.894240 | 0.968532 | 0.920219 |
| TR20 | 2 | 0.890570 | 0.970671 | 0.918572 |
| TR20 | 42 | 0.879787 | 0.968274 | 0.913918 |
| TR20 | 72 | 0.884516 | 0.969000 | 0.914534 |
| TR10 | 2 | 0.863040 | 0.954565 | 0.913605 |
| TR10 | 42 | 0.868034 | 0.958085 | 0.909550 |
| TR10 | 72 | 0.874988 | 0.960968 | 0.911646 |

## Audit and reproducibility

- `results/tuning/`: 12 trial histories, per-epoch timings, stored validation scores, frozen winner and grid manifest.
- `results/finals/`: 12 final training records with validation histories, per-epoch timings, validation scores, test-once claims, saved test scores, test metrics and both frozen summaries.
- `scripts/`: actual Stage 3/4/5/6 runner scripts including the exact, forensically attested import-path repair. The original author model is linked via its immutable upstream commit.
- `evidence/`: provenance, source/split hashes, independent pre-test audit, pilot evidence and available execution logs.
- Excluded from Git: raw source graph, frozen split data, all checkpoint `.pt` weights, recovery states, CUDA/Python environments and private archives. Independently verified Mac private backup holds frozen checkpoints and validation evidence. Canonical graph and split retain separate verified Mac originals.
- Runtime measurements reflect RTX A6000 instance 57212396; cross-instance performance is not a controlled comparison.
