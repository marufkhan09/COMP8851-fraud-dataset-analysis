# Contract for model-specific adapters (engineering specification, not implementation)

The model-specific adapter implementation is NOT included because directly rewriting these models without tests would be scientifically unsafe. Each new adapter must expose:

- `read_canonical_input(dataset_name, source_path, source_sha, split_path, split_sha)`; reject unknown or mismatched SHA; preserve source features, edge types, node labels and time ordering.
- `make_fixed_masks(split_file, ratio, train_seed)`; split registry seed=2, same val/test IDs for all ratios, Elliptic time blocks unchanged, no accidental first-40%-of-nodes split.
- `run_smoke(training_seed=2, ratio=.40, epochs=2)`; **no test tensor access**; log exact code commit and effective configuration, all GPU timings and failures.
- `tune(trials=12, ratio=.40, seed=2)`; select from validation AUPRC only, early stop where compatible, TEST MASK UNAVAILABLE and raw data access audited. Save 12 trial configs, per-epoch validation and winner manifest.
- `fit_final(training_seed, ratio, frozen_winner)`; fixed source+split; no hyperparameter re-search for lower ratios, select checkpoint by validation AUPRC and evaluate test only after selection.
- `report_metrics(scores_test,labels_test)` once per final run: test AUPRC, AUROC, Macro-F1, fraud precision/recall/F1, G-Mean; record validation threshold, calibration choices if any. Keep raw probabilities for Rec@K post hoc, never use test to choose thresholds.
- `record_cost(preprocess_s, train_s, val_s, test_s, gpu_max_allocated_gib, hw_info)` with matched boundaries.
- `save_provenance(run_id, git_sha, input_sha, split_sha, config_sha, run_seed, ratio, status, artifacts)` append-only, no overwrite.

**Never patch model loss/architecture/fraud-specific logic just for runtime improvements.** Only known input/plumbing changes that preserve semantics may be candidate adapters. Native and canonical claims must be stored in distinct namespaces.
