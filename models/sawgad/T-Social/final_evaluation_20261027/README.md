# SAWGAD × T-Social — separately reported final evaluation

This release preserves the 12-row, four-ratio by three-seed test metric table supplied from a separate RTX A6000 48-GB Vast.ai GPU run. The per-seed CSV contains test AUPRC, test AUROC and test macro-F1; the summary CSV contains means and sample standard deviations (`ddof=1`) calculated solely from those rows.

The source screenshot is retained under `evidence/source_test_results.png`. The archived training and validation evidence in `../completed_frozen_training_20261024/` remains unchanged. Independent recomputation from the separate GPU's ordered prediction scores and labels has not been performed; no additional test metrics are inferred.
