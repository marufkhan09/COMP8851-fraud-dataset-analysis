# RGTAN × T-Social — separately reported final evaluation

This directory records the user-supplied final-evaluation table from a separate Vast.ai GPU experiment. It is distinct from the earlier I1 TR40 hyperparameter-tuning evidence in `../tuning_evidence_20261025/`.

The per-seed CSV contains all 12 reported metric rows for four training ratios and seeds 2, 42 and 72. The summary CSV contains arithmetic means and sample standard deviations (`ddof=1`) calculated directly from those same rows, rounded to six decimal places.

The user reports that these are held-out test measurements. The label-score arrays, exact evaluation commands, evaluation logs and frozen decision thresholds from that separate GPU were not supplied with this evidence release; metric provenance and internal consistency must be checked against those primary artifacts for independent scientific verification.

The original T-Social graph and all private model artifacts are not included here. The earlier I1 Stage 2 and Stage 3 experiments remain archived separately with their own SHA-256 manifest.
