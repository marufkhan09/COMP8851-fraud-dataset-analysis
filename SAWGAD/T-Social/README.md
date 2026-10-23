# SAWGAD × T-Social — frozen validation-only training completed

**Status as recorded on 24 October 2026:** the canonical T-Social graph and frozen training/validation/test splits have been verified. **All 12/12 TR40 validation-only tuning trials and 12/12 frozen multi-ratio final training runs have completed on I2.** The independent metadata and SHA-256 audit passed for all 24 runs. The independent checkpoint-validation replay still requires review; held-out test evaluation has not taken place. No held-out test performance can yet be reported.

## Provenance and reproducibility

- Original pinned author source: `yj-zhou/SAWGAD`, commit `64d6d531a2f03549f0bfbe461092beedd6f70bd2`; original author source remains pinned and unchanged.
- Canonical source: 5,781,065 nodes, 146,211,016 stored directed edges, 10 features; source SHA-256 `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`.
- Frozen nested split SHA-256 `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519` (split seed 2); TR40/TR30/TR20/TR10 with training seeds 2, 42, 72.
- SAWGAD uses an attention-based graph encoder, synthetic anomaly training, and separate synthetic warm-up and supervised phases. Full frozen validation covers **1,156,213 nodes**, including **34,856 fraud-labelled nodes**; validation AUPRC was used for tuning and checkpoint selection.
- **Adaptation:** author full-graph training was replaced by bounded one-hop inbound neighbourhood sampling (fanout 10), with 32 warm-up updates, 8 × 1,024 supervised updates per tuning configuration and 16 × 1,024 per final run. This is **not** an author-identical or fully harmonised training protocol, and methodological approval has not been recorded.

## Validation-only tuning — available evidence

- **12 TR40 configurations completed:** `hidden_dim ∈ {32,64}`, learning rates `{0.001,0.003,0.005}`, and dropout rates `{0.0,0.25}`, using training seed 2.
- Frozen tuning winner **trial 9**, with `hidden_dim=64`, learning rate `0.005`, dropout `0.0`; best full-validation AUPRC **0.167209** at round 8. The winning trial and its recorded checkpoint hash were independently verified.
- **12 frozen final training runs completed:** TR40/TR30/TR20/TR10 × seeds 2, 42, 72. The full-validation AUPRC results (mean ± sample SD across seeds) are:
  - **TR40:** **0.171703 ± 0.002465**; seeds 2/42/72: **0.170170 / 0.174546 / 0.170391**.
  - **TR30:** **0.171425 ± 0.005535**; seeds 2/42/72: **0.173258 / 0.175812 / 0.165206**.
  - **TR20:** **0.173948 ± 0.005635**; seeds 2/42/72: **0.180091 / 0.169018 / 0.172736**.
  - **TR10:** **0.176150 ± 0.002892**; seeds 2/42/72: **0.172999 / 0.176769 / 0.178683**.
- All **24/24** saved tuning/final completion records, validation histories and checkpoint fingerprints passed independent **metadata and SHA-256 integrity checks**. Public scripts, JSON/CSV results, and the publication manifest are archived in `completed_frozen_training_20261024/`.
- Independent checkpoint-based replay of the tuning winner and 12 final checkpoints found full-validation AUPRC differences of **0.000122–0.001805** relative to recorded scores; all **13/13** exceeded the `0.00001` replay tolerance. This requires further evaluator-reproducibility investigation; the independent prediction-score replay **did not pass**.
- **No held-out test scoring has taken place.** These validation AUPRC values must not be reported as held-out test performance or used to revise the model after testing.

## Planned remaining phases

1. Investigate the independent checkpoint-validation replay discrepancy and reconcile evaluator sampling and scoring before approving the pretest gate.
2. Preserve and independently verify private checkpoints, frozen configuration, validation-only selection and label isolation. Select any classification thresholds using validation data only.
3. After pretest verification and any required approvals, evaluate each frozen TR40/TR30/TR20/TR10 × seed 2/42/72 model once on the held-out test set.
4. Independently audit saved test predictions and metrics, then publish verified held-out results and update this root README.

No original graph data, frozen-split files, model/optimizer checkpoints or node-level predictions should be published to this public repository.
