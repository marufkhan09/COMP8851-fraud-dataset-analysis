# EOGFD × T-Social — I2 Stage 5, once-only held-out test

Target Vast instance: **57231045**, hostname **a92ee6fa0813**.

This kit consumes the exact frozen outputs of Stage 4 v3 (`EOGFD_TSOCIAL_STAGE4_I2_20261023`) and the original pinned Stage 3 v3 optimization source. It does **not** retrain, retune, modify the original EOGFD repository, or repeat validation selection. It remains the documented *unified frozen-split adapted benchmark*, not an author-identical reproduction; Venus methodological approval is not claimed.

## Commands (I2 terminal, after uploading ZIP to `/workspace`)

```
cd /workspace
sha256sum EOGFD_TSOCIAL_STAGE5_I2_20261023.zip
python3 -m zipfile -e EOGFD_TSOCIAL_STAGE5_I2_20261023.zip /workspace
cd /workspace/EOGFD_TSOCIAL_STAGE5_I2_20261023
sha256sum -c KIT_SHA256SUMS.txt
bash vast_i2_stage5.sh --selftest
bash vast_i2_stage5.sh --audit-only
```

The pretest audit checks all 12 immutable Stage 4 checkpoints, model configurations, validation-selected thresholds, associated hashes and the Stage 3 frozen winner. It indexes **no test labels** and runs **no test inference**. It creates `/workspace/comp8851/results/eogfd/tsocial/stage5_heldout_test_once_v1/pretest_audit.json`.

After the pretest audit passes, authorize the once-only test operation:

```
bash /workspace/EOGFD_TSOCIAL_STAGE5_I2_20261023/vast_i2_stage5.sh --all
```

The script creates a non-overwritable `test_once_claim.json` for each job *before* accessing its test labels or running model inference; uses the frozen validation threshold with a strict `scores > threshold` decision rule; writes `test_scores.npy` atomically; and saves metric records. It reuses already-completed records without new inference if restarted partway through the 12 runs. If a claim exists but `test_metrics.json` is missing, it **halts** and requires manual review rather than rerunning test inference. The original graph and split are verified by SHA-256; test labels are not used for fitting, model selection, or choosing thresholds.

The output summary is `EOGFD_TSocial_Stage5_12TestOnceSummary.json`; each run contains its independent test score array, metrics and durable claim. Keep all private files for the subsequent independent test-metric audit, external backups and GitHub evidence-only publication. Do not destroy I2 before backup verification.

## Caveats

- This kit has passed local syntax, unit and ZIP-integrity checks but has **not** executed against the Vast GPU. `--audit-only` must pass first.
- The full-graph forward-pass uses the same process-local optimized EOGFD inference path and random evaluation seed `100921` as validation.
- Changing any Stage 3/Stage 4 source or frozen checkpoint makes the audit fail.
- After test scoring, do **not** rerun `--all` with a different runner, regenerate checkpoints, alter saved thresholds, or select any hyperparameters from held-out metrics.
