# COMP8851 | SAWGAD × T-Social | 12 TR40 tunes then 3-seed × 4-ratio training

23 October 2026. **Exploratory adapted protocol; not author-identical or supervisor-approved.** Neither a final benchmark nor held-out testing is performed by this kit.

## Before launching

- Run only on Vast I2 `a92ee6fa0813`; original pinned SAWGAD author clone must remain unmodified.
- Verify the prior sustained full-validation seed-2 pilot completion, best full-validation AUPRC `0.1596089366267632` and preserved checkpoint. The kit refuses to start otherwise.
- The original dataset and split hashes are checked before launch and by every worker.
- Original full T-Social graph remains on CPU; each model step uses an inbound fanout-10 sampled subgraph and the original SAWGAD model/head functions with local synthetic perturbation. Results cannot be described as a full-graph original-author reproduction. Approved comparative ranking must be reviewed with Dr Venus Haghighi.
- One training worker runs at a time. A process file-lock prevents duplicate pipeline writers. No test-label indexing or test-score calculation is performed.

## Execution

Run these with the isolated SAWGAD environment:

```
export DGLBACKEND=pytorch
PY=/workspace/comp8851/repos/sawgad_tsocial_author/.venv/bin/python
KIT=/workspace/COMP8851_I2_SAWGAD_12TUNE_3x4_20261023
"$PY" "$KIT/sawgad_job.py" --selftest
"$PY" "$KIT/sawgad_pipeline.py" --selftest
"$PY" "$KIT/sawgad_pipeline.py" --preflight
"$PY" -u "$KIT/sawgad_pipeline.py" --run
```

The pipeline performs 12 distinct TR40/seed-2 configurations, measured on the full frozen 1,156,213-node validation set. Configurations vary hidden dimensions (32/64), dropout (0/.25), and joint-stage learning rate (.001/.003/.005), with the same 32 synthetic warm-up steps and eight training *rounds* of 1,024 minibatch updates per trial. An optimizer update on a sampled minibatch **is not** a pass over all training nodes. Trial selection is maximum full-validation AUPRC, tie broken by earlier trial index. Test never participates.

When all 12 TR40 trials pass independent summary/history/checkpoint checks, the selected configuration is immutably frozen in `frozen_tr40_winner.json` with supporting 12-trial evidence and a SHA256. Finals rerun the selected settings on TR40/TR30/TR20/TR10 with independent model seeds 2/42/72. All 12 finals use 32 warm-up steps and 16 rounds × 1,024 sampled minibatch updates, full frozen validation at every round, and validation-only best checkpoint selection. This finite-sampling protocol is an adaptation and shorter training budgets are not equivalent to the author's full-graph train epochs.

## Private evidence paths

Root: `/workspace/comp8851/results/sawgad/tsocial/stage3_tune12_final3x4_adapted_v1/`

- `plan.json`, `.single_writer.lock`
- `tune_01` ... `tune_12` each with immutable `job_spec.json`, `pilot_protocol.json`, `resume_state.pt`, `pilot_history.json`, `best_full_validation.pt`, `pilot_completed.json`, `training.log`
- `frozen_tr40_winner.json` after ALL 12 tuning trials
- `final_TR40_seed2` etc, one per ratio/seed
- `training_matrix_completed.json` only after all final jobs finish

If interrupted, rerun only the **same** `sawgad_pipeline.py --run` command. The pipeline verifies completed jobs and resumes the active checkpoint; never delete or overwrite pre-existing evidence folders. If a job fails, inspect its `training.log`. Preserve the original script bytes, run plan and sha manifest.

After completion, independent cross-check of node splits, trainer label accesses, best-checkpoint hashes, methodology approval, private off-host backup, and any RGTAN-like suspicious validation behavior is required before final test. The Stage-5 test script is deliberately absent.

Static tests passed in the creation environment for the 12 tuning specs and 12 final specs plus negative/mismatched-protocol cases. **No GPU training from this kit was executed in the creation environment.**
