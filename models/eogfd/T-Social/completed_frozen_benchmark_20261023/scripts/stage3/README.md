# EOGFD × T-Social — Stage 3 v3 on I2 57231045

## Exact repaired failure
v1 hit OOM in validation; v2 streamed ES candidate fitness in validation but left **training** `evolutionary_step` unchanged. v2 completed epoch 1, then epoch 2's training call tried to allocate a 4.31 GiB tensor and failed. v3 streams candidate projection fitness in **both training and validation**. The population, fitness objective, author graph, polynomial order, hidden size, and population sizes are not reduced.

## Integrity and scientific status
Uses source SHA 8d577114…c8d0 and frozen split SHA b7e76835…bc6519, pinned original author code SHA a4726f1… unchanged. All monkeypatching is model-instance/process-scoped; the pinned repository is unmodified. This is a **documented memory-engineered implementation**, not an untouched-author reproduction. Retains prior failed attempts and logs in their original directories. New v3 output tree separates these results to prevent mixing with v2's failed attempt.

Runs equivalence before graph loading: CPU/CUDA elite selector + evolutionary-step theta and RNG tests, **plus small-graph entire forward, loss, backward gradient and RNG equivalence**. Numeric equivalence is checked to tolerances; bitwise full-graph equality cannot be asserted.

Runs with `--trial8` first. Saves each completed epoch into `epoch_recovery.pt` for restart safety, 100 epoch maximum, patience 20, validation AUPRC for selecting checkpoints, no held-out test labels. Only after a complete trial8 should `--all` be used. Do not launch simultaneous Stage 3 jobs.

Run: `bash /workspace/EOGFD_TSOCIAL_STAGE3_V3_I2_20261023/vast_i2_stage3_v3.sh --trial8`.
