# EOGFD × T-Social — I2 Stage 4 (frozen final training)

Instance **57231045**, hostname **a92ee6fa0813** only.

Stage 3 `--all` previously produced 12 validation-only candidate trials and locked
**trial 9**, validation AUPRC ~0.859637, with zero held-out test evaluations.
The Stage 4 script audits all 12 Stage 3 candidate files and SHA-256 hashes,
verifies trial 9 against the frozen-winner manifest, and pins the same exact
model / streamed ES-fitness runtime optimization from the verified Stage 3 V3.
The unmodified upstream EOGFD repository remains untouched.

Final training grid: TR40/TR30/TR20/TR10 × training seeds 2/42/72. All use
**the same frozen trial-9 hyperparameters**. Each final uses only labels of
its own training ratio and validation labels to select best checkpoint by
validation AUPRC and threshold by validation macro-F1. No test labels are
indexed and no test inference is executed. Per-epoch recovery is atomic;
complete final records and their hashes are verified and reused upon restart.

This is a unified benchmark split/feature adapter and process-local memory
engineering, **not** an author-identical reproduction. Supervisor approval is
not documented.

## On I2 after upload of ZIP to `/workspace`

Verify ZIP against the checksum communicated with delivery, then extract and
check internal hashes. Run `bash vast_i2_stage4.sh --audit-only` first, then
`bash vast_i2_stage4.sh --all`. For an incremental first-run gate, `--one TR40 2`
may be run before `--all`, but it is optional and will be SHA-verified/reused
when `--all` is run; avoid duplicate concurrent processes.

The `--all` operation is **resume-safe from completed epochs** and reuses
completed final records. It will not overwrite Stage 3 files, and will refuse
to rerun when the 12-final Stage 4 summary is frozen. Keep GPU instance and
Jupyter terminal running during training; do not run another Stage 3 job.

Required success: `I2_TSOCIAL_STAGE4_12_FINALS_FROZEN=PASS TEST_SCORED=NO TEST_EVALUATIONS=0`.

Then **stop**: separate independent audit, private backup, and final test-once
protocol are the next stages. Never score held-out test early.
