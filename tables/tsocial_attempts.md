# T-Social — every recorded attempt

T-Social produced no test evaluation. It was **not skipped**: both models were attempted repeatedly, and each attempt below is taken either from its own artefact bundle in `../evidence/tsocial/` or from the run console, as the final column states.

| Attempt | Model | Phase | Status | Device | Config | Epochs | Wall | Best val AUPRC | Test | Failure |
|---|---|---|---|---|---|---|---|---|---|---|
| final_seed2 | CARE-GNN | final run | **FAILED** | cuda:0 | emb_size=64, lr=0.01 | 16 | 106 m | — | no | KeyboardInterrupt |
| tuning_trial00 | CARE-GNN | tuning | **COMPLETE** | cuda:0 | emb_size=64, lr=0.01 | 10 | 65 m | 0.0859 | no | — |
| tuning_trial01 | CARE-GNN | tuning | **COMPLETE** | cuda:0 | emb_size=32, lr=0.0005 | 10 | 66 m | 0.0520 | no | — |
| probe_ghrn_1 | GHRN | escalation rung 1 of 5 | **FAILED** | cuda:0 | hid_dim=64, order=2 (published config) | — | 2 m | — | no | OutOfMemoryError: CUDA out of memory — tried to allocate 1.38 GiB with 638.81 MiB free of 14.56 GiB |
| probe_ghrn_2 | GHRN | escalation rung 2 of 5 | **FAILED** | cuda:0 | hid_dim=32, order=2 (half width) | — | 2 m | — | no | DGLError: Expect argument "u" to have data type torch.int32. But got torch.int64. |
| probe_ghrn_3 | GHRN | escalation rung 3 of 5 | **FAILED** | cuda:0 | hid_dim=32, order=1 (lower order) | — | 2 m | — | no | DGLError: Expect argument "u" to have data type torch.int32. But got torch.int64. |
| probe_ghrn_4 | GHRN | escalation rung 4 of 5 | **FAILED** | cpu | hid_dim=64, order=2 (published config) | — | 7 m | — | no | DGLError: Expect argument "u" to have data type torch.int32. But got torch.int64. |
| probe_ghrn_5 | GHRN | escalation rung 5 of 5 | **FAILED** | cpu | hid_dim=32, order=2 (half width) | — | 5 m | — | no | DGLError: Expect argument "u" to have data type torch.int32. But got torch.int64. |

**8 attempts, not one test evaluation.** The CARE-GNN rows are read from artefact bundles in `../evidence/tsocial/`. The GHRN rows are transcribed from the execution notebook preserved in `../notebooks/`; that ladder ran on the benchmark host and its bundles stayed there under `probe_ghrn_1..5/`.

## Failure modes

| Failure | Attempts |
|---|---|
| KeyboardInterrupt | 1 |
| OutOfMemoryError | 1 |
| DGLError | 4 |

The two GHRN modes are worth separating. **Rung 1 ran out of GPU memory** at the published width: it needed a further 1.38 GiB with 638.81 MiB free of the T4's 14.56 GiB. That is a capacity limit and a genuine RQ3 finding — T-Social does not fit GHRN at published width on a 16 GB card. **Rungs 2 to 5 then hit an index-dtype defect**, on GPU and CPU alike, which is a code bug rather than a resource limit and is fixed in `../source_code/models/ghrn/ghrnlib/backend.py` but not yet re-run. Shrinking the model got past the memory wall and straight into the bug, so GHRN's true cost on T-Social is still unmeasured.

## Validation curve of the final CARE-GNN attempt

The run trained for 16 epochs before it was stopped. It was learning: validation AUPRC rose from 0.0366 at epoch 0 to a best of **0.1006** at epoch 14 (AUROC 0.7927).

| Epoch | Val AUPRC | Val AUROC | Val Macro-F1 | Threshold |
|---|---|---|---|---|
| 0 | 0.0366 | 0.5879 | 0.2309 | 0.99 |
| 1 | 0.0563 | 0.7131 | 0.5115 | 0.01 |
| 2 | 0.0560 | 0.7121 | 0.4254 | 0.99 |
| 3 | 0.0485 | 0.6799 | 0.4040 | 0.99 |
| 4 | 0.0720 | 0.7413 | 0.5099 | 0.01 |
| 5 | 0.0557 | 0.7124 | 0.4543 | 0.99 |
| 6 | 0.0859 | 0.7661 | 0.5538 | 0.59 |
| 7 | 0.0578 | 0.7136 | 0.4236 | 0.99 |
| 8 | 0.0573 | 0.7303 | 0.4930 | 0.99 |
| 9 | 0.0680 | 0.7504 | 0.5121 | 0.99 |
| 10 | 0.0556 | 0.6946 | 0.4098 | 0.99 |
| 11 | 0.0794 | 0.7605 | 0.5446 | 0.08 |
| 12 | 0.0542 | 0.7032 | 0.4074 | 0.99 |
| 13 | 0.0752 | 0.7688 | 0.5380 | 0.95 |
| 14 **(best)** | 0.1006 | 0.7927 | 0.5787 | 0.37 |
| 15 | 0.0821 | 0.7673 | 0.5602 | 0.46 |

**These are validation figures from an incomplete run.** They are reported here to show what happened, and they are deliberately kept out of every results table in this package. No T-Social number appears anywhere a test metric is expected.
