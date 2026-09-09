# Required T-Finance dataset-version decision

Do not place these BWGNN results in a cross-model comparison table until the
team freezes one T-Finance raw-file version.

| Record | SHA-256 | Fraud | Normal |
|---|---|---:|---:|
| Current BWGNN execution | `b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700` | 1,804 | 37,553 |
| Earlier dataset investigation | `bd8fa6bb279bea1bda0e4891976edd41d67d041821c9f3d2c484c1b19e3a1b8d` | 1,803 | 37,554 |

Recommended resolution:

1. Adopt the current BWGNN-execution file.
2. Recalculate the T-Finance dataset and heterophily statistics on that file.
3. Record the current SHA-256 in the master dataset table.
4. Make all later T-Finance models reuse the split files in
   `shared/splits/tfinance/`.

If the earlier file is retained, rerun BWGNN with that exact file and regenerate
the shared splits before comparison.

