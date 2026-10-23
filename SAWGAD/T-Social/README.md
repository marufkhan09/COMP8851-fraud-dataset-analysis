# SAWGAD × T-Social — GPU feasibility preflight pending

**Status as of 23 October 2026:** the SAWGAD × T-Social benchmark is **not started**. The previous SAWGAD × T-Finance benchmark is complete and documented separately, but those results must not be presented as T-Social results. The EOGFD × T-Social benchmark is complete on I2, and that same rented GPU is a potential new SAWGAD execution host after separate preflight.

## Model and source

- Original pinned [SAWGAD author repository](https://github.com/yj-zhou/SAWGAD/tree/64d6d531a2f03549f0bfbe461092beedd6f70bd2) commit `64d6d531a2f03549f0bfbe461092beedd6f70bd2`, officially developed for weakly supervised graph anomaly detection with synthetic perturbations and a two-phase warm-up / supervised-training strategy.
- Canonical T-Social source: 5,781,065 nodes, 146,211,016 directed graph edges, 10 node features; SHA-256 `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`.
- Frozen nested split SHA-256 `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519`; TR40/TR30/TR20/TR10 and model seeds 2, 42, 72 remain the desired benchmark design.

## Implementation constraints to resolve before training

- Upstream `pyproject.toml` pins Python `>=3.12,<3.13`, PyTorch `2.4.0`, DGL `2.4.0`, CUDA 11.8. It **must not** be installed over I2's established EOGFD environment.
- Upstream encoder computes attention over the **entire graph**, and the synthetic-anomaly pipeline deep-copies/perturbs that graph. Batch-size arguments do not alone bound GPU memory on T-Social's 146 million edges.
- The author entry point defaults to Amazon and builds its own split; using the immutable COMP8851 T-Social split requires a carefully tested adapter and independent test-label isolation audit.
- Memory/capacity and numerical consistency must pass bounded dry runs before authorizing full training; no full-graph or author-equivalent performance is claimed yet.

## Planned experiment protocol (not yet executed)

1. On isolated I2 Python/CUDA environment, pin the original source commit, verify source/split SHA-256, inspect graph preprocessing and exact edge/memory behavior, and remove held-out labels from model features.
2. Run minimal graph and synthetic-perturbation memory pilots without model selection or test access. Document any computational adaptation and seek methodological approval if required.
3. If feasible, conduct 12 TR40 seed-2 validation-only trials, freeze validation-AUPRC winner, and train four ratios × seeds 2/42/72.
4. Select thresholds on validation only and carry out independent once-only held-out evaluation after all final artifacts are frozen.
5. Publish complete per-seed test results and reproduction evidence in **this root README**, keeping graph data, checkpoints and per-node predictions private.

**No SAWGAD × T-Social AUPRC, AUROC or F1 results exist yet.**
