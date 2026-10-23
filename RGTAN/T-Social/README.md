# RGTAN × T-Social — TR40 validation-only tuning completed

**Status as recorded on 23 October 2026:** the canonical T-Social graph, frozen training/validation/test splits, exact sparse two-hop risk features and independent Stage 2 crossfit audit have passed. **Trial 1/12 has completed; trial 2 is actively training on I1. Final tuning, frozen multi-ratio training and held-out test evaluation have not completed.** No test performance can yet be reported.

## Provenance and reproducibility

- Original pinned author source: [AI4Risk/antifraud](https://github.com/AI4Risk/antifraud/tree/65e583f03ac5c5983e49cc35179fe7a65908b9c1), commit `65e583f03ac5c5983e49cc35179fe7a65908b9c1`; original author model source is unchanged.
- Canonical source: 5,781,065 nodes, 146,211,016 stored directed edges, 10 features; source SHA-256 `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`.
- Frozen nested split SHA-256 `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519` (split seed 2).
- Five-fold **training-only** out-of-fold label contexts and exactly computed one-hop/two-hop risk features. Independent checks passed for 16 saved artifacts, masked labels, OOF scaling, binary CSR walk count (18,777,751,032 candidate two-hop walks), and sampled exact two-hop features.
- **Adaptation:** full-neighbour author propagation is replaced by bounded `MultiLayerNeighborSampler([10,10])`, batch size 1024. This is **not** an author-identical reproduction and methodological approval has not been recorded.

## Validation-only tuning — available evidence

- 12 configurations planned: `hidden_dim ∈ {32,64}`, base learning rates `{0.001,0.003,0.005}`, dropout pairs `[0.2,0.1]` and `[0.3,0.15]`.
- Trial 1 completed after 43 epochs (patience 20), best validation AUPRC **0.584989** at epoch 23; checkpoint and score hashes independently verified.
- Trial 2 was actively training on I1 at the latest observed checkpoint, epoch 12; early best validation AUPRC **0.991912** at epoch 2 is **provisional** and requires a separate leakage/metric/protocol audit before winner selection.
- **No held-out test scoring has taken place.** Do not publish provisional validation AUPRC as test performance or choose a winning checkpoint from test outcomes.

## Planned remaining phases

1. Finish and independently audit all 12 TR40 validation-only trials, investigating near-perfect scores.
2. Freeze hyperparameters by validation AUPRC only; rebuild label-derived risk features separately for TR30/TR20/TR10 using only each ratio's training labels.
3. Train TR40/TR30/TR20/TR10 × seeds 2, 42, 72 with immutable checkpoints and validation-selected thresholds.
4. After an independent pretest gate, evaluate each frozen model exactly once on the held-out test set.
5. Preserve private checkpoint backup, publish sanitized scripts/evidence, and update this root README with complete results.

No graphs, frozen-split files, model checkpoints or node-level predictions should be published to this public repository.
