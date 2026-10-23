# RGTAN × T-Social — Frozen-split assessment benchmark (validation-only tuning in progress)

**Status at the latest verified I1 snapshot (24 October 2026, 05:05 AEDT):** the canonical graph, frozen splits, training-only risk-feature construction and Stage 2 crossfit checks had passed. **4/12 TR40 validation-only tuning trials were complete; trial 5 was training.** The remaining tuning trials, frozen final trainings (TR40/TR30/TR20/TR10 × seeds 2, 42, 72) and once-only held-out test evaluations had **not** been verified complete. The snapshot is dated; it is not a live progress indicator. **No held-out test metrics are available.** The author RGTAN model source is pinned and unchanged, but the T-Social runner uses bounded-neighbour sampling; this is not an author-identical full-neighbour experiment.

## Provenance

- Original RGTAN author: `AI4Risk/antifraud` (pinned commit `65e583f03ac5c5983e49cc35179fe7a65908b9c1`); original author model source is unchanged.
- Canonical T-Social graph: **5,781,065 nodes**, **146,211,016 stored directed edges**, **10 node features**; source SHA-256 `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`.
- Frozen Assessment 2 nested split SHA-256 `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519`; split seed **2**, ratios **TR40/TR30/TR20/TR10**, planned final seeds **2, 42, 72**, fixed validation and test membership.
- Five-fold training-only out-of-fold label contexts and exact sparse one-hop/two-hop risk features. The independent Stage 2 audit checked **16 saved artifacts**, label masking, crossfit scaling and sampled exact two-hop features; **18,777,751,032** candidate binary-CSR two-hop walks were recorded during construction. Validation/test labels were excluded from training risk-feature fitting.
- T-Social training adaptation: two RGTAN graph layers with `MultiLayerNeighborSampler([10,10])`, batch **1024**, rather than original full-neighbour propagation. Trial training allows up to **100 epochs** with **early-stopping patience 20**; full frozen-validation AUPRC selects checkpoints. This is **not** an author-identical reproduction, and supervisor methodological approval has not been recorded.
- TR40 tuning grid: **12** candidate configurations; hidden dimensions `{32,64}`, base learning rates `{0.001,0.003,0.005}`, dropout pairs `[0.2,0.1]` and `[0.3,0.15]`. The winner is **not yet verified or frozen**. Test labels are not used for tuning or checkpoint selection.

## Validation-only TR40 tuning results — last verified snapshot

These are the recorded **validation** AUPRC values available from the I1 terminal on **24 October 2026 at 05:05 AEDT**. They are neither a completed 12-trial ranking nor held-out test results. Trial 2's near-perfect value requires a separate alignment, metric and feature-provenance audit before winner selection.

| Trial | Training status at snapshot | Epochs completed | Best validation AUPRC | Verification note |
|---|---|---:|---:|---|
| 01 | Complete | 43 | 0.584989 | Stored checkpoint/score hashes checked |
| 02 | Complete | 22 | 0.991912 | Exceptional score; independent integrity review required |
| 03 | Complete | 26 | 0.477618 | Training completed |
| 04 | Complete | 39 | 0.576455 | Training completed |
| 05 | Active | 35 | 0.519692 | Provisional; stale epochs 18/20 at snapshot |
| 06–12 | Not started at snapshot | — | — | No result available |

## Held-out test results across three final seeds

**Not available.** At the last verified I1 snapshot, the pipeline was still tuning TR40; no frozen winning configuration, completed 12-run final matrix or once-only held-out test records were available. The four training ratios below are the planned final evaluation matrix, **not completed results**.

| Ratio | Planned final seeds | Held-out test AUPRC | Held-out test AUROC | Held-out test macro-F1 |
|---|---|---|---|---|
| TR40 | 2, 42, 72 | Not evaluated | Not evaluated | Not evaluated |
| TR30 | 2, 42, 72 | Not evaluated | Not evaluated | Not evaluated |
| TR20 | 2, 42, 72 | Not evaluated | Not evaluated | Not evaluated |
| TR10 | 2, 42, 72 | Not evaluated | Not evaluated | Not evaluated |

## Individual once-only held-out test results

**Not available:** no verified held-out test evaluations or validation-selected classification thresholds have been published for this model/dataset pair. Do not treat TR40 tuning values as per-seed final or test metrics.

## Public evidence and scripts

- `README.md`: this dated, evidence-limited progress snapshot. The corresponding `status.json` on this branch is historical and may not match the latest I1 training output.
- Stage 2 exact sparse crossfit verification, original graph/split hashes and risk-context evidence are retained on I1; the original author model remains pinned at the commit above.
- I1 Stage 3 tuning runner: `RGTAN_TSOCIAL_I1_STAGE3_GRID_V2_20261023/stage3_tr40_rgtan_grid_v2.py` (private I1 workspace). The most recent verified invocation used `--mode all`; it **only** performs tuning and winner freezing, not Stage 4 final training.
- Before test access: finish and independently audit the full 12-trial validation grid; inspect the trial-2 anomaly; freeze the winner; rebuild ratio-specific training-only risk contexts; complete and audit all 12 seed/ratio final checkpoints with validation-selected thresholds; then perform exactly one held-out evaluation per frozen run.

**Withheld privately:** original graph, frozen split arrays, risk-feature arrays, training/optimizer checkpoints and node-level predictions. No unverified completion claim or held-out test metric should be inferred from this README.
