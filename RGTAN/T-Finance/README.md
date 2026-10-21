# RGTAN × T-Finance — Candidate validation complete (NOT approved benchmark)

**Status:** 12/12 TR40 *validation-only* tuning candidates and 12/12 frozen *validation-only* candidate trainings. **No held-out test scoring has been performed.** The GPU work is complete for this provisional configuration. This GitHub directory is a public evidence record, not a private GPU image or full restartable checkpoint archive.

## Explicit methodological deviation and approval

- Original RGTAN model architecture is pinned to upstream [AI4Risk/antifraud commit](https://github.com/AI4Risk/antifraud/tree/65e583f03ac5c5983e49cc35179fe7a65908b9c1) `65e583f03ac5c5983e49cc35179fe7a65908b9c1`.
- Canonical BWGNN-released T-Finance source is preserved: 39,357 nodes, 10 features, **42,445,086 stored directed edges**, 1,804 fraud labels; canonical source SHA-256 `b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700`.
- Original frozen Assessment 2 split SHA-256 `de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1`; split seed 2. Four nested training fractions TR40, TR30, TR20, TR10 with fixed validation/test membership, seeds 2,42,72.
- Risk features use full canonical topology, training-only five-fold OOF label contexts and **exact two-hop** statistics with training-only OOF normalisation. Validation/test targets do not create model risk statistics.
- **Critical difference:** Original two-layer *full-neighbour* propagation was computationally infeasible on this graph for dense nodes (one seed expands >33 million edges). The RGTAN model uses *bounded neighbour fanouts `[10,10]`* at batch 1024. It is a **candidate adaptation**; **supervisor approval is required** before reporting it as a directly comparable accepted benchmark.
- TR40 12-trial validation-only tuning; best candidate **trial 7**, validation AUPRC **0.888112**, selected configuration `{'base_lr': 0.001, 'drop': [0.2, 0.1], 'hidden_dim': 64}`.
- Four ratios × three seeds = 12 validation-frozen candidate trainings. Checkpoints selected by validation AUPRC and thresholds by validation macro-F1.
- **Held-out test evaluation:** NOT PERFORMED, NOT AUTHORISED. Do not label the figures below as test-set performance.

## Validation performance — candidate results only

Three-seed mean ± **sample standard deviation** (ddof=1).

| Training ratio | Validation AUPRC | Validation macro-F1 |
|---|---:|---:|
| TR40 | 0.840388 ± 0.037157 | 0.895616 ± 0.017511 |
| TR30 | 0.865161 ± 0.031851 | 0.907152 ± 0.016315 |
| TR20 | 0.860401 ± 0.028896 | 0.907414 ± 0.013502 |
| TR10 | 0.810456 ± 0.061881 | 0.881548 ± 0.033606 |

## Reproducibility / evidence

- `scripts/` — model-specific source scripts, launchers, preflight and crossfit code used on I2.
- `risk_features/` — four SHA-256-pinned crossfit JSON manifests, excluding the underlying private NPZ arrays.
- `results/tuning/` — 12 trial JSON records, training histories, validation score arrays and frozen winner.
- `results/finals/` — 12 candidate training records, histories, validation score arrays, frozen config and candidate validation summary. **No final test scores or test predictions.**
- `evidence/` — instance validation diagnostics, logs, environment inventory, audit and publication SHA-256 manifest.

Raw T-Finance graph, frozen split arrays, crossfit data arrays, model checkpoint binaries and Python/CUDA environments are excluded from the public repo. Stored checkpoint SHA-256 references can be verified only while the original private files are retained.
