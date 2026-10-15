# Reconstruction of PMP, GAAP, BWGNN and current evidence status

## Where the existing evidence is

- Team repository: `https://github.com/marufkhan09/COMP8851-fraud-dataset-analysis`, branch `Pathik`, pinned SHA in `configs/source_lock.json`.
- BWGNN: `bwgnn notebook and docx/BWGNN_ALL_PY_EXPORT/02_bwgnn_yelp_final.py` through `08_bwgnn_tsocial_final.py`, `90/91` reports, `SHA256SUMS.txt`, `adapters/BWGNN_gpu.py`.
- PMP: `pmp notebooks and docx/` with `smoke/`, `tuning/`, `final/`, `recovery/` and `SHA256SUMS.txt`.
- GAAP: `gaap evidence/GAAP_AMAZON_COMPLETE_20260918/`, `GAAP_TFINANCE_COMPLETE_20260918/`, `GAAP_FDCOMPCN_COMPLETE_20260919/`, and code/outputs under `models/gaap/evidence/`. The author code was pinned historically at `6a7dbb0447c4897504525de49e41a0526ee777f8` for GAAP work.
- Actual canonical `shared/splits/` on the inspected team GitHub Pathik branch contains only `.gitkeep`, **not** complete split assets. Search local/Vast archives and verify SHA256 before reuse.

## How the prior protocol actually worked

1. Author-source and dependency environment isolated per model (A6000 if possible).
2. Freeze upstream dataset source and exact split registry (split seed 2), raw label counts and input graph shape. Run smoke with **no test access**.
3. Tune TR40 with seed 2, 12 candidate trials, up to 100 epochs, patience 20, checkpoint chosen solely by validation AUPRC.
4. Store frozen TR40 winner; run 4 ratios × 3 training seeds = 12 final runs on fixed validation/test IDs, never retuning lower ratios.
5. Save raw per-run, per-ratio summary, full config, manifest, provenance, GPU/time logs and SHA256 files; do not replace old result archives.

## BWGNN

- Submitted final Ass2 reports 72/72 final model/dataset/ratio/seed combinations across six datasets. T-Social was completed on A6000. It is the best template for coverage and evidence packaging.
- Historical issue: FDCompCN had a relation-duplication problem that was subsequently reported corrected, but exact correction and hash still need confirmation from the underlying artifacts.
- Earlier T-Social result used **22.5 GB peak GPU memory** and about **160 s training** under its timing boundary; don't assume other models share its speed.

## PMP

- LA-SAGE-S retained full-neighbour message passing; T-Social used batch 256 and reportedly **9,033 training batches/epoch**. Its exact validation loop was much smaller (283 validation batches in recovered logs).
- Environment needed CUDA/libcusparse and Setuptools compatibility repairs; these may not transfer to new images. Keep author version isolated.
- Recovered Vast log shows **Trial 09** best validation AUPRC **0.94977266** around epoch 84, with later tuning incomplete in the visible logs. Earlier Amazon tuning completed 12/12, winner trial 2, LR 0.01, weight_decay 1e-5. Do **not** infer three-seed T-Social final results from incomplete logs alone.

## GAAP

- Original GAAP repository historically pinned at `6a7dbb0447c4897504525de49e41a0526ee777f8` for past work. A6000 dependency fixes included NVRTC, Pydantic and Setuptools, without changing architecture/loss.
- Earlier T-Social smoke had validation AUPRC ~0.926813; trial 2 at validation AUPRC ~0.99433 in available logs. Later OOM and stalled/partial instance recovery were reported. This is **not proof** of a completed 12-trial sweep or complete 12 final runs.
- Previous recovery of one copied GAAP environment on i3 found **6.9 GB of packages but no actual GAAP repository source, T-Social assets, scripts or evidence** in the copied directories. Reusing the Python environment alone is not enough. Do not graft Python 3.12 over broken Python 3.10 symlinks.

## Canonical T-Social historical verification targets

- Source file historical path: `/workspace/pmp_vast/shared/tsocial/tsocial` (DGL graph binary, no extension).
- Frozen split file: `/workspace/pmp_vast/shared/tsocial/tsocial_seed2_nested_splits.npz`.
- Source SHA256: `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0`.
- Split SHA256: `b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519`.
- Expected processed graph scale in Ass2: 5,781,065 nodes, 10 features, about 3.01% fraud, 146,211,016 stored directed entries / 73,105,508 logical edges. A filesystem hash is necessary but full schema/label graph audit is also required.

## Conflicting evidence and how to handle it

The submitted final Ass2 text claims PMP and GAAP T-Social multi-ratio execution was complete, but its label-scarcity results section says that only BWGNN has the full T-Social ladder and PMP/GAAP are TR40-only. Later recovered logs likewise show unfinished tuning. **Before using these results as final in a new project, reconcile source-of-truth raw artifacts.** Leave them marked `CLAIMED_IN_REPORT__NOT_YET_VERIFIED_FROM_ARTIFACTS` rather than asserting completion or rerunning blindly.

Other outstanding work: T-Finance fraud count corrected to 1,804 and canonical split must be revalidated; †/* models are not comparable without canonicalisation; Elliptic retains chronological time-step split; T-Social failures must be recorded as failures, not zeros.
