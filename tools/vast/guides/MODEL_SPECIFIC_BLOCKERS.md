# New model design gates: do not silently rewrite research methods

## RGTAN (2025) — AntiFraud

Pinned to `AI4Risk/antifraud`; originally uses PyTorch 1.12.1+cu113, DGL CUDA 0.8.1, older numpy/scipy/sklearn. Author runner `methods/rgtan/rgtan_main.py` trains `n_fold: 5` with `MultiLayerFullNeighborSampler`, default config has batch_size 64, 2 layers, up to 15 epochs, and uses risk/neighbor categorical features. Native setup is **not** a direct replacement for canonical fixed training+validation IDs. Leakage-safe risk-statistic feature engineering must use only permitted training labels; run an independent feature provenance audit. Ask Venus whether native folds are considered a core mechanism or may be aligned to the shared validation fold. Do not shrink neighbour fanout without permission. Audit full-neighbour scale on T-Finance/T-Social.

## SAWGAD (2026) — synthetic-anomaly weak supervision

Pinned to `yj-zhou/SAWGAD`, Python >=3.12,<3.13; PyTorch 2.4.0, DGL 2.4.0 CUDA 11.8 as upstream `uv` project. Native default: 30 labelled anomalies, contamination_rate 0.01, 100 warmup epochs with synthetic graph perturbations, plus 350 supervised full epochs. The default `build_weakly_supervised_split` **does not match** canonical TR40/TR30/TR20/TR10 splits: it constructs an 80/10/10 split and moves some positives to validation/test, and marks chosen contamination as normal. Run native author-mode replication separately; adapt to canonical protocol only after conceptual approval. Do not conflate weakly supervised experiments with the same-label-budget supervised ladder. T-Social's repeated graph `deepcopy` and full-graph attention is a memory risk.

## EOGFD (2026) — evolution-optimised dynamic polynomial convolution

Pinned to `BERA-wx/EOGFD`. Model file exports `EOGFDModel` and `EOGFDHeteroModel`, while `run_eogfd.py` imports `EOGFD` and `EOGFDHetero`: the names do **not** match. The author runner also performs test metrics within its epoch loop; canonical TR40 tuning must never read/test-score inside training. Imported-runner repair should only address plumbing (e.g. alias import), and the canonical evaluator must use validation-only checkpoint selection. The code constructs evolutionary population of node-specific polynomial coefficients; memory may be catastrophic on the 5.78M-node T-Social graph. Profiling must not secretly change population size, loss, polynomial order or graph semantics. No verified upstream dependency lock was available in the inspected repo; create a reproducible environment only after a native smoke is possible.

## All three

Canonical eligibility is a **decision**, not a filename. A passed smoke run is necessary but insufficient: source hash, split hash, training supervision comparability, test isolation, and consistent evaluation must be verified before calling evidence canonical. Failed pairs are acceptable benchmark outcomes when their cause and attempted settings are documented. Direct comparison with an unmodified native score must be explicitly separated from canonical comparable evidence.
