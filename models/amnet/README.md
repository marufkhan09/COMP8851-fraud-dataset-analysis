# GDN under the unified benchmark protocol

Graph Decomposition Network (GDN), from Gao et al., *Alleviating Structural Distribution Shift in Graph Anomaly Detection*, WSDM 2023.

- Source: https://github.com/blacksingular/wsdm_GDN, commit `0383e0d94519dcf45142538f1118b3b440198c1a`
- The repository has **no LICENSE file**. The model code is used unmodified from a clone on the run machine and is not redistributed here; only our harness scripts are.
- Runs: 6 datasets x 4 training ratios (TR40/30/20/10) x 3 seeds (2, 42, 72) = **72 runs, all completed**.
- Hardware: Vast.ai, 1x RTX A6000 (48 GB), Ubuntu 24.04, driver 595.91.07, CUDA 13.2. Per-run GPU details in each `hardware.json`.

## Results

Test AUPRC (primary metric), mean ± std over 3 seeds. SplitGNN on the same splits shown for reference (mean of its 3 seeds, from `models/splitgnn/full_metrics_summary.csv`).

| Dataset | TR40 | TR30 | TR20 | TR10 | SplitGNN TR40 |
|---|---|---|---|---|---|
| YelpChi | 0.700 ± 0.008 | 0.669 ± 0.005 | 0.654 ± 0.007 | 0.598 ± 0.006 | 0.496 |
| Amazon | 0.858 ± 0.001 | 0.857 ± 0.001 | 0.851 ± 0.002 | 0.846 ± 0.000 | 0.550 |
| FDCompCN | 0.144 ± 0.006 | 0.171 ± 0.007 | 0.124 ± 0.002 | 0.124 ± 0.003 | 0.110 |
| T-Finance | 0.831 ± 0.001 | 0.831 ± 0.001 | 0.829 ± 0.002 | 0.818 ± 0.003 | 0.798 |
| Elliptic | 0.918 ± 0.002 | 0.915 ± 0.002 | 0.898 ± 0.002 | 0.877 ± 0.003 | 0.577 |
| T-Social (15% subsample) | 0.283 ± 0.001 | 0.268 ± 0.001 | 0.271 ± 0.001 | 0.270 ± 0.001 | 0.437 |

All seven metrics at TR40 (mean ± std over seeds):

| Dataset | AUPRC | AUROC | Macro-F1 | Fraud P | Fraud R | Fraud F1 | G-Mean |
|---|---|---|---|---|---|---|---|
| YelpChi | 0.700 ± 0.008 | 0.912 ± 0.002 | 0.787 ± 0.002 | 0.646 ± 0.018 | 0.624 ± 0.017 | 0.634 ± 0.002 | 0.767 ± 0.008 |
| Amazon | 0.858 ± 0.001 | 0.961 ± 0.000 | 0.917 ± 0.001 | 0.885 ± 0.004 | 0.809 ± 0.002 | 0.845 ± 0.001 | 0.896 ± 0.001 |
| FDCompCN | 0.144 ± 0.006 | 0.592 ± 0.006 | 0.497 ± 0.034 | 0.104 ± 0.091 | 0.125 ± 0.201 | 0.086 ± 0.122 | 0.221 ± 0.276 |
| T-Finance | 0.831 ± 0.001 | 0.947 ± 0.000 | 0.894 ± 0.001 | 0.882 ± 0.012 | 0.729 ± 0.011 | 0.798 ± 0.002 | 0.852 ± 0.006 |
| Elliptic | 0.918 ± 0.002 | 0.981 ± 0.001 | 0.920 ± 0.001 | 0.902 ± 0.018 | 0.812 ± 0.019 | 0.854 ± 0.003 | 0.897 ± 0.010 |
| T-Social | 0.283 ± 0.001 | 0.886 ± 0.000 | 0.663 ± 0.001 | 0.400 ± 0.002 | 0.303 ± 0.002 | 0.344 ± 0.002 | 0.546 ± 0.002 |

Full per-run numbers: `full_metrics_summary.csv`. Per-ratio means and standard deviations for every metric: `aggregate_summary.csv`. Per-epoch timing and peak memory: `timing_summary.csv`. Per-run evidence bundles: `results/unified/gdn_results/<dataset>/unified/tr<XX>/seed_<N>/`.

## Findings and limitations (report these)

1. **Still improving at the epoch cap.** 44 of 72 runs reached the protocol's 100-epoch maximum with the best validation epoch at 95 or later (all 12 YelpChi and T-Finance runs, 11 of 12 Amazon, 6 Elliptic and 3 FDCompCN). GDN's default learning rate (1e-3, and 1e-5 for the feature table) converges slowly, so these numbers are a lower bound on GDN's performance. The cap is the shared protocol and was not changed.
2. **FDCompCN is close to chance.** Fraud is 10.5% of nodes, so chance AUPRC is about 0.105; GDN reaches 0.12 to 0.17 (AUROC 0.54 to 0.60). At TR40, seed 42 recovered no fraud on the test set (fraud precision and recall both 0 at the selected threshold) and seed 72 recalled only 1.8%. Several TR10/TR20 runs early-stopped within the first 10 to 40 epochs. SplitGNN shows the same difficulty on this dataset (AUROC below 0.5, zero fraud predictions in 10 of 12 runs).
3. **GDN fails on Elliptic's raw features.** On the standardized features as distributed (range -13 to 289), training produces NaN parameters within 4 batches. Diagnosis (`diagnostics/elliptic_raw_features.log`): the loss terms stay finite, but GDN's neighbour constraint applies a softmax to raw feature values; with values this large the softmax underflows to exactly 0, and the KL gradient through a zero target is non-finite. Per-feature min-max scaling to [0, 1] removes it (`diagnostics/elliptic_column_minmax.log`). This is a robustness finding about GDN, not only a setup detail.
4. **Homogeneous graphs fill all three relation slots.** GDN is hard-wired to exactly three relations. T-Finance, T-Social and Elliptic have one, so the same adjacency list is used in all three slots (`adapter_version: homo_replicated_v1`). GDN then runs three aggregators over one graph, which it was not designed for. YelpChi, Amazon and FDCompCN (invest / provide / sale) use their native three relations.
5. **T-Social is a 15% stratified node subsample** (867,160 nodes, 3,297,570 edges, 3.01% fraud), identical to SplitGNN's view, not the full 5.78M-node graph. GDN scores lower than SplitGNN here (AUPRC 0.28 vs 0.44) while having a similar AUROC (0.89 vs 0.91). Every T-Social run early-stopped (best epoch 10 to 50).
6. **Seed variance is very small** (AUPRC std at most 0.009 outside FDCompCN). Splits are fixed (split seed 2) and only model initialisation varies with the seed, so the std reflects initialisation sensitivity only, not split sensitivity.
7. **Amazon split includes the 3,305 nodes the authors treat as unlabelled** (indices 0 to 3304, label 0 in the .mat file), because the team's canonical splits do. GDN's own code excludes them. Kept for comparability with the other models.

## Protocol

Identical for every model in the benchmark:

- Canonical nested splits from the team's `models/splitgnn/splits/generate_splits.py` (split seed 2, stratified, valid 20%, test 40%, train pool 40%, TR40 ⊃ TR30 ⊃ TR20 ⊃ TR10), generated over labelled nodes only.
- Every run checks its masks against SplitGNN's stored `split_indices.npz` and stops on any difference. **All 72 runs matched** (`split_identical_to_splitgnn: true` in each `protocol_identity.json`), and the split-file SHA256 equals SplitGNN's recorded `split_sha256` for every dataset.
- Max 100 epochs, early stopping patience 20, validation every epoch, checkpoint on best validation AUPRC.
- Decision threshold: best validation Macro-F1 over 0.05, 0.10, ..., 0.95. No test data used for any selection; the test set is evaluated once.
- Every run reloads its best checkpoint and confirms the validation AUPRC matches the logged value exactly (all 72 matched).
- GDN hyperparameters: author defaults from `config/gdn_yelpchi.yml` (the only config the authors ship) for all datasets. No tuning.

## Deviations from GDN's native setup

Recorded per run in `run_config.yml` under `protocol_deviations_from_native`.

Protocol harness (all datasets):
- Team canonical splits instead of the author's `train_test_split` 40/20/40.
- Validation every epoch instead of every 5; checkpoint on validation AUPRC instead of AUROC.
- Max 100 epochs with patience 20 instead of a fixed 101 epochs.
- Threshold tuned on validation Macro-F1 instead of fixed 0.2.
- Training node order: canonical train mask shuffled once with `RandomState(2)` (the author's order comes from `train_test_split(random_state=2)`).
- GDN's prototype vectors are snapshotted before and restored after each evaluation pass, so per-epoch validation does not change training dynamics (the forward output does not depend on them).
- Environment: Python 3.10, torch 2.5.1+cu121, numpy 1.26.4, scikit-learn 1.7.2, scipy 1.15.3, instead of the README pins (torch 1.4, numpy 1.16.4, scikit-learn 0.21rc2). The code uses only APIs that are unchanged across these versions.

Dataset-specific (data translation only; model, losses and mechanism untouched):

| Dataset | Relations | Features | topk |
|---|---|---|---|
| YelpChi | native 3 | raw (author) | 10 |
| Amazon | native 3 | GDN `normalize()` (author) | 10 |
| FDCompCN | native 3 (invest, provide, sale) | raw, already in [0, 1] | 10 |
| T-Finance | homo x 3 | GDN `normalize()`; raw values reach 508,174 | **4** |
| T-Social | homo x 3 | GDN `normalize()`; raw values reach 850,934 | **4** |
| Elliptic | homo x 3 | per-feature min-max to [0, 1] (label-free); raw causes NaN, see finding 3 | 10 |

- **Normalization for T-Finance and T-Social:** GDN applies softmax to feature values in its constraint losses, so unscaled values saturate it. GDN's authors handle Amazon's raw count features with their own `utils.normalize()` (row normalization); the same function is applied here. It cannot be used on Elliptic (negative features make row sums near zero), hence min-max there.
- **topk = 4 for T-Finance and T-Social:** GDN's default topk = 10 leaves no features for its second constraint step on 10-feature datasets (the run refuses to start). 4/10 keeps the author's Amazon ratio of 10/25.

## Data sources and identity

| Dataset | Source | SHA256 |
|---|---|---|
| YelpChi | CARE-GNN repo `data/YelpChi.zip` → `YelpChi.mat` | `fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42` (identical to the team's SplitGNN copy) |
| Amazon | CARE-GNN repo `data/Amazon.zip` → `Amazon.mat` | `4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63` (identical to the team's SplitGNN copy) |
| FDCompCN | SplitGNN repo `data/FDCompCN.zip` → `comp.dgl` | `e252b9a6b619b28a7bf6d9f5b16aacc232d43207d87ac26cd5160fb0d8a98baa` (equals SplitGNN's recorded `dgl_sha256`) |
| T-Finance | BWGNN authors' Google Drive, `tfinance.zip` → `tfinance` | `b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700` |
| T-Social | BWGNN authors' Google Drive, `tsocial.zip` (744,239,223 bytes) → `tsocial` | `8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0` (equals the team's recorded official hash) |
| Elliptic | `https://data.pyg.org/datasets/elliptic/` (mirror of the Kaggle CSVs) | features `fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0`, edgelist `a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e`, classes `93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493` |

These are also recorded in each run's `dataset_manifest.json` and `protocol_identity.json`. Node order, labels and the T-Social subsample reproduce SplitGNN's adapters in `models/splitgnn/adapters/` exactly; the split check above confirms it node for node.

## Scripts (`scripts/`)

| File | Purpose |
|---|---|
| `run_gdn_unified.py` | One run: loads data, verifies canonical splits against SplitGNN, replays GDN's training step unchanged, applies the protocol, writes the evidence bundle. |
| `run_gdn_unified_v1.py` | The version used for YelpChi, Amazon and FDCompCN. Same training and evaluation path; the later version only adds bundle-level feature transforms and per-dataset deviation notes. |
| `batch_run_gdn.py` | Idempotent loop over ratio x seed; skips finished runs. |
| `build_gdn_views.py` | Adapter: converts FDCompCN, T-Finance, T-Social and Elliptic into GDN's adjacency-list format, reproducing SplitGNN's node order. |
| `summarize_gdn.py` | Builds the three summary CSVs and flags runs to report. |
| `diag_nan.py` | Replays GDN's training step batch by batch to locate non-finite values (used for finding 3). |

Evidence bundle per run: `run_config.yml`, `protocol_identity.json`, `environment.txt`, `hardware.json`, `dataset_manifest.json`, `split_indices.npz`, `split_summary.json`, `epoch_times.csv`, `validation_metrics.csv`, `test_metrics.json`, `summary.json`, `terminal.log`, plus `valid_probs.npy` and `test_probs.npy`. Checkpoints (`checkpoints/best_model.pt`) are kept off GitHub because of size (Elliptic checkpoints exceed GitHub's 100 MB file limit) and archived separately.

## Native reproduction (before the unified runs)

GDN's own pipeline on YelpChi with its own 40/20/40 split, one seed: AUROC 0.906, Macro-F1 0.730, G-Mean 0.831, against the paper's AUROC 0.903 ± 0.008, Macro-F1 0.761 ± 0.006, G-Mean 0.808 ± 0.001 (5-run mean). AUROC, the threshold-free metric, reproduces; the Macro-F1 / G-Mean trade-off reflects the fixed 0.2 threshold on a single seed. Log: `native_reproduction/native_yelp.log`.

## How to reproduce

```bash
# environments
conda create -y -n gdn python=3.10 && conda activate gdn
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2" scipy scikit-learn pyyaml
conda create -y -n gdn-adapt python=3.10 && conda activate gdn-adapt
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/repo.html
pip install "numpy<2" pandas gdown pyyaml pydantic

# code: GDN at /workspace/gdn (commit above), team repo (branch raiyan) at /workspace/team,
# these scripts at /workspace/gdn_vast

# YelpChi / Amazon: GDN's own preprocessing (Amazon lines are commented out in utils/data_process.py
# and were run via the same sparse_to_adjlist calls)
cd /workspace/gdn && python -m utils.data_process

# other datasets (gdn-adapt env)
python build_gdn_views.py --dataset comp     --src /workspace/raw/comp.dgl
python build_gdn_views.py --dataset tfinance --src /workspace/raw/tfinance --feature_transform gdn_row_normalize
python build_gdn_views.py --dataset tsocial  --src /workspace/raw/tsocial  --feature_transform gdn_row_normalize
python build_gdn_views.py --dataset elliptic --src /workspace/raw/elliptic --prescale column_minmax

# runs (gdn env)
python batch_run_gdn.py --dataset yelp
python batch_run_gdn.py --dataset amazon
python batch_run_gdn.py --dataset comp
python batch_run_gdn.py --dataset tfinance --extra "--topk 4"
python batch_run_gdn.py --dataset tsocial  --extra "--topk 4"
python batch_run_gdn.py --dataset elliptic
python summarize_gdn.py
```
