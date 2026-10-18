# H2-FDetector: unified benchmark runs

H2-FDetector (Shi et al., *H2-FDetector: A GNN-based Fraud Detector with Homophilic and Heterophilic Connections*, WWW 2022), run under the group's unified protocol on all six benchmark datasets.

- Model source: https://github.com/shifengzhao/H2-FDetector, commit `10710d08a8461f418da2db8625a6f23199051785` (2 June 2022)
- Hardware: Vast.ai, 1x NVIDIA RTX A6000 (48 GB), 80 GB disk
- Environment (same stack as the SplitGNN runs): Python 3.12.15, torch 2.2.1+cu121, DGL 2.4.0+cu121, numpy 1.26.4, scipy 1.17.1, scikit-learn 1.9.1. Exact versions per run are in each run's `environment.txt`.

## Status

| Dataset | Unified runs | Data / split check | Notes |
|---|---|---|---|
| YelpChi | 12/12 | splits identical to SplitGNN | `YelpChi.mat` SHA256 matches the repo's shipped copy |
| Amazon | 12/12 | splits identical to SplitGNN | `Amazon.mat` SHA256 matches the repo's shipped copy |
| FDCompCN | 12/12 | splits identical to SplitGNN | `comp.dgl` from FDCompCN.zip, node type renamed to `r` |
| Elliptic | 12/12 | splits identical to SplitGNN | unknown-label nodes excluded, as in SplitGNN |
| T-Social | 12/12 | identical 15% subsample (867,160 nodes / 3,297,570 edges) and splits as SplitGNN | shared limitation: subsample, not full graph |
| T-Finance (full) | 0/12 | splits identical to SplitGNN | **CUDA out of memory on 48 GB before the first epoch completed** (see Limitations) |
| T-Finance 15% (`tfinance_sub15`) | 12/12 | own canonical splits (no SplitGNN reference exists) | **supplementary**, same subsampling procedure as T-Social; not like-for-like with SplitGNN's full-graph T-Finance |

72 completed runs, 12 documented failures. "Splits identical" means the train/valid/test masks were compared array-for-array with SplitGNN's `split_indices.npz` at all four ratios (file hashes are not comparable because `np.savez` embeds a timestamp).

## Changes to the authors' code (compatibility only)

The model, loss, optimiser and all hyperparameters are unchanged. The full diff is in `model_patch/h2fd_local_changes.diff`.

1. `src/data_preprocess.py` lines 75 and 145: `add_self_loop(t)` changed to `add_self_loop(etype=t)`. DGL >= 1.0 reads the second positional argument as an edge-feature name, so the original call raised an error. The return value is still not assigned, exactly as in the original code, so these lines remain the no-ops they were under DGL 0.8 and the graph the model sees is unchanged.
2. `config/yelp.yaml` and `config/amazon.yaml`: `cuda: 1` / `cuda: 2` changed to `cuda: 0`. These are GPU indices from the authors' multi-GPU machine; this instance has one GPU. Hardware setting, not a hyperparameter.

README errors found in the authors' repository (not changed, documented only): the requirements lines are swapped (Python 3.9.12 is listed as "pytorch" with a DGL link, DGL with a PyTorch link), and the unzip instruction is misspelled `upzip`.

## Native reproduction (authors' code, split and settings)

Authors' 40/20/40 split (`random_state=2`), seed 20, up to 1000 epochs, early stopping on validation AUC (patience 50 for YelpChi, 200 for Amazon), default 0.5 threshold. Logs: `model_patch/native_yelp.log`, `model_patch/native_amazon.log`.

| Dataset | AUC | F1-macro | G-Mean | Recall | Epochs (best) |
|---|---|---|---|---|---|
| YelpChi | 0.806 | 0.552 | 0.712 | 0.930 | 277 (227) |
| Amazon | 0.956 | 0.830 | 0.902 | 0.870 | 480 |

Amazon reproduces closely; YelpChi is below the published figure (single seed, newer library versions). Check the paper's results table for the exact reference values before citing.

## Unified protocol

Identical to the SplitGNN unified runs (`models/splitgnn/scripts/run_evidence.py`):

- Canonical nested splits from `models/splitgnn/splits/generate_splits.py` (split seed 2, stratified; validation 20%, test 40%, TR40 ⊃ TR30 ⊃ TR20 ⊃ TR10 drawn from the remaining 40%)
- Seeds 2, 42, 72; max 100 epochs; early stopping patience 20
- Checkpoint = best validation AUPRC; threshold = best validation macro-F1 over `np.arange(0.05, 0.96, 0.05)`; no test data used for any selection
- Seven metrics: AUPRC (primary), AUC, macro-F1, fraud precision, fraud recall, fraud F1, G-Mean (same definitions as `recompute_full_metrics.py`)
- Author hyperparameters, no tuning. YelpChi and Amazon use the authors' configs; the other datasets have no author config and use `yelp.yaml` unchanged apart from the dataset name (see `model_patch/*.yaml`)

### Adapter (data translation only)

H2-FDetector requires node type `r`, a `homo` edge type (used for its edge-label loss) and at least one further relation; relations are read from the graph, so no model change is needed. For each training ratio the adapter swaps in the canonical node masks and recomputes the `homo` edge labels and edge train mask with a vectorised version of the authors' `generate_edges_labels()`. Its output was verified **bit-identical** to the authors' function on the full YelpChi (7,693,958 edges) and Amazon (8,796,784 edges) graphs, and it runs in under a second instead of about 24 minutes.

Leakage check: edge labels are computed for all edges, but the model only reads them through the edge train mask (both endpoints in the training set), and node labels only through the node train mask (`model.py` lines 115-130, 196-219). Validation and test labels never reach the loss.

T-Finance and T-Social use the original BWGNN release (the files SplitGNN used). The GADBench release adds reverse edges and a self-loop on every node and did not reproduce SplitGNN's split or T-Social subsample; the safety checks rejected it.

## Results (mean ± std over 3 seeds, test set)

Full tables: `summary/aggregated_metrics.csv` (all 7 metrics), `summary/full_metrics_summary.csv` (per run), `summary/health_report.md`.

| Dataset | Ratio | AUPRC | AUC | Macro-F1 | G-Mean |
|---|---|---|---|---|---|
| YelpChi | TR40 | 0.352 ± 0.099 | 0.775 ± 0.058 | 0.629 ± 0.070 | 0.677 ± 0.010 |
| YelpChi | TR10 | 0.367 ± 0.111 | 0.767 ± 0.055 | 0.645 ± 0.069 | 0.661 ± 0.012 |
| Amazon | TR40 | 0.672 ± 0.102 | 0.919 ± 0.005 | 0.854 ± 0.041 | 0.874 ± 0.020 |
| Amazon | TR10 | 0.591 ± 0.214 | 0.852 ± 0.075 | 0.825 ± 0.094 | 0.786 ± 0.091 |
| FDCompCN | TR40 | 0.131 ± 0.018 | 0.542 ± 0.031 | 0.495 ± 0.049 | 0.387 ± 0.110 |
| FDCompCN | TR10 | 0.129 ± 0.013 | 0.549 ± 0.024 | 0.503 ± 0.028 | 0.212 ± 0.198 |
| Elliptic | TR40 | 0.138 ± 0.012 | 0.558 ± 0.033 | 0.535 ± 0.029 | 0.302 ± 0.108 |
| Elliptic | TR10 | 0.132 ± 0.012 | 0.553 ± 0.029 | 0.545 ± 0.013 | 0.356 ± 0.048 |
| T-Social (15%) | TR40 | 0.634 ± 0.021 | 0.936 ± 0.002 | 0.812 ± 0.008 | 0.800 ± 0.011 |
| T-Social (15%) | TR10 | 0.638 ± 0.006 | 0.938 ± 0.002 | 0.814 ± 0.002 | 0.798 ± 0.017 |
| T-Finance 15% (suppl.) | TR40 | 0.846 ± 0.009 | 0.956 ± 0.004 | 0.889 ± 0.012 | 0.839 ± 0.031 |
| T-Finance 15% (suppl.) | TR10 | 0.856 ± 0.010 | 0.957 ± 0.010 | 0.899 ± 0.012 | 0.887 ± 0.012 |

## Health checks (from `summary/health_report.md`)

- Checkpoint integrity: all 72 runs reproduce their best validation AUPRC exactly after reloading the saved checkpoint.
- Collapsed runs (no fraud predicted): FDCompCN TR10 seed 2 and TR20 seed 72.
- Near-random datasets: FDCompCN and Elliptic (AUC about 0.52-0.60, AUPRC close to the ~10% fraud rate). SplitGNN also failed on FDCompCN. A possible contributor on Elliptic is the authors' row normalisation (`utils.normalize`, divides each row by its sum + 0.01) applied to standardised features that contain negative values; this is a hypothesis, not tested.
- Weak seeds (AUPRC > 0.10 below the other seeds): Amazon seed 42 at TR10/TR30/TR40; YelpChi TR10 seed 2, TR30 seed 72, TR40 seed 42. These runs peaked at epochs 2-14 and stopped 20 epochs later; with the authors' learning rate (0.1) validation AUPRC is noisy, and the native YelpChi run found its best epoch at 227. All are kept in the averages.
- Threshold at the grid edge in 10 runs (mostly 0.95), so macro-F1 and recall may be slightly underestimated for those runs; the grid is shared with SplitGNN.
- Timing: no material anomalies (two runs with sub-second blips).

## Efficiency

Per-run and per-dataset timing and peak memory: `summary/timing_summary.csv`, `summary/timing_by_dataset.csv`. Indicative values: YelpChi about 3.5 s/epoch and 26 GB; Amazon about 35 s/epoch and 27 GB; T-Social (15%) about 1.7 s/epoch and 9.4 GB. Cost grows with graph density: the authors' message passing uses Python UDF reduce functions, which DGL processes in degree buckets, so Amazon (about 600 neighbours per node on its densest relation) is roughly 10x slower per epoch than YelpChi despite a similar edge count.

## Limitations

1. **T-Finance (full graph): not runnable on a 48 GB GPU.** All 12 runs failed with CUDA OOM in the first forward pass of epoch 0 (`model.py` line 75, edge features): 39 GB allocated plus a further 15.18 GB requested (47.4 GB total). Not fragmentation (11 MB reserved but unused). Evidence: `logs/batch2_console.log` and the `tfinance/` run folders (header-only `epoch_times.csv`).
2. **T-Finance 15% subsample** (`tfinance_sub15`, 5,904 nodes / 902,212 edges, fraud rate 4.59% vs 4.58% full) was run as a documented supplementary result with the T-Social procedure. SplitGNN was not re-run on it, so it is not a like-for-like comparison.
3. **T-Social** uses the group's 15% stratified subsample (shared with SplitGNN).
4. Datasets without an author config use YelpChi defaults with no tuning.
5. The shared threshold grid tops out at 0.95.

## Reproducing

```
# environment (conda, see environment.txt in any run folder)
pip install torch==2.2.1 numpy==1.26.4 --extra-index-url https://download.pytorch.org/whl/cu121
pip install dgl==2.4.0 scikit-learn scipy pyyaml numpy==1.26.4 -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html

# native graphs (authors' preprocessing, with the add_self_loop fix)
cd H2-FDetector/src && python data_preprocess.py --dataset yelp && python data_preprocess.py --dataset amazon

# canonical views
python scripts/h2fd_build_views.py --dataset yelp
python scripts/h2fd_build_views.py --dataset amazon
python scripts/h2fd_build_views_ext.py --dataset comp
python scripts/h2fd_build_views_ext.py --dataset elliptic
python scripts/h2fd_build_views_ext.py --dataset tsocial  --data-dir <BWGNN dataset dir>
python scripts/h2fd_build_views_ext.py --dataset tfinance --data-dir <BWGNN dataset dir>
python scripts/h2fd_build_tfinance_sub.py --dataset tfinance_sub15 --data-dir <BWGNN dataset dir>

# runs (idempotent, restartable) and summary tables
python scripts/h2fd_batch_run.py --datasets yelp amazon comp elliptic tsocial tfinance tfinance_sub15
python3 scripts/h2fd_collect_results.py --results-root ../../results/unified/h2fdetector_results --out-dir summary
```

## Folder layout

```
models/h2fdetector/
  README.md
  scripts/        view builders, single-run script, batch loop, results collector, data probe
  views/          per-dataset view manifests (hashes, counts, split cross-checks) and split files
  model_patch/    diff of the two compatibility edits, configs used, native-run logs, model commit
  summary/        aggregated tables, per-run tables, timing tables, health report
  logs/           batch logs, view-build logs, smoke test log
results/unified/h2fdetector_results/<dataset>/unified/tr<R>/seed_<S>/
  run_config.yml, protocol_identity.json, environment.txt, hardware.json, dataset_manifest.json,
  dataset_view_manifest.json, split_indices.npz, split_summary.json, epoch_times.csv,
  validation_metrics.csv, test_metrics.json, full_test_metrics.json, predictions.npz,
  summary.json, terminal.log, checkpoints/best_model.pt
```
