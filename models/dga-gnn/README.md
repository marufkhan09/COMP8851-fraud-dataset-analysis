# DGA-GNN on the COMP8851 fraud-graph benchmark

Owner: Md Maruf Khan.

## The model

DGA-GNN, "Dynamic Grouping Aggregation GNN for Fraud Detection" (Duan et al.,
AAAI 2024). Official code: https://github.com/AtwoodDuan/DGA-GNN, commit
`0907392f6060e18230339ca30eaca0c917414820`.

Two ideas matter for fraud graphs with low homophily:
- **Dynamic grouping.** During training the model predicts a label for every node
  and uses those predictions to split each node's neighbours into groups, which are
  aggregated separately. The grouping therefore changes as the model improves.
- **Bin encoding of features.** Numeric features are encoded into `k` bins. The
  authors use `k=32` for their two largest datasets (YelpChi, T-Social) and `k=4`
  for the smaller ones.

## Status

| Dataset | Support | Status |
|---|---|---|
| YelpChi | native | seed-2 native-protocol run |
| Amazon | native | seed-2 native-protocol run |
| T-Finance | native | seed-2 native-protocol run |
| Elliptic | native | seed-2 native-protocol run |
| T-Social | native | seed-2 native run completed |
| FDCompCN | adapter (not native) | TR40 to TR10, plus extra seeds, on a provisional split |

"Native protocol" here means the authors' default hyperparameters, a single seed
(2), run through the protocol-safe adapter. These runs are not the frozen COMP8851
TR40 to TR10 grid, so the report marks them separately and does not rank them
against the unified-protocol models.

## Results

| Dataset | Test AUPRC (APS) | Test AUROC | Test fraud-F1 |
|---|---:|---:|---:|
| YelpChi | 0.922 | 0.977 | 0.848 |
| Amazon | 0.927 | 0.982 | 0.871 |
| T-Finance | 0.904 | 0.976 | 0.846 |
| Elliptic | 0.813 | 0.942 | 0.774 |
| T-Social | 0.9805 | 0.9988 | see `tsocial/README.md` |

Source: `evidence/benchmark_parallel_default_seed2/accuracy_summary.csv` (first four
rows) and `tsocial/README.md`. The four default runs shared one GPU, so their wall
time and memory are marked not reportable for the efficiency question
(`benchmark_adapter/PROTOCOL.md`).

**FDCompCN.** TR40 and TR30 train normally (test AUC about 0.78). TR20 and TR10
collapse to near chance (test AUC about 0.54), with the dynamic-grouping proportion
`g0` stuck near zero, so the grouping mechanism does not activate when few fraud
labels are available. Seeds 42 and 72 show that the result also depends on the
seed, and TR30 mostly fails except for seed 2. Details: `fdcompcn/README.md` and
`fdcompcn/evidence/benchmark/seed_variance/`.

**T-Social.** DGA-GNN is the only one of PC-GNN, DGA-GNN and GAGA that completed
T-Social (5,781,065 nodes, 146,211,016 directed edges). Its preprocessing is
vectorised. Best validation checkpoint at epoch 97, test AUC 0.9988, APS 0.9805.
PC-GNN and GAGA did not finish on the same hardware (see their READMEs).

## How to reproduce

1. Get the code at the recorded commit (see `dga_gnn_commit.txt`) and the data
   (root README). YelpChi, Amazon, T-Finance, Elliptic and T-Social use the paper's own
   configs (`configs/<dataset>.yaml` in the upstream repository).
2. Preprocess with the upstream `data_handle.py`, then train with the benchmark
   adapter in `benchmark_adapter/` (`train_benchmark.py`; `train_tuning.py` for tuning).
   The exact launch used for the four default runs is
   `benchmark_adapter/parallel_default_driver.executed.sh`.
3. Protocol: validation-only selection by validation AP, threshold chosen on
   validation nodes, test metrics computed once (`benchmark_adapter/PROTOCOL.md`).
4. T-Social: use the config in `tsocial/evidence/benchmark/native_tsocial_seed2/resolved_config.yaml`.
   It adds `nowandb=True` so training does not stop at a Weights & Biases prompt.
5. FDCompCN: the adapter is in `fdcompcn/evidence/preprocessing/`.
   The source is `data/FDCompCN.zip` in the SplitGNN repository (SHA-256 in
   `fdcompcn/evidence/preprocessing/fdcompcn_zip.sha256`).
6. Environment: `constraints.txt` and `environment_requested_vs_installed.txt`.
   The DGL CUDA fix is documented in `evidence/environment/dgl_cuda_repair/`.

## Known limitations

- FDCompCN uses a provisional split generated locally with the same convention as
  the other datasets (stratified, seed 2). It must be replaced by the team's frozen
  split before it is compared with other models.
- Native-protocol results are single-seed. FDCompCN shows that seed variation can be large.
- FDCompCN `k=32` overfits badly (train AUC 0.989 against test 0.690) and was rejected
  in favour of `k=4`. The rejected log is kept as evidence of that decision.
