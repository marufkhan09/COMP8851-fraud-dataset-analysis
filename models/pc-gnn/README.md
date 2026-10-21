# PC-GNN on the COMP8851 fraud-graph benchmark

Owner: Md Maruf Khan.

## The model

PC-GNN, "Pick and Choose: A GNN-Based Imbalanced Learning Approach for Fraud
Detection" (Liu et al., The Web Conference 2021,
https://doi.org/10.1145/3442381.3449989).

Fraud graphs have few fraud nodes and many camouflaged neighbours. PC-GNN
addresses both problems before aggregation:
- **Pick**: a label-balanced sampler builds the training sub-graph so that fraud
  nodes are not drowned out by the majority class.
- **Choose**: a neighbourhood sampler keeps the neighbours that look most
  relevant to the target node, instead of aggregating all of them.
- **Aggregate**: the chosen neighbours are aggregated within each relation, and
  the relations are then combined.

Official code: https://github.com/PonderLY/PC-GNN, commit
`9d7d7fae491081178b6e12e193fb89cfc330f2be`. The architecture was not changed.
The instrumented copy used here only adds timing, metric recording and
persistent split files (see `source/COMP8851_RUN_GUIDE.md`).

## Status

| Dataset | Final runs | Notes |
|---|---|---|
| YelpChi | 12 of 12 | rho 0.4, chosen on validation AUROC |
| Amazon | 12 of 12 | rho 0.2; 51 epochs per run |
| T-Finance | 12 of 12 | see `tfinance/evidence/raw/` |
| Elliptic | 12 of 12 | see `elliptic/evidence/raw/` |
| FDCompCN | 12 of 12 | see `fdcompcn/evidence/raw/` |
| T-Social | not completed | scaling failure, described below |

Each "run" is one training ratio (TR40, TR30, TR20, TR10) and one seed (2, 42, 72).

## Results (test, mean +/- SD over 3 seeds)

| Dataset | TR40 AUPRC | TR10 AUPRC | TR40 AUROC | TR40 fraud-F1 |
|---|---:|---:|---:|---:|
| YelpChi | 0.524 +/- 0.011 | 0.458 | 0.839 | 0.452 |
| Amazon | 0.882 +/- 0.006 | 0.857 | 0.970 | 0.707 |
| T-Finance | 0.751 +/- 0.002 | 0.742 | 0.915 | 0.725 |
| Elliptic | 0.465 +/- 0.063 | 0.333 | 0.865 | 0.535 |
| FDCompCN | 0.246 +/- 0.020 | 0.178 | 0.654 | 0.260 |

Full tables with AUPRC, AUROC, Macro-F1, fraud-F1, G-Mean and epoch times for all
four ratios are in `<dataset>/results/aggregate/pcgnn_<dataset>_mean_sd.csv`, with
per-seed rows in `..._seed_level_results.csv`. For T-Finance, Elliptic and
FDCompCN these tables are generated from the committed terminal logs by
`scripts/aggregate_from_logs.py`. Only three seeds were run, so no significance
claim is made.

## How to reproduce

```bash
cd models/pc-gnn/source
# author-mode check on YelpChi (needs the YelpChi .mat and adjacency pickles in data/)
python main.py --config ./config/pcgnn_yelpchi.yml
# unified runs: nested training ratios, persistent splits
python main.py --config ./config/unified/pcgnn_yelpchi_TR40.yml
```

- Data: see the root README. Required file names are listed in `source/COMP8851_RUN_GUIDE.md`.
- Split files: `../../shared/splits/yelpchi/` and `.../amazon/`. For T-Finance,
  Elliptic and FDCompCN, `<dataset>/evidence/raw/splits/team_frozen_split_manifest.json`.
  Do not regenerate splits per model. `source/scripts/create_nested_splits.py` is the generator.
- Selection protocol: hyperparameters and checkpoints by validation only. Test is evaluated once.
- Regenerate the three log-based tables from the repository root:
  `python3 models/pc-gnn/scripts/aggregate_from_logs.py`
- Environment: the original code targets an old stack (see `source/README.md`).
  YelpChi and Amazon were run on a Tesla T4 (Kaggle). Environment evidence is in each dataset's `evidence/` folder.

## T-Social: scaling failure

T-Social (5,781,065 nodes, 146,211,016 directed edges) was attempted on the same hardware
as DGA-GNN and GAGA. PC-GNN's data loaded correctly, but no training epoch
completed in 2 hours 6 minutes. CPU and memory were flat the whole time, which
points to the adjacency-construction step as the bottleneck. The run was stopped
and recorded as a non-completion, not a crash. This is a measured result for the
reference implementation on this hardware. It does not show that PC-GNN cannot
handle large graphs. The evidence and comparison with the other two models are in
`../dga-gnn/tsocial/README.md`. The small data-loading patch used for the attempt
was not committed, so that attempt cannot be rerun from this repository.

## Known limitations

- The Amazon author-mode reproduction is approximate: the supplied graph has 25
  features and the run used 51 epochs, whereas the paper reports 100 features and 100 epochs.
  Use the 12-run unified block, not the author-mode run.
- Some random-search tuning trials hit the time limit. They are logged as
  `TIMEOUT` in `<dataset>/evidence/raw/tuning/tr40/all_trials_summary.json`.
- Elliptic at TR40 varies a lot between seeds (AUPRC SD 0.063), and FDCompCN is weak at every ratio (AUROC 0.57 to 0.68).
