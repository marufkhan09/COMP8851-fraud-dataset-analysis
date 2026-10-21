# GAGA on the COMP8851 fraud-graph benchmark

Owner: Md Maruf Khan.

## The model

GAGA, "Label Information Enhanced Fraud Detection against Low Homophily in
Graphs" (Wang et al., The Web Conference 2023). Official code:
https://github.com/Orion-wyc/GAGA, commit `8cc835842305aa9487d5d019d485a102c93989fe`
(`gaga_commit.txt`).

Standard GNNs mix fraud and benign neighbours when homophily is low. GAGA uses
training labels to separate them:
- **Group aggregation (preprocessing).** For every node, its multi-hop neighbours are
  grouped by label information, and each group is aggregated into one vector.
  Each node becomes a short sequence of group vectors (`graph2seq`).
- **Transformer encoder.** A Transformer reads the sequence with learnable group
  encodings, so the model weighs fraud-like, benign-like and unlabelled neighbourhoods
  differently.

The preprocessing is the expensive part, and it is what limits GAGA on large graphs
(see T-Social below).

## Status

| Dataset | Support | Status |
|---|---|---|
| YelpChi | native | complete, native paper split |
| Amazon | native | complete, native paper split |
| T-Finance | adapter | complete, native-style split |
| Elliptic | adapter | complete, native-style split |
| FDCompCN | adapter | complete, native-style split |
| T-Social | adapter | not completed (preprocessing too slow) |

These are single-seed runs under GAGA's own split settings, not the frozen COMP8851
TR40 to TR10 grid. The report marks them as native-protocol and does not rank them
against the unified-protocol models.

## Results (test, single seed)

| Dataset | AUC | AP (AUPRC) | F1-macro | F1-fraud | G-Mean |
|---|---:|---:|---:|---:|---:|
| YelpChi | 0.9428 | 0.7952 | 0.8313 | 0.7110 | 0.8232 |
| Amazon | 0.9616 | 0.8708 | 0.9098 | 0.8355 | 0.8775 |
| T-Finance | 0.9577 | 0.8449 | 0.8952 | 0.7991 | 0.8461 |
| Elliptic | 0.8817 | 0.5985 | 0.7870 | 0.6053 | 0.8114 |
| FDCompCN | 0.6657 | 0.2278 | 0.6009 | 0.3089 | 0.5839 |

Taken from the last block of `<dataset>/evidence/benchmark/*/train.log`. YelpChi and
Amazon use the paper split (train 0.4, validation 0.1, test 0.5). The other
three use train 0.4 and validation 0.198. All five use split seed 717. YelpChi AUC (0.943) is lower than the AUC
that PC-GNN and DGA-GNN reach on YelpChi (about 0.98 for DGA-GNN). It was not tuned
or rerun to close that gap.

## How to reproduce

Run from `models/gaga/gaga_upstream/pytorch_gaga/`. Example for Elliptic:

```bash
python dataset_split.py --dataset elliptic --save_dir seq_data --train_size 0.4 --val_size 0.198
python preprocessing/graph2seq_single.py --dataset elliptic --fanouts -1 -1 --save_dir seq_data \
       --train_size 0.4 --val_size 0.198 --add_self_loop --norm_feat
python main_transformer.py --config configs/elliptic_paper.json --gpu 0 --log_dir logs --early_stop 100
```

- YelpChi and Amazon: `--dataset yelp` or `amazon`, `--val_size 0.1`, configs `*_paper.json`.
- Elliptic and FDCompCN adapters: `elliptic-fdcompcn-adapter.md`. They load the
  preprocessed graphs built for DGA-GNN, and keep FDCompCN's three relations.
- Run `graph2seq_single.py` and not the upstream `graph2seq_mp.py` (see below).
- Environment: Python 3.7, `torch==1.7.1+cu110`, `dgl==0.6.1`, and the CUDA 11
  libraries installed with pip (`nvidia-cublas-cu11`, `nvidia-cusparse-cu11`, and so on)
  with `LD_LIBRARY_PATH` set to include them. `dgl-cu110` stops at 0.6.1, and the
  0.7.2 wheel is CPU-only and fails at `.to('cuda')`. Package list:
  `yelpchi/evidence/environment/environment_installed.txt`.

## Fix to the upstream preprocessing

`graph2seq_mp.py` with more than one worker deadlocks on this setup: all workers sit at
0% CPU. This is a known class of failure for fork-based multiprocessing after PyTorch
and DGL have started their thread pools. The replacement `graph2seq_single.py`
does the same work in one process, in chunks of 2000 nodes, with progress logging.
YelpChi (45,954 nodes) preprocesses in about 178 seconds this way. Copy of the file:
`yelpchi/evidence/preprocessing/graph2seq_single.py`.

## T-Social: scaling failure

T-Social (5,781,065 nodes) was attempted on the same hardware as DGA-GNN. GAGA's
reference preprocessing (`GroupFeatureSequenceLoader.load_batch`) handles nodes one at
a time in a Python loop. At T-Social scale it ran at about 1.4 iterations per second,
which projects to roughly 1,100 hours (about 46 days). The run was stopped once the rate
was stable. DGA-GNN completed the same graph because its preprocessing is vectorised.
This describes the reference implementation as shipped, on this hardware. The
evidence is in `../dga-gnn/tsocial/README.md`. The small adapter used for the attempt
(a `tsocial` branch in `preprocessing/data_utils.py`) was not committed.

## Known limitations

- Single-seed, native-style splits, so not directly comparable with the unified grid.
- T-Social was not completed.
