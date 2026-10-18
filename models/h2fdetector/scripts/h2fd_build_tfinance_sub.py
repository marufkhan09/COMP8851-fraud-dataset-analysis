#!/usr/bin/env python
"""
Supplementary T-Finance view for H2-FDetector: 15% stratified node subsample.

Why: the full T-Finance graph (BWGNN release, the file SplitGNN used) does not fit
H2-FDetector on a 48 GB GPU (CUDA OOM in the first forward pass, all 12 unified runs).
Following the group rule (run in the same environment; if it fails, document it), this
adds a clearly separate, documented dataset 'tfinance_sub15' built with exactly the
same subsampling procedure the group used for T-Social:
  * RandomState(2), stratified by label, 15% of the nodes of each class
  * induced subgraph (only edges between sampled nodes kept)
  * 'homo' = 'e' = the subgraph edges (same convention as every other adapter)
Results on this view are supplementary and NOT like-for-like with SplitGNN's
full-graph T-Finance runs (SplitGNN was not re-run on this subsample).

Everything else (canonical splits, edge labels, views, manifest, config) is done by
h2fd_build_views_ext.py, which is imported unchanged; this wrapper only registers one
extra dataset builder.

Usage:
  python h2fd_build_tfinance_sub.py --dataset tfinance_sub15 --data-dir /workspace/data/bwgnn/dataset
"""
import hashlib
import os
import sys

import numpy as np
import torch
import dgl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h2fd_build_views_ext as ext  # noqa: E402

DATASET = 'tfinance_sub15'
FRACTION = 0.15
WRAPPER_VERSION = 'h2fd_tfinance_sub15_v1'


def build_tfinance_sub15(a):
    p = f'{a.data_dir}/tfinance'
    ext.log(f'[{DATASET}] loading full T-Finance graph {p}')
    g_full = dgl.load_graphs(p)[0][0]
    label_full = ext.flat_label(g_full.ndata['label'])
    n_full, e_full = g_full.num_nodes(), g_full.num_edges()

    rng = np.random.RandomState(2)
    picked = []
    for c in torch.unique(label_full):
        class_idx = torch.where(label_full == c)[0].numpy()
        n_take = int(round(FRACTION * len(class_idx)))
        picked.append(rng.choice(class_idx, size=n_take, replace=False))
    idx = np.sort(np.concatenate(picked))
    idx_t = torch.from_numpy(idx)

    sub = dgl.node_subgraph(g_full, idx_t)
    src, dst = sub.edges()
    g = ext.single_rel_graph(src, dst, sub.num_nodes())
    g.nodes['r'].data['feature'] = g_full.ndata['feature'][idx_t].float()
    g.nodes['r'].data['label'] = label_full[idx_t].long()

    with open(os.path.abspath(__file__), 'rb') as f:
        wrapper_sha = hashlib.sha256(f.read()).hexdigest()
    info = {
        'source_path': p,
        'source_sha256': ext.sha256_file(p),
        'source_release': 'BWGNN (Tang et al. 2022) Google Drive release, the file SplitGNN used',
        'full_graph': {'n_nodes': int(n_full), 'n_edges': int(e_full),
                       'h2fd_full_graph_status': 'CUDA OOM on 48 GB RTX A6000 (all 12 unified runs)'},
        'subsample': {'fraction': FRACTION, 'rng': 'RandomState(2)', 'stratified_by': 'label',
                      'procedure': 'identical to splitgnn tsocial_adapter.py',
                      'n_nodes': int(sub.num_nodes()), 'n_edges': int(sub.num_edges()),
                      'n_fraud': int(g.nodes['r'].data['label'].sum()),
                      'fraud_rate': float(g.nodes['r'].data['label'].float().mean()),
                      'full_fraud_rate': float(label_full.float().mean())},
        'subsample_node_ids_sha256': hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest(),
        'wrapper_version': WRAPPER_VERSION,
        'wrapper_sha256': wrapper_sha,
        'comparability_note': 'supplementary; not like-for-like with SplitGNN full-graph T-Finance',
        'transform': '15% stratified induced subgraph; homo = e = subgraph edges',
    }
    ext.log(f'[{DATASET}] subsample: {info["subsample"]}')
    del g_full
    return g, info, None


ext.BUILDERS[DATASET] = build_tfinance_sub15
if DATASET not in ext.DATASETS:
    ext.DATASETS.append(DATASET)

if __name__ == '__main__':
    ext.main()
