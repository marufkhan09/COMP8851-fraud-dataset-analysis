"""
Builds GDN input views for the four datasets GDN cannot read natively.

Each view reproduces the node order, labels, features and edges of the SplitGNN
dataset view for the same dataset (see models/splitgnn/adapters/*.py in the team
repo), so the canonical splits line up node-for-node. run_gdn_unified.py then
verifies this against SplitGNN's stored split_indices.npz before training.

Output, in /workspace/gdn/data/:
    <ds>_bundle.npz            features, labels, relation_files, adapter_version, stats
    <ds>_<rel>_adjlists.pickle adjacency lists in GDN's format (self loops, symmetric),
                               same semantics as utils.utils.sparse_to_adjlist

Adapter rule: data translation only. GDN's model, loss and mechanism are untouched.
Homogeneous graphs (T-Finance, T-Social, Elliptic) have one relation, so the same
adjacency list fills all three of GDN's relation slots (adapter_version
'homo_replicated_v1'). This is recorded as a limitation.

Usage (in the gdn-adapt env, which has DGL):
    python build_gdn_views.py --dataset comp     --src /workspace/raw/comp.dgl
    python build_gdn_views.py --dataset tfinance --src /workspace/raw/tfinance
    python build_gdn_views.py --dataset tsocial  --src /workspace/raw/tsocial
    python build_gdn_views.py --dataset elliptic --src /workspace/raw/elliptic   (folder with the 3 CSVs)
"""
import argparse
import hashlib
import json
import os
import pickle
import time

import numpy as np

OUT_DIR = '/workspace/gdn/data'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def to_adjlist(src, dst, n):
    """Equivalent to GDN's sparse_to_adjlist: add a self loop to every node, make edges
    symmetric, store dict[node] -> set(neighbours). Built via CSR to stay fast on big graphs."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    loops = np.arange(n, dtype=np.int64)
    u = np.concatenate([src, dst, loops])
    v = np.concatenate([dst, src, loops])
    order = np.lexsort((v, u))
    u, v = u[order], v[order]
    keep = np.ones(len(u), dtype=bool)
    keep[1:] = (u[1:] != u[:-1]) | (v[1:] != v[:-1])
    u, v = u[keep], v[keep]
    ptr = np.searchsorted(u, np.arange(n + 1))
    vl = v.tolist()
    return {i: set(vl[ptr[i]:ptr[i + 1]]) for i in range(n)}, len(u)


def load_dgl(path):
    import dgl
    return dgl.load_graphs(path)[0][0]


def view_comp(src_path):
    """FDCompCN as shipped by SplitGNN (comp.dgl): heterograph with its own relations plus 'homo'."""
    g = load_dgl(src_path)
    assert len(g.ntypes) == 1, g.ntypes
    nt = g.ntypes[0]
    rels = [et for et in g.etypes if et != 'homo']
    print(f'comp.dgl node type {nt!r}, relations {g.etypes}')
    feat = g.nodes[nt].data['feature'].numpy()
    label = g.nodes[nt].data['label'].numpy().astype(np.int64).reshape(-1)
    n = g.num_nodes(nt)
    if len(rels) == 3:
        edges = {r: tuple(t.numpy() for t in g.edges(etype=r)) for r in rels}
        return feat, label, n, edges, rels, 'comp_native_relations_v1'
    raise SystemExit(f'Expected 3 non-homo relations in comp.dgl, found {rels}. Stopping for a decision.')


def view_tfinance(src_path):
    """Matches models/splitgnn/adapters/tfinance_adapter.py: original node order, label = onehot[:, 1]."""
    g = load_dgl(src_path)
    s, d = g.edges()
    raw = g.ndata['label']
    label = (raw[:, 1] if raw.dim() == 2 else raw).numpy().astype(np.int64)
    feat = g.ndata['feature'].float().numpy()
    return feat, label, g.num_nodes(), {'homo': (s.numpy(), d.numpy())}, None, 'homo_replicated_v1'


def view_tsocial(src_path):
    """Matches models/splitgnn/adapters/tsocial_adapter.py: 15% stratified node subsample, RandomState(2),
    classes in torch.unique order, sorted node ids, induced subgraph."""
    import dgl
    import torch
    g_full = load_dgl(src_path)
    label_full = g_full.ndata['label']
    rng = np.random.RandomState(2)
    picked = []
    for c in torch.unique(label_full):
        class_idx = torch.where(label_full == c)[0].numpy()
        n_take = int(round(0.15 * len(class_idx)))
        picked.append(rng.choice(class_idx, size=n_take, replace=False))
    picked = np.concatenate(picked)
    picked.sort()
    sub = dgl.node_subgraph(g_full, torch.from_numpy(picked))
    del g_full
    s, d = sub.edges()
    print(f'T-Social subsample: {sub.num_nodes()} nodes, {sub.num_edges()} edges (SplitGNN: 867160 / 3297570)')
    feat = sub.ndata['feature'].float().numpy()
    label = sub.ndata['label'].long().numpy()
    return feat, label, sub.num_nodes(), {'homo': (s.numpy(), d.numpy())}, None, 'homo_replicated_tsocial15pct_v1'


def view_elliptic(src_dir):
    """Matches models/splitgnn/adapters/elliptic_adapter.py: order from features CSV, features = columns 2:,
    class '1' -> 1 (illicit), '2' -> 0 (licit), 'unknown' -> -1 (kept for message passing only)."""
    import pandas as pd
    classes = pd.read_csv(f'{src_dir}/elliptic_txs_classes.csv')
    edges_df = pd.read_csv(f'{src_dir}/elliptic_txs_edgelist.csv')
    features_df = pd.read_csv(f'{src_dir}/elliptic_txs_features.csv', header=None)
    tx_ids = features_df[0].values
    txid_to_idx = {tx: i for i, tx in enumerate(tx_ids)}
    n = len(tx_ids)
    feat = features_df.iloc[:, 2:].values.astype(np.float32)
    cls = classes.set_index('txId')['class'].astype(str)
    label = np.full(n, -1, dtype=np.int64)
    for tx, idx in txid_to_idx.items():
        c = cls.loc[tx]
        if c == '1':
            label[idx] = 1
        elif c == '2':
            label[idx] = 0
    s = edges_df['txId1'].map(txid_to_idx)
    d = edges_df['txId2'].map(txid_to_idx)
    ok = s.notna() & d.notna()
    return (feat, label, n, {'homo': (s[ok].values.astype(np.int64), d[ok].values.astype(np.int64))},
            None, 'homo_replicated_v1')


VIEWS = {'comp': view_comp, 'tfinance': view_tfinance, 'tsocial': view_tsocial, 'elliptic': view_elliptic}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(VIEWS))
    ap.add_argument('--src', required=True, help='source .dgl file, or folder of CSVs for elliptic')
    ap.add_argument('--feature_transform', default='none', choices=['none', 'gdn_row_normalize'],
                    help='applied at load time by run_gdn_unified.py using GDN utils.normalize')
    ap.add_argument('--prescale', default='none', choices=['none', 'column_minmax'],
                    help='label-free per-feature min-max to [0,1], baked into the bundle at build time '
                         '(for features with negatives, where row normalization is undefined)')
    cli = ap.parse_args()
    t0 = time.time()

    feat, label, n, edges, rels, adapter = VIEWS[cli.dataset](cli.src)
    assert feat.shape[0] == n == len(label)
    raw_min, raw_max = float(feat.min()), float(feat.max())
    if cli.prescale == 'column_minmax':
        # Uses feature values of all nodes and no labels (transductive, label-free), so no label leakage.
        lo, hi = feat.min(0), feat.max(0)
        feat = ((feat - lo) / np.where(hi > lo, hi - lo, 1.0)).astype(np.float32)
        adapter = f'{adapter}+column_minmax'

    rel_files, edge_counts = [], {}
    for r, (s, d) in edges.items():
        fname = f'{cli.dataset}_{r}_adjlists.pickle'
        adj, n_entries = to_adjlist(s, d, n)
        with open(f'{OUT_DIR}/{fname}', 'wb') as f:
            pickle.dump(adj, f, protocol=pickle.HIGHEST_PROTOCOL)
        edge_counts[r] = {'raw_edges': int(len(s)), 'adjlist_entries_incl_self_loops': int(n_entries)}
        print(f'wrote {fname}: {n_entries} entries')
        del adj
        rel_files.append(fname)
    if rels is None:  # homogeneous: one adjacency list in all three GDN relation slots
        rel_files = rel_files * 3

    stats = {
        'n_nodes': int(n), 'n_features': int(feat.shape[1]),
        'n_fraud': int((label == 1).sum()), 'n_normal': int((label == 0).sum()),
        'n_unlabelled': int((label == -1).sum()),
        'feature_min': float(feat.min()), 'feature_max': float(feat.max()),
        'feature_mean': float(feat.mean()), 'feature_abs_row_sum_median': float(np.median(np.abs(feat).sum(1))),
        'any_negative_features': bool((feat < 0).any()),
        'prescale': cli.prescale, 'raw_feature_min': raw_min, 'raw_feature_max': raw_max,
        'edges': edge_counts, 'source_sha256': sha256_file(cli.src) if os.path.isfile(cli.src) else 'folder',
    }
    if os.path.isdir(cli.src):
        stats['source_sha256'] = {fn: sha256_file(os.path.join(cli.src, fn)) for fn in sorted(os.listdir(cli.src))
                                  if fn.endswith('.csv')}
    np.savez(f'{OUT_DIR}/{cli.dataset}_bundle.npz', features=feat.astype(np.float32), labels=label,
             relation_files=np.array(rel_files), adapter_version=adapter,
             feature_transform=cli.feature_transform, stats_json=json.dumps(stats))
    print(json.dumps(stats, indent=2))
    print(f'bundle: {OUT_DIR}/{cli.dataset}_bundle.npz  relation slots: {rel_files}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
