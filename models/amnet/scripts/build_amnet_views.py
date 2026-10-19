"""
Builds AMNet input views for the COMP8851 unified benchmark (Track B).

Output, one file per dataset: /workspace/amnet_vast/views/<ds>_view.npz
    x            float32 [n, d]   node features, as stored in the canonical source (no scaling)
    y            int64   [n]      1 = fraud, 0 = normal, -1 = unlabelled (Elliptic 'unknown')
    edge_index   int64   [2, E]   undirected (both directions), de-duplicated, no self loops
    adapter_version, stats_json

Node order, labels and features reproduce the SplitGNN / GDN dataset views exactly
(see /workspace/gdn_vast/build_gdn_views.py and models/splitgnn/adapters/*.py), so the
canonical split masks line up node for node. run_amnet_unified.py re-checks this against
SplitGNN's stored split_indices.npz before every run.

AMNet is a single-relation model (one edge_index), so each dataset is given as one
undirected edge set:
    yelp, amazon                 the 'homo' adjacency stored in the canonical .mat file
    comp                         the 'homo' edge type of comp.dgl (union of the 3 relations if absent)
    tfinance, tsocial, elliptic  their single relation
Symmetrising matches AMNet's own loader (utils.load_pyg_data converts graphs to undirected),
and BernConv removes self loops itself, so this is a representation step only.
Adapter rule: data translation only. AMNet's model, loss and mechanism are untouched.

Python env:
    yelp, amazon, elliptic   -> amnet env (numpy, scipy, pandas)
    comp, tfinance, tsocial  -> an env with DGL (the gdn-adapt env)

Usage:
    python build_amnet_views.py --dataset elliptic
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

OUT_DIR = '/workspace/amnet_vast/views'

SOURCES = {
    'yelp': '/workspace/gdn/data/YelpChi.mat',
    'amazon': '/workspace/gdn/data/Amazon.mat',
    'comp': '/workspace/raw/comp.dgl',
    'tfinance': '/workspace/raw/tfinance',
    'tsocial': '/workspace/raw/tsocial',
    'elliptic': '/workspace/raw/elliptic',
}

# Canonical source hashes (team registry, as verified by the GDN runner). Mismatch = stop.
EXPECTED_SOURCE_SHA256 = {
    'yelp': {'YelpChi.mat': 'fedb35a8fa539b27866244d3515a47a76b20080cdacb33112da3458fd2487b42'},
    'amazon': {'Amazon.mat': '4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63'},
    'comp': {'comp.dgl': 'e252b9a6b619b28a7bf6d9f5b16aacc232d43207d87ac26cd5160fb0d8a98baa'},
    'tfinance': {'tfinance': 'b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700'},
    'tsocial': {'tsocial': '8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0'},
    'elliptic': {
        'elliptic_txs_features.csv': 'fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0',
        'elliptic_txs_edgelist.csv': 'a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e',
        'elliptic_txs_classes.csv': '93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493',
    },
}

# (n_nodes, n_fraud) of the canonical views, where known. None = not checked.
EXPECTED_COUNTS = {
    'yelp': (45954, 6677),
    'amazon': (11944, 821),
    'comp': None,
    'tfinance': (39357, 1804),
    'tsocial': (867160, 26142),
    'elliptic': (203769, 4545),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_sources(name, src):
    expected = EXPECTED_SOURCE_SHA256[name]
    got = {}
    for fname, want in expected.items():
        path = os.path.join(src, fname) if os.path.isdir(src) else src
        got[fname] = sha256_file(path)
        if got[fname] != want:
            raise SystemExit(f'SOURCE HASH MISMATCH for {name}/{fname}: {got[fname]} != {want}. Stopping.')
    print(f'source hashes verified: {list(got)}')
    return got


def undirected(src, dst, n):
    """Both directions, no self loops, no duplicates. Returns edge_index [2, E] and counts."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    n_raw = int(len(src))
    n_loops = int((src == dst).sum())
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    keep = u != v
    key = np.unique(u[keep] * n + v[keep])
    edge_index = np.stack([key // n, key % n]).astype(np.int64)
    return edge_index, {'raw_edges': n_raw, 'raw_self_loops': n_loops,
                        'undirected_edges_both_directions': int(edge_index.shape[1])}


def load_dgl(path):
    import dgl
    return dgl.load_graphs(path)[0][0]


# ---------------------------------------------------------------- views
def view_mat(path):
    """YelpChi / Amazon: .mat order, 'homo' adjacency, features as stored."""
    from scipy.io import loadmat
    d = loadmat(path)
    feat = d['features']
    feat = (feat.toarray() if hasattr(feat, 'toarray') else np.asarray(feat)).astype(np.float32)
    label = d['label'].flatten().astype(np.int64)
    homo = d['homo'].tocoo()
    return feat, label, homo.row, homo.col, 'mat_homo_undirected_v1'


def view_comp(path):
    """FDCompCN (comp.dgl): node type 'company'; 'homo' edge type, else union of the 3 relations."""
    g = load_dgl(path)
    assert len(g.ntypes) == 1, g.ntypes
    nt = g.ntypes[0]
    print(f'comp.dgl node type {nt!r}, edge types {g.etypes}')
    feat = g.nodes[nt].data['feature'].float().numpy()
    label = g.nodes[nt].data['label'].numpy().astype(np.int64).reshape(-1)
    n = g.num_nodes(nt)
    rels = [et for et in g.etypes if et != 'homo']
    union_s = np.concatenate([g.edges(etype=r)[0].numpy() for r in rels])
    union_d = np.concatenate([g.edges(etype=r)[1].numpy() for r in rels])
    if 'homo' in g.etypes:
        s, d = (t.numpy() for t in g.edges(etype='homo'))
        same = np.array_equal(undirected(s, d, n)[0], undirected(union_s, union_d, n)[0])
        print(f"'homo' edge type equals union of {rels} after symmetrising: {same}")
        return feat, label, s, d, f'comp_homo_etype_undirected_v1(homo_equals_union={same})'
    return feat, label, union_s, union_d, 'comp_relation_union_undirected_v1'


def view_tfinance(path):
    """Matches SplitGNN/GDN: original node order, label = onehot[:, 1]."""
    g = load_dgl(path)
    s, d = g.edges()
    raw = g.ndata['label']
    label = (raw[:, 1] if raw.dim() == 2 else raw).numpy().astype(np.int64)
    feat = g.ndata['feature'].float().numpy()
    return feat, label, s.numpy(), d.numpy(), 'homo_undirected_v1'


def view_tsocial(path):
    """Matches SplitGNN/GDN: 15% stratified node subsample, RandomState(2), classes in torch.unique
    order, sorted node ids, induced subgraph."""
    import dgl
    import torch
    g_full = load_dgl(path)
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
    return feat, label, s.numpy(), d.numpy(), 'homo_undirected_tsocial15pct_v1'


def view_elliptic(src_dir):
    """Matches SplitGNN/GDN: features-CSV row order, features = columns 2:,
    class '1' -> 1, '2' -> 0, 'unknown' -> -1 (kept in the graph, excluded from splits)."""
    import pandas as pd
    classes = pd.read_csv(f'{src_dir}/elliptic_txs_classes.csv')
    edges_df = pd.read_csv(f'{src_dir}/elliptic_txs_edgelist.csv')
    features_df = pd.read_csv(f'{src_dir}/elliptic_txs_features.csv', header=None)
    tx_ids = features_df[0].values
    txid_to_idx = {tx: i for i, tx in enumerate(tx_ids)}
    feat = features_df.iloc[:, 2:].values.astype(np.float32)
    cls = classes.set_index('txId')['class'].astype(str)
    mapped = pd.Series(tx_ids).map(cls).astype(str).values
    label = np.full(len(tx_ids), -1, dtype=np.int64)
    label[mapped == '1'] = 1
    label[mapped == '2'] = 0
    s = edges_df['txId1'].map(txid_to_idx)
    d = edges_df['txId2'].map(txid_to_idx)
    ok = s.notna() & d.notna()
    return (feat, label, s[ok].values.astype(np.int64), d[ok].values.astype(np.int64),
            'homo_undirected_v1')


VIEWS = {'yelp': view_mat, 'amazon': view_mat, 'comp': view_comp, 'tfinance': view_tfinance,
         'tsocial': view_tsocial, 'elliptic': view_elliptic}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(VIEWS))
    ap.add_argument('--src', default=None, help='override the canonical source path')
    cli = ap.parse_args()
    t0 = time.time()
    src = cli.src or SOURCES[cli.dataset]
    os.makedirs(OUT_DIR, exist_ok=True)

    source_sha = verify_sources(cli.dataset, src)
    feat, label, s, d, adapter = VIEWS[cli.dataset](src)
    n = feat.shape[0]
    assert n == len(label), (n, len(label))
    edge_index, edge_stats = undirected(s, d, n)
    assert edge_index.max() < n and edge_index.min() >= 0

    exp = EXPECTED_COUNTS[cli.dataset]
    if exp is not None:
        got = (int(n), int((label == 1).sum()))
        if got != exp:
            raise SystemExit(f'COUNT MISMATCH for {cli.dataset}: (nodes, fraud) = {got}, expected {exp}. Stopping.')

    stats = {
        'dataset': cli.dataset, 'adapter_version': adapter,
        'n_nodes': int(n), 'n_features': int(feat.shape[1]),
        'n_fraud': int((label == 1).sum()), 'n_normal': int((label == 0).sum()),
        'n_unlabelled': int((label == -1).sum()),
        'feature_min': float(feat.min()), 'feature_max': float(feat.max()),
        'any_negative_features': bool((feat < 0).any()),
        'feature_transform': 'none',
        'edges': edge_stats, 'source_sha256': source_sha, 'source_path': src,
    }
    out = f'{OUT_DIR}/{cli.dataset}_view.npz'
    tmp = f'{OUT_DIR}/{cli.dataset}_view.tmp.npz'
    np.savez(tmp, x=feat.astype(np.float32), y=label.astype(np.int64), edge_index=edge_index,
             adapter_version=adapter, stats_json=json.dumps(stats))
    os.replace(tmp, out)  # only a complete file ever has the final name
    with open(f'{OUT_DIR}/{cli.dataset}_view_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f'wrote {out} ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
