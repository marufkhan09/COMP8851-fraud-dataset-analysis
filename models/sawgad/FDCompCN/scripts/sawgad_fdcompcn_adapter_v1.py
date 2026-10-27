#!/usr/bin/env python3
"""SAWGAD FDCompCN adapter: original topology, frozen nested splits, masked test labels."""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/workspace/comp8851')
AUTHOR_REPO = ROOT / 'repos/SAWGAD'
SOURCE = ROOT / 'assets/fdcompcn/source/comp.dgl'
SPLIT = Path('/workspace/uploads/ass2_split_candidates/FDCompCN/fdcompcn_seed2_nested_splits.npz')
DIAG = ROOT / 'diagnostics/sawgad/fdcompcn_adapter_v1.json'
AUTHOR_SHA = '64d6d531a2f03549f0bfbe461092beedd6f70bd2'
SOURCE_SHA = 'e252b9a6b619b28a7bf6d9f5b16aacc232d43207d87ac26cd5160fb0d8a98baa'
SPLIT_SHA = '5cd7084342f4af48e9b82828dcac0e97879314cd5c9c010642b30c89e7c95168'
NODES = 5317
FEATURES = 57
COUNTS = {'TR40': 2126, 'TR30': 1594, 'TR20': 1062, 'TR10': 531,
          'val': 1064, 'test': 2127}


def checksum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs():
    for p, expected in ((SOURCE, SOURCE_SHA), (SPLIT, SPLIT_SHA)):
        assert p.is_file(), f'MISSING: {p}'
        assert checksum(p) == expected, f'SHA256 MISMATCH: {p}'
    actual = subprocess.check_output(['git', '-C', str(AUTHOR_REPO), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == AUTHOR_SHA, f'Author commit mismatch: {actual}'
    print('FDCOMPCN_PINNED_INPUTS=PASS', flush=True)


def load_source_and_splits():
    import numpy as np
    from dgl.data.utils import load_graphs
    graphs, _ = load_graphs(str(SOURCE))
    assert len(graphs) == 1
    graph = graphs[0]
    assert len(graph.ntypes) == 1, ('Unexpected node types', graph.ntypes)
    node_type = graph.ntypes[0]
    assert all(e[0] == node_type and e[2] == node_type for e in graph.canonical_etypes), 'Cross-node-type relations unsupported'
    print('FDCOMPCN_SOURCE_RELATIONS=', [(str(e), graph.num_edges(e)) for e in graph.canonical_etypes], flush=True)
    assert graph.num_nodes() == NODES and graph.num_edges() == 30752
    assert set(graph.ndata).issuperset({'label', 'feature'})
    x = graph.ndata['feature'].float()
    y = graph.ndata['label'].long().reshape(-1)
    assert tuple(x.shape) == (NODES, FEATURES)
    assert bool(x.isfinite().all())
    assert set(y.unique().tolist()) == {0, 1} and int(y.sum()) == 559
    a = y.numpy()
    with np.load(SPLIT, allow_pickle=False) as z:
        assert int(z['seed'].item()) == 2
        ids = {}
        for k, n in COUNTS.items():
            raw = np.asarray(z[k], dtype=np.int64)
            assert raw.shape == (n,) and len(np.unique(raw)) == n, (k, 'length/duplicate')
            assert (raw >= 0).all() and (raw < NODES).all(), (k, 'range')
            assert 0 < int(a[raw].sum()) < len(raw), (k, 'class distribution')
            ids[k] = raw.copy()
    for small, big in (('TR10','TR20'), ('TR20','TR30'), ('TR30','TR40')):
        assert set(ids[small]).issubset(set(ids[big]))
    assert not np.intersect1d(ids['TR40'], ids['val']).size
    assert not np.intersect1d(ids['TR40'], ids['test']).size
    assert not np.intersect1d(ids['val'], ids['test']).size
    assert len(np.union1d(np.union1d(ids['TR40'], ids['val']), ids['test'])) == NODES
    return graph, ids, y


def load_fdcompcn_split(ratio='TR40', device='cpu', checked=False):
    if ratio not in ('TR40','TR30','TR20','TR10'):
        raise ValueError(ratio)
    if not checked: verify_inputs()
    import torch
    import numpy as np
    sys.path.insert(0, str(AUTHOR_REPO))
    from sawgad.data import DataSplit, _to_homogeneous, _normalize_features
    source, ids, labels = load_source_and_splits()
    # Preserve source edge direction / IDs. Strip all source masks and edge labels.
    import dgl
    relation_edges = [source.edges(order='eid', etype=e) for e in source.canonical_etypes]
    assert sum(int(u.numel()) for u, _ in relation_edges) == 30752
    s = torch.cat([u.long() for u, _ in relation_edges])
    t = torch.cat([v.long() for _, v in relation_edges])
    graph = dgl.graph((s, t), num_nodes=NODES, idtype=torch.int64)
    graph.ndata['feature'] = source.ndata['feature'].float().clone()
    graph.ndata['label'] = labels.clone()
    graph = _to_homogeneous(graph)
    for field in list(graph.edata.keys()):
        del graph.edata[field]
    _normalize_features(graph)
    masked = torch.full((NODES,), -1, dtype=torch.int64)
    for name in (ratio,'val'):
        node_ids = ids[name]
        masked[torch.from_numpy(node_ids)] = labels[torch.from_numpy(node_ids)]
    assert bool((masked[torch.from_numpy(ids['test'])] == -1).all())
    assert int((masked >= 0).sum()) == COUNTS[ratio] + COUNTS['val']
    graph.ndata['label'] = masked
    assert set(graph.ndata) == {'feature','label'} and not list(graph.edata)
    graph = graph.to(device)
    split = DataSplit(graph=graph,
            train_nodes=torch.as_tensor(ids[ratio].copy(), dtype=torch.int64, device=device),
            val_nodes=torch.as_tensor(ids['val'].copy(), dtype=torch.int64, device=device),
            test_nodes=torch.empty(0, dtype=torch.int64, device=device))
    meta = {'model':'SAWGAD','dataset':'FDCompCN','ratio':ratio,
          'source_sha256':SOURCE_SHA,'split_sha256':SPLIT_SHA,
          'author_commit':AUTHOR_SHA,'nodes':NODES,'features':FEATURES,
          'original_edges':30752,'model_graph_edges_after_author_cleanup':int(graph.num_edges()),
          'train_count':len(ids[ratio]),'train_positives':int(labels[torch.from_numpy(ids[ratio])].sum()),
          'val_count':len(ids['val']),'val_positives':int(labels[torch.from_numpy(ids['val'])].sum()),
          'test_count':len(ids['test']),'test_labels_in_model_graph':False,
          'source_mask_and_edge_metadata_removed':True,'test_scored':False}
    return split, ids['test'].copy(), meta


def main():
    verify_inputs()
    split, test_ids, meta = load_fdcompcn_split('TR40', 'cpu', checked=True)
    assert len(test_ids) == COUNTS['test']
    DIAG.parent.mkdir(parents=True, exist_ok=True)
    if DIAG.exists():
        assert json.loads(DIAG.read_text()) == meta, 'Existing manifest mismatch'
    else:
        with DIAG.open('x') as f:
            json.dump(meta, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps(meta, indent=2), flush=True)
    print('FDCOMPCN_SAWGAD_ADAPTER_PREFLIGHT=PASS', flush=True)
    print('TRAINING_STARTED=NO; TEST_EVALUATED=NO', flush=True)

if __name__ == '__main__':
    main()
