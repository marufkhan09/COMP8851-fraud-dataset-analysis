#!/usr/bin/env python3
"""SAWGAD T-Finance adapter: checksum-pinned BWGNN canonical graph and Ass2 frozen splits.

No preflight training. Original full directed topology retained; the exact author's
remove-self-loops/add-self-loops/to_simple cleanup occurs only in load_tfinance_split.
Test labels are excluded from model graph. Evaluation reads test labels separately
only after frozen checkpoint/threshold and a durable one-time test claim.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/workspace/comp8851')
AUTHOR_REPO = ROOT / 'repos/SAWGAD'
SOURCE = ROOT / 'assets/tfinance/canonical_candidate/tfinance_canonical.dgl'
SPLIT = Path('/workspace/uploads/ass2_split_candidates/T-Finance/tfinance_seed2_nested_splits.npz')
DIAG = ROOT / 'diagnostics/sawgad/tfinance_canonical_adapter_v2.json'
AUTHOR_SHA = '64d6d531a2f03549f0bfbe461092beedd6f70bd2'
SOURCE_SHA = 'b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700'
SPLIT_SHA = 'de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1'
NODES = 39357
FEATURES = 10
EDGES = 42445086
COUNTS = {'TR40': 15742, 'TR30': 11806, 'TR20': 7870, 'TR10': 3935,
          'val': 7872, 'test': 15743}


def checksum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs():
    for p, expected in ((SOURCE, SOURCE_SHA), (SPLIT, SPLIT_SHA)):
        assert p.is_file(), f'STOP: Missing source/split {p}'
        assert checksum(p) == expected, f'STOP: Source/split SHA mismatch {p}'
    actual = subprocess.check_output(['git', '-C', str(AUTHOR_REPO), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == AUTHOR_SHA, f'STOP: Wrong author commit {actual}'
    print('TFINANCE_PINNED_INPUTS=PASS', flush=True)


def load_source_and_splits():
    import numpy as np
    import torch
    from dgl.data.utils import load_graphs
    graphs, _ = load_graphs(str(SOURCE))
    assert len(graphs) == 1, 'Expected one original DGL graph'
    graph = graphs[0]
    assert graph.is_homogeneous and graph.num_nodes() == NODES and graph.num_edges() == EDGES
    assert {'feature', 'label'}.issubset(set(graph.ndata))
    x = graph.ndata['feature']
    raw = graph.ndata['label']
    # Original BWGNN release uses 2-column one-hot labels [normal, fraud].
    # Decode using exactly its author's argmax convention, without relabelling.
    assert raw.shape == (NODES, 2), f'STOP: Canonical one-hot label shape mismatch {tuple(raw.shape)}'
    assert bool(((raw == 0) | (raw == 1)).all()) and bool((raw.sum(1) == 1).all())
    y = raw.argmax(dim=1).long()
    assert tuple(x.shape) == (NODES, FEATURES) and bool(torch.isfinite(x).all())
    assert set(y.unique().tolist()) == {0, 1}, f'STOP: Source label classes changed: {y.unique()}'
    assert int(y[27028]) == 1, 'STOP: Canonical fraud label 27028 is not 1'
    a = y.numpy()
    with np.load(SPLIT, allow_pickle=False) as z:
        assert int(z['seed'].item()) == 2
        assert int(z['source_nodes'].item()) == NODES
        # These are the canonical Ass2 counts, not the alternate GADBench copy.
        split_normal = int(z['source_normal'].item())
        split_fraud = int(z['source_fraud'].item())
        graph_normal = int((a == 0).sum())
        graph_fraud = int((a == 1).sum())
        assert (split_normal, split_fraud) == (37553, 1804), (split_normal, split_fraud)
        assert (graph_normal, graph_fraud) == (37553, 1804), (graph_normal, graph_fraud)
        print('TFINANCE_CANONICAL_SOURCE_LABELS_MATCH_FROZEN_SPLIT=PASS',
              f'normal={graph_normal}', f'fraud={graph_fraud}', flush=True)
        ids = {}
        for k, n in COUNTS.items():
            raw = np.asarray(z[k], dtype=np.int64)
            assert raw.shape == (n,) and len(np.unique(raw)) == n, (k, 'count/duplicates')
            assert (raw >= 0).all() and (raw < NODES).all(), (k, 'ID bounds')
            assert 0 < int(a[raw].sum()) < len(raw), (k, 'class presence')
            ids[k] = raw.copy()
    for small, big in (('TR10', 'TR20'), ('TR20', 'TR30'), ('TR30', 'TR40')):
        assert set(ids[small]).issubset(set(ids[big])), (small, big, 'not nested')
    assert not np.intersect1d(ids['TR40'], ids['val']).size
    assert not np.intersect1d(ids['TR40'], ids['test']).size
    assert not np.intersect1d(ids['val'], ids['test']).size
    assert len(np.unique(np.concatenate([ids['TR40'], ids['val'], ids['test']]))) == NODES
    # Frozen split memberships and exactly corrected canonical fraud labels.
    counts = {k: int(a[ids[k]].sum()) for k in COUNTS}
    assert counts == {'TR40': 721, 'TR30': 541, 'TR20': 361,
                      'TR10': 181, 'val': 361, 'test': 722}, counts
    assert counts['TR40'] + counts['val'] + counts['test'] == 1804
    assert all(27028 in ids[r] for r in ('TR40','TR30','TR20','TR10'))
    assert 27028 not in ids['val'] and 27028 not in ids['test']
    return graph, ids, y


def load_tfinance_split(ratio='TR40', device='cpu', checked=False):
    if ratio not in ('TR40', 'TR30', 'TR20', 'TR10'):
        raise ValueError(ratio)
    if not checked:
        verify_inputs()
    import torch
    import dgl
    sys.path.insert(0, str(AUTHOR_REPO))
    from sawgad.data import DataSplit, _to_homogeneous, _normalize_features
    graph, ids, labels = load_source_and_splits()
    # Decode canonical one-hot labels before homogeneous graph conversion.
    # Never expose original source labels on non-train/val nodes to model.
    graph.ndata['label'] = labels.clone()
    # Remove any source metadata if added in a future DGL release.
    for field in list(graph.ndata.keys()):
        if field not in ('feature', 'label'):
            del graph.ndata[field]
    for field in list(graph.edata.keys()):
        del graph.edata[field]
    assert graph.num_edges() == EDGES
    # Preserve all original directed relations. Author cleanup only (no subsampling).
    graph = _to_homogeneous(graph)
    # DGL to_simple (used inside the pinned author's _to_homogeneous) adds
    # edge multiplicities under edata['count'] by default. This metadata is
    # not part of SAWGAD's model input and must be removed AFTER cleanup.
    edge_fields_after_cleanup = set(graph.edata.keys())
    print('TFINANCE_GRAPH_EDGE_FIELDS_AFTER_AUTHOR_CLEANUP=',
          sorted(edge_fields_after_cleanup), flush=True)
    assert edge_fields_after_cleanup <= {'count'}, (
        'STOP: Unexpected graph edge fields after author cleanup',
        sorted(edge_fields_after_cleanup))
    if 'count' in graph.edata:
        del graph.edata['count']
    assert not list(graph.edata), 'STOP: edge metadata not cleaned'
    _normalize_features(graph)
    masked = torch.full((NODES,), -1, dtype=torch.int64)
    for name in (ratio, 'val'):
        idx = torch.from_numpy(ids[name])
        masked[idx] = labels[idx]
    assert bool((masked[torch.from_numpy(ids['test'])] == -1).all()), 'STOP: Test labels visible'
    assert int((masked >= 0).sum()) == COUNTS[ratio] + COUNTS['val']
    graph.ndata['label'] = masked
    assert set(graph.ndata) == {'feature', 'label'} and not list(graph.edata)
    graph = graph.to(device)
    split = DataSplit(
        graph=graph,
        train_nodes=torch.as_tensor(ids[ratio].copy(), dtype=torch.int64, device=device),
        val_nodes=torch.as_tensor(ids['val'].copy(), dtype=torch.int64, device=device),
        test_nodes=torch.empty(0, dtype=torch.int64, device=device),
    )
    meta = dict(
        model='SAWGAD', dataset='T-Finance', ratio=ratio,
        source_sha256=SOURCE_SHA, split_sha256=SPLIT_SHA, author_commit=AUTHOR_SHA,
        nodes=NODES, features=FEATURES, original_directed_edges=EDGES,
        model_graph_edges_after_author_cleanup=int(graph.num_edges()),
        train_count=len(ids[ratio]),
        train_positives=int(labels[torch.from_numpy(ids[ratio])].sum()),
        val_count=len(ids['val']),
        val_positives=int(labels[torch.from_numpy(ids['val'])].sum()),
        test_count=len(ids['test']), test_labels_in_model_graph=False,
        original_source_masks_removed=True, original_graph_subsampled=False,
        test_scored=False,
    )
    return split, ids['test'].copy(), meta


def main():
    verify_inputs()
    graph, ids, labels = load_source_and_splits()
    # Data-only preflight avoids 42 million-edge graph conversion and GPU work.
    data = dict(
        model='SAWGAD', dataset='T-Finance', source_sha256=SOURCE_SHA,
        split_sha256=SPLIT_SHA, author_commit=AUTHOR_SHA,
        nodes=NODES, original_directed_edges=EDGES, features=FEATURES,
        node_fields=sorted(graph.ndata.keys()),
        positive_count=int(labels.sum()),
        source_normal_count=int((labels == 0).sum()),
        source_fraud_count=int((labels == 1).sum()),
        frozen_split_metadata_normal_count=37553,
        frozen_split_metadata_fraud_count=1804,
        source_split_label_metadata_discrepancy='none; canonical 1804 fraud labels match Ass2 source metadata',
        split_counts={k: len(ids[k]) for k in COUNTS},
        split_positives={k: int(labels[__import__('torch').from_numpy(ids[k])].sum()) for k in COUNTS},
        test_labels_passed_to_model=False,
        dataset_preflight_only=True, graph_cleanup_performed=False,
        training_started=False, test_evaluated=False,
    )
    DIAG.parent.mkdir(parents=True, exist_ok=True)
    if DIAG.exists():
        assert json.loads(DIAG.read_text()) == data, 'STOP: Preflight manifest changed'
    else:
        with DIAG.open('x') as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write('\n')
    print(json.dumps(data, indent=2), flush=True)
    print('TFINANCE_CANONICAL_SOURCE_SPLIT_PREFLIGHT=PASS', flush=True)
    # v1.1 missed the runtime graph conversion entirely. Test that real path
    # on CPU before declaring preflight safe; do not execute model or read tests.
    del graph, labels, ids
    import gc
    gc.collect()
    print('TFINANCE_CANONICAL_FULL_GRAPH_CONVERSION_PREFLIGHT=START', flush=True)
    checked, withheld_test_ids, graph_meta = load_tfinance_split(
        'TR40', device='cpu', checked=True)
    assert graph_meta['original_directed_edges'] == EDGES
    assert graph_meta['model_graph_edges_after_author_cleanup'] == EDGES + NODES
    assert graph_meta['train_positives'] == 721
    assert graph_meta['val_positives'] == 361
    assert graph_meta['original_graph_subsampled'] is False
    assert graph_meta['train_count'] == COUNTS['TR40']
    assert graph_meta['test_labels_in_model_graph'] is False
    assert set(checked.graph.ndata.keys()) == {'feature', 'label'}
    assert checked.graph.ndata['feature'].dtype == __import__('torch').float32
    assert bool(__import__('torch').isfinite(checked.graph.ndata['feature']).all())
    assert int(checked.graph.ndata['label'][27028]) == 1
    assert int((checked.graph.ndata['label'] >= 0).sum()) == COUNTS['TR40'] + COUNTS['val']
    assert not list(checked.graph.edata)
    assert len(withheld_test_ids) == COUNTS['test']
    print('TFINANCE_CANONICAL_FULL_GRAPH_CONVERSION_PREFLIGHT=PASS '
          f'model_edges={graph_meta["model_graph_edges_after_author_cleanup"]} '
          'test_scored=NO', flush=True)
    del checked, withheld_test_ids
    gc.collect()
    print('MODEL_TUNING_STARTED=NO; TEST_EVALUATED=NO', flush=True)


if __name__ == '__main__':
    main()
