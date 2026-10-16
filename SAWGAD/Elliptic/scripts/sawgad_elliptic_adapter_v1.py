#!/usr/bin/env python3
"""Elliptic chronological SAWGAD adapter, stable source-order node IDs.

Model input uses temporal graph prefixes. Training-only prefix supplies feature
normalization. Validation/test edges and features never enter training forward.
All graph label fields mask validation, test and unknown labels; validation
targets are held separately and test targets read only by the final scorer.
"""
from __future__ import annotations
import hashlib
from functools import lru_cache
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/comp8851')
AUTHOR_REPO = ROOT / 'repos/SAWGAD'
SOURCE = ROOT / 'assets/elliptic/source'
SPLIT = Path('/workspace/uploads/ass2_split_candidates/Elliptic/elliptic_seed2_nested_splits.npz')
CACHE = ROOT / 'assets/elliptic/chronological_cache_v1.npz'
MANIFEST = ROOT / 'assets/elliptic/chronological_cache_v1.json'
DIAG = ROOT / 'diagnostics/sawgad/elliptic_adapter_v1.json'
AUTHOR_SHA = '64d6d531a2f03549f0bfbe461092beedd6f70bd2'
SPLIT_SHA = '1e7963e9d09935fb33e0df74cfb786cab826d1b95ae10b9592dbc240b48ffdbc'
HASHES = {
    'elliptic_txs_features.csv': 'fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0',
    'elliptic_txs_classes.csv': '93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493',
    'elliptic_txs_edgelist.csv': 'a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e',
}
COMBINED_SHA = hashlib.sha256(''.join(HASHES[n] for n in sorted(HASHES)).encode()).hexdigest()
RATIO_TIMES = {'TR10': 3, 'TR20': 7, 'TR30': 12, 'TR40': 20}
COUNTS = {'TR10': 4543, 'TR20': 9553, 'TR30': 13670, 'TR40': 18889, 'val': 8726, 'test': 18949}
N = 203769


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def verify_inputs():
    assert sha(SPLIT) == SPLIT_SHA, 'Frozen split hash changed'
    for name, checksum in HASHES.items():
        assert sha(SOURCE/name) == checksum, 'Source hash changed: '+name
    author = subprocess.check_output(['git', '-C', str(AUTHOR_REPO), 'rev-parse', 'HEAD'], text=True).strip()
    assert author == AUTHOR_SHA, 'SAWGAD author commit changed'
    print('ELLIPTIC_PINNED_SOURCE_AND_SPLIT=PASS', flush=True)


def build_cache(checked=False):
    if not checked: verify_inputs()
    if CACHE.exists() or MANIFEST.exists():
        assert CACHE.is_file() and MANIFEST.is_file(), 'Incomplete cache: STOP'
        m = json.loads(MANIFEST.read_text())
        assert m['status']=='ELLIPTIC_CHRONOLOGICAL_CACHE_PASS'
        assert m['cache_sha256'] == sha(CACHE)
        assert m['source_bundle_sha256'] == COMBINED_SHA
        assert m['split_sha256'] == SPLIT_SHA
        print('ELLIPTIC_EXISTING_CACHE=PASS', flush=True)
        return

    import pandas as pd
    print('ELLIPTIC_CACHE_BUILD_START', flush=True)
    dtype = {0: np.int64, 1: np.int16}
    dtype.update({i: np.float32 for i in range(2, 167)})
    features_df = pd.read_csv(SOURCE/'elliptic_txs_features.csv', header=None, dtype=dtype)
    assert features_df.shape == (N, 167), features_df.shape
    ids = features_df.iloc[:, 0].to_numpy(dtype=np.int64, copy=True)
    times = features_df.iloc[:, 1].to_numpy(dtype=np.int16, copy=True)
    features = features_df.iloc[:, 2:].to_numpy(dtype=np.float32, copy=True)
    del features_df
    assert len(np.unique(ids))==N and times.min()==1 and times.max()==49
    assert np.isfinite(features).all()
    id_index = pd.Index(ids)
    cls = pd.read_csv(SOURCE/'elliptic_txs_classes.csv', dtype={'txId':np.int64,'class':str})
    assert len(cls)==N and set(cls['class'].unique()) == {'1','2','unknown'}
    cls_indices = id_index.get_indexer(cls['txId'])
    assert np.all(cls_indices>=0) and np.unique(cls_indices).size==N
    # Cache labels for train + validation only; never persist test labels here.
    labels = np.full(N, -1, dtype=np.int8)
    with np.load(SPLIT, allow_pickle=False) as z:
        part = {key: np.asarray(z[key], dtype=np.int64) for key in COUNTS}
        assert int(z['split_seed'].item()) == 2
        assert len(part['TR40'])==18889 and len(part['val'])==8726 and len(part['test'])==18949
        assert len(np.intersect1d(part['TR40'],part['val'])) == 0
        assert len(np.intersect1d(part['TR40'],part['test'])) == 0
        assert len(np.intersect1d(part['val'],part['test'])) == 0
        for a,b in (('TR10','TR20'),('TR20','TR30'),('TR30','TR40')):
            assert np.isin(part[a],part[b]).all()
        for r,cutoff in RATIO_TIMES.items():
            assert np.all(times[part[r]]<=cutoff)
        assert np.all((times[part['val']]>=21)&(times[part['val']]<=31))
        assert np.all(times[part['test']]>=32)
    label_lookup = {'1':1, '2':0, 'unknown':-1}
    label_values = np.array([label_lookup[v] for v in cls['class'].to_numpy()], dtype=np.int8)
    all_labels = np.full(N, -1, dtype=np.int8)
    all_labels[cls_indices]=label_values
    assert int(np.sum(all_labels>=0)) == 46564
    for r in COUNTS:
        assert part[r].shape == (COUNTS[r],)
        assert np.all(all_labels[part[r]]>=0)
    labels[np.r_[part['TR40'],part['val']]] = all_labels[np.r_[part['TR40'],part['val']]]
    assert np.all(labels[part['test']] == -1)
    assert int(np.sum(labels>=0)) == 18889+8726
    assert int(np.sum(labels[part['val']]))==1289
    # No test targets in cache; test source labels only read after final checkpoint claim.
    del all_labels, label_values, cls
    edges = pd.read_csv(SOURCE/'elliptic_txs_edgelist.csv',dtype=np.int64)
    assert list(edges.columns) == ['txId1','txId2'] and len(edges)==234355
    src = id_index.get_indexer(edges['txId1']).astype(np.int64)
    dst = id_index.get_indexer(edges['txId2']).astype(np.int64)
    assert (src>=0).all() and (dst>=0).all()
    assert np.all(times[src] == times[dst]), 'Unexpected cross-timestep edges'
    CACHE.parent.mkdir(parents=True,exist_ok=True)
    partial = CACHE.with_suffix('.npz.partial')
    assert not partial.exists(), 'Stale partial cache; STOP'
    with partial.open('xb') as f:
        np.savez(f, ids=ids, times=times, features=features, labels=labels, src=src, dst=dst)
        f.flush(); os.fsync(f.fileno())
    os.replace(partial,CACHE)
    m={'status':'ELLIPTIC_CHRONOLOGICAL_CACHE_PASS','cache_sha256':sha(CACHE),
       'source_bundle_sha256':COMBINED_SHA,'source_sha256':HASHES,
       'split_sha256':SPLIT_SHA,'nodes':N,'features':165,'edges':234355,
       'known_labels_cached_train_val_only':int(np.sum(labels>=0)),
       'test_labels_cached':False,'test_scored':False}
    with MANIFEST.open('x') as f: json.dump(m,f,indent=2,sort_keys=True);f.write('\n')
    print('ELLIPTIC_CHRONOLOGICAL_CACHE_PASS',flush=True)


@lru_cache(maxsize=2)
def _get_cache(checked=False):
    if not checked: verify_inputs()
    build_cache(checked=True)
    with np.load(CACHE,allow_pickle=False) as z:
        c={key:z[key] for key in z.files}
    assert c['features'].shape==(N,165)
    assert c['labels'].shape==(N,)
    with np.load(SPLIT,allow_pickle=False) as z:
        s={key:np.asarray(z[key],dtype=np.int64) for key in COUNTS}
    assert np.all(c['labels'][s['test']]==-1)
    return c,s


def _view(cache, cutoff, visible_labels, min_feat, max_feat, device):
    import torch
    import dgl
    from sawgad.data import _to_homogeneous, DataSplit
    times=cache['times']
    allowed=np.flatnonzero(times<=cutoff)
    mapping=np.full(N,-1,dtype=np.int64);mapping[allowed]=np.arange(len(allowed))
    edge_ok=(times[cache['src']]<=cutoff)&(times[cache['dst']]<=cutoff)
    src=mapping[cache['src'][edge_ok]]
    dst=mapping[cache['dst'][edge_ok]]
    assert (src>=0).all() and (dst>=0).all()
    graph=dgl.graph((torch.from_numpy(src.copy()),torch.from_numpy(dst.copy())),num_nodes=len(allowed))
    values=cache['features'][allowed]
    denom=np.maximum(max_feat-min_feat,1e-12)
    normalized=(values-min_feat)/denom
    assert np.isfinite(normalized).all()
    graph.ndata['feature']=torch.from_numpy(normalized.astype(np.float32,copy=False).copy())
    masked_labels=np.full(len(allowed),-1,dtype=np.int64)
    present=visible_labels[times[visible_labels]<=cutoff]
    masked_labels[mapping[present]]=cache['labels'][present]
    assert np.all(masked_labels[mapping[present]]>=0)
    graph.ndata['label']=torch.from_numpy(masked_labels)
    graph=_to_homogeneous(graph)
    for name in list(graph.edata.keys()):del graph.edata[name]
    assert set(graph.ndata.keys())=={'feature','label'} and not graph.edata
    return graph.to(device),mapping,allowed


def load_elliptic_split(ratio='TR40',device='cpu',checked=False):
    if ratio not in RATIO_TIMES:raise ValueError(ratio)
    c,s=_get_cache(checked=checked)
    import torch
    from sawgad.data import DataSplit
    cutoff=RATIO_TIMES[ratio]
    train_global=s[ratio]
    fit=c['features'][c['times']<=cutoff]
    low=fit.min(axis=0);high=fit.max(axis=0)
    # Train cutoff excludes all future nodes. Validation's held-out labels are masked.
    train_graph,trmap,_=_view(c,cutoff,train_global,low,high,device)
    val_graph,vmap,_=_view(c,31,train_global,low,high,device)
    train_local=trmap[train_global];val_local=vmap[s['val']]
    assert (train_local>=0).all() and (val_local>=0).all()
    assert not np.isin(s['val'], np.flatnonzero(c['times']<=cutoff)).any()
    assert not np.isin(s['test'],np.flatnonzero(c['times']<=31)).any()
    assert int((train_graph.ndata['label']>=0).sum())==len(train_global)
    assert int((val_graph.ndata['label']>=0).sum())==len(train_global)
    assert torch.all(val_graph.ndata['label'][torch.from_numpy(val_local).to(device)]==-1).item()
    train=DataSplit(train_graph,torch.as_tensor(train_local,dtype=torch.long,device=device),torch.empty(0,dtype=torch.long,device=device),torch.empty(0,dtype=torch.long,device=device))
    val=DataSplit(val_graph,torch.empty(0,dtype=torch.long,device=device),torch.as_tensor(val_local,dtype=torch.long,device=device),torch.empty(0,dtype=torch.long,device=device))
    metadata={'dataset':'Elliptic','model':'SAWGAD','ratio':ratio,'train_cutoff':cutoff,
              'val_cutoff':31,'test_start':32,'train_count':len(train_global),
              'val_count':len(s['val']),'test_count':len(s['test']),
              'train_graph_nodes':train_graph.num_nodes(),'train_graph_edges':train_graph.num_edges(),
              'val_graph_nodes':val_graph.num_nodes(),'val_graph_edges':val_graph.num_edges(),
              'source_bundle_sha256':COMBINED_SHA,'split_sha256':SPLIT_SHA,
              'test_labels_in_model_graph':False,'test_graph_built':False,'test_scored':False}
    return train,val,s['test'].copy(),metadata


def validation_targets():
    c,s=_get_cache(checked=True)
    vals=c['labels'][s['val']].astype(np.int64)
    assert len(vals)==8726 and int(vals.sum())==1289 and set(np.unique(vals))=={0,1}
    return vals


def load_test_view(ratio,device):
    assert ratio in RATIO_TIMES
    c,s=_get_cache(checked=True)
    cutoff=RATIO_TIMES[ratio]
    fit=c['features'][c['times']<=cutoff]
    graph,mapping,_=_view(c,49,s[ratio],fit.min(axis=0),fit.max(axis=0),device)
    test_local=mapping[s['test']]
    assert np.all(test_local>=0)
    import torch
    assert torch.all(graph.ndata['label'][torch.as_tensor(test_local,dtype=torch.long,device=device)]==-1).item()
    return graph,test_local,s['test'].copy()


def final_test_targets(test_global):
    """Call only after durable test claim and frozen checkpoint."""
    import pandas as pd
    with np.load(SPLIT,allow_pickle=False) as z:
        assert np.array_equal(test_global,np.asarray(z['test'],dtype=np.int64))
    with np.load(CACHE,allow_pickle=False) as z: ids=z['ids'][test_global]
    cls=pd.read_csv(SOURCE/'elliptic_txs_classes.csv',dtype={'txId':np.int64,'class':str})
    lookup=cls.set_index('txId')['class'].reindex(ids)
    assert lookup.notna().all()
    label=lookup.map({'1':1,'2':0}).to_numpy(dtype=np.int64)
    assert len(label)==18949 and int(label.sum())==1485
    return label


def main():
    verify_inputs()
    build_cache(checked=True)
    tr,val,ids,meta=load_elliptic_split('TR40','cpu',checked=True)
    assert len(ids)==18949
    assert int(tr.graph.ndata['label'][tr.train_nodes].sum())==1771
    DIAG.parent.mkdir(parents=True,exist_ok=True)
    if DIAG.exists():assert json.loads(DIAG.read_text())==meta,'Existing preflight changed; STOP'
    else:
        with DIAG.open('x') as f:json.dump(meta,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(meta,indent=2),flush=True)
    print('ELLIPTIC_TEMPORAL_ADAPTER_PREFLIGHT=PASS',flush=True)
    print('TUNING_STARTED=NO; TEST_EVALUATED=NO',flush=True)


if __name__=='__main__':main()
