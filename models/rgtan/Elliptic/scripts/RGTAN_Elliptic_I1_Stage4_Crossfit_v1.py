#!/usr/bin/env python3
"""RGTAN Elliptic ratio-specific crossfit, frozen chronological training-prefix only.

This script produces ONLY TRAIN/VALIDATION model-input artifacts. No tuning,
no model selection, no final testing, no held-out test label access.
Five-fold out-of-fold label context, exact directed two-hop reachability
(excluding first hop and self), training-only OOF scaler.

The actual RGTAN graph in the pilot is the original directed prefix with
self-loops solely to support zero-degree attention; original risk topology
uses directed edge reachability without self-loops. All choices in manifest.
"""
import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path('/workspace/comp8851')
SOURCE = ROOT / 'assets/elliptic/source'
SPLIT = Path('/workspace/uploads/ass2_split_candidates/Elliptic/elliptic_seed2_nested_splits.npz')
RATIO='TR30'
CUT={'TR10':3,'TR20':7,'TR30':12,'TR40':20}
OUTPUT = None

MANIFEST = None
PIN = '65e583f03ac5c5983e49cc35179fe7a65908b9c1'
SHA = {
    'elliptic_txs_features.csv': 'fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0',
    'elliptic_txs_classes.csv': '93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493',
    'elliptic_txs_edgelist.csv': 'a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e',
}
SPLIT_SHA='1e7963e9d09935fb33e0df74cfb786cab826d1b95ae10b9592dbc240b48ffdbc'
COUNTS={'TR10':4543,'TR20':9553,'TR30':13670,'TR40':18889,'val':8726,'test':18949}
N=203769
NAMES=('degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat')


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for buf in iter(lambda:f.read(8*1024*1024),b''):h.update(buf)
    return h.hexdigest()


def verify():
    assert socket.gethostname()=='2be00658fac4','Wrong Vast instance: I1 only'
    assert subprocess.check_output(['git','-C',str(ROOT/'repos/antifraud'),'rev-parse','HEAD'],text=True).strip()==PIN
    for filename,expected in SHA.items():
        p=SOURCE/filename
        assert p.is_file() and digest(p)==expected, f'Canonical file changed: {filename}'
    assert digest(SPLIT)==SPLIT_SHA,'Split changed'
    with np.load(SPLIT,allow_pickle=False) as z:
        s={k:np.asarray(z[k],dtype=np.int64).copy() for k in COUNTS}
        assert int(z['split_seed'].item())==2
    for name,n in COUNTS.items():
        ids=s[name]; assert ids.shape==(n,) and len(np.unique(ids))==n and (ids>=0).all() and (ids<N).all()
    for a,b in (('TR10','TR20'),('TR20','TR30'),('TR30','TR40')):
        assert np.isin(s[a],s[b]).all()
    assert not np.intersect1d(s['TR40'],s['val']).size
    assert not np.intersect1d(s['TR40'],s['test']).size
    assert not np.intersect1d(s['val'],s['test']).size
    report=ROOT/'diagnostics/rgtan/elliptic/preflight_v1.json'
    assert report.is_file() and json.loads(report.read_text()).get('no_training_run') is True, 'I1 preflight missing'
    print('I1_PINNED_SOURCE_SPLIT_AND_PRIOR_PREFLIGHT=PASS',flush=True)
    return s


def load_data(s):
    print('READING_ORIGINAL_ELLIPTIC_FEATURES',flush=True)
    types={0:np.int64,1:np.int16}
    types.update({i:np.float32 for i in range(2,167)})
    t=pd.read_csv(SOURCE/'elliptic_txs_features.csv',header=None,dtype=types)
    assert t.shape==(N,167)
    txids=t.iloc[:,0].to_numpy(dtype=np.int64,copy=True)
    times=t.iloc[:,1].to_numpy(dtype=np.int16,copy=True)
    features=t.iloc[:,2:].to_numpy(dtype=np.float32,copy=True)
    del t
    assert len(np.unique(txids))==N and times.min()==1 and times.max()==49
    assert np.isfinite(features).all()
    assert (times[s[RATIO]]<=CUT[RATIO]).all()
    assert ((times[s['val']]>=21)&(times[s['val']]<=31)).all()
    assert (times[s['test']]>=32).all()
    # Filter original labels at parse-time to the selected ratio training nodes ONLY.
    # Validation and test labels are deliberately not collected.
    allowed=set(map(int,txids[s[RATIO]]))
    record={}
    with (SOURCE/'elliptic_txs_classes.csv').open(newline='') as f:
        reader=csv.DictReader(f)
        assert set(reader.fieldnames)=={'txId','class'}
        for row in reader:
            tx=int(row['txId'])
            if tx in allowed:
                assert tx not in record and row['class'] in ('1','2')
                record[tx]=int(row['class']=='1')
    assert len(record)==len(allowed)
    train_y=np.array([record[int(tx)] for tx in txids[s[RATIO]]],dtype=np.int8)
    assert train_y.shape==(COUNTS[RATIO],) and 5 < int(train_y.sum()) < len(train_y)-5
    edges=pd.read_csv(SOURCE/'elliptic_txs_edgelist.csv',dtype=np.int64)
    assert tuple(edges.columns)==('txId1','txId2') and len(edges)==234355
    index=pd.Index(txids)
    src=index.get_indexer(edges['txId1']).astype(np.int64)
    dst=index.get_indexer(edges['txId2']).astype(np.int64)
    del edges
    assert (src>=0).all() and (dst>=0).all() and (times[src]==times[dst]).all()
    print('I1_CHRONOLOGY_AND_TRAIN_LABELS_ONLY=PASS',flush=True)
    return txids,times,features,src,dst,train_y


def prefix(times,src,dst,cutoff):
    ids=np.flatnonzero(times<=cutoff)
    local=np.full(N,-1,dtype=np.int64)
    local[ids]=np.arange(len(ids),dtype=np.int64)
    keep=(times[src]<=cutoff)&(times[dst]<=cutoff)
    u=local[src[keep]];v=local[dst[keep]]
    assert (u>=0).all() and (v>=0).all()
    # Rows are RGTAN aggregation destinations; columns are inbound sources.
    a=sp.coo_matrix((np.ones(len(u),dtype=np.int8),(v,u)),shape=(len(ids),len(ids))).tocsr()
    a.sum_duplicates();a.data.fill(1)
    a.setdiag(0);a.eliminate_zeros();a.sort_indices()
    assert a.data.size==a.nnz and (a.data==1).all()
    return ids,local,a


def exact_reachability(a, tag):
    # Sparse exact A*A, not dense N x N; binary 2-step reachability semantics.
    print('EXACT_SPARSE_TWO_HOP_START',tag,'nodes',a.shape[0],'edges',a.nnz,flush=True)
    start=time.perf_counter()
    two=(a.astype(np.int32)@a.astype(np.int32)).tocsr()
    assert (two.data>0).all()
    two.data.fill(1)
    two=two-two.multiply(a.astype(np.int32))
    two.setdiag(0);two.eliminate_zeros();two.sort_indices()
    assert (two.data==1).all()
    print('EXACT_SPARSE_TWO_HOP=PASS',tag,'reachable',two.nnz,'seconds',round(time.perf_counter()-start,2),flush=True)
    return two


def components(a,two,pos):
    # YelpChi prior RGTAN six-feature definitions; use exact directed topology.
    degree=np.diff(a.indptr).astype(np.float32)
    positive=(a@pos).astype(np.float32)
    first_degree=np.asarray(a@degree,dtype=np.float32)
    second_degree=np.asarray(two@degree,dtype=np.float32)
    first_risk=np.asarray(a@positive,dtype=np.float32)
    second_risk=np.asarray(two@positive,dtype=np.float32)
    return degree,positive,first_degree,second_degree,first_risk,second_risk


def risk_view(comp,k):
    deg,positive,fd,sd,fr,sr=comp
    return np.column_stack((deg,positive[:,k],fd,sd,fr[:,k],sr[:,k])).astype(np.float32)


def main():
    global RATIO, OUTPUT, MANIFEST
    pa=argparse.ArgumentParser();pa.add_argument('--ratio',choices=('TR30','TR20','TR10'),required=True)
    RATIO=pa.parse_args().ratio
    OUTPUT=ROOT/f'assets/rgtan/elliptic/RGTAN_Elliptic_{RATIO}_TemporalCrossfit_v1.npz'
    MANIFEST=OUTPUT.with_suffix('.json')
    start=time.perf_counter()
    s=verify()
    if OUTPUT.exists() or MANIFEST.exists():
        assert OUTPUT.exists() and MANIFEST.exists(),'Partial artifact; inspect before retrying'
        meta=json.loads(MANIFEST.read_text())
        assert meta.get('status')==f'RGTAN_ELLIPTIC_{RATIO}_REAL_CROSSFIT_PASS' and meta['artifact_sha256']==digest(OUTPUT)
        print(f'RGTAN_Elliptic_{RATIO}_EXISTING_CROSSFIT_REUSED=PASS',flush=True)
        return
    txids,times,features,src,dst,y=load_data(s)
    trids,trmap,train_a=prefix(times,src,dst,CUT[RATIO])
    vlids,vlmap,val_a=prefix(times,src,dst,31)
    assert not np.isin(s['val'],trids).any() and not np.isin(s['test'],vlids).any()
    train_local=trmap[s[RATIO]]
    val_local=vlmap[s['val']]
    assert (train_local>=0).all() and (val_local>=0).all()
    print('TRAIN_VALIDATION_GRAPH_PREFIX_ISOLATION=PASS','train',len(trids),train_a.nnz,'validation',len(vlids),val_a.nnz,flush=True)
    folds=np.full(len(trids),-1,dtype=np.int8)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=2)
    for k,(_,held) in enumerate(skf.split(np.zeros(len(y)),y)):
        folds[train_local[held]]=k
    assert set(np.unique(folds[train_local]))==set(range(5))
    ctx=np.full((6,len(trids)),2,dtype=np.int8)
    positives=np.zeros((len(trids),6),dtype=np.float32)
    for k in range(5):
        can=(folds[train_local]!=k)
        nodes=train_local[can]
        ctx[k,nodes]=y[can]
        positives[nodes,k]=y[can]
        assert np.all(ctx[k,train_local[~can]]==2)
    ctx[5,train_local]=y
    positives[train_local,5]=y
    assert np.all(ctx[:,np.setdiff1d(np.arange(len(trids)),train_local,assume_unique=False)]==2)
    for k in range(5):
        assert np.all(positives[train_local[folds[train_local]==k],k]==0)
    print(f'RGTAN_Elliptic_{RATIO}_TRAIN_ONLY_5_FOLD_MASKING=PASS',flush=True)
    train_two=exact_reachability(train_a,f'RGTAN_Elliptic_{RATIO}_TRAIN_T{CUT[RATIO]}')
    train_comp=components(train_a,train_two,positives)
    # These views are the graph-specific OOF risk features for fitting the scaler.
    raw=[risk_view(train_comp,k) for k in range(6)]
    oof=np.empty((len(train_local),6),dtype=np.float32)
    for k in range(5):
        ix=folds[train_local]==k
        oof[ix]=raw[k][train_local[ix]]
    scaler=StandardScaler().fit(oof)
    views=[scaler.transform(z).astype(np.float32) for z in raw]
    del raw,train_comp,train_two,oof
    val_two=exact_reachability(val_a,'VALIDATION_T31')
    val_pos=np.zeros((len(vlids),1),dtype=np.float32)
    val_ctx=np.full(len(vlids),2,dtype=np.int8)
    val_pos[vlmap[s[RATIO]],0]=y
    val_ctx[vlmap[s[RATIO]]]=y
    assert (val_ctx[val_local]==2).all()
    # Compute full training-context risk features on historical t<=31 only.
    val_comp=components(val_a,val_two,val_pos)
    validation_view=scaler.transform(risk_view(val_comp,0)).astype(np.float32)
    assert np.isfinite(validation_view).all() and all(np.isfinite(v).all() for v in views)
    assert all((ctx[k,train_local[folds[train_local]==k]]==2).all() for k in range(5))
    assert np.all(val_ctx[val_local]==2)
    print('VAL_VIEW_TRAIN_ONLY_CONTEXT_AND_OOF_SCALING=PASS',flush=True)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    tmp=OUTPUT.with_suffix('.npz.partial')
    assert not tmp.exists(),'Stale partial file: inspect first'
    payload={'train_prefix_global_ids':trids,'val_prefix_global_ids':vlids,
       'train_global_ids':s[RATIO],'train_prefix_local_ids':train_local,'train_labels':y,
       'val_global_ids':s['val'],'val_prefix_local_ids':val_local,
       'train_fold_of_local':folds,'val_context':val_ctx,'val_risk_view':validation_view,
       'risk_feature_names':np.asarray(NAMES),'scaler_mean':scaler.mean_,
       'scaler_scale':scaler.scale_}
    for k in range(6):
        payload[f'train_risk_view_{k}']=views[k]
        payload[f'train_label_context_{k}']=ctx[k]
    with tmp.open('xb') as f:
        np.savez_compressed(f,**payload)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,OUTPUT)
    metadata={
       'status':f'RGTAN_ELLIPTIC_{RATIO}_REAL_CROSSFIT_PASS',
       'source_sha256':SHA,'split_sha256':SPLIT_SHA,'author_commit':PIN,
       'model':'RGTAN','dataset':'Elliptic','ratio':RATIO,'fold_seed':2,
       'training_count':len(train_local),'training_positives':int(y.sum()),
       'validation_count':len(val_local),
       'training_timestep_max':CUT[RATIO],'validation_timestep_max':31,'test_timestep_start':32,
       'training_prefix_nodes':len(trids),'validation_prefix_nodes':len(vlids),
       'train_original_directed_binary_nonself_edges':int(train_a.nnz),
       'validation_original_directed_binary_nonself_edges':int(val_a.nnz),
       'risk_feature_names':NAMES,'twohop_semantics':'exact directed two-hop reachability excluding one-hop and self',
       'normalizer':'StandardScaler fit on training-node heldout-fold OOF risk only',
       'training_label_contexts':5,'validation_train_only_context':True,
       'validation_labels_accessed':False,'test_labels_accessed':False,
       'future_nodes_used_for_training':False,'training_started':False,'test_scored':False,
       'artifact_sha256':digest(OUTPUT),'seconds':round(time.perf_counter()-start,2),
       'script_sha256':digest(__file__),
    }
    with MANIFEST.open('x') as f:
        json.dump(metadata,f,indent=2,sort_keys=True);f.write('\n')
    print(f'RGTAN_ELLIPTIC_{RATIO}_REAL_CROSSFIT_PASS',flush=True)
    print('ARTIFACT',OUTPUT,'SHA256',metadata['artifact_sha256'],flush=True)
    print('TUNING_STARTED=NO TEST_SCORED=NO',flush=True)

if __name__=='__main__':main()
