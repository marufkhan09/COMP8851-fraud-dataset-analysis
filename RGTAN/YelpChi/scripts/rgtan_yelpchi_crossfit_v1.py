#!/usr/bin/env python3
"""YelpChi RGTAN six-view risk statistics, exact two-hop excluding one-hop.

Reuses I2 RGTAN Amazon OOF semantics, but constructs original YelpChi
homogeneous relation-union topology. Computes two-hop reachability on CUDA
in bounded row tiles instead of holding a 45954 x 45954 float32 two-hop matrix.
No val/test labels enter risk-source arrays, normalizer, model or checkpoint.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat

from rgtan_yelpchi_core_v1 import (
    N,NF,EXPECTED_EDGES,EXPECTED_TRAIN,EXPECTED_POS,ROOT,ASSET_DIR,
    MAT,SOURCE_SHA,SPLIT_SHA,AUTHOR_COMMIT,sha256,load_splits,
    verify_input_files,assert_environment,get_graph_and_features)

NAMES=['degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat']

def atomic_json(path, data):
    path=Path(path);tmp=path.with_name(path.name+'.partial')
    with tmp.open('w') as f:
        json.dump(data,f,indent=2,sort_keys=True);f.write('\n')
    os.replace(tmp,path)

def risk_components(adj, pos, tile_rows=256):
    """Exact binary 1hop, exact >0 path 2hop minus 1hop, 6 contexts.

    Dense FP16 is used *only* to decide binary two-hop reachability. Every
    degree and risk sum is accumulated as FP32 (same as I2 Amazon reference).
    Row tiling avoids massive simultaneous dense float32 allocations.
    """
    assert adj.shape==(N,N) and adj.nnz==EXPECTED_EDGES
    assert pos.shape==(N,6) and pos.dtype==np.float32
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False
    degree=np.diff(adj.indptr).astype(np.float32)
    positive_count=adj @ pos
    first_degree=np.asarray(adj @ degree,dtype=np.float32)
    first_risk=np.asarray(adj @ positive_count,dtype=np.float32)
    positive_count=np.asarray(positive_count,dtype=np.float32)
    # Binary graph: float16 reachability matmul sums at most 45,954 ones, less
    # than FP16 maximum 65,504. Any positive value means a 2-step path exists.
    print('CUDA_EXACT_TWOHOP_DENSE_HALF_BEGIN nodes='+str(N),flush=True)
    t0=time.perf_counter()
    dense_int8=adj.toarray()
    assert dense_int8.dtype==np.int8 and dense_int8.shape==(N,N)
    dense_half=torch.from_numpy(dense_int8).to(device='cuda:0',dtype=torch.float16)
    del dense_int8
    degree_gpu=torch.from_numpy(degree).to('cuda:0')
    risk_gpu=torch.from_numpy(positive_count).to('cuda:0')
    second_degree=np.empty(N,dtype=np.float32)
    second_risk=np.empty((N,6),dtype=np.float32)
    for start in range(0,N,tile_rows):
        end=min(N,start+tile_rows)
        reachable=(dense_half[start:end] @ dense_half)>0
        reachable &= (dense_half[start:end] == 0)
        weights=reachable.to(torch.float32)
        second_degree[start:end]=(weights @ degree_gpu).cpu().numpy()
        second_risk[start:end]=(weights @ risk_gpu).cpu().numpy()
        del reachable,weights
        if start==0 or end==N or (start//tile_rows+1)%25==0:
            print(f'TWOHOP_ROWS_DONE={end}/{N} seconds={time.perf_counter()-t0:.1f}',flush=True)
    torch.cuda.synchronize()
    del dense_half,degree_gpu,risk_gpu
    torch.cuda.empty_cache()
    assert np.isfinite(second_risk).all() and np.isfinite(second_degree).all()
    # Independently check three fixed nodes against sparse CSR traversal.
    for node in (0,23000,45953):
        first=set(adj.indices[adj.indptr[node]:adj.indptr[node+1]].tolist())
        second=set()
        for nb in first:
            second.update(adj.indices[adj.indptr[nb]:adj.indptr[nb+1]].tolist())
        second.difference_update(first)
        assert np.isclose(first_degree[node],degree[list(first)].sum(),rtol=2e-5,atol=0.5),node
        assert np.isclose(second_degree[node],degree[list(second)].sum(),rtol=2e-5,atol=0.5),node
        reference=positive_count[list(second)].sum(axis=0,dtype=np.float64)
        assert np.allclose(second_risk[node],reference,rtol=2e-5,atol=0.5),(node,second_risk[node],reference)
    print('TWOHOP_INDEPENDENT_CSR_CHECK=PASS seconds=%.1f'%(time.perf_counter()-t0),flush=True)
    return degree,positive_count,first_degree,second_degree,first_risk,second_risk

def build(ratio):
    start=time.perf_counter()
    verify_input_files();assert_environment()
    parts=load_splits()
    train=parts[ratio];val=parts['val'];test=parts['test']
    asset=ASSET_DIR/f'yelpchi_{ratio.lower()}_crossfit.npz'
    manifest=ASSET_DIR/f'yelpchi_{ratio.lower()}_crossfit_manifest.json'
    ASSET_DIR.mkdir(parents=True,exist_ok=True)
    if asset.exists() or manifest.exists():
        assert asset.exists() and manifest.exists(), 'Partial artifact; manual audit required'
        meta=json.loads(manifest.read_text())
        assert meta['artifact_sha256']==sha256(asset)
        assert meta['source_sha256']==SOURCE_SHA and meta['split_sha256']==SPLIT_SHA
        assert meta['ratio']==ratio and meta['status']=='RGTAN_YELPCHI_CROSSFIT_PASS'
        print('EXISTING_CROSSFIT_VERIFIED='+ratio,flush=True)
        return
    adj,features=get_graph_and_features()
    raw=np.asarray(loadmat(MAT,variable_names=['label'])['label']).reshape(-1)
    y=np.asarray(raw[train],dtype=np.int8).copy()
    del raw
    assert set(np.unique(y).tolist())=={0,1} and int(y.sum())==EXPECTED_POS[ratio]
    fold=np.full(N,-1,dtype=np.int8)
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=2)
    for k,(_,heldout_pos) in enumerate(splitter.split(np.zeros(len(train)),y)):
        fold[train[heldout_pos]]=k
    assert np.array_equal(np.unique(fold[train]),np.arange(5))
    contexts=np.full((6,N),2,dtype=np.int8)
    positive=np.zeros((N,6),dtype=np.float32)
    for k in range(5):
        fit=fold[train]!=k
        nodes=train[fit]
        contexts[k,nodes]=y[fit]
        positive[nodes,k]=y[fit].astype(np.float32)
    contexts[5,train]=y
    positive[train,5]=y.astype(np.float32)
    outside=np.setdiff1d(np.arange(N),train,assume_unique=False)
    assert np.all(contexts[:,outside]==2)
    assert np.all(positive[outside]==0)
    assert np.all(contexts[:,val]==2) and np.all(contexts[:,test]==2)
    for k in range(5):
        held=train[fold[train]==k]
        assert np.all(contexts[k,held]==2) and np.all(positive[held,k]==0)
    print(f'TRAIN_ONLY_5FOLD_CONTEXTS_PASS ratio={ratio} train={len(train)} fraud={int(y.sum())}',flush=True)
    (degree,risk,first_deg,second_deg,first_risk,second_risk)=risk_components(adj,positive)
    del adj,positive
    raw_views=[]
    for k in range(6):
        z=np.column_stack([degree,risk[:,k],first_deg,second_deg,first_risk[:,k],second_risk[:,k]]).astype(np.float32)
        assert z.shape==(N,6) and np.isfinite(z).all()
        raw_views.append(z)
    oof=np.empty((len(train),6),dtype=np.float32)
    for k in range(5):
        which=fold[train]==k
        oof[which]=raw_views[k][train[which]]
    scaler=StandardScaler().fit(oof)
    views=[scaler.transform(z).astype(np.float32) for z in raw_views]
    assert all(np.isfinite(v).all() for v in views)
    payload={'train_ids':train,'train_labels':y,'val_ids':val,
             'fold_of_node':fold,'features':features,
             'risk_feature_names':np.asarray(NAMES),
             'scaler_mean':scaler.mean_.astype(np.float64),
             'scaler_scale':scaler.scale_.astype(np.float64)}
    for k in range(6):
        payload[f'risk_view_{k}']=views[k]
        payload[f'label_context_{k}']=contexts[k]
    temp=asset.with_suffix('.npz.partial')
    with temp.open('wb') as f:np.savez_compressed(f,**payload)
    assert temp.is_file()
    os.replace(temp,asset)
    meta={'status':'RGTAN_YELPCHI_CROSSFIT_PASS','model':'RGTAN',
          'dataset':'YelpChi','ratio':ratio,'split_seed':2,
          'author_commit':AUTHOR_COMMIT,'source_sha256':SOURCE_SHA,
          'split_sha256':SPLIT_SHA,'artifact_sha256':sha256(asset),
          'nodes':N,'features':NF,'edges':EXPECTED_EDGES,
          'train_count':len(train),'train_positive':int(y.sum()),
          'val_count':len(val),'test_count':len(test),
          'fold_seed':2,'fold_labels_masked_from_their_own_risk_views':True,
          'validation_test_labels_used_in_features':False,
          'normalization_fit_on_oof_training_only':True,
          'risk_feature_names':NAMES,'risk_feature_count':6,
          'training_contexts':5,'validation_contexts':1,
          'twohop_semantics':'exact two-step reachability excluding one-hop and self loops',
          'twohop_implementation':'GPU tiled fp16 binary reachability; fp32 feature aggregation',
          'seconds':round(time.perf_counter()-start,3),
          'training_started':False,'test_scoring_performed':False}
    atomic_json(manifest,meta)
    print('RGTAN_YELPCHI_CROSSFIT_READY='+ratio,flush=True)
    print('CROSSFIT_ASSET_SHA256='+meta['artifact_sha256'],flush=True)
    print('TEST_SCORING_PERFORMED=NO',flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ratio',choices=['TR40','TR30','TR20','TR10'],required=True)
    x=p.parse_args()
    build(x.ratio)

if __name__=='__main__':main()
