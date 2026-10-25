#!/usr/bin/env python3
"""Independent, read-only integrity + mathematics audit for RGTAN T-Social Stage 2.

Does not import DGL/PyTorch or access source graph labels; does not train/test.
Only writes a fresh audit report (never edits saved research artifacts).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path('/workspace/comp8851')
CACHE = ROOT / 'assets/rgtan/tsocial/exact_risk_sparse_stage2'
REPORT = ROOT / 'diagnostics/rgtan/tsocial'
STAGE2 = Path('/workspace/RGTAN_TSOCIAL_STAGE2_SPARSE_RISK_I1_20261023')
AUTHOR = ROOT / 'repos/antifraud'
SOURCE = ROOT / 'assets/tsocial/tsocial'
SPLIT = ROOT / 'assets/tsocial/tsocial_seed2_nested_splits.npz'
SOURCE_SHA = '8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0'
SPLIT_SHA = 'b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519'
PIN = '65e583f03ac5c5983e49cc35179fe7a65908b9c1'
N, E = 5781065, 146211016
COUNT = {'TR10':578106,'TR20':1156212,'TR30':1734319,'TR40':2312426,'val':1156213,'test':2312426}
RISK_NAMES = ['degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat']
EXPECTED_FILES = ([f'risk_view_{k}.npy' for k in range(6)]
                  + [f'label_context_{k}.npy' for k in range(6)]
                  + ['train_ids.npy','train_labels.npy','val_ids.npy','fold_of_node.npy'])


def req(condition: bool, message: str):
    if not condition:
        raise RuntimeError('STOP_' + message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(16*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    req(path.is_file(), 'MISSING_' + str(path))
    return json.loads(path.read_text())


def make_pos(n: int, train_ids: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    p = np.zeros((n, 6), dtype=np.float32)
    for k in range(5):
        keep = folds != k
        p[train_ids[keep], k] = y[keep]
    p[train_ids, 5] = y
    return p


def raw_rows_independent(ptr, idx, degree, one, rows, *, node_limit: int = 5_000_000):
    """Pure Python/set independent reference of the C++ stamp-table kernel."""
    out = []
    diag = []
    for target in rows:
        target = int(target)
        inbound = np.asarray(idx[int(ptr[target]):int(ptr[target+1])], dtype=np.int64)
        req(inbound.size < node_limit, 'AUDIT_ROW_EXCEEDS_SAFETY_CAP')
        inbound_set = set(int(i) for i in inbound)
        distance2 = set()
        visited_walks = 0
        for neighbor in inbound:
            ns = idx[int(ptr[neighbor]):int(ptr[neighbor+1])]
            visited_walks += len(ns)
            req(visited_walks <= 8_000_000, 'AUDIT_SAMPLE_TWOHOP_WORK_CAP_EXCEEDED')
            distance2.update(int(x) for x in ns)
        distance2.difference_update(inbound_set)
        distance2.discard(target)
        secondary = np.fromiter(distance2,dtype=np.int64,count=len(distance2))
        first_degree = float(np.sum(degree[inbound], dtype=np.float64))
        second_degree = float(np.sum(degree[secondary], dtype=np.float64))
        first_risk = np.sum(one[inbound],axis=0,dtype=np.float64)
        second_risk = np.sum(one[secondary],axis=0,dtype=np.float64)
        arr = np.empty((6,6), dtype=np.float32)
        for k in range(6):
            arr[k] = np.asarray([degree[target],one[target,k],first_degree,
                                  second_degree,first_risk[k],second_risk[k]],dtype=np.float32)
        diag.append({'node_id':target,'incoming_neighbors':len(inbound),
                     'distinct_exclusive_twohop':len(distance2),'candidate_twohop_walks':visited_walks})
        out.append(arr)
    return np.stack(out),diag


def selftest():
    # Independent dense-reachability oracle across three small directed graphs.
    for seed in (2,42,72):
        rng = np.random.RandomState(seed)
        n=47;uu=rng.randint(0,n,700);vv=rng.randint(0,n,700)
        a=sp.coo_matrix((np.ones(len(uu),np.int8),(vv,uu)),shape=(n,n)).tocsr()
        a.sum_duplicates();a.data[:]=1;a.setdiag(0);a.eliminate_zeros();a.sort_indices()
        ptr=a.indptr.astype(np.int32);idx=a.indices.astype(np.int32)
        degree=np.diff(ptr).astype(np.float32)
        train_ids=np.arange(0,n,2,dtype=np.int64)
        y=rng.randint(0,2,size=len(train_ids)).astype(np.int8)
        folds=(np.arange(len(train_ids))%5).astype(np.int8)
        pos=make_pos(n,train_ids,y,folds)
        one=np.asarray(a@pos,dtype=np.float32)
        rows=np.arange(n,dtype=np.int64)
        observed,_=raw_rows_independent(ptr,idx,degree,one,rows)
        binary=a.toarray()!=0
        reach=(binary.astype(np.int32)@binary.astype(np.int32))>0
        reach[binary]=False
        np.fill_diagonal(reach,False)
        expected=np.empty_like(observed)
        for target in rows:
            one_nodes=np.flatnonzero(binary[target])
            two_nodes=np.flatnonzero(reach[target])
            for k in range(6):
                expected[target,k] = np.asarray([
                    degree[target],one[target,k],degree[one_nodes].sum(),
                    degree[two_nodes].sum(),one[one_nodes,k].sum(),
                    one[two_nodes,k].sum()],dtype=np.float32)
        np.testing.assert_allclose(observed,expected,rtol=0,atol=5e-4)
    print('I1_RGTAN_TSOCIAL_INDEPENDENT_ORACLE_SELFTEST=PASS seeds=2_42_72',flush=True)


def main():
    start = time.perf_counter()
    req(socket.gethostname()=='60d39211b1fd','WRONG_INSTANCE')
    manifest = read_json(CACHE/'risk_crossfit_complete.json')
    req(manifest.get('status')=='RGTAN_TSOCIAL_I1_EXACT_TRAIN_ONLY_5FOLD_CROSSFIT_COMPLETE','BUILD_NOT_COMPLETE')
    req(manifest['source_sha256']==SOURCE_SHA and manifest['split_sha256']==SPLIT_SHA,'CANONICAL_HASH_MANIFEST_CHANGED')
    req(manifest['author_commit']==PIN and manifest['nodes']==N and manifest['original_graph_edges']==E,'PROVENANCE_CHANGED')
    req(manifest['train_count']==COUNT['TR40'] and manifest['train_fraud']==69712 and manifest['fold_count']==5,'TRAIN_METADATA')
    req(manifest['ratio']=='TR40' and manifest['features']==RISK_NAMES,'FEATURE_SCHEMA')
    req(manifest['validation_labels_used'] is False and manifest['test_labels_used'] is False and manifest['test_scored'] is False,'LABEL_ISOLATION_CLAIM')
    req(manifest['tuning_started'] is False and manifest['no_test_model_inference'] is True,'STAGE2_PROTOCOL')
    req(manifest['twohop_total_candidate_walks']==18777751032,'TWOHOP_WALK_COUNT_CHANGED')
    req(manifest['risk_topology']=='exact unique 2hop excluding first hop and self; binary directed original edges','GRAPH_SEMANTICS')
    req(set(manifest['files'])==set(EXPECTED_FILES),'WRONG_OR_MISSING_ARTIFACT_FILES')
    budget=read_json(REPORT/'stage2_sparse_twohop_budget.json')
    req(budget['status']=='RGTAN_TSOCIAL_I1_EXACT_SPARSE_TWOHOP_WORK_BUDGET_PASS' and budget['test_labels_indexed'] is False,'BUDGET_NOT_PASSED')
    req(budget['total_exact_candidate_twohop_walks_all_nodes']==manifest['twohop_total_candidate_walks'],'BUDGET_WALKS_DISAGREE')

    pin=subprocess.check_output(['git','-C',str(AUTHOR),'rev-parse','HEAD'],text=True).strip()
    req(pin==PIN,'PINNED_AUTHOR_COMMIT_CHANGED')
    dirty=subprocess.check_output(['git','-C',str(AUTHOR),'status','--porcelain'],text=True).strip()
    req(not dirty,'PINNED_AUTHOR_REPOSITORY_DIRTY')
    for name,path in [('script_sha256',STAGE2/'stage2_sparse_risk.py'),('cpp_sha256',STAGE2/'sparse_twohop.cpp'),('compiled_helper_sha256',STAGE2/'sparse_twohop.so')]:
        req(path.is_file() and sha(path)==manifest[name],'BUILDER_FINGERPRINT_'+name)

    print('I1_RGTAN_TSOCIAL_AUDIT_VERIFYING_CANONICAL_SOURCE_SHA256=START',flush=True)
    req(SOURCE.stat().st_size==4110300257 and sha(SOURCE)==SOURCE_SHA,'ORIGINAL_SOURCE_SHA256')
    req(SPLIT.stat().st_size==14180251 and sha(SPLIT)==SPLIT_SHA,'FROZEN_SPLIT_SHA256')
    print('I1_RGTAN_TSOCIAL_AUDIT_CANONICAL_SOURCE_SPLIT_PIN=PASS',flush=True)

    for fname in EXPECTED_FILES:
        path=CACHE/fname
        meta=manifest['files'][fname]
        req(path.is_file() and path.stat().st_size==meta['bytes'] and sha(path)==meta['sha256'],'SAVED_ARTIFACT_HASH_'+fname)
    print('I1_RGTAN_TSOCIAL_AUDIT_ALL_16_ARTIFACT_HASHES=PASS',flush=True)

    binmeta=read_json(CACHE/'binary_adjacency_manifest.json')
    ptrp=CACHE/'inbound_binary_indptr.npy'; idxp=CACHE/'inbound_binary_indices.npy'
    for path in (ptrp,idxp):
        req(path.is_file() and sha(path)==binmeta[path.name],'SAVED_BINARY_CSR_HASH_'+path.name)
    req(binmeta['source_sha256']==SOURCE_SHA and binmeta['split_sha256']==SPLIT_SHA and binmeta['orientation']=='row=destination,column=original_inbound_source','BINARY_CSR_PROVENANCE')
    ptr=np.load(ptrp,mmap_mode='r',allow_pickle=False)
    idx=np.load(idxp,mmap_mode='r',allow_pickle=False)
    req(ptr.dtype==np.int32 and idx.dtype==np.int32 and ptr.shape==(N+1,) and idx.shape==(E,),'CSR_SHAPE')
    req(int(ptr[0])==0 and int(ptr[-1])==E,'CSR_BOUNDS')
    req(bool(np.all(np.diff(ptr)>=0)),'CSR_OFFSETS_NOT_MONOTONIC')
    req(int(idx.min())>=0 and int(idx.max())<N,'CSR_INDICES_RANGE')
    degree=np.diff(ptr).astype(np.float32)
    walks=int(np.dot(degree.astype(np.int64),np.bincount(idx,minlength=N).astype(np.int64)))
    req(walks==manifest['twohop_total_candidate_walks'],'CSR_INDEPENDENT_TOTAL_WALK_COUNT')
    print('I1_RGTAN_TSOCIAL_AUDIT_BINARY_CSR_AND_WALK_COUNT=PASS',flush=True)

    with np.load(SPLIT,allow_pickle=False) as z:
        ids={name:np.asarray(z[name],dtype=np.int64) for name in COUNT}
    for name,arr in ids.items():
        req(arr.ndim==1 and len(arr)==COUNT[name] and arr.min()>=0 and arr.max()<N,'SPLIT_COUNT_'+name)
        req(np.unique(arr).size==len(arr),'SPLIT_DUPLICATES_'+name)
    for small,big in [('TR10','TR20'),('TR20','TR30'),('TR30','TR40')]:
        req(np.isin(ids[small],ids[big]).all(),'SPLIT_NOT_NESTED')
    for left,right in [('TR40','val'),('TR40','test'),('val','test')]:
        req(np.intersect1d(ids[left],ids[right]).size==0,'SPLIT_LEAK_'+left+'_'+right)
    tr=np.load(CACHE/'train_ids.npy',mmap_mode='r',allow_pickle=False)
    y=np.load(CACHE/'train_labels.npy',mmap_mode='r',allow_pickle=False)
    val=np.load(CACHE/'val_ids.npy',mmap_mode='r',allow_pickle=False)
    fold=np.load(CACHE/'fold_of_node.npy',mmap_mode='r',allow_pickle=False)
    req(tr.shape==(COUNT['TR40'],) and np.array_equal(tr,ids['TR40']),'CROSSFIT_TRAIN_IDS')
    req(val.shape==(COUNT['val'],) and np.array_equal(val,ids['val']),'CROSSFIT_VAL_IDS')
    req(y.shape==(COUNT['TR40'],) and np.isin(np.unique(y),[0,1]).all() and int(y.sum())==69712,'CROSSFIT_TRAIN_LABELS')
    req(fold.shape==(N,) and set(np.unique(fold).tolist())=={-1,0,1,2,3,4},'CROSSFIT_FOLD_VALUES')
    istrain=np.zeros(N,dtype=np.bool_);istrain[tr]=True
    req(np.all(fold[~istrain]==-1),'NONTRAIN_FOLD_EXPOSURE')
    req(set(np.unique(fold[tr]).tolist())=={0,1,2,3,4},'TRAIN_FOLDS_NOT_COMPLETE')
    ctx=[];view=[]
    for k in range(6):
        c=np.load(CACHE/f'label_context_{k}.npy',mmap_mode='r',allow_pickle=False)
        v=np.load(CACHE/f'risk_view_{k}.npy',mmap_mode='r',allow_pickle=False)
        req(c.shape==(N,) and c.dtype==np.int8,'CONTEXT_SHAPE_'+str(k))
        req(v.shape==(N,6) and v.dtype==np.float32,'RISK_VIEW_SHAPE_'+str(k))
        req(np.all(c[~istrain]==2),'NONTRAIN_LABEL_CONTEXT_LEAK_'+str(k))
        if k<5:
            keep=fold[tr]!=k
            req(np.all(c[tr[~keep]]==2),'HELDOUT_TRAIN_FOLD_NOT_MASKED_'+str(k))
            req(np.array_equal(c[tr[keep]],y[keep]),'TRAIN_FOLD_CONTEXT_DISAGREES_'+str(k))
        else:
            req(np.array_equal(c[tr],y),'TRAIN_FULL_CONTEXT_DISAGREES')
        ctx.append(c);view.append(v)
    print('I1_RGTAN_TSOCIAL_AUDIT_FIVEFOLD_TRAIN_LABEL_ISOLATION=PASS TEST_LABELS_INDEXED=NO',flush=True)

    means=np.asarray(manifest['scaler_mean'],dtype=np.float64)
    scales=np.asarray(manifest['scaler_scale'],dtype=np.float64)
    req(means.shape==(6,) and scales.shape==(6,) and np.isfinite(means).all() and np.isfinite(scales).all() and (scales>0).all(),'SCALER_PARAMS')
    # Independently verify all OOF training rows are standardized by the train-only scaler.
    cnt=0;sum1=np.zeros(6,np.float64);sum2=np.zeros(6,np.float64)
    for k in range(5):
        fold_ids=np.asarray(tr[fold[tr]==k],dtype=np.int64)
        for start in range(0,len(fold_ids),100000):
            chunk=np.asarray(view[k][fold_ids[start:start+100000]],dtype=np.float64)
            req(np.isfinite(chunk).all(),'NONFINITE_TRAIN_OOF_RISK_'+str(k))
            cnt+=len(chunk);sum1+=chunk.sum(axis=0,dtype=np.float64);sum2+=(chunk*chunk).sum(axis=0,dtype=np.float64)
    req(cnt==len(tr),'OOF_TRAIN_ROW_COUNT')
    avg=sum1/cnt;var=sum2/cnt-avg*avg
    req(np.max(np.abs(avg))<0.01,'OOF_STANDARDIZATION_MEAN')
    req(np.all((np.abs(var-1)<0.05)|(np.abs(var)<0.05)),'OOF_STANDARDIZATION_SCALE')
    print('I1_RGTAN_TSOCIAL_AUDIT_ALL_TRAIN_OOF_STANDARDIZATION=PASS',flush=True)

    # Independently compute the direct-neighbour six risk label counts through SciPy sparse.
    pos=make_pos(N,np.asarray(tr,dtype=np.int64),np.asarray(y,dtype=np.int8),np.asarray(fold[tr],dtype=np.int8))
    a=sp.csr_matrix((np.ones(E,dtype=np.uint8),idx,ptr),shape=(N,N),copy=False)
    one=np.asarray(a@pos,dtype=np.float32)
    del a,pos
    rng=np.random.RandomState(8721)
    samples=[]
    for k in range(5):
        candidates=np.asarray(tr[fold[tr]==k],dtype=np.int64)
        samples.extend(rng.choice(candidates,2,replace=False).tolist())
    samples.extend(rng.choice(np.asarray(val),6,replace=False).tolist())
    # Add a high-degree TR40 node that is still cheap enough for independent set traversal.
    subset=np.asarray(tr[::997],dtype=np.int64)
    samples.append(int(subset[np.argmax(degree[subset])]))
    samples=list(dict.fromkeys(int(x) for x in samples))
    oracle,diag=raw_rows_independent(ptr,idx,degree,one,samples)
    max_abs=0.0;max_rel=0.0
    for i,row in enumerate(samples):
        for k in range(6):
            expected=oracle[i,k].copy()
            expected-=means
            expected/=scales
            actual=np.asarray(view[k][row],dtype=np.float32)
            diff=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
            max_abs=max(max_abs,float(diff.max()))
            max_rel=max(max_rel,float(np.max(diff/np.maximum(1,abs(expected)))))
            try:
                np.testing.assert_allclose(actual,expected,rtol=4e-4,atol=3e-3)
            except AssertionError as e:
                raise RuntimeError('STOP_EXACT_RISK_ORACLE_MISMATCH_node_'+str(row)+'_view_'+str(k)+' '+str(e))
    print('I1_RGTAN_TSOCIAL_AUDIT_INDEPENDENT_PYTHON_TWOHOP_REFERENCE=PASS rows='+str(len(samples))+' views=6',flush=True)

    record={'status':'I1_RGTAN_TSOCIAL_STAGE2_INDEPENDENT_AUDIT_PASS',
       'instance_id':'57212396','dataset':'T-Social','model':'RGTAN',
       'canonical_source_sha256':SOURCE_SHA,'frozen_split_sha256':SPLIT_SHA,
       'original_rgtan_commit':PIN,'source_graph_labels_loaded':False,
       'heldout_test_ids_read_for_disjointness':True,
       'heldout_test_labels_accessed':False,'test_scored':False,'model_training_started':False,
       'stage2_complete_manifest_sha256':sha(CACHE/'risk_crossfit_complete.json'),
       'risk_views_checked':6,'label_contexts_checked':6,'files_sha256_checked':16,
       'train_rows_OOF_standardized':cnt,'train_fraud_count':int(y.sum()),
       'candidate_twohop_walks_independently_verified':walks,
       'independent_reference_rows':diag,'independent_reference_view_count':6,
       'max_reference_absolute_difference':max_abs,'max_reference_scaled_relative_difference':max_rel,
       'scaler_OOF_mean':avg.tolist(),'scaler_OOF_variance':var.tolist(),
       'stage2_original_script_sha256':manifest['script_sha256'],
       'elapsed_wall_seconds':time.perf_counter()-start,
       'auditor_sha256':sha(Path(__file__))}
    REPORT.mkdir(parents=True,exist_ok=True)
    stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
    out=REPORT/('stage2_independent_audit_'+stamp+'.json')
    req(not out.exists(),'EXISTING_AUDIT_REPORT')
    with out.open('x') as f:
        json.dump(record,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    print('I1_RGTAN_TSOCIAL_INDEPENDENT_STAGE2_AUDIT=PASS',flush=True)
    print('I1_RGTAN_TSOCIAL_TUNING_AUTHORIZATION_GATE=STAGE2_DATA_READY_ONLY',flush=True)
    print('I1_RGTAN_TSOCIAL_AUDIT_REPORT='+str(out),flush=True)
    print('RGTAN_TUNING_STARTED=NO HELDOUT_TEST_EVALUATIONS=0',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',required=True,choices=['selftest','audit'])
    args=p.parse_args()
    try:
        if args.mode=='selftest':selftest()
        else:main()
    except BaseException as e:
        print('I1_RGTAN_TSOCIAL_INDEPENDENT_STAGE2_AUDIT_FAIL='+repr(e),flush=True)
        raise
