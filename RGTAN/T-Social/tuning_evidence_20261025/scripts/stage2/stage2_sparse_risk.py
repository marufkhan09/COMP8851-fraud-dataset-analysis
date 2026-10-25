#!/usr/bin/env python3
"""I1 RGTAN x T-Social: EXACT sparse, training-only OOF 2-hop risk features.

Stages:
 --selftest: synthetic graph proof vs dense T-Finance reference, no dataset.
 --budget: canonical SHA+split audit, disk-backed binary CSR, 2048-row exact
           two-hop timing sample + exact global walk count. No label read.
 --build: optional and irreversible artifact build; no model training/test scoring.

C++ kernel only computes exact exclusive 2-hop features; original RGTAN code unchanged.
"""
from __future__ import annotations
import argparse, ctypes, hashlib, json, os, subprocess, socket, sys, time, math
from pathlib import Path
import numpy as np
import scipy.sparse as sp

BASE=Path('/workspace/comp8851')
SRC=BASE/'assets/tsocial/tsocial'
SPLIT=BASE/'assets/tsocial/tsocial_seed2_nested_splits.npz'
AUTHOR=BASE/'repos/antifraud'
CACHE=BASE/'assets/rgtan/tsocial/exact_risk_sparse_stage2'
REPORT=BASE/'diagnostics/rgtan/tsocial'
SRC_SHA='8d577114cff12f7de35eda2974f17825ef9febe20c661e95e8d21072a6dfc8d0'
SPLIT_SHA='b7e76835612e72a8fe5c69bbe87c8ea7b2e5790cbd8849e2a3ac776364bc6519'
PIN='65e583f03ac5c5983e49cc35179fe7a65908b9c1'
HOST='60d39211b1fd'
N=5781065;E=146211016;F=10
COUNTS={'TR10':578106,'TR20':1156212,'TR30':1734319,'TR40':2312426,'val':1156213,'test':2312426}
RISK_NAMES=['degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat']
CPP=Path(__file__).with_name('sparse_twohop.cpp')
SO=Path(__file__).with_name('sparse_twohop.so')
WORKERS=6
BUDGET_N=2048


def require(c, msg):
    if not c: raise RuntimeError('STOP_'+str(msg))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(16*1024*1024),b''):h.update(x)
    return h.hexdigest()


def atomic_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    require(not path.exists() and not tmp.exists(),'EXISTING_REPORT_OR_PARTIAL_'+str(path))
    with tmp.open('x') as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def npy_write_once(path,value):
    tmp=path.with_name(path.name+'.partial')
    require(not path.exists() and not tmp.exists(),'EXISTING_OUTPUT_OR_PARTIAL_'+str(path))
    with tmp.open('xb') as f:
        np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def env_check(check_graph=True):
    require(socket.gethostname()==HOST,'WRONG_INSTANCE')
    import torch,dgl,psutil
    require(torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1','WRONG_MODEL_ENV')
    require(subprocess.check_output(['git','-C',str(AUTHOR),'rev-parse','HEAD'],text=True).strip()==PIN,'WRONG_AUTHOR_PIN')
    require(not subprocess.check_output(['git','-C',str(AUTHOR),'status','--porcelain'],text=True).strip(),'AUTHOR_REPO_DIRTY')
    require(psutil.virtual_memory().available>24*(1024**3),'LESS_THAN_24GIB_AVAILABLE_RAM')
    if check_graph:
        require(SRC.stat().st_size==4110300257 and sha(SRC)==SRC_SHA,'CANONICAL_SOURCE_CHANGED')
        require(sha(SPLIT)==SPLIT_SHA,'FROZEN_SPLIT_CHANGED')
    return torch,dgl,psutil


def read_splits():
    with np.load(SPLIT,allow_pickle=False) as z:
        ids={k:np.array(z[k],dtype=np.int64,copy=True) for k in COUNTS}
        for seed_key in ('split_seed','seed'):
            if seed_key in z.files: require(int(z[seed_key])==2,'WRONG_SEED')
    for k,a in ids.items():
        require(a.ndim==1 and len(a)==COUNTS[k] and a.min()>=0 and a.max()<N,'SPLIT_SHAPE_'+k)
        require(len(np.unique(a))==len(a),'SPLIT_DUPLICATES_'+k)
    for a,b in (('TR10','TR20'),('TR20','TR30'),('TR30','TR40')):
        require(np.isin(ids[a],ids[b]).all(),'TRAIN_NOT_NESTED')
    for a,b in (('TR40','val'),('TR40','test'),('val','test')):
        require(not np.intersect1d(ids[a],ids[b],assume_unique=True).size,'SPLIT_OVERLAP')
    return ids


def compile_cpp():
    if not SO.is_file():
        require(subprocess.run(['which','g++'],capture_output=True).returncode==0,'MISSING_GPP_COMPILER')
        tmp=SO.with_suffix('.so.partial')
        require(not tmp.exists(),'STALE_PARTIAL_SO')
        subprocess.run(['g++','-std=c++17','-O3','-fPIC','-shared','-fopenmp',str(CPP),'-o',str(tmp)],check=True)
        os.replace(tmp,SO)
    dll=ctypes.CDLL(str(SO))
    func=dll.exact_exclusive_twohop
    i32p=np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    fp=np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    i64p=np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    func.argtypes=[i32p,i32p,ctypes.c_int32,fp,fp,i32p,ctypes.c_int32,fp,fp,i64p,ctypes.c_int]
    func.restype=ctypes.c_int
    return func


def twohop_kernel(func,indptr,indices,degree,one6,rows,threads=WORKERS):
    rows=np.ascontiguousarray(rows,dtype=np.int32)
    require(rows.ndim==1 and len(np.unique(rows))==len(rows) and (not len(rows) or (rows.min()>=0 and rows.max()<len(degree))),'KERNEL_ROW_IDS_INVALID')
    require(one6.shape==(len(degree),6) and one6.dtype==np.float32,'ONEHOP_RISK_LAYOUT_INVALID')
    count=len(rows)
    outdegree=np.zeros(count,dtype=np.float32)
    outrisk=np.zeros((count,6),dtype=np.float32)
    work=np.zeros(count,dtype=np.int64)
    rv=func(indptr,indices,len(degree),degree,one6,rows,count,outdegree,outrisk,work,threads)
    require(rv==0,'CPP_KERNEL_FAILED')
    require(np.isfinite(outdegree).all() and np.isfinite(outrisk).all(),'NONFINITE_TWOHOP')
    return outdegree,outrisk,work


def selftest():
    func=compile_cpp()
    rng=np.random.RandomState(8)
    for seed in [2,42,72]:
        rs=np.random.RandomState(seed)
        n=49;uu=rs.randint(0,n,400);vv=rs.randint(0,n,400)
        # Include duplicates and self loops in source graph; risk adjacency removes them.
        rr=sp.coo_matrix((np.ones(400,np.uint8),(vv,uu)),shape=(n,n)).tocsr()
        rr.sum_duplicates();rr.data[:]=1;rr.setdiag(0);rr.eliminate_zeros();rr.sort_indices()
        degree=np.diff(rr.indptr).astype(np.float32)
        rng6=rs.randint(0,2,(n,6)).astype(np.float32)
        one6=np.asarray(rr@rng6,dtype=np.float32)
        rows=np.arange(n,dtype=np.int32)
        d,r,w=twohop_kernel(func,rr.indptr,rr.indices,degree,one6,rows,threads=3)
        direct=(rr.toarray()!=0)
        reach=(direct.astype(np.int32)@direct.astype(np.int32)>0)
        reach[direct]=False
        np.fill_diagonal(reach,False)
        assert np.allclose(d,reach.astype(np.float32)@degree,rtol=1e-5,atol=1e-3)
        assert np.allclose(r,reach.astype(np.float32)@one6,rtol=1e-5,atol=1e-3)
        assert np.array_equal(w,np.asarray((rr.astype(np.int32)@degree.astype(np.int32)).reshape(-1),dtype=np.int64))
    print('I1_RGTAN_TSOCIAL_EXACT_TWOHOP_EQUIVALENT_TO_DENSE_REFERENCE=PASS seeds=2_42_72',flush=True)
    print('I1_RGTAN_TSOCIAL_CPP_KERNEL_SELFTEST=PASS',flush=True)


def persist_csr(dgl):
    CACHE.mkdir(parents=True,exist_ok=True)
    ptr=CACHE/'inbound_binary_indptr.npy';idx=CACHE/'inbound_binary_indices.npy';meta=CACHE/'binary_adjacency_manifest.json'
    if any(f.exists() for f in (ptr,idx,meta)):
        require(all(f.is_file() for f in (ptr,idx,meta)),'INCOMPLETE_BINARY_ADJACENCY_CACHE')
        m=json.loads(meta.read_text())
        for p in (ptr,idx):require(sha(p)==m[p.name],'BINARY_CACHE_HASH_MISMATCH_'+p.name)
        require(m['source_sha256']==SRC_SHA and m['split_sha256']==SPLIT_SHA,'BINARY_CACHE_CANONICAL_MISMATCH')
        pi=np.load(ptr,mmap_mode='r',allow_pickle=False);ix=np.load(idx,mmap_mode='r',allow_pickle=False)
        require(pi.dtype==np.int32 and ix.dtype==np.int32 and pi.shape==(N+1,) and pi[-1]==ix.shape[0],'CACHED_CSR_SHAPE_MISMATCH')
        print(f'I1_RGTAN_TSOCIAL_BINARY_CSR_CACHE_REUSED=PASS unique_edges={len(ix)}',flush=True)
        return pi,ix,m
    print('I1_RGTAN_TSOCIAL_BUILDING_BINARY_ADJACENCY_FROM_ORIGINAL_DIRECTED_GRAPH=START',flush=True)
    g_list,_=dgl.load_graphs(str(SRC))
    require(len(g_list)==1,'SOURCE_GRAPH_COUNT')
    g=g_list[0]
    require(g.num_nodes()==N and g.num_edges()==E,'GRAPH_NODE_OR_EDGE_COUNT')
    require(list(g.canonical_etypes)==[('_U','_V','_U')],'GRAPH_RELATION')
    for k in list(g.ndata):g.ndata.pop(k)
    for k in list(g.edata):g.edata.pop(k)
    u,v=g.edges(order='eid')
    # The risk-only binary matrix uses (destination,source). Model graph is unchanged.
    uu=u.numpy().astype(np.int32,copy=False);vv=v.numpy().astype(np.int32,copy=False)
    import gc
    coo=sp.coo_matrix((np.ones(E,dtype=np.uint8),(vv,uu)),shape=(N,N),dtype=np.uint8)
    adj=coo.tocsr()
    del coo,uu,vv,u,v,g,g_list
    gc.collect()
    adj.sum_duplicates();adj.data.fill(1)
    adj.setdiag(0);adj.eliminate_zeros();adj.sort_indices()
    require(adj.indptr.dtype==np.int32 and adj.indices.dtype==np.int32,'CSR_INDEX_FORMAT_NOT_INT32')
    require(adj.nnz>0 and adj.nnz<=E,'BAD_BINARY_ADJACENCY_NONZERO_COUNT')
    for p,value in ((ptr,adj.indptr),(idx,adj.indices)):
        npy_write_once(p,value)
    details={'status':'CANONICAL_T_SOCIAL_BINARY_INBOUND_ADJACENCY_BUILT','source_sha256':SRC_SHA,
      'split_sha256':SPLIT_SHA,'nodes':N,'original_directed_edges':E,
      'risk_binary_directed_edges_no_self':int(adj.nnz),
      'orientation':'row=destination,column=original_inbound_source',
      'exclusion':'risk-only adjacency deduplicated edges and removes self-loops; original model graph unaffected',
      ptr.name:sha(ptr),idx.name:sha(idx),'test_labels_indexed':False}
    atomic_json(meta,details)
    print(f'I1_RGTAN_TSOCIAL_BINARY_CSR_BUILT=PASS unique_edges={adj.nnz}',flush=True)
    return np.load(ptr,mmap_mode='r',allow_pickle=False),np.load(idx,mmap_mode='r',allow_pickle=False),details


def sampling_ids(ids, degree, n=BUDGET_N):
    # Stratified by observed degree to include high-degree nodes, plus random seeds.
    tr=ids['TR40'];rng=np.random.RandomState(2)
    choices=[]
    for q in [0,.1,.25,.5,.75,.9,.95,.99,1]:
        ix=min(len(tr)-1,int((len(tr)-1)*q))
        sorted_tr=tr[np.argsort(degree[tr],kind='stable')]
        choices.append(int(sorted_tr[ix]))
    remaining=rng.choice(tr,size=n-len(choices),replace=False)
    choices=np.unique(np.concatenate([np.asarray(choices,np.int64),remaining])).astype(np.int32)
    return choices


def budget(ids, ptr,idx):
    func=compile_cpp()
    degree=np.diff(ptr).astype(np.float32)
    # All candidate two-hop walk counts are exact and computable as sum(outdeg*indeg).
    outdegree=np.bincount(idx,minlength=N).astype(np.int64)
    total=int(np.dot(degree.astype(np.int64),outdegree))
    require(total>0,'BAD_TWOHOP_WORK_COUNT')
    work_samples=sampling_ids(ids,degree)
    # One-hop risk inputs are zeros only for structural timing; not saved as model inputs.
    riskzero=np.zeros((N,6),dtype=np.float32)
    start=time.perf_counter()
    sd,sr,w=twohop_kernel(func,ptr,idx,degree,riskzero,work_samples)
    elapsed=time.perf_counter()-start
    weighted_rate=float(w.sum())/max(elapsed,1e-9)
    report={'status':'RGTAN_TSOCIAL_I1_EXACT_SPARSE_TWOHOP_WORK_BUDGET_PASS',
      'source_sha256':SRC_SHA,'split_sha256':SPLIT_SHA,'author_commit':PIN,
      'original_edges':E,'binary_edges':len(idx),'nodes':N,
      'total_exact_candidate_twohop_walks_all_nodes':total,
      'sample_unique_nodes':int(len(work_samples)),
      'sample_candidate_walks':int(w.sum()),'sample_max_walks_per_node':int(w.max()),
      'sample_seconds':elapsed,'sample_walks_per_second':weighted_rate,
      'idealized_estimate_full_seconds':float(total/weighted_rate),
      'idealized_estimate_warning':'Approximate only; sample may be unrepresentative and full run includes sparse preprocessing and file I/O',
      'workers':WORKERS,'method':'exact CSR streamed unique exclusive 2-hop; no NxN or full A squared',
      'test_labels_indexed':False,'tuning_started':False,'model_training_started':False,
      'script_sha256':sha(Path(__file__)),'cpp_sha256':sha(CPP),'compiled_helper_sha256':sha(SO)}
    path=REPORT/'stage2_sparse_twohop_budget.json'
    if path.exists():
        old=json.loads(path.read_text());require(old['status']==report['status'] and old['source_sha256']==SRC_SHA,'OLD_BUDGET_MANIFEST_MISMATCH')
        print('I1_RGTAN_TSOCIAL_PREVIOUS_SPARSE_WORK_BUDGET_PRESERVED=YES',flush=True)
        path=REPORT/f'stage2_sparse_twohop_budget_repeat_{int(time.time())}.json'
    atomic_json(path,report)
    print('I1_RGTAN_TSOCIAL_TWOHOP_WORK_BUDGET='+json.dumps(report,sort_keys=True),flush=True)
    print('I1_RGTAN_TSOCIAL_TWOHOP_BUDGET_PASS=PASS TEST_LABELS_INDEXED=NO',flush=True)


def build(ids, ptr, idx, workers):
    require(not (CACHE/'risk_crossfit_complete.json').exists(),'CROSSFIT_ALREADY_COMPLETE_DO_NOT_OVERWRITE')
    # Never allow a half-output to be mistaken for verified finished inputs.
    pending=list(CACHE.glob('risk_view_*.npy'))+list(CACHE.glob('label_context_*.npy'))+list(CACHE.glob('*.partial'))
    require(not pending,'EXISTING_RISK_FILES_OR_PARTIAL_INSPECT_BEFORE_RETRY')
    func=compile_cpp()
    import torch,dgl
    import sklearn.preprocessing
    from sklearn.model_selection import StratifiedKFold
    gs,_=dgl.load_graphs(str(SRC));require(len(gs)==1 and gs[0].num_edges()==E,'CANONICAL_GRAPH_NOT_FOUND')
    g=gs[0]
    raw=g.ndata['label'];require(tuple(raw.shape) in ((N,),(N,1)),'BAD_SOURCE_LABEL_SHAPE')
    tr=ids['TR40']
    yy=raw[torch.from_numpy(tr.copy())].long().reshape(-1).numpy().astype(np.int8,copy=True)
    require(int(yy.sum())==69712,'TR40_FRAUD_COUNT_MISMATCH')
    del raw,g,gs
    fold_of=np.full(N,-1,dtype=np.int8)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=2)
    for k,(_,held) in enumerate(skf.split(np.zeros(len(tr),dtype=np.int8),yy)):
        fold_of[tr[held]]=k
    require(np.array_equal(np.unique(fold_of[tr]),np.arange(5)),'CROSSFIT_FOLDS_NOT_POPULATED')
    pos=np.zeros((N,6),dtype=np.float32)
    ctx=np.full((N,6),2,dtype=np.int8)
    for k in range(5):
        other=fold_of[tr]!=k
        nodes=tr[other];pos[nodes,k]=yy[other];ctx[nodes,k]=yy[other]
        require((pos[tr[~other],k]==0).all() and (ctx[tr[~other],k]==2).all(),'FOLD_LEAK')
    pos[tr,5]=yy;ctx[tr,5]=yy
    # Train labels outside TR40 never accessed; all risk-label sources are TR40-only.
    mask=np.zeros(N,dtype=bool);mask[tr]=True
    require((pos[~mask]==0).all() and (ctx[~mask]==2).all(),'VALIDATION_OR_TEST_LABEL_LEAK')
    print('I1_RGTAN_TSOCIAL_5FOLD_TRAIN_ONLY_RISK_LABEL_CONTEXT=PASS',flush=True)
    degree=np.diff(ptr).astype(np.float32)
    a=sp.csr_matrix((np.ones(len(idx),np.uint8),idx,ptr),shape=(N,N),copy=False)
    start=time.perf_counter()
    # 1-hop scores: direct incident risk; direct degree; 1-hop second risk sum.
    one=np.asarray(a@pos,dtype=np.float32)
    first_d=np.asarray(a@degree,dtype=np.float32).reshape(-1)
    first_risk=np.asarray(a@one,dtype=np.float32)
    del a,pos
    print(f'I1_RGTAN_TSOCIAL_ONEHOP_RISK_FEATURES_BUILT=PASS seconds={time.perf_counter()-start:.1f}',flush=True)
    two_d=np.zeros(N,dtype=np.float32)
    two_risk=np.zeros((N,6),dtype=np.float32)
    work=np.zeros(N,dtype=np.int64)
    chunk=100_000
    for offset in range(0,N,chunk):
        end=min(N,offset+chunk)
        rows=np.arange(offset,end,dtype=np.int32)
        t=time.perf_counter()
        d,r,w=twohop_kernel(func,ptr,idx,degree,one,rows,threads=workers)
        two_d[offset:end]=d;two_risk[offset:end]=r;work[offset:end]=w
        print(f'I1_RGTAN_TSOCIAL_EXACT_TWOHOP_ROWS={end}/{N} elapsed_chunk_s={time.perf_counter()-t:.1f} total_walks={int(work[:end].sum())}',flush=True)
    require(np.isfinite(two_d).all() and np.isfinite(two_risk).all(),'NONFINITE_EXACT_TWOHOP_AGGREGATES')
    # A train-only OOF scaler, built in chunks. Never fit on val/test IDs.
    scaler=sklearn.preprocessing.StandardScaler()
    for start in range(0,len(tr),100_000):
        idslice=tr[start:start+100_000]
        k=fold_of[idslice]
        feats=np.stack([degree[idslice],one[idslice,k],first_d[idslice],two_d[idslice],
                        first_risk[idslice,k],two_risk[idslice,k]],axis=1).astype(np.float32)
        require(np.isfinite(feats).all(),'BAD_OOF_FEATURES')
        scaler.partial_fit(feats)
    print('I1_RGTAN_TSOCIAL_SCALER_FIT_TRAIN_ONLY_OOF=PASS',flush=True)
    # Save each context and risk-view as standalone NPY to enable memory-mapped Stage 3.
    outfiles=[]
    for k in range(6):
        f=CACHE/f'risk_view_{k}.npy';tmp=f.with_name(f.name+'.partial')
        require(not f.exists() and not tmp.exists(),'RISK_VIEW_ALREADY_EXISTS')
        # Create a standalone memory-mappable matrix; never overwrite an existing artifact.
        mat=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(N,6))
        for start in range(0,N,100_000):
            end=min(N,start+100_000)
            raw=np.stack([degree[start:end],one[start:end,k],first_d[start:end],two_d[start:end],
                          first_risk[start:end,k],two_risk[start:end,k]],axis=1).astype(np.float32)
            x=scaler.transform(raw).astype(np.float32,copy=False)
            require(np.isfinite(x).all(),'NORMALIZED_RISK_NONFINITE')
            mat[start:end]=x
        mat.flush();del mat
        os.replace(tmp,f)
        outfiles.append(f)
        c=CACHE/f'label_context_{k}.npy';npy_write_once(c,ctx[:,k]);outfiles.append(c)
        print(f'I1_RGTAN_TSOCIAL_SAVED_EXACT_RISK_VIEW={k} context=TRAIN_ONLY',flush=True)
    for name,value in [('train_ids.npy',tr),('train_labels.npy',yy),('val_ids.npy',ids['val']),('fold_of_node.npy',fold_of)]:
        p=CACHE/name;npy_write_once(p,value);outfiles.append(p)
    for k in range(5):
        selected=(fold_of[tr]==k)
        assert np.all(ctx[tr[selected],k]==2) and np.all(ctx[tr[~selected],k]==yy[~selected])
    manifest={'status':'RGTAN_TSOCIAL_I1_EXACT_TRAIN_ONLY_5FOLD_CROSSFIT_COMPLETE',
      'source_sha256':SRC_SHA,'split_sha256':SPLIT_SHA,'author_commit':PIN,
      'ratio':'TR40','nodes':N,'original_graph_edges':E,'binary_directed_edges_no_self':len(idx),
      'train_count':len(tr),'train_fraud':int(yy.sum()),'fold_count':5,
      'features':RISK_NAMES,'scaler_mean':scaler.mean_.tolist(),'scaler_scale':scaler.scale_.tolist(),
      'source_labels':'TR40 ONLY','validation_labels_used':False,'test_labels_used':False,
      'no_test_model_inference':True,'risk_topology':'exact unique 2hop excluding first hop and self; binary directed original edges',
      'fivefold_OOF':'5 train-only stratified folds seed 2; fold excluded for its own training rows',
      'script_sha256':sha(Path(__file__)),'cpp_sha256':sha(CPP),'compiled_helper_sha256':sha(SO),
      'risk_feature_storage':'six stand-alone uncompressed memory-mappable NPY matrices instead of T-Finance NPZ',
      'twohop_total_candidate_walks':int(work.sum()),'workers':workers,
      'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in outfiles},
      'tuning_started':False,'test_scored':False,'supervisor_method_approval_recorded':False}
    atomic_json(CACHE/'risk_crossfit_complete.json',manifest)
    print(f'I1_RGTAN_TSOCIAL_EXACT_5FOLD_RISK_CROSSFIT=PASS files={len(outfiles)}',flush=True)
    print('I1_RGTAN_TSOCIAL_STAGE2_COMPLETE=PASS TEST_SCORED=NO TUNING_STARTED=NO',flush=True)


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--mode',choices=['selftest','budget','build'],required=True)
    pa.add_argument('--workers',type=int,default=WORKERS)
    args=pa.parse_args()
    require(1<=args.workers<=8,'INVALID_WORKER_COUNT_1_TO_8')
    if args.mode=='selftest': selftest();return
    torch,dgl,psutil=env_check()
    ids=read_splits()
    ptr,idx,cache_meta=persist_csr(dgl)
    if args.mode=='budget':budget(ids,ptr,idx);return
    # Full feature construction is deliberately never automatically invoked by budget.
    require((REPORT/'stage2_sparse_twohop_budget.json').is_file(),'MISSING_PRIOR_CAPACITY_BUDGET')
    print('I1_RGTAN_TSOCIAL_EXACT_SPARSE_CROSSFIT_BUILD_START=YES',flush=True)
    build(ids,ptr,idx,args.workers)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        print('I1_RGTAN_TSOCIAL_STAGE2_FAIL='+repr(e),flush=True)
        raise
