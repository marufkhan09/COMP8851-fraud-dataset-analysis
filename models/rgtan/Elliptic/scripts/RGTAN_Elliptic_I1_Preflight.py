#!/usr/bin/env python3
"""Read-only identity/graph-capacity preflight. Does not tune or score held-out test."""
import hashlib, json, os, subprocess, time
from pathlib import Path
import numpy as np
import torch
import dgl
ROOT=Path('/workspace/comp8851')
PIN='65e583f03ac5c5983e49cc35179fe7a65908b9c1'
SEEDS=(2,42,72)
RATIOS=('TR10','TR20','TR30','TR40')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify_file(path, expected):
    assert path.is_file(), 'STOP missing: '+str(path)
    actual=sha(path)
    assert actual==expected,(str(path),actual,expected)
    print('SHA256_PASS',path.name,flush=True)

def check_env():
    assert torch.__version__=='1.12.1+cu113',torch.__version__
    assert dgl.__version__=='0.8.1',dgl.__version__
    assert torch.cuda.is_available(),'No available GPU'
    head=subprocess.check_output(['git','-C',str(ROOT/'repos/antifraud'),'rev-parse','HEAD'],text=True).strip()
    assert head==PIN,head
    print('RGTAN_AUTHOR_ENV_GPU_PIN=PASS',flush=True)

def split_load(path, counts, n):
    with np.load(path,allow_pickle=False) as z:
        print('FROZEN_SPLIT_KEYS',list(z.files),flush=True)
        parts={k:np.asarray(z[k],dtype=np.int64).copy() for k in counts}
    for k,ids in parts.items():
        assert len(ids)==counts[k],(k,len(ids),counts[k])
        assert ids.ndim==1 and np.unique(ids).size==ids.size,(k,'duplicate')
        assert ids.min()>=0 and ids.max()<n,(k,'out of bounds')
    for a,b in zip(RATIOS,RATIOS[1:]):
        assert np.isin(parts[a],parts[b]).all(),(a,b,'not nested')
    for k in ('val','test'):
        assert np.intersect1d(parts['TR40'],parts[k]).size==0,(k,'overlaps train')
    assert np.intersect1d(parts['val'],parts['test']).size==0
    print('FROZEN_SPLIT_NESTING_AND_ISOLATION=PASS',flush=True)
    return parts

def save_report(dataset,report):
    out=ROOT/'diagnostics/rgtan'/dataset/'preflight_v1.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    report['dataset']=dataset
    report['tuning_started']=False
    report['test_scored']=False
    report['author_commit']=PIN
    if out.exists():
        old=json.loads(out.read_text())
        # Preflight runs are repeatable; log a new numbered copy, never overwrite.
        out=out.with_name('preflight_v1_repeat_'+str(int(time.time()))+'.json')
    with open(out,'x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print('REPORT',out,flush=True)

import csv
import pandas as pd
DATA=ROOT/'assets/elliptic/source'
SPLIT=Path('/workspace/uploads/ass2_split_candidates/Elliptic/elliptic_seed2_nested_splits.npz')
HASHES={
 'elliptic_txs_features.csv':'fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0',
 'elliptic_txs_classes.csv':'93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493',
 'elliptic_txs_edgelist.csv':'a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e',
}
SPLIT_SHA='1e7963e9d09935fb33e0df74cfb786cab826d1b95ae10b9592dbc240b48ffdbc'
COUNTS={'TR10':4543,'TR20':9553,'TR30':13670,'TR40':18889,'val':8726,'test':18949}
CUTOFF={'TR10':3,'TR20':7,'TR30':12,'TR40':20}
N=203769

def main():
    assert os.uname().nodename=='2be00658fac4','STOP: wrong I1 machine'
    check_env()
    for name,h in HASHES.items():verify_file(DATA/name,h)
    verify_file(SPLIT,SPLIT_SHA)
    parts=split_load(SPLIT,COUNTS,N)
    with np.load(SPLIT,allow_pickle=False) as z:
        assert int(z['split_seed'])==2
    print('ELLIPTIC_LOADING_FEATURES',flush=True)
    dtypes={0:np.int64,1:np.int16}
    dtypes.update({i:np.float32 for i in range(2,167)})
    table=pd.read_csv(DATA/'elliptic_txs_features.csv',header=None,dtype=dtypes)
    assert table.shape==(N,167),table.shape
    txids=table.iloc[:,0].to_numpy(dtype=np.int64,copy=True)
    times=table.iloc[:,1].to_numpy(dtype=np.int16,copy=True)
    features=table.iloc[:,2:].to_numpy(dtype=np.float32,copy=True)
    del table
    assert np.unique(txids).size==N
    assert int(times.min())==1 and int(times.max())==49
    assert np.isfinite(features).all()
    for r, cutoff in CUTOFF.items():assert np.all(times[parts[r]]<=cutoff),(r,'train future contamination')
    assert np.all((times[parts['val']]>=21)&(times[parts['val']]<=31))
    assert np.all(times[parts['test']]>=32)
    print('ELLIPTIC_CHRONOLOGICAL_BOUNDARIES=PASS',flush=True)
    # Only collect the labels of the TR40 training partition and validation.
    allowed=set(txids[np.r_[parts['TR40'],parts['val']]].tolist())
    records={}
    with open(DATA/'elliptic_txs_classes.csv',newline='') as f:
        reader=csv.DictReader(f)
        assert set(reader.fieldnames)=={'txId','class'},reader.fieldnames
        for row in reader:
            node=int(row['txId'])
            if node in allowed:
                assert node not in records
                assert row['class'] in ('1','2')
                records[node]=int(row['class']=='1')
            # Deliberately ignore all held-out test/unknown class values.
    assert len(records)==len(allowed),(len(records),len(allowed))
    ytrain=np.asarray([records[int(v)] for v in txids[parts['TR40']]],dtype=np.int8)
    yval=np.asarray([records[int(v)] for v in txids[parts['val']]],dtype=np.int8)
    assert int(ytrain.sum())==1771 and int(yval.sum())==1289,(ytrain.sum(),yval.sum())
    print('TRAIN_VALIDATION_ONLY_LABELS=PASS; TEST_LABELS_NOT_COLLECTED',flush=True)
    edges=pd.read_csv(DATA/'elliptic_txs_edgelist.csv',dtype=np.int64)
    assert tuple(edges.columns)==('txId1','txId2') and len(edges)==234355
    idx=pd.Index(txids)
    src=idx.get_indexer(edges['txId1']).astype(np.int64)
    dst=idx.get_indexer(edges['txId2']).astype(np.int64)
    assert np.all(src>=0) and np.all(dst>=0)
    assert np.all(times[src]==times[dst])
    del edges
    # Candidate directed training topology: only times <=20, no future nodes/edges.
    valid=np.flatnonzero(times<=20)
    mapping=np.full(N,-1,dtype=np.int64)
    mapping[valid]=np.arange(len(valid))
    edge_mask=(times[src]<=20)&(times[dst]<=20)
    u=mapping[src[edge_mask]];v=mapping[dst[edge_mask]]
    assert u.min()>=0 and v.min()>=0
    graph=dgl.graph((torch.from_numpy(u),torch.from_numpy(v)),num_nodes=len(valid))
    # A self-loop is needed for any zero-in-degree destination under original RGTAN attention.
    # This is a diagnostic candidate, not frozen production graph selection.
    graph=dgl.remove_self_loop(graph)
    graph=dgl.add_self_loop(graph)
    assert len(graph.ndata)==0 and len(graph.edata)==0
    train_local=mapping[parts['TR40']]
    assert np.all(train_local>=0)
    assert not np.isin(parts['val'],valid).any() and not np.isin(parts['test'],valid).any()
    print('ELLIPTIC_TRAIN_PREFIX_GRAPH_ONLY=PASS',graph.num_nodes(),graph.num_edges(),flush=True)
    # Architecture-preserving bounded forward/backward; risk contexts all 2 (unknown), risk features zero.
    import sys
    sys.path.insert(0,str(ROOT/'repos/antifraud'))
    from methods.rgtan.rgtan_model import RGTAN
    from dgl.dataloading import NodeDataLoader,MultiLayerFullNeighborSampler
    ids=train_local[:4].copy()
    sampler=MultiLayerFullNeighborSampler(2)
    loader=NodeDataLoader(graph,torch.from_numpy(ids),sampler,batch_size=4,shuffle=False,drop_last=False,num_workers=0)
    start=time.perf_counter()
    input_ids,outputs,blocks=next(iter(loader))
    assert len(blocks)==2 and len(outputs)==4
    assert torch.equal(blocks[0].srcdata[dgl.NID],input_ids)
    print('ELLIPTIC_FULL_NEIGHBOR_SAMPLE=PASS', [int(b.num_edges()) for b in blocks],flush=True)
    # Check model initialization at 165 features and a compatible attention-head divisor (3).
    names=['degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat']
    zeros={name:torch.zeros(N,device='cuda') for name in names}
    model=RGTAN(in_feats=165,hidden_dim=32,n_layers=2,n_classes=2,heads=[4,4],
       activation=torch.nn.PReLU(),drop=[.2,.1],device='cuda:0',gated=True,
       ref_df=pd.DataFrame(features),cat_features={},neigh_features=zeros,
       nei_att_head=3).to('cuda')
    xx=torch.from_numpy(features[valid[input_ids.numpy()]]).to('cuda')
    cc=torch.full((len(input_ids),),2,dtype=torch.int64,device='cuda')
    risk={name:torch.zeros(len(input_ids),device='cuda') for name in names}
    gpu_blocks=[b.to('cuda') for b in blocks]
    torch.cuda.reset_peak_memory_stats()
    out=model(gpu_blocks,xx,cc,{},risk)
    assert out.shape==(4,2) and torch.isfinite(out).all()
    # No optimizer step; confirms the graph forward path only.
    torch.cuda.synchronize()
    peak=round(torch.cuda.max_memory_allocated()/1024**3,4)
    sec=round(time.perf_counter()-start,3)
    print('ELLIPTIC_ONE_BATCH_FORWARD=PASS seconds='+str(sec)+' GPU_PEAK_GIB='+str(peak),flush=True)
    save_report('elliptic',{'model':'RGTAN','ratio':'TR40','train_count':len(parts['TR40']),
      'validation_count':len(parts['val']),'test_count':len(parts['test']),
      'train_positive':int(ytrain.sum()),'val_positive':int(yval.sum()),
      'train_cutoff':20,'val_cutoff':31,'test_start':32,
      'train_graph_nodes':int(graph.num_nodes()),'train_graph_edges_with_self_loops':int(graph.num_edges()),
      'one_batch_forward_gpu_peak_gib':peak,'seconds_sample_and_forward':sec,
      'original_directed_topology':True,'diagnostic_self_loops_added':True,
      'real_train_crossfit_built':False,'graph_adapter_finalized':False,
      'test_labels_collected':False,'no_training_run':True})
    print('I1_ELLIPTIC_RGTAN_PREFLIGHT=PASS',flush=True)
if __name__=='__main__':main()
