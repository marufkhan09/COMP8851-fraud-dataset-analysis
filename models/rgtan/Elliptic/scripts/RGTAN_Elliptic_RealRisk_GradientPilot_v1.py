#!/usr/bin/env python3
"""Single RGTAN Elliptic TR40 REAL crossfit-risk forward/backward pilot.

One 4-seed gradient computation but NO optimizer.step(), training trial,
checkpoint, hyperparameter selection, test labels, or test scoring.
Only 5-fold training-only risk features are supplied as RGTAN inputs.
"""
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
import torch
import dgl
import torch.nn.functional as F
from dgl.dataloading import NodeDataLoader, MultiLayerFullNeighborSampler

ROOT=Path('/workspace/comp8851')
SRC=ROOT/'assets/elliptic/source'
ASSET=ROOT/'assets/rgtan/elliptic/RGTAN_Elliptic_TR40_TemporalCrossfit_v1.npz'
MANIFEST=ASSET.with_suffix('.json')
OUT=ROOT/'diagnostics/rgtan/elliptic/RGTAN_Elliptic_RealRisk_GradientPilot_v1.json'
PIN='65e583f03ac5c5983e49cc35179fe7a65908b9c1'
NAMES=('degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    assert socket.gethostname()=='2be00658fac4','Wrong instance'
    assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1' and torch.cuda.is_available()
    assert subprocess.check_output(['git','-C',str(ROOT/'repos/antifraud'),'rev-parse','HEAD'],text=True).strip()==PIN
    if OUT.exists():
        old=json.loads(OUT.read_text())
        assert old.get('status')=='RGTAN_ELLIPTIC_REAL_RISK_GRADIENT_PASS', 'Existing result status mismatch'
        assert old.get('crossfit_sha256')==sha(ASSET), 'Existing pilot asset changed'
        assert old.get('script_sha256')==sha(__file__), 'Existing pilot source changed'
        assert old.get('optimizer_steps')==0 and old.get('test_scored') is False, 'Invalid prior pilot'
        print('RGTAN_Elliptic_EXISTING_REALRISK_PILOT_REUSED=PASS',flush=True)
        return
    m=json.loads(MANIFEST.read_text())
    assert m['status']=='RGTAN_ELLIPTIC_TR40_REAL_CROSSFIT_PASS'
    assert m['artifact_sha256']==sha(ASSET)
    with np.load(ASSET,allow_pickle=False) as z:
        d={key:z[key] for key in z.files}
    train_global=d['train_global_ids'];train_local=d['train_prefix_local_ids']
    train_prefix=d['train_prefix_global_ids'];y=d['train_labels']
    assert len(train_global)==len(train_local)==len(y)==18889 and int(y.sum())==1771
    folds=d['train_fold_of_local'][train_local]
    assert set(folds)==set(range(5))
    # 2 fraud, 2 normal, all held out from risk/context sources of fold 0.
    pos=np.flatnonzero((folds==0)&(y==1))[:2]
    neg=np.flatnonzero((folds==0)&(y==0))[:2]
    assert len(pos)==len(neg)==2
    ix=np.r_[pos,neg]
    chosen=train_local[ix]
    assert np.all(d['train_label_context_0'][chosen]==2)
    assert np.all(y[ix]==[1,1,0,0])
    print('TRAINING_ONLY_OOF_FOLD_ZERO_SEEDS=PASS',chosen.tolist(),flush=True)

    # Exact same directed training-prefix graph construction as earlier preflight.
    t0=time.perf_counter()
    types={0:np.int64,1:np.int16}
    types.update({i:np.float32 for i in range(2,167)})
    table=pd.read_csv(SRC/'elliptic_txs_features.csv',header=None,dtype=types)
    assert table.shape==(203769,167)
    txids=table.iloc[:,0].to_numpy(dtype=np.int64,copy=True)
    times=table.iloc[:,1].to_numpy(dtype=np.int16,copy=True)
    features=table.iloc[:,2:].to_numpy(dtype=np.float32,copy=True)
    del table
    assert np.isfinite(features).all()
    assert np.array_equal(np.flatnonzero(times<=20),train_prefix)
    ed=pd.read_csv(SRC/'elliptic_txs_edgelist.csv',dtype=np.int64)
    assert tuple(ed.columns)==('txId1','txId2') and len(ed)==234355
    idx=pd.Index(txids)
    u=idx.get_indexer(ed['txId1']).astype(np.int64)
    v=idx.get_indexer(ed['txId2']).astype(np.int64)
    del ed
    assert (u>=0).all() and (v>=0).all() and (times[u]==times[v]).all()
    mapping=np.full(203769,-1,dtype=np.int64)
    mapping[train_prefix]=np.arange(len(train_prefix))
    edgeok=(times[u]<=20)&(times[v]<=20)
    g=dgl.graph((torch.from_numpy(mapping[u[edgeok]].copy()),
                  torch.from_numpy(mapping[v[edgeok]].copy())),num_nodes=len(train_prefix))
    g=dgl.add_self_loop(dgl.remove_self_loop(g))
    assert len(g.ndata)==0 and len(g.edata)==0
    print('REALRISK_PILOT_TRAIN_GRAPH_ISOLATED=PASS',int(g.num_nodes()),int(g.num_edges()),flush=True)
    torch.set_num_threads(6)
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    loader=NodeDataLoader(g,torch.as_tensor(chosen.copy()),MultiLayerFullNeighborSampler(2),
           batch_size=4,shuffle=False,drop_last=False,num_workers=0)
    input_nodes,out_nodes,blocks=next(iter(loader))
    assert torch.equal(out_nodes,torch.as_tensor(chosen))
    assert torch.equal(blocks[0].srcdata[dgl.NID],input_nodes)
    print('REALRISK_PILOT_FULL_NEIGHBOR_SAMPLER=PASS', [int(b.num_edges()) for b in blocks],flush=True)
    src_nodes=input_nodes.numpy()
    # Context for fold 0 contains labels only of non-fold-0 training seeds.
    context=d['train_label_context_0'][src_nodes]
    assert (context[np.isin(src_nodes,chosen)]==2).all()
    risk=d['train_risk_view_0'][src_nodes]
    assert risk.shape==(len(src_nodes),6) and np.isfinite(risk).all()
    sys.path.insert(0,str(ROOT/'repos/antifraud'))
    from methods.rgtan.rgtan_model import RGTAN
    dummy_neigh={n:torch.zeros(len(train_prefix),device='cuda:0') for n in NAMES}
    model=RGTAN(in_feats=165,hidden_dim=32,n_layers=2,n_classes=2,heads=[4,4],
       activation=torch.nn.PReLU(),drop=[.2,.1],device='cuda:0',gated=True,
       ref_df=pd.DataFrame(features[train_prefix]),cat_features={},
       neigh_features=dummy_neigh,nei_att_head=3).to('cuda:0')
    model.train()
    X=torch.from_numpy(features[train_prefix[src_nodes]].copy()).to('cuda:0')
    C=torch.from_numpy(context.astype(np.int64)).to('cuda:0')
    risks={n:torch.from_numpy(risk[:,i].copy()).to('cuda:0') for i,n in enumerate(NAMES)}
    labels=torch.from_numpy(y[ix].astype(np.int64)).to('cuda:0')
    blocks=[b.to('cuda:0') for b in blocks]
    torch.cuda.reset_peak_memory_stats()
    logits=model(blocks,X,C,{},risks)
    assert logits.shape==(4,2) and bool(torch.isfinite(logits).all())
    loss=F.cross_entropy(logits,labels)
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.cuda.synchronize()
    peak=round(torch.cuda.max_memory_allocated()/1024**3,5)
    duration=round(time.perf_counter()-t0,2)
    result={
      'status':'RGTAN_ELLIPTIC_REAL_RISK_GRADIENT_PASS',
      'dataset':'Elliptic','model':'RGTAN','ratio':'TR40','seed':2,
      'crossfit_sha256':sha(ASSET),'risk_context_fold':0,
      'train_graph_nodes':int(g.num_nodes()),'train_graph_edges_with_selfloops':int(g.num_edges()),
      'selected_training_nodes_global':train_global[ix].tolist(),
      'fold_zero_heldout_seed_contexts_masked':True,
      'heldout_test_labels_accessed':False,'validation_labels_accessed':False,
      'model_forward_and_gradient_pass':True,'optimizer_steps':0,
      'checkpoint_written':False,'tuning_trials_completed':0,'test_scored':False,
      'cross_entropy_pilot_loss':float(loss.item()),
      'gpu_peak_gib':peak,'elapsed_seconds':duration,'script_sha256':sha(__file__)
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print('RGTAN_ELLIPTIC_REAL_RISK_GRADIENT=PASS',flush=True)
    print('REALRISK_GPU_PEAK_GIB',peak,'SECONDS',duration,flush=True)
    print('TUNING_STARTED=NO OPTIMIZER_STEPS=0 TEST_SCORED=NO',flush=True)

if __name__=='__main__':main()
