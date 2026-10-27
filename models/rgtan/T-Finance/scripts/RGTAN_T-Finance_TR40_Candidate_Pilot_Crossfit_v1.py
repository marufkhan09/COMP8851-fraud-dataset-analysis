#!/usr/bin/env python3
"""RGTAN T-Finance: capped-neighbour *candidate* pilot and exact-risk crossfit.

Unchanged canonical 42,445,086 directed edge graph used for model/signal.
Sampling ONLY for message passing is explicitly different from author's
unbounded full-neighbour sampler: fanouts [10, 10]. This requires reporting
and supervisor acceptance to use in a benchmark. Crossfit is exact-original-
graph 2hop reachability; validation/test labels never feed model risk/context.
Pilot uses zero risk features solely to test architecture and memory; no update.
NO hyperparameter tuning, final runs, or test scoring.
"""
import argparse
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT=Path('/workspace/comp8851')
DATA=ROOT/'assets/tfinance/canonical_candidate/tfinance_canonical.dgl'
SPLIT=Path('/workspace/uploads/ass2_split_candidates/T-Finance/tfinance_seed2_nested_splits.npz')
PIN='65e583f03ac5c5983e49cc35179fe7a65908b9c1'
DATA_SHA='b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700'
SPLIT_SHA='de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1'
OUT=ROOT/'assets/rgtan/tfinance/RGTAN_T-Finance_TR40_ExactRiskCrossfit_v1.npz'
MAN=OUT.with_suffix('.json')
PILOT=ROOT/'diagnostics/rgtan/tfinance/RGTAN_T-Finance_TR40_Fanout10x10_Pilot_v1.json'
N=39357
NF=10
EDGES=42445086
FANOUTS=[10,10]
NAMES=('degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat')


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 return h.hexdigest()


def jwrite(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=Path(str(path)+'.partial')
 with tmp.open('w') as f:
  json.dump(data,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)


def verify():
 assert socket.gethostname()=='f24d1bb2672d','STOP: This is not T-Finance I2'
 assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1' and torch.cuda.is_available()
 assert subprocess.check_output(['git','-C',str(ROOT/'repos/antifraud'),'rev-parse','HEAD'],text=True).strip()==PIN
 assert sha(DATA)==DATA_SHA and sha(SPLIT)==SPLIT_SHA
 prev=json.loads((ROOT/'diagnostics/rgtan/tfinance/RGTAN_T-Finance_TwoHop_ExpansionBudget_v1.json').read_text())
 assert prev['status']=='RGTAN_TFINANCE_TWOHOP_EXPANSION_BUDGET_PASS'
 assert prev['two_hop_sampler_executed'] is False and prev['test_scored'] is False
 with np.load(SPLIT,allow_pickle=False) as z:
  assert int(z['seed'].item())==2
  assert int(z['source_normal'].item())==37553 and int(z['source_fraud'].item())==1804
  parts={key:np.asarray(z[key],dtype=np.int64).copy() for key in ('TR40','TR30','TR20','TR10','val')}
 assert len(parts['TR40'])==15742 and len(parts['val'])==7872
 for a,b in (('TR10','TR20'),('TR20','TR30'),('TR30','TR40')):
  assert np.isin(parts[a],parts[b]).all()
 assert not np.intersect1d(parts['TR40'],parts['val']).size
 print('RGTAN_T-Finance_CANONICAL_SHA_AND_FROZEN_TR40_VALIDATION_ISOLATION=PASS',flush=True)
 return parts


def load(parts):
 graph_list,_=dgl.load_graphs(str(DATA))
 assert len(graph_list)==1
 g=graph_list[0]
 assert g.is_homogeneous and g.num_nodes()==N and g.num_edges()==EDGES
 assert set(g.ndata)>={'feature','label'}
 x=g.ndata['feature'].to(torch.float32).numpy().copy()
 raw=g.ndata['label']
 assert raw.shape==(N,2) and bool(((raw==0)|(raw==1)).all()) and bool((raw.sum(dim=1)==1).all())
 yy=raw.argmax(dim=1)
 assert int(yy.sum())==1804 and int(yy[27028])==1
 tr=parts['TR40']
 y=yy[torch.from_numpy(tr.copy())].numpy().astype(np.int64,copy=True)
 assert int(y.sum())==721
 # Do not retain original labels, including validation/test labels, anywhere in model graph.
 del yy,raw
 for k in list(g.ndata.keys()):del g.ndata[k]
 for k in list(g.edata.keys()):del g.edata[k]
 assert not g.ndata and not g.edata
 assert x.shape==(N,NF) and np.isfinite(x).all()
 print('RGTAN_T-Finance_ORIGINAL_42M_EDGE_GRAPH_TR40_LABEL_ONLY=PASS',flush=True)
 return g,x,y


def candidate_pilot(g,x,y,parts):
 if PILOT.exists():
  d=json.loads(PILOT.read_text())
  assert d['status']=='RGTAN_TFINANCE_CAPPED_FANOUT_PILOT_PASS' and d['fanouts']==FANOUTS
  assert d['script_sha256']==sha(__file__) and d['test_scored'] is False
  print('RGTAN_T-Finance_CAPPED_FANOUT_PILOT_REUSED=PASS',flush=True)
  return
 sys.path.insert(0,str(ROOT/'repos/antifraud'))
 from methods.rgtan.rgtan_model import RGTAN
 tr=parts['TR40']
 rng=np.random.RandomState(2)
 pos=tr[y==1];neg=tr[y==0]
 assert len(pos)>64 and len(neg)>1000
 ids=np.concatenate([rng.choice(pos,16,replace=False),rng.choice(neg,1008,replace=False)])
 rng.shuffle(ids)
 torch.manual_seed(2);torch.cuda.manual_seed_all(2)
 torch.set_num_threads(6)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 model=RGTAN(in_feats=NF,hidden_dim=32,n_layers=2,n_classes=2,heads=[4,4],
   activation=torch.nn.PReLU(),drop=[.2,.1],device='cuda:0',gated=True,
   ref_df=pd.DataFrame(x),cat_features={},
   neigh_features={n:torch.zeros(N,device='cuda:0') for n in NAMES},
   nei_att_head=2).to('cuda:0')
 model.eval()
 records=[]
 proc=psutil.Process(os.getpid())
 for batch in (32,256,1024):
  src_seeds=torch.from_numpy(ids[:batch].copy())
  loader=NodeDataLoader(g,src_seeds,MultiLayerNeighborSampler(FANOUTS),
    batch_size=batch,shuffle=False,drop_last=False,num_workers=0)
  tic=time.perf_counter()
  inbound,seeds,blocks=next(iter(loader))
  assert torch.equal(seeds,src_seeds)
  assert len(blocks)==2
  assert torch.equal(blocks[0].srcdata[dgl.NID],inbound)
  assert all(int(b.num_edges())<=batch*FANOUTS[0]*FANOUTS[1]+batch*FANOUTS[1] for b in blocks)
  with torch.no_grad():
   xx=torch.from_numpy(x[inbound.numpy()].copy()).to('cuda:0')
   context=torch.full((len(inbound),),2,device='cuda:0',dtype=torch.long)
   risks={n:torch.zeros(len(inbound),device='cuda:0') for n in NAMES}
   torch.cuda.reset_peak_memory_stats()
   logits=model([b.to('cuda:0') for b in blocks],xx,context,{},risks)
   assert logits.shape==(batch,2) and bool(torch.isfinite(logits).all())
   torch.cuda.synchronize()
  rec={'batch_size':batch,'block_edges':[int(b.num_edges()) for b in blocks],
    'sampled_input_nodes':len(inbound),'gpu_peak_gib':round(torch.cuda.max_memory_allocated()/1024**3,4),
    'process_rss_gib':round(proc.memory_info().rss/1024**3,3),
    'elapsed_seconds':round(time.perf_counter()-tic,3)}
  records.append(rec)
  print('RGTAN_T-Finance_CAPPED_10x10_ARCHITECTURE_PILOT',json.dumps(rec,sort_keys=True),flush=True)
 meta={'status':'RGTAN_TFINANCE_CAPPED_FANOUT_PILOT_PASS',
    'model':'RGTAN','dataset':'T-Finance','source_sha256':DATA_SHA,'split_sha256':SPLIT_SHA,
    'author_commit':PIN,'fanouts':FANOUTS,'sampler':'MultiLayerNeighborSampler',
    'deviation_from_author_full_neighbor_sampling':True,
    'benchmark_supervisor_approval_required':True,
    'risk_features':'placeholder zeros in architecture-only pilot, NOT evaluated',
    'real_risk_crossfit_required_before_tuning':True,
    'optimizer_steps':0,'test_scored':False,'validation_labels_used':False,
    'records':records,'script_sha256':sha(__file__)}
 jwrite(PILOT,meta)
 print('RGTAN_T-Finance_TR40_BOUNDED_FANOUT_PILOT=PASS',flush=True)


def exact_risk(a,views):
 """Binary EXACT directed 2-hop reachability excluding first hop and self.

 Matrix product in FP16 is exact for binary incidence because maximum shared
 intermediate nodes <=N(39357)<FP16 integer exactness? FP16 integers only
 exact through 2048! Therefore use float32 GPU matmul to ensure exact counts.
 Under 39k intermediate nodes, sums <=39357 and float32 integers exact.
 """
 # Use float32 dense A to avoid FP16 count rounding; 6.2GB device memory.
 # GPU memory 48GB, tile on host RAM 55GB; 1.55GB host adjacency allocation.
 tic=time.perf_counter()
 torch.backends.cuda.matmul.allow_tf32=False
 free,total=torch.cuda.mem_get_info()
 assert free>=14*1024**3, f'STOP: Free GPU VRAM {free/1024**3:.1f} GiB < 14 GiB minimum for exact-risk preflight'
 h=torch.from_numpy(a.toarray()).to(device='cuda:0',dtype=torch.float32)
 degree=np.diff(a.indptr).astype(np.float32)
 one_risk=np.asarray(a@views,dtype=np.float32)
 first_degree=np.asarray(a@degree,dtype=np.float32)
 first_risk=np.asarray(a@one_risk,dtype=np.float32)
 dg=torch.from_numpy(degree).to('cuda:0')
 rg=torch.from_numpy(one_risk).to('cuda:0')
 second_degree=np.empty(N,dtype=np.float32)
 second_risk=np.empty((N,views.shape[1]),dtype=np.float32)
 tile=128
 for start in range(0,N,tile):
  end=min(N,start+tile)
  reach=(h[start:end]@h)>0
  reach &= (h[start:end]==0)
  ii=torch.arange(end-start,device='cuda:0')
  reach[ii,torch.arange(start,end,device='cuda:0')]=False
  w=reach.float()
  second_degree[start:end]=(w@dg).cpu().numpy()
  second_risk[start:end]=(w@rg).cpu().numpy()
  del reach,w
  if start==0 or end==N or (start//tile+1)%50==0:
   print(f'RGTAN_T-Finance_TWOHOP_RISK_ROWS={end}/{N} elapsed={time.perf_counter()-tic:.1f}s',flush=True)
 torch.cuda.synchronize()
 del h,dg,rg
 torch.cuda.empty_cache()
 assert np.isfinite(second_degree).all() and np.isfinite(second_risk).all()
 print('RGTAN_T-Finance_EXACT_TWOHOP_RISK=PASS elapsed_s',round(time.perf_counter()-tic,2),flush=True)
 return degree,one_risk,first_degree,second_degree,first_risk,second_risk


def crossfit(g,x,train_y,parts):
 if OUT.exists() or MAN.exists():
  assert OUT.is_file() and MAN.is_file(),'STOP: partial crossfit artifact; inspect'
  meta=json.loads(MAN.read_text())
  assert meta['status']=='RGTAN_TFINANCE_TR40_EXACT_RISK_CROSSFIT_PASS'
  assert meta['artifact_sha256']==sha(OUT) and meta['script_sha256']==sha(__file__)
  print('RGTAN_T-Finance_TR40_CROSSFIT_REUSED=PASS',flush=True)
  return
 tr=parts['TR40'];val=parts['val']
 folds=np.full(N,-1,dtype=np.int8)
 skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=2)
 for k,(_,held) in enumerate(skf.split(np.zeros(len(tr)),train_y)):
  folds[tr[held]]=k
 assert set(folds[tr])==set(range(5))
 ctx=np.full((6,N),2,dtype=np.int8)
 pos=np.zeros((N,6),dtype=np.float32)
 for k in range(5):
  fit=(folds[tr]!=k);nodes=tr[fit]
  ctx[k,nodes]=train_y[fit];pos[nodes,k]=train_y[fit]
  assert np.all(ctx[k,tr[~fit]]==2) and np.all(pos[tr[~fit],k]==0)
 ctx[5,tr]=train_y;pos[tr,5]=train_y
 other=np.setdiff1d(np.arange(N),tr)
 assert np.all(ctx[:,other]==2) and np.all(pos[other]==0)
 assert np.all(ctx[:,val]==2) and np.all(pos[val]==0)
 print('RGTAN_T-Finance_TR40_5FOLD_TRAIN_ONLY_CONTEXT=PASS',flush=True)
 t0=time.perf_counter()
 # Graph adjacency orientation: row=destination, column=inbound source.
 # COO takes original directed edges; binary deduplicate for *risk view only*.
 u,v=g.edges(order='eid')
 adj=sp.coo_matrix((np.ones(len(u),dtype=np.int8),(v.numpy(),u.numpy())),shape=(N,N)).tocsr()
 del u,v
 adj.sum_duplicates();adj.data.fill(1);adj.setdiag(0);adj.eliminate_zeros();adj.sort_indices()
 assert adj.nnz>21000000 and adj.nnz<=EDGES
 print('RGTAN_T-Finance_EXACT_RISK_BINARY_ADJACENCY_EDGES',adj.nnz,flush=True)
 d,r,fd,sd,fr,sr=exact_risk(adj,pos)
 raw_views=[np.column_stack((d,r[:,k],fd,sd,fr[:,k],sr[:,k])).astype(np.float32) for k in range(6)]
 oof=np.empty((len(tr),6),dtype=np.float32)
 for k in range(5):
  mask=folds[tr]==k;oof[mask]=raw_views[k][tr[mask]]
 scaler=StandardScaler().fit(oof)
 views=[scaler.transform(z).astype(np.float32) for z in raw_views]
 assert all(np.isfinite(z).all() for z in views)
 payload={'train_ids':tr,'train_labels':train_y,'val_ids':val,
   'fold_of_node':folds,'features':x,
   'risk_feature_names':np.asarray(NAMES),
   'scaler_mean':scaler.mean_,'scaler_scale':scaler.scale_}
 for k in range(6):
  payload[f'risk_view_{k}']=views[k]
  payload[f'label_context_{k}']=ctx[k]
 OUT.parent.mkdir(parents=True,exist_ok=True)
 tmp=OUT.with_suffix('.npz.partial')
 assert not tmp.exists(),'STOP: stale partial NPZ; inspect'
 with tmp.open('xb') as f:
  np.savez_compressed(f,**payload);f.flush();os.fsync(f.fileno())
 os.replace(tmp,OUT)
 meta={'status':'RGTAN_TFINANCE_TR40_EXACT_RISK_CROSSFIT_PASS',
  'model':'RGTAN','dataset':'T-Finance','ratio':'TR40',
  'original_source_graph_edges':EDGES,'risk_binary_graph_edges':int(adj.nnz),
  'risk_topology':'binary direct edges excluding self; exact 2hop excluding 1hop and self',
  'train_count':len(tr),'train_fraud':int(train_y.sum()),'val_count':len(val),
  'risk_sources':'training labelled nodes only, 5 fold held-out train contexts',
  'scaler':'OOF train only','test_labels_used':False,'validation_labels_used':False,
  'author_commit':PIN,'source_sha256':DATA_SHA,'split_sha256':SPLIT_SHA,
  'artifact_sha256':sha(OUT),'script_sha256':sha(__file__),
  'elapsed_seconds':round(time.perf_counter()-t0,2),'training_started':False,'test_scored':False,
  'sampling_change_proposed':[10,10], 'benchmark_supervisor_approval_required':True}
 jwrite(MAN,meta)
 print('RGTAN_T-Finance_TR40_EXACT_5FOLD_CROSSFIT=PASS',flush=True)
 print('RGTAN_T-Finance_TR40_CROSSFIT_ARTIFACT_SHA256',meta['artifact_sha256'],flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--pilot-only',action='store_true')
 args=p.parse_args()
 parts=verify()
 g,x,y=load(parts)
 candidate_pilot(g,x,y,parts)
 if args.pilot_only:
  print('RGTAN_T-Finance_CAPPED_SAMPLER_PILOT_ONLY=PASS; CROSSFIT_NOT_RUN',flush=True)
 else:
  crossfit(g,x,y,parts)
  print('RGTAN_T-Finance_TR40_RISK_CROSSFIT_AND_CAPPED_SAMPLER_PREFLIGHT=PASS',flush=True)
 print('RGTAN_T-Finance_TUNING_STARTED=NO TEST_SCORED=NO',flush=True)

if __name__=='__main__':main()
