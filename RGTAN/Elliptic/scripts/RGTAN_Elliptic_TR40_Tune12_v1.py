#!/usr/bin/env python3
"""RGTAN Elliptic TR40 12 validation-only trials; chronological leakage-safe.

Uses frozen SHA-pinned TR40 5-fold OOF risk/context artifact from prior stage.
One author RGTAN model per trial. Each training seed uses its withheld-fold risk
view. Validation forward uses full training-label context, prefix <= timestep31.
The source CSV is parsed to select chronological prefixes; post-cutoff nodes/edges are never passed to a model, and no test labels are collected or scored.
"""
import argparse
import csv
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
import torch
import torch.nn.functional as F
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

ROOT = Path('/workspace/comp8851')
SRC = ROOT/'assets/elliptic/source'
ASSET = ROOT/'assets/rgtan/elliptic/RGTAN_Elliptic_TR40_TemporalCrossfit_v1.npz'
MANIFEST = ASSET.with_suffix('.json')
SPLIT = Path('/workspace/uploads/ass2_split_candidates/Elliptic/elliptic_seed2_nested_splits.npz')
OUT = ROOT/'results/rgtan/elliptic/RGTAN_Elliptic_TR40_Tune12_v1'
PIN = '65e583f03ac5c5983e49cc35179fe7a65908b9c1'
SPLIT_SHA='1e7963e9d09935fb33e0df74cfb786cab826d1b95ae10b9592dbc240b48ffdbc'
DATA_HASHES={
 'elliptic_txs_features.csv':'fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0',
 'elliptic_txs_classes.csv':'93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493',
 'elliptic_txs_edgelist.csv':'a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e'}
NAMES=('degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat')
GRID=[dict(hidden_dim=h,drop=[d,d/2],base_lr=lr)
      for h in (32,64) for d in (.2,.3) for lr in (.001,.003,.005)]
assert len(GRID)==12
BATCH=1024
MAX_EPOCHS=100
PATIENCE=20
DEVICE='cuda:0'


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()


def atomic_json(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=Path(str(p)+'.partial')
 with open(tmp,'w') as f:
  json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)


def atomic_torch(p,obj):
 tmp=Path(str(p)+'.partial');torch.save(obj,str(tmp));os.replace(tmp,p)


def check_and_read():
 assert socket.gethostname()=='2be00658fac4','STOP: wrong instance'
 assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1'
 assert torch.cuda.is_available() and 'A6000' in torch.cuda.get_device_name(0)
 assert subprocess.check_output(['git','-C',str(ROOT/'repos/antifraud'),'rev-parse','HEAD'],text=True).strip()==PIN
 for name,expected in DATA_HASHES.items():assert sha(SRC/name)==expected,(name,'SHA mismatch')
 assert sha(SPLIT)==SPLIT_SHA
 m=json.loads(MANIFEST.read_text())
 assert m['status']=='RGTAN_ELLIPTIC_TR40_REAL_CROSSFIT_PASS'
 assert m['artifact_sha256']==sha(ASSET)
 with np.load(ASSET,allow_pickle=False) as z:
  d={k:z[k] for k in z.files}
 assert tuple(str(x) for x in d['risk_feature_names'])==NAMES
 assert len(d['train_global_ids'])==18889 and len(d['val_global_ids'])==8726
 assert int(d['train_labels'].sum())==1771
 with np.load(SPLIT,allow_pickle=False) as z:
  assert np.array_equal(z['TR40'],d['train_global_ids'])
  assert np.array_equal(z['val'],d['val_global_ids'])
  # Frozen test partition is neither materialized nor scored during tuning.
 for k in range(5):
  fold=(d['train_fold_of_local'][d['train_prefix_local_ids']]==k)
  assert fold.any()
  assert np.all(d[f'train_label_context_{k}'][d['train_prefix_local_ids'][fold]]==2)
  assert np.isfinite(d[f'train_risk_view_{k}']).all()
 assert np.all(d['val_context'][d['val_prefix_local_ids']]==2)
 assert np.isfinite(d['val_risk_view']).all()
 print('RGTAN_Elliptic_TR40_FROZEN_SOURCE_SPLIT_OOF_ISOLATION=PASS',flush=True)
 return d,m


def load_graphs(d):
 dtype={0:np.int64,1:np.int16}
 dtype.update({i:np.float32 for i in range(2,167)})
 t=pd.read_csv(SRC/'elliptic_txs_features.csv',header=None,dtype=dtype)
 assert t.shape==(203769,167)
 txid=t.iloc[:,0].to_numpy(dtype=np.int64,copy=True)
 times=t.iloc[:,1].to_numpy(dtype=np.int16,copy=True)
 x=t.iloc[:,2:].to_numpy(dtype=np.float32,copy=True)
 del t
 assert np.isfinite(x).all()
 tr=d['train_prefix_global_ids'];vl=d['val_prefix_global_ids']
 assert np.array_equal(np.flatnonzero(times<=20),tr)
 assert np.array_equal(np.flatnonzero(times<=31),vl)
 assert (times[d['train_global_ids']]<=20).all()
 assert ((times[d['val_global_ids']]>=21)&(times[d['val_global_ids']]<=31)).all()
 ed=pd.read_csv(SRC/'elliptic_txs_edgelist.csv',dtype=np.int64)
 assert len(ed)==234355
 ix=pd.Index(txid)
 u=ix.get_indexer(ed['txId1']).astype(np.int64)
 v=ix.get_indexer(ed['txId2']).astype(np.int64)
 del ed
 assert (u>=0).all() and (v>=0).all() and (times[u]==times[v]).all()
 graphs=[]
 for cutoff,prefix in [(20,tr),(31,vl)]:
  m=np.full(203769,-1,dtype=np.int64);m[prefix]=np.arange(len(prefix))
  mask=(times[u]<=cutoff)&(times[v]<=cutoff)
  # Model retains directed edges and attaches exactly one self-loop/node.
  g=dgl.graph((torch.from_numpy(m[u[mask]].copy()),torch.from_numpy(m[v[mask]].copy())),num_nodes=len(prefix))
  g=dgl.add_self_loop(dgl.remove_self_loop(g))
  assert len(g.ndata)==0 and len(g.edata)==0
  graphs.append(g)
 gtr,gvl=graphs
 assert (gtr.num_nodes(),gtr.num_edges())==(90762,197569)
 assert (gvl.num_nodes(),gvl.num_edges())==(126103,271936)
 # 145833 non-self + 126103 self-loops = 271936
 assert np.isin(d['train_global_ids'],tr).all()
 assert np.isin(d['val_global_ids'],vl).all()
 # Validation targets are loaded by transaction ID only, never raw test labels.
 allowed=set(map(int,txid[d['val_global_ids']]))
 labels={}
 with (SRC/'elliptic_txs_classes.csv').open(newline='') as f:
  reader=csv.DictReader(f)
  for row in reader:
   tid=int(row['txId'])
   if tid in allowed:
    assert tid not in labels and row['class'] in ('1','2')
    labels[tid]=int(row['class']=='1')
 assert len(labels)==8726
 val_y=np.array([labels[int(tid)] for tid in txid[d['val_global_ids']]],dtype=np.int64)
 assert int(val_y.sum())==1289
 print('RGTAN_Elliptic_TR40_CHRONOLOGICAL_GRAPHS_AND_VALIDATION_ONLY_TARGETS=PASS',flush=True)
 return gtr,gvl,x[tr],x[vl],val_y


class Engine:
 def __init__(self, d, graphs, config, seed):
  from methods.rgtan.rgtan_model import RGTAN
  self.d=d
  self.gtr,self.gvl,self.xtr,self.xvl,self.val_y=graphs
  self.config=config
  self.seed=seed
  self.train_global=d['train_global_ids'];self.train_local=d['train_prefix_local_ids']
  self.val_local=d['val_prefix_local_ids']
  self.y=d['train_labels'].astype(np.int64)
  self.folds=d['train_fold_of_local'][self.train_local]
  self.fids=[]
  self.labs=[]
  for k in range(5):
   ids=self.train_local[self.folds==k]
   self.fids.append(torch.from_numpy(ids.copy()))
   self.labs.append(dict(zip(map(int,ids),map(int,self.y[self.folds==k]))))
  self.val_lab=dict(zip(map(int,self.val_local),map(int,self.val_y)))
  for f in self.fids:assert len(f)>1
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  np.random.seed(seed);random.seed(seed)
  torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
  torch.set_num_threads(6)
  zero={n:torch.zeros(self.gtr.num_nodes(),device=DEVICE) for n in NAMES}
  self.model=RGTAN(in_feats=165,hidden_dim=config['hidden_dim'],n_layers=2,n_classes=2,
   heads=[4,4],activation=torch.nn.PReLU(),drop=config['drop'],device=DEVICE,
   gated=True,ref_df=pd.DataFrame(self.xtr),cat_features={},
   neigh_features=zero,nei_att_head=3).to(DEVICE)
  self.lr=config['base_lr']*(BATCH/1024)**.5
  self.opt=torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=.0001)
  self.tr_loaders=[NodeDataLoader(self.gtr, ids, MultiLayerFullNeighborSampler(2),
     batch_size=BATCH,shuffle=True,drop_last=False,num_workers=0) for ids in self.fids]
  self.val_loader=NodeDataLoader(self.gvl,torch.from_numpy(self.val_local.copy()),
    MultiLayerFullNeighborSampler(2),batch_size=BATCH,shuffle=False,drop_last=False,num_workers=0)
  self.steps=0

 def forward(self,batch,stage,fold=5):
  inn,outs,blocks=batch
  a=inn.numpy();b=outs.numpy()
  if stage=='train':
   feats=self.xtr[a]
   ctx=self.d[f'train_label_context_{fold}'][a]
   risk=self.d[f'train_risk_view_{fold}'][a]
   # Every loss target is from one heldout fold; labels in their contexts masked.
   assert np.all(ctx[np.isin(a,b)]==2)
  else:
   feats=self.xvl[a]
   ctx=self.d['val_context'][a]
   risk=self.d['val_risk_view'][a]
   assert np.all(ctx[np.isin(a,b)]==2)
  x=torch.from_numpy(feats.copy()).to(DEVICE)
  c=torch.from_numpy(ctx.astype(np.int64)).to(DEVICE)
  rd={n:torch.from_numpy(risk[:,j].copy()).to(DEVICE) for j,n in enumerate(NAMES)}
  logits=self.model([blk.to(DEVICE) for blk in blocks],x,c,{},rd)
  assert logits.shape==(len(b),2) and bool(torch.isfinite(logits).all())
  return logits,b

 def train_epoch(self,ep):
  self.model.train();loss_sum=0;seen=0
  order=np.random.RandomState(self.seed+ep).permutation(5)
  for fold in order:
   for batch in self.tr_loaders[fold]:
    logits,b=self.forward(batch,'train',fold)
    targets=torch.from_numpy(np.asarray([self.labs[fold][int(i)] for i in b],dtype=np.int64)).to(DEVICE)
    loss=F.cross_entropy(logits,targets)
    assert bool(torch.isfinite(loss))
    self.opt.zero_grad(set_to_none=True)
    loss.backward()
    self.opt.step()
    self.steps+=1
    loss_sum+=float(loss.item())*len(b);seen+=len(b)
  assert seen==18889,seen
  return loss_sum/seen

 def evaluate(self):
  self.model.eval();scores=np.empty(len(self.val_local),dtype=np.float32)
  pos={int(i):j for j,i in enumerate(self.val_local)}
  with torch.no_grad():
   for batch in self.val_loader:
    logits,b=self.forward(batch,'validation')
    p=torch.softmax(logits,dim=1)[:,1].cpu().numpy()
    for ix,node in enumerate(b):scores[pos[int(node)]]=p[ix]
  assert np.isfinite(scores).all()
  return float(average_precision_score(self.val_y,scores)), scores


def run_trial(d,graphs,j,config,args,asset_sha):
 out=OUT/f'trial_{j:02d}';out.mkdir(parents=True,exist_ok=True)
 done=out/'RGTAN_Elliptic_TR40_TrialResult.json'
 if done.exists():
  row=json.loads(done.read_text())
  assert row['status']=='COMPLETE' and row['asset_sha256']==asset_sha and row['config']==config and row['script_sha256']==sha(__file__)
  assert (out/'RGTAN_Elliptic_TR40_BestModel.pt').exists()
  print(f'RGTAN_Elliptic_TR40_TRIAL_{j:02d}_REUSED=PASS val_AUPRC={row["best_val_auprc"]:.6f}',flush=True)
  return row
 eng=Engine(d,graphs,config,2)
 start=time.perf_counter();best=-1.;best_epoch=0;stale=0;history=[]
 ckpt=out/'RGTAN_Elliptic_TR40_BestModel.pt'
 for epoch in range(1,MAX_EPOCHS+1):
  loss=eng.train_epoch(epoch)
  ap,scores=eng.evaluate()
  history.append({'epoch':epoch,'train_loss':loss,'val_auprc':ap})
  if ap>best+1e-9:
   best=ap;best_epoch=epoch;stale=0
   atomic_torch(ckpt,{'model':eng.model.state_dict(),'config':config,'seed':2,
     'epoch':epoch,'val_auprc':ap,'asset_sha256':asset_sha,'author_pin':PIN,
     'dataset':'Elliptic','architecture':'RGTAN','test_scored':False})
   # Val scores are safe for downstream threshold selection.
   temp=out/'RGTAN_Elliptic_TR40_BestValidationScores.npy.partial'
   with open(temp,'wb') as f:np.save(f,scores);f.flush();os.fsync(f.fileno())
   os.replace(temp,out/'RGTAN_Elliptic_TR40_BestValidationScores.npy')
  else:stale+=1
  if epoch<=2 or epoch%5==0 or stale>=PATIENCE:
   print(f'RGTAN_Elliptic_TR40 trial={j}/12 epoch={epoch} loss={loss:.5f} val_AUPRC={ap:.6f} best={best:.6f} no_improve={stale}',flush=True)
  if args.mode=='pilot':break
  if stale>=PATIENCE:break
 row={'status':'COMPLETE' if args.mode=='tune' else 'PILOT_ONLY',
      'model':'RGTAN','dataset':'Elliptic','ratio':'TR40','trial':j,
      'best_val_auprc':best,'best_epoch':best_epoch,'epochs_completed':epoch,
      'config':config,'batch_size':BATCH,'early_stopping_patience':PATIENCE,
      'max_epochs':MAX_EPOCHS,'seed':2,'asset_sha256':asset_sha,'author_commit':PIN,
      'test_scored':False,'test_labels_used':False,
      'optimizer_steps':eng.steps,'wall_seconds':round(time.perf_counter()-start,3),
      'script_sha256':sha(__file__)}
 if args.mode=='tune':
  atomic_json(out/'RGTAN_Elliptic_TR40_History.json',history)
  atomic_json(done,row)
 else:
  atomic_json(OUT/'RGTAN_Elliptic_TR40_OneEpochPilot.json',row)
 print(f'RGTAN_Elliptic_TR40_{args.mode.upper()}_TRIAL_{j:02d}=PASS',flush=True)
 del eng;torch.cuda.empty_cache()
 return row


def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['pilot','tune'],required=True)
 args=p.parse_args()
 # Change into pinned author repo for module import resolution.
 sys.path.insert(0,str(ROOT/'repos/antifraud'))
 d,m=check_and_read()
 OUT.mkdir(parents=True,exist_ok=True)
 graphs=load_graphs(d)
 asset_sha=m['artifact_sha256']
 if args.mode=='pilot':
  row=run_trial(d,graphs,1,GRID[0],args,asset_sha)
  print('RGTAN_Elliptic_TR40_REAL_ONE_EPOCH_PILOT=PASS test_scored=NO',flush=True)
  return
 results=[]
 for i,conf in enumerate(GRID,1):results.append(run_trial(d,graphs,i,conf,args,asset_sha))
 winner=max(results,key=lambda r:(r['best_val_auprc'],-r['trial']))
 atomic_json(OUT/'RGTAN_Elliptic_TR40_TuningWinner.json',{
  'status':'RGTAN_ELLIPTIC_TR40_12_VALIDATION_TRIALS_COMPLETE',
  'model':'RGTAN','dataset':'Elliptic','ratio':'TR40','winner_trial':winner['trial'],
  'winner_configuration':winner['config'],'winner_epoch':winner['best_epoch'],
  'winner_val_auprc':winner['best_val_auprc'],'crossfit_sha256':asset_sha,
  'trials_completed':12,'test_scored':False,'author_commit':PIN})
 print('RGTAN_Elliptic_TR40_12_VAL_ONLY_TUNING_TRIALS=PASS',flush=True)

if __name__=='__main__':main()
