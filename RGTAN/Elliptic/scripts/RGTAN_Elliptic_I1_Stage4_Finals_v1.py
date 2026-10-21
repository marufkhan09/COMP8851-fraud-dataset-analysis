#!/usr/bin/env python3
"""RGTAN × Elliptic: frozen TR40 tuning winner, 4 ratios × 3 seeds.

Train mode NEVER reads held-out test targets, builds test graph, or scores tests.
Test mode is enabled only after all 12 validation-selected checkpoints and
thresholds are durably complete. Each final model gets a unique exclusive test
claim BEFORE its one held-out scoring pass; incomplete claims stop on resume.

Training-prefix graph t<=3/7/12/20; validation prefix t<=31; testing t<=49.
Edges have same-timestep endpoints in the verified canonical graph. No future
information enters training, and risk label sources are always ratio's train.
"""
import argparse
import csv
import hashlib
import importlib.util
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
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score

ROOT=Path('/workspace/comp8851')
SRC=ROOT/'assets/elliptic/source'
SPLIT=Path('/workspace/uploads/ass2_split_candidates/Elliptic/elliptic_seed2_nested_splits.npz')
AUTHOR=ROOT/'repos/antifraud'
STAGE3=ROOT/'pipeline/RGTAN_Elliptic_TR40_Tune12_v1.py'
TUNE=ROOT/'results/rgtan/elliptic/RGTAN_Elliptic_TR40_Tune12_v1'
WINNER= TUNE/'RGTAN_Elliptic_TR40_TuningWinner.json'
FREEZE=ROOT/'results/rgtan/elliptic/RGTAN_Elliptic_I1_TR40_FrozenWinner_v1.json'
FINALS=ROOT/'results/rgtan/elliptic/RGTAN_Elliptic_I1_12Finals_v1'
PIN='65e583f03ac5c5983e49cc35179fe7a65908b9c1'
SPLIT_SHA='1e7963e9d09935fb33e0df74cfb786cab826d1b95ae10b9592dbc240b48ffdbc'
DATA_HASHES={
 'elliptic_txs_features.csv':'fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0',
 'elliptic_txs_classes.csv':'93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493',
 'elliptic_txs_edgelist.csv':'a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e',
}
CUT={'TR40':20,'TR30':12,'TR20':7,'TR10':3}
COUNTS={'TR40':18889,'TR30':13670,'TR20':9553,'TR10':4543,'val':8726,'test':18949}
SEEDS=(2,42,72)
NAMES=('degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat')
N=203769
DEVICE='cuda:0'
BATCH=1024
PATIENCE=20
MAX_EPOCHS=100


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()


def save_json(p,obj,exclusive=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if exclusive:
  with p.open('x') as f:
   json.dump(obj,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 else:
  tmp=Path(str(p)+'.partial')
  with tmp.open('w') as f:
   json.dump(obj,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,p)


def save_ckpt(p,obj):
 tmp=Path(str(p)+'.partial');torch.save(obj,tmp);os.replace(tmp,p)


def input_preflight():
 assert socket.gethostname()=='2be00658fac4','STOP: Must run I1 RGTAN_Elliptic only'
 assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1'
 assert torch.cuda.is_available() and 'A6000' in torch.cuda.get_device_name(0)
 assert subprocess.check_output(['git','-C',str(AUTHOR),'rev-parse','HEAD'],text=True).strip()==PIN
 assert sha(SPLIT)==SPLIT_SHA
 for name,h in DATA_HASHES.items():assert sha(SRC/name)==h,('STOP: source checksum mismatch',name)
 assert sha(ROOT/'assets/rgtan/elliptic/RGTAN_Elliptic_TR40_TemporalCrossfit_v1.npz')=='4413a8a5af304deb40bd602d160bc1a07e1197587dedc1ecd9ffb79bea37c718', 'STOP: verified TR40 crossfit artifact changed'
 assert STAGE3.is_file()
 with np.load(SPLIT,allow_pickle=False) as z:
  parts={k:np.asarray(z[k],dtype=np.int64).copy() for k in COUNTS}
  assert int(z['split_seed'].item())==2
 for a,b in (('TR10','TR20'),('TR20','TR30'),('TR30','TR40')):
  assert np.isin(parts[a],parts[b]).all()
 for a,b in (('TR40','val'),('TR40','test'),('val','test')):
  assert np.intersect1d(parts[a],parts[b]).size==0
 for key,n in COUNTS.items():
  assert len(parts[key])==n and len(np.unique(parts[key]))==n
 print('RGTAN_Elliptic_I1_STAGE4_SOURCE_SPLIT_ENV=PASS',flush=True)
 return parts


def freeze_winner():
 assert WINNER.exists(), 'STOP: tuning winner missing; 12-trial tuning must complete'
 rows=[];stage_sha=sha(STAGE3)
 for j in range(1,13):
  trial_dir=TUNE/f'trial_{j:02d}'
  result=trial_dir/'RGTAN_Elliptic_TR40_TrialResult.json'
  ckpt=trial_dir/'RGTAN_Elliptic_TR40_BestModel.pt'
  val_file=trial_dir/'RGTAN_Elliptic_TR40_BestValidationScores.npy'
  assert all(p.is_file() for p in (result,ckpt,val_file)),('STOP: missing tuning trial files',j)
  r=json.loads(result.read_text())
  assert r['status']=='COMPLETE' and r['trial']==j and r['script_sha256']==stage_sha
  assert r['seed']==2 and r['test_scored'] is False and r['test_labels_used'] is False
  assert r['config']==M.GRID[j-1] and 0.0 < r['best_val_auprc'] <= 1.0
  assert r['asset_sha256']==sha(ROOT/'assets/rgtan/elliptic/RGTAN_Elliptic_TR40_TemporalCrossfit_v1.npz')
  v=np.load(val_file,allow_pickle=False)
  assert v.shape==(8726,) and np.isfinite(v).all()
  rows.append(r)
 expected=max(rows,key=lambda r:(r['best_val_auprc'],-r['trial']))
 w=json.loads(WINNER.read_text())
 assert w['status']=='RGTAN_ELLIPTIC_TR40_12_VALIDATION_TRIALS_COMPLETE'
 assert w['trials_completed']==12 and w['test_scored'] is False
 assert w['winner_trial']==expected['trial'] and w['winner_configuration']==expected['config']
 assert abs(w['winner_val_auprc']-expected['best_val_auprc'])<1e-10
 assert w['crossfit_sha256']==expected['asset_sha256']
 f={'status':'RGTAN_Elliptic_I1_FROZEN_TR40_WINNER','model':'RGTAN','dataset':'Elliptic',
    'winning_trial':expected['trial'],'winning_config':expected['config'],
    'winner_val_auprc':expected['best_val_auprc'],'winner_epoch':expected['best_epoch'],
    'tuning_winner_sha256':sha(WINNER),'tuning_script_sha256':stage_sha,
    'source_sha256':DATA_HASHES,'split_sha256':SPLIT_SHA,'author_commit':PIN,
    'crossfit_TR40_sha256':w['crossfit_sha256'],'test_scored_in_tuning':False,
    'selection_metric':'validation average precision; no test selection',
    'seeds':list(SEEDS),'ratios':list(CUT)}
 if FREEZE.exists():
  assert json.loads(FREEZE.read_text())==f, 'STOP: frozen winner changed; investigate'
 else:save_json(FREEZE,f,exclusive=True)
 print(f'RGTAN_Elliptic_I1_FROZEN_WINNER=PASS trial={expected["trial"]} val_AUPRC={expected["best_val_auprc"]:.6f} config={expected["config"]}',flush=True)
 return f


def load_stage3():
 spec=importlib.util.spec_from_file_location('rgtan_elliptic_stage3',STAGE3)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 assert mod.GRID==[dict(hidden_dim=h,drop=[d,d/2],base_lr=lr) for h in (32,64) for d in (.2,.3) for lr in (.001,.003,.005)]
 return mod


def read_assets(parts):
 data={}
 for ratio in CUT:
  path=ROOT/f'assets/rgtan/elliptic/RGTAN_Elliptic_{ratio}_TemporalCrossfit_v1.npz'
  manifest=path.with_suffix('.json')
  assert path.is_file() and manifest.is_file(),('STOP: crossfit missing',ratio)
  mm=json.loads(manifest.read_text())
  assert mm['status']==f'RGTAN_ELLIPTIC_{ratio}_REAL_CROSSFIT_PASS'
  assert mm['artifact_sha256']==sha(path)
  assert mm['author_commit']==PIN and mm['split_sha256']==SPLIT_SHA
  assert mm['source_sha256']==DATA_HASHES and mm['ratio']==ratio
  assert mm['training_timestep_max']==CUT[ratio] and mm['validation_timestep_max']==31
  assert mm['validation_labels_accessed'] is False and mm['test_labels_accessed'] is False
  with np.load(path,allow_pickle=False) as z: d={k:z[k] for k in z.files}
  assert np.array_equal(d['train_global_ids'],parts[ratio]) and np.array_equal(d['val_global_ids'],parts['val'])
  assert len(d['train_global_ids'])==COUNTS[ratio]
  assert tuple(str(n) for n in d['risk_feature_names'])==NAMES
  y=d['train_labels'];assert y.shape==(COUNTS[ratio],) and set(np.unique(y))=={0,1}
  assert (d['train_fold_of_local'][d['train_prefix_local_ids']]>=0).all()
  assert set(d['train_fold_of_local'][d['train_prefix_local_ids']].tolist())==set(range(5))
  for k in range(5):
   ids=d['train_prefix_local_ids'][d['train_fold_of_local'][d['train_prefix_local_ids']]==k]
   assert ids.size>0 and np.all(d[f'train_label_context_{k}'][ids]==2)
   assert np.isfinite(d[f'train_risk_view_{k}']).all()
  assert np.all(d['val_context'][d['val_prefix_local_ids']]==2)
  assert np.isfinite(d['val_risk_view']).all()
  data[ratio]=(d,mm)
 print('RGTAN_Elliptic_I1_ALL_RATIO_OOF_AND_CONTEXT_ISOLATION=PASS',flush=True)
 return data


def read_graph_inputs(parts,read_val_targets=True):
 dtype={0:np.int64,1:np.int16};dtype.update({i:np.float32 for i in range(2,167)})
 print('RGTAN_Elliptic_I1_READING_CANONICAL_FEATURES',flush=True)
 t=pd.read_csv(SRC/'elliptic_txs_features.csv',header=None,dtype=dtype)
 assert t.shape==(N,167)
 txid=t.iloc[:,0].to_numpy(dtype=np.int64,copy=True)
 times=t.iloc[:,1].to_numpy(dtype=np.int16,copy=True)
 x=t.iloc[:,2:].to_numpy(dtype=np.float32,copy=True)
 del t
 assert len(np.unique(txid))==N and np.isfinite(x).all()
 for ratio,cut in CUT.items():assert (times[parts[ratio]]<=cut).all()
 assert ((times[parts['val']]>=21)&(times[parts['val']]<=31)).all()
 assert (times[parts['test']]>=32).all()
 edges=pd.read_csv(SRC/'elliptic_txs_edgelist.csv',dtype=np.int64)
 assert len(edges)==234355
 ix=pd.Index(txid)
 u=ix.get_indexer(edges['txId1']).astype(np.int64)
 v=ix.get_indexer(edges['txId2']).astype(np.int64)
 del edges
 assert (u>=0).all() and (v>=0).all() and (times[u]==times[v]).all()
 val_y=None
 if read_val_targets:
  val_y=read_targets(txid,parts['val'],expected_sum=1289)
 return dict(txid=txid,times=times,x=x,u=u,v=v,val_y=val_y)


def read_targets(txid,ids,expected_sum):
 allowed=set(map(int,txid[ids])); found={}
 with (SRC/'elliptic_txs_classes.csv').open(newline='') as f:
  reader=csv.DictReader(f)
  for row in reader:
   key=int(row['txId'])
   if key in allowed:
    assert row['class'] in ('1','2') and key not in found
    found[key]=int(row['class']=='1')
 assert len(found)==len(allowed)
 y=np.asarray([found[int(k)] for k in txid[ids]],dtype=np.int64)
 assert int(y.sum())==expected_sum
 return y


def graph_prefix(raw,cut):
 times=raw['times']
 glob=np.flatnonzero(times<=cut)
 mapping=np.full(N,-1,dtype=np.int64);mapping[glob]=np.arange(len(glob))
 keep=(times[raw['u']]<=cut)&(times[raw['v']]<=cut)
 source=mapping[raw['u'][keep]];dest=mapping[raw['v'][keep]]
 assert (source>=0).all() and (dest>=0).all()
 g=dgl.graph((torch.from_numpy(source.copy()),torch.from_numpy(dest.copy())),num_nodes=len(glob))
 g=dgl.add_self_loop(dgl.remove_self_loop(g))
 assert not g.ndata and not g.edata
 return glob,g


class FinalEngine:
 # Use Stage3's exact RGTAN architecture, masking and OOF feature semantics.
 # Only generalization: the Stage3 hardcoded 18,889 train count becomes the frozen ratio count.
 def __init__(self, d, graphs, config, seed):
  self.engine=M.Engine(d,graphs,config,seed)
  self.n=len(d['train_global_ids'])

 def train_epoch(self,ep):
  e=self.engine;e.model.train();loss_sum=0;seen=0
  order=np.random.RandomState(e.seed+ep).permutation(5)
  for fold in order:
   for batch in e.tr_loaders[fold]:
    logits,b=e.forward(batch,'train',fold)
    targets=torch.from_numpy(np.asarray([e.labs[fold][int(i)] for i in b],dtype=np.int64)).to(DEVICE)
    loss=F.cross_entropy(logits,targets)
    assert bool(torch.isfinite(loss))
    e.opt.zero_grad(set_to_none=True);loss.backward();e.opt.step();e.steps+=1
    loss_sum+=float(loss.item())*len(b);seen+=len(b)
  assert seen==self.n,('STOP: training examples mismatch',seen,self.n)
  return loss_sum/seen


def val_best_threshold(y,scores):
 """Exact validation macro-F1 sweep, threshold scores>=cut, deterministic tie: higher cut."""
 y=np.asarray(y,dtype=np.int8);s=np.asarray(scores,dtype=np.float64)
 assert y.ndim==s.ndim==1 and len(y)==len(s) and np.isfinite(s).all()
 assert set(np.unique(y))=={0,1}
 order=np.argsort(-s,kind='mergesort')
 ss=s[order];yy=y[order]
 tp_c=np.cumsum(yy,dtype=np.int64)
 n=len(y);positives=int(yy.sum());negatives=n-positives
 ends=np.r_[np.flatnonzero(np.diff(ss)!=0),n-1]
 tp=tp_c[ends];fp=ends+1-tp;fn=positives-tp;tn=negatives-fp
 f1pos=2*tp/(2*tp+fp+fn)
 f1neg=2*tn/(2*tn+fp+fn)
 f1macro=(f1pos+f1neg)*.5
 # also threshold above max => all predicted normal
 allnormal=(2*negatives/(2*negatives+positives))/2
 ix=int(np.argmax(np.r_[allnormal,f1macro]))
 if ix==0:return float(np.nextafter(ss[0],np.inf)),float(allnormal)
 return float(ss[ends[ix-1]]),float(f1macro[ix-1])


def per_seed_paths(ratio,seed):
 folder=FINALS/f'RGTAN_Elliptic_{ratio}_Seed{seed}'
 return (folder,folder/f'RGTAN_Elliptic_{ratio}_Seed{seed}_BestModel.pt',
         folder/f'RGTAN_Elliptic_{ratio}_Seed{seed}_FinalTraining.json',
         folder/f'RGTAN_Elliptic_{ratio}_Seed{seed}_BestValidationScores.npy')


def train_one(ratio,seed,info,config,graphs,freeze_sha):
 d,meta=info
 path,ckpt,done,valfile=per_seed_paths(ratio,seed)
 path.mkdir(parents=True,exist_ok=True)
 if done.exists():
  rr=json.loads(done.read_text())
  assert rr['status']=='RGTAN_Elliptic_FINAL_TRAINING_FROZEN'
  assert rr['training_script_sha256']==sha(__file__)
  assert rr['frozen_winner_sha256']==freeze_sha and rr['crossfit_sha256']==meta['artifact_sha256']
  assert rr['seed']==seed and rr['ratio']==ratio and rr['config']==config
  assert ckpt.exists() and valfile.exists() and rr['checkpoint_sha256']==sha(ckpt)
  assert rr['validation_scores_sha256']==sha(valfile)
  assert rr['test_scored'] is False
  print(f'RGTAN_Elliptic_{ratio}_Seed{seed}_FINAL_TRAINING_REUSED=PASS',flush=True)
  return rr
 e=FinalEngine(d,graphs,config,seed); engine=e.engine
 ckpt_tmp=Path(str(ckpt)+'.partial')
 if ckpt_tmp.exists():raise RuntimeError(f'STOP: incomplete checkpoint write: {ckpt_tmp}')
 start=time.perf_counter();best=-1.0;epoch_best=0;stale=0;hist=[]
 torch.cuda.reset_peak_memory_stats()
 for epoch in range(1,MAX_EPOCHS+1):
  loss=e.train_epoch(epoch)
  score,pred=engine.evaluate()
  hist.append({'epoch':epoch,'train_loss':loss,'val_auprc':score})
  if score>best+1e-9:
   best=score;epoch_best=epoch;stale=0
   save_ckpt(ckpt,{'model':engine.model.state_dict(),'config':config,'seed':seed,
     'ratio':ratio,'best_epoch':epoch,'val_auprc':score,'author_commit':PIN,
     'crossfit_sha256':meta['artifact_sha256'],'frozen_winner_sha256':freeze_sha,
     'model_name':'RGTAN','dataset_name':'Elliptic','test_scored':False})
   tmp=Path(str(valfile)+'.partial')
   with tmp.open('wb') as f:np.save(f,pred);f.flush();os.fsync(f.fileno())
   os.replace(tmp,valfile)
  else:stale+=1
  if epoch<=2 or epoch%5==0 or stale>=PATIENCE:
   print(f'RGTAN_Elliptic_{ratio}_Seed{seed} epoch={epoch} loss={loss:.5f} val_AUPRC={score:.6f} best={best:.6f} stale={stale}',flush=True)
  if stale>=PATIENCE:break
 assert best>0 and ckpt.exists() and valfile.exists()
 val_scores=np.load(valfile,allow_pickle=False)
 threshold,macro_f1=val_best_threshold(engine.val_y,val_scores)
 actual=float(f1_score(engine.val_y,(val_scores>=threshold).astype(int),average='macro',zero_division=0))
 assert abs(actual-macro_f1)<1e-10,(actual,macro_f1)
 result={'status':'RGTAN_Elliptic_FINAL_TRAINING_FROZEN','model':'RGTAN','dataset':'Elliptic',
  'ratio':ratio,'seed':seed,'config':config,'batch_size':BATCH,'max_epochs':MAX_EPOCHS,
  'patience':PATIENCE,'epochs_completed':epoch,'best_epoch':epoch_best,'best_val_auprc':best,
  'val_macro_f1':macro_f1,'val_macro_f1_threshold':threshold,
  'validation_label_count':8726,'validation_positive_count':1289,
  'training_count':len(d['train_global_ids']),'training_positive_count':int(d['train_labels'].sum()),
  'optimizer_steps':engine.steps,'train_seconds':round(time.perf_counter()-start,3),
  'gpu_peak_gib':round(torch.cuda.max_memory_allocated()/1024**3,5),
  'author_commit':PIN,'crossfit_sha256':meta['artifact_sha256'],
  'frozen_winner_sha256':freeze_sha,'checkpoint_sha256':sha(ckpt),
  'validation_scores_sha256':sha(valfile),'training_script_sha256':sha(__file__),
  'train_time_cutoff':CUT[ratio],'validation_time_cutoff':31,
  'test_label_accessed':False,'test_scored':False}
 save_json(path/f'RGTAN_Elliptic_{ratio}_Seed{seed}_TrainingHistory.json',hist)
 save_json(done,result)
 print(f'RGTAN_Elliptic_{ratio}_Seed{seed}_FINAL_VAL_FREEZE=PASS best_AUPRC={best:.6f} val_macroF1={macro_f1:.6f}',flush=True)
 del e,engine;torch.cuda.empty_cache()
 return result


def run_train(parts,config,freeze):
 assets=read_assets(parts)
 raw=read_graph_inputs(parts)
 valglob,valgraph=graph_prefix(raw,31)
 assert len(valglob)==126103 and valgraph.num_edges()==271936
 freeze_sha=sha(FREEZE)
 completed=[]
 for ratio in CUT:
  d,meta=assets[ratio]
  trglob,trgraph=graph_prefix(raw,CUT[ratio])
  assert np.array_equal(trglob,d['train_prefix_global_ids'])
  assert np.array_equal(valglob,d['val_prefix_global_ids'])
  assert np.isin(parts[ratio],trglob).all() and not np.isin(parts['val'],trglob).any()
  graphs=(trgraph,valgraph,raw['x'][trglob],raw['x'][valglob],raw['val_y'])
  for seed in SEEDS:
   row=train_one(ratio,seed,(d,meta),config,graphs,freeze_sha)
   completed.append(row)
  del trgraph
 summary={'status':'RGTAN_Elliptic_I1_ALL_12_FINAL_TRAININGS_FROZEN',
  'model':'RGTAN','dataset':'Elliptic','winner_sha256':freeze_sha,
  'ratios':list(CUT),'seeds':list(SEEDS),'model_count':len(completed),
  'all_no_test_scoring':all(r['test_scored'] is False for r in completed),
  'rows':[{'ratio':r['ratio'],'seed':r['seed'],'val_auprc':r['best_val_auprc'],
           'val_macro_f1':r['val_macro_f1'],'checkpoint_sha256':r['checkpoint_sha256']} for r in completed]}
 summfile=FINALS/'RGTAN_Elliptic_I1_All12_FrozenTrainingManifest.json'
 if summfile.exists():assert json.loads(summfile.read_text())==summary
 else:save_json(summfile,summary,exclusive=True)
 print('RGTAN_Elliptic_I1_ALL_12_FINAL_TRAININGS_FROZEN=PASS TEST_SCORED=NO',flush=True)


def full_test_risk(raw,d):
 """Unchanged directed binary exact-2hop view, fit normalization on train OOF only."""
 u=raw['u'];v=raw['v'];N=len(raw['times'])
 adj=sp.coo_matrix((np.ones(len(u),dtype=np.int8),(v,u)),shape=(N,N)).tocsr()
 adj.sum_duplicates();adj.data.fill(1);adj.setdiag(0);adj.eliminate_zeros();adj.sort_indices()
 two=(adj.astype(np.int32)@adj.astype(np.int32)).tocsr()
 two.data.fill(1);two=two-two.multiply(adj.astype(np.int32))
 two.setdiag(0);two.eliminate_zeros();two.sort_indices()
 train=d['train_global_ids'];y=d['train_labels']
 pos=np.zeros((N,1),dtype=np.float32);pos[train,0]=y
 degree=np.diff(adj.indptr).astype(np.float32)
 first=(adj@pos).astype(np.float32)
 fd=np.asarray(adj@degree,dtype=np.float32)
 sd=np.asarray(two@degree,dtype=np.float32)
 fr=np.asarray(adj@first,dtype=np.float32)
 sr=np.asarray(two@first,dtype=np.float32)
 feat=np.column_stack([degree,first[:,0],fd,sd,fr[:,0],sr[:,0]]).astype(np.float32)
 mean=np.asarray(d['scaler_mean'],dtype=np.float64)
 scale=np.asarray(d['scaler_scale'],dtype=np.float64)
 assert mean.shape==scale.shape==(6,) and np.all(scale>0)
 risk=((feat.astype(np.float64)-mean)/scale).astype(np.float32)
 assert risk.shape==(N,6) and np.isfinite(risk).all()
 ctx=np.full(N,2,dtype=np.int64);ctx[train]=y
 assert np.all(ctx[d['val_global_ids']]==2)
 return risk,ctx


def result_metrics(targets,prob,threshold):
 targets=np.asarray(targets,dtype=np.int64);prob=np.asarray(prob,dtype=np.float64)
 assert np.isfinite(prob).all() and len(targets)==len(prob)
 pred=(prob>=threshold).astype(np.int64)
 return {'auprc':float(average_precision_score(targets,prob)),
  'auroc':float(roc_auc_score(targets,prob)),
  'macro_f1':float(f1_score(targets,pred,average='macro',zero_division=0)),
  'fraud_f1':float(f1_score(targets,pred,pos_label=1,zero_division=0)),
  'fraud_precision':float(precision_score(targets,pred,pos_label=1,zero_division=0)),
  'fraud_recall':float(recall_score(targets,pred,pos_label=1,zero_division=0)),
  'predicted_fraud_count':int(pred.sum()),'true_fraud_count':int(targets.sum())}


def run_test(parts,config,freeze):
 # Stop before the FIRST test claim unless every training is successfully frozen.
 summfile=FINALS/'RGTAN_Elliptic_I1_All12_FrozenTrainingManifest.json'
 assert summfile.is_file(), 'STOP: All 12 final models must be trained first'
 summary=json.loads(summfile.read_text())
 assert summary['status']=='RGTAN_Elliptic_I1_ALL_12_FINAL_TRAININGS_FROZEN' and summary['model_count']==12
 frozen=[];assets=read_assets(parts);freeze_sha=sha(FREEZE)
 for ratio in CUT:
  d,meta=assets[ratio]
  for seed in SEEDS:
   folder,ckpt,done,valfile=per_seed_paths(ratio,seed)
   assert all(p.is_file() for p in (done,ckpt,valfile))
   r=json.loads(done.read_text())
   assert r['status']=='RGTAN_Elliptic_FINAL_TRAINING_FROZEN'
   assert r['training_script_sha256']==sha(__file__)
   assert r['frozen_winner_sha256']==freeze_sha and r['crossfit_sha256']==meta['artifact_sha256']
   assert r['checkpoint_sha256']==sha(ckpt) and r['validation_scores_sha256']==sha(valfile)
   assert r['test_scored'] is False and r['test_label_accessed'] is False
   frozen.append((ratio,seed,r))
 assert len(frozen)==12
 print('RGTAN_Elliptic_I1_ALL_12_MODELS_FROZEN_BEFORE_HELDOUT=PASS',flush=True)
 raw=read_graph_inputs(parts,read_val_targets=False)
 testglob,gtest=graph_prefix(raw,49)
 assert len(testglob)==203769
 loader=NodeDataLoader(gtest,torch.as_tensor(parts['test'].copy()),
    MultiLayerFullNeighborSampler(2),batch_size=BATCH,shuffle=False,drop_last=False,num_workers=0)
 assert np.all(raw['times'][parts['test']]>=32)
 print('RGTAN_Elliptic_I1_FULL_TEST_GRAPH_BUILT_LABEL_FREE=PASS',flush=True)
 risk_cache={}
 for ratio in CUT:
  d,meta=assets[ratio]
  risk,ctx=full_test_risk(raw,d)
  assert np.all(ctx[parts['test']]==2), 'STOP: TEST labels used in inference context'
  risk_cache[ratio]=(risk,ctx)
 print('RGTAN_Elliptic_I1_FULL_TEST_RISK_USES_TRAIN_LABELS_ONLY=PASS',flush=True)
 for ratio,seed,r in frozen:
  folder,ckpt,done,valfile=per_seed_paths(ratio,seed)
  metrics_file=folder/f'RGTAN_Elliptic_{ratio}_Seed{seed}_FinalTestMetrics.json'
  claim=folder/f'RGTAN_Elliptic_{ratio}_Seed{seed}_ONE_TIME_TEST_CLAIM.json'
  scores_file=folder/f'RGTAN_Elliptic_{ratio}_Seed{seed}_TestScores.npy'
  if metrics_file.exists():
   m=json.loads(metrics_file.read_text())
   assert m['status']=='RGTAN_Elliptic_FINAL_HELDOUT_SCORED_ONCE' and claim.exists()
   assert m['checkpoint_sha256']==r['checkpoint_sha256'] and m['source_sha256']==DATA_HASHES
   assert scores_file.exists() and m['test_scores_sha256']==sha(scores_file)
   print(f'RGTAN_Elliptic_{ratio}_Seed{seed}_TEST_RESULT_REUSED=PASS',flush=True)
   continue
  assert not claim.exists(), f'STOP: one-time test claim already exists, no metrics; audit before retry: {claim}'
  print(f'RGTAN_Elliptic_{ratio}_Seed{seed}_CHECKING_FROZEN_MODEL_BEFORE_TEST_CLAIM',flush=True)
  from methods.rgtan.rgtan_model import RGTAN
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  train_count=len(assets[ratio][0]['train_prefix_global_ids'])
  zero={n:torch.zeros(train_count,device=DEVICE) for n in NAMES}
  model=RGTAN(in_feats=165,hidden_dim=config['hidden_dim'],n_layers=2,n_classes=2,
   heads=[4,4],activation=torch.nn.PReLU(),drop=config['drop'],device=DEVICE,
   gated=True,ref_df=pd.DataFrame(raw['x'][assets[ratio][0]['train_prefix_global_ids']]),
   cat_features={},neigh_features=zero,nei_att_head=3).to(DEVICE)
  saved=torch.load(ckpt,map_location=DEVICE)
  assert saved['crossfit_sha256']==r['crossfit_sha256'] and saved['frozen_winner_sha256']==freeze_sha
  assert saved['seed']==seed and saved['ratio']==ratio and saved['config']==config
  model.load_state_dict(saved['model'],strict=True);model.eval()
  # Only now claim the single held-out score: package, weights and architecture
  # are validated beforehand, reducing avoidable half-claimed evaluations.
  save_json(claim,{'model':'RGTAN','dataset':'Elliptic','ratio':ratio,'seed':seed,
   'checkpoint_sha256':r['checkpoint_sha256'],'threshold':r['val_macro_f1_threshold'],
   'one_time_test_scoring_claimed':True},exclusive=True)
  target=read_targets(raw['txid'],parts['test'],expected_sum=1485)
  risk,ctx=risk_cache[ratio]
  scores=np.empty(len(parts['test']),dtype=np.float32)
  lookup={int(k):i for i,k in enumerate(parts['test'])}
  torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
  with torch.no_grad():
   for input_nodes,seeds,blocks in loader:
    a=input_nodes.numpy();b=seeds.numpy()
    assert np.all(ctx[b]==2)
    x=torch.from_numpy(raw['x'][a].copy()).to(DEVICE)
    labelctx=torch.from_numpy(ctx[a].copy()).long().to(DEVICE)
    risk_dict={n:torch.from_numpy(risk[a,j].copy()).to(DEVICE) for j,n in enumerate(NAMES)}
    logits=model([block.to(DEVICE) for block in blocks],x,labelctx,{},risk_dict)
    assert logits.shape==(len(b),2) and bool(torch.isfinite(logits).all())
    pred=torch.softmax(logits,dim=1)[:,1].cpu().numpy()
    for j,node in enumerate(b):scores[lookup[int(node)]]=pred[j]
  assert np.isfinite(scores).all()
  score_temp=Path(str(scores_file)+'.partial')
  with score_temp.open('wb') as f:np.save(f,scores);f.flush();os.fsync(f.fileno())
  os.replace(score_temp,scores_file)
  measurements=result_metrics(target,scores,r['val_macro_f1_threshold'])
  result={'status':'RGTAN_Elliptic_FINAL_HELDOUT_SCORED_ONCE',
   'model':'RGTAN','dataset':'Elliptic','ratio':ratio,'seed':seed,
   'checkpoint_sha256':r['checkpoint_sha256'],'crossfit_sha256':r['crossfit_sha256'],
   'frozen_winner_sha256':freeze_sha,'source_sha256':DATA_HASHES,'split_sha256':SPLIT_SHA,
   'test_size':len(target),'test_timestep_range':[32,49],
   'decision_threshold_frozen_validation_macro_f1':r['val_macro_f1_threshold'],
   'validation_auprc':r['best_val_auprc'],'validation_macro_f1':r['val_macro_f1'],
   'test_scores_sha256':sha(scores_file),'test_scoring_seconds':round(time.perf_counter()-start,3),
   'gpu_peak_gib':round(torch.cuda.max_memory_allocated()/1024**3,5),
   'test_scored_once':True,'metrics':measurements}
  save_json(metrics_file,result)
  print(f'RGTAN_Elliptic_{ratio}_Seed{seed}_ONE_TIME_TEST_SCORE=PASS '
        f'AUPRC={measurements["auprc"]:.6f} macroF1={measurements["macro_f1"]:.6f}',flush=True)
  del model,saved;torch.cuda.empty_cache()
 aggregated=[]
 for ratio in CUT:
  vals=[]
  for seed in SEEDS:
   f=per_seed_paths(ratio,seed)[0]/f'RGTAN_Elliptic_{ratio}_Seed{seed}_FinalTestMetrics.json'
   vals.append(json.loads(f.read_text())['metrics'])
  keys=['auprc','auroc','macro_f1','fraud_f1','fraud_precision','fraud_recall']
  summaryrow={'ratio':ratio,'seeds':list(SEEDS)}
  for k in keys:
   numbers=np.array([v[k] for v in vals],dtype=np.float64)
   summaryrow[k+'_mean']=float(numbers.mean())
   summaryrow[k+'_sample_sd']=float(numbers.std(ddof=1))
  aggregated.append(summaryrow)
 report={'status':'RGTAN_Elliptic_I1_ALL_12_FINAL_HELDOUT_ONCE_COMPLETE',
    'model':'RGTAN','dataset':'Elliptic','ratios':list(CUT),'seeds':list(SEEDS),
    'summary_rows':aggregated,'std_convention':'sample SD ddof=1',
    'author_commit':PIN,'source_sha256':DATA_HASHES,'split_sha256':SPLIT_SHA,
    'full_neighbour_sampling':True,'chronological_train_val_test_isolation':True,
    'heldout_scored_once_per_final_run':True}
 path=FINALS/'RGTAN_Elliptic_I1_AllRatios_FinalTestSummary.json'
 if path.exists():assert json.loads(path.read_text())==report
 else:save_json(path,report)
 for row in aggregated:
  print(f'RGTAN_Elliptic_{row["ratio"]}_FINAL_TEST_AUPRC_MEAN={row["auprc_mean"]:.6f} sampleSD={row["auprc_sample_sd"]:.6f} '
        f'macroF1_MEAN={row["macro_f1_mean"]:.6f}',flush=True)
 print('RGTAN_Elliptic_I1_ALL_12_FINAL_TESTS_COMPLETE=PASS',flush=True)


def main():
 global M
 parser=argparse.ArgumentParser()
 parser.add_argument('--mode',choices=['freeze','train','test'],required=True)
 args=parser.parse_args()
 sys.path.insert(0,str(AUTHOR))
 M=load_stage3()
 parts=input_preflight();freeze=freeze_winner()
 if args.mode=='freeze':return
 config=freeze['winning_config']
 if args.mode=='train':run_train(parts,config,freeze)
 else:run_test(parts,config,freeze)

if __name__=='__main__':main()
