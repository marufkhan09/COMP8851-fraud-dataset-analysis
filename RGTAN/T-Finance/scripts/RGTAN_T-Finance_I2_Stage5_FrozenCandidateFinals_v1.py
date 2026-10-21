#!/usr/bin/env python3
"""RGTAN x T-Finance: freeze completed TR40 VAL-only candidate tuning winner,
then train 4 ratios x seeds [2,42,72]. NO heldout test labels read or scored.

Provisional candidate ONLY: full original graph + exact train-only 2-hop OOF
risk, but bounded message-passing neighbor fanouts [10,10] differ from author
full-neighbor sampling. Report as candidate; supervisor approval required for
benchmark. Rerun reuses verified completed training assets/checkpoints.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, random, socket, subprocess, sys, time
from pathlib import Path
import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score

ROOT=Path('/workspace/comp8851')
STAGE4=ROOT/'pipeline/RGTAN_T-Finance_TR40_Tune12_Candidate_v1.py'
SPLIT=Path('/workspace/uploads/ass2_split_candidates/T-Finance/tfinance_seed2_nested_splits.npz')
SOURCE=ROOT/'assets/tfinance/canonical_candidate/tfinance_canonical.dgl'
ARTIFACT_DIR=ROOT/'assets/rgtan/tfinance'
TUNE=ROOT/'results/rgtan/tfinance/RGTAN_T-Finance_TR40_Tune12_Fanout10x10_Candidate_v1'
WINNER=TUNE/'RGTAN_T-Finance_TR40_CandidateTune12Winner.json'
FREEZE=ROOT/'results/rgtan/tfinance/RGTAN_T-Finance_I2_TR40_FrozenCandidateWinner_v1.json'
OUT=ROOT/'results/rgtan/tfinance/RGTAN_T-Finance_I2_12CandidateFinals_Fanout10x10_v1'
MODEL='RGTAN'; DATASET='T-Finance'; PIN='65e583f03ac5c5983e49cc35179fe7a65908b9c1'
DATA_SHA='b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700'
SPLIT_SHA='de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1'
TR40_SHA='798daba5c4a131159208d242b71e03ef3d1c79aa44f5cfd599c7fd97b98dab81'
COUNTS={'TR40':15742,'TR30':11806,'TR20':7870,'TR10':3935,'val':7872,'test':15743}
POSITIVES={'TR40':721,'TR30':541,'TR20':361,'TR10':181,'val':361}
RATIOS=('TR40','TR30','TR20','TR10'); SEEDS=(2,42,72); FANOUTS=[10,10]
NAMES=('degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat')
MAX_EPOCHS=100; PATIENCE=20


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()


def atomic_json(p,obj,exclusive=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if exclusive:
  with p.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 else:
  tmp=Path(str(p)+'.partial')
  with tmp.open('w') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,p)


def save_checkpoint(path,obj):
 tmp=Path(str(path)+'.partial')
 torch.save(obj,tmp);os.replace(tmp,path)


def import_stage4():
 assert STAGE4.is_file(),'STOP: Stage4 tuned code missing'
 spec=importlib.util.spec_from_file_location('RGTAN_TFinance_Stage4',STAGE4)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert m.DATA_SHA==DATA_SHA and m.SPLIT_SHA==SPLIT_SHA and m.RISK_SHA==TR40_SHA
 assert m.FANOUTS==tuple(FANOUTS) and m.BATCH==1024 and m.MAX_EPOCHS==100
 return m


def preflight():
 assert socket.gethostname()=='f24d1bb2672d','STOP: Wrong instance, I2 ONLY'
 assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1'
 assert torch.cuda.is_available() and 'A6000' in torch.cuda.get_device_name(0)
 assert subprocess.check_output(['git','-C',str(ROOT/'repos/antifraud'),'rev-parse','HEAD'],text=True).strip()==PIN
 assert sha(SOURCE)==DATA_SHA and sha(SPLIT)==SPLIT_SHA
 with np.load(SPLIT,allow_pickle=False) as z:
  assert int(z['seed'].item())==2 and int(z['source_fraud'].item())==1804
  parts={key:np.asarray(z[key],dtype=np.int64).copy() for key in COUNTS}
 for k,n in COUNTS.items():assert len(parts[k])==n and len(np.unique(parts[k]))==n
 for a,b in (('TR10','TR20'),('TR20','TR30'),('TR30','TR40')):assert np.isin(parts[a],parts[b]).all()
 for a,b in (('TR40','val'),('TR40','test'),('val','test')):assert not np.intersect1d(parts[a],parts[b]).size
 print('RGTAN_T-Finance_I2_STAGE5_FROZEN_SOURCES_SPLITS_ENV=PASS',flush=True)
 return parts


def freeze_winner(T):
 assert WINNER.is_file(),'STOP: Stage4 12-trial tuning winner file missing'
 rows=[];script_sha=sha(STAGE4)
 for j in range(1,13):
  folder=TUNE/f'trial_{j:02d}'
  manifest=folder/'RGTAN_T-Finance_TR40_CandidateTrialResult.json'
  checkpoint=folder/'RGTAN_T-Finance_TR40_CandidateBestModel.pt'
  valfile=folder/'RGTAN_T-Finance_TR40_CandidateBestValidationScores.npy'
  assert all(x.is_file() for x in (manifest,checkpoint,valfile)),('STOP: Missing Stage4 tune trial artifact',j)
  r=json.loads(manifest.read_text())
  assert r['status']=='COMPLETE' and r['trial']==j and r['config']==T.GRID[j-1]
  assert r['seed']==2 and r['risk_sha256']==TR40_SHA and r['source_sha256']==DATA_SHA and r['split_sha256']==SPLIT_SHA
  assert r['script_sha256']==script_sha and r['checkpoint_sha256']==sha(checkpoint)
  assert r['fanouts']==FANOUTS and r['original_full_neighbor_sampling_replaced'] is True
  assert r['test_scored'] is False and r['test_labels_used'] is False
  preds=np.load(valfile,allow_pickle=False)
  assert preds.shape==(COUNTS['val'],) and np.isfinite(preds).all()
  rows.append(r)
 w=json.loads(WINNER.read_text())
 best=max(rows,key=lambda r:(r['best_val_auprc'],-r['trial']))
 assert w['status']=='RGTAN_TFINANCE_TR40_CANDIDATE_12_VAL_ONLY_TUNE_COMPLETE'
 assert w['trial_count']==12 and w['winner_trial']==best['trial']
 assert w['winner_configuration']==best['config']
 assert abs(w['winner_val_auprc']-best['best_val_auprc'])<1e-12
 assert w['benchmark_supervisor_approval_required'] is True
 assert w['test_scored'] is False and w['test_labels_used'] is False
 record={'status':'RGTAN_T-Finance_I2_TR40_VALIDATION_WINNER_FROZEN_CANDIDATE',
  'model':MODEL,'dataset':DATASET,'trial_count':12,'trial':best['trial'],
  'configuration':best['config'],'winner_epoch':best['best_epoch'],
  'winner_val_auprc':best['best_val_auprc'],
  'tuning_winner_sha256':sha(WINNER),'tuning_script_sha256':script_sha,
  'source_sha256':DATA_SHA,'split_sha256':SPLIT_SHA,'crossfit_TR40_sha256':TR40_SHA,
  'author_commit':PIN,'fanouts':FANOUTS,'original_full_neighbor_sampling_replaced':True,
  'benchmark_supervisor_approval_required':True,'candidate_only':True,
  'heldout_test_evaluation_authorized':False,'test_scored':False,'seeds':list(SEEDS),'ratios':list(RATIOS)}
 if FREEZE.exists():assert json.loads(FREEZE.read_text())==record,'STOP: Existing frozen winner differs'
 else:atomic_json(FREEZE,record,exclusive=True)
 print(f'RGTAN_T-Finance_I2_TR40_CANDIDATE_WINNER=PASS trial={best["trial"]} '
       f'val_AUPRC={best["best_val_auprc"]:.6f} config={best["config"]}',flush=True)
 return record


def read_assets(parts):
 allfiles={}
 for ratio in RATIOS:
  p=ARTIFACT_DIR/f'RGTAN_T-Finance_{ratio}_ExactRiskCrossfit_v1.npz'
  m=p.with_suffix('.json')
  assert p.is_file() and m.is_file(),('STOP: missing exact risk asset',ratio)
  data=json.loads(m.read_text());actual=sha(p)
  assert data['artifact_sha256']==actual and data['source_sha256']==DATA_SHA and data['split_sha256']==SPLIT_SHA
  if ratio=='TR40':
   assert actual==TR40_SHA and data['status']=='RGTAN_TFINANCE_TR40_EXACT_RISK_CROSSFIT_PASS'
  else:
   assert data['status']=='RGTAN_TFINANCE_'+ratio+'_EXACT_RISK_CROSSFIT_PASS'
  assert data['test_labels_used'] is False and data['validation_labels_used'] is False
  with np.load(p,allow_pickle=False) as z:fields={key:z[key] for key in z.files}
  assert tuple(map(str,fields['risk_feature_names']))==NAMES
  assert np.array_equal(fields['train_ids'],parts[ratio]) and np.array_equal(fields['val_ids'],parts['val'])
  assert fields['features'].shape==(39357,10)
  assert len(fields['train_labels'])==COUNTS[ratio] and int(fields['train_labels'].sum())==POSITIVES[ratio]
  assert set(fields['fold_of_node'][parts[ratio]])==set(range(5))
  assert np.isfinite(fields['features']).all()
  for k in range(6):
   assert fields[f'risk_view_{k}'].shape==(39357,6)
   assert fields[f'label_context_{k}'].shape==(39357,)
   assert np.isfinite(fields[f'risk_view_{k}']).all()
   assert np.all(fields[f'label_context_{k}'][parts['val']]==2)
   assert np.all(fields[f'label_context_{k}'][parts['test']]==2)
  for k in range(5):
   held=parts[ratio][fields['fold_of_node'][parts[ratio]]==k]
   assert len(held)>0 and np.all(fields[f'label_context_{k}'][held]==2)
  assert np.all(fields['label_context_5'][parts[ratio]]==fields['train_labels'])
  allfiles[ratio]=(fields,actual)
 print('RGTAN_T-Finance_I2_ALL_RATIO_TRAIN_ONLY_OOF_AND_RISK_SHA=PASS',flush=True)
 return allfiles


def load_graph(parts,assets):
 graphs,_=dgl.load_graphs(str(SOURCE))
 assert len(graphs)==1
 g=graphs[0]
 assert g.is_homogeneous and g.num_nodes()==39357 and g.num_edges()==42445086
 onehot=g.ndata['label']
 assert tuple(onehot.shape)==(39357,2)
 # Training targets are solely from the OOF assets; read only validation labels.
 vy=onehot[torch.from_numpy(parts['val'].copy())].argmax(1).numpy().astype(np.int64,copy=True)
 assert vy.shape==(7872,) and int(vy.sum())==361
 assert set(np.unique(vy))=={0,1}
 for key in list(g.ndata):del g.ndata[key]
 for key in list(g.edata):del g.edata[key]
 del onehot,graphs
 assert not g.ndata and not g.edata
 assert np.array_equal(assets['TR40'][0]['features'],assets['TR30'][0]['features'])
 assert np.array_equal(assets['TR40'][0]['features'],assets['TR20'][0]['features'])
 assert np.array_equal(assets['TR40'][0]['features'],assets['TR10'][0]['features'])
 print('RGTAN_T-Finance_I2_FULL_42M_ORIGINAL_GRAPH_LABEL_FIELDS_REMOVED=PASS',flush=True)
 return g,vy


def threshold_macro_f1(y,scores):
 y=np.asarray(y,dtype=np.int8);scores=np.asarray(scores,dtype=np.float64)
 assert len(y)==len(scores) and np.isfinite(scores).all() and set(np.unique(y))=={0,1}
 order=np.argsort(-scores,kind='mergesort')
 ss=scores[order]; yy=y[order]
 tp=np.cumsum(yy,dtype=np.int64);n=len(y);positive=int(yy.sum());negative=n-positive
 ends=np.r_[np.flatnonzero(np.diff(ss)!=0),n-1]
 tp=tp[ends];fp=ends+1-tp;fn=positive-tp;tn=negative-fp
 fm=(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))*.5
 normal=(2*negative/(2*negative+positive))*.5
 choice=int(np.argmax(np.r_[normal,fm]))
 if choice==0:return float(np.nextafter(ss[0],np.inf)),float(normal)
 return float(ss[ends[choice-1]]),float(fm[choice-1])


def paths(ratio,seed):
 folder=OUT/f'RGTAN_T-Finance_{ratio}_Seed{seed}'
 return folder,folder/f'RGTAN_T-Finance_{ratio}_Seed{seed}_BestModel.pt',\
  folder/f'RGTAN_T-Finance_{ratio}_Seed{seed}_FrozenCandidateTraining.json',\
  folder/f'RGTAN_T-Finance_{ratio}_Seed{seed}_BestValidationScores.npy'


def run_one(T,g,val_y,ratio,seed,asset,cfg,freeze_sha):
 fields,artifact_sha=asset
 folder,checkpoint,done,valfile=paths(ratio,seed);folder.mkdir(parents=True,exist_ok=True)
 if done.exists():
  r=json.loads(done.read_text())
  assert r['status']=='RGTAN_TFINANCE_CANDIDATE_FINAL_TRAINING_FROZEN'
  assert r['ratio']==ratio and r['seed']==seed and r['config']==cfg
  assert r['source_sha256']==DATA_SHA and r['crossfit_sha256']==artifact_sha
  assert r['frozen_winner_sha256']==freeze_sha and r['training_script_sha256']==sha(__file__)
  assert r['checkpoint_sha256']==sha(checkpoint) and r['validation_scores_sha256']==sha(valfile)
  assert r['candidate_only'] is True and r['test_scored'] is False
  print(f'RGTAN_T-Finance_{ratio}_Seed{seed}_FROZEN_CANDIDATE_REUSED=PASS',flush=True)
  return r
 engine=T.Engine(g,fields,fields['train_labels'],val_y,cfg,seed=seed)
 # T.Engine.one_epoch hardcodes 15,742 for TR40; preserve all other
 # author-compatible operations and generalize only the train set count.
 assert engine.tr.shape==(COUNTS[ratio],) and len(engine.targets)==5
 start=time.perf_counter();best=-1.;best_ep=0;stale=0;history=[]
 torch.cuda.reset_peak_memory_stats()
 for epoch in range(1,MAX_EPOCHS+1):
  engine.model.train();total=0.;count=0
  order=np.random.RandomState(seed+epoch).permutation(5)
  for k in order:
   dgl.seed(seed+epoch*17+int(k))
   for batch in engine.train_loaders[k]:
    logits,outs=engine.forward(batch,'train',k)
    target=torch.as_tensor([engine.targets[k][int(i)] for i in outs],dtype=torch.long,device='cuda:0')
    loss=F.cross_entropy(logits,target)
    assert bool(torch.isfinite(loss))
    engine.opt.zero_grad(set_to_none=True);loss.backward();engine.opt.step()
    engine.steps+=1;total+=float(loss.item())*len(outs);count+=len(outs)
  assert count==COUNTS[ratio],(ratio,count,COUNTS[ratio])
  val_ap,val_scores=engine.validation()
  history.append({'epoch':epoch,'train_loss':total/count,'val_auprc':val_ap})
  if val_ap>best+1e-9:
   best=val_ap;best_ep=epoch;stale=0
   save_checkpoint(checkpoint,{'model':engine.model.state_dict(),'config':cfg,
     'seed':seed,'ratio':ratio,'best_epoch':epoch,'val_auprc':best,
     'source_sha256':DATA_SHA,'crossfit_sha256':artifact_sha,
     'frozen_winner_sha256':freeze_sha,'author_commit':PIN,'fanouts':FANOUTS,
     'candidate_only':True,'benchmark_supervisor_approval_required':True,'test_scored':False})
   tempfile=Path(str(valfile)+'.partial')
   with tempfile.open('wb') as f:np.save(f,val_scores);f.flush();os.fsync(f.fileno())
   os.replace(tempfile,valfile)
  else:stale+=1
  if epoch<=2 or epoch%5==0 or stale>=PATIENCE:
   print(f'RGTAN_T-Finance_{ratio}_Seed{seed} epoch={epoch} loss={total/count:.5f} '
     f'val_AUPRC={val_ap:.6f} best={best:.6f} stale={stale}',flush=True)
  if stale>=PATIENCE:break
 assert best>0 and checkpoint.exists() and valfile.exists()
 frozen_scores=np.load(valfile,allow_pickle=False)
 threshold,macro=threshold_macro_f1(val_y,frozen_scores)
 observed=float(f1_score(val_y,(frozen_scores>=threshold).astype(int),average='macro',zero_division=0))
 assert abs(observed-macro)<1e-10,(observed,macro)
 r={'status':'RGTAN_TFINANCE_CANDIDATE_FINAL_TRAINING_FROZEN','model':MODEL,'dataset':DATASET,
  'ratio':ratio,'seed':seed,'config':cfg,'batch':1024,'max_epochs':MAX_EPOCHS,
  'patience':PATIENCE,'best_epoch':best_ep,'epochs_completed':epoch,
  'best_val_auprc':best,'best_val_macro_f1':macro,'frozen_val_macro_f1_threshold':threshold,
  'train_count':COUNTS[ratio],'train_positive_count':POSITIVES[ratio],
  'val_count':COUNTS['val'],'val_positive_count':POSITIVES['val'],
  'optimizer_steps':engine.steps,'train_seconds':round(time.perf_counter()-start,3),
  'gpu_peak_allocated_gib':round(torch.cuda.max_memory_allocated()/1024**3,5),
  'source_sha256':DATA_SHA,'split_sha256':SPLIT_SHA,'crossfit_sha256':artifact_sha,
  'frozen_winner_sha256':freeze_sha,'checkpoint_sha256':sha(checkpoint),
  'validation_scores_sha256':sha(valfile),'training_script_sha256':sha(__file__),
  'author_commit':PIN,'fanouts':FANOUTS,'original_full_neighbor_sampling_replaced':True,
  'benchmark_supervisor_approval_required':True,'candidate_only':True,
  'test_labels_accessed':False,'test_scored':False,'test_evaluation_authorized':False}
 atomic_json(folder/f'RGTAN_T-Finance_{ratio}_Seed{seed}_CandidateTrainingHistory.json',history)
 atomic_json(done,r)
 print(f'RGTAN_T-Finance_{ratio}_Seed{seed}_FROZEN_CANDIDATE_VAL=PASS '
       f'AUPRC={best:.6f} macroF1={macro:.6f}',flush=True)
 del engine;torch.cuda.empty_cache()
 return r


def train(parts,freeze,T):
 assets=read_assets(parts)
 g,vy=load_graph(parts,assets)
 completed=[];fhash=sha(FREEZE)
 for ratio in RATIOS:
  for seed in SEEDS:
   row=run_one(T,g,vy,ratio,seed,assets[ratio],freeze['configuration'],fhash)
   completed.append(row)
 assert len(completed)==12
 report={'status':'RGTAN_T-Finance_I2_ALL_12_CANDIDATE_TRAININGS_FROZEN',
  'model':MODEL,'dataset':DATASET,'candidate_only':True,'ratios':list(RATIOS),'seeds':list(SEEDS),
  'candidate_training_count':12,'frozen_winner_sha256':fhash,
  'author_commit':PIN,'source_sha256':DATA_SHA,'split_sha256':SPLIT_SHA,
  'fanouts':FANOUTS,'original_full_neighbor_sampling_replaced':True,
  'benchmark_supervisor_approval_required':True,'test_labels_accessed':False,'test_scored':False,
  'validation_metrics_summary':[]}
 for ratio in RATIOS:
  rows=[r for r in completed if r['ratio']==ratio]
  vals=np.array([r['best_val_auprc'] for r in rows]);f1s=np.array([r['best_val_macro_f1'] for r in rows])
  report['validation_metrics_summary'].append({'ratio':ratio,
   'AUPRC_mean':float(vals.mean()),'AUPRC_sample_SD':float(vals.std(ddof=1)),
   'macro_F1_mean':float(f1s.mean()),'macro_F1_sample_SD':float(f1s.std(ddof=1))})
 path=OUT/'RGTAN_T-Finance_I2_AllRatios_CandidateValidationSummary.json'
 if path.exists():assert json.loads(path.read_text())==report
 else:atomic_json(path,report,exclusive=True)
 for row in report['validation_metrics_summary']:
  print(f'RGTAN_T-Finance_{row["ratio"]}_CANDIDATE_VALIDATION_AUPRC_MEAN='
    f'{row["AUPRC_mean"]:.6f} sampleSD={row["AUPRC_sample_SD"]:.6f}',flush=True)
 print('RGTAN_T-Finance_I2_ALL_12_CANDIDATE_FINAL_TRAININGS=PASS TEST_SCORED=NO APPROVAL_REQUIRED=YES',flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['freeze','train'],required=True)
 a=p.parse_args();sys.path.insert(0,str(ROOT/'repos/antifraud'))
 T=import_stage4();parts=preflight()
 frozen=freeze_winner(T)
 if a.mode=='freeze':return
 train(parts,frozen,T)

if __name__=='__main__':main()
