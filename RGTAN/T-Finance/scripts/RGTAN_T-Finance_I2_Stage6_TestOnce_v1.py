#!/usr/bin/env python3
"""Complete RGTAN x T-Finance ADAPTED [10,10] benchmark by one-time test scoring.

This script never retunes, retrains, changes a validation threshold or uses test
labels for configuration selection. Stage5's 12 immutable candidate checkpoints
are checked BEFORE any test partition label is read. All 12 score-once claims
are exclusive; incomplete claims halt rather than silently rerun a heldout test.

Sampling deviation from original full neighbors remains explicit: completed
adapted RGTAN result != author-identical full-neighbor baseline. User requested
completion; supervisor approval is NOT assumed or asserted.
"""
from __future__ import annotations
import hashlib, importlib.util, json, os, random, socket, subprocess, sys, time
from pathlib import Path
import dgl
import numpy as np
import pandas as pd
import torch
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score

ROOT=Path('/workspace/comp8851')
STAGE5=ROOT/'pipeline/RGTAN_T-Finance_I2_Stage5_FrozenCandidateFinals_v1.py'
STAGE4=ROOT/'pipeline/RGTAN_T-Finance_TR40_Tune12_Candidate_v1.py'
SOURCE=ROOT/'assets/tfinance/canonical_candidate/tfinance_canonical.dgl'
SPLIT=Path('/workspace/uploads/ass2_split_candidates/T-Finance/tfinance_seed2_nested_splits.npz')
FINALS=ROOT/'results/rgtan/tfinance/RGTAN_T-Finance_I2_12CandidateFinals_Fanout10x10_v1'
TEST_DIR=ROOT/'results/rgtan/tfinance/RGTAN_T-Finance_I2_Stage6_AdaptedHeldout_v1'
DATA_SHA='b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700'
SPLIT_SHA='de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1'
AUTHOR_SHA='65e583f03ac5c5983e49cc35179fe7a65908b9c1'
N=39357; EDGES=42445086; TEST_COUNT=15743; TEST_POSITIVES=722
RATIOS=('TR40','TR30','TR20','TR10'); SEEDS=(2,42,72); FANOUTS=[10,10]; BATCH=1024
NAMES=('degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat')
DEVICE='cuda:0'


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()


def write_json(path,object,exclusive=False):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if exclusive:
  with path.open('x') as f:json.dump(object,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
 else:
  tmp=Path(str(path)+'.partial')
  with tmp.open('x') as f:json.dump(object,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)


def save_npy_once(path,arr):
 path=Path(path);partial=Path(str(path)+'.partial')
 assert not path.exists() and not partial.exists(),f'STOP: existing score or partial file: {path}'
 with partial.open('xb') as f:np.save(f,arr,allow_pickle=False);f.flush();os.fsync(f.fileno())
 os.replace(partial,path)


def metrics(y,sc,threshold):
 y=np.asarray(y,dtype=np.int64);sc=np.asarray(sc,dtype=np.float64)
 assert y.ndim==sc.ndim==1 and len(y)==len(sc)==TEST_COUNT
 assert np.isfinite(sc).all() and np.isfinite(threshold)
 assert set(np.unique(y))=={0,1} and int(y.sum())==TEST_POSITIVES
 p=(sc>=threshold).astype(np.int64)
 return {'auprc':float(average_precision_score(y,sc)),
  'auroc':float(roc_auc_score(y,sc)),
  'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),
  'fraud_f1':float(f1_score(y,p,pos_label=1,zero_division=0)),
  'fraud_precision':float(precision_score(y,p,pos_label=1,zero_division=0)),
  'fraud_recall':float(recall_score(y,p,pos_label=1,zero_division=0)),
  'predicted_fraud_count':int(p.sum()),'true_fraud_count':int(y.sum())}


def import_stage5():
 assert STAGE5.is_file() and STAGE4.is_file(),'STOP: Missing original frozen training/tuner source'
 sys.path.insert(0,str(ROOT/'repos/antifraud'))
 spec=importlib.util.spec_from_file_location('RGTAN_TFinance_Stage5',STAGE5)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert m.DATA_SHA==DATA_SHA and m.SPLIT_SHA==SPLIT_SHA and m.PIN==AUTHOR_SHA
 assert m.RATIOS==RATIOS and m.SEEDS==SEEDS and m.FANOUTS==FANOUTS
 assert m.OUT==FINALS and m.FREEZE.is_file()
 return m


def preflight(m):
 assert socket.gethostname()=='f24d1bb2672d','STOP: I2 ONLY'
 assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1'
 assert torch.cuda.is_available() and 'A6000' in torch.cuda.get_device_name(0)
 assert subprocess.check_output(['git','-C',str(ROOT/'repos/antifraud'),'rev-parse','HEAD'],text=True).strip()==AUTHOR_SHA
 assert sha(SOURCE)==DATA_SHA and sha(SPLIT)==SPLIT_SHA
 with np.load(SPLIT,allow_pickle=False) as z:
  assert int(z['seed'].item())==2
  parts={k:np.asarray(z[k],dtype=np.int64).copy() for k in (*RATIOS,'val','test')}
 assert len(parts['test'])==TEST_COUNT and len(np.unique(parts['test']))==TEST_COUNT
 assert min(parts['test'])>=0 and max(parts['test'])<N
 for a,b in [('TR40','val'),('TR40','test'),('val','test')]:
  assert np.intersect1d(parts[a],parts[b]).size==0
 for a,b in [('TR10','TR20'),('TR20','TR30'),('TR30','TR40')]:
  assert np.isin(parts[a],parts[b]).all()
 assert int(sum(len(parts[k]) for k in ('TR40','val','test')))==N
 print('RGTAN_T-Finance_I2_STAGE6_FROZEN_SOURCE_SPLIT_ENV=PASS',flush=True)
 return parts


def verify_all_12(m,parts):
 summary=FINALS/'RGTAN_T-Finance_I2_AllRatios_CandidateValidationSummary.json'
 assert summary.is_file(),'STOP: Stage5 summary missing'
 s=json.loads(summary.read_text())
 assert s['status']=='RGTAN_T-Finance_I2_ALL_12_CANDIDATE_TRAININGS_FROZEN'
 assert s['candidate_training_count']==12 and s['test_scored'] is False
 assert s['candidate_only'] is True and s['fanouts']==FANOUTS
 assert s['original_full_neighbor_sampling_replaced'] is True
 assert s['source_sha256']==DATA_SHA and s['split_sha256']==SPLIT_SHA
 assert s['frozen_winner_sha256']==sha(m.FREEZE)
 w=json.loads(m.FREEZE.read_text())
 assert w['status']=='RGTAN_T-Finance_I2_TR40_VALIDATION_WINNER_FROZEN_CANDIDATE'
 assert w['tuning_winner_sha256']==sha(m.WINNER) and w['fanouts']==FANOUTS
 assert w['original_full_neighbor_sampling_replaced'] is True
 cfg=w['configuration']; assert cfg in m.import_stage4().GRID
 assert w['candidate_only'] is True and w['test_scored'] is False
 verified=[]
 for ratio in RATIOS:
  artifact=ROOT/f'assets/rgtan/tfinance/RGTAN_T-Finance_{ratio}_ExactRiskCrossfit_v1.npz'
  meta=artifact.with_suffix('.json')
  assert artifact.is_file() and meta.is_file(),f'STOP: Missing {ratio} risk artifact'
  rmeta=json.loads(meta.read_text())
  riskhash=sha(artifact)
  assert rmeta['artifact_sha256']==riskhash
  assert rmeta['source_sha256']==DATA_SHA and rmeta['split_sha256']==SPLIT_SHA
  assert rmeta['test_labels_used'] is False and rmeta['validation_labels_used'] is False
  with np.load(artifact,allow_pickle=False) as z:
   assert np.array_equal(z['train_ids'],parts[ratio]);assert np.array_equal(z['val_ids'],parts['val'])
   assert np.array_equal(z['risk_feature_names'],np.asarray(NAMES))
   assert z['features'].shape==(N,10)
   assert np.all(z['label_context_5'][parts['test']]==2)
   assert z['risk_view_5'].shape==(N,6)
  for seed in SEEDS:
   folder,ckpt,recfile,valfile=m.paths(ratio,seed)
   assert all(p.is_file() for p in (ckpt,recfile,valfile)),f'STOP: Frozen assets absent for {ratio} seed {seed}'
   r=json.loads(recfile.read_text())
   assert r['status']=='RGTAN_TFINANCE_CANDIDATE_FINAL_TRAINING_FROZEN'
   assert r['ratio']==ratio and r['seed']==seed and r['config']==cfg
   assert r['test_scored'] is False and r['test_labels_accessed'] is False
   assert r['source_sha256']==DATA_SHA and r['split_sha256']==SPLIT_SHA
   assert r['fanouts']==FANOUTS and r['original_full_neighbor_sampling_replaced'] is True
   assert r['crossfit_sha256']==riskhash and r['frozen_winner_sha256']==sha(m.FREEZE)
   assert r['training_script_sha256']==sha(STAGE5)
   assert r['checkpoint_sha256']==sha(ckpt) and r['validation_scores_sha256']==sha(valfile)
   assert np.load(valfile,allow_pickle=False).shape==(len(parts['val']),)
   assert 0<float(r['frozen_val_macro_f1_threshold'])<=1.000001
   # Explicitly examine all saved model metadata and weights BEFORE test access.
   saved=torch.load(ckpt,map_location='cpu')
   assert saved['ratio']==ratio and saved['seed']==seed and saved['config']==cfg
   assert saved['source_sha256']==DATA_SHA and saved['crossfit_sha256']==riskhash
   assert saved['frozen_winner_sha256']==r['frozen_winner_sha256']
   assert saved['fanouts']==FANOUTS and saved['test_scored'] is False
   assert len(saved['model'])>0
   del saved
   verified.append({'ratio':ratio,'seed':seed,'frozen':r,'crossfit_sha256':riskhash,
    'checkpoint':str(ckpt),'record':str(recfile), 'frozen_record_sha256':sha(recfile)})
 assert len(verified)==12
 print('RGTAN_T-Finance_I2_ALL_12_FROZEN_CHECKPOINTS_AND_VAL_THRESHOLDS_VERIFIED=PASS',flush=True)
 return verified,cfg


def build_model(cfg,features,seed):
 from methods.rgtan.rgtan_model import RGTAN
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 initial={n:torch.zeros(N,device=DEVICE) for n in NAMES}
 model=RGTAN(in_feats=10,hidden_dim=cfg['hidden_dim'],n_layers=2,n_classes=2,
  heads=[4,4],activation=torch.nn.PReLU(),drop=cfg['drop'],device=DEVICE,
  gated=True,ref_df=pd.DataFrame(features),cat_features={},neigh_features=initial,nei_att_head=2).to(DEVICE)
 return model


def read_graph_and_test_truth(parts):
 # ENTERED only after every frozen training/checkpoint/hash has been validated.
 graphs,_=dgl.load_graphs(str(SOURCE))
 assert len(graphs)==1
 graph=graphs[0]
 assert graph.is_homogeneous and graph.num_nodes()==N and graph.num_edges()==EDGES
 onehot=graph.ndata['label']
 assert tuple(onehot.shape)==(N,2)
 assert bool((((onehot==0)|(onehot==1)).all()) and (onehot.sum(1)==1).all())
 test_y=onehot[torch.from_numpy(parts['test'].copy())].argmax(1).numpy().astype(np.int64,copy=True)
 assert len(test_y)==TEST_COUNT and int(test_y.sum())==TEST_POSITIVES
 # Model will never see source node/edge labels or extra graph feature tensors.
 for k in list(graph.ndata):del graph.ndata[k]
 for k in list(graph.edata):del graph.edata[k]
 assert not graph.ndata and not graph.edata
 del graphs,onehot
 print('RGTAN_T-Finance_I2_HELDOUT_LABELS_READ_AFTER_12_FROZEN_VALIDATIONS=PASS',flush=True)
 print('RGTAN_T-Finance_I2_42M_EDGE_GRAPH_MODEL_LABEL_FREE=PASS',flush=True)
 return graph,test_y


def infer_one(graph,parts,cfg,record,fields):
 seed=record['seed'];ratio=record['ratio'];r=record['frozen']
 model=build_model(cfg,fields['features'],seed)
 checkpoint=torch.load(record['checkpoint'],map_location=DEVICE)
 assert checkpoint['source_sha256']==DATA_SHA
 assert checkpoint['crossfit_sha256']==record['crossfit_sha256']
 assert checkpoint['frozen_winner_sha256']==r['frozen_winner_sha256']
 assert checkpoint['config']==cfg and checkpoint['seed']==seed and checkpoint['ratio']==ratio
 model.load_state_dict(checkpoint['model'],strict=True);model.eval()
 del checkpoint
 test_ids=parts['test']; lookup={int(x):j for j,x in enumerate(test_ids)}
 # Same deterministic evaluation fanout sampler as Stage4 validation,
 # not repeated random ensembles, and not the original full-neighbor model.
 sampler=MultiLayerNeighborSampler(FANOUTS)
 loader=NodeDataLoader(graph,torch.from_numpy(test_ids.copy()),sampler,batch_size=BATCH,
  shuffle=False,drop_last=False,num_workers=0)
 dgl.seed(90210)
 random.seed(90210);np.random.seed(90210);torch.manual_seed(90210);torch.cuda.manual_seed_all(90210)
 scores=np.full(len(test_ids),np.nan,dtype=np.float32)
 ctx=fields['label_context_5'];risk=fields['risk_view_5'];feats=fields['features']
 assert np.all(ctx[test_ids]==2)
 torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 with torch.no_grad():
  for in_nodes,out_nodes,blocks in loader:
   a=in_nodes.numpy();b=out_nodes.numpy()
   assert np.all(ctx[b]==2)
   xx=torch.from_numpy(feats[a].astype(np.float32,copy=False).copy()).to(DEVICE)
   cc=torch.from_numpy(ctx[a].astype(np.int64,copy=False).copy()).to(DEVICE)
   rd={name:torch.from_numpy(risk[a,j].copy()).to(DEVICE) for j,name in enumerate(NAMES)}
   logits=model([blk.to(DEVICE) for blk in blocks],xx,cc,{},rd)
   assert logits.shape==(len(b),2) and bool(torch.isfinite(logits).all())
   probs=torch.softmax(logits,dim=1)[:,1].cpu().numpy()
   for i,node in enumerate(b):
    j=lookup[int(node)]
    assert np.isnan(scores[j]), 'STOP: duplicate test node scored'
    scores[j]=probs[i]
 assert np.isfinite(scores).all()
 sec=time.perf_counter()-start; gib=torch.cuda.max_memory_allocated()/1024**3
 del model,loader;torch.cuda.empty_cache()
 return scores,sec,gib


def evaluate(parts,verified,cfg):
 # Precheck saved test-state before reading target labels: each model either
 # verifiably complete or entirely unclaimed. Partial claim means STOP.
 for item in verified:
  ratio=item['ratio'];seed=item['seed']
  folder=TEST_DIR/f'RGTAN_T-Finance_{ratio}_Seed{seed}'
  base=f'RGTAN_T-Finance_{ratio}_Seed{seed}'
  claim=folder/f'{base}_ONE_TIME_TEST_CLAIM.json'
  result=folder/f'{base}_FinalAdaptedTestMetrics.json'
  scores=folder/f'{base}_TestScores.npy'
  item.update(folder=folder,claim=claim,result=result,scores=scores)
  if result.is_file():
   assert claim.is_file() and scores.is_file(),f'STOP: corrupted completed result: {base}'
   rec=json.loads(result.read_text())
   assert rec['status']=='RGTAN_TFINANCE_ADAPTED_HELDOUT_SCORED_ONCE'
   assert rec['checkpoint_sha256']==item['frozen']['checkpoint_sha256']
   assert rec['test_scores_sha256']==sha(scores)
   assert rec['frozen_training_record_sha256']==item['frozen_record_sha256']
   assert rec['fanouts']==FANOUTS and rec['test_scored_once'] is True
   item['complete']=True
  else:
   assert not any(p.exists() for p in (claim,scores,Path(str(scores)+'.partial'),Path(str(result)+'.partial'))),\
    f'STOP: Incomplete one-time test scoring, inspect and recover manually: {base}'
   item['complete']=False
 print('RGTAN_T-Finance_I2_SINGLE_PASS_CLAIM_RESUME_GUARDS=PASS',flush=True)
 if all(item['complete'] for item in verified):
  print('RGTAN_T-Finance_I2_ALL_12_TEST_RECORDS_VERIFIED_REUSE=PASS',flush=True)
  aggregate(verified);return
 graph,test_y=read_graph_and_test_truth(parts)
 for ratio in RATIOS:
  pending=[item for item in verified if item['ratio']==ratio and not item['complete']]
  if not pending:continue
  crossfit=ROOT/f'assets/rgtan/tfinance/RGTAN_T-Finance_{ratio}_ExactRiskCrossfit_v1.npz'
  with np.load(crossfit,allow_pickle=False) as f:fields={k:f[k] for k in ('features','risk_view_5','label_context_5')}
  assert fields['features'].shape==(N,10) and fields['risk_view_5'].shape==(N,6)
  assert fields['label_context_5'].shape==(N,) and np.all(fields['label_context_5'][parts['test']]==2)
  for item in pending:
   ratio=item['ratio'];seed=item['seed'];r=item['frozen']
   print(f'RGTAN_T-Finance_{ratio}_Seed{seed}_VERIFYING_MODEL_AND_EXCLUSIVE_TEST_CLAIM',flush=True)
   # Verify model architecture and checkpoint restore BEFORE claiming test use.
   # Validation was already frozen; no new thresholds or model selection.
   model=build_model(cfg,fields['features'],seed)
   saved=torch.load(item['checkpoint'],map_location='cpu')
   model.load_state_dict(saved['model'],strict=True)
   del model,saved;torch.cuda.empty_cache()
   write_json(item['claim'],{
    'model':'RGTAN','dataset':'T-Finance','ratio':ratio,'seed':seed,
    'checkpoint_sha256':r['checkpoint_sha256'],'validation_threshold':r['frozen_val_macro_f1_threshold'],
    'fanouts':FANOUTS,'sampling_method':'bounded_neighbor_sampling_10x10',
    'method_deviation_reported':True,'user_directed_completion':True,
    'one_time_test_scoring_claimed':True},exclusive=True)
   probs,secs,gib=infer_one(graph,parts,cfg,item,fields)
   met=metrics(test_y,probs,r['frozen_val_macro_f1_threshold'])
   save_npy_once(item['scores'],probs)
   result={
    'status':'RGTAN_TFINANCE_ADAPTED_HELDOUT_SCORED_ONCE','model':'RGTAN','dataset':'T-Finance',
    'ratio':ratio,'seed':seed,'test_count':TEST_COUNT,'test_positive_count':TEST_POSITIVES,
    'validation_auprc':r['best_val_auprc'],'validation_macro_f1':r['best_val_macro_f1'],
    'decision_threshold_frozen_validation_macro_f1':r['frozen_val_macro_f1_threshold'],
    'test_scored_once':True,'test_scores_sha256':sha(item['scores']),
    'frozen_training_record_sha256':item['frozen_record_sha256'],
    'checkpoint_sha256':r['checkpoint_sha256'],'crossfit_sha256':item['crossfit_sha256'],
    'source_sha256':DATA_SHA,'split_sha256':SPLIT_SHA,'author_commit':AUTHOR_SHA,
    'test_scoring_seconds':round(secs,3),'gpu_peak_allocated_gib':round(gib,5),
    'sampler':'MultiLayerNeighborSampler','fanouts':FANOUTS,'evaluation_sampler_seed':90210,
    'original_full_neighbor_sampling_replaced':True,'author_identical_benchmark':False,
    'supervisor_approval_documented':False,'user_directed_test_completion':True,
    'metrics':met,
   }
   write_json(item['result'],result,exclusive=True)
   item['complete']=True
   print(f'RGTAN_T-Finance_{ratio}_Seed{seed}_ONE_TIME_ADAPTED_TEST_SCORE=PASS '
    f'AUPRC={met["auprc"]:.6f} AUROC={met["auroc"]:.6f} macroF1={met["macro_f1"]:.6f}',flush=True)
 aggregate(verified)


def aggregate(verified):
 assert len(verified)==12 and all(i['result'].is_file() for i in verified)
 rows=[]; keynames=('auprc','auroc','macro_f1','fraud_f1','fraud_precision','fraud_recall')
 for ratio in RATIOS:
  run=[]
  for seed in SEEDS:
   item=next(i for i in verified if i['ratio']==ratio and i['seed']==seed)
   m=json.loads(item['result'].read_text())
   assert m['test_scores_sha256']==sha(item['scores'])
   run.append(m['metrics'])
  row={'ratio':ratio,'seeds':list(SEEDS)}
  for name in keynames:
   vals=np.array([x[name] for x in run],dtype=np.float64)
   row[name+'_mean']=float(vals.mean())
   row[name+'_sample_sd']=float(vals.std(ddof=1))
  rows.append(row)
 summary={
  'status':'RGTAN_T-Finance_I2_ALL_12_ADAPTED_FINAL_TESTS_COMPLETE',
  'model':'RGTAN','dataset':'T-Finance','ratios':list(RATIOS),'seeds':list(SEEDS),
  'run_count':12,'test_scores_per_run':1,'test_sample_count':TEST_COUNT,
  'summary_rows':rows,'standard_deviation':'sample ddof=1',
  'source_sha256':DATA_SHA,'split_sha256':SPLIT_SHA,'author_commit':AUTHOR_SHA,
  'sampler':'MultiLayerNeighborSampler','fanouts':FANOUTS,
  'full_original_graph_edges':EDGES,'original_full_neighbor_sampling_replaced':True,
  'author_identical_benchmark':False,'supervisor_approval_documented':False,
  'user_directed_test_completion':True,'test_scored_once_per_model':True,
 }
 path=TEST_DIR/'RGTAN_T-Finance_I2_AllRatios_AdaptedFinalTestSummary.json'
 if path.exists():assert json.loads(path.read_text())==summary,'STOP: Summary conflicts with frozen results'
 else:write_json(path,summary,exclusive=True)
 for row in rows:
  print(f'RGTAN_T-Finance_{row["ratio"]}_ADAPTED_FINAL_TEST_AUPRC_MEAN={row["auprc_mean"]:.6f} '
   f'sampleSD={row["auprc_sample_sd"]:.6f} macroF1_MEAN={row["macro_f1_mean"]:.6f}',flush=True)
 print('RGTAN_T-Finance_I2_STAGE6_ALL_12_ADAPTED_FINAL_TESTS_COMPLETE=PASS',flush=True)


def main():
 m=import_stage5();parts=preflight(m)
 verified,cfg=verify_all_12(m,parts)
 evaluate(parts,verified,cfg)

if __name__=='__main__':main()
