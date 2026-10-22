#!/usr/bin/env python3
"""EOGFD 57212396 combined Stage-6: offline result audit, private backup, public evidence export.
NO model forwards, checkpoint loading, training or new test inference. Read-only originals.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, socket, subprocess, sys, zipfile
from pathlib import Path, PurePosixPath
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, brier_score_loss, confusion_matrix)

BASE=Path('/workspace/comp8851')
W=Path('/workspace')
REPO=BASE/'repos/EOGFD'
BRANCH='Pathik-ass3'
PIN='a4726f173bc8c923109ddcb2786f4d9c5206ae7f'
DS=('FDCompCN','Elliptic')
RATIOS=('TR40','TR30','TR20','TR10')
SEEDS=(2,42,72)
PUB=W/'EOGFD_57212396_GITHUB_PUBLICATION.zip'
PRIVATE=W/'EOGFD_57212396_PRIVATE_FINAL_CHECKPOINTS.zip'
DATA={
 'FDCompCN':{'source':{'comp.dgl':'e252b9a6b619b28a7bf6d9f5b16aacc232d43207d87ac26cd5160fb0d8a98baa'},'split':'5cd7084342f4af48e9b82828dcac0e97879314cd5c9c010642b30c89e7c95168','winner':1,'winner_ap':0.215919,'testn':2127,'counts':{'TR40':2126,'TR30':1594,'TR20':1062,'TR10':531,'val':1064,'test':2127}},
 'Elliptic':{'source':{
 'elliptic_txs_features.csv':'fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0',
 'elliptic_txs_classes.csv':'93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493',
 'elliptic_txs_edgelist.csv':'a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e'},
 'split':'1e7963e9d09935fb33e0df74cfb786cab826d1b95ae10b9592dbc240b48ffdbc','winner':10,'winner_ap':0.708897,'testn':18949,'counts':{'TR40':18889,'TR30':13670,'TR20':9553,'TR10':4543,'val':8726,'test':18949}}
}
SOURCES={'FDCompCN':BASE/'assets/fdcompcn/source','Elliptic':BASE/'assets/elliptic/source'}
SPLITS={'FDCompCN':W/'uploads/ass2_split_candidates/FDCompCN/fdcompcn_seed2_nested_splits.npz',
        'Elliptic':W/'uploads/ass2_split_candidates/Elliptic/elliptic_seed2_nested_splits.npz'}
ALLOWED_PUBLIC_EXT={'.json','.csv','.md','.txt','.log','.py','.sh','.npy'}


def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def checked_json(p):
 assert p.is_file(),('MISSING',str(p))
 return json.loads(p.read_text())

def jbytes(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False,ensure_ascii=False)+'\n').encode()

def assert_close(a,b,field,tol=5e-10):
 assert math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=tol,(field,a,b,tol)

def metric(y,s,threshold):
 y=np.asarray(y,dtype=np.int64);s=np.asarray(s,dtype=np.float64)
 assert len(y)==len(s) and len(y)>0 and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
 assert set(np.unique(y).tolist())=={0,1}
 pr=np.asarray(s>threshold,dtype=np.int64)
 tn,fp,fn,tp=map(int,confusion_matrix(y,pr,labels=[0,1]).ravel())
 return dict(auprc=float(average_precision_score(y,s)),auroc=float(roc_auc_score(y,s)),
  macro_f1=float(f1_score(y,pr,average='macro',zero_division=0)),
  fraud_f1=float(f1_score(y,pr,average='binary',pos_label=1,zero_division=0)),
  fraud_precision=float(precision_score(y,pr,pos_label=1,zero_division=0)),
  fraud_recall=float(recall_score(y,pr,pos_label=1,zero_division=0)),
  accuracy=float(accuracy_score(y,pr)),balanced_accuracy=float(balanced_accuracy_score(y,pr)),
  mcc=float(matthews_corrcoef(y,pr)),brier_score=float(brier_score_loss(y,s)),
  tp=tp,tn=tn,fp=fp,fn=fn)


def cpyfile(pubfiles,rel,source):
 assert isinstance(rel,str) and rel.startswith('EOGFD/')
 pos=PurePosixPath(rel)
 assert not pos.is_absolute() and '..' not in pos.parts
 assert pos.suffix in ALLOWED_PUBLIC_EXT,('FORBIDDEN_PUBLIC_EXTENSION',rel)
 assert not any(s in rel.lower() for s in ('/source/','/.git/','/private/','/venv/'))
 p=Path(source);assert p.is_file(),('MISSING_PUBLISH_FILE',str(p))
 assert p.stat().st_size<24*1024*1024,('PUBLIC_FILE_TOO_LARGE',str(p))
 assert rel not in pubfiles,('DUPLICATE_PUBLIC_PATH',rel)
 pubfiles[rel]=p.read_bytes()

def put(pubfiles,rel,data):
 assert rel not in pubfiles
 assert PurePosixPath(rel).suffix in ALLOWED_PUBLIC_EXT
 pubfiles[rel]=data if isinstance(data,bytes) else data.encode()

def verify_canonical(ds):
 cfg=DATA[ds]
 for name,h in cfg['source'].items():
  f=SOURCES[ds]/name
  assert f.is_file() and sha(f)==h,('CANONICAL_SOURCE_CHANGED',ds,name)
 assert SPLITS[ds].is_file() and sha(SPLITS[ds])==cfg['split'],('FROZEN_SPLIT_CHANGED',ds)
 with np.load(SPLITS[ds],allow_pickle=False) as z:
  ids={r:np.asarray(z[r],dtype=np.int64).copy() for r in (*RATIOS,'val','test')}
  for r,count in cfg['counts'].items():assert len(ids[r])==count and len(np.unique(ids[r]))==count,(ds,r)
  for a,b in (('TR10','TR20'),('TR20','TR30'),('TR30','TR40')): assert np.isin(ids[a],ids[b]).all()
  for a,b in (('TR40','val'),('TR40','test'),('val','test')):assert not np.intersect1d(ids[a],ids[b]).size
  seed_key='seed' if ds=='FDCompCN' else 'split_seed'
  assert int(np.asarray(z[seed_key]).reshape(-1)[0])==2
 if ds=='FDCompCN':
  from dgl.data.utils import load_graphs
  gg,_=load_graphs(str(SOURCES[ds]/'comp.dgl'))
  assert len(gg)==1 and gg[0].num_nodes()==5317 and gg[0].num_edges()==30752
  y=np.asarray(gg[0].ndata['label'].long().reshape(-1).numpy(),dtype=np.int64)[ids['test']]
 else:
  import pandas as pd
  f=pd.read_csv(SOURCES[ds]/'elliptic_txs_features.csv',usecols=[0,1],header=None)
  assert f.shape==(203769,2)
  txids=f.iloc[:,0].to_numpy(dtype=np.int64)
  times=f.iloc[:,1].to_numpy(dtype=np.int64)
  assert np.all(times[ids['TR40']]<=20) and np.all((times[ids['val']]>=21)&(times[ids['val']]<=31)) and np.all(times[ids['test']]>=32)
  selected=set(map(int,txids[ids['test']]))
  labels={}
  with (SOURCES[ds]/'elliptic_txs_classes.csv').open(newline='') as fp:
   rd=csv.DictReader(fp)
   assert set(rd.fieldnames)=={'txId','class'}
   for row in rd:
    key=int(row['txId'])
    if key in selected:
     assert key not in labels and row['class'] in ('1','2')
     labels[key]=int(row['class']=='1')
  assert len(labels)==cfg['testn']
  y=np.asarray([labels[int(i)] for i in txids[ids['test']]],dtype=np.int64)
 assert len(y)==cfg['testn'] and 0<int(y.sum())<len(y)
 return ids,y


def verify_finished():
 assert socket.gethostname()=='60d39211b1fd','STOP: wrong GPU instance hostname'
 assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==PIN
 assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
 s4=BASE/'logs/EOGFD_57212396_Stage4_Master.log'
 s5=BASE/'logs/EOGFD_57212396_Stage5_AuditRecovery.log'
 assert 'EOGFD_NEWI_BOTH_CONDITIONAL_STAGE4_12X2_FROZEN_TRAININGS_COMPLETE=PASS TEST_SCORED=NO' in s4.read_text()
 assert 'EOGFD_NEWI_BOTH_CONDITIONAL_STAGE5_12X2_HELDOUT_TEST_ONCE_COMPLETE=PASS' in s5.read_text()
 # Verify the original stage3/4/5 bundle files remain unchanged.
 for stage in ('STAGE3','STAGE4','STAGE5','STAGE5_RECOVERY'):
  root=W/('EOGFD_NEWI_'+stage)
  manifest=checked_json(root/'KIT_SHA256_MANIFEST.json')
  for name,expected in manifest.items():assert sha(root/name)==expected,('STAGE_SCRIPT_CHANGED',str(root/name))
 assert not PUB.exists() and not PRIVATE.exists(),'STOP: Stage6 output already exists; verify existing file rather than overwrite'


def export_dataset(ds,files,backup):
 cfg=DATA[ds]
 ids,y=verify_canonical(ds)
 p=f'EOGFD_{ds}_NewI'
 pref=f'EOGFD/{ds}/'
 root=BASE/'results/eogfd'/ds.lower()
 tune=root/'stage3_tr40_tuning';fin=root/'stage4_frozen_finals';test=root/'stage5_heldout_test_once'
 wp=tune/f'{p}_TR40_FrozenWinner.json'
 win=checked_json(wp)
 assert win['winner_trial']==cfg['winner'] and abs(float(win['winner_val_auprc'])-cfg['winner_ap'])<5e-7
 assert win['trial_count']==12 and win['test_scored'] is False and win['venus_approval_recorded'] is False
 assert win['split_sha256']==cfg['split'] and win['upstream_commit']==PIN
 r3=[]
 for i in range(1,13):
  d=tune/f'trial_{i:02d}';row=checked_json(d/'trial_result.json')
  assert row['trial']==i and row['test_scored'] is False and row['test_labels_accessed'] is False
  assert row['mode']=='UNIFIED_CONDITIONAL_ADAPTED' and row['upstream_commit']==PIN
  assert row['split_sha256']==cfg['split'] and sha(d/'best_validation.pt')==row['checkpoint_sha256']
  assert sha(d/'best_validation_scores.npy')==row['validation_scores_sha256']
  for name in ('trial_result.json','validation_history.json','epoch_times.csv','best_validation_scores.npy'):
   cpyfile(files,pref+f'results/tuning/trial_{i:02d}/'+name,d/name)
  backup[f'PRIVATE/tuning/{ds}/trial_{i:02d}/best_validation.pt']=d/'best_validation.pt'
  r3.append(row)
 best=max(r3,key=lambda t:(t['best_val_auprc'],-t['trial']))
 assert best['trial']==win['winner_trial'] and best['config']==win['winner_config']
 for n in (wp.name,f'{p}_TR40_TuningSummary.json','stage3_run_manifest.json','population20_cost_gate.json'):
  cpyfile(files,pref+'results/tuning/'+n,tune/n)
 s4=fin/f'EOGFD_{ds}_Stage4_12FrozenFinalsSummary.json'
 s5=test/f'EOGFD_{ds}_Stage5_AllRatios_Summary.json'
 s4data=checked_json(s4);s5data=checked_json(s5)
 assert s4data['completed_finals']==12 and s4data['test_scored'] is False and s4data['venus_approval_recorded'] is False
 assert s5data['status']=='ALL12_CONDITIONAL_ONE_TIME_HELDOUT_TESTS_COMPLETE'
 assert s5data['test_scoring_per_final_checkpoint']==1 and s5data['venus_approval_recorded'] is False
 assert s5data['frozen_config']==win['winner_config'] and s5data['stage3_tuning_trials']==12 and s5data['stage4_frozen_final_trainings']==12
 assert s5data['split_sha256']==cfg['split'] and s5data['upstream_commit']==PIN
 testrows=[]
 for ratio in RATIOS:
  for seed in SEEDS:
   fd=fin/ratio/f'seed_{seed}';td=test/ratio/f'seed_{seed}'
   frozen=checked_json(fd/'frozen_training.json')
   claim=checked_json(td/'test_once_claim.json')
   result=checked_json(td/'test_metrics.json')
   assert frozen['dataset']==ds and frozen['ratio']==ratio and frozen['seed']==seed
   assert frozen['config']==win['winner_config'] and frozen['split_sha256']==cfg['split']
   assert frozen['test_scored'] is False and frozen['test_labels_accessed'] is False
   assert frozen['venus_approval_recorded'] is False and frozen['mode']=='UNIFIED_CONDITIONAL_ADAPTED'
   assert sha(fd/'best_validation.pt')==frozen['checkpoint_sha256']
   assert sha(fd/'best_validation_scores.npy')==frozen['validation_scores_sha256']
   assert sha(fd/'epoch_times.csv')==frozen['epoch_times_sha256']
   assert sha(fd/'validation_history.json')==frozen['history_sha256']
   assert claim['claimed_attempt_count']==1 and claim['dataset']==ds and claim['ratio']==ratio and claim['seed']==seed
   assert claim['checkpoint_sha256']==frozen['checkpoint_sha256'] and claim['status']=='TEST_SCORES_SAVED_ONCE'
   assert result['test_scored_once'] is True and result['test_inference_claims']==1
   assert result['training_during_test'] is False and result['test_threshold_retuned'] is False
   assert result['venus_approval_recorded'] is False and result['mode']=='UNIFIED_CONDITIONAL_ADAPTED'
   assert result['checkpoint_sha256']==frozen['checkpoint_sha256']
   assert result['ratio']==ratio and result['seed']==seed and result['dataset']==ds
   assert_close(result['frozen_validation_threshold'],frozen['frozen_validation_threshold'],'threshold',1e-12)
   assert result['test_count']==cfg['testn'] and result['test_positives']==int(y.sum())
   assert sha(td/'test_scores.npy')==claim['test_scores_sha256']==result['test_scores_sha256']
   scores=np.asarray(np.load(td/'test_scores.npy',allow_pickle=False),dtype=np.float64).reshape(-1)
   mm=metric(y,scores,float(frozen['frozen_validation_threshold']))
   for key,val in mm.items():
    if key in ('tp','tn','fp','fn'):assert val==result[key],(ds,ratio,seed,key)
    else:assert_close(val,result[key],f'{ds}/{ratio}/seed{seed}/{key}',1e-9)
   for name in ('frozen_training.json','validation_history.json','epoch_times.csv','best_validation_scores.npy'):
    cpyfile(files,pref+f'results/finals/{ratio}/seed_{seed}/'+name,fd/name)
   for name in ('test_metrics.json','test_once_claim.json','test_scores.npy'):
    cpyfile(files,pref+f'results/finals/{ratio}/seed_{seed}/'+name,td/name)
   backup[f'PRIVATE/finals/{ds}/{ratio}/seed_{seed}/best_validation.pt']=fd/'best_validation.pt'
   testrows.append(result)
 for ratio in RATIOS:
  pair=[r for r in testrows if r['ratio']==ratio]
  assert len(pair)==3 and {r['seed'] for r in pair}==set(SEEDS)
  summaryrow=next(r for r in s5data['metrics'] if r['ratio']==ratio)
  for key in ('auprc','auroc','macro_f1','fraud_f1','g_mean','fraud_precision','fraud_recall','accuracy','balanced_accuracy','mcc','brier_score'):
   a=np.asarray([r[key] for r in pair],dtype=np.float64)
   assert_close(summaryrow[key+'_mean'],float(a.mean()),f'{ds}/{ratio}/{key}_mean')
   assert_close(summaryrow[key+'_sample_sd'],float(a.std(ddof=1)),f'{ds}/{ratio}/{key}_sample_sd')
 for n in (s4.name,'stage4_frozen_training_manifest.json'):
  cpyfile(files,pref+'results/finals/'+n,fin/n)
 for n in (s5.name,f'EOGFD_{ds}_Stage5_All12_PretestAudit.json',f'EOGFD_{ds}_Stage5_ValidationReplayProvenance.json'):
  cpyfile(files,pref+'results/finals/'+n,test/n)
 # The actual stages, original and corrective scripts; never upstream .git, checkpoints or input datasets.
 scriptfiles=[('stage2_pilot_original.py',W/'EOGFD_NEWI_KIT/vast_preflight_and_realgraph_pilots.py'),
  ('stage2_pilot_elliptic_corrected.py',W/'EOGFD_NEWI_KIT/vast_preflight_and_realgraph_pilots_elliptic_corrected.py'),
  ('stage3_runner.py',W/'EOGFD_NEWI_STAGE3/stage3_runner.py'),
  ('stage3_launch_both.sh',W/'EOGFD_NEWI_STAGE3/stage3_launch_both.sh'),
  ('stage4_runner.py',W/'EOGFD_NEWI_STAGE4/stage4_runner.py'),
  ('stage4_launch_both.sh',W/'EOGFD_NEWI_STAGE4/stage4_launch_both.sh'),
  ('stage5_test_once_original.py',W/'EOGFD_NEWI_STAGE5/stage5_test_once.py'),
  ('stage5_launch_both.sh',W/'EOGFD_NEWI_STAGE5/stage5_launch_both.sh'),
  ('stage5_pretest_forensics.py',W/'EOGFD_NEWI_STAGE5_RECOVERY/stage5_pretest_forensics.py'),
  ('stage5_test_once_dualsnapshot_audited.py',W/'EOGFD_NEWI_STAGE5_RECOVERY/stage5_test_once_dualsnapshot_audited.py'),
  ('stage5_integrity_and_safe_resume.sh',W/'EOGFD_NEWI_STAGE5_RECOVERY/stage5_integrity_and_safe_resume.sh'),
  ('stage5_launch_both_dualsnapshot_audited.sh',W/'EOGFD_NEWI_STAGE5_RECOVERY/stage5_launch_both_dualsnapshot_audited.sh'),
  ('stage6_offline_export_and_private_backup.py',W/'EOGFD_NEWI_STAGE6/export_and_backup.py'),
  ('stage6_mac_verify_and_publish.sh',W/'EOGFD_NEWI_STAGE6/mac_verify_and_publish.sh'),
  ('enable_eogfd_cuda11.sh',BASE/'bootstrap/enable_eogfd_cuda11.sh')]
 for name,path in scriptfiles:cpyfile(files,pref+'scripts/'+name,path)
 envinfo={'python':sys.version,'pinned_torch':'1.12.1+cu113','pinned_dgl':'0.8.1','GPU':'NVIDIA RTX A6000','instance_id':'57212396','host':socket.gethostname(),'model_upstream_commit':PIN,'python_environment':'/workspace/comp8851/env/eogfd-legacy','source_SHA256':cfg['source'],'split_SHA256':cfg['split'],'hardware_note':'Runtime values are GPU-instance-specific and not T4-comparable'}
 put(files,pref+'evidence/EOGFD_57212396_Environment.json',jbytes(envinfo))
 cpyfile(files,pref+'evidence/stage1_stage2_pilot.json',BASE/'diagnostics/eogfd_newi_57212396'/f'{ds}_Stage1_Stage2_ReadOnlyThenOneStep.json')
 if ds=='Elliptic':
  cpyfile(files,pref+'evidence/elliptic_edgecount_assertion_repair.json',BASE/'diagnostics/eogfd_newi_57212396/Elliptic_EdgeCount_Assertion_Repair.json')
 for log in ('eogfd_elliptic_fdcompcn_stage12.log','EOGFD_57212396_Stage3_Master.log',
  'EOGFD_57212396_Stage3_ImportRecovery.log','EOGFD_57212396_Stage4_Master.log',
  'EOGFD_57212396_Stage5_Master.log','EOGFD_57212396_Stage5_AuditRecovery.log'):
  path=BASE/'logs'/log
  if path.is_file():cpyfile(files,pref+'evidence/logs/'+log,path)
 bundle=BASE/'logs/eogfd_newi_57212396'/f'{ds}_Stage3.log'
 if bundle.is_file():cpyfile(files,pref+'evidence/logs/'+f'{ds}_Stage3.log',bundle)
 summarytable='\n'.join('| '+ratio+' | '+' | '.join(f'{next(r for r in s5data["metrics"] if r["ratio"]==ratio)[metric+"_mean"]:.6f} ± {next(r for r in s5data["metrics"] if r["ratio"]==ratio)[metric+"_sample_sd"]:.6f}' for metric in ('auprc','auroc','macro_f1'))+' |' for ratio in RATIOS)
 individual='\n'.join(f'| {ratio} | {seed} | {next(r for r in testrows if r["ratio"]==ratio and r["seed"]==seed)["auprc"]:.6f} | {next(r for r in testrows if r["ratio"]==ratio and r["seed"]==seed)["auroc"]:.6f} | {next(r for r in testrows if r["ratio"]==ratio and r["seed"]==seed)["macro_f1"]:.6f} |' for ratio in RATIOS for seed in SEEDS)
 relation='original four directed company relations (static transductive unlabeled graph)' if ds=='FDCompCN' else 'single original directed transaction relation with ratio-specific training-prefix cutoffs TR40 ≤20, TR30 ≤12, TR20 ≤7, TR10 ≤3, validation ≤31, test period 32–49; no invented or reversed edges'
 readme=f'''# EOGFD × {ds} — completed conditional unified benchmark

**Status:** 12/12 TR40 validation-only tuning trials, 12/12 fresh frozen final trainings, 12/12 once-only held-out test evaluations. These are completed experimental outputs, **not an approved author-identical reproduction**.

- Model: [BERA-wx/EOGFD](https://github.com/BERA-wx/EOGFD/tree/{PIN}) pinned at `{PIN}`; original `EOGFDHeteroModel` implementation unchanged.
- Assessment 2 canonical source SHA-256: `{json.dumps(cfg['source'],sort_keys=True)}`.
- Frozen split SHA-256: `{cfg['split']}`; split seed 2; training seeds 2, 42, 72.
- Input adapter: {relation}.
- Stage 3 tuning: TR40 seed 2, **12 attempts**, selection by validation AUPRC. Winner trial **{win['winner_trial']}**; validation AUPRC **{win['winner_val_auprc']:.6f}**. Frozen configuration: `{json.dumps(win['winner_config'],sort_keys=True)}`.
- Stage 4: four nested training ratios × three training seeds, maximum 100 epochs, patience 20; checkpoint selection by validation AUPRC, classification threshold selected by validation macro-F1 (fraud-recall/0.5 tie-breaking).
- Stage 5: one exclusive test claim per selected checkpoint; exactly one test inference and saved scores. Independent Stage 6 metric recomputation uses **saved scores only**, not new GPU inference.
- EOGFD evaluation RNG seed 100921 fixed for validation/test. Adam optimizer beta1=.9, beta2=.999, eps=1e-8; training loss includes model-specific class weighting; parameters recorded for each trial.
- **Methodological status:** `UNIFIED_CONDITIONAL_ADAPTED`; Venus's model-specific adaptation sign-off has **not been recorded**. This flag reflects documentation status, not rejection. Dataset eligibility should be reviewed before interpreting as a supervisor-approved benchmark.
- This benchmark's time/memory figures are from RTX A6000 instance 57212396, not the Kaggle T4 described as a planning baseline; treat cross-instance runtime comparisons as non-controlled.

## Once-only held-out test metrics (mean ± sample SD; three seeds)

| Ratio | AUPRC | AUROC | Macro-F1 |
|---|---:|---:|---:|
{summarytable}

### Individual held-out runs

| Ratio | Seed | AUPRC | AUROC | Macro-F1 |
|---|---:|---:|---:|---:|
{individual}

## Public evidence

- `results/tuning/`: 12 trial histories, timing CSVs, saved validation scores, frozen winning configuration, manifest and capacity gate.
- `results/finals/`: all 12 final validation records, full epoch timing CSVs, validation-score arrays, one-time held-out test claims, test-score arrays, frozen summaries and audit notes.
- `scripts/`: actual adapted runner scripts, corrected Stage 2 pilot and Stage 5 discrepancy recovery. Original EOGFD author's source lives only in linked pinned upstream.
- `evidence/`: execution logs, pinned source/split fingerprints, independent saved-score audit and publication hash manifest.
- **Excluded from Git:** source datasets, frozen split arrays, model `.pt` weights, full training recovery `.pt` files, Python/CUDA environments and private backup.
- A separate private ZIP preserves the 24 final and 24 tuning checkpoints across both datasets, including the frozen split arrays. Keep it outside the public repository.

**Original released `run_eogfd.py` evaluates held-out data each epoch; this external benchmark runner instead froze Assessment 2 split IDs and isolated test access. Not an author-identical reproduction.**
'''
 put(files,pref+'README.md',readme)
 status=dict(model='EOGFD',dataset=ds,instance_id='57212396',machine_hostname='60d39211b1fd',
  status='COMPLETED_CONDITIONAL_UNIFIED_BENCHMARK',mode='UNIFIED_CONDITIONAL_ADAPTED',
  venus_approval_recorded=False,model_specific_adaptation_requires_review=True,
  author_identical_reproduction=False,original_upstream_model_implementation=True,upstream_commit=PIN,
  source_sha256=cfg['source'],split_sha256=cfg['split'],split_seed=2,training_seeds=list(SEEDS),
  ratios=list(RATIOS),completed_tuning_trials=12,completed_frozen_trainings=12,
  completed_once_only_heldout_test_evaluations=12,winning_trial=win['winner_trial'],
  winning_validation_auprc=win['winner_val_auprc'],frozen_config=win['winner_config'],
  one_time_test_inference_per_run=True,publication_performed_no_model_inference=True,
  test_metrics_summary=s5data['metrics'])
 put(files,pref+'status.json',jbytes(status))
 audit={'status':'EOGFD_'+ds+'_57212396_PUBLIC_EVIDENCE_AUDIT_PASS',
  'author_identical_benchmark':False,'venus_approval_recorded':False,
  'dataset':ds,'model':'EOGFDHeteroModel','instance_id':'57212396',
  'original_author_source_unmodified':True,'source_sha256':cfg['source'],
  'split_sha256':cfg['split'],'frozen_split_seed':2,'training_seeds':list(SEEDS),
  'stage3_tuning_trials':12,'stage4_frozen_trainings':12,'heldout_test_once_completed':12,
  'test_count':len(y),'test_positives':int(y.sum()),
  'test_labels_not_used_in_model_selection':True,'independent_saved_score_metric_recomputation':True,
  'additional_model_inference_in_export':False,'raw_data_and_weights_excluded_from_public_zip':True,
  'frozen_winner_trial':win['winner_trial'],'frozen_winner_validation_auprc':win['winner_val_auprc'],
  'final_metrics':s5data['metrics']}
 put(files,pref+'evidence/EOGFD_'+ds+'_57212396_IndependentPublicationAudit.json',jbytes(audit))
 manifest_path=pref+'evidence/EOGFD_'+ds+'_57212396_PublicationFileManifest.json'
 manifest=dict(branch=BRANCH,repository='marufkhan09/COMP8851-fraud-dataset-analysis',
  destination=pref,status='COMPLETED_CONDITIONAL_UNIFIED_BENCHMARK',model='EOGFD',dataset=ds,
  files={k:{'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)} for k,v in sorted(files.items()) if k.startswith(pref)})
 put(files,manifest_path,jbytes(manifest))
 print(f'EOGFD_{ds}_STAGE6_OFFLINE_METRICS_AND_PUBLICATION_AUDIT=PASS FILES={len(manifest["files"])+1} TEST_SCORINGS_REPEATED=NO',flush=True)
 return {'ds':ds,'status':status,'counts':len(manifest['files'])+1}


def private_export(backup,public_summary):
 privfiles={}
 for n,path in sorted(backup.items()):
  assert n.startswith('PRIVATE/') and n.endswith('.pt')
  assert Path(path).is_file()
  privfiles[n]=Path(path).read_bytes()
 assert len(privfiles)==48,('NOT_48_PRIVATE_WEIGHTS',len(privfiles))
 for ds in DS:privfiles[f'PRIVATE/splits/{ds}/{SPLITS[ds].name}']=SPLITS[ds].read_bytes()
 info={'note':'PRIVATE: DO NOT PUSH TO GITHUB. Model weights and frozen seed-2 split arrays. Raw dataset sources excluded.',
       'instance_id':'57212396','upstream_model_commit':PIN,
       'final_weights':24,'tuning_weights':24,'split_files':2,
       'canonical_sources_not_in_archive':{ds:DATA[ds]['source'] for ds in DS},
       'dataset_publication_status':{x['ds']:x['status']['status'] for x in public_summary},
       'restore_note':'Use this private ZIP together with the separately published scripts and your canonical original sources.'}
 privfiles['PRIVATE/README.json']=jbytes(info)
 manifest={n:{'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)} for n,b in privfiles.items()}
 privfiles['PRIVATE/SHA256_MANIFEST.json']=jbytes(manifest)
 with zipfile.ZipFile(PRIVATE,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for n,b in sorted(privfiles.items()):z.writestr(n,b)
 with zipfile.ZipFile(PRIVATE,'r') as z:
  assert z.testzip() is None and set(z.namelist())==set(privfiles)
  for n,m in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==m['sha256']
 print('EOGFD_STAGE6_PRIVATE_48_CHECKPOINTS_PLUS_2_SPLITS=PASS')
 print('EOGFD_PRIVATE_BACKUP='+str(PRIVATE))
 print('EOGFD_PRIVATE_BACKUP_BYTES='+str(PRIVATE.stat().st_size))
 print('EOGFD_PRIVATE_BACKUP_SHA256='+sha(PRIVATE))


def public_export(files):
 assert sum(1 for n in files if n.endswith('/test_metrics.json'))==24
 assert sum(1 for n in files if n.endswith('/test_once_claim.json'))==24
 assert sum(1 for n in files if n.endswith('/test_scores.npy'))==24
 assert sum(1 for n in files if n.endswith('/trial_result.json'))==24
 for name in files:
  assert name.startswith(('EOGFD/FDCompCN/','EOGFD/Elliptic/'))
  assert PurePosixPath(name).suffix in ALLOWED_PUBLIC_EXT
  assert all(w not in name.lower() for w in ('/private/','/.git/','/source/'))
 with zipfile.ZipFile(PUB,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for n,b in sorted(files.items()):z.writestr(n,b)
 with zipfile.ZipFile(PUB,'r') as z:
  assert z.testzip() is None and set(z.namelist())==set(files)
  for ds in DS:
   pref=f'EOGFD/{ds}/';mn=pref+f'evidence/EOGFD_{ds}_57212396_PublicationFileManifest.json'
   m=json.loads(z.read(mn));assert set(m['files'])==set(n for n in files if n.startswith(pref))-{mn}
   for name,meta in m['files'].items():
    data=z.read(name)
    assert hashlib.sha256(data).hexdigest()==meta['sha256'] and len(data)==meta['bytes']
 print(f'EOGFD_STAGE6_BOTH_DATASETS_GITHUB_EXPORT=PASS FILES={len(files)} NO_NEW_INFERENCE=YES')
 print('EOGFD_PUBLICATION_ZIP='+str(PUB))
 print('EOGFD_PUBLICATION_BYTES='+str(PUB.stat().st_size))
 print('EOGFD_PUBLICATION_SHA256='+sha(PUB))


def main():
 verify_finished()
 files={};backup={};summ=[]
 for ds in DS:summ.append(export_dataset(ds,files,backup))
 private_export(backup,summ)
 public_export(files)
 print('EOGFD_57212396_STAGE6_EXPORT_AND_PRIVATE_BACKUP=PASS GITHUB_NOT_YET_PUSHED=YES')

if __name__=='__main__':
 try:main()
 except BaseException as e:
  print('EOGFD_STAGE6_STOP='+repr(e),file=sys.stderr,flush=True)
  raise
