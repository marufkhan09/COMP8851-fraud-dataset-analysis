#!/usr/bin/env python3
"""Read-only forensic check of original immutable Stage 4 checkpoint vs replay metrics.
Does not load test labels, graphs or perform inference or modify any run artifact.
"""
from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
import torch
ROOT=Path('/workspace/comp8851/results/eogfd')
S=('FDCompCN','Elliptic')
R=('TR40','TR30','TR20','TR10')
SEEDS=(2,42,72)
FIELDS={
 'val_auprc':'best_val_auprc',
 'val_auroc':'best_val_auroc',
 'val_macro_f1':'best_val_macro_f1',
 'val_fraud_recall':'fraud_recall_validation',
 'threshold':'frozen_validation_threshold'
}
# Checkpoint vs record allowable drifts are narrowly bounded and separately disclosed.
LIMITS={'val_auprc':1e-8,'val_auroc':1e-6,'val_macro_f1':1e-6,
 'val_fraud_recall':1e-6,'threshold':1e-12}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for buf in iter(lambda:f.read(8*1024*1024),b''):h.update(buf)
 return h.hexdigest()

def main():
 issues=[]; anomalies=[]; checked=0
 for dataset in S:
  for ratio in R:
   for seed in SEEDS:
    d=ROOT/dataset.lower()/'stage4_frozen_finals'/ratio/f'seed_{seed}'
    record_p=d/'frozen_training.json'; checkpoint_p=d/'best_validation.pt'; history_p=d/'validation_history.json'
    assert all(p.is_file() for p in (record_p,checkpoint_p,history_p)), ('MISSING',dataset,ratio,seed)
    rec=json.loads(record_p.read_text()); hist=json.loads(history_p.read_text())
    assert sha(checkpoint_p)==rec['checkpoint_sha256'],('CHECKPOINT_CHANGED',dataset,ratio,seed)
    assert sha(history_p)==rec['history_sha256'],('HISTORY_CHANGED',dataset,ratio,seed)
    cp=torch.load(checkpoint_p,map_location='cpu')
    epoch=int(rec['best_epoch'])
    assert 1<=epoch<=len(hist) and hist[epoch-1]['epoch']==epoch and hist[epoch-1]['improved'] is True,('BAD_BEST_EPOCH',dataset,ratio,seed)
    assert cp['epoch']==epoch and cp['ratio']==ratio and cp['seed']==seed and cp['dataset']==dataset, ('CHECKPOINT_IDENTITY',dataset,ratio,seed)
    assert cp['test_scored'] is False and rec['test_scored'] is False
    assert not (d/'recovery_progress.pt').exists()
    assert set(cp['validation'])==set(FIELDS),('CHECKPOINT_VAL_FIELDS',dataset,ratio,seed)
    checked+=1
    for key,rkey in FIELDS.items():
     c=float(cp['validation'][key]);e=float(hist[epoch-1][key]);v=float(rec[rkey])
     assert all(math.isfinite(n) for n in (c,e,v))
     if abs(c-e)>1e-12: issues.append((dataset,ratio,seed,key,'CHECKPOINT_VS_ORIGINAL_BEST_EPOCH_HISTORY',abs(c-e)))
     delta=abs(c-v)
     if delta>1e-8:
      print(f'DIFFERENCE {dataset} {ratio} seed{seed} {key}: best_epoch={c:.15g} post_restore_replay={v:.15g} absolute_delta={delta:.12g} allowed={LIMITS[key]:g}',flush=True)
     if delta>LIMITS[key]: anomalies.append((dataset,ratio,seed,key,delta,LIMITS[key]))
 print(f'EOGFD_STAGE5_READ_ONLY_INSPECTION=CHECKED_{checked}_CHECKPOINTS',flush=True)
 if issues:
  print('CHECKPOINT_VS_EPOCH_HISTORY_MISMATCHES='+repr(issues),flush=True)
  raise AssertionError('STOP: checkpoint does not match its original best-epoch history')
 print('ORIGINAL_CHECKPOINTS_MATCH_ORIGINAL_BEST_EPOCH_HISTORIES=PASS',flush=True)
 if anomalies:
  print('NUMERICAL_REPLAY_DRIFT_EXCEEDS_STRICT_GATE='+repr(anomalies),flush=True)
  raise AssertionError('STOP: validation drift requires investigation before held-out test')
 print('EOGFD_STAGE5_ALL24_DUAL_SNAPSHOT_CHECKS=PASS WITHIN_MICRO_TOLERANCES; TEST_LABELS_NOT_OPENED=YES',flush=True)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  print('EOGFD_PRETEST_INSPECTION_STOP='+repr(e),flush=True)
  raise
