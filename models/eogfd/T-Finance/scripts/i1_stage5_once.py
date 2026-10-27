#!/usr/bin/env python3
"""T-Finance Stage5 once-only held-out evaluation.

Read-only forensic frozen audit must pass before all scoring. Persist an
exclusive claim BEFORE each sole forward pass. Existing claims are never
replayed; completed scores are audited offline. No training/tuning or threshold
adjustment, unchanged author EOGFDHeteroModel with frozen-split adapter.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, random, socket, sys, traceback
from pathlib import Path
import numpy as np
import torch
import dgl
from dgl.data.utils import load_graphs
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, brier_score_loss, confusion_matrix)

BASE=Path('/workspace/comp8851'); W=Path('/workspace')
S3=BASE/'results/eogfd/tfinance/stage3_tr40_tuning'
S4=BASE/'results/eogfd/tfinance/stage4_frozen_finals'
S5=BASE/'results/eogfd/tfinance/stage5_heldout_test_once'
S3PY=W/'EOGFD_TFINANCE_STAGE3_I1_20261023/stage3_tfinance.py'
AUDITPY=W/'EOGFD_TFINANCE_PRETEST_AUDIT_I1_V2_20261023/i1_pretest_audit_backup_v2.py'
AUTHOR=BASE/'repos/EOGFD'; SRC=BASE/'assets/tfinance/source/tfinance'
SPLIT=BASE/'assets/tfinance/tfinance_seed2_nested_splits.npz'
RATIOS=('TR40','TR30','TR20','TR10');SEEDS=(2,42,72)
PIN='a4726f173bc8c923109ddcb2786f4d9c5206ae7f'
SOURCE_SHA='b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700'
SPLIT_SHA='de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1'
EVAL_SEED=100921


def log(s): print(s,flush=True)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def require(ok,why):
    if not ok: raise RuntimeError('STOP_'+str(why))
def jread(p): return json.loads(Path(p).read_text())
def write_new_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
def write_new_npy(p,values):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:
        np.save(f,np.asarray(values,dtype=np.float64),allow_pickle=False)
        f.flush();os.fsync(f.fileno())
def import_verified(path,name,expected_sha=None):
    if expected_sha is not None: require(sha(path)==expected_sha,name+'_HASH_CHANGED')
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj

def metric(y,s,threshold):
    y=np.asarray(y,dtype=np.int64).ravel();s=np.asarray(s,dtype=np.float64).ravel()
    require(len(y)==15743 and len(s)==len(y),'TEST_SCORES_LENGTH')
    require(set(np.unique(y).tolist())=={0,1},'TEST_LABEL_CLASSES')
    require(np.isfinite(s).all() and ((s>=0)&(s<=1)).all(),'NONFINITE_PROBS')
    require(0<threshold<1,'FROZEN_THRESHOLD')
    pred=np.asarray(s>threshold,dtype=np.int64)
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
    recall=tp/max(tp+fn,1);specificity=tn/max(tn+fp,1)
    return dict(auprc=float(average_precision_score(y,s)),
       auroc=float(roc_auc_score(y,s)),
       macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
       fraud_f1=float(f1_score(y,pred,pos_label=1,zero_division=0)),
       fraud_precision=float(precision_score(y,pred,pos_label=1,zero_division=0)),
       fraud_recall=float(recall_score(y,pred,pos_label=1,zero_division=0)),
       g_mean=float(math.sqrt(recall*specificity)),
       accuracy=float(accuracy_score(y,pred)),
       balanced_accuracy=float(balanced_accuracy_score(y,pred)),
       mcc=float(matthews_corrcoef(y,pred)),
       brier_score=float(brier_score_loss(y,s)),
       tp=tp,tn=tn,fp=fp,fn=fn)

def audit_frozen():
    require(socket.gethostname()=='60d39211b1fd','WRONG_I1_INSTANCE')
    require(sha(SRC)==SOURCE_SHA and sha(SPLIT)==SPLIT_SHA,'ORIGINAL_INPUT_CHANGED')
    with torch.no_grad():
        mod=import_verified(AUDITPY,'i1_pretest_existing_audit')
        a=mod.check() # Full 12-checkpoint, validation-scores and threshold forensic audit
    require(a['completed_finals']==12 and a['test_evaluations']==0,'PRETEST_AUDIT')
    require(a['status']=='ALL_12_FROZEN_PRETEST_AUDITS_PASS','PRETEST_STATUS')
    require(a['source_sha256']==SOURCE_SHA and a['split_sha256']==SPLIT_SHA,'AUDIT_HASH')
    pre=W/'EOGFD_TFINANCE_I1_PRETEST_BACKUP_20261023.tar.gz'
    sums=Path(str(pre)+'.sha256')
    require(pre.is_file() and sums.is_file(),'PRETEST_BACKUP_MISSING')
    row=sums.read_text().split()
    require(len(row)==2 and row[1]==pre.name and sha(pre)==row[0],'PRETEST_BACKUP_CHANGED')
    require(pre.stat().st_size>100000,'BACKUP_SIZE')
    frozen=jread(S3/'EOGFD_TFinance_TR40_FrozenWinner.json')
    require(frozen['winner_trial']==8 and abs(frozen['winner_val_auprc']-.883130)<1e-6,'WINNER_CHANGED')
    log('I1_TFINANCE_STAGE5_PRETEST_ALL12_AND_OFFINSTANCE_BACKUP_REFERENCE=PASS')
    return frozen,a

def read_test_labels_only_when_scoring(ids):
    # Never called by --audit-only. No graph tensor has labels during any forward.
    g,_=load_graphs(str(SRC));require(len(g)==1,'GRAPH_COUNT')
    raw=g[0].ndata['label'];ix=torch.from_numpy(ids['test'].copy())
    if raw.shape==(39357,2):y=raw[ix].argmax(dim=1).long().numpy().copy()
    else:y=raw[ix].long().view(-1).numpy().copy()
    require(len(y)==15743 and 0<int(y.sum())<len(y),'TEST_LABEL_COUNTS')
    del g,raw
    return y

def verify_finished(ratio,seed,rec,y):
    folder=S5/ratio/f'seed_{seed}'
    claim=folder/'test_once_claim.json';scores=folder/'test_scores.npy'
    result=folder/'test_metrics.json'
    require(claim.exists() and scores.exists() and result.exists(),'INCOMPLETE_TEST_CLAIM_DO_NOT_RERUN')
    cl=jread(claim);r=jread(result)
    require(cl['claimed_attempt_count']==1 and cl['claimed_before_test_forward'] is True,'TEST_CLAIM')
    require(cl['ratio']==ratio and cl['seed']==seed and cl['checkpoint_sha256']==rec['checkpoint_sha256'],'CLAIM_ID')
    require(r['ratio']==ratio and r['seed']==seed and r['checkpoint_sha256']==rec['checkpoint_sha256'],'TEST_RECORD_ID')
    require(r['test_inference_claims']==1 and r['test_scored_once'] is True and not r['training_during_test'] and not r['threshold_retuned'],'ONCE_ONLY_FLAGS')
    require(sha(scores)==r['test_scores_sha256'],'SCORES_HASH')
    require(r['test_count']==15743 and r['test_positives']==int(y.sum()),'TEST_LABEL_COUNTS')
    s=np.load(scores,allow_pickle=False)
    m=metric(y,s,float(rec['frozen_validation_threshold']))
    for k,v in m.items():
        if k in ('tp','tn','fp','fn'): require(v==r[k],'OFFLINE_'+k)
        else:require(abs(v-r[k])<1e-9,'OFFLINE_'+k)
    log(f'I1_TFINANCE_STAGE5_{ratio}_SEED{seed}_SAVED_SCORES_REPLAY=PASS no_new_inference')
    return r

def exclusive_claims_precheck():
    for ratio in RATIOS:
        for seed in SEEDS:
            path=S5/ratio/f'seed_{seed}'
            claim=path/'test_once_claim.json';scores=path/'test_scores.npy';result=path/'test_metrics.json'
            if claim.exists():require(scores.exists() and result.exists(),('UNRECOVERABLE_CLAIM_WITHOUT_SCORES',ratio,seed))
            if scores.exists() or result.exists():require(claim.exists(),'UNCLAIMED_TEST_SCORES')

def score_all():
    frozen,a=audit_frozen()
    import subprocess
    require(subprocess.check_output(['git','-C',str(AUTHOR),'rev-parse','HEAD'],text=True).strip()==PIN,'AUTHOR_PIN')
    require(not subprocess.check_output(['git','-C',str(AUTHOR),'status','--porcelain'],text=True).strip(),'AUTHOR_DIRTY')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','GPU_ASSIGNMENT')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'CUDA')
    exclusive_claims_precheck()
    S5.mkdir(parents=True,exist_ok=True)
    summary_path=S5/'EOGFD_TFinance_Stage5_AllRatios_Summary.json'
    if summary_path.exists():
        log('I1_TFINANCE_STAGE5_SUMMARY_EXISTS_VERIFY_ONLY_NO_EXTRA_INFERENCE')
    s3sha=None
    for row in (W/'EOGFD_TFINANCE_STAGE3_I1_20261023/KIT_SHA256SUMS.txt').read_text().splitlines():
        parts=row.split()
        if parts[-1]=='stage3_tfinance.py':s3sha=parts[0]
    require(s3sha is not None,'STAGE3_SHA_NOT_FOUND')
    s3=import_verified(S3PY,'stage3_tfinance_stage5_verified',s3sha)
    ids=s3.validate_assets()
    y=read_test_labels_only_when_scoring(ids)
    # If everything already finished, only replay offline without loading GPU graph.
    all_done=all((S5/r/f'seed_{seed}'/'test_metrics.json').exists() for r in RATIOS for seed in SEEDS)
    b=None
    if not all_done:
        b=s3.dataset_gpu(ids)
        # dataset_gpu strips labels from graph before forwarding.
        require(not list(b['graph'].ndata.keys()) and not list(b['graph'].edata.keys()),'GRAPH_LABEL_LEAK')
        test_ix=torch.from_numpy(ids['test'].copy()).long().to('cuda:0')
        from models.eogfd import EOGFDHeteroModel
    import gc
    winner_sha=sha(S3/'EOGFD_TFinance_TR40_FrozenWinner.json')
    out=[]
    for ratio in RATIOS:
        for seed in SEEDS:
            folder=S5/ratio/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
            rec=jread(S4/ratio/f'seed_{seed}'/'frozen_training.json')
            require(rec['config']==frozen['winner_config'] and rec['frozen_winner_sha256']==winner_sha,'FROZEN_CONF')
            require(rec['test_scored'] is False and rec['test_evaluations']==0,'FROZEN_TEST_FLAG_CHANGED')
            claim=folder/'test_once_claim.json';sp=folder/'test_scores.npy';result=folder/'test_metrics.json'
            if claim.exists():
                out.append(verify_finished(ratio,seed,rec,y));continue
            require(not sp.exists() and not result.exists(),'UNCLAIMED_SCORES')
            cp=S4/ratio/f'seed_{seed}'/'best_validation.pt'
            require(sha(cp)==rec['checkpoint_sha256'],'CHECKPOINT_HASH')
            ckpt=torch.load(cp,map_location='cpu')
            require(ckpt['epoch']==rec['best_epoch'] and ckpt['config']==rec['config'],'CKPT_LOCK')
            model=s3.build_model(rec['config'])
            s3.restore_state(model,ckpt['model_state'])
            model.eval()
            request={'dataset':'T-Finance','ratio':ratio,'seed':seed,
                 'status':'CLAIMED_BEFORE_SINGLE_TEST_INFERENCE','claimed_before_test_forward':True,
                 'claimed_attempt_count':1,'checkpoint_sha256':rec['checkpoint_sha256'],
                 'frozen_training_sha256':sha(S4/ratio/f'seed_{seed}'/'frozen_training.json'),
                 'frozen_validation_threshold':rec['frozen_validation_threshold'],
                 'upstream_commit':PIN,'source_sha256':SOURCE_SHA,'split_sha256':SPLIT_SHA,
                 'evaluation_rng_seed':EVAL_SEED,'training_during_test':False,
                 'threshold_retuned':False,'no_retry_if_claim_exists_without_scores':True}
            write_new_json(claim,request)  # DURABLE, EXCLUSIVE before test inference
            with torch.random.fork_rng(devices=[0],enabled=True):
                torch.manual_seed(EVAL_SEED);torch.cuda.manual_seed_all(EVAL_SEED)
                with torch.no_grad():
                    logits,_=model(b['graph'],b['features'])
                    scores=torch.softmax(logits[test_ix],dim=1)[:,1].detach().cpu().numpy().astype(np.float64)
            del logits,model
            require(len(scores)==15743 and np.isfinite(scores).all(),'TEST_SCORE_SHAPE')
            write_new_npy(sp,scores)
            m=metric(y,scores,float(rec['frozen_validation_threshold']))
            row=dict(dataset='T-Finance',ratio=ratio,seed=seed,model='EOGFDHeteroModel',
              mode='UNIFIED_INPUT_ADAPTER_NOT_AUTHOR_IDENTICAL',venus_methodological_approval_recorded=False,
              checkpoint_sha256=rec['checkpoint_sha256'],source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
              frozen_winner_sha256=winner_sha,upstream_commit=PIN,
              test_count=15743,test_positives=int(y.sum()),
              frozen_validation_threshold=float(rec['frozen_validation_threshold']),
              test_scores_sha256=sha(sp),test_scored_once=True,test_inference_claims=1,
              training_during_test=False,threshold_retuned=False,
              original_model_code_unchanged=True,eval_seed=EVAL_SEED,**m)
            write_new_json(result,row)
            out.append(verify_finished(ratio,seed,rec,y))
            del scores
            gc.collect();torch.cuda.empty_cache()
            log(f'I1_TFINANCE_STAGE5_{ratio}_SEED{seed}_ONCE_ONLY_TEST=PASS test_AUPRC={row["auprc"]:.6f}')
    require(len(out)==12,'EXACTLY_12_TEST_RECORDS')
    summary={'status':'EOGFD_TFINANCE_12_ONCE_ONLY_HELDOUT_TESTS_COMPLETE',
      'dataset':'T-Finance','mode':'UNIFIED_INPUT_ADAPTER_NOT_AUTHOR_IDENTICAL',
      'venus_methodological_approval_recorded':False,
      'upstream_commit':PIN,'source_sha256':SOURCE_SHA,'split_sha256':SPLIT_SHA,
      'frozen_config':frozen['winner_config'],'winner_trial':8,'winner_validation_AUPRC':frozen['winner_val_auprc'],
      'tuning_trials_completed':12,'frozen_final_trainings_completed':12,'test_evaluations':12,
      'test_inference_claims_per_final':1,'threshold_retuning_during_test':False,
      'test_scoring_mode':'single original EOGFDHeteroModel forward / frozen checkpoint / fixed RNG',
      'ratios':list(RATIOS),'training_seeds':list(SEEDS),'metrics':[]}
    for ratio in RATIOS:
        sub=[r for r in out if r['ratio']==ratio]
        require({r['seed'] for r in sub}==set(SEEDS),'PER_RATIO_SEEDS')
        obj={'ratio':ratio,'seeds':list(SEEDS)}
        for k in ('auprc','auroc','macro_f1','fraud_f1','fraud_precision','fraud_recall','g_mean',
                  'accuracy','balanced_accuracy','mcc','brier_score'):
            z=np.asarray([r[k] for r in sub],dtype=np.float64)
            obj[k+'_mean']=float(z.mean());obj[k+'_sample_sd']=float(z.std(ddof=1))
        summary['metrics'].append(obj)
    if summary_path.exists():require(jread(summary_path)==summary,'EXISTING_SUMMARY_CONFLICT')
    else:write_new_json(summary_path,summary)
    log('I1_TFINANCE_STAGE5_12_TEST_ONCE_SAVED_AND_OFFLINE_VERIFIED=PASS')
    log('I1_TFINANCE_STAGE5_NO_TRAINING_OR_RETUNING=PASS')
    log('I1_TFINANCE_STAGE5_READY_FOR_READONLY_GITHUB_EXPORT=PASS')


def main():
    ap=argparse.ArgumentParser();a=ap.add_mutually_exclusive_group(required=True)
    a.add_argument('--audit-only',action='store_true');a.add_argument('--score-once',action='store_true')
    args=ap.parse_args()
    if args.audit_only:
        audit_frozen();exclusive_claims_precheck()
        log('I1_TFINANCE_STAGE5_AUDIT_ONLY=PASS TEST_LABELS_NOT_READ=YES')
    else:score_all()

if __name__=='__main__':
    try:main()
    except BaseException as e:
        log('I1_TFINANCE_STAGE5_STOP='+repr(e));traceback.print_exc();sys.exit(2)
