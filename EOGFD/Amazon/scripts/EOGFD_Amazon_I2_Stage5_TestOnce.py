#!/usr/bin/env python3
"""EOGFD Stage5: independently verify 12 frozen validation models, THEN score held-out test once per model.
No training, tuning, threshold selection or repeated test inference. A claimed test inference without
persisted scores is fail-closed and must be audited manually. Immutable upstream model code used.
"""
import os,sys,json,time,hashlib,importlib.util,traceback,gc
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score,roc_auc_score,f1_score,precision_score,
    recall_score,accuracy_score,balanced_accuracy_score,matthews_corrcoef,brier_score_loss,
    confusion_matrix)

C=json.loads(Path(__file__).with_name('EOGFD_Amazon_I2_Stage5_Config.json').read_text())
P=f"EOGFD_{C['dataset']}_{C['id']}"
BASE=Path('/workspace/comp8851')
PIPE=BASE/'eogfd/pipeline'
FIN=BASE/'results/eogfd'/C['dataset'].lower()/'stage4_frozen_finals'
OUT=BASE/'results/eogfd'/C['dataset'].lower()/'stage5_heldout_test_once'
SPLIT=Path('/workspace/uploads/ass2_split_candidates')/C['dataset']/C['split_name']
SRC=BASE/'assets/eogfd'/C['dataset']/C['source_name']
RATIOS=['TR40','TR30','TR20','TR10'];SEEDS=[2,42,72]
EVAL_SEED=100921


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def atomjson(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.writing')
    with open(tmp,'x') as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def atomnpy(path,arr):
    tmp=path.with_suffix('.writing.npy')
    assert not tmp.exists(),('UNRESOLVED PARTIAL SCORES',tmp)
    with open(tmp,'xb') as f:
        np.save(f,arr,allow_pickle=False)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def oneclaim(path,obj):
    with open(path,'x') as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())

def stop(msg):raise RuntimeError(msg)

def metric(y,s,threshold):
    assert s.ndim==1 and len(s)==len(y) and np.isfinite(s).all()
    assert np.isfinite(threshold) and 0<threshold<1
    pred=np.asarray(s>threshold,dtype=np.int64)
    tn,fp,fn,tp=[int(z) for z in confusion_matrix(y,pred,labels=[0,1]).ravel()]
    return dict(
      auprc=float(average_precision_score(y,s)),
      auroc=float(roc_auc_score(y,s)),
      macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
      fraud_f1=float(f1_score(y,pred,pos_label=1,zero_division=0)),
      fraud_precision=float(precision_score(y,pred,pos_label=1,zero_division=0)),
      fraud_recall=float(recall_score(y,pred,pos_label=1,zero_division=0)),
      accuracy=float(accuracy_score(y,pred)),
      balanced_accuracy=float(balanced_accuracy_score(y,pred)),
      mcc=float(matthews_corrcoef(y,pred)),
      brier_score=float(brier_score_loss(y,s)),
      tp=tp,tn=tn,fp=fp,fn=fn)

def preflight():
    assert OUT.exists() is False or OUT.is_dir()
    assert SPLIT.is_file() and SRC.is_file()
    assert sha(SRC)==C['source_sha256'] and sha(SPLIT)==C['split_sha256']
    master=BASE/'logs/eogfd'/(P+'_Stage4_Master.log')
    assert master.is_file() and (P+'_STAGE4_12_FROZEN_FINAL_TRAININGS_COMPLETE=PASS TEST_SCORED=NO') in master.read_text(), 'STAGE4 MASTER LOG NOT COMPLETE'
    st4py=PIPE/(P+'_Stage4_FrozenFinals12.py')
    st4cfg=PIPE/(P+'_Stage4_Config.json')
    assert st4py.is_file() and st4cfg.is_file()
    assert sha(st4py)==C['stage4_script_sha256'], 'STAGE4 SCRIPT CHANGED'
    assert sha(st4cfg)==C['stage4_config_sha256'], 'STAGE4 CONFIG CHANGED'
    spec=importlib.util.spec_from_file_location('stage4_validation_source',st4py)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.C==C['stage4_config']
    K,g,x,ids,yt,validx,yval,device,winner_sha,config=mod.validate()
    sumfile=FIN/(P+'_Stage4_12FrozenTrainingSummary.json')
    assert sumfile.is_file(), 'STAGE4 SUMMARY MISSING'
    summary=json.loads(sumfile.read_text())
    assert summary['status']=='EOGFD_12_FROZEN_TRAININGS_COMPLETE' and summary['training_count']==12
    assert summary['config']==config and summary['frozen_winner_sha256']==winner_sha
    assert summary['source_sha256']==C['source_sha256'] and summary['split_sha256']==C['split_sha256']
    assert summary['ratios']==RATIOS and summary['seeds']==SEEDS
    assert summary['test_scored'] is False and summary['test_labels_accessed'] is False and summary['heldout_test_scored_count']==0
    records=[]
    for ratio in RATIOS:
      for seed in SEEDS:
        run=FIN/ratio/f'seed_{seed}'
        rowfile=run/'frozen_training.json'
        checkpoint=run/'best_validation.pt'
        valfile=run/'best_validation_scores.npy'
        hist=run/'validation_history.json'
        assert all(z.is_file() for z in [rowfile,checkpoint,valfile,hist]),('MISSING STAGE4 ARTIFACT',ratio,seed)
        row=json.loads(rowfile.read_text())
        assert row['ratio']==ratio and row['seed']==seed and row['dataset']==C['dataset']
        assert row['model']=='EOGFDHeteroModel' and row['config']==config
        assert row['frozen_winner_sha256']==winner_sha
        assert row['source_sha256']==C['source_sha256'] and row['split_sha256']==C['split_sha256']
        assert row['upstream_commit']==C['upstream_commit']
        assert row['validation_only_selection'] is True and row['test_labels_accessed'] is False and row['test_scored'] is False
        assert row['heldout_test_pending'] is True and row['full_original_heterograph'] is True
        assert row['validation_evolution_rng_seed']==EVAL_SEED
        assert row['epochs_completed']<=100 and 1<=row['best_epoch']<=row['epochs_completed']
        assert row['val_count']==C['validation_nodes']
        assert sha(checkpoint)==row['checkpoint_sha256'],('CHECKPOINT HASH BAD',ratio,seed)
        assert sha(valfile)==row['validation_scores_sha256'],('VALIDATION HASH BAD',ratio,seed)
        vs=np.load(valfile,allow_pickle=False)
        assert len(vs)==len(yval) and np.isfinite(vs).all()
        val_metrics=metric(np.asarray(yval,dtype='int64'),np.asarray(vs,dtype='float64'),float(row['frozen_validation_threshold']))
        assert abs(val_metrics['auprc']-row['best_val_auprc'])<1e-8,('FROZEN VALIDATION AUPRC INVALID',ratio,seed)
        assert abs(val_metrics['auroc']-row['best_val_auroc'])<1e-8,('FROZEN VALIDATION AUROC INVALID',ratio,seed)
        assert abs(val_metrics['macro_f1']-row['best_val_macro_f1'])<1e-8,('FROZEN VALIDATION MACROF1 INVALID',ratio,seed)
        obj=torch.load(checkpoint,map_location='cpu')
        assert obj['ratio']==ratio and obj['seed']==seed and obj['epoch']==row['best_epoch']
        assert obj['config']==config and obj['frozen_winner_sha256']==winner_sha
        assert obj['source_sha256']==C['source_sha256'] and obj['split_sha256']==C['split_sha256']
        assert obj['test_scored'] is False and obj['theta_last_excluded'] is True
        assert 'model_state' in obj
        records.append((ratio,seed,row,checkpoint))
        del obj,vs
        print(f'{P}_{ratio}_Seed{seed}_FROZEN_PRETEST_VERIFIED=PASS',flush=True)
    assert len(records)==12
    # Inspect existing scoring claims BEFORE test access. No unsafe repeat after interrupted claimed inference.
    for ratio,seed,row,checkpoint in records:
        dest=OUT/ratio/f'seed_{seed}'
        claim=dest/'test_once_claim.json';score=dest/'test_scores.npy';result=dest/'test_metrics.json'
        if result.exists():
            assert claim.is_file() and score.is_file(),('INCOMPLETE EXISTING RESULT',ratio,seed)
            c=json.loads(claim.read_text());r=json.loads(result.read_text())
            assert c['checkpoint_sha256']==row['checkpoint_sha256'] and c['ratio']==ratio and c['seed']==seed
            assert r['checkpoint_sha256']==row['checkpoint_sha256'] and r['test_score_sha256']==sha(score)
            assert c['status']=='TEST_INFERENCE_COMPLETED_ONCE',('BAD TEST CLAIM',ratio,seed)
        elif claim.exists():
            if not score.exists():stop(f'STOP_UNRECOVERABLE_CLAIMED_WITHOUT_SCORES {ratio} {seed}: manual audit required, NO re-inference')
            assert json.loads(claim.read_text())['checkpoint_sha256']==row['checkpoint_sha256']
        elif score.exists():stop(f'UNCLAIMED TEST SCORE EXIST {ratio} {seed}')
    print(P+'_STAGE5_ALL_12_FROZEN_CHECKPOINTS_VALIDATION_SCORES_AND_TEST_GUARDS=PASS',flush=True)
    return K,g,x,device,records,config,winner_sha

def main():
    import socket
    assert socket.gethostname()==C['hostname'],('WRONG INSTANCE',socket.gethostname())
    assert torch.cuda.is_available()
    K,g,x,device,records,config,winner_sha=preflight()
    print(P+'_STAGE5_PRETEST_VALIDATIONS_COMPLETE=PASS TEST_LABELS_NOT_ACCESSED_YET=YES',flush=True)
    with np.load(SPLIT,allow_pickle=False) as z:
        testidx=np.asarray(z['test'],dtype='int64').copy()
        assert len(testidx)==C['test_nodes'] and len(np.unique(testidx))==len(testidx)
    # Load ONLY the label matrix, AFTER every checkpoint and validation selection is sealed.
    arr=sio.loadmat(SRC,variable_names=['label'])['label']
    yall=np.asarray(arr).reshape(-1)
    assert len(yall)==C['nodes']
    y=np.asarray(yall[testidx],dtype='int64').copy()
    del yall,arr
    assert set(np.unique(y).tolist())=={0,1}
    ti=torch.from_numpy(testidx).to(device)
    print(P+f'_STAGE5_HELDOUT_LABELS_OPENED_AFTER_ALL_12_FROZEN_CHECKS=PASS test_nodes={len(y)} test_positives={int(y.sum())}',flush=True)
    outputs=[]
    for ratio,seed,row,checkpoint in records:
        dest=OUT/ratio/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
        claim=dest/'test_once_claim.json';scorefile=dest/'test_scores.npy';result=dest/'test_metrics.json'
        if result.exists():
            saved=json.loads(result.read_text())
            scores=np.asarray(np.load(scorefile,allow_pickle=False),dtype='float64')
            calculated=metric(y,scores,float(row['frozen_validation_threshold']))
            for key,value in calculated.items():
                assert abs(saved[key]-value)<1e-10,('STORED TEST METRIC CHANGED',ratio,seed,key)
            outputs.append(saved)
            print(P+f'_{ratio}_Seed{seed}_PREVIOUS_ONE_TIME_TEST_RESULT_AUDITED=PASS',flush=True)
            continue
        if scorefile.exists():
            # Only offline metrics recovery is allowed; no additional model forward.
            c=json.loads(claim.read_text())
            assert c['checkpoint_sha256']==row['checkpoint_sha256']
            scores=np.asarray(np.load(scorefile,allow_pickle=False),dtype='float64')
            print(P+f'_{ratio}_Seed{seed}_OFFLINE_RESULT_RECOVERY_FROM_SAVED_SCORES=PASS',flush=True)
            inference_secs=c.get('inference_seconds')
            c['status']='TEST_INFERENCE_COMPLETED_ONCE'
            c['test_score_sha256']=sha(scorefile)
            atomjson(claim,c)
        else:
            if claim.exists():stop(f'TEST_ALREADY_CLAIMED_NO_SCORES {ratio} {seed}')
            c=dict(status='TEST_INFERENCE_CLAIMED_ONCE',dataset=C['dataset'],ratio=ratio,seed=seed,
                   checkpoint_sha256=row['checkpoint_sha256'],frozen_winner_sha256=winner_sha,
                   split_sha256=C['split_sha256'],validation_threshold=row['frozen_validation_threshold'],
                   test_inference_attempts=1,stochastic_eval_rng_seed=EVAL_SEED)
            # Exclusive disk claim is written BEFORE the only held-out forward pass.
            oneclaim(claim,c)
            print(P+f'_{ratio}_Seed{seed}_EXCLUSIVE_ONE_TIME_TEST_CLAIM=PASS',flush=True)
            torch.cuda.reset_peak_memory_stats(device)
            model=K(C['features'],config['hidden_dim'],2,d=config['order'],beta_p=config['beta_p'],population_size=config['population_size'],mutation_rate=config['mutation_rate']).to(device)
            obj=torch.load(checkpoint,map_location=device)
            missing=model.load_state_dict(obj['model_state'],strict=False)
            assert missing.missing_keys==['theta_last'] and not missing.unexpected_keys,missing
            model.eval()
            tic=time.perf_counter()
            # EOGFD draws evolutionary proposals in eval(); fix RNG exactly as Stage4 validation.
            with torch.random.fork_rng(devices=[0],enabled=True):
                torch.manual_seed(EVAL_SEED);torch.cuda.manual_seed_all(EVAL_SEED)
                with torch.no_grad():
                    logits=model(g,x)[0]
                    scores=F.softmax(logits[ti],dim=1)[:,1].cpu().numpy().astype('float64')
            inference_secs=round(time.perf_counter()-tic,5)
            assert len(scores)==len(y) and np.isfinite(scores).all()
            atomnpy(scorefile,scores)
            del logits,model,obj
            gc.collect();torch.cuda.empty_cache()
            c['inference_seconds']=inference_secs
            c['peak_gpu_allocated_gib']=round(torch.cuda.max_memory_allocated(device)/1073741824,6)
            c['test_score_sha256']=sha(scorefile)
            c['status']='TEST_INFERENCE_COMPLETED_ONCE'
            atomjson(claim,c)
        metrics=metric(y,scores,float(row['frozen_validation_threshold']))
        record=dict(status='ONE_TIME_FROZEN_HELDOUT_TEST_COMPLETE',model='EOGFDHeteroModel',dataset=C['dataset'],ratio=ratio,
                    seed=seed,checkpoint_sha256=row['checkpoint_sha256'],frozen_winner_sha256=winner_sha,
                    source_sha256=C['source_sha256'],split_sha256=C['split_sha256'],upstream_commit=C['upstream_commit'],
                    frozen_validation_threshold=row['frozen_validation_threshold'],best_epoch=row['best_epoch'],
                    best_validation_auprc=row['best_val_auprc'],test_count=len(y),test_positive_count=int(y.sum()),
                    inference_seconds=inference_secs,stochastic_eval_rng_seed=EVAL_SEED,
                    test_score_sha256=sha(scorefile),test_scored_once=True,train_or_tune_during_test=False,**metrics)
        atomjson(result,record)
        outputs.append(record)
        print(P+f'_{ratio}_Seed{seed}_ONE_TIME_FINAL_HELDOUT_TEST=PASS AUPRC={metrics["auprc"]:.6f} AUROC={metrics["auroc"]:.6f} macroF1={metrics["macro_f1"]:.6f}',flush=True)
        del scores
    assert len(outputs)==12
    groups=[]
    names=['auprc','auroc','macro_f1','fraud_f1','fraud_precision','fraud_recall','accuracy','balanced_accuracy','mcc','brier_score']
    for r in RATIOS:
        chunk=[row for row in outputs if row['ratio']==r]
        assert len(chunk)==3 and sorted(x['seed'] for x in chunk)==sorted(SEEDS)
        v={'ratio':r,'seeds':SEEDS}
        for key in names:
            data=np.asarray([row[key] for row in chunk],dtype='float64')
            v[key+'_mean']=float(np.mean(data));v[key+'_sample_sd']=float(np.std(data,ddof=1))
        groups.append(v)
        print(P+f'_{r}_FINAL_TEST_AUPRC_MEAN={v["auprc_mean"]:.6f} sampleSD={v["auprc_sample_sd"]:.6f} macroF1_MEAN={v["macro_f1_mean"]:.6f}',flush=True)
    summary=dict(status='EOGFD_12_HELDOUT_TESTS_COMPLETED_ONCE',dataset=C['dataset'],model='EOGFDHeteroModel',
                 original_upstream_model_implementation=True,author_identical_benchmark=False,author_source_commit=C['upstream_commit'],
                 training_protocol_adapted_frozen_nested_splits=True,source_sha256=C['source_sha256'],
                 split_sha256=C['split_sha256'],ratios=RATIOS,seeds=SEEDS,test_run_count=12,
                 validation_only_tuning_trials=12,final_training_count=12,all_model_thresholds_validation_frozen=True,
                 heldout_test_scoring_once_per_model=True,test_membership=C['test_nodes'],
                 test_fraud_count=int(y.sum()),summary_metrics=groups)
    atomjson(OUT/(P+'_Stage5_AllRatios_FinalHeldoutTestSummary.json'),summary)
    print(P+'_STAGE5_ALL_12_FINAL_HELDOUT_TESTS_COMPLETE=PASS TEST_SCORING_ONCE_PER_MODEL=YES',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print(P+'_STAGE5_STOP_ERROR='+repr(exc),flush=True)
        traceback.print_exc();sys.exit(2)
