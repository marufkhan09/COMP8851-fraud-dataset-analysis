#!/usr/bin/env python3
"""Assessment 3 RGTAN Amazon canonical finals, resuming exact TR40 tuning architecture.

Frozen source/splits (seed 2); validation-selected winner; training seeds 2,42,72.
Every run: train OOF label contexts, best validation-AUPRC checkpoint, validation-only
macro-F1 threshold, then exactly one test pass. A durable test claim prevents
silent rescoring if interrupted. Test labels are loaded only after claim.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from scipy.io import loadmat
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, brier_score_loss)

ROOT = Path('/workspace/comp8851')
WORK = Path('/workspace')
SRC = ROOT/'repos/antifraud'
TUNER_PATH = WORK/'rgtan_amazon_tr40_tuning_v1.py'
WINNER = ROOT/'results/rgtan/amazon_tr40_tuning_v1/winner.json'
SPLIT = Path('/workspace/uploads/ass2_split_candidates/Amazon/amazon_seed2_nested_splits.npz')
MAT = Path('/root/.dgl/amazon_1e446145/Amazon.mat')
ASSET_DIR = ROOT/'assets/rgtan/amazon'
OUT = ROOT/'results/rgtan/amazon_finals_v2'
SOURCE_SHA = '4b7e3f9cccc62b736792707393ccd74332a1a0592dba128ac6b2989bf1ee9d63'
SPLIT_SHA = '0fd96816c440b247b8cac768896bc454b1ecf4abebaf5001c1a3e1d900efa4fd'
TR40_ADAPTER_SHA = 'fd3f871171c795999e6a1470857b270d5a605bb3f7cb3c3ea296cd5b3d1c6efe'
AUTHOR_COMMIT = '65e583f03ac5c5983e49cc35179fe7a65908b9c1'
RATIOS = ('TR40', 'TR30', 'TR20', 'TR10')
SEEDS = (2,42,72)
EXPECTED_TRAIN = {'TR40':3455,'TR30':2591,'TR20':1727,'TR10':863}
EXPECTED_CONFIG = {'hidden_dim':32, 'drop':[.2,.1], 'base_lr':.001}
FINAL_IDS = [(r,seed) for r in RATIOS for seed in SEEDS]
BATCH, MAX_EPOCHS, PATIENCE, N = 32, 100, 20, 11944
DEVICE = 'cuda:0'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for ch in iter(lambda:f.read(4*1024*1024),b''): h.update(ch)
    return h.hexdigest()


def atomic_json(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('w') as f: json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')
    os.replace(tmp,path)


def save_torch(path,obj):
    tmp=str(path)+'.partial'
    torch.save(obj,tmp)
    os.replace(tmp,str(path))


def assert_dataset():
    import subprocess
    assert digest(MAT)==SOURCE_SHA, 'Amazon source sha mismatch'
    assert digest(SPLIT)==SPLIT_SHA, 'Amazon frozen split sha mismatch'
    head=subprocess.check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip()
    assert head==AUTHOR_COMMIT, ('author source mismatch',head)
    assert torch.__version__=='1.12.1+cu113', torch.__version__
    assert dgl.__version__=='0.8.1', dgl.__version__
    assert torch.cuda.is_available(), 'CUDA unavailable'
    with np.load(SPLIT,allow_pickle=False) as z:
        assert int(z['seed'])==2
        tr40=np.asarray(z['TR40'],dtype=np.int64)
        val=np.asarray(z['val'],dtype=np.int64)
        test=np.asarray(z['test'],dtype=np.int64)
        assert len(val)==1728 and len(test)==3456
        assert len(np.unique(np.concatenate([tr40,val,test])))==8639
        assert set(tr40).isdisjoint(val) and set(tr40).isdisjoint(test) and set(val).isdisjoint(test)
        for ratio in RATIOS:
            t=np.asarray(z[ratio],dtype=np.int64)
            assert len(t)==EXPECTED_TRAIN[ratio] and len(np.unique(t))==len(t)
            assert np.isin(t,tr40).all() and t.min()>=3305
    assert digest(ASSET_DIR/'amazon_tr40_crossfit.npz')==TR40_ADAPTER_SHA
    print('FINALS_SOURCE_SPLIT_AND_TR40_ASSET=PASS',flush=True)


def assert_winner():
    rec=json.loads(WINNER.read_text())
    assert rec=={
        'status':'WINNER_FROZEN','ratio':'TR40','split_seed':2,'seed':2,
        'winner_trial':1,'config':EXPECTED_CONFIG,
        'best_val_auprc':0.922785248270213,
        'source_sha256':SOURCE_SHA,'split_sha256':SPLIT_SHA,
        'adapter_sha256':TR40_ADAPTER_SHA,'test_scoring_performed':False
    }, 'winner differs from audited frozen record'
    trial_results=[]
    for i in range(1,13):
        trial_dir=ROOT/'results/rgtan/amazon_tr40_tuning_v1/tuning'/f'trial_{i:02d}'
        report=json.loads((trial_dir/'tuning.json').read_text())
        assert report['trial']==i and report['status']=='TUNING_COMPLETE'
        assert report['adapter_sha256']==TR40_ADAPTER_SHA
        assert report['split_sha256']==SPLIT_SHA and report['source_sha256']==SOURCE_SHA
        assert digest(trial_dir/'best_val_auprc.pt')==report['checkpoint_sha256']
        trial_results.append(report)
    winner=max(trial_results,key=lambda d:(d['best_val_auprc'],-d['trial']))
    assert winner['trial']==1 and winner['config']==EXPECTED_CONFIG
    assert abs(winner['best_val_auprc']-rec['best_val_auprc'])<1e-12
    print('FROZEN_WINNER_AND_12_TUNING_CHECKPOINTS=PASS',flush=True)
    return rec


def asset_paths(ratio):
    low=ratio.lower()
    return ASSET_DIR/f'amazon_{low}_crossfit.npz',ASSET_DIR/f'amazon_{low}_crossfit_manifest.json'


def verify_assets(need_all=True):
    with np.load(SPLIT,allow_pickle=False) as z:
        splits={r:np.asarray(z[r],dtype=np.int64) for r in RATIOS}
        val=np.asarray(z['val'],dtype=np.int64)
        test=np.asarray(z['test'],dtype=np.int64)
    hashes={}
    for ratio in RATIOS:
        a,m=asset_paths(ratio)
        if not a.exists() or not m.exists():
            if need_all: raise AssertionError(f'Crossfit not yet built for {ratio}')
            print('CROSSFIT_PENDING='+ratio,flush=True)
            continue
        rec=json.loads(m.read_text())
        ah=digest(a)
        assert ah==rec['artifact_sha256'],ratio
        assert rec['status']=='RGTAN_CROSSFIT_ADAPTER_PASS',ratio
        assert rec['ratio']==ratio and rec['split_seed']==2,ratio
        assert rec['source_sha256']==SOURCE_SHA and rec['split_sha256']==SPLIT_SHA,ratio
        assert rec['train_count']==EXPECTED_TRAIN[ratio] and rec['val_count']==1728 and rec['test_count']==3456
        assert rec['fold_labels_masked_from_their_own_risk_views'] is True
        assert rec['validation_test_labels_used_in_features'] is False
        assert rec['normalization_fit_on_oof_training_only'] is True
        with np.load(a,allow_pickle=False) as z:
            train=np.asarray(z['train_ids'],dtype=np.int64)
            y=np.asarray(z['train_labels'],dtype=np.int64)
            f=np.asarray(z['features'],dtype=np.float32)
            folds=np.asarray(z['fold_of_node'],dtype=np.int64)
            assert np.array_equal(train,splits[ratio]) and np.array_equal(val,z['val_ids'])
            assert len(y)==len(train) and set(y.tolist())=={0,1}
            assert f.shape==(11944,25) and np.isfinite(f).all()
            assert np.array_equal(np.unique(folds[train]),np.arange(5))
            assert (folds[np.setdiff1d(np.arange(N),train)]==-1).all()
            names=[str(x) for x in z['risk_feature_names']]
            assert names==['degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat']
            for k in range(6):
                context=np.asarray(z[f'label_context_{k}'],dtype=np.int64)
                view=np.asarray(z[f'risk_view_{k}'],dtype=np.float32)
                assert context.shape==(N,) and view.shape==(N,6)
                assert np.isfinite(view).all()
                assert (context[val]==2).all() and (context[test]==2).all() and (context[:3305]==2).all()
                assert (context[np.setdiff1d(np.arange(N),train)]==2).all()
                if k<5:
                    assert (context[train[folds[train]==k]]==2).all()
                    ids=train[folds[train]!=k]
                    assert np.array_equal(context[ids],y[folds[train]!=k])
                else:
                    assert np.array_equal(context[train],y)
        hashes[ratio]=ah
        print(f'CROSSFIT_{ratio}=PASS hash={ah} train={len(train)} fraud={int(y.sum())}',flush=True)
    return hashes


def preflight(need_all=False):
    assert_dataset()
    assert_winner()
    hashes=verify_assets(need_all)
    return hashes


def graph_and_data(ratio):
    # Exact same adjacency reconstruction and graph conditions as tuning v1.
    with np.load(asset_paths(ratio)[0],allow_pickle=False) as z:
        features=z['features'].astype(np.float32)
        train=z['train_ids'].astype(np.int64)
        y_train=z['train_labels'].astype(np.int64)
        val=z['val_ids'].astype(np.int64)
        fold=z['fold_of_node'].astype(np.int64)
        views=[z[f'risk_view_{i}'].astype(np.float32) for i in range(6)]
        contexts=[z[f'label_context_{i}'].astype(np.int64) for i in range(6)]
        names=[str(x) for x in z['risk_feature_names']]
    with np.load(SPLIT,allow_pickle=False) as z:
        test=np.asarray(z['test'],dtype=np.int64)
    # Load ONLY validation truth for checkpoint and threshold selection.
    raw_labels=np.asarray(loadmat(MAT,variable_names=['label'])['label']).reshape(-1)
    val_y=np.array(raw_labels[val],dtype=np.int64,copy=True)
    assert np.array_equal(raw_labels[train].astype(np.int64),y_train)
    del raw_labels
    assert set(val_y.tolist())=={0,1}
    raw=loadmat(MAT,variable_names=['homo'])
    adj=sp.csr_matrix(raw['homo'],dtype=np.float32)
    del raw
    adj=(adj+adj.T).tocsr()
    adj.sum_duplicates()
    adj.data[:]=1.0
    adj=adj.tolil()
    adj.setdiag(1.0)
    adj=adj.tocsr()
    adj.eliminate_zeros()
    adj.sort_indices()
    assert adj.nnz==8808728
    graph=dgl.from_scipy(adj)
    del adj
    assert graph.num_nodes()==N and len(graph.ndata)==0
    return (graph,features,train,y_train,val,val_y,fold,views,contexts,names),test


def import_tuner():
    import importlib.util
    spec=importlib.util.spec_from_file_location('rgtan_original_tune_i2',str(TUNER_PATH))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SOURCE_SHA==SOURCE_SHA and mod.SPLIT_SHA==SPLIT_SHA
    assert mod.ADAPTER_SHA==TR40_ADAPTER_SHA and mod.SEED==2
    assert mod.BATCH==BATCH and mod.MAX_EPOCHS==MAX_EPOCHS and mod.PATIENCE==PATIENCE
    assert EXPECTED_CONFIG in mod.GRID
    return mod


def metric_with_threshold(truth,probs,threshold):
    pred=(probs>=threshold).astype(np.int64)
    tn,fp,fn,tp=confusion_matrix(truth,pred,labels=[0,1]).ravel().tolist()
    return dict(auprc=float(average_precision_score(truth,probs)),
                auroc=float(roc_auc_score(truth,probs)),
                macro_f1=float(f1_score(truth,pred,average='macro',zero_division=0)),
                fraud_f1=float(f1_score(truth,pred,pos_label=1,zero_division=0)),
                precision=float(precision_score(truth,pred,zero_division=0)),
                recall=float(recall_score(truth,pred,zero_division=0)),
                accuracy=float(accuracy_score(truth,pred)),
                balanced_accuracy=float(balanced_accuracy_score(truth,pred)),
                mcc=float(matthews_corrcoef(truth,pred)),
                brier=float(brier_score_loss(truth,probs)),
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),threshold=float(threshold))


def choose_validation_threshold(y,scores):
    # O(N log N) exact macro-F1 for every distinct observed threshold + all-negative.
    assert len(y)==len(scores) and set(np.unique(y))=={0,1}
    assert np.isfinite(scores).all()
    order=np.argsort(-scores,kind='mergesort')
    p=scores[order]
    positives=y[order].astype(np.int64)
    truepos=np.cumsum(positives)
    selected=np.arange(1,len(y)+1,dtype=np.int64)
    total_pos=int(y.sum()); total_neg=len(y)-total_pos
    keep=np.r_[p[:-1]!=p[1:],True]
    idx=np.flatnonzero(keep)
    tp=truepos[idx].astype(np.float64)
    fp=selected[idx]-tp
    fn=total_pos-tp
    tn=total_neg-fp
    f1pos=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=(2*tp+fp+fn)>0)
    f1neg=np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tn),where=(2*tn+fp+fn)>0)
    macros=(f1pos+f1neg)/2
    # Include all-negative option with threshold > largest score.
    allneg=(total_neg/(2*total_neg+total_pos)) if (2*total_neg+total_pos)>0 else 0.0
    if allneg>float(macros.max())+1e-12:
        threshold=float(np.nextafter(float(p[0]),float('inf')))
        selection_score=float(allneg)
    else:
        # Tie: highest threshold first, deterministic.
        j=int(np.argmax(macros))
        threshold=float(p[idx[j]])
        selection_score=float(macros[j])
    check=float(f1_score(y,(scores>=threshold).astype(int),average='macro',zero_division=0))
    assert abs(check-selection_score)<1e-10, (check,selection_score,threshold)
    return threshold,check


def predict_scores(exp, ids):
    exp.model.eval()
    probs=np.full(N,np.nan,dtype=np.float32)
    observed=[]
    with torch.no_grad():
        for batch in exp.loader(ids,shuffle=False):
            batch_ids,logits=exp.predict(batch,5)
            assert np.all(np.isnan(probs[batch_ids]))
            probs[batch_ids]=torch.softmax(logits,dim=1)[:,1].cpu().numpy()
            observed.extend(batch_ids.tolist())
    assert len(observed)==len(ids) and set(observed)==set(ids.tolist())
    assert np.isfinite(probs[ids]).all()
    return probs[ids].astype(np.float64)


def run_one(ratio,seed,tune):
    directory=OUT/ratio/f'seed_{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    final=directory/'final.json'
    if final.exists():
        report=json.loads(final.read_text())
        assert report['status']=='FINAL_COMPLETE' and report['ratio']==ratio and report['seed']==seed
        assert report['checkpoint_sha256']==digest(directory/'best_val_auprc.pt')
        assert report['source_sha256']==SOURCE_SHA and report['split_sha256']==SPLIT_SHA
        assert report['crossfit_sha256']==digest(asset_paths(ratio)[0])
        assert report['winner_sha256']==digest(WINNER)
        print(f'FINAL_ALREADY_COMPLETE ratio={ratio} seed={seed} testAUPRC={report["test"]["auprc"]:.6f}',flush=True)
        return report
    # Test score claim protects exactly-once evaluation in this runner.
    claim=directory/'test_evaluation_started.json'
    assert not claim.exists(), f'Existing test claim; manual audit needed before any retry: {claim}'
    data,test_ids=graph_and_data(ratio)
    tune.SEED=seed
    exp=tune.Experiment(data,EXPECTED_CONFIG)
    ckpt=directory/'best_val_auprc.pt'
    latest=directory/'latest.pt'
    best_ap,best_auc,best_ep,wait=-1.0,-1.0,0,0
    history=[]
    start_epoch=1
    started_at=time.perf_counter()
    ash=digest(asset_paths(ratio)[0])
    wsha=digest(WINNER)
    if latest.exists():
        st=torch.load(latest,map_location=DEVICE)
        assert st['ratio']==ratio and st['seed']==seed
        assert st['config']==EXPECTED_CONFIG
        assert st['crossfit_sha256']==ash and st['winner_sha256']==wsha
        exp.model.load_state_dict(st['model'])
        exp.optimizer.load_state_dict(st['optimizer'])
        start_epoch=st['epoch']+1
        best_ap,best_auc,best_ep,wait=st['best_val_auprc'],st['best_val_auroc'],st['best_epoch'],st['patience_wait']
        history=st['history']
        assert ckpt.exists() and digest(ckpt)==st['best_checkpoint_sha256']
        torch.set_rng_state(st['torch_rng'])
        torch.cuda.set_rng_state(st['cuda_rng'],device=DEVICE)
        np.random.set_state(st['numpy_rng'])
        random.setstate(st['python_rng'])
        if wait>=PATIENCE: start_epoch=MAX_EPOCHS+1
        print(f'FINAL_RESUME ratio={ratio} seed={seed} epoch={start_epoch}',flush=True)
    else:
        assert not ckpt.exists(), 'Checkpoint without resumable state; manual audit required'
        print(f'FINAL_STARTED ratio={ratio} seed={seed} winner_trial=1 config={EXPECTED_CONFIG}',flush=True)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,MAX_EPOCHS+1):
        t0=time.perf_counter()
        loss,steps=exp.epoch()
        ap,auc=exp.validate()
        if ap>best_ap+1e-12:
            best_ap,best_auc,best_ep,wait=float(ap),float(auc),epoch,0
            save_torch(ckpt,dict(
                model_state_dict={k:v.detach().cpu().clone() for k,v in exp.model.state_dict().items()},
                ratio=ratio,seed=seed,config=EXPECTED_CONFIG,epoch=epoch,
                best_val_auprc=best_ap,best_val_auroc=best_auc,
                crossfit_sha256=ash,winner_sha256=wsha))
        else: wait+=1
        rec=dict(epoch=epoch,loss=loss,optimizer_steps=steps,val_auprc=float(ap),
                 val_auroc=float(auc),seconds=round(time.perf_counter()-t0,3))
        history.append(rec)
        save_torch(latest,dict(ratio=ratio,seed=seed,config=EXPECTED_CONFIG,epoch=epoch,
            model=exp.model.state_dict(),optimizer=exp.optimizer.state_dict(),
            best_val_auprc=best_ap,best_val_auroc=best_auc,best_epoch=best_ep,patience_wait=wait,
            history=history,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
            winner_sha256=wsha,crossfit_sha256=ash,best_checkpoint_sha256=digest(ckpt),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(DEVICE),
            numpy_rng=np.random.get_state(),python_rng=random.getstate()))
        print(f'FINAL ratio={ratio} seed={seed} epoch={epoch:03d}/{MAX_EPOCHS} '
              f'loss={loss:.5f} valAUPRC={ap:.6f} best={best_ap:.6f}@{best_ep} '
              f'wait={wait}/{PATIENCE} seconds={rec["seconds"]:.1f}',flush=True)
        if wait>=PATIENCE:
            print(f'EARLY_STOP=YES ratio={ratio} seed={seed} epoch={epoch}',flush=True)
            break
    assert ckpt.exists() and best_ep>=1
    # Freeze checkpoint/threshold; do not use test truth or predictions here.
    best=torch.load(ckpt,map_location=DEVICE)
    assert best['ratio']==ratio and best['seed']==seed and best['config']==EXPECTED_CONFIG
    assert abs(best['best_val_auprc']-best_ap)<1e-12
    exp.model.load_state_dict(best['model_state_dict'])
    val_scores=predict_scores(exp,data[4])
    val_truth=data[5].astype(np.int64)
    assert abs(float(average_precision_score(val_truth,val_scores))-best_ap)<1e-5
    threshold,val_macro=choose_validation_threshold(val_truth,val_scores)
    validation=metric_with_threshold(val_truth,val_scores,threshold)
    assert abs(validation['macro_f1']-val_macro)<1e-10
    atomic_json(directory/'validation_frozen.json',dict(
        ratio=ratio,seed=seed,best_epoch=best_ep,best_val_auprc=best_ap,
        best_checkpoint_sha256=digest(ckpt),threshold=threshold,
        threshold_selection='maximize validation macro F1 only; ties prefer higher threshold',
        validation=validation,crossfit_sha256=ash,winner_sha256=wsha,
        test_scoring_performed=False))
    print(f'VALIDATION_FROZEN ratio={ratio} seed={seed} bestAUPRC={best_ap:.6f} '
          f'epoch={best_ep} threshold={threshold:.8f} macroF1={val_macro:.6f}',flush=True)
    # O_EXCL durable marker ensures no rerun silently scores test twice.
    claim_data=dict(ratio=ratio,seed=seed,checkpoint_sha256=digest(ckpt),
                    frozen_validation_file_sha256=digest(directory/'validation_frozen.json'),
                    threshold=threshold,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
                    crossfit_sha256=ash,winner_sha256=wsha)
    flags=os.O_CREAT|os.O_EXCL|os.O_WRONLY
    fd=os.open(str(claim),flags,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(claim_data,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(f'TEST_EVALUATION_CLAIMED ratio={ratio} seed={seed}',flush=True)
    # Single test scoring. Only now load test truth. Never feed it to the model.
    test_scores=predict_scores(exp,test_ids)
    raw_truth=np.asarray(loadmat(MAT,variable_names=['label'])['label']).reshape(-1)
    test_y=np.asarray(raw_truth[test_ids],dtype=np.int64).copy()
    del raw_truth
    assert set(test_y.tolist())=={0,1}
    metrics=metric_with_threshold(test_y,test_scores,threshold)
    np.savez_compressed(directory/'test_predictions_once.npz',
                        node_ids=test_ids,score=test_scores.astype(np.float32),truth=test_y)
    report=dict(status='FINAL_COMPLETE',dataset='Amazon',model='RGTAN',ratio=ratio,seed=seed,
                frozen_split_seed=2,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
                author_commit=AUTHOR_COMMIT,winner_trial=1,config=EXPECTED_CONFIG,
                winner_sha256=wsha,crossfit_sha256=ash,checkpoint_sha256=digest(ckpt),
                best_epoch=best_ep,max_epochs=MAX_EPOCHS,patience=PATIENCE,
                completed_epochs=len(history),batch_size=BATCH,
                effective_lr=exp.learning_rate,train_count=len(data[2]),
                val_count=len(data[4]),test_count=len(test_ids),
                trained_with='five out-of-fold label and risk contexts; one model',
                model_selection='maximum validation AUPRC',
                threshold_selection='maximum validation macro F1',
                test_scoring_count=1,test_used_for_model_selection=False,
                validation=validation,test=metrics,
                test_predictions_sha256=digest(directory/'test_predictions_once.npz'),
                test_evaluation_claim_sha256=digest(claim),
                peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                elapsed_seconds=round(time.perf_counter()-started_at,3))
    atomic_json(directory/'history.json',history)
    atomic_json(final,report)
    latest.unlink(missing_ok=True)
    print(f'FINAL_COMPLETE ratio={ratio} seed={seed} testAUPRC={metrics["auprc"]:.6f} '
          f'testAUROC={metrics["auroc"]:.6f} macroF1={metrics["macro_f1"]:.6f}',flush=True)
    return report


def summary_status():
    for ratio,seed in FINAL_IDS:
        folder=OUT/ratio/f'seed_{seed}'
        result=folder/'final.json'
        partial=folder/'latest.pt'
        claimed=folder/'test_evaluation_started.json'
        if result.exists():
            j=json.loads(result.read_text())
            state=f'COMPLETE testAUPRC={j["test"]["auprc"]:.6f}'
        elif claimed.exists(): state='TEST_CLAIM_PRESENT; MANUAL_AUDIT_REQUIRED'
        elif partial.exists(): state='TRAINING_RESUMABLE'
        else: state='PENDING'
        print(f'{ratio} seed={seed} {state}')


def aggregate():
    rows=[]
    for ratio,seed in FINAL_IDS:
        f=OUT/ratio/f'seed_{seed}'/'final.json'
        assert f.exists(),f'Not all 12 finals complete: {f}'
        j=json.loads(f.read_text())
        assert j['status']=='FINAL_COMPLETE' and j['ratio']==ratio and j['seed']==seed
        assert j['test_scoring_count']==1
        rows.append(j)
    report=dict(status='TWELVE_FINALS_COMPLETE',model='RGTAN',dataset='Amazon',
                source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
                winner_sha256=digest(WINNER),seeds=list(SEEDS),
                ratios=list(RATIOS),test_scored_once_per_run=True,
                by_ratio={r:{'n':3,
                    'test_auprc_mean':float(np.mean([x['test']['auprc'] for x in rows if x['ratio']==r])),
                    'test_auprc_sd_sample':float(np.std([x['test']['auprc'] for x in rows if x['ratio']==r],ddof=1)),
                    'test_auroc_mean':float(np.mean([x['test']['auroc'] for x in rows if x['ratio']==r])),
                    'macro_f1_mean':float(np.mean([x['test']['macro_f1'] for x in rows if x['ratio']==r]))}
                    for r in RATIOS})
    atomic_json(OUT/'complete_summary.json',report)
    print('RGTAN_AMAZON_TWELVE_FINALS_COMPLETE=PASS',flush=True)
    print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['preflight','pilot','finals','status'])
    a=p.parse_args()
    if a.mode=='status':
        summary_status()
        return
    preflight(need_all=a.mode!='preflight')
    print('FINALS_PREFLIGHT=PASS',flush=True)
    if a.mode=='preflight':
        print('TRAINING_STARTED=NO; TEST_EVALUATED=NO',flush=True)
        return
    tune=import_tuner()
    if a.mode=='pilot':
        run_one('TR40',2,tune)
        return
    for r,seed in FINAL_IDS: run_one(r,seed,tune)
    aggregate()

if __name__=='__main__':
    main()
