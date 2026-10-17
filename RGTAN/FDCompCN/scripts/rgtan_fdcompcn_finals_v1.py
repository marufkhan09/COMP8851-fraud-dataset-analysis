#!/usr/bin/env python3
"""RGTAN FDCompCN 4 ratios x 3 seeds; audited validation-selected finals.

Uses existing, tested I2 RGTAN TR40 tuner Experiment implementation and 4
frozen cross-fit adapters. One durable O_EXCL test claim per run prevents
silent rescoring. No retuning or use of test labels before frozen validation.
"""
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

ROOT=Path('/workspace/comp8851')
SRC=ROOT/'repos/antifraud'
DIR=Path('/workspace/uploads/rgtan_fdcompcn_i2_incoming')
ASSETS=ROOT/'assets/rgtan/fdcompcn'
TUNE_DIR=ROOT/'results/rgtan/fdcompcn_tr40_tuning_v1'
TUNER=Path('/workspace/rgtan_fdcompcn_tr40_tune_fast_v1.py')
WINNER=TUNE_DIR/'winner.json'
OUT=ROOT/'results/rgtan/fdcompcn_finals_v1'
SPLIT=DIR/'fdcompcn_seed2_nested_splits.npz'
GRAPH=DIR/'comp.dgl'
PIN='65e583f03ac5c5983e49cc35179fe7a65908b9c1'
SOURCE_SHA='e252b9a6b619b28a7bf6d9f5b16aacc232d43207d87ac26cd5160fb0d8a98baa'
SPLIT_SHA='5cd7084342f4af48e9b82828dcac0e97879314cd5c9c010642b30c89e7c95168'
RATIOS=('TR40','TR30','TR20','TR10')
SEEDS=(2,42,72)
COUNT={'TR40':(2126,223),'TR30':(1594,167),'TR20':(1062,111),'TR10':(531,56)}
KNOWN={'TR40':'79c88a3421795dc4c420781567f7bdb9288b1ff1aee9ec551ba4b3f669878744',
       'TR30':'d1a7ae0cd5352b10499232b673d2f64b034651d1d7a4d575da45013827fa8e1e',
       'TR20':'4c60d021d6f274792dadb710b4b0938f302e69dd8141cffe77c5a714ac7ebc3f',
       'TR10':'51d43146af8c774966f13fd453fdb826b66e2f6afa20529814afcc9baa8ea6b2'}
WINNER_CONFIG={'hidden_dim':64,'drop':[.3,.15],'base_lr':.005}
WINNER_AUPRC=0.1764374792213923
N=5317


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def atomic_json(path,obj):
    path=Path(path)
    temp=path.with_name(path.name+'.partial')
    assert not temp.exists(), 'partial JSON needs audit: '+str(temp)
    with temp.open('w') as f:
        json.dump(obj,f,indent=2,sort_keys=True)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def atomic_pt(path,obj):
    path=Path(path)
    temp=path.with_name(path.name+'.partial')
    assert not temp.exists(), 'partial checkpoint needs audit: '+str(temp)
    torch.save(obj,temp)
    os.replace(temp,path)


def import_tuner():
    assert TUNER.is_file()
    spec=importlib.util.spec_from_file_location('fdcompcn_tune_existing_i2',str(TUNER))
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.PIN==PIN and mod.SOURCE_SHA==SOURCE_SHA and mod.SPLIT_SHA==SPLIT_SHA
    assert mod.ADAPTER_SHA==KNOWN['TR40']
    assert mod.N==N and mod.SEED==2 and mod.BATCH==256
    assert mod.MAX_EPOCHS==100 and mod.PATIENCE==20
    assert len(mod.GRID)==12 and mod.GRID[11]==WINNER_CONFIG
    assert mod.NAMES==['degree','riskstat','1hop_degree','2hop_degree','1hop_riskstat','2hop_riskstat']
    return mod


def asset_path(r):
    low=r.lower()
    return ASSETS/f'fdcompcn_{low}_crossfit.npz',ASSETS/f'fdcompcn_{low}_crossfit_manifest.json'


def preflight():
    assert subprocess.check_output(['git','-C',str(SRC),'rev-parse','HEAD'],text=True).strip()==PIN
    assert digest(GRAPH)==SOURCE_SHA and digest(SPLIT)==SPLIT_SHA
    assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1'
    assert torch.cuda.is_available()
    tune=import_tuner()
    with np.load(SPLIT,allow_pickle=False) as z:
        assert int(z['seed'].item())==2
        frozen={r:np.asarray(z[r],dtype=np.int64) for r in RATIOS}
        val=np.asarray(z['val'],dtype=np.int64)
        test=np.asarray(z['test'],dtype=np.int64)
        assert len(val)==1064 and len(test)==2127
        assert len(set(frozen['TR40'])|set(val)|set(test))==N
        assert len(set(frozen['TR40'])&set(val))==0 and len(set(frozen['TR40'])&set(test))==0 and len(set(val)&set(test))==0
        for r in RATIOS:
            assert len(frozen[r])==COUNT[r][0]
            assert len(set(frozen[r]))==COUNT[r][0]
            assert np.isin(frozen[r],frozen['TR40']).all()
    print('SOURCE_SPLITS_PINNED=PASS',flush=True)
    record=json.loads(WINNER.read_text())
    assert record['status']=='WINNER_FROZEN' and record['winner_trial']==12
    assert record['config']==WINNER_CONFIG and record['batch_size']==256
    assert record['source_sha256']==SOURCE_SHA and record['split_sha256']==SPLIT_SHA
    assert record['adapter_sha256']==KNOWN['TR40'] and record['test_scoring_performed'] is False
    assert abs(record['best_val_auprc']-WINNER_AUPRC)<1e-12
    rows=[]
    for i in range(1,13):
        base=TUNE_DIR/'tuning'/f'trial_{i:02d}'
        row=json.loads((base/'tuning.json').read_text())
        assert row['trial']==i and row['status']=='TUNING_COMPLETE'
        assert row['source_sha256']==SOURCE_SHA and row['split_sha256']==SPLIT_SHA
        assert row['adapter_sha256']==KNOWN['TR40']
        assert row['test_scoring_performed'] is False and row['test_used_for_selection'] is False
        assert digest(base/'best_val_auprc.pt')==row['checkpoint_sha256']
        rows.append(row)
    top=max(rows,key=lambda row:(row['best_val_auprc'],-row['trial']))
    assert top['trial']==12 and top['config']==WINNER_CONFIG
    assert abs(top['best_val_auprc']-record['best_val_auprc'])<1e-12
    print('FROZEN_WINNER_AND_12_TUNING_CHECKPOINTS=PASS',flush=True)
    graphs,_=dgl.load_graphs(str(GRAPH))
    assert len(graphs)==1
    base=graphs[0]
    assert base.num_nodes()==N and base.num_edges()==30752
    features=base.nodes['company'].data['feature'].cpu().numpy()
    assert features.shape==(N,57) and np.isfinite(features).all()
    for r in RATIOS:
        ap,mp=asset_path(r)
        assert digest(ap)==KNOWN[r]
        meta=json.loads(mp.read_text())
        assert meta['status']=='RGTAN_CROSSFIT_ADAPTER_PASS' and meta['ratio']==r
        assert meta['source_sha256']==SOURCE_SHA and meta['split_sha256']==SPLIT_SHA
        assert meta['artifact_sha256']==KNOWN[r] and meta['risk_attention_heads']==3
        assert meta['training_nodes']==COUNT[r][0] and meta['training_positives']==COUNT[r][1]
        assert meta['validation_nodes']==1064 and meta['test_nodes']==2127
        assert meta['validation_test_labels_used_in_features'] is False
        assert meta['training_target_fold_labels_excluded'] is True
        assert meta['normalization_fit_on_oof_training_only'] is True
        assert meta['source_edge_label_and_mask_fields_ignored'] is True
        assert meta['source_node_split_masks_ignored'] is True
        assert meta['training_started'] is False and meta['test_evaluated'] is False
        with np.load(ap,allow_pickle=False) as z:
            train=np.asarray(z['train_ids'],dtype=np.int64)
            label=np.asarray(z['train_labels'],dtype=np.int64)
            fold=np.asarray(z['fold_of_node'],dtype=np.int64)
            assert np.array_equal(train,frozen[r]) and np.array_equal(val,z['val_ids'])
            assert len(label)==len(train) and int(label.sum())==COUNT[r][1]
            assert np.array_equal(z['features'],features)
            assert set(fold[train])==set(range(5)) and np.all(fold[val]==-1) and np.all(fold[test]==-1)
            for j in range(6):
                ctx=np.asarray(z[f'label_context_{j}'])
                view=np.asarray(z[f'risk_view_{j}'])
                assert ctx.shape==(N,) and view.shape==(N,6) and np.isfinite(view).all()
                assert np.all(ctx[val]==2) and np.all(ctx[test]==2)
                expected=np.full(N,2,dtype=np.int64)
                mask=(fold[train]!=j) if j<5 else np.ones(len(train),dtype=bool)
                expected[train[mask]]=label[mask]
                assert np.array_equal(ctx,expected)
    print('ALL_FOUR_CROSSFIT_ADAPTERS_AND_LABEL_CONTEXTS=PASS',flush=True)
    assert not (OUT/'complete_summary.json').exists(), 'Existing aggregate requires manual audit'
    return tune, frozen, val, test, base, features


def prepare(r, frozen, val, test, base, features):
    ap,_=asset_path(r)
    with np.load(ap,allow_pickle=False) as z:
        train=np.array(z['train_ids'],dtype=np.int64,copy=True)
        label=np.array(z['train_labels'],dtype=np.int64,copy=True)
        fold=np.array(z['fold_of_node'],dtype=np.int64,copy=True)
        x=np.array(z['features'],dtype=np.float32,copy=True)
        views=[np.array(z[f'risk_view_{j}'],dtype=np.float32,copy=True) for j in range(6)]
        contexts=[np.array(z[f'label_context_{j}'],dtype=np.int64,copy=True) for j in range(6)]
    assert np.array_equal(train,frozen[r]) and np.array_equal(x,features)
    u,v=base.edges(etype=('company','homo','company'))
    adj=sp.csr_matrix((np.ones(len(u),dtype=np.float32),(u.numpy(),v.numpy())),shape=(N,N))
    adj=(adj+adj.T).tocsr();adj.sum_duplicates();adj.data[:]=1.0;adj.setdiag(1);adj.eliminate_zeros();adj.sort_indices()
    assert adj.nnz==12585 and (adj!=adj.T).nnz==0
    graph=dgl.from_scipy(adj)
    assert not len(graph.ndata) and not len(graph.edata)
    val_truth=base.nodes['company'].data['label'][val].cpu().numpy().astype(np.int64,copy=True)
    assert len(val_truth)==1064 and set(val_truth.tolist())=={0,1}
    # Test labels deliberately not indexed, read, or passed here.
    return (graph,x,train,label,val,val_truth,fold,views,contexts)


def metrics(y,score,threshold):
    pred=(score>=threshold).astype(np.int64)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel().tolist()
    return dict(auprc=float(average_precision_score(y,score)),auroc=float(roc_auc_score(y,score)),
                macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),
                fraud_f1=float(f1_score(y,pred,pos_label=1,zero_division=0)),
                precision=float(precision_score(y,pred,zero_division=0)),
                recall=float(recall_score(y,pred,zero_division=0)),
                accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                mcc=float(matthews_corrcoef(y,pred)),brier=float(brier_score_loss(y,score)),
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),threshold=float(threshold))


def threshold_on_validation(y,score):
    """Exact O(n log n) validation macro F1, prefer high threshold on ties."""
    assert len(y)==len(score) and set(np.unique(y))=={0,1}
    order=np.argsort(-score,kind='mergesort');p=score[order];c=np.cumsum(y[order].astype(np.int64))
    k=np.arange(1,len(y)+1,dtype=np.int64)
    inds=np.flatnonzero(np.r_[p[:-1]!=p[1:],True])
    tp=c[inds].astype(np.float64);fp=k[inds]-tp
    fn=int(y.sum())-tp;tn=len(y)-int(y.sum())-fp
    f1p=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=(2*tp+fp+fn)>0)
    f1n=np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tn),where=(2*tn+fp+fn)>0)
    macros=(f1p+f1n)/2
    nneg=len(y)-int(y.sum())
    allneg=nneg/(2*nneg+int(y.sum()))
    if allneg>float(macros.max())+1e-12:
        threshold=float(np.nextafter(float(p[0]),float('inf')))
        value=float(allneg)
    else:
        idx=int(np.argmax(macros))
        threshold=float(p[inds[idx]])
        value=float(macros[idx])
    check=float(f1_score(y,score>=threshold,average='macro',zero_division=0))
    assert abs(check-value)<1e-10,(check,value,threshold)
    return threshold,check


def infer(tune,exp,ids):
    exp.model.eval()
    scores=np.full(N,np.nan,dtype=np.float32)
    seen=[]
    with torch.no_grad():
        for batch in exp.loader(ids,False):
            target,logits=exp.predict(batch,5)
            assert np.all(np.isnan(scores[target]))
            scores[target]=torch.softmax(logits,dim=1)[:,1].cpu().numpy()
            seen.extend(target.tolist())
    assert len(seen)==len(ids) and set(seen)==set(ids.tolist())
    assert np.isfinite(scores[ids]).all()
    return scores[ids].astype(np.float64)


def run_one(tune,r,seed,frozen,val,test,base,features):
    folder=OUT/r/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    final=folder/'final.json'
    claim=folder/'test_evaluation_started.json'
    validation_file=folder/'validation_frozen.json'
    checkpoint=folder/'best_val_auprc.pt'
    latest=folder/'latest.pt'
    if final.exists():
        rec=json.loads(final.read_text())
        assert rec['status']=='FINAL_COMPLETE' and rec['ratio']==r and rec['seed']==seed
        assert rec['test_scoring_count']==1 and rec['test_used_for_model_selection'] is False
        assert rec['crossfit_sha256']==KNOWN[r] and rec['winner_sha256']==digest(WINNER)
        assert digest(checkpoint)==rec['checkpoint_sha256']
        assert digest(claim)==rec['test_evaluation_claim_sha256']
        assert digest(folder/'test_predictions_once.npz')==rec['test_predictions_sha256']
        print(f'FINAL_ALREADY_COMPLETE ratio={r} seed={seed} testAUPRC={rec["test"]["auprc"]:.6f}',flush=True)
        return rec
    assert not claim.exists(), f'Test already claimed without final; manual audit required: {claim}'
    assert not validation_file.exists(),f'Frozen validation without final requires manual audit: {validation_file}'
    assert not (folder/'test_predictions_once.npz').exists(), 'Orphan test scores; audit required'
    data=prepare(r,frozen,val,test,base,features)
    tune.SEED=seed
    tune.seed_all()
    exp=tune.Experiment(data,WINNER_CONFIG)
    start=1;best_ap=-1.0;best_auc=-1.0;best_epoch=0;wait=0;hist=[]
    started=time.perf_counter()
    if latest.exists():
        st=torch.load(latest,map_location=tune.DEVICE)
        assert st['ratio']==r and st['seed']==seed and st['config']==WINNER_CONFIG
        assert st['crossfit_sha256']==KNOWN[r] and st['winner_sha256']==digest(WINNER)
        assert digest(checkpoint)==st['best_sha256']
        exp.model.load_state_dict(st['model']);exp.optimizer.load_state_dict(st['optim'])
        best_ap=float(st['best_ap']);best_auc=float(st['best_auc']);best_epoch=st['best_epoch'];wait=st['wait'];hist=st['history']
        start=st['epoch']+1
        torch.set_rng_state(st['torch_rng']);torch.cuda.set_rng_state(st['cuda_rng'],device=tune.DEVICE)
        np.random.set_state(st['numpy_rng']);random.setstate(st['python_rng'])
        if wait>=tune.PATIENCE: start=tune.MAX_EPOCHS+1
        print(f'RESUME_FINAL ratio={r} seed={seed} epoch={start}',flush=True)
    else:
        assert not checkpoint.exists(), 'Best checkpoint without latest; manual audit required'
        print(f'START_FINAL ratio={r} seed={seed} config={WINNER_CONFIG}',flush=True)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start,tune.MAX_EPOCHS+1):
        tick=time.perf_counter()
        loss,steps=exp.epoch()
        ap,auc=exp.validate()
        if ap>best_ap+1e-12:
            best_ap,best_auc,best_epoch,wait=float(ap),float(auc),epoch,0
            atomic_pt(checkpoint,dict(model_state_dict={k:v.detach().cpu().clone() for k,v in exp.model.state_dict().items()},
                                     ratio=r,seed=seed,epoch=epoch,config=WINNER_CONFIG,
                                     crossfit_sha256=KNOWN[r],winner_sha256=digest(WINNER),best_val_auprc=best_ap))
        else: wait+=1
        hist.append(dict(epoch=epoch,loss=float(loss),optimizer_steps=int(steps),val_auprc=float(ap),
                         val_auroc=float(auc),seconds=round(time.perf_counter()-tick,3)))
        atomic_pt(latest,dict(ratio=r,seed=seed,config=WINNER_CONFIG,epoch=epoch,model=exp.model.state_dict(),
                              optim=exp.optimizer.state_dict(),best_ap=best_ap,best_auc=best_auc,best_ep=best_epoch,
                              wait=wait,history=hist,crossfit_sha256=KNOWN[r],winner_sha256=digest(WINNER),
                              best_sha256=digest(checkpoint),torch_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state(tune.DEVICE),numpy_rng=np.random.get_state(),python_rng=random.getstate()))
        print(f'FINAL ratio={r} seed={seed} epoch={epoch:03d}/{tune.MAX_EPOCHS} loss={loss:.5f} '
              f'valAUPRC={ap:.6f} best={best_ap:.6f}@{best_epoch} wait={wait}/{tune.PATIENCE} '
              f'seconds={hist[-1]["seconds"]:.2f}',flush=True)
        if wait>=tune.PATIENCE: break
    assert best_epoch>=1 and checkpoint.exists()
    best=torch.load(checkpoint,map_location=tune.DEVICE)
    assert best['ratio']==r and best['seed']==seed and best['config']==WINNER_CONFIG
    assert best['crossfit_sha256']==KNOWN[r] and best['winner_sha256']==digest(WINNER)
    assert abs(best['best_val_auprc']-best_ap)<1e-12
    exp.model.load_state_dict(best['model_state_dict'])
    val_scores=infer(tune,exp,val)
    val_y=data[5]
    assert abs(average_precision_score(val_y,val_scores)-best_ap)<1e-5
    threshold,macro=threshold_on_validation(val_y,val_scores)
    validation=metrics(val_y,val_scores,threshold)
    assert abs(validation['macro_f1']-macro)<1e-10
    atomic_json(validation_file,dict(ratio=r,seed=seed,best_epoch=best_epoch,best_val_auprc=best_ap,
                                     best_checkpoint_sha256=digest(checkpoint),threshold=threshold,
                                     threshold_selection='maximize validation macro F1 only; higher threshold wins ties',
                                     validation=validation,crossfit_sha256=KNOWN[r],winner_sha256=digest(WINNER),
                                     test_scoring_performed=False))
    print(f'VALIDATION_FROZEN ratio={r} seed={seed} valAUPRC={best_ap:.6f} '
          f'epoch={best_epoch} threshold={threshold:.8f} macroF1={macro:.6f}',flush=True)
    claim_payload=dict(ratio=r,seed=seed,checkpoint_sha256=digest(checkpoint),
                       frozen_validation_sha256=digest(validation_file),threshold=threshold,
                       source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
                       crossfit_sha256=KNOWN[r],winner_sha256=digest(WINNER))
    fd=os.open(str(claim),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f:
        json.dump(claim_payload,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(f'TEST_EVALUATION_CLAIMED ratio={r} seed={seed}',flush=True)
    # Exactly one held-out inference, after checkpoint, threshold and claim are frozen.
    test_scores=infer(tune,exp,test)
    # Load test truth ONLY here (after durable O_EXCL claim).
    test_truth=base.nodes['company'].data['label'][test].cpu().numpy().astype(np.int64,copy=True)
    assert set(test_truth.tolist())=={0,1}
    test_metrics=metrics(test_truth,test_scores,threshold)
    predfile=folder/'test_predictions_once.npz'
    assert not predfile.exists()
    np.savez_compressed(predfile,node_ids=test,score=test_scores.astype(np.float32),truth=test_truth)
    rec=dict(status='FINAL_COMPLETE',dataset='FDCompCN',model='RGTAN',ratio=r,seed=seed,
             frozen_split_seed=2,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,author_commit=PIN,
             winner_trial=12,config=WINNER_CONFIG,winner_sha256=digest(WINNER),crossfit_sha256=KNOWN[r],
             checkpoint_sha256=digest(checkpoint),best_epoch=best_epoch,max_epochs=tune.MAX_EPOCHS,
             patience=tune.PATIENCE,completed_epochs=len(hist),batch_size=tune.BATCH,
             effective_lr=float(exp.lr),train_count=len(data[2]),val_count=len(val),test_count=len(test),
             trained_with='five OOF label and risk contexts; one model',
             model_selection='maximum validation AUPRC',threshold_selection='maximum validation macro F1',
             test_scoring_count=1,test_used_for_model_selection=False,validation=validation,test=test_metrics,
             test_predictions_sha256=digest(predfile),test_evaluation_claim_sha256=digest(claim),
             peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
             elapsed_seconds=round(time.perf_counter()-started,3))
    atomic_json(folder/'history.json',hist)
    atomic_json(final,rec)
    latest.unlink(missing_ok=True)
    print(f'FINAL_COMPLETE ratio={r} seed={seed} testAUPRC={test_metrics["auprc"]:.6f} '
          f'testAUROC={test_metrics["auroc"]:.6f} macroF1={test_metrics["macro_f1"]:.6f}',flush=True)
    return rec


def status():
    for r in RATIOS:
        for seed in SEEDS:
            folder=OUT/r/f'seed_{seed}'
            if (folder/'final.json').exists():
                j=json.loads((folder/'final.json').read_text());s=f'COMPLETE testAUPRC={j["test"]["auprc"]:.6f}'
            elif (folder/'test_evaluation_started.json').exists(): s='TEST_CLAIM_PRESENT; MANUAL_AUDIT_REQUIRED'
            elif (folder/'latest.pt').exists(): s='TRAINING_RESUMABLE'
            else: s='PENDING'
            print(f'{r} seed={seed}: {s}')


def aggregate(rows):
    assert len(rows)==12
    for r in RATIOS:
        for seed in SEEDS:
            final=OUT/r/f'seed_{seed}'/'final.json'
            j=json.loads(final.read_text())
            assert j['status']=='FINAL_COMPLETE' and j['ratio']==r and j['seed']==seed
            assert j['test_scoring_count']==1 and j['test_used_for_model_selection'] is False
    rep=dict(status='TWELVE_FINALS_COMPLETE',model='RGTAN',dataset='FDCompCN',
             source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,winner_sha256=digest(WINNER),
             seeds=list(SEEDS),ratios=list(RATIOS),test_scored_once_per_run=True,
             by_ratio={r:dict(n=3,
                              test_auprc_mean=float(np.mean([j['test']['auprc'] for j in rows if j['ratio']==r])),
                              test_auprc_sd_sample=float(np.std([j['test']['auprc'] for j in rows if j['ratio']==r],ddof=1)),
                              test_auroc_mean=float(np.mean([j['test']['auroc'] for j in rows if j['ratio']==r])),
                              macro_f1_mean=float(np.mean([j['test']['macro_f1'] for j in rows if j['ratio']==r])))
                       for r in RATIOS})
    atomic_json(OUT/'complete_summary.json',rep)
    print('I2_FDCOMPCN_TWELVE_FINALS_COMPLETE=PASS',flush=True)
    print(json.dumps(rep,indent=2),flush=True)


def main():
    assert len(sys.argv)==2 and sys.argv[1] in ('preflight','finals','status')
    if sys.argv[1]=='status':status();return
    tune,frozen,val,test,base,features=preflight()
    if sys.argv[1]=='preflight':
        print('I2_FDCOMPCN_FINALS_PREFLIGHT=PASS',flush=True)
        print('TRAINING_STARTED=NO; TEST_EVALUATED=NO',flush=True)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[run_one(tune,r,seed,frozen,val,test,base,features) for r in RATIOS for seed in SEEDS]
    aggregate(rows)

if __name__=='__main__':main()
