#!/usr/bin/env python3
"""EOGFD T-Finance: frozen TR40 winner, 4 ratios x 3 seeds, validation only.

Original EOGFD model source untouched. Same canonical full graph and exact frozen
nested splits. This is the unified COMP8851 input/split adapter, not an exact
reproduction of the author's dataset protocol. No held-out labels or scoring.
"""
from __future__ import annotations
import argparse, datetime, gc, hashlib, importlib.util, json, os, random, socket, statistics, subprocess, sys, traceback, uuid
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/workspace/comp8851')
S3PATH=Path('/workspace/EOGFD_TFINANCE_STAGE3_I1_20261023/stage3_tfinance.py')
S3RESULTS=ROOT/'results/eogfd/tfinance/stage3_tr40_tuning'
OUT=ROOT/'results/eogfd/tfinance/stage4_frozen_finals'
RATIOS=('TR40','TR30','TR20','TR10')
SEEDS=(2,42,72)


def checked_stage3():
    assert socket.gethostname()=='60d39211b1fd','STOP_WRONG_INSTANCE'
    assert S3PATH.is_file()
    manifests=Path('/workspace/EOGFD_TFINANCE_STAGE3_I1_20261023/KIT_SHA256SUMS.txt').read_text().splitlines()
    s3_hash=[x.split()[0] for x in manifests if x.split()[-1]=='stage3_tfinance.py']
    assert len(s3_hash)==1 and sha(S3PATH)==s3_hash[0],'STOP_STAGE3_SOURCE_CHANGED'
    spec=importlib.util.spec_from_file_location('stage3_tfinance_verified',S3PATH)
    s3=importlib.util.module_from_spec(spec);spec.loader.exec_module(s3)
    ids=s3.validate_assets()
    assert tuple(s3.RATIOS)==RATIOS and tuple(s3.SEEDS)==SEEDS
    assert s3.MAX_EPOCHS==100 and s3.PATIENCE==20 and s3.EVAL_SEED==100921
    assert s3.RESULTS==S3RESULTS
    manifest=json.loads((S3RESULTS/'stage3_run_manifest.json').read_text())
    assert manifest['source_sha256']==s3.SOURCE_SHA and manifest['split_sha256']==s3.SPLIT_SHA
    assert manifest['test_scored'] is False and manifest['test_labels_indexed'] is False
    trials=[]
    for i in range(1,13):
        folder=S3RESULTS/f'trial_{i:02d}'
        r=json.loads((folder/'trial_result.json').read_text())
        assert r['trial']==i and r['config']==s3.GRID[i-1]
        assert r['source_sha256']==s3.SOURCE_SHA and r['split_sha256']==s3.SPLIT_SHA
        assert r['test_scored'] is False and r['test_labels_indexed'] is False
        for fname,key in [('best_validation.pt','checkpoint_sha256'),('best_validation_scores.npy','validation_scores_sha256'),('validation_history.json','history_sha256'),('epoch_times.csv','epoch_times_sha256')]:
            assert sha(folder/fname)==r[key],('STAGE3_HASH_MISMATCH',fname,i)
        assert not (folder/'epoch_recovery.pt').exists(),'UNFINISHED_STAGE3_TRIAL'
        trials.append(r)
    winner=max(trials,key=lambda x:(x['best_val_auprc'],-x['trial']))
    frozenpath=S3RESULTS/'EOGFD_TFinance_TR40_FrozenWinner.json'
    frozen=json.loads(frozenpath.read_text())
    assert frozen['status']=='TR40_12_VALIDATION_ONLY_TRIALS_COMPLETE'
    assert frozen['winner_trial']==8 and winner['trial']==8
    assert abs(frozen['winner_val_auprc']-0.883130)<0.000001
    assert frozen['winner_config']==winner['config']
    assert frozen['winner_checkpoint_sha256']==winner['checkpoint_sha256']
    assert frozen['source_sha256']==s3.SOURCE_SHA and frozen['split_sha256']==s3.SPLIT_SHA
    assert frozen['test_scored'] is False and frozen['venus_methodological_approval_recorded'] is False
    assert sha(S3RESULTS/'trial_08/best_validation.pt')==frozen['winner_checkpoint_sha256']
    print(f'I1_TFINANCE_STAGE4_FROZEN_WINNER_AUDIT=PASS winner=8 validation_AUPRC={winner["best_val_auprc"]:.6f}',flush=True)
    return s3,ids,frozen,sha(frozenpath)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()


def clean_resume(s3, r, cfg, ratio, seed, dest, locksha):
    assert r['ratio']==ratio and r['seed']==seed and r['config']==cfg
    assert r['frozen_winner_sha256']==locksha
    assert r['source_sha256']==s3.SOURCE_SHA and r['split_sha256']==s3.SPLIT_SHA
    assert r['upstream_commit']==s3.PIN and r['test_scored'] is False and r['test_labels_indexed'] is False
    for path,field in [('best_validation.pt','checkpoint_sha256'),('best_validation_scores.npy','validation_scores_sha256'),('validation_history.json','history_sha256'),('epoch_times.csv','epoch_times_sha256')]:
        assert sha(dest/path)==r[field],('STOP_FROZEN_RUN_FILE_CHANGED',ratio,seed,path)
    recovery=dest/'epoch_recovery.pt'
    if recovery.exists():
        old=torch.load(recovery,map_location='cpu')
        assert old['ratio']==ratio and old['seed']==seed and old['config']==cfg
        assert old['frozen_winner_sha256']==locksha
        assert old['source_sha256']==s3.SOURCE_SHA and old['split_sha256']==s3.SPLIT_SHA
        assert old['test_scored'] is False and len(old['history'])==r['epochs_completed']
        recovery.unlink()
        print(f'I1_TFINANCE_STAGE4_{ratio}_SEED{seed}_POSTCOMMIT_RECOVERY_CLEANUP=PASS',flush=True)
    assert len(json.loads((dest/'validation_history.json').read_text()))==r['epochs_completed']
    print(f'I1_TFINANCE_STAGE4_{ratio}_SEED{seed}_VERIFIED_REUSE=PASS',flush=True)
    return r


def run_one(s3,base,ids,cfg,ratio,seed,locksha,labels):
    dest=OUT/ratio/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
    rp=dest/'frozen_training.json';recovery=dest/'epoch_recovery.pt'
    cp=dest/'best_validation.pt';sp=dest/'best_validation_scores.npy';hp=dest/'validation_history.json';ep=dest/'epoch_times.csv'
    if rp.exists():return clean_resume(s3,json.loads(rp.read_text()),cfg,ratio,seed,dest,locksha)
    assert not any(p.exists() for p in (cp,sp,hp,ep)),'STOP_PARTIAL_FINAL_FILES_REQUIRE_INSPECTION'
    index=np.asarray(ids[ratio],dtype=np.int64)
    train_label=labels[index].astype(np.int64)
    assert len(train_label)==len(index) and np.all((train_label==0)|(train_label==1))
    pos=int(train_label.sum());assert 0<pos<len(train_label)
    b=dict(base)
    b['train_ix']=torch.from_numpy(index.copy()).long().to('cuda:0')
    b['ytrain']=torch.from_numpy(train_label.copy()).long().to('cuda:0')
    b['weight']=torch.tensor([1.0,(len(index)-pos)/pos],device='cuda:0')
    s3.set_seeds(seed);model=s3.build_model(cfg)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=cfg['weight_decay'])
    start=1;stale=0;best=-1.;best_epoch=None;best_metrics=None;best_state=None;best_scores=None;history=[]
    if recovery.exists():
        r=torch.load(recovery,map_location='cpu')
        assert r['source_sha256']==s3.SOURCE_SHA and r['split_sha256']==s3.SPLIT_SHA
        assert r['frozen_winner_sha256']==locksha and r['config']==cfg
        assert r['ratio']==ratio and r['seed']==seed and r['test_scored'] is False
        s3.restore_state(model,r['model_state']);opt.load_state_dict(r['optimizer_state'])
        for state in opt.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.to('cuda:0')
        start=r['epoch']+1;stale=r['stale'];best=r['best'];best_epoch=r['best_epoch']
        best_metrics=r['best_metrics'];best_state=r['best_state'];best_scores=r['best_scores'];history=r['history']
        torch.set_rng_state(r['cpu_torch_rng']);torch.cuda.set_rng_state(r['cuda_torch_rng'],device='cuda:0')
        np.random.set_state(r['numpy_rng']);random.setstate(r['python_rng'])
        assert len(history)==start-1 and (best_state is not None)==(best_epoch is not None)
        print(f'I1_TFINANCE_STAGE4_{ratio}_SEED{seed}_RESUME_EPOCH={start}',flush=True)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start,101):
        if stale>=20:break
        model.train();opt.zero_grad(set_to_none=True);t0=s3.time.perf_counter()
        logits,_=model(b['graph'],b['features'])
        loss=F.cross_entropy(logits[b['train_ix']],b['ytrain'],weight=b['weight'])
        assert bool(torch.isfinite(loss).item()),'NONFINITE_TRAIN_LOSS'
        loss.backward();opt.step();torch.cuda.synchronize()
        train_s=s3.time.perf_counter()-t0;lossval=float(loss.detach().cpu().item());del logits,loss
        t0=s3.time.perf_counter();val,scores=s3.eval_val(model,b);torch.cuda.synchronize();val_s=s3.time.perf_counter()-t0
        if val['val_auprc']>best+1e-12:
            best=val['val_auprc'];best_epoch=epoch;best_metrics=val;best_state=s3.model_state(model);best_scores=scores.copy();stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,train_loss=lossval,training_seconds=train_s,validation_seconds=val_s,
                            improved=(best_epoch==epoch),best_so_far=best,stale=stale,**val))
        state=dict(epoch=epoch,ratio=ratio,seed=seed,config=cfg,
            source_sha256=s3.SOURCE_SHA,split_sha256=s3.SPLIT_SHA,frozen_winner_sha256=locksha,
            model_state=s3.model_state(model),optimizer_state=opt.state_dict(),history=history,
            best=best,best_epoch=best_epoch,best_metrics=best_metrics,best_state=best_state,
            best_scores=best_scores,stale=stale,cpu_torch_rng=torch.get_rng_state(),
            cuda_torch_rng=torch.cuda.get_rng_state(device='cuda:0'),numpy_rng=np.random.get_state(),
            python_rng=random.getstate(),test_scored=False,test_labels_indexed=False)
        s3.pt_atomic(recovery,state);del state,scores
        peak=torch.cuda.max_memory_allocated()/2**30
        if epoch in (1,2) or epoch%10==0 or stale>=20:
            print(f'I1_TFINANCE_STAGE4_{ratio}_SEED{seed} epoch={epoch} best_val_AP={best:.6f} train_s={train_s:.3f} val_s={val_s:.3f} peak_gib={peak:.3f} stale={stale}',flush=True)
        assert peak<43,'STOP_UNSAFE_GPU_MEMORY'
    assert best_epoch is not None and best_state is not None and best_scores is not None
    s3.pt_atomic(cp,dict(model_state=best_state,epoch=best_epoch,ratio=ratio,seed=seed,
        config=cfg,frozen_winner_sha256=locksha,dataset='T-Finance',
        source_sha256=s3.SOURCE_SHA,split_sha256=s3.SPLIT_SHA,upstream_commit=s3.PIN,
        validation=best_metrics,theta_last_buffer_excluded=True,test_scored=False))
    s3.npy_atomic(sp,best_scores)
    s3.write_json_atomic(hp,history);s3.epoch_csv(ep,history)
    rec=dict(dataset='T-Finance',ratio=ratio,seed=seed,config=cfg,
        model='EOGFDHeteroModel',mode='UNIFIED_INPUT_ADAPTER_NOT_AUTHOR_IDENTICAL',
        venus_methodological_approval_recorded=False,train_count=len(index),train_pos=pos,
        val_count=len(ids['val']),upstream_commit=s3.PIN,source_sha256=s3.SOURCE_SHA,
        split_sha256=s3.SPLIT_SHA,frozen_winner_sha256=locksha,
        best_val_auprc=best,best_epoch=best_epoch,epochs_completed=len(history),
        best_validation_metrics=best_metrics,frozen_validation_threshold=best_metrics['threshold'],
        checkpoint_sha256=sha(cp),validation_scores_sha256=sha(sp),history_sha256=sha(hp),
        epoch_times_sha256=sha(ep),max_epochs=100,patience=20,eval_seed=100921,
        peak_gpu_allocated_gib=float(torch.cuda.max_memory_allocated()/2**30),
        test_labels_indexed=False,test_scored=False,test_evaluations=0,
        original_model_code_unchanged=True)
    s3.write_json_atomic(rp,rec)
    recovery.unlink()
    print(f'I1_TFINANCE_STAGE4_{ratio}_SEED{seed}_FROZEN_FINAL=PASS val_AUPRC={best:.6f} threshold={best_metrics["threshold"]:.2f}',flush=True)
    del model,opt,b;gc.collect();torch.cuda.empty_cache()
    return rec


def main():
    arg=argparse.ArgumentParser()
    arg.add_argument('--audit-only',action='store_true')
    arg.add_argument('--all',action='store_true')
    args=arg.parse_args();assert args.audit_only != args.all,'CHOOSE --audit-only OR --all'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    torch.set_num_threads(min(8,os.cpu_count() or 2))
    s3,ids,win,locksha=checked_stage3()
    cfg=win['winner_config'];OUT.mkdir(parents=True,exist_ok=True)
    manifest=dict(dataset='T-Finance',mode='UNIFIED_INPUT_ADAPTER_NOT_AUTHOR_IDENTICAL',
        status='STAGE4_FROZEN_12_FINAL_TRAININGS',frozen_winner_sha256=locksha,
        frozen_winner_trial=8,frozen_config=cfg,source_sha256=s3.SOURCE_SHA,
        split_sha256=s3.SPLIT_SHA,upstream_commit=s3.PIN,
        ratios=list(RATIOS),seeds=list(SEEDS),max_epochs=100,patience=20,
        threshold_metric='validation_macroF1',selection_metric='validation_AUPRC',
        test_labels_indexed=False,test_scored=False,test_evaluations=0,
        original_model_code_unchanged=True,venus_methodological_approval_recorded=False)
    s3.create_or_identical(OUT/'stage4_plan.json',manifest)
    if args.audit_only:
        print('I1_TFINANCE_STAGE4_PRELAUNCH_AUDIT=PASS; no training and no test access',flush=True)
        return
    if (OUT/'EOGFD_TFinance_Stage4_12FrozenFinalsSummary.json').exists():
        raise RuntimeError('STOP_STAGE4_SUMMARY_ALREADY_FROZEN; use separate auditor, do not rerun')
    b=s3.dataset_gpu(ids)
    labels=np.full(39357,-1,dtype=np.int8)
    labels[np.asarray(ids['TR40'],dtype=np.int64)]=b['ytrain'].cpu().numpy().astype(np.int8)
    assert (labels[np.asarray(ids['TR40'],dtype=np.int64)]>=0).all()
    results=[]
    for ratio in RATIOS:
        for seed in SEEDS:
            results.append(run_one(s3,b,ids,cfg,ratio,seed,locksha,labels))
    assert len(results)==12 and all(not r['test_scored'] for r in results)
    summary=dict(status='EOGFD_TFINANCE_12_FROZEN_VALIDATION_ONLY_FINALS_COMPLETE',
        dataset='T-Finance',mode='UNIFIED_INPUT_ADAPTER_NOT_AUTHOR_IDENTICAL',
        frozen_winner_trial=8,frozen_winner_sha256=locksha,config=cfg,
        source_sha256=s3.SOURCE_SHA,split_sha256=s3.SPLIT_SHA,upstream_commit=s3.PIN,
        ratios=list(RATIOS),seeds=list(SEEDS),completed_finals=12,
        test_scored=False,test_evaluations=0,venus_methodological_approval_recorded=False,
        run_record_hashes={f'{r["ratio"]}_seed{r["seed"]}':sha(OUT/r['ratio']/f'seed_{r["seed"]}'/'frozen_training.json') for r in results})
    s3.create_or_identical(OUT/'EOGFD_TFinance_Stage4_12FrozenFinalsSummary.json',summary)
    print('I1_TFINANCE_STAGE4_12_FINALS_FROZEN=PASS TEST_SCORED=NO TEST_EVALUATIONS=0',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        print('I1_TFINANCE_STAGE4_STOP='+repr(e),flush=True);traceback.print_exc();sys.exit(2)
