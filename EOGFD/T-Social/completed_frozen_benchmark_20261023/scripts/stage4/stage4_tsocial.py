#!/usr/bin/env python3
"""EOGFDHeteroModel x T-Social, I2 Stage 4: frozen-trial9 validation-only finals.

Original EOGFD upstream source unmodified. Stage3 process-local optimizer selection
and streamed ES fitness used identically in train/validation. Frozen original graph,
seed-2 split, validation-only checkpoint/threshold selection. NEVER index test labels.
"""
from __future__ import annotations
import argparse, datetime, gc, hashlib, importlib.util, json, os, random, socket, statistics, sys, time, traceback, uuid
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path('/workspace/comp8851')
S3KIT=Path('/workspace/EOGFD_TSOCIAL_STAGE3_V3_I2_20261023')
S3PATH=S3KIT/'stage3_tsocial_v3.py'
S3RESULTS=ROOT/'results/eogfd/tsocial/stage3_scoped_elite_plus_streamed_train_eval_tr40_tuning'
OUT=ROOT/'results/eogfd/tsocial/stage4_frozen_finals_v3'
RATIOS=('TR40','TR30','TR20','TR10')
SEEDS=(2,42,72)
STAGE3_SOURCE_SHA='1f5feabdc3045d61fcc258568bb7648225bcc4d267b3dd616d698beb52256f94'
STAGE3_LAUNCHER_SHA='958eaa822d657ecfb89352801141c0d366d59d27ef8478e1c76f39cee6857022'
EXPECTED_CONFIG=dict(lr=.01,population_size=10,mutation_rate=.05,hidden_dim=10,
                     order=8,beta_p=1,weight_decay=0.,lamb=0.)
EXPECTED_STAGE3_BEST=0.859637
STAGE4_STATUS='EOGFD_TSOCIAL_STAGE4_12_FROZEN_VALIDATION_ONLY_FINALS_COMPLETE'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()

def note(*a):print(*a,flush=True)

def audit_stage3():
    assert socket.gethostname()=='a92ee6fa0813', 'STOP: This is not I2 57231045'
    assert S3PATH.is_file() and sha(S3PATH)==STAGE3_SOURCE_SHA, 'STOP: unexpected Stage3 V3 runner'
    assert sha(S3KIT/'vast_i2_stage3_v3.sh')==STAGE3_LAUNCHER_SHA, 'STOP: Stage3 launcher changed'
    assert torch.__version__=='1.12.1+cu113', ('TORCH_CHANGED',torch.__version__)
    spec=importlib.util.spec_from_file_location('eogfd_tsocial_verified_stage3_v3',str(S3PATH))
    assert spec and spec.loader
    s3=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s3;spec.loader.exec_module(s3)
    ids=s3.validate_assets()
    assert s3.RESULTS==S3RESULTS and tuple(s3.RATIOS)==RATIOS and tuple(s3.SEEDS)==SEEDS
    assert s3.MAX_EPOCHS==100 and s3.PATIENCE==20 and s3.EVAL_SEED==100921
    assert s3.MODE=='UNIFIED_FROZEN_SPLITS_SCOPED_ELITE_PLUS_STREAMED_TRAIN_EVAL_IMPL_OPT_NOT_AUTHOR_UNMODIFIED'
    assert s3.GRID[8]==EXPECTED_CONFIG and len(s3.GRID)==12
    stage3plan=json.loads((S3RESULTS/'stage3_run_manifest.json').read_text())
    assert stage3plan['grid']==s3.GRID
    assert stage3plan['source_sha256']==s3.SOURCE_SHA and stage3plan['split_sha256']==s3.SPLIT_SHA
    assert stage3plan['test_scored'] is False and stage3plan['test_labels_indexed'] is False
    rows=[]
    for i in range(1,13):
        dest=S3RESULTS/f'trial_{i:02d}'
        r=json.loads((dest/'trial_result.json').read_text())
        assert r['trial']==i and r['config']==s3.GRID[i-1],('STAGE3_GRID_MISMATCH',i)
        assert r['source_sha256']==s3.SOURCE_SHA and r['split_sha256']==s3.SPLIT_SHA
        assert r['upstream_commit']==s3.PIN
        assert r['test_scored'] is False and r['test_labels_indexed'] is False
        assert r['streamed_fitness_training_and_validation'] is True
        assert r['original_repository_unchanged'] is True
        for name,key in [('best_validation.pt','checkpoint_sha256'),('best_validation_scores.npy','validation_scores_sha256'),('validation_history.json','history_sha256'),('epoch_times.csv','epoch_times_sha256')]:
            path=dest/name
            assert path.is_file() and sha(path)==r[key],('STOP_STAGE3_ARTIFACT_HASH',i,name)
        assert not (dest/'epoch_recovery.pt').exists(),('STOP_UNFINISHED_STAGE3_TRIAL',i)
        history=json.loads((dest/'validation_history.json').read_text())
        assert len(history)==r['epochs_completed'] and r['epochs_completed']<=100
        assert r['best_epoch'] in [v['epoch'] for v in history]
        assert abs(max(v['val_auprc'] for v in history)-r['best_val_auprc'])<1e-11
        rows.append(r)
    winner=max(rows,key=lambda r:(r['best_val_auprc'],-r['trial']))
    winfile=S3RESULTS/'EOGFD_TSocial_TR40_FrozenWinner.json'
    frozen=json.loads(winfile.read_text())
    assert frozen['status']=='TR40_12_VALIDATION_ONLY_TRIALS_COMPLETE'
    assert winner['trial']==9 and frozen['winner_trial']==9
    assert abs(winner['best_val_auprc']-EXPECTED_STAGE3_BEST)<.000001
    assert frozen['winner_config']==EXPECTED_CONFIG and frozen['winner_config']==winner['config']
    assert frozen['winner_checkpoint_sha256']==winner['checkpoint_sha256']
    assert frozen['source_sha256']==s3.SOURCE_SHA and frozen['split_sha256']==s3.SPLIT_SHA
    assert frozen['test_scored'] is False and frozen['author_identical_reproduction'] is False
    assert frozen['venus_methodological_approval_recorded'] is False
    assert frozen['tuning_trials_completed']==12
    assert sha(S3RESULTS/'trial_09/best_validation.pt')==frozen['winner_checkpoint_sha256']
    best_checkpoint=torch.load(S3RESULTS/'trial_09/best_validation.pt',map_location='cpu')
    assert best_checkpoint['config']==EXPECTED_CONFIG and best_checkpoint['epoch']==winner['best_epoch']
    assert best_checkpoint['test_scored'] is False and best_checkpoint['test_labels_indexed'] is False
    assert best_checkpoint['source_sha256']==s3.SOURCE_SHA and best_checkpoint['split_sha256']==s3.SPLIT_SHA
    note(f'I2_TSOCIAL_STAGE4_12_TUNING_TRIAL_SHA_AUDIT=PASS WINNER=9 best_val_AUPRC={winner["best_val_auprc"]:.6f} test_scored=NO')
    note('I2_TSOCIAL_STAGE4_FROZEN_CONFIG=PASS '+json.dumps(EXPECTED_CONFIG,sort_keys=True))
    return s3,ids,frozen,sha(winfile)

def record_complete(s3,dest,ratio,seed,cfg,locksha):
    r=json.loads((dest/'final_training.json').read_text())
    assert r['ratio']==ratio and r['seed']==seed and r['config']==cfg and r['frozen_winner_sha256']==locksha
    assert r['source_sha256']==s3.SOURCE_SHA and r['split_sha256']==s3.SPLIT_SHA
    assert r['upstream_commit']==s3.PIN and r['test_scored'] is False and r['test_labels_indexed'] is False
    assert r['test_evaluations']==0 and r['model']=='EOGFDHeteroModel'
    for path,key in [('best_validation.pt','checkpoint_sha256'),('best_validation_scores.npy','validation_scores_sha256'),('validation_history.json','history_sha256'),('epoch_times.csv','epoch_times_sha256')]:
        assert sha(dest/path)==r[key],('STOP_STAGE4_COMPLETED_FILE_CHANGED',ratio,seed,path)
    hist=json.loads((dest/'validation_history.json').read_text())
    assert len(hist)==r['epochs_completed'] and r['epochs_completed']<=100
    assert abs(max(x['val_auprc'] for x in hist)-r['best_val_auprc'])<1e-11
    recovery=dest/'epoch_recovery.pt'
    if recovery.exists():
        state=torch.load(recovery,map_location='cpu')
        assert state['ratio']==ratio and state['seed']==seed and state['config']==cfg
        assert state['frozen_winner_sha256']==locksha
        assert state['source_sha256']==s3.SOURCE_SHA and state['split_sha256']==s3.SPLIT_SHA
        assert state['test_scored'] is False and state['test_labels_indexed'] is False
        assert len(state['history'])==r['epochs_completed']
        recovery.unlink()
        note(f'I2_TSOCIAL_STAGE4_{ratio}_SEED{seed}_POSTCOMMIT_RECOVERY_CLEANUP=PASS')
    note(f'I2_TSOCIAL_STAGE4_{ratio}_SEED{seed}_VERIFIED_REUSE=PASS val_AUPRC={r["best_val_auprc"]:.6f}')
    return r

def one(s3,base,ids,labels,ratio,seed,cfg,locksha):
    dest=OUT/ratio/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
    result=dest/'final_training.json';recovery=dest/'epoch_recovery.pt'
    checkpoint=dest/'best_validation.pt';score_path=dest/'best_validation_scores.npy'
    history_path=dest/'validation_history.json';times_path=dest/'epoch_times.csv'
    if result.exists():return record_complete(s3,dest,ratio,seed,cfg,locksha)
    assert not any(p.exists() for p in [checkpoint,score_path,history_path,times_path]), ('STOP_INCOMPLETE_FINAL_OUTPUTS_REQUIRE_INSPECTION',ratio,seed)
    train_ix=np.asarray(ids[ratio],dtype=np.int64)
    train_y=labels[train_ix].astype(np.int64)
    assert len(train_ix)==len(train_y) and (train_y>=0).all() and (train_y<=1).all()
    train_pos=int(train_y.sum())
    assert 0<train_pos<len(train_y)
    batch=dict(base)
    batch['train_ix']=torch.from_numpy(train_ix.copy()).long().to('cuda:0')
    batch['ytrain']=torch.from_numpy(train_y.copy()).long().to('cuda:0')
    batch['weight']=torch.tensor([1.0,(len(train_y)-train_pos)/train_pos],device='cuda:0')
    assert len(batch['yval'])==len(ids['val'])
    s3.set_seeds(seed)
    model=s3.build_model(cfg)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=cfg['weight_decay'])
    start=1;stale=0;best=-1.;best_epoch=None;best_metrics=None;best_state=None;best_scores=None;history=[]
    if recovery.exists():
        state=torch.load(recovery,map_location='cpu')
        assert state['ratio']==ratio and state['seed']==seed and state['config']==cfg
        assert state['frozen_winner_sha256']==locksha
        assert state['source_sha256']==s3.SOURCE_SHA and state['split_sha256']==s3.SPLIT_SHA
        assert state['test_scored'] is False and state['test_labels_indexed'] is False
        s3.restore_state(model,state['model_state']);opt.load_state_dict(state['optimizer_state'])
        for entry in opt.state.values():
            for k,v in entry.items():
                if isinstance(v,torch.Tensor):entry[k]=v.to('cuda:0')
        start=state['epoch']+1;stale=state['stale'];best=state['best'];best_epoch=state['best_epoch']
        best_metrics=state['best_metrics'];best_state=state['best_state'];best_scores=state['best_scores'];history=state['history']
        torch.set_rng_state(state['cpu_torch_rng']);torch.cuda.set_rng_state(state['cuda_torch_rng'],device='cuda:0')
        np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        assert len(history)==start-1 and (best_state is not None)==(best_epoch is not None)
        note(f'I2_TSOCIAL_STAGE4_{ratio}_SEED{seed}_RESUMING_EPOCH={start}')
    torch.cuda.reset_peak_memory_stats()
    run_start=time.perf_counter()
    for epoch in range(start,s3.MAX_EPOCHS+1):
        if stale>=s3.PATIENCE:break
        model.train();opt.zero_grad(set_to_none=True);t=time.perf_counter()
        logits,_=model(batch['graph'],batch['features'])
        loss=F.cross_entropy(logits[batch['train_ix']],batch['ytrain'],weight=batch['weight'])
        assert bool(torch.isfinite(loss).item()),('STOP_NONFINITE_TRAIN_LOSS',ratio,seed,epoch)
        loss.backward();opt.step();torch.cuda.synchronize()
        train_s=time.perf_counter()-t; lossval=float(loss.detach().item())
        del logits,loss
        torch.cuda.empty_cache()
        t=time.perf_counter();val,scores=s3.eval_val(model,batch);torch.cuda.synchronize()
        val_s=time.perf_counter()-t
        improvement=val['val_auprc']>best+1e-12
        if improvement:
            best=val['val_auprc'];best_epoch=epoch;best_metrics=val
            best_state=s3.model_state(model);best_scores=scores.copy();stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,train_loss=lossval,training_seconds=train_s,
             validation_seconds=val_s,improved=bool(improvement),best_so_far=best,stale=stale,**val))
        recovery_state=dict(epoch=epoch,ratio=ratio,seed=seed,config=cfg,
             source_sha256=s3.SOURCE_SHA,split_sha256=s3.SPLIT_SHA,frozen_winner_sha256=locksha,
             model_state=s3.model_state(model),optimizer_state=opt.state_dict(),history=history,
             best=best,best_epoch=best_epoch,best_metrics=best_metrics,best_state=best_state,
             best_scores=best_scores,stale=stale,cpu_torch_rng=torch.get_rng_state(),
             cuda_torch_rng=torch.cuda.get_rng_state(device='cuda:0'),
             numpy_rng=np.random.get_state(),python_rng=random.getstate(),test_scored=False,test_labels_indexed=False)
        s3.pt_atomic(recovery,recovery_state)
        del recovery_state,scores
        torch.cuda.empty_cache()
        peak=torch.cuda.max_memory_allocated()/2**30
        if epoch in (1,2) or epoch%10==0 or stale>=s3.PATIENCE:
            note(f'I2_TSOCIAL_STAGE4_{ratio}_SEED{seed} epoch={epoch} loss={lossval:.5f} val_AUPRC={val["val_auprc"]:.6f} best={best:.6f} train_s={train_s:.3f} val_s={val_s:.3f} peak_gib={peak:.3f} stale={stale}')
        assert peak<43,('STOP_UNSAFE_GPU_MEMORY',ratio,seed,peak)
    assert best_state is not None and best_scores is not None and best_epoch is not None and best_metrics is not None
    s3.pt_atomic(checkpoint,dict(model_state=best_state,epoch=best_epoch,config=cfg,
         dataset='T-Social',ratio=ratio,seed=seed,frozen_winner_sha256=locksha,
         source_sha256=s3.SOURCE_SHA,split_sha256=s3.SPLIT_SHA,upstream_commit=s3.PIN,
         validation=best_metrics,theta_last_buffer_excluded=True,test_scored=False,test_labels_indexed=False))
    s3.npy_atomic(score_path,best_scores)
    s3.write_json_atomic(history_path,history);s3.epoch_csv(times_path,history)
    record=dict(dataset='T-Social',ratio=ratio,seed=seed,model='EOGFDHeteroModel',config=cfg,
        mode=s3.MODE,author_identical_reproduction=False,original_repository_unchanged=True,
        venus_methodological_approval_recorded=False,frozen_winner_trial=9,
        train_count=len(train_ix),train_pos=train_pos,val_count=len(ids['val']),
        upstream_commit=s3.PIN,source_sha256=s3.SOURCE_SHA,split_sha256=s3.SPLIT_SHA,
        frozen_winner_sha256=locksha,best_val_auprc=best,best_epoch=best_epoch,
        epochs_completed=len(history),best_validation_metrics=best_metrics,
        frozen_validation_threshold=best_metrics['threshold'],
        checkpoint_sha256=sha(checkpoint),validation_scores_sha256=sha(score_path),
        history_sha256=sha(history_path),epoch_times_sha256=sha(times_path),
        max_epochs=s3.MAX_EPOCHS,patience=s3.PATIENCE,eval_seed=s3.EVAL_SEED,
        elapsed_seconds=sum(h['training_seconds']+h['validation_seconds'] for h in history),
        peak_gpu_allocated_gib=float(torch.cuda.max_memory_allocated()/2**30),
        test_labels_indexed=False,test_scored=False,test_evaluations=0,
        validation_only_selection=True,runtime_select_override=True,
        streamed_fitness_training_and_validation=True)
    s3.write_json_atomic(result,record)
    recovery.unlink(missing_ok=True)
    note(f'I2_TSOCIAL_STAGE4_{ratio}_SEED{seed}_FROZEN_FINAL=PASS val_AUPRC={best:.6f} threshold={best_metrics["threshold"]:.2f} epochs={len(history)}')
    del model,opt,batch;gc.collect();torch.cuda.empty_cache()
    return record

def main():
    p=argparse.ArgumentParser(description='I2 TSocial Stage4: frozen validation-only EOGFD finals, no test')
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--audit-only',action='store_true')
    group.add_argument('--all',action='store_true')
    group.add_argument('--one',nargs=2,metavar=('RATIO','SEED'))
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0','STOP: expected explicit CUDA_VISIBLE_DEVICES=0'
    torch.set_num_threads(min(8,os.cpu_count() or 2))
    s3,ids,frozen,locksha=audit_stage3()
    cfg=frozen['winner_config']
    OUT.mkdir(parents=True,exist_ok=True)
    plan=dict(dataset='T-Social',model='EOGFDHeteroModel',stage='4',
        status='FROZEN_12_VALIDATION_ONLY_FINAL_TRAINING_PLAN',mode=s3.MODE,
        frozen_winner_trial=9,frozen_winner_sha256=locksha,
        source_sha256=s3.SOURCE_SHA,split_sha256=s3.SPLIT_SHA,
        upstream_commit=s3.PIN,config=cfg,ratios=list(RATIOS),seeds=list(SEEDS),
        max_epochs=s3.MAX_EPOCHS,patience=s3.PATIENCE,eval_seed=s3.EVAL_SEED,
        selected_metric='validation_AUPRC',threshold_metric='validation_macroF1',
        independent_test_evaluations=0,test_labels_indexed=False,test_scored=False,
        author_identical_reproduction=False,venus_methodological_approval_recorded=False,
        process_local_optimizations=['elite_only_selection','streamed_ES_fitness_train_and_val'])
    s3.create_or_identical(OUT/'stage4_plan.json',plan)
    if args.audit_only:
        note('I2_TSOCIAL_STAGE4_PRELAUNCH_AUDIT=PASS stage3=12/12 winner=9 TEST_LABELS_INDEXED=NO TRAINING_STARTED=NO')
        return
    summary_file=OUT/'EOGFD_TSocial_Stage4_12FrozenFinalsSummary.json'
    if summary_file.exists():
        raise RuntimeError('STOP_STAGE4_SUMMARY_ALREADY_FROZEN; use future independent audit; do not retrain')
    if args.one:
        ratio,seed_text=args.one
        assert ratio in RATIOS and seed_text in ('2','42','72'),'STOP: invalid --one ratio/seed'
        jobs=[(ratio,int(seed_text))]
    else:jobs=[(ratio,seed) for ratio in RATIOS for seed in SEEDS]
    s3.install_scoped_optimizer()
    s3.assert_streamed_equivalence_on_small_graphs()
    s3.assert_streamed_forward_backward_equivalence()
    gc.collect();torch.cuda.empty_cache()
    base=s3.dataset_gpu(ids)
    # Extract ONLY labels for TR40 train and frozen validation using the
    # independently audited Stage 3 loader. Every lower ratio is a TR40 subset.
    labels=np.full(5781065,-1,dtype=np.int8)
    tr40_ix=np.asarray(ids['TR40'],dtype=np.int64)
    ytr40=base['ytrain'].detach().cpu().numpy().astype(np.int8)
    assert ytr40.shape==(2312426,) and int(ytr40.sum())==69712
    labels[tr40_ix]=ytr40
    assert np.all(labels[np.asarray(ids['val'],dtype=np.int64)]==-1)
    assert np.all(labels[np.asarray(ids['test'],dtype=np.int64)]==-1)
    del ytr40
    note('I2_TSOCIAL_STAGE4_ONLY_TR40_TRAIN_LABELS_AND_VAL_LABELS_AVAILABLE=PASS TEST_LABELS_INDEXED=NO')
    results=[]
    for ratio,seed in jobs:
        results.append(one(s3,base,ids,labels,ratio,seed,cfg,locksha))
    if args.one:
        note(f'I2_TSOCIAL_STAGE4_SINGLE_FINAL=PASS {jobs[0][0]} seed={jobs[0][1]} TEST_SCORED=NO')
        return
    assert len(results)==12 and all(r['test_scored'] is False and r['test_evaluations']==0 for r in results)
    summary=dict(status=STAGE4_STATUS,dataset='T-Social',model='EOGFDHeteroModel',
        mode=s3.MODE,upstream_commit=s3.PIN,source_sha256=s3.SOURCE_SHA,
        split_sha256=s3.SPLIT_SHA,frozen_winner_trial=9,
        frozen_winner_sha256=locksha,frozen_config=cfg,
        ratios=list(RATIOS),seeds=list(SEEDS),completed_finals=12,
        test_scored=False,test_labels_indexed=False,test_evaluations=0,
        author_identical_reproduction=False,venus_methodological_approval_recorded=False,
        run_record_hashes={f'{r["ratio"]}_seed{r["seed"]}':sha(OUT/r['ratio']/f'seed_{r["seed"]}'/'final_training.json') for r in results})
    s3.create_or_identical(summary_file,summary)
    note('I2_TSOCIAL_STAGE4_12_FINALS_FROZEN=PASS TEST_SCORED=NO TEST_EVALUATIONS=0')

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        note('I2_TSOCIAL_STAGE4_STOP='+repr(exc))
        traceback.print_exc();sys.exit(2)
