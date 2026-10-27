#!/usr/bin/env python3
"""Validated EOGFD (original EOGFDHeteroModel) x canonical T-Finance.
Stage 3 ONLY: TR40 seed2 validation-only tuning, 12 configs, no test scoring.
Single writer + durable epoch recovery; tuning winner frozen only after 12.
This is a unified frozen-split adapter, not author-identical reproduction.
"""
from __future__ import annotations
import argparse, csv, datetime, gc, hashlib, json, os, random, socket, statistics, subprocess, sys, time, traceback, uuid
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.data.utils import load_graphs
from sklearn.metrics import average_precision_score, roc_auc_score

BASE=Path('/workspace/comp8851')
KIT=Path('/workspace/EOGFD_DUAL_I1_I2_20261023')
SRC=BASE/'assets/tfinance/source/tfinance'
SPLIT=BASE/'assets/tfinance/tfinance_seed2_nested_splits.npz'
PILOT=BASE/'diagnostics/EOGFD_57212396_tfinance/train_only_one_update.json'
AUTHOR=BASE/'repos/EOGFD'
RESULTS=BASE/'results/eogfd/tfinance/stage3_tr40_tuning'
SOURCE_SHA='b7d853ec4079e9f7137c03f78a044b1c33d0ff5ed6caa297b895542a7c8e3700'
SPLIT_SHA='de18651a281c3f7d098a89987b05313bbbd2386556a97c61344a6280d9426ff1'
PIN='a4726f173bc8c923109ddcb2786f4d9c5206ae7f'
EVAL_SEED=100921
MAX_EPOCHS=100
PATIENCE=20
RATIOS=('TR40','TR30','TR20','TR10')
SEEDS=(2,42,72)
GRID=[dict(lr=lr,population_size=pop,mutation_rate=mutation,
           hidden_dim=64,order=7,beta_p=1,weight_decay=0.,lamb=0.)
      for lr in (.001,.003,.01)
      for pop in (10,20)
      for mutation in (.05,.10)]
assert len(GRID)==12

def log(*args):print(*args,flush=True)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''): h.update(block)
    return h.hexdigest()

def write_json_atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing.'+uuid.uuid4().hex)
    with tmp.open('x') as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def create_or_identical(path,obj):
    if path.exists():
        assert json.loads(path.read_text())==obj,('CONFLICT_EXISTING_RECORD',str(path))
    else:write_json_atomic(path,obj)

def pt_atomic(path,obj):
    tmp=path.with_name(path.name+'.writing.'+uuid.uuid4().hex)
    with tmp.open('xb') as f:
        torch.save(obj,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def npy_atomic(path,arr):
    tmp=path.with_name(path.name+'.writing.'+uuid.uuid4().hex)
    with tmp.open('xb') as f:
        np.save(f,arr,allow_pickle=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def set_seeds(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def model_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k!='theta_last'}

def restore_state(model,state):
    r=model.load_state_dict(state,strict=False)
    assert r.missing_keys==['theta_last'] and not r.unexpected_keys,r

def decode_only(raw,ids):
    idx=torch.from_numpy(np.asarray(ids,dtype=np.int64).copy())
    if raw.shape==(39357,2):y=raw[idx].argmax(dim=1).long()
    elif raw.shape in ((39357,),(39357,1)):y=raw[idx].long().view(-1)
    else:raise RuntimeError(('STOP_UNEXPECTED_LABEL_ENCODING',tuple(raw.shape)))
    assert set(torch.unique(y).tolist())=={0,1},'SPLIT_LABELS_NOT_BOTH_CLASSES'
    return y

def validate_assets():
    assert socket.gethostname()=='60d39211b1fd',('STOP_WRONG_I1_HOST',socket.gethostname())
    assert KIT.is_dir() and (KIT/'preflight_pilot.py').is_file()
    assert torch.__version__=='1.12.1+cu113' and dgl.__version__=='0.8.1'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert subprocess.check_output(['git','-C',str(AUTHOR),'rev-parse','HEAD'],text=True).strip()==PIN
    assert not subprocess.check_output(['git','-C',str(AUTHOR),'status','--porcelain'],text=True).strip()
    assert digest(SRC)==SOURCE_SHA, 'STOP SOURCE HASH CHANGED'
    assert digest(SPLIT)==SPLIT_SHA, 'STOP FROZEN SPLIT HASH CHANGED'
    pilot=json.loads(PILOT.read_text())
    assert pilot['status']=='TRAIN_ONLY_REAL_FULL_GRAPH_ONE_UPDATE_PASS'
    assert pilot['source_sha256']==SOURCE_SHA and pilot['split_sha256']==SPLIT_SHA
    assert pilot['upstream_commit']==PIN and pilot['test_scored'] is False
    with np.load(SPLIT,allow_pickle=False) as z:
        ids={k:np.asarray(z[k],dtype=np.int64).copy() for k in ('TR10','TR20','TR30','TR40','val','test')}
        for key in ('seed','split_seed'):
            if key in z.files:assert int(z[key].item())==2
    expected={'TR10':3935,'TR20':7870,'TR30':11806,'TR40':15742,'val':7872,'test':15743}
    for key,num in expected.items():
        a=ids[key];assert len(a)==num and (a>=0).all() and (a<39357).all()
        assert len(np.unique(a))==num
    for small,large in [('TR10','TR20'),('TR20','TR30'),('TR30','TR40')]:assert np.isin(ids[small],ids[large]).all()
    for a,b in [('TR40','val'),('TR40','test'),('val','test')]:assert not np.intersect1d(ids[a],ids[b],assume_unique=True).size
    log('I1_TFINANCE_TUNING_CANONICAL_INPUTS=PASS TEST_SCORED=NO')
    return ids

def dataset_gpu(ids):
    grs,_=load_graphs(str(SRC));assert len(grs)==1
    graph=grs[0]
    assert graph.num_nodes()==39357 and graph.num_edges()==42445086
    assert len(graph.ntypes)==1 and len(graph.canonical_etypes)==1
    assert 'feature' in graph.ndata and 'label' in graph.ndata
    features=graph.ndata.pop('feature').float()
    raw=graph.ndata.pop('label')
    ytrain=decode_only(raw,ids['TR40']);yval=decode_only(raw,ids['val'])
    assert int(ytrain.sum())==721 and int(yval.sum())==361
    del raw
    for key in list(graph.ndata.keys()):graph.ndata.pop(key)
    for key in list(graph.edata.keys()):graph.edata.pop(key)
    assert not list(graph.ndata) and not list(graph.edata)
    assert features.shape==(39357,10) and bool(torch.isfinite(features).all())
    graph=graph.to('cuda:0');features=features.to('cuda:0')
    train_ix=torch.from_numpy(ids['TR40']).long().to('cuda:0')
    val_ix=torch.from_numpy(ids['val']).long().to('cuda:0')
    ytrain=ytrain.to('cuda:0');yval=yval.cpu().numpy().astype(np.int64)
    weight=torch.tensor([1.0,(15742-721)/721],device='cuda:0')
    log('I1_TFINANCE_TUNING_GRAPH_READY=PASS nodes=39357 original_edges=42445086 relation_count=1 train=15742 val=7872 test_labels_indexed=NO')
    return dict(graph=graph,features=features,train_ix=train_ix,val_ix=val_ix,ytrain=ytrain,yval=yval,weight=weight)

def metrics(y,scores):
    y=np.asarray(y,dtype=np.int64).reshape(-1)
    scores=np.asarray(scores,dtype=np.float64).reshape(-1)
    assert len(y)==len(scores) and np.isfinite(scores).all()
    assert np.all((scores>=0)&(scores<=1)) and set(np.unique(y).tolist())=={0,1}
    ts=np.arange(1,100,dtype=np.float64)/100
    positive=y.astype(bool)[:,None];pred=scores[:,None]>ts[None,:]
    tp=np.sum(pred&positive,axis=0);tn=np.sum((~pred)&(~positive),axis=0)
    fp=np.sum(pred&(~positive),axis=0);fn=np.sum((~pred)&positive,axis=0)
    fpos=2*tp/np.maximum(2*tp+fp+fn,1)
    fneg=2*tn/np.maximum(2*tn+fp+fn,1)
    macro=(fpos+fneg)/2;recall=tp/np.maximum(tp+fn,1)
    ix=max(range(99),key=lambda i:(round(float(macro[i]),12),round(float(recall[i]),12),-abs(float(ts[i])-.5),-float(ts[i])))
    return dict(val_auprc=float(average_precision_score(y,scores)),val_auroc=float(roc_auc_score(y,scores)),
                val_macro_f1=float(macro[ix]),val_fraud_recall=float(recall[ix]),threshold=float(ts[ix]))

def eval_val(model,b):
    model.eval()
    with torch.random.fork_rng(devices=[0],enabled=True):
        torch.manual_seed(EVAL_SEED);torch.cuda.manual_seed_all(EVAL_SEED)
        with torch.no_grad():
            logits,_=model(b['graph'],b['features'])
            scores=torch.softmax(logits[b['val_ix']],dim=1)[:,1].detach().cpu().numpy().astype(np.float64)
    assert len(scores)==7872
    return metrics(b['yval'],scores),scores

def build_model(cfg):
    from models.eogfd import EOGFDHeteroModel
    return EOGFDHeteroModel(in_feats=10,hid_feats=cfg['hidden_dim'],num_classes=2,
        d=cfg['order'],beta_p=cfg['beta_p'],population_size=cfg['population_size'],
        mutation_rate=cfg['mutation_rate']).to('cuda:0')

def epoch_csv(path,history):
    tmp=path.with_name(path.name+'.writing.'+uuid.uuid4().hex)
    with tmp.open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['epoch','training_seconds','validation_seconds','cumulative_training_seconds'])
        cumulative=0.
        for h in history:
            cumulative+=h['training_seconds'];w.writerow([h['epoch'],f'{h["training_seconds"]:.8f}',f'{h["validation_seconds"]:.8f}',f'{cumulative:.8f}'])
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def trial_run(i,b):
    cfg=GRID[i-1];name=f'trial_{i:02d}'
    dest=RESULTS/name;dest.mkdir(parents=True,exist_ok=True)
    final=dest/'trial_result.json';recfile=dest/'epoch_recovery.pt'
    ckpt=dest/'best_validation.pt';scofile=dest/'best_validation_scores.npy'
    historyfile=dest/'validation_history.json';csvfile=dest/'epoch_times.csv'
    if final.exists():
        obj=json.loads(final.read_text())
        assert obj['trial']==i and obj['config']==cfg and obj['source_sha256']==SOURCE_SHA and obj['split_sha256']==SPLIT_SHA
        assert obj['test_scored'] is False and obj['test_labels_indexed'] is False
        for p,k in ((ckpt,'checkpoint_sha256'),(scofile,'validation_scores_sha256'),(historyfile,'history_sha256'),(csvfile,'epoch_times_sha256')):
            assert p.is_file() and digest(p)==obj[k],('STOP_COMPLETE_TRIAL_BAD_ARTIFACT',str(p))
        if recfile.exists():
            old=torch.load(recfile,map_location='cpu')
            assert old['config']==cfg and old['source_sha256']==SOURCE_SHA and old['split_sha256']==SPLIT_SHA
            assert old['test_scored'] is False and len(old['history'])==obj['epochs_completed']
            recfile.unlink()
            log(f'I1_TFINANCE_{name}_RECOVERY_AFTER_COMPLETE_CLEANUP=PASS')
        log(f'I1_TFINANCE_STAGE3_{name}_VERIFIED_REUSE=PASS AUPRC={obj["best_val_auprc"]:.6f}')
        return obj
    assert not any(p.exists() for p in (ckpt,scofile,historyfile,csvfile)), ('STOP_PARTIAL_FINAL_WITHOUT_TRIAL_JSON',name)
    set_seeds(2)
    model=build_model(cfg)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=cfg['weight_decay'])
    start=1;stale=0;best=-1.;best_epoch=None;best_metrics=None;best_state=None;best_scores=None;history=[]
    if recfile.exists():
        r=torch.load(recfile,map_location='cpu')
        assert r['config']==cfg and r['source_sha256']==SOURCE_SHA and r['split_sha256']==SPLIT_SHA
        assert r['test_scored'] is False and r['test_labels_indexed'] is False
        restore_state(model,r['model_state']);opt.load_state_dict(r['optimizer_state'])
        for state in opt.state.values():
            for k,v in state.items():
                if isinstance(v,torch.Tensor):state[k]=v.to('cuda:0')
        start=r['epoch']+1;stale=r['stale'];best=r['best'];best_epoch=r['best_epoch']
        best_metrics=r['best_metrics'];best_state=r['best_state'];best_scores=r['best_scores'];history=r['history']
        torch.set_rng_state(r['cpu_torch_rng']);torch.cuda.set_rng_state(r['cuda_torch_rng'],device='cuda:0')
        np.random.set_state(r['numpy_rng']);random.setstate(r['python_rng'])
        assert len(history)==start-1 and (best_state is not None)==(best_epoch is not None)
        log(f'I1_TFINANCE_{name}_RECOVERY_EPOCH_START={start}')
    torch.cuda.reset_peak_memory_stats()
    tic=time.perf_counter()
    for epoch in range(start,MAX_EPOCHS+1):
        if stale>=PATIENCE:break
        model.train();opt.zero_grad(set_to_none=True);ts=time.perf_counter()
        logits,_=model(b['graph'],b['features'])
        loss=F.cross_entropy(logits[b['train_ix']],b['ytrain'],weight=b['weight'])
        assert bool(torch.isfinite(loss).item()),('STOP_NONFINITE_TRAIN_LOSS',i,epoch)
        loss.backward();opt.step();torch.cuda.synchronize()
        train_s=time.perf_counter()-ts;vloss=float(loss.detach().cpu().item())
        del logits,loss
        tv=time.perf_counter();val,scores=eval_val(model,b);torch.cuda.synchronize();val_s=time.perf_counter()-tv
        improved=val['val_auprc']>best+1e-12
        if improved:
            best=val['val_auprc'];best_epoch=epoch;best_metrics=val;best_state=model_state(model);best_scores=scores.copy();stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,train_loss=vloss,training_seconds=train_s,validation_seconds=val_s,
                            improved=bool(improved),best_so_far=float(best),stale=stale,**val))
        prog=dict(epoch=epoch,config=cfg,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
          model_state=model_state(model),optimizer_state=opt.state_dict(),history=history,
          best=best,best_epoch=best_epoch,best_metrics=best_metrics,best_state=best_state,best_scores=best_scores,
          stale=stale,cpu_torch_rng=torch.get_rng_state(),cuda_torch_rng=torch.cuda.get_rng_state(device='cuda:0'),
          numpy_rng=np.random.get_state(),python_rng=random.getstate(),test_scored=False,test_labels_indexed=False)
        pt_atomic(recfile,prog)
        del prog,scores
        peak=torch.cuda.max_memory_allocated()/2**30
        if epoch in (1,2) or epoch%10==0 or stale>=PATIENCE:
            log(f'I1_TFINANCE_TRIAL_{i:02d} epoch={epoch} loss={vloss:.5f} val_AUPRC={val["val_auprc"]:.6f} best={best:.6f} train_s={train_s:.3f} val_s={val_s:.3f} peak_gib={peak:.3f} stale={stale}')
        if peak>43:raise RuntimeError(('STOP_UNSAFE_GPU_MEMORY_ALLOCATED_GIB',peak))
    assert best_state is not None and best_scores is not None and best_epoch is not None
    # The snapshot used for the winner is the exact validation-selected best-epoch model.
    pt_atomic(ckpt,dict(model_state=best_state,epoch=best_epoch,config=cfg,
             dataset='T-Finance',ratio='TR40',seed=2,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
             upstream_commit=PIN,validation=best_metrics,theta_last_buffer_excluded=True,
             test_scored=False,test_labels_indexed=False))
    npy_atomic(scofile,best_scores)
    write_json_atomic(historyfile,history)
    epoch_csv(csvfile,history)
    obj=dict(dataset='T-Finance',model='EOGFDHeteroModel',trial=i,ratio='TR40',seed=2,
       config=cfg,upstream_commit=PIN,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
       mode='UNIFIED_INPUT_ADAPTER_NOT_AUTHOR_IDENTICAL',
       venus_methodological_approval_recorded=False,
       best_val_auprc=best,best_epoch=best_epoch,epochs_completed=len(history),
       best_validation_metrics=best_metrics,
       checkpoint_sha256=digest(ckpt),validation_scores_sha256=digest(scofile),
       history_sha256=digest(historyfile),epoch_times_sha256=digest(csvfile),
       per_epoch_timings_preserved=True,max_epochs=MAX_EPOCHS,patience=PATIENCE,
       elapsed_seconds=sum(h['training_seconds']+h['validation_seconds'] for h in history),
       peak_gpu_allocated_gib=float(torch.cuda.max_memory_allocated()/2**30),
       fixed_eval_rng_seed=EVAL_SEED,threshold_selection='validation_macroF1_then_fraud_recall_then_nearest_0.5',
       selection_metric='validation_AUPRC',test_labels_indexed=False,test_scored=False)
    write_json_atomic(final,obj)
    if recfile.exists():recfile.unlink()
    log(f'I1_TFINANCE_STAGE3_TRIAL_{i:02d}_VALIDATION_ONLY_COMPLETE=PASS best_val_AUPRC={best:.6f} best_epoch={best_epoch} epochs={len(history)}')
    log(f'I1_TFINANCE_STAGE3_TRIAL_{i:02d}_MEASURED_EPOCH_MEDIAN_TRAIN_SECONDS={statistics.median(h["training_seconds"] for h in history):.3f}')
    log(f'I1_TFINANCE_STAGE3_TRIAL_{i:02d}_MEASURED_EPOCH_MEDIAN_VALIDATION_SECONDS={statistics.median(h["validation_seconds"] for h in history):.3f}')
    del model,opt;gc.collect();torch.cuda.empty_cache()
    return obj

def main():
    p=argparse.ArgumentParser(description='I1 EOGFD T-Finance Stage3 validation-only: no test access')
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--trial',type=int,choices=range(1,13),metavar='1..12')
    g.add_argument('--all',action='store_true')
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0',('STOP_ASSIGN_ONE_GPU_EXPLICITLY',os.environ.get('CUDA_VISIBLE_DEVICES'))
    torch.set_num_threads(min(8,os.cpu_count() or 2))
    ids=validate_assets()
    RESULTS.mkdir(parents=True,exist_ok=True)
    m=dict(dataset='T-Finance',mode='UNIFIED_INPUT_ADAPTER_NOT_AUTHOR_IDENTICAL',
        original_eogfd_author_commit=PIN,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
        split_seed=2,tuning_seed=2,ratios=list(RATIOS),final_seeds=list(SEEDS),grid=GRID,
        max_epochs=MAX_EPOCHS,patience=PATIENCE,eval_seed=EVAL_SEED,
        selection='TR40_seed2_validation_AUPRC',test_labels_indexed=False,test_scored=False,
        original_native_relation_count=1,original_model_code_unchanged=True,
        note='Inputs are the original canonical graph with exact frozen node splits, no synthetic relations; not an author-identical reproduction.')
    create_or_identical(RESULTS/'stage3_run_manifest.json',m)
    if (RESULTS/'EOGFD_TFinance_TR40_FrozenWinner.json').exists() and args.trial:
        raise RuntimeError('STOP WINNER ALREADY FROZEN; DO NOT CREATE NEW TRIAL')
    attempt_name=(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex)
    attempt_path=RESULTS/'attempts'/(attempt_name+'.json')
    attempt_record={'attempt_id':attempt_name,'scope':'all' if args.all else f'trial_{args.trial:02d}',
      'dataset':'T-Finance','status':'STARTED','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'source_sha256':SOURCE_SHA,'split_sha256':SPLIT_SHA,'test_scored':False}
    write_json_atomic(attempt_path,attempt_record)
    try:
        b=dataset_gpu(ids)
        which=range(1,13) if args.all else [args.trial]
        for i in which:trial_run(i,b)
        attempt_record['status']='COMPLETED'
    except BaseException as err:
        attempt_record['status']='FAILED_OR_INTERRUPTED'
        attempt_record['error']=repr(err)
        write_json_atomic(attempt_path,attempt_record)
        raise
    write_json_atomic(attempt_path,attempt_record)
    if args.all:
        results=[json.loads((RESULTS/f'trial_{i:02d}'/'trial_result.json').read_text()) for i in range(1,13)]
        assert all(r['test_scored'] is False for r in results)
        winner=max(results,key=lambda r:(r['best_val_auprc'],-r['trial']))
        frozen=dict(status='TR40_12_VALIDATION_ONLY_TRIALS_COMPLETE',dataset='T-Finance',
           upstream_commit=PIN,source_sha256=SOURCE_SHA,split_sha256=SPLIT_SHA,
           winner_trial=winner['trial'],winner_config=winner['config'],
           winner_val_auprc=winner['best_val_auprc'],winner_checkpoint_sha256=winner['checkpoint_sha256'],
           selection='validation_AUPRC',ratio='TR40',training_seed=2,
           final_ratios=list(RATIOS),final_seeds=list(SEEDS),
           tuning_trials_completed=12,test_scored=False,
           author_identical_reproduction=False,venus_methodological_approval_recorded=False)
        create_or_identical(RESULTS/'EOGFD_TFinance_TR40_FrozenWinner.json',frozen)
        log(f'I1_TFINANCE_STAGE3_12_TRIALS_FROZEN=PASS WINNER_TRIAL={winner["trial"]} VAL_AUPRC={winner["best_val_auprc"]:.6f} TEST_SCORED=NO')
    else:
        log('I1_TFINANCE_STAGE3_ONE_COMPLETE_TRIAL_GATE=PASS NEXT_REVIEW_REAL_EPOCH_TIMINGS_BEFORE_ALL_12')
    log('I1_TFINANCE_STAGE3_HELDOUT_TEST_EVALUATIONS=0')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        log('I1_TFINANCE_STAGE3_STOP='+repr(e));traceback.print_exc();sys.exit(2)
