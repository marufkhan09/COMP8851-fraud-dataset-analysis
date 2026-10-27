#!/usr/bin/env python3
"""EOGFD Stage4: freeze independently verified Stage3 winner and fit 12 final models.
NO held-out test labels/metrics, NO retuning. Tested frozen Ass2 split and original full heterograph.
This uses original pinned EOGFDHeteroModel (no author tracked-file edits).
"""
import sys,os,json,socket,hashlib,subprocess,time,random,gc,traceback,itertools
from pathlib import Path
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import dgl
from sklearn.metrics import average_precision_score,roc_auc_score

CFG=Path(__file__).with_name('EOGFD_YelpChi_I1_Stage4_Config.json')
C=json.loads(CFG.read_text())
PREFIX=f"EOGFD_{C['dataset']}_{C['id']}"
BASE=Path('/workspace/comp8851')
REPO=BASE/'repos/EOGFD'
TUNE=BASE/'results/eogfd'/C['dataset'].lower()/'stage3_tr40_tuning'
FINAL=BASE/'results/eogfd'/C['dataset'].lower()/'stage4_frozen_finals'
SRC=BASE/'assets/eogfd'/C['dataset']/C['source_name']
SPLIT=Path('/workspace/uploads/ass2_split_candidates')/C['dataset']/C['split_name']
RATIOS=('TR40','TR30','TR20','TR10')
SEEDS=(2,42,72)
MAX_EPOCHS=100
PATIENCE=20
EVAL_SEED=100921
GRID=[{'lr':lr,'population_size':pop,'mutation_rate':mut,'hidden_dim':64,'order':C['order'],'beta_p':1,'lamb':0.0,'beta':C['beta']}
      for lr,pop,mut in itertools.product([.001,.003,.01],[10,20],[.05,.10])]

def hash_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for blk in iter(lambda:f.read(4*1024*1024),b''):h.update(blk)
    return h.hexdigest()

def atomic_json(path,thing):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(thing,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def atomic_pt(path,thing):
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(thing,tmp);os.replace(tmp,path)

def atomic_np(path,arr):
    tmp=path.with_suffix('.tmp.npy')
    np.save(tmp,arr,allow_pickle=False);os.replace(tmp,path)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def frozen_state(model):
    return {k:v for k,v in model.state_dict().items() if k!='theta_last'}

def load_state(model,weights):
    result=model.load_state_dict(weights,strict=False)
    assert result.missing_keys==['theta_last'] and not result.unexpected_keys,result

def validate():
    assert socket.gethostname()==C['hostname'],('WRONG HOST',socket.gethostname())
    assert torch.cuda.is_available(),'CUDA UNAVAILABLE'
    assert SRC.is_file() and SPLIT.is_file(),'MISSING CANONICAL INPUTS'
    assert hash_file(SRC)==C['source_sha256'],'SOURCE HASH MISMATCH'
    assert hash_file(SPLIT)==C['split_sha256'],'SPLIT HASH MISMATCH'
    print(PREFIX+'_STAGE4_CANONICAL_SOURCE_SPLIT=PASS',flush=True)
    assert REPO.is_dir(), 'SOURCE CHECKOUT MISSING'
    head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    assert head==C['upstream_commit'],('UPSTREAM COMMIT MISMATCH',head)
    tracked=subprocess.run(['git','-C',str(REPO),'diff','--exit-code','HEAD','--'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    assert tracked.returncode==0,'UPSTREAM TRACKED FILE EDITS DETECTED'
    dirty=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    assert not dirty,('UPSTREAM DIRTY',dirty)
    sys.path.insert(0,str(REPO))
    from models.eogfd import EOGFDHeteroModel
    winner_path=TUNE/(PREFIX+'_TR40_FrozenWinner.json')
    assert winner_path.is_file(),'STAGE3 NOT COMPLETE: WINNER MISSING'
    w=json.loads(winner_path.read_text())
    assert w['status']=='VALIDATION_ONLY_TR40_12_TRIALS_COMPLETE'
    assert w['test_scored'] is False and w['heldout_test_evaluations']==0
    assert w['source_sha256']==C['source_sha256'] and w['split_sha256']==C['split_sha256']
    assert w['upstream_commit']==C['upstream_commit']
    assert w['ratios']==list(RATIOS) and w['final_training_seeds']==list(SEEDS)
    assert w['grid']==GRID and w['trials_completed']==12,'STAGE3 GRID OR TRIAL COUNT MISMATCH'
    trial_records=[]
    for k,cfg in enumerate(GRID,1):
        trial=TUNE/f'trial_{k:02d}'
        result_path=trial/'trial_result.json'
        checkpoint=trial/'best_validation.pt'
        assert result_path.is_file() and checkpoint.is_file(),('TUNING TRIAL INCOMPLETE',k)
        row=json.loads(result_path.read_text())
        assert row['trial']==k and row['config']==cfg
        assert row['source_sha256']==C['source_sha256'] and row['split_sha256']==C['split_sha256']
        assert row['upstream_commit']==C['upstream_commit']
        assert row['test_scored'] is False and row['test_labels_accessed'] is False
        assert hash_file(checkpoint)==row['checkpoint_sha256'],('TUNING CHECKPOINT HASH CHANGED',k)
        trial_records.append(row)
    selected=max(trial_records,key=lambda r:(r['best_val_auprc'],-r['trial']))
    assert w['winner_trial']==selected['trial'] and w['winner_config']==selected['config']
    assert abs(w['winner_val_auprc']-selected['best_val_auprc'])<1e-12
    assert w['winner_checkpoint_sha256']==selected['checkpoint_sha256']
    winner_sha=hash_file(winner_path)
    print(PREFIX+f'_STAGE4_ALL_12_TUNE_TRIALS_VERIFIED=PASS WINNER=TRIAL_{selected["trial"]:02d} val_AUPRC={selected["best_val_auprc"]:.6f}',flush=True)
    with np.load(SPLIT,allow_pickle=False) as z:
        assert int(np.asarray(z['seed']).reshape(-1)[0])==2
        assert int(np.asarray(z['source_nodes']).reshape(-1)[0])==C['nodes']
        ids={r:np.asarray(z[r],dtype=np.int64).copy() for r in RATIOS}
        val=np.asarray(z['val'],dtype=np.int64).copy()
        test=np.asarray(z['test'],dtype=np.int64).copy() # used ONLY for disjoint-set check
    assert [len(ids[r]) for r in RATIOS]==C['train_counts']
    assert len(val)==C['validation_nodes'] and len(test)==C['test_nodes']
    for name,arr in list(ids.items())+[('val',val),('test',test)]:
        assert len(np.unique(arr))==len(arr) and arr.min()>=0 and arr.max()<C['nodes'],('BAD_SPLIT',name)
    for r,arr in ids.items():
        assert not np.intersect1d(arr,val).size and not np.intersect1d(arr,test).size
    assert not np.intersect1d(val,test).size
    for smaller,bigger in [('TR10','TR20'),('TR20','TR30'),('TR30','TR40')]:
        assert np.isin(ids[smaller],ids[bigger]).all()
    mat=sio.loadmat(SRC)
    feature=mat['features']
    if sp.issparse(feature):feature=feature.toarray()
    x=np.asarray(feature,dtype=np.float32)
    assert x.shape==(C['nodes'],C['features']) and np.isfinite(x).all()
    raw_labels=np.asarray(mat['label']).reshape(-1)
    assert len(raw_labels)==C['nodes']
    # Crucial: NOT indexed for held-out test IDs; discard source immediately after extracting train + val.
    yt={r:raw_labels[ids[r]].astype('int64',copy=True) for r in RATIOS}
    yv=raw_labels[val].astype('int64',copy=True)
    del raw_labels
    mat.pop('label',None)
    for r,yy in yt.items():assert np.array_equal(np.unique(yy),[0,1]),('ONE_CLASS_TRAIN',r)
    assert np.array_equal(np.unique(yv),[0,1])
    edges={};edge_counts={}
    for rel in C['relations']:
        m=mat[rel]
        assert sp.issparse(m) and m.shape==(C['nodes'],C['nodes'])
        m=m.tocoo(copy=False)
        edge_counts[rel]=int(m.nnz)
        edges[('user',rel,'user')]=(torch.from_numpy(np.asarray(m.row,dtype=np.int64)),torch.from_numpy(np.asarray(m.col,dtype=np.int64)))
    del mat
    g=dgl.heterograph(edges,num_nodes_dict={'user':C['nodes']}).to('cuda:0')
    del edges
    device=torch.device('cuda:0')
    features=torch.from_numpy(x).to(device)
    vidx=torch.from_numpy(val).to(device)
    print(PREFIX+'_STAGE4_FULL_HETEROGRAPH_READY=PASS EDGES='+json.dumps(edge_counts,sort_keys=True),flush=True)
    print(PREFIX+'_STAGE4_HELDOUT_LABELS_NOT_INDEXED=PASS TEST_SCORED=NO',flush=True)
    return EOGFDHeteroModel,g,features,ids,yt,vidx,yv,device,winner_sha,w['winner_config']

def evaluate(model,graph,x,validx,yval):
    model.eval()
    with torch.random.fork_rng(devices=[0],enabled=True):
        torch.manual_seed(EVAL_SEED);torch.cuda.manual_seed_all(EVAL_SEED)
        with torch.no_grad():
            score=F.softmax(model(graph,x)[0][validx],dim=1)[:,1].detach().cpu().numpy().astype('float64')
    assert len(score)==len(yval) and np.isfinite(score).all()
    ap=float(average_precision_score(yval,score));roc=float(roc_auc_score(yval,score))
    thresh=np.arange(1,100,dtype=np.float64)/100.0
    positive=score[:,None]>thresh[None,:];yy=yval.astype(bool)[:,None]
    tp=(positive & yy).sum(axis=0);tn=((~positive) & (~yy)).sum(axis=0)
    fp=(positive & (~yy)).sum(axis=0);fn=((~positive) & yy).sum(axis=0)
    macro=(2*tp/np.maximum(2*tp+fp+fn,1)+2*tn/np.maximum(2*tn+fp+fn,1))/2.0
    best=int(macro.argmax())
    return dict(val_auprc=ap,val_auroc=roc,val_macro_f1=float(macro[best]),threshold=float(thresh[best])),score

def fit_run(ratio,seed,config,winner_sha,K,graph,x,ids,labels,validx,yval,device):
    dest=FINAL/ratio/f'seed_{seed}'
    dest.mkdir(parents=True,exist_ok=True)
    final=dest/'frozen_training.json'
    best=dest/'best_validation.pt'
    recovery=dest/'recovery_progress.pt'
    valfile=dest/'best_validation_scores.npy'
    historyfile=dest/'validation_history.json'
    if final.exists():
        row=json.loads(final.read_text())
        assert row['ratio']==ratio and row['seed']==seed and row['config']==config
        assert row['frozen_winner_sha256']==winner_sha
        assert row['source_sha256']==C['source_sha256'] and row['split_sha256']==C['split_sha256']
        assert not row['test_labels_accessed'] and not row['test_scored']
        assert best.is_file() and hash_file(best)==row['checkpoint_sha256']
        assert valfile.is_file() and hash_file(valfile)==row['validation_scores_sha256']
        assert historyfile.is_file()
        print(PREFIX+f'_{ratio}_Seed{seed}_FINAL_FROZEN_REUSED=PASS',flush=True)
        return row
    assert not valfile.exists(),'PREEXISTING VALIDATION SCORE WITHOUT FINAL RECORD: STOP'
    seed_all(seed)
    model=K(C['features'],config['hidden_dim'],2,d=config['order'],beta_p=config['beta_p'],population_size=config['population_size'],mutation_rate=config['mutation_rate']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['lr'])
    tx=torch.from_numpy(ids[ratio]).to(device);ty=torch.from_numpy(labels[ratio]).long().to(device)
    classw=torch.tensor([1.,float((labels[ratio]==0).sum())/float((labels[ratio]==1).sum())],device=device)
    hist=[];best_metric=-1.;best_epoch=0;best_metrics=None;start=1;stale=0
    if recovery.exists():
        state=torch.load(recovery,map_location=device)
        assert state['ratio']==ratio and state['seed']==seed and state['config']==config
        assert state['frozen_winner_sha256']==winner_sha
        assert state['source_sha256']==C['source_sha256'] and state['split_sha256']==C['split_sha256']
        load_state(model,state['model_state']);optimizer.load_state_dict(state['optim_state'])
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'],device)
        hist=state['history'];best_metric=state['best_metric'];best_metrics=state['best_metrics'];best_epoch=state['best_epoch'];stale=state['stale'];start=state['epoch']+1
        assert best.is_file() or best_epoch==0
        print(PREFIX+f'_{ratio}_Seed{seed}_RESUMED_EPOCH={start}',flush=True)
    else:
        assert not best.exists(),'UNTRACKED PREEXISTING BEST CHECKPOINT: STOP'
    tic=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start,MAX_EPOCHS+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        logits=model(graph,x)[0]
        loss=F.cross_entropy(logits[tx],ty,weight=classw)
        assert torch.isfinite(loss).item(),('NONFINITE LOSS',ratio,seed,epoch)
        loss.backward();optimizer.step()
        metrics,_=evaluate(model,graph,x,validx,yval)
        improved=metrics['val_auprc']>best_metric+1e-12
        if improved:
            best_metric=metrics['val_auprc'];best_epoch=epoch;best_metrics=metrics;stale=0
            atomic_pt(best,dict(model_state=frozen_state(model),ratio=ratio,seed=seed,epoch=epoch,config=config,validation=metrics,frozen_winner_sha256=winner_sha,source_sha256=C['source_sha256'],split_sha256=C['split_sha256'],upstream_commit=C['upstream_commit'],theta_last_excluded=True,test_scored=False))
        else:stale+=1
        hist.append(dict(epoch=epoch,training_loss=float(loss.detach().cpu()),**metrics,best_val_auprc_so_far=best_metric,stale=stale))
        atomic_pt(recovery,dict(ratio=ratio,seed=seed,epoch=epoch,config=config,model_state=frozen_state(model),optim_state=optimizer.state_dict(),history=hist,best_metric=best_metric,best_epoch=best_epoch,best_metrics=best_metrics,stale=stale,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),frozen_winner_sha256=winner_sha,source_sha256=C['source_sha256'],split_sha256=C['split_sha256']))
        if epoch<=2 or epoch%10==0:
            print(PREFIX+f'_{ratio}_Seed{seed} epoch={epoch} loss={float(loss.detach().cpu()):.5f} val_AUPRC={metrics["val_auprc"]:.6f} best={best_metric:.6f} stale={stale}',flush=True)
        del logits,loss
        if stale>=PATIENCE:break
    assert best.is_file() and best_metrics is not None
    # Repeat one FIXED RNG validation inference from frozen best for independent score record.
    obj=torch.load(best,map_location=device)
    assert obj['epoch']==best_epoch and obj['config']==config and obj['frozen_winner_sha256']==winner_sha
    load_state(model,obj['model_state'])
    confirm,scores=evaluate(model,graph,x,validx,yval)
    assert abs(confirm['val_auprc']-best_metric)<1e-8,('FROZEN CHECKPOINT REPLAY VALIDATION DIFFERENCE',ratio,seed)
    assert abs(confirm['threshold']-best_metrics['threshold'])<1e-12
    assert len(scores)==C['validation_nodes']
    atomic_np(valfile,scores)
    atomic_json(historyfile,hist)
    row=dict(model='EOGFDHeteroModel',dataset=C['dataset'],ratio=ratio,seed=seed,config=config,frozen_winner_sha256=winner_sha,epochs_completed=hist[-1]['epoch'],best_epoch=best_epoch,best_val_auprc=best_metric,best_val_auroc=confirm['val_auroc'],best_val_macro_f1=confirm['val_macro_f1'],frozen_validation_threshold=confirm['threshold'],checkpoint_sha256=hash_file(best),validation_scores_sha256=hash_file(valfile),train_count=len(ids[ratio]),train_positive_count=int(labels[ratio].sum()),val_count=len(yval),val_positive_count=int(yval.sum()),train_seconds=round(time.perf_counter()-tic,4),peak_gpu_allocated_gib=round(torch.cuda.max_memory_allocated(device)/1073741824,5),full_original_heterograph=True,source_sha256=C['source_sha256'],split_sha256=C['split_sha256'],upstream_commit=C['upstream_commit'],epochs_cap=MAX_EPOCHS,patience=PATIENCE,validation_evolution_rng_seed=EVAL_SEED,validation_only_selection=True,test_labels_accessed=False,test_scored=False,heldout_test_pending=True)
    atomic_json(final,row)
    if recovery.exists():recovery.unlink()
    print(PREFIX+f'_{ratio}_Seed{seed}_FROZEN_VALIDATION_FINAL=PASS val_AUPRC={best_metric:.6f} threshold={confirm["threshold"]:.2f}',flush=True)
    del model,optimizer
    gc.collect();torch.cuda.empty_cache()
    return row

def main():
    FINAL.mkdir(parents=True,exist_ok=True)
    K,g,x,ids,yt,validx,yv,device,winner_sha,config=validate()
    print(PREFIX+'_STAGE4_12_FROZEN_FINALS_START=PASS NO_TUNING=YES TEST_SCORED=NO',flush=True)
    results=[]
    for ratio in RATIOS:
        for seed in SEEDS:
            results.append(fit_run(ratio,seed,config,winner_sha,K,g,x,ids,yt,validx,yv,device))
    assert len(results)==12
    report=[]
    for r in RATIOS:
        vals=np.array([row['best_val_auprc'] for row in results if row['ratio']==r])
        report.append(dict(ratio=r,validation_auprc_mean=float(vals.mean()),validation_auprc_sample_sd=float(vals.std(ddof=1)),seeds=list(SEEDS)))
    summary=dict(status='EOGFD_12_FROZEN_TRAININGS_COMPLETE',dataset=C['dataset'],model='EOGFDHeteroModel',training_count=12,config=config,frozen_winner_sha256=winner_sha,source_sha256=C['source_sha256'],split_sha256=C['split_sha256'],upstream_commit=C['upstream_commit'],ratios=list(RATIOS),seeds=list(SEEDS),validation_results=report,heldout_test_scored_count=0,test_labels_accessed=False,test_scored=False,heldout_test_pending=True)
    atomic_json(FINAL/(PREFIX+'_Stage4_12FrozenTrainingSummary.json'),summary)
    for row in report:
        print(PREFIX+'_'+row['ratio']+f'_STAGE4_VALIDATION_AUPRC_MEAN={row["validation_auprc_mean"]:.6f} sampleSD={row["validation_auprc_sample_sd"]:.6f}',flush=True)
    print(PREFIX+'_STAGE4_12_FROZEN_FINAL_TRAININGS_COMPLETE=PASS TEST_SCORED=NO',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        print(PREFIX+'_STAGE4_STOP_ERROR='+repr(e),flush=True)
        traceback.print_exc();sys.exit(2)
