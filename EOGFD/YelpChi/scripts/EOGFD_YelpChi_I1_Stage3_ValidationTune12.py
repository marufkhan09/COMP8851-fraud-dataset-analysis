#!/usr/bin/env python3
"""EOGFD Stage3: full-real-heterograph TR40 validation-only tuning (12 trials).
Unmodified pinned upstream EOGFDHeteroModel; original frozen splits; no test-label indexing
or test scoring; test indices are read only to verify disjoint membership.
"""
import sys, os, json, socket, hashlib, subprocess, time, random, gc, traceback, itertools
from pathlib import Path
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import dgl
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

CONFIG=Path(__file__).with_name('EOGFD_YelpChi_I1_Stage3_Config.json')
C=json.loads(CONFIG.read_text())
PREFIX='EOGFD_'+C['dataset']+'_'+C['id']
REPO=Path('/workspace/comp8851/repos/EOGFD')
ROOT=Path('/workspace/comp8851/results/eogfd')/C['dataset'].lower()/ 'stage3_tr40_tuning'
LOGROOT=Path('/workspace/comp8851/logs/eogfd')
SRC=Path('/workspace/comp8851/assets/eogfd')/C['dataset']/C['source_name']
SPLIT=Path('/workspace/uploads/ass2_split_candidates')/C['dataset']/C['split_name']
GRIDS=[{'lr':lr,'population_size':ps,'mutation_rate':mut,'hidden_dim':64,'order':C['order'],'beta_p':1,'lamb':0.0,'beta':C['beta']}
       for lr,ps,mut in itertools.product([0.001,0.003,0.01],[10,20],[0.05,0.1])]
EVAL_SEED=100921 # fixed validation stochastic evolution path across all tuning trials
MAX_EPOCHS=100
PATIENCE=20

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def atomic_json(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp')
    with open(tmp,'w') as f:
        json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n')
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)

def atomic_pt(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(obj,tmp)
    os.replace(tmp,path)

def clean_model_state(model):
    # Upstream theta_last is a variable-size diagnostic buffer, not a trained parameter.
    # It cannot be loaded into the fresh empty buffer by strict load_state_dict.
    return {k:v for k,v in model.state_dict().items() if k!='theta_last'}

def reload_model_state(model,sd):
    problem=model.load_state_dict(sd,strict=False)
    assert problem.missing_keys==['theta_last'] and not problem.unexpected_keys,problem

def validate_inputs():
    assert socket.gethostname()==C['hostname'],('WRONG INSTANCE',socket.gethostname())
    assert torch.cuda.is_available(),'NO CUDA'
    for name,p,want in [('SOURCE',SRC,C['source_sha256']),('SPLIT',SPLIT,C['split_sha256'])]:
        assert p.is_file(),('MISSING',str(p))
        assert sha(p)==want,(name,'SOURCE/SPLIT SHA MISMATCH')
        print(PREFIX+'_STAGE3_'+name+'_SHA256=PASS',flush=True)
    assert REPO.is_dir(), 'UPSTREAM EOGFD SOURCE MISSING'
    head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    assert head==C['upstream_commit'],('UPSTREAM COMMIT CHANGED',head)
    tracked=subprocess.run(['git','-C',str(REPO),'diff','--exit-code','HEAD','--'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    assert tracked.returncode==0,'UPSTREAM TRACKED CONTENT CHANGED'
    dirty=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    assert not dirty,('UPSTREAM UNEXPECTED FILES',dirty)
    sys.path.insert(0,str(REPO))
    from models.eogfd import EOGFDHeteroModel
    # No aliases added to pinned original. Our runner uses the actual released class.
    with np.load(SPLIT,allow_pickle=False) as z:
        assert int(np.asarray(z['seed']).reshape(-1)[0])==2
        assert int(np.asarray(z['source_nodes']).reshape(-1)[0])==C['nodes']
        ratios={r:np.asarray(z[r],dtype=np.int64).copy() for r in ['TR40','TR30','TR20','TR10']}
        val=np.asarray(z['val'],dtype=np.int64).copy()
        test=np.asarray(z['test'],dtype=np.int64).copy()  # only membership, no labels
    assert [len(ratios[r]) for r in ['TR40','TR30','TR20','TR10']]==C['train_counts']
    assert len(val)==C['validation_nodes'] and len(test)==C['test_nodes']
    for r,arr in list(ratios.items())+[('val',val),('test',test)]:
        assert np.unique(arr).size==len(arr),('DUPLICATES',r)
        assert arr.min()>=0 and arr.max()<C['nodes'],('OUT OF BOUNDS',r)
    for r,a in ratios.items():
        assert not np.intersect1d(a,val).size and not np.intersect1d(a,test).size,('SPLIT LEAK',r)
    assert not np.intersect1d(val,test).size
    for child,parent in [('TR10','TR20'),('TR20','TR30'),('TR30','TR40')]:
        assert np.isin(ratios[child],ratios[parent]).all(),'NOT NESTED'
    print(PREFIX+'_STAGE3_FROZEN_SPLIT_DISJOINT_AND_NESTED=PASS TEST_LABELS_NOT_INDEXED=YES',flush=True)
    mat=sio.loadmat(SRC)
    features=mat['features']
    if sp.issparse(features):features=features.toarray()
    features=np.asarray(features,dtype=np.float32)
    assert features.shape==(C['nodes'],C['features']) and np.isfinite(features).all()
    all_labels=np.asarray(mat['label']).reshape(-1)
    assert len(all_labels)==C['nodes']
    # Only train and validation labels may be indexed. No test labels are accessed.
    ytrain=all_labels[ratios['TR40']].astype('int64',copy=True)
    yval=all_labels[val].astype('int64',copy=True)
    del all_labels
    mat.pop('label',None)
    assert np.array_equal(np.unique(ytrain),np.array([0,1]))
    assert np.array_equal(np.unique(yval),np.array([0,1]))
    edges={};counts={}
    for relation in C['relations']:
        mx=mat[relation]
        assert sp.issparse(mx) and mx.shape==(C['nodes'],C['nodes']),relation
        mx=mx.tocoo(copy=False)
        counts[relation]=int(mx.nnz)
        edges[('user',relation,'user')]=(torch.from_numpy(np.asarray(mx.row,dtype=np.int64)),torch.from_numpy(np.asarray(mx.col,dtype=np.int64)))
    del mat
    graph=dgl.heterograph(edges,num_nodes_dict={'user':C['nodes']}).to('cuda:0')
    del edges
    assert graph.num_nodes('user')==C['nodes']
    device=torch.device('cuda:0')
    x=torch.from_numpy(features).to(device)
    t_ix=torch.from_numpy(ratios['TR40']).long().to(device)
    y=torch.from_numpy(ytrain).long().to(device)
    class_weights=torch.tensor([1.0,float((ytrain==0).sum())/float((ytrain==1).sum())],device=device)
    print(PREFIX+'_STAGE3_GRAPH_READY_NODES='+str(C['nodes'])+' RELATION_EDGES='+json.dumps(counts,sort_keys=True),flush=True)
    return EOGFDHeteroModel,graph,x,t_ix,y,val,yval,class_weights,device,counts

def make_model(K,cfg,device):
    return K(C['features'],cfg['hidden_dim'],2,d=cfg['order'],beta_p=cfg['beta_p'],population_size=cfg['population_size'],mutation_rate=cfg['mutation_rate']).to(device)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def eval_only(model,graph,x,val_ix,yval,device):
    model.eval()
    # Evolution algorithm draws random population during inference. Hold the same
    # validation RNG path across epochs/trials to avoid selection from fresh random draws.
    with torch.random.fork_rng(devices=[0],enabled=True):
        torch.manual_seed(EVAL_SEED)
        torch.cuda.manual_seed_all(EVAL_SEED)
        with torch.no_grad():
            scores=torch.softmax(model(graph,x)[0][val_ix],dim=1)[:,1].detach().cpu().numpy().astype('float64')
    assert len(scores)==len(yval) and np.isfinite(scores).all()
    ap=float(average_precision_score(yval,scores))
    roc=float(roc_auc_score(yval,scores))
    # Fixed 99-point threshold grid, validation labels only.
    thresholds=np.arange(1,100,dtype=np.float64)/100.0
    preds=scores[:,None]>thresholds[None,:]
    yy=yval.astype(bool)[:,None]
    tp=(preds&yy).sum(axis=0);tn=((~preds)&(~yy)).sum(axis=0)
    fp=(preds&(~yy)).sum(axis=0);fn=((~preds)&yy).sum(axis=0)
    fpos=2*tp/np.maximum(2*tp+fp+fn,1)
    fneg=2*tn/np.maximum(2*tn+fp+fn,1)
    fmac=(fpos+fneg)/2.0
    best=int(np.argmax(fmac))
    return {'val_auprc':ap,'val_auroc':roc,'val_macro_f1':float(fmac[best]),'threshold':float(thresholds[best])}

def gate(K,graph,x,t_ix,y,weights,device):
    path=ROOT/'EOGFD_STAGE3_POP20_COST_GATE.json'
    if path.exists():
        d=json.loads(path.read_text())
        assert d.get('status')=='PASS' and d.get('source_sha256')==C['source_sha256'] and d.get('split_sha256')==C['split_sha256']
        print(PREFIX+'_STAGE3_POP20_GATE_REUSED=PASS',flush=True)
        return
    seed_all(2)
    cfg=GRIDS[2]  # population 20, mutation .05, lr .001 (product ordering)
    assert cfg['population_size']==20
    model=make_model(K,cfg,device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    torch.cuda.reset_peak_memory_stats(device)
    tic=time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits=model(graph,x)[0]
    loss=F.cross_entropy(logits[t_ix],y,weight=weights)
    assert torch.isfinite(loss).item(),'NONFINITE GATE LOSS'
    loss.backward();optimizer.step();torch.cuda.synchronize()
    seconds=time.perf_counter()-tic
    gib=torch.cuda.max_memory_allocated(device)/1073741824
    del model,optimizer,logits,loss
    gc.collect();torch.cuda.empty_cache()
    print(PREFIX+'_STAGE3_POP20_GATE_SECONDS='+format(seconds,'.3f')+' GPU_ALLOC_GIB='+format(gib,'.3f'),flush=True)
    # Explicit budget gate; no 12-trial launch if an epoch is unexpectedly expensive.
    assert gib<8.0 and seconds<18.0,('STOP_COST_OR_MEMORY_GATE',seconds,gib)
    atomic_json(path,dict(status='PASS',source_sha256=C['source_sha256'],split_sha256=C['split_sha256'],pop_size=20,seconds=round(seconds,5),peak_allocated_gib=round(gib,5),train_only_step=True,test_scored=False))
    print(PREFIX+'_STAGE3_POP20_COST_GATE=PASS TEST_SCORED=NO',flush=True)

def run_trial(index,cfg,K,graph,x,t_ix,y,val_ix,yval,weights,device):
    dirname=ROOT/('trial_%02d'%index)
    dirname.mkdir(exist_ok=True)
    final=dirname/'trial_result.json';best=dirname/'best_validation.pt';progress=dirname/'recovery_progress.pt'
    if final.exists():
        rec=json.loads(final.read_text())
        assert rec['config']==cfg and rec['source_sha256']==C['source_sha256'] and rec['split_sha256']==C['split_sha256']
        assert best.is_file() and sha(best)==rec['checkpoint_sha256'],('FINAL_CHECKPOINT_MISSING_OR_CHANGED',index)
        print(PREFIX+f'_TR40_TRIAL_{index:02d}_EXISTING_VALIDATION_RESULT=PASS val_AUPRC={rec["best_val_auprc"]:.6f}',flush=True)
        return rec
    seed_all(2)
    model=make_model(K,cfg,device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    history=[];best_metric=-1.0;best_epoch=0;best_metrics=None;start_epoch=1;stale=0
    if progress.exists():
        rec=torch.load(progress,map_location=device)
        assert rec['config']==cfg and rec['source_sha256']==C['source_sha256'] and rec['split_sha256']==C['split_sha256'],('WRONG_RESUME',index)
        reload_model_state(model,rec['model_state']); opt.load_state_dict(rec['optim_state'])
        torch.set_rng_state(rec['cpu_rng']);torch.cuda.set_rng_state(rec['cuda_rng'],device)
        start_epoch=rec['epoch']+1;history=rec['history'];best_metric=rec['best_metric'];best_metrics=rec['best_metrics'];best_epoch=rec['best_epoch'];stale=rec['stale']
        assert best.exists() or best_epoch==0,('MISSING_BEST_AT_RESUME',index)
        print(PREFIX+f'_TR40_TRIAL_{index:02d}_RESUME_FROM_EPOCH={start_epoch}',flush=True)
    else:
        assert not best.exists(),('PREEXISTING_BEST_WITH_NO_RECOVERY: STOP',index)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed=time.perf_counter()
    for epoch in range(start_epoch,MAX_EPOCHS+1):
        model.train();opt.zero_grad(set_to_none=True)
        logits=model(graph,x)[0]
        loss=F.cross_entropy(logits[t_ix],y,weight=weights)
        assert torch.isfinite(loss).item(),('NONFINITE_LOSS',index,epoch)
        loss.backward();opt.step()
        current=eval_only(model,graph,x,val_ix,yval,device)
        assert np.isfinite(current['val_auprc']),('BAD_VAL',index,epoch)
        improved=current['val_auprc']>best_metric+1e-12
        if improved:
            best_metric=current['val_auprc'];best_epoch=epoch;best_metrics=current;stale=0
            atomic_pt(best,dict(model_state=clean_model_state(model),epoch=epoch,config=cfg,val=current,upstream_commit=C['upstream_commit'],source_sha256=C['source_sha256'],split_sha256=C['split_sha256'],ratio='TR40',tuning_seed=2,test_scored=False,theta_last_buffer_excluded=True))
        else:stale+=1
        history.append(dict(epoch=epoch,train_loss=float(loss.detach().cpu()),**current,best_so_far=best_metric,stale=stale))
        # Exact optimizer/torch-RNG resume. Save after every evaluated epoch, independent of best ckpt.
        atomic_pt(progress,dict(epoch=epoch,config=cfg,model_state=clean_model_state(model),optim_state=opt.state_dict(),history=history,best_metric=best_metric,best_epoch=best_epoch,best_metrics=best_metrics,stale=stale,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),source_sha256=C['source_sha256'],split_sha256=C['split_sha256']))
        if epoch<=2 or epoch%5==0 or improved and epoch==1:
            print(PREFIX+f'_TR40_TRIAL_{index:02d} epoch={epoch} loss={float(loss.detach().cpu()):.5f} val_AUPRC={current["val_auprc"]:.6f} best={best_metric:.6f} stale={stale}',flush=True)
        del logits,loss
        if stale>=PATIENCE: break
    assert best_metrics is not None and best.is_file() and len(history)>=1
    atomic_json(dirname/'validation_history.json',history)
    rec=dict(dataset=C['dataset'],model='EOGFDHeteroModel',ratio='TR40',trial=index,config=cfg,seed=2,epochs_completed=history[-1]['epoch'],best_epoch=best_epoch,best_val_auprc=float(best_metric),best_validation_metrics=best_metrics,checkpoint_sha256=sha(best),source_sha256=C['source_sha256'],split_sha256=C['split_sha256'],upstream_commit=C['upstream_commit'],sampling='FULL_HETEROGRAPH_ORIGINAL_RELATIONS',selection_metric='validation_AUPRC',validation_threshold_metric='validation_macro_F1',fixed_validation_evolution_rng_seed=EVAL_SEED,trial_wall_seconds=round(time.perf_counter()-elapsed,4),peak_gpu_allocated_gib=round(torch.cuda.max_memory_allocated(device)/1073741824,4),test_labels_accessed=False,test_scored=False,full_heldout_test_pending=True)
    atomic_json(final,rec)
    if progress.exists():progress.unlink()
    print(PREFIX+f'_TR40_TRIAL_{index:02d}_VAL_ONLY=PASS AUPRC={best_metric:.6f} epoch={best_epoch} threshold={best_metrics["threshold"]:.3f}',flush=True)
    del model,opt
    gc.collect();torch.cuda.empty_cache()
    return rec

def main():
    assert len(GRIDS)==12 and len({json.dumps(g,sort_keys=True) for g in GRIDS})==12
    ROOT.mkdir(parents=True,exist_ok=True);LOGROOT.mkdir(parents=True,exist_ok=True)
    K,graph,x,t_ix,y,val_ix,yval,weights,device,counts=validate_inputs()
    from models.eogfd import EOGFDHeteroModel
    assert K is EOGFDHeteroModel
    print(PREFIX+'_STAGE3_VALIDATION_ONLY_TUNING_START=PASS GRID=12 MAX_EPOCHS=100 PATIENCE=20',flush=True)
    gate(K,graph,x,t_ix,y,weights,device)
    records=[]
    for index,cfg in enumerate(GRIDS,1):
        records.append(run_trial(index,cfg,K,graph,x,t_ix,y,val_ix,yval,weights,device))
    selected=max(records,key=lambda r:(r['best_val_auprc'],-r['trial']))
    winner=dict(dataset=C['dataset'],model='EOGFDHeteroModel',status='VALIDATION_ONLY_TR40_12_TRIALS_COMPLETE',winner_trial=selected['trial'],winner_config=selected['config'],winner_val_auprc=selected['best_val_auprc'],winner_checkpoint_sha256=selected['checkpoint_sha256'],source_sha256=C['source_sha256'],split_sha256=C['split_sha256'],upstream_commit=C['upstream_commit'],tuning_seed=2,final_training_seeds=[2,42,72],ratios=['TR40','TR30','TR20','TR10'],grid=GRIDS,trials_completed=len(records),validation_metric='AUPRC',test_scored=False,heldout_test_evaluations=0)
    atomic_json(ROOT/(PREFIX+'_TR40_FrozenWinner.json'),winner)
    atomic_json(ROOT/(PREFIX+'_TR40_TuningSummary.json'),{'winner':winner,'trial_metrics':[dict(trial=r['trial'],best_val_auprc=r['best_val_auprc'],best_epoch=r['best_epoch'],config=r['config']) for r in records]})
    print(PREFIX+'_STAGE3_TUNING_12_VAL_ONLY_COMPLETE=PASS TEST_SCORED=NO',flush=True)
    print(PREFIX+f'_STAGE3_WINNER=TRIAL_{selected["trial"]:02d} VAL_AUPRC={selected["best_val_auprc"]:.6f} CONFIG={json.dumps(selected["config"],sort_keys=True)}',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        print(PREFIX+'_STAGE3_STOP_ERROR='+repr(e),flush=True)
        traceback.print_exc()
        sys.exit(2)
